// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/cooperative_groups.h"
#include "gpu_iface/fastdiv.cuh"
#include "gpu_iface/math_ops.hpp"
#include "gpu_iface/memory_ops.hpp"
#include "gpu_iface/mma_ops.hpp"
#include "gpu_iface/platform.hpp"
#include "gpu_iface/utils.cuh"

#ifdef FP16_QK_REDUCTION_SUPPORTED
#include "../../fp16.h"
#endif
#include <type_traits>

#include "cascade.cuh"
#include "dispatch.cuh"
#include "frag_layout_swizzle.cuh"
#include "page.cuh"
#include "permuted_smem.cuh"
#include "pos_enc.cuh"
#include "variants.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(maybe_q_rope_offset)
DEFINE_HAS_MEMBER(maybe_k_rope_offset)

namespace cg = gpu_iface::cg;
namespace memory = gpu_iface::memory;
namespace mma = gpu_iface::mma;

using gpu_iface::vec_dtypes::vec_cast;
using mma::MMAMode;

constexpr uint32_t WARP_SIZE = gpu_iface::kWarpSize;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}

template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];
      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
    };
    struct {  // NOTE(Zihao): synchronize attention states across warps
      alignas(
          16) std::conditional_t<NUM_WARPS_KV == 1, float[1],
                                 float[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO]> cta_sync_o_smem;
      alignas(16) std::conditional_t<NUM_WARPS_KV == 1, float2[1],
                                     float2[NUM_WARPS_KV * CTA_TILE_Q]> cta_sync_md_smem;
    };
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];
  };
};

template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_, uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_, uint32_t NUM_WARPS_Q_,
          uint32_t NUM_WARPS_KV_, PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_>
struct KernelTraits {
  static constexpr MaskMode MASK_MODE = MASK_MODE_;
  static constexpr uint32_t NUM_MMA_Q = NUM_MMA_Q_;
  static constexpr uint32_t NUM_MMA_KV = NUM_MMA_KV_;
  static constexpr uint32_t NUM_MMA_D_QK = NUM_MMA_D_QK_;
  static constexpr uint32_t NUM_MMA_D_VO = NUM_MMA_D_VO_;
  static constexpr uint32_t NUM_WARPS_Q = NUM_WARPS_Q_;
  static constexpr uint32_t NUM_WARPS_KV = NUM_WARPS_KV_;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  static constexpr uint32_t HEAD_DIM_QK = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO = NUM_MMA_D_VO * 16;
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16;
  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using DTypeQKAccum = DTypeQKAccum_;
  using IdType = IdType_;
  using AttentionVariant = AttentionVariant_;

  static_assert(sizeof(DTypeKV_) != 1, "8-bit types not supported for CDNA3");

  using SmemBasePtrTy = uint2;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * 64;
  static constexpr uint32_t WARP_THREAD_ROWS = 4;
  static constexpr uint32_t WARP_THREAD_COLS = 16;
  static constexpr uint32_t HALF_ELEMS_PER_THREAD = 4;
  static constexpr uint32_t INT32_ELEMS_PER_THREAD = 2;
  static constexpr uint32_t VECTOR_BIT_WIDTH = HALF_ELEMS_PER_THREAD * 16;

  // k128B_16Row extends the XOR period from 8 to 16, eliminating the 8-way
  // LDS bank conflicts that k128B exhibits in the Q-smem read path on MI300x
  // (CDNA3 issues LDS in 16-thread phases; k128B only de-aliases 8 of them).
  static constexpr SwizzleMode SWIZZLE_MODE_Q = SwizzleMode::k128B_16Row;
  static constexpr SwizzleMode SWIZZLE_MODE_KV = SwizzleMode::k128B_16Row;

  // Presently we use 16x4 thread layout for all cases.
  static constexpr uint32_t KV_THR_LAYOUT_ROW = WARP_THREAD_ROWS;
  static constexpr uint32_t KV_THR_LAYOUT_COL = WARP_THREAD_COLS;
  // FIXME: [The comment is not correct] The constant is defined based on the
  // matrix layout of the "D/C" accumulator matrix in a D = A*B+C computation.
  // On CDNA3 the D/C matrices are distributed as four 4x16 bands across the
  // 64 threads. Each thread owns one element from four different rows.
  static constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD = 4;
  // Number of threads that collaboratively handle the same set of matrix rows
  // in attention score computation and cross-warp synchronization.
  // CDNA3: 16 threads (each thread handles 1 element from same row group)
  static constexpr uint32_t THREADS_PER_BMATRIX_ROW_SET = 16;
  // controls the indexing stride used in logits-related functions
  // (logits_transform, logits_mask, and LSE writing).
  static constexpr uint32_t LOGITS_INDEX_STRIDE = 4;
  static constexpr uint32_t UPCAST_STRIDE_Q =
      HEAD_DIM_QK / upcast_size<DTypeQ_, VECTOR_BIT_WIDTH>();
  static constexpr uint32_t UPCAST_STRIDE_K =
      HEAD_DIM_QK / upcast_size<DTypeKV_, VECTOR_BIT_WIDTH>();
  static constexpr uint32_t UPCAST_STRIDE_V =
      HEAD_DIM_VO / upcast_size<DTypeKV_, VECTOR_BIT_WIDTH>();
  static constexpr uint32_t UPCAST_STRIDE_O =
      HEAD_DIM_VO / upcast_size<DTypeO_, VECTOR_BIT_WIDTH>();

  static constexpr bool IsInvalid() {
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && NUM_MMA_D_VO > 4 &&
             NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
            (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
  }

  using SharedStorage = SharedStorageQKVO<NUM_WARPS_KV, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_QK,
                                          HEAD_DIM_VO, DTypeQ, DTypeKV, DTypeO>;
#ifdef FP16_QK_REDUCTION_SUPPORTED
  template <typename DT>
  static constexpr DT getNegInf() {
    if constexpr (std::is_same<DT, __half>::value) {
      return std::bit_cast<half>(fp16_ieee_from_fp32_value(-gpu_iface::math::inf));
    } else {
      return static_cast<DTypeQKAccum>(-gpu_iface::math::inf);
    }
  }

  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? getNegInf<DTypeQKAccum>() : DTypeQKAccum(0.f);
#else
  static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                "then recompile to support fp16 reduction");
  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? DTypeQKAccum(-gpu_iface::math::inf) : DTypeQKAccum(0.f);
#endif
};

namespace {

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q(const uint32_t tid_y = threadIdx.y) {
  if constexpr (KTraits::NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return tid_y;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_kv(const uint32_t tid_z = threadIdx.z) {
  if constexpr (KTraits::NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return tid_z;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx(const uint32_t tid_y = threadIdx.y,
                                                 const uint32_t tid_z = threadIdx.z) {
  return get_warp_idx_kv<KTraits>(tid_z) * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid_y);
}

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <typename T, uint32_t HALF_ELEMS_PER_THREAD>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < HALF_ELEMS_PER_THREAD; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7

    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T, uint32_t HALF_ELEMS_PER_THREAD>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < HALF_ELEMS_PER_THREAD; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    uint32_t freq_idx = 2 * j + reg_id % 2;
    uint32_t position = qo_packed_offset + 8 * i;
    __sincosf(float(position / group_size) * rope_freq[freq_idx], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T, typename IdType, uint32_t HALF_ELEMS_PER_THREAD>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_rope_offset) {
  float pos[2] = {static_cast<float>(q_rope_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_rope_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < HALF_ELEMS_PER_THREAD; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    // NOTE: The following indexing logic is CUDA-specific and is temporarily used pending HIP port
    //       completion. This matches the CUDA register layout; update as needed when porting to
    //       HIP.
    const uint32_t i = (reg_id % 4) / 2;
    const uint32_t j = reg_id / 4;
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename KTraits, bool produce_v, SharedMemFillMode fill_mode>
__device__ __forceinline__ void produce_kv_impl(
    uint32_t warp_idx, uint32_t lane_idx,
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> smem, uint32_t* smem_offset,
    typename KTraits::DTypeKV** gptr, const uint32_t stride_n, const uint32_t kv_idx_base,
    const uint32_t kv_len) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;  // 16
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  // NOTE: NUM_MMA_KV*4/NUM_WARPS_Q = NUM_WARPS_KV*NUM_MMA_KV*4/num_warps
  static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / KV_THR_LAYOUT_COL;

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
      smem.template load_vector_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.template advance_offset_by_column<16>(*smem_offset, j);
      *gptr += 16 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
    }
    kv_idx += NUM_WARPS * 4;
    *smem_offset = smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
                   (sizeof(DTypeKV) * NUM_MMA_D * 2);
    *gptr += NUM_WARPS * 4 * stride_n -
             sizeof(DTypeKV) * NUM_MMA_D * 2 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
  }
  *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <bool produce_v, SharedMemFillMode fill_mode, typename KTraits>
__device__ __forceinline__ void produce_kv(
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> smem, uint32_t* smem_offset,
    typename KTraits::DTypeKV** gptr, const uint32_t stride_n, const uint32_t kv_idx_base,
    const uint32_t kv_len, const dim3 tid = threadIdx) {
  const uint32_t warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), lane_idx = tid.x;
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;  // 16
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  // NOTE: NUM_MMA_KV*4/NUM_WARPS_Q = NUM_WARPS_KV*NUM_MMA_KV*4/num_warps
  static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / KV_THR_LAYOUT_COL;

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
      smem.template load_vector_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.template advance_offset_by_column<16>(*smem_offset, j);
      *gptr += 16 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
    }
    kv_idx += NUM_WARPS * 4;
    *smem_offset = smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
                   (sizeof(DTypeKV) * NUM_MMA_D * 2);
    *gptr += NUM_WARPS * 4 * stride_n -
             sizeof(DTypeKV) * NUM_MMA_D * 2 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
  }
  *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
}

template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t kv_idx_base, const size_t* thr_local_kv_offset, const uint32_t kv_len,
    const dim3 tid = threadIdx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  const uint32_t warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), lane_idx = tid.x;

  // NOTE: NUM_MMA_KV * 4/NUM_WARPS_Q=NUM_WARPS_KV*NUM_MMA_KV*4/num_warps
  static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / KV_THR_LAYOUT_COL;

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
    DTypeKV* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
      smem.template load_vector_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);

      *smem_offset = smem.template advance_offset_by_column<16>(*smem_offset, j);
      gptr += 16 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
    }
    kv_idx += NUM_WARPS * 4;
    *smem_offset = smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
                   (sizeof(DTypeKV) * NUM_MMA_D * 2);
  }
  *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
}

template <uint32_t HEAD_DIM>
__device__ __forceinline__ uint32_t get_feature_index(uint32_t mma_d, uint32_t lane_idx,
                                                      uint32_t j) {
  // CUDA A-matrix MMA tile to thread mapping for a 32 thread warp:
  // Each group of four consecutive threads map four different features for
  // the same sequence.
  // T0: {0,1,8,9}, T1: {2,3,10,11}, T2: {4,5,12,13}, T3: {6,7,14,15}
  //
  // The pattern repeats across 8 rows with each row mapped to a set of four
  // consecutive threads.
  //      row 0 --> T0, T1, T2, T3
  //      row 1 --> T4, T5, T6, T7
  //      ...
  //      row 7 --> T28, T29, T30, T31
  // The full data to thread mapping repeats again for the next set of 16
  // rows. Thereby, forming a 16x16 MMA tile dubdivided into four 8x8
  // quadrants.
  uint32_t feature_index =
      ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) % (HEAD_DIM / 2));
  return feature_index;
}

template <typename KTraits>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4], const float rope_rcp_scale,
                                               const float rope_rcp_theta,
                                               const uint32_t tid_x = threadIdx.x) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  const uint32_t lane_idx = tid_x;

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t feature_index = get_feature_index<HEAD_DIM>(mma_d, lane_idx, j);
      float freq_base = float(2 * feature_index) / float(HEAD_DIM);
      rope_freq[mma_d][j] = rope_rcp_scale * __powf(rope_rcp_theta, freq_base);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_states(
    typename KTraits::AttentionVariant variant,
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    typename KTraits::DTypeQKAccum (*m)[KTraits::NUM_ACCUM_ROWS_PER_THREAD],
    float (*d)[KTraits::NUM_ACCUM_ROWS_PER_THREAD]) {
  constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD = KTraits::NUM_ACCUM_ROWS_PER_THREAD;
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
        m[mma_q][j] = typename KTraits::DTypeQKAccum(-gpu_iface::math::inf);
        d[mma_q][j] = 1.f;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy>* q_smem,
    const dim3 tid = threadIdx) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t WARP_THREAD_COLS = KTraits::WARP_THREAD_COLS;
  constexpr uint32_t WARP_THREAD_ROWS = KTraits::WARP_THREAD_ROWS;
  constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  constexpr uint32_t COLUMN_RESET_OFFSET = (NUM_MMA_D_QK / 4) * WARP_THREAD_COLS;
  const uint32_t lane_idx = tid.x, warp_idx_x = get_warp_idx_q<KTraits>(tid.y);
  uint32_t row = lane_idx / WARP_THREAD_COLS;
  uint32_t col = lane_idx % WARP_THREAD_COLS;

  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    uint32_t q_smem_offset_w = q_smem->template get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + row, col);
    // row_idx_w: the logical smem row immediately before each advance_offset_by_row<4>
    // call.  Required by k128B_16Row to select the correct xor_mask; ignored by k128B.
    uint32_t row_idx_w = warp_idx_x * KTraits::NUM_MMA_Q * 16 + row;

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + row + mma_q * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h +
                        col * upcast_size<DTypeQ, VECTOR_BIT_WIDTH>();
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          // load q fragment from gmem to smem
          q_smem->template load_vector_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                                         q_idx < qo_upper_bound);
          q_smem_offset_w =
              q_smem->template advance_offset_by_column<WARP_THREAD_COLS>(q_smem_offset_w, mma_do);
          q_ptr += WARP_THREAD_COLS * upcast_size<DTypeQ, VECTOR_BIT_WIDTH>();
        }
        q_smem_offset_w = q_smem->template advance_offset_by_row<WARP_THREAD_ROWS, UPCAST_STRIDE_Q>(
                              q_smem_offset_w, row_idx_w) -
                          COLUMN_RESET_OFFSET;
        row_idx_w += WARP_THREAD_ROWS;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy>* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) != 0) return;

  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t COL_ADVANCE_TO_NEXT = 16 / KTraits::HALF_ELEMS_PER_THREAD;
  constexpr uint32_t COL_ADVANCE_TO_LAST_HALF = KTraits::NUM_MMA_D_QK;

  const uint32_t lane_idx = tid.x;
  uint32_t q_frag_local[2][KTraits::INT32_ELEMS_PER_THREAD];
  static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
  // q_col_rope: starting column for this thread (lane / WARP_THREAD_COLS).
  // The column of *q_smem_offset_r never changes between mma_q iterations
  // (only the row advances), so q_col_rope is constant across the outer loop.
  const uint32_t q_col_rope = lane_idx / KTraits::WARP_THREAD_COLS;
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
    uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
    const uint32_t seq_id =
        q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4;
    // col_idx of first_half advances by COL_ADVANCE_TO_NEXT each mma_di step.
    uint32_t q_col_first = q_col_rope;
#pragma unroll
    for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
      q_smem->load_fragment(q_smem_offset_r_first_half, q_frag_local[0]);
      uint32_t q_smem_offset_r_last_half =
          q_smem->template advance_offset_by_column<COL_ADVANCE_TO_LAST_HALF, UPCAST_STRIDE_Q>(
              q_smem_offset_r_first_half, 0, q_col_first);
      q_smem->load_fragment(q_smem_offset_r_last_half, q_frag_local[1]);
      q_frag_apply_llama_rope<typename KTraits::DTypeQ, KTraits::HALF_ELEMS_PER_THREAD>(
          (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
          rope_freq[mma_di], seq_id, group_size);
      q_smem->store_fragment(q_smem_offset_r_last_half, q_frag_local[1]);
      q_smem->store_fragment(q_smem_offset_r_first_half, q_frag_local[0]);
      q_smem_offset_r_first_half =
          q_smem->template advance_offset_by_column<COL_ADVANCE_TO_NEXT, UPCAST_STRIDE_Q>(
              q_smem_offset_r_first_half, mma_di, q_col_first);
      q_col_first += COL_ADVANCE_TO_NEXT;
    }
    *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
  }
  *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const typename KTraits::IdType* q_rope_offset,
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy>* q_smem,
    const uint_fastdiv group_size, uint32_t* q_smem_offset_r, float (*rope_freq)[4],
    const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = tid.x;
    uint32_t q_frag_local[2][KTraits::INT32_ELEMS_PER_THREAD];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
    const uint32_t q_col_rope = lane_idx / KTraits::WARP_THREAD_COLS;
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
      uint32_t q_col_first = q_col_rope;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->load_fragment(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK, UPCAST_STRIDE_Q>(
                q_smem_offset_r_first_half, 0, q_col_first);
        q_smem->load_fragment(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<typename KTraits::DTypeQ, typename KTraits::IdType,
                                         KTraits::HALF_ELEMS_PER_THREAD>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di],
            q_packed_idx_base + mma_q * 16 + lane_idx / KTraits::THREADS_PER_BMATRIX_ROW_SET,
            group_size, q_rope_offset);
        q_smem->store_fragment(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->store_fragment(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half = q_smem->template advance_offset_by_column<2, UPCAST_STRIDE_Q>(
            q_smem_offset_r_first_half, mma_di, q_col_first);
        q_col_first += 2;
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base,
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy>* k_smem,
    uint32_t* k_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  using DTypeKV = typename KTraits::DTypeKV;
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  constexpr uint32_t THREADS_PER_BMATRIX_ROW_SET = KTraits::THREADS_PER_BMATRIX_ROW_SET;
  constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  uint32_t k_frag_local[2][KTraits::INT32_ELEMS_PER_THREAD];
  const uint32_t lane_idx = tid.x;
  if constexpr (KTraits::NUM_MMA_D_QK == 4 && KTraits::NUM_WARPS_Q == 4) {
    static_assert(KTraits::NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<KTraits>(tid.y);
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(KTraits::NUM_MMA_KV % 2 == 0,
                  "when NUM_MMA_D_QK == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / THREADS_PER_BMATRIX_ROW_SET;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * UPCAST_STRIDE_K;
    // After the XOR above, the logical column of *k_smem_offset_r is
    // (lane_idx/TPBRS) ^ (2*warp_idx%2).  Required by k128B_16Row for the
    // advance_offset_by_column<4> call inside the loop.
    const uint32_t k_col_rope_if =
        (lane_idx / THREADS_PER_BMATRIX_ROW_SET) ^ (0x2u * (warp_idx % 2u));
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV / 2; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->load_fragment(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4, UPCAST_STRIDE_K>(k_smem_offset_r_first_half,
                                                                        0, k_col_rope_if);
      k_smem->load_fragment(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV, HALF_ELEMS_PER_THREAD>(
          (DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1], rope_freq[mma_di], kv_idx);
      k_smem->store_fragment(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->store_fragment(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * UPCAST_STRIDE_K;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + KTraits::NUM_MMA_KV) * 16 * UPCAST_STRIDE_K;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y),
                   warp_idx_z = get_warp_idx_kv<KTraits>(tid.z);
    static_assert(KTraits::NUM_MMA_D_QK % (2 * KTraits::NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)             | 1-16   | 16-32  | 32-48  | 48-64
    // | ... | 1-16*NUM_MMA_KV                | (0, 0) | (0, 1) | (0, 2) | (0, 3)
    // | ... | 16*NUM_MMA_KV-32*NUM_MMA_KV    | (1, 0) | (1, 1) | (1, 2) | (1, 3)
    // | ...   ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * KTraits::NUM_MMA_KV * 16) +
                      lane_idx / THREADS_PER_BMATRIX_ROW_SET;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
    // Starting logical column after the XOR: j_new = (lane_idx/TPBRS) ^ (2*warp_idx_x).
    // Required by k128B_16Row for advance_offset_by_column<step> when step ∈ {2,4}.
    const uint32_t k_col_rope = (lane_idx / THREADS_PER_BMATRIX_ROW_SET) ^ (2u * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t k_col_rope_cur = k_col_rope;
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_MMA_D_QK / (2 * KTraits::NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * KTraits::NUM_WARPS_Q;
        k_smem->load_fragment(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK, UPCAST_STRIDE_K>(
                k_smem_offset_r_first_half, 0, k_col_rope_cur);
        k_smem->load_fragment(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV, HALF_ELEMS_PER_THREAD>(
            (DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1], rope_freq[mma_di], kv_idx);
        k_smem->store_fragment(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->store_fragment(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->template advance_offset_by_column<2 * KTraits::NUM_WARPS_Q, UPCAST_STRIDE_K>(
                k_smem_offset_r_first_half, mma_di, k_col_rope_cur);
        k_col_rope_cur += 2u * KTraits::NUM_WARPS_Q;
      }
      *k_smem_offset_r += 16 * UPCAST_STRIDE_K;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy>* q_smem,
    uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy>* k_smem,
    uint32_t* k_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][KTraits::HALF_ELEMS_PER_THREAD],
    const dim3 tid = threadIdx) {
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  constexpr uint32_t QK_SMEM_COLUMN_ADVANCE = 16 / KTraits::HALF_ELEMS_PER_THREAD;

  uint32_t a_frag[KTraits::NUM_MMA_Q][KTraits::INT32_ELEMS_PER_THREAD],
      b_frag[KTraits::INT32_ELEMS_PER_THREAD];

  // q_col_idx: current column j of *q_smem_offset_r before each advance_offset_by_column<4>.
  // Needed by k128B_16Row to apply the exact XOR correction; ignored by k128B.
  // Initial column = lane_idx / WARP_THREAD_COLS (the "j" dimension of the read layout).
  uint32_t q_col_idx = tid.x / KTraits::WARP_THREAD_COLS;
  // k_col_idx: same role for *k_smem_offset_r in the K read path.
  uint32_t k_col_idx = tid.x / KTraits::WARP_THREAD_COLS;

  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      q_smem->load_fragment(*q_smem_offset_r, a_frag[mma_q]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(*q_smem_offset_r);
    }

    *q_smem_offset_r =
        q_smem->template advance_offset_by_column<QK_SMEM_COLUMN_ADVANCE, UPCAST_STRIDE_Q>(
            *q_smem_offset_r, mma_d, q_col_idx) -
        KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
    q_col_idx += QK_SMEM_COLUMN_ADVANCE;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        static_assert(false, "FP8 support not yet implemented for CDNA3");
      } else {
        k_smem->load_fragment(*k_smem_offset_r, b_frag);
      }

      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(*k_smem_offset_r);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          if (mma_d == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          }
        } else if (std::is_same_v<typename KTraits::DTypeQKAccum, half>) {
          static_assert(false, "FP16 DTypeQKAccum not yet implemented for CDNA3");
        }
      }
    }
    if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
      if (mma_d % 2 == 1) {
        *k_smem_offset_r = k_smem->template advance_offset_by_column<QK_SMEM_COLUMN_ADVANCE>(
            *k_smem_offset_r, mma_d / 2);
      }
      *k_smem_offset_r -= KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    } else {
      *k_smem_offset_r =
          k_smem->template advance_offset_by_column<QK_SMEM_COLUMN_ADVANCE, UPCAST_STRIDE_K>(
              *k_smem_offset_r, mma_d, k_col_idx) -
          KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
      k_col_idx += QK_SMEM_COLUMN_ADVANCE;
    }
  }
  *q_smem_offset_r -= KTraits::NUM_MMA_D_QK * QK_SMEM_COLUMN_ADVANCE;
  *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * (QK_SMEM_COLUMN_ADVANCE);
}

template <typename KTraits, typename Params, typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint_fastdiv group_size,
    DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][KTraits::HALF_ELEMS_PER_THREAD],
    const dim3 tid = threadIdx, const uint32_t kv_head_idx = blockIdx.z) {
  constexpr uint32_t TPR = KTraits::THREADS_PER_BMATRIX_ROW_SET;
  constexpr uint32_t NAPTR = KTraits::NUM_ACCUM_ROWS_PER_THREAD;

  const uint32_t lane_idx = tid.x;
  uint32_t q[KTraits::NUM_MMA_Q][NAPTR], r[KTraits::NUM_MMA_Q][NAPTR];
  float logits = 0., logitsTransformed = 0.;

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < NAPTR; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + (lane_idx / TPR) * NAPTR + j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
        const uint32_t q_idx = q[mma_q][reg_id % NAPTR];
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][reg_id % NAPTR];
        const uint32_t kv_idx = kv_idx_base + mma_kv * 16 + (lane_idx % TPR);

#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          logits = std::bit_cast<float>(fp16_ieee_to_fp32_value(s_frag[mma_q][mma_kv][reg_id]));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          logits = s_frag[mma_q][mma_kv][reg_id];
        }
#else
        static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                      "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                      "then recompile to support fp16 reduction");
        logits = s_frag[mma_q][mma_kv][reg_id];
#endif
        logitsTransformed = variant.LogitsTransform(params, logits, batch_idx, q_idx, kv_idx,
                                                    qo_head_idx, kv_head_idx);
#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] =
              std::bit_cast<half>(fp16_ieee_from_fp32_value(logitsTransformed));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
        }
#else
        s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
#endif
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t chunk_end, const uint_fastdiv group_size,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][KTraits::HALF_ELEMS_PER_THREAD],
    const dim3 tid = threadIdx, const uint32_t kv_head_idx = blockIdx.z) {
  const uint32_t lane_idx = tid.x;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  constexpr uint32_t TPR = KTraits::THREADS_PER_BMATRIX_ROW_SET;
  constexpr uint32_t NAPTR = KTraits::NUM_ACCUM_ROWS_PER_THREAD;

  uint32_t q[NUM_MMA_Q][NAPTR], r[NUM_MMA_Q][NAPTR];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < NAPTR; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + (lane_idx / TPR) * NAPTR + j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % NAPTR)];
        const uint32_t kv_idx = kv_idx_base + mma_kv * 16 + (lane_idx % TPR);
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % NAPTR)];
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states(
    typename KTraits::AttentionVariant variant,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][KTraits::HALF_ELEMS_PER_THREAD],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    typename KTraits::DTypeQKAccum (*m)[KTraits::NUM_ACCUM_ROWS_PER_THREAD],
    float (*d)[KTraits::NUM_ACCUM_ROWS_PER_THREAD]) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD = KTraits::NUM_ACCUM_ROWS_PER_THREAD;
  constexpr bool use_softmax = AttentionVariant::use_softmax;

  if constexpr (use_softmax) {
    const float sm_scale = variant.sm_scale_log2;
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
          float m_prev = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            m[mma_q][j] = max(m[mma_q][j], s_frag[mma_q][mma_kv][j]);
          }
          // Butterfly reduction across all threads in the band
          m[mma_q][j] = max(m[mma_q][j], gpu_iface::math::shfl_xor_sync(m[mma_q][j], 0x8));
          m[mma_q][j] = max(m[mma_q][j], gpu_iface::math::shfl_xor_sync(m[mma_q][j], 0x4));
          m[mma_q][j] = max(m[mma_q][j], gpu_iface::math::shfl_xor_sync(m[mma_q][j], 0x2));
          m[mma_q][j] = max(m[mma_q][j], gpu_iface::math::shfl_xor_sync(m[mma_q][j], 0x1));
          float o_scale = gpu_iface::math::ptx_exp2(m_prev * sm_scale - m[mma_q][j] * sm_scale);
          d[mma_q][j] *= o_scale;

#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j] *= o_scale;
          }
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            s_frag[mma_q][mma_kv][j] = gpu_iface::math::ptx_exp2(
                s_frag[mma_q][mma_kv][j] * sm_scale - m[mma_q][j] * sm_scale);
          }
        }
      }
    } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
      static_assert(false, "Half precision accumulator not yet implemented for AMD");
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy>* v_smem,
    uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][KTraits::HALF_ELEMS_PER_THREAD],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    float (*d)[KTraits::NUM_ACCUM_ROWS_PER_THREAD], const dim3 tid = threadIdx) {
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  constexpr uint32_t INT32_ELEMS_PER_THREAD = KTraits::INT32_ELEMS_PER_THREAD;
  constexpr uint32_t V_SMEM_COLUMN_ADVANCE = 16 / KTraits::HALF_ELEMS_PER_THREAD;
  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV]
                                     [HALF_ELEMS_PER_THREAD];

  if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        vec_cast<typename KTraits::DTypeQ, float>::template cast<HALF_ELEMS_PER_THREAD>(
            s_frag_f16[mma_q][mma_kv], s_frag[mma_q][mma_kv]);
      }
    }
  }

// In-place transposition of the s_frag MMA tile to get the data into CDNA3 A-matrix layout.
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      mma::transpose_mma_tile(reinterpret_cast<uint32_t*>(s_frag_f16[mma_q][mma_kv]));
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);
        } else {
          static_assert(!std::is_same_v<typename KTraits::DTypeQKAccum, __half>,
                        "FP16 reduction path not implemented for CDNA3");
        }
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
    // v_col_idx: current column j of *v_smem_offset_r before each advance_offset_by_column.
    // Reset per KV row: each row's V fragment starts at column tid.x / WARP_THREAD_COLS.
    // Needed by k128B_16Row; ignored by k128B.
    uint32_t v_col_idx = tid.x / KTraits::WARP_THREAD_COLS;
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
      uint32_t b_frag[INT32_ELEMS_PER_THREAD];
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        static_assert(false, "FP8 V path not implemented for CDNA3 yet");
      } else {
        v_smem->load_matrix_m16n16_trans(*v_smem_offset_r, b_frag);
      }
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
        }
      }
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
          *v_smem_offset_r = v_smem->template advance_offset_by_column<V_SMEM_COLUMN_ADVANCE>(
              *v_smem_offset_r, mma_d / 2);
        }
      } else {
        *v_smem_offset_r =
            v_smem->template advance_offset_by_column<V_SMEM_COLUMN_ADVANCE, UPCAST_STRIDE_V>(
                *v_smem_offset_r, mma_d, v_col_idx);
        v_col_idx += V_SMEM_COLUMN_ADVANCE;
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V>(*v_smem_offset_r) -
        V_SMEM_COLUMN_ADVANCE * KTraits::NUM_MMA_D_VO;
  }
  *v_smem_offset_r -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V;
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    typename KTraits::DTypeQKAccum (*m)[KTraits::NUM_ACCUM_ROWS_PER_THREAD],
    float (*d)[KTraits::NUM_ACCUM_ROWS_PER_THREAD]) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr uint32_t NAPTR = KTraits::NUM_ACCUM_ROWS_PER_THREAD;

  if constexpr (AttentionVariant::use_softmax) {
    float d_rcp[KTraits::NUM_MMA_Q][KTraits::NUM_ACCUM_ROWS_PER_THREAD];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_ACCUM_ROWS_PER_THREAD; ++j) {
        d_rcp[mma_q][j] = (m[mma_q][j] != typename KTraits::DTypeQKAccum(-gpu_iface::math::inf))
                              ? gpu_iface::math::ptx_rcp(d[mma_q][j])
                              : 0.f;
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
          o_frag[mma_q][mma_d][reg_id] =
              o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][reg_id % NAPTR];
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m(
    typename KTraits::AttentionVariant variant,
    typename KTraits::DTypeQKAccum (*m)[KTraits::NUM_ACCUM_ROWS_PER_THREAD]) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_ACCUM_ROWS_PER_THREAD; ++j) {
        if (m[mma_q][j] != typename KTraits::DTypeQKAccum(-gpu_iface::math::inf)) {
          m[mma_q][j] *= variant.sm_scale_log2;
        }
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along threadIdx.z.
 */
template <typename KTraits>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    typename KTraits::SharedStorage* smem_storage,
    typename KTraits::DTypeQKAccum (*m)[KTraits::NUM_ACCUM_ROWS_PER_THREAD],
    float (*d)[KTraits::NUM_ACCUM_ROWS_PER_THREAD], const uint32_t warp_idx,
    const uint32_t lane_idx, const dim3 tid = threadIdx) {
  constexpr uint32_t THREADS_PER_LANE_GROUP = KTraits::THREADS_PER_BMATRIX_ROW_SET;
  constexpr uint32_t NARPT = KTraits::NUM_ACCUM_ROWS_PER_THREAD;

  static_assert(WARP_SIZE % THREADS_PER_LANE_GROUP == 0,
                "THREADS_PER_BMATRIX_ROW_SET must divide WARP_SIZE");
  constexpr uint32_t GROUPS_PER_WARP = WARP_SIZE / THREADS_PER_LANE_GROUP;
  const uint32_t ln_grp_idx = lane_idx / THREADS_PER_LANE_GROUP;

  // only necessary when blockDim.z > 1
  if constexpr (KTraits::NUM_WARPS_KV > 1) {
    float* smem_o = smem_storage->cta_sync_o_smem;
    float2* smem_md = smem_storage->cta_sync_md_smem;
    // o: [num_warps,
    //     NUM_MMA_Q,
    //     NUM_MMA_D_VO,
    //     WARP_SIZE,
    //     HALF_ELEMS_PER_THREAD]
    // md: [num_warps, NUM_MMA_Q, 16, 2 (m/d)]
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        // Write o_frag to smem_o with transposed layout (column-major storage)
        // to enable efficient contiguous writes despite column-major register layout.
        // smem_o layout: [warp][mma_q][col][row] where col varies fastest
        // Index breakdown:
        //   - warp_idx * (NUM_MMA_Q * CTA_TILE_Q * HEAD_DIM_VO): warp partition offset
        //   - mma_q * (CTA_TILE_Q * HEAD_DIM_VO): mma_q tile offset within warp
        //   - mma_d * 256: mma_d tile offset (each tile is 16x16)
        //   - (lane_idx / 16) * 4: row group offset (4 contiguous rows per thread group)
        //   - (lane_idx % 16) * 16: column stride (16 threads per col, CTA_TILE_Q rows per col)
        const uint32_t smem_o_idx =
            warp_idx * (KTraits::NUM_MMA_Q * KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) +
            mma_q * (KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) + mma_d * (16 * 16) +
            (lane_idx / THREADS_PER_LANE_GROUP) * KTraits::HALF_ELEMS_PER_THREAD +
            (lane_idx % THREADS_PER_LANE_GROUP) * THREADS_PER_LANE_GROUP;
        vec_t<float, KTraits::HALF_ELEMS_PER_THREAD>::memcpy(smem_o + smem_o_idx,
                                                             o_frag[mma_q][mma_d]);
      }
    }

    if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < NARPT; ++j) {
          auto warp_offset = warp_idx * KTraits::NUM_MMA_Q;
          auto row_offset = warp_offset + mma_q;
          smem_md[row_offset * THREADS_PER_LANE_GROUP + ln_grp_idx * NARPT + j] =
              make_float2(float(m[mma_q][j]), d[mma_q][j]);
        }
      }
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        float o_scale[NARPT][KTraits::NUM_WARPS_KV];
#pragma unroll
        for (uint32_t j = 0; j < NARPT; ++j) {
          float m_new = -gpu_iface::math::inf, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[i * KTraits::NUM_MMA_Q * 16 + mma_q * 16 + ln_grp_idx * NARPT + j];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * gpu_iface::math::ptx_exp2(m_prev - m_new) +
                    md.y * gpu_iface::math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[i * KTraits::NUM_MMA_Q * 16 + mma_q * 16 + ln_grp_idx * NARPT + j];
            float mi = md.x;
            o_scale[j][i] = gpu_iface::math::ptx_exp2(float(mi - m_new));
          }
          m[mma_q][j] = typename KTraits::DTypeQKAccum(m_new);
          d[mma_q][j] = d_new;
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, KTraits::HALF_ELEMS_PER_THREAD> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, KTraits::HALF_ELEMS_PER_THREAD> oi;
            const uint32_t smem_o_read_idx =
                i * (KTraits::NUM_MMA_Q * KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) +
                mma_q * (KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) + mma_d * (16 * 16) +
                (lane_idx / THREADS_PER_LANE_GROUP) * KTraits::HALF_ELEMS_PER_THREAD +
                (lane_idx % THREADS_PER_LANE_GROUP) * THREADS_PER_LANE_GROUP;
            oi.load(smem_o + smem_o_read_idx);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
              // CDNA3: Direct mapping - each reg_id corresponds to one accumulator row
              o_new[reg_id] += oi[reg_id] * o_scale[reg_id][i];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    } else {
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, KTraits::HALF_ELEMS_PER_THREAD> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, KTraits::HALF_ELEMS_PER_THREAD> oi;
            const uint32_t smem_o_read_idx =
                i * (KTraits::NUM_MMA_Q * KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) +
                mma_q * (KTraits::CTA_TILE_Q * KTraits::HEAD_DIM_VO) + mma_d * (16 * 16) +
                (lane_idx / THREADS_PER_LANE_GROUP) * KTraits::HALF_ELEMS_PER_THREAD +
                (lane_idx % THREADS_PER_LANE_GROUP) * THREADS_PER_LANE_GROUP;
            oi.load(smem_o + smem_o_read_idx);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id) {
              o_new[reg_id] += oi[reg_id];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][KTraits::HALF_ELEMS_PER_THREAD],
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size, const dim3 tid = threadIdx) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  constexpr uint32_t TPR = KTraits::THREADS_PER_BMATRIX_ROW_SET;
  constexpr uint32_t NAPTR = KTraits::NUM_ACCUM_ROWS_PER_THREAD;
  constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  constexpr uint32_t WARP_THREAD_COLS = KTraits::WARP_THREAD_COLS;
  constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;
  constexpr uint32_t COLUMN_RESET_OFFSET = KTraits::NUM_MMA_D_VO / 4 * WARP_THREAD_COLS;

  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y);
  const uint32_t lane_idx = tid.x;

  if constexpr (sizeof(DTypeO) == 4) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < NAPTR; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / TPR + mma_q * 16 + j * 8, q, r);
        const uint32_t o_idx = q;
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          if (o_idx < qo_upper_bound) {
            auto base_addr = o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16;
            auto col_offset = lane_idx % 16;
            *(base_addr + col_offset) = o_frag[mma_q][mma_d][j];
          }
        }
      }
    }
  } else {
    if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          uint32_t o_frag_f16[HALF_ELEMS_PER_THREAD / 2];
          vec_cast<DTypeO, float>::template cast<HALF_ELEMS_PER_THREAD>((DTypeO*)o_frag_f16,
                                                                        o_frag[mma_q][mma_d]);
          const int lane_id_in_warp = tid.x % WARP_SIZE;
          const int warp_idx_q = get_warp_idx_q<KTraits>(tid.y);
          const uint32_t warp_base_row = warp_idx_q * KTraits::NUM_MMA_Q * 16;
          const uint32_t frag_row_offset = mma_q * 16;
          const uint32_t frag_col_offset = mma_d * 16;
          const uint32_t thread_start_row_in_frag = (lane_id_in_warp / 16) * NAPTR;
          const uint32_t thread_col_in_frag = (lane_id_in_warp % 16);
          // Calculate base row (the first of 4 rows this thread writes)
          const uint32_t base_row = warp_base_row + frag_row_offset + thread_start_row_in_frag;
          // Column in units of 4-element vectors
          const uint32_t col_vec = (frag_col_offset + thread_col_in_frag) / HALF_ELEMS_PER_THREAD;
          // Index within the 4-element vector (0-3)
          const uint32_t col_idx = (frag_col_offset + thread_col_in_frag) % HALF_ELEMS_PER_THREAD;
          // Cast to DTypeO* and write all 4 elements with swizzled addressing
          DTypeO* o_frag_f16_half = reinterpret_cast<DTypeO*>(o_frag_f16);
          // Calculate column index in DTypeO units
          const uint32_t col_dtype = col_vec * HALF_ELEMS_PER_THREAD + col_idx;
          // Convert to BasePtrTy (uint2) units: 1 uint2 = 4 DTypeO elements (for fp16)
          constexpr uint32_t elems_per_base_ptr_type =
              sizeof(typename KTraits::SmemBasePtrTy) / sizeof(DTypeO);
          const uint32_t col_base = col_dtype / elems_per_base_ptr_type;
          const uint32_t col_offset = col_dtype % elems_per_base_ptr_type;
          // Write each of the 4 rows handled by this thread using swizzled offsets
          for (uint32_t row_offset = 0; row_offset < HALF_ELEMS_PER_THREAD; ++row_offset) {
            const uint32_t row = base_row + row_offset;
            uint32_t swizzled_offset =
                o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(row, col_base);
            DTypeO* o_smem_typed = reinterpret_cast<DTypeO*>(o_smem->base + swizzled_offset);
            o_smem_typed[col_offset] = o_frag_f16_half[row_offset];
          }
        }
      }

      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / WARP_THREAD_COLS,
          lane_idx % WARP_THREAD_COLS);
      // row_idx_ow mirrors the row_idx_w tracking in load_q_global_smem.
      // Required by k128B_16Row advance_offset_by_row<4>; ignored by k128B.
      uint32_t row_idx_ow = warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / WARP_THREAD_COLS;

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2 * 2; ++j) {
          uint32_t q, r;
          group_size.divmod(o_packed_idx_base + lane_idx / WARP_THREAD_COLS + mma_q * 16 + j * 4, q,
                            r);
          const uint32_t o_idx = q;
          DTypeO* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h +
                          (lane_idx % WARP_THREAD_COLS) * upcast_size<DTypeO, VECTOR_BIT_WIDTH>();
#pragma unroll
          for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_VO / 4; ++mma_do) {
            if (o_idx < qo_upper_bound) {
              o_smem->store_vector(o_smem_offset_w, o_ptr);
            }
            o_ptr += WARP_THREAD_COLS * upcast_size<DTypeO, VECTOR_BIT_WIDTH>();
            o_smem_offset_w = o_smem->template advance_offset_by_column<WARP_THREAD_COLS>(
                o_smem_offset_w, mma_do);
          }
          o_smem_offset_w = o_smem->template advance_offset_by_row<4, UPCAST_STRIDE_O>(
                                o_smem_offset_w, row_idx_ow) -
                            COLUMN_RESET_OFFSET;
          row_idx_ow += 4;
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
 * \tparam POS_ENCODING_MODE The positional encoding mode.
 * \tparam NUM_MMA_Q The number of fragments in x dimension.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
 * \tparam DTypeO The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param rope_rcp_scale 1/(rope_scale), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param rope_rcp_theta 1/(rope_theta), where rope_theta is the theta
 *   used in RoPE.
 */
template <typename KTraits, typename Params>
__device__ __forceinline__ void SinglePrefillWithKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t chunk_idx = blockIdx.y,
    const uint32_t kv_head_idx = blockIdx.z, const uint32_t num_chunks = gridDim.y,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  [[maybe_unused]] constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD =
      KTraits::NUM_ACCUM_ROWS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t THREADS_PER_BMATRIX_ROW_SET =
      KTraits::THREADS_PER_BMATRIX_ROW_SET;
  [[maybe_unused]] constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  DTypeQ* q = params.q;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const uint32_t qo_len = params.qo_len;
  const uint32_t kv_len = params.kv_len;
  const bool partition_kv = params.partition_kv;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t k_stride_h = params.k_stride_h;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t v_stride_h = params.v_stride_h;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
  const uint32_t num_qo_heads = num_kv_heads * group_size;

  const uint32_t max_chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
  const uint32_t chunk_start = partition_kv ? chunk_idx * max_chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min((chunk_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  auto block = cg::this_thread_block();
  auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t window_left = variant.window_left;

  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][HALF_ELEMS_PER_THREAD];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][HALF_ELEMS_PER_THREAD];
  DTypeQKAccum m[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float d[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float rope_freq[NUM_MMA_D_QK / 2][4];
  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;
    init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
  }
  init_states<KTraits>(variant, o_frag, m, d);

  // cooperative fetch q fragment from gmem to reg
  const uint32_t qo_packed_idx_base =
      (bx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
  smem_t<SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
  DTypeQ* q_ptr_base = q + (kv_head_idx * group_size) * q_stride_h;
  DTypeO* o_ptr_base = partition_kv
                           ? o + chunk_idx * o_stride_n + (kv_head_idx * group_size) * o_stride_h
                           : o + (kv_head_idx * group_size) * o_stride_h;

  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_ptr_base, q_stride_n, q_stride_h,
                              group_size, &qo_smem, tid);

  uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

  memory::commit_group();
  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    memory::wait_group<0>();
    block.sync();
    q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                         &q_smem_offset_r, rope_freq, tid);
    block.sync();
  }

  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> k_smem(smem_storage.k_smem);
  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> v_smem(smem_storage.v_smem);

  const uint32_t num_iterations =
      ceil_div(MASK_MODE == MaskMode::kCausal
                   ? min(chunk_size,
                         sub_if_greater_or_zero(
                             kv_len - qo_len + ((bx + 1) * CTA_TILE_Q) / group_size, chunk_start))
                   : chunk_size,
               CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size, sub_if_greater_or_zero(
                                 kv_len + (bx * CTA_TILE_Q) / group_size - qo_len, chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

  DTypeKV* k_ptr =
      k + (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * k_stride_n +
      kv_head_idx * k_stride_h +
      (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();

  DTypeKV* v_ptr =
      v + (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * v_stride_n +
      kv_head_idx * v_stride_h +
      (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();

  uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, (lane_idx / 16));
  uint32_t v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16);
  uint32_t k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL),
           v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL);
  produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                         k_stride_n, 0, chunk_size, tid);
  memory::commit_group();
  produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                          v_stride_n, 0, chunk_size, tid);
  memory::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    memory::wait_group<1>();
    block.sync();
    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<KTraits>(chunk_start + iter * CTA_TILE_KV, &k_smem,
                                           &k_smem_offset_r, rope_freq, tid);
      block.sync();
    }
    // compute attention score
    compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag, tid);
    // logits transformation
    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/0, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
        qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);
    // apply mask
    if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
      logits_mask<KTraits>(
          params, variant, /*batch_idx=*/0, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
    }
    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
    block.sync();
    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
        k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
    memory::commit_group();
    memory::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d, tid);
    block.sync();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
        v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
    memory::commit_group();
  }
  memory::wait_group<0>();
  block.sync();

  finalize_m<KTraits>(variant, m);
  // threadblock synchronization
  threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);
  // normalize d
  normalize_d<KTraits>(o_frag, m, d);
  // write back
  write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                            /*o_stride_n=*/
                            partition_kv ? num_chunks * o_stride_n : o_stride_n,
                            /*o_stride_h=*/o_stride_h, group_size, tid);
  // write lse
  if constexpr (variant.use_softmax) {
    if (lse != nullptr || partition_kv) {
      if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
            uint32_t q, r;

            group_size.divmod(
                qo_packed_idx_base +
                    (lane_idx / THREADS_PER_BMATRIX_ROW_SET) * NUM_ACCUM_ROWS_PER_THREAD + j +
                    mma_q * 16,
                q, r);
            const uint32_t qo_head_idx = kv_head_idx * group_size + r;
            const uint32_t qo_idx = q;
            if (qo_idx < qo_len) {
              if (partition_kv) {
                lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                    gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              } else {
                lse[qo_idx * num_qo_heads + qo_head_idx] =
                    gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              }
            }
          }
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void SinglePrefillWithKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  SinglePrefillWithKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
gpuError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              gpuStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t qo_len = params.qo_len;
  const uint32_t kv_len = params.kv_len;
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be "
               "greater than or equal to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  constexpr uint32_t ELEMS_PER_FRAGMENT = 16 * 16 / WARP_SIZE;
  int64_t packed_qo_len = qo_len * group_size;
  uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO);

  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                  float>::type;

    int dev_id = 0;
    FI_GPU_CALL(gpuGetDevice(&dev_id));
    const int max_smem_per_threadblock = getMaxSharedMemPerBlock(dev_id);
    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (ELEMS_PER_FRAGMENT / NUM_MMA_Q);
    // On HIP (CDNA3), cap KV smem at half of LDS per CU to allow 2 workgroups/CU.
    // Without this cap, CTA_TILE_Q=64 savings in q_smem are automatically
    // consumed by a larger NUM_MMA_KV, keeping smem at 48 KB (1 block/CU).
    // With the cap, CTA_TILE_Q=64+head_dim=128 → 32 KB smem → 2 blocks/CU.
    // Always use at least min_valid_mma_kv (IsInvalid: NUM_MMA_D_VO==4 → must be even).
#if defined(PLATFORM_HIP_DEVICE)
    const uint32_t q_smem_bytes_ = CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
    const uint32_t kv_budget_ =
        (static_cast<uint32_t>(max_smem_per_threadblock) / 2u > q_smem_bytes_)
            ? static_cast<uint32_t>(max_smem_per_threadblock) / 2u - q_smem_bytes_
            : 0u;
    constexpr uint32_t min_valid_mma_kv_ = (HEAD_DIM_VO / 16u == 4u) ? 2u : 1u;
#else
    const uint32_t kv_budget_ =
        static_cast<uint32_t>(max_smem_per_threadblock) -
        CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
    constexpr uint32_t min_valid_mma_kv_ = 1u;
#endif
    const uint32_t max_num_mma_kv_smem = std::max(
        min_valid_mma_kv_, static_cast<uint32_t>(kv_budget_ / ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 *
                                                               NUM_WARPS_KV * sizeof(DTypeKV))));

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      using KTraits =
          KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                       NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                       DTypeQKAccum, typename Params::IdType, AttentionVariant>;
      if constexpr (KTraits::IsInvalid()) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid "
                   "configuration : NUM_MMA_Q="
                << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
                << " NUM_WARPS_KV=" << NUM_WARPS_KV
                << " please create an issue "
                   "(https://github.com/flashinfer-ai/flashinfer/"
                   "issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
        auto kernel = SinglePrefillWithKVCacheKernel<KTraits, Params>;
        size_t smem_size = sizeof(typename KTraits::SharedStorage);
        // Check if shared memory requirement exceeds hardware limit (important for AMD GPUs with
        // 64KB limit)
        if (smem_size > static_cast<size_t>(max_smem_per_threadblock)) {
          std::ostringstream err_msg;
          err_msg << "FlashInfer: Shared memory requirement (" << smem_size
                  << " bytes) exceeds hardware limit (" << max_smem_per_threadblock
                  << " bytes). Consider using smaller head_dim or CTA_TILE_Q.";
          FLASHINFER_ERROR(err_msg.str());
        }
        FI_GPU_CALL(
            gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FI_GPU_CALL(gpuDeviceGetAttribute(&num_sm, gpuDevAttrMultiProcessorCount, dev_id));
        FI_GPU_CALL(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                 num_threads, smem_size));
        uint32_t max_num_kv_chunks = (num_blocks_per_sm * num_sm) /
                                     (num_kv_heads * ceil_div(qo_len * group_size, CTA_TILE_Q));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          params.partition_kv = false;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1, num_kv_heads);
          dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);
          FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          params.partition_kv = true;
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
          auto o = params.o;
          auto lse = params.lse;
          params.o = tmp;
          params.lse = tmp_lse;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), num_chunks, num_kv_heads);
          dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);
          FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          if constexpr (AttentionVariant::use_softmax) {
            FI_GPU_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                    HEAD_DIM_VO, stream));
          } else {
            FI_GPU_CALL(
                AttentionSum(tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
          }
        }
      }
    })
  });
  return gpuSuccess;
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  [[maybe_unused]] constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD =
      KTraits::NUM_ACCUM_ROWS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t THREADS_PER_BMATRIX_ROW_SET =
      KTraits::THREADS_PER_BMATRIX_ROW_SET;
  [[maybe_unused]] constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  DTypeQ* q = params.q;
  IdType* request_indices = params.request_indices;
  IdType* qo_tile_indices = params.qo_tile_indices;
  IdType* kv_tile_indices = params.kv_tile_indices;
  IdType* q_indptr = params.q_indptr;
  IdType* kv_indptr = params.kv_indptr;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  IdType* o_indptr = params.o_indptr;
  DTypeO* o = params.o;
  float* lse = params.lse;
  bool* block_valid_mask = params.block_valid_mask;
  const bool partition_kv = params.partition_kv;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t k_stride_h = params.k_stride_h;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t v_stride_h = params.v_stride_h;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const dim3& tid = threadIdx;

  auto block = cg::this_thread_block();
  const uint32_t bx = blockIdx.x, lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z),
                 kv_head_idx = blockIdx.z;
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = group_size * num_kv_heads;
  const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];

  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][HALF_ELEMS_PER_THREAD];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][HALF_ELEMS_PER_THREAD];
  DTypeQKAccum m[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float d[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float rope_freq[NUM_MMA_D_QK / 2][4];

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;
    init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
  }
  init_states<KTraits>(variant, o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

  DTypeQ* q_ptr_base =
      q + q_indptr[request_idx] * q_stride_n + kv_head_idx * group_size * q_stride_h;

  DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h
                                    : o + o_indptr[request_idx] * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h;

  uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                              q_stride_h, group_size, &qo_smem, tid);

  memory::commit_group();

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    memory::wait_group<0>();
    block.sync();
    IdType* q_rope_offset = nullptr;

    if constexpr (has_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.maybe_q_rope_offset;
    }
    if (!q_rope_offset) {
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq, tid);
    } else {
      q_smem_inplace_apply_rotary_with_pos<KTraits>(qo_packed_idx_base,
                                                    q_rope_offset + q_indptr[request_idx], &qo_smem,
                                                    group_size, &q_smem_offset_r, rope_freq, tid);
    }
    block.sync();
  }

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size, chunk_start))
           : chunk_size),
      CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                        chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> k_smem(smem_storage.k_smem);
  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> v_smem(smem_storage.v_smem);
  uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, (lane_idx / 16));

  uint32_t v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16);

  uint32_t k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL),
           v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL);

  DTypeKV* k_ptr = k +
                   (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                    lane_idx / KV_THR_LAYOUT_COL) *
                       k_stride_n +
                   kv_head_idx * k_stride_h +
                   (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
  DTypeKV* v_ptr = v +
                   (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                    lane_idx / KV_THR_LAYOUT_COL) *
                       v_stride_n +
                   kv_head_idx * v_stride_h +
                   (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();

  produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                         k_stride_n, 0, chunk_size, tid);
  memory::commit_group();
  produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                          v_stride_n, 0, chunk_size, tid);

  memory::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    memory::wait_group<1>();
    block.sync();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      IdType* k_rope_offset = nullptr;
      if constexpr (has_maybe_k_rope_offset_v<Params>) {
        k_rope_offset = params.maybe_k_rope_offset;
      }
      k_smem_inplace_apply_rotary<KTraits>(
          (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
              iter * CTA_TILE_KV,
          &k_smem, &k_smem_offset_r, rope_freq, tid);
      block.sync();
    }

    // compute attention score
    compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
        qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

    // apply mask
    if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
      logits_mask<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
    }

    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

    block.sync();
    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
        k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
    memory::commit_group();
    memory::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

    block.sync();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
        v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
    memory::commit_group();
  }
  memory::wait_group<0>();
  block.sync();

  finalize_m<KTraits>(variant, m);

  // threadblock synchronization
  threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

  // normalize d
  normalize_d<KTraits>(o_frag, m, d);

  const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write back
  write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                            /*o_stride_n=*/
                            partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                            /*o_stride_h=*/o_stride_h, group_size, tid);

  // write lse
  if constexpr (AttentionVariant::use_softmax) {
    if (lse != nullptr) {
      if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
            uint32_t q, r;
            group_size.divmod(
                qo_packed_idx_base +
                    (lane_idx / THREADS_PER_BMATRIX_ROW_SET) * NUM_ACCUM_ROWS_PER_THREAD + j +
                    mma_q * 16,
                q, r);
            const uint32_t qo_head_idx = kv_head_idx * group_size + r;
            const uint32_t qo_idx = q;
            if (qo_idx < qo_len) {
              if (partition_kv) {
                lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                    qo_head_idx] = gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              } else {
                lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                    gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              }
            }
          }
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t kv_head_idx = blockIdx.z,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  [[maybe_unused]] constexpr uint32_t HALF_ELEMS_PER_THREAD = KTraits::HALF_ELEMS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t NUM_ACCUM_ROWS_PER_THREAD =
      KTraits::NUM_ACCUM_ROWS_PER_THREAD;
  [[maybe_unused]] constexpr uint32_t THREADS_PER_BMATRIX_ROW_SET =
      KTraits::THREADS_PER_BMATRIX_ROW_SET;
  [[maybe_unused]] constexpr uint32_t VECTOR_BIT_WIDTH = KTraits::VECTOR_BIT_WIDTH;

  IdType* request_indices = params.request_indices;
  IdType* qo_tile_indices = params.qo_tile_indices;
  IdType* kv_tile_indices = params.kv_tile_indices;
  DTypeQ* q = params.q;
  IdType* q_indptr = params.q_indptr;
  IdType* o_indptr = params.o_indptr;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
  bool* block_valid_mask = params.block_valid_mask;
  const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
  const bool partition_kv = params.partition_kv;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  auto block = cg::this_thread_block();
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

  const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_qo_heads = num_kv_heads * group_size;

  const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));
  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][HALF_ELEMS_PER_THREAD];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][HALF_ELEMS_PER_THREAD];
  DTypeQKAccum m[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float d[NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD];
  float rope_freq[NUM_MMA_D_QK / 2][4];

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;
    init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
  }
  init_states<KTraits>(variant, o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;

  smem_t<SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
  DTypeQ* q_ptr_base =
      q + q_indptr[request_idx] * q_stride_n + (kv_head_idx * group_size) * q_stride_h;
  DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h
                                    : o + o_indptr[request_idx] * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h;
  uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                              q_stride_h, group_size, &qo_smem, tid);

  memory::commit_group();

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    memory::wait_group<0>();
    block.sync();
    IdType* q_rope_offset = nullptr;
    if constexpr (has_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.maybe_q_rope_offset;
    }
    if (q_rope_offset == nullptr) {
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq, tid);
    } else {
      q_smem_inplace_apply_rotary_with_pos<KTraits>(qo_packed_idx_base,
                                                    q_rope_offset + q_indptr[request_idx], &qo_smem,
                                                    group_size, &q_smem_offset_r, rope_freq, tid);
    }
    block.sync();
  }

  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> k_smem(smem_storage.k_smem);
  smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> v_smem(smem_storage.v_smem);

  // The thr_local_kv_offset array stores the offsets into the paged kv cache for each
  // thread. The size of the array should be equal to the trip count of the initialization loop.
  size_t thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_ROW / NUM_WARPS_Q];

  uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, (lane_idx / 16));
  uint32_t v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
      get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16);

  uint32_t k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL),
           v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
               warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
               lane_idx % KV_THR_LAYOUT_COL);

  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  uint32_t packed_page_iter_base = paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * KV_THR_LAYOUT_ROW / NUM_WARPS_Q; ++i) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                  lane_idx / KV_THR_LAYOUT_COL +
                                  KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                              page_iter, entry_idx);
    thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
        page_iter, kv_head_idx, entry_idx,
        (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>(), last_indptr);
  }
  page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                  chunk_size, tid);
  memory::commit_group();
  page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                 chunk_size, tid);
  memory::commit_group();

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size, chunk_start))
           : chunk_size),
      CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                        chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    packed_page_iter_base += CTA_TILE_KV;
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * KV_THR_LAYOUT_ROW / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                    lane_idx / KV_THR_LAYOUT_COL +
                                    KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>(), last_indptr);
    }
    memory::wait_group<1>();
    block.sync();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<KTraits>(
          (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
              chunk_start + iter * CTA_TILE_KV,
          &k_smem, &k_smem_offset_r, rope_freq, tid);
      block.sync();
    }

    // compute attention score
    compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag, tid);

    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
        qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

    // apply mask
    if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
      logits_mask<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
    }

    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

    block.sync();
    page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                    thr_local_kv_offset, chunk_size, tid);
    memory::commit_group();
    memory::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d, tid);

    block.sync();
    page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                   thr_local_kv_offset, chunk_size, tid);
    memory::commit_group();
  }
  memory::wait_group<0>();
  block.sync();

  finalize_m<KTraits>(variant, m);

  // threadblock synchronization
  threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

  // normalize d
  normalize_d<KTraits>(o_frag, m, d);

#ifdef PLATFORM_HIP_DEVICE
  // Cascade epilogue: merge with a prior cascade level's output in-register.
  // Skipped for split-KV chunks (partition_kv=true); those are merged in
  // BatchPrefillWithPagedKVCacheDispatched after VariableLengthMergeStates.
  if constexpr (AttentionVariant::use_softmax) {
    if (params.partial_o != nullptr && !partition_kv) {
      if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
            uint32_t q_idx, r;
            group_size.divmod(
                qo_packed_idx_base +
                    (lane_idx / THREADS_PER_BMATRIX_ROW_SET) * NUM_ACCUM_ROWS_PER_THREAD + j +
                    mma_q * 16,
                q_idx, r);
            const uint32_t qo_head_idx = kv_head_idx * group_size + r;
            const uint32_t qo_idx = q_idx;
            if (qo_idx < qo_upper_bound) {
              const float s_cur = gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              const float s_partial =
                  params.partial_lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx];
              const float s_max = fmaxf(s_cur, s_partial);
              const float scale_a = exp2f(s_cur - s_max);
              const float scale_b = exp2f(s_partial - s_max);
              const float inv_denom = 1.0f / (scale_a + scale_b);
              const uint32_t po_base =
                  (o_indptr[request_idx] + qo_idx) * o_stride_n + qo_head_idx * o_stride_h;
#pragma unroll
              for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
                const float p_o =
                    (float)params
                        .partial_o[po_base + mma_d * 16 + lane_idx % THREADS_PER_BMATRIX_ROW_SET];
                o_frag[mma_q][mma_d][j] =
                    (o_frag[mma_q][mma_d][j] * scale_a + p_o * scale_b) * inv_denom;
              }
              m[mma_q][j] = static_cast<DTypeQKAccum>(s_max);
              d[mma_q][j] = scale_a + scale_b;
            }
          }
        }
      }
    }
  }
#endif  // PLATFORM_HIP_DEVICE

  const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write_back
  write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                            /*o_stride_n=*/
                            partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                            /*o_stride_h=*/o_stride_h, group_size, tid);

  // write lse
  if constexpr (variant.use_softmax) {
    if (lse != nullptr) {
      if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
            uint32_t q, r;
            group_size.divmod(
                qo_packed_idx_base +
                    (lane_idx / THREADS_PER_BMATRIX_ROW_SET) * NUM_ACCUM_ROWS_PER_THREAD + j +
                    mma_q * 16,
                q, r);
            const uint32_t qo_head_idx = kv_head_idx * group_size + r;
            const uint32_t qo_idx = q;
            if (qo_idx < qo_upper_bound) {
              if (partition_kv) {
                lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                    qo_head_idx] = gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              } else {
                lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                    gpu_iface::math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
              }
            }
          }
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
gpuError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, gpuStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;

  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the
    // padded_batch_size
    return gpuSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  constexpr uint32_t ELEMS_PER_FRAGMENT = 16 * 16 / WARP_SIZE;

  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FI_GPU_CALL(gpuGetDevice(&dev_id));
  const int max_smem_per_threadblock = getMaxSharedMemPerBlock(dev_id);

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (ELEMS_PER_FRAGMENT / NUM_MMA_Q);

  // On HIP (CDNA3), cap KV smem at half of LDS per CU to allow 2 workgroups/CU.
  // Without this cap, CTA_TILE_Q=64 savings in q_smem are automatically
  // consumed by a larger NUM_MMA_KV, keeping smem at 48 KB (1 block/CU).
  // With the cap, CTA_TILE_Q=64+head_dim=128 → 32 KB smem → 2 blocks/CU.
  // Always use at least min_valid_mma_kv (IsInvalid: NUM_MMA_D_VO==4 → must be even).
#if defined(PLATFORM_HIP_DEVICE)
  const uint32_t q_smem_bytes_ = CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
  const uint32_t kv_budget_ =
      (static_cast<uint32_t>(max_smem_per_threadblock) / 2u > q_smem_bytes_)
          ? static_cast<uint32_t>(max_smem_per_threadblock) / 2u - q_smem_bytes_
          : 0u;
  constexpr uint32_t min_valid_mma_kv_ = (HEAD_DIM_VO / 16u == 4u) ? 2u : 1u;
#else
  const uint32_t kv_budget_ =
      static_cast<uint32_t>(max_smem_per_threadblock) - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
  constexpr uint32_t min_valid_mma_kv_ = 1u;
#endif
  const uint32_t max_num_mma_kv_smem = std::max(
      min_valid_mma_kv_, static_cast<uint32_t>(kv_budget_ / ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 *
                                                             NUM_WARPS_KV * sizeof(DTypeKV))));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : "
                 "NUM_MMA_Q="
              << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue "
                 "(https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithRaggedKVCacheKernel<KTraits, Params>;
      FI_GPU_CALL(
          gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // partition kv
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
        FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FI_GPU_CALL(VariableLengthMergeStates(tmp_v, tmp_s, params.merge_indptr, o, lse,
                                                params.max_total_num_rows, params.total_num_rows,
                                                num_qo_heads, HEAD_DIM_VO, stream));
        } else {
          FI_GPU_CALL(VariableLengthAttentionSum(tmp_v, params.merge_indptr, o,
                                                 params.max_total_num_rows, params.total_num_rows,
                                                 num_qo_heads, HEAD_DIM_VO, stream));
        }
      }
    }
  });
  return gpuSuccess;
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
gpuError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, gpuStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the
    // padded_batch_size
    return gpuSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  constexpr uint32_t ELEMS_PER_FRAGMENT = 16 * 16 / WARP_SIZE;

  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FI_GPU_CALL(gpuGetDevice(&dev_id));
  const int max_smem_per_threadblock = getMaxSharedMemPerBlock(dev_id);

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (ELEMS_PER_FRAGMENT / NUM_MMA_Q);

  // On HIP (CDNA3), cap KV smem at half of LDS per CU to allow 2 workgroups/CU.
  // Without this cap, CTA_TILE_Q=64 savings in q_smem are automatically
  // consumed by a larger NUM_MMA_KV, keeping smem at 48 KB (1 block/CU).
  // With the cap, CTA_TILE_Q=64+head_dim=128 → 32 KB smem → 2 blocks/CU.
  // Always use at least min_valid_mma_kv (IsInvalid: NUM_MMA_D_VO==4 → must be even).
#if defined(PLATFORM_HIP_DEVICE)
  const uint32_t q_smem_bytes_ = CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
  const uint32_t kv_budget_ =
      (static_cast<uint32_t>(max_smem_per_threadblock) / 2u > q_smem_bytes_)
          ? static_cast<uint32_t>(max_smem_per_threadblock) / 2u - q_smem_bytes_
          : 0u;
  constexpr uint32_t min_valid_mma_kv_ = (HEAD_DIM_VO / 16u == 4u) ? 2u : 1u;
#else
  const uint32_t kv_budget_ =
      static_cast<uint32_t>(max_smem_per_threadblock) - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ);
  constexpr uint32_t min_valid_mma_kv_ = 1u;
#endif
  const uint32_t max_num_mma_kv_smem = std::max(
      min_valid_mma_kv_, static_cast<uint32_t>(kv_budget_ / ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 *
                                                             NUM_WARPS_KV * sizeof(DTypeKV))));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : "
                 "NUM_MMA_Q="
              << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue "
                 "(https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithPagedKVCacheKernel<KTraits, Params>;
      FI_GPU_CALL(
          gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
        FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FI_GPU_CALL(VariableLengthMergeStates(tmp_v, tmp_s, params.merge_indptr, o, lse,
                                                params.max_total_num_rows, params.total_num_rows,
                                                num_qo_heads, HEAD_DIM_VO, stream));
#ifdef PLATFORM_HIP_DEVICE
          if (params.partial_o != nullptr) {
            FI_GPU_CALL(MergeStateInPlace(o, lse, params.partial_o, params.partial_lse,
                                          params.max_total_num_rows, num_qo_heads, HEAD_DIM_VO,
                                          nullptr, stream));
          }
#endif
        } else {
          FI_GPU_CALL(VariableLengthAttentionSum(tmp_v, params.merge_indptr, o,
                                                 params.max_total_num_rows, params.total_num_rows,
                                                 num_qo_heads, HEAD_DIM_VO, stream));
        }
      }
    }
  });
  return gpuSuccess;
}

}  // namespace flashinfer
