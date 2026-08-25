// SPDX-FileCopyrightText: 2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cascade.cuh"
#include "dispatch.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"
#include "gpu_iface/math_ops.hpp"
#include "gpu_iface/platform.hpp"
#include "gpu_iface/sm_id.hpp"
#include "gpu_iface/utils.cuh"
#include "prefill.cuh"
#include "variants.cuh"

namespace flashinfer {

enum PODOperation {
  POD_PREFILL = 0,
  POD_DECODE = 1,
};

template <typename KTraits_P, typename KTraits_D, bool PartitionKV_P, typename PrefillParams,
          typename DecodeParams>
__global__ __launch_bounds__(std::max(
    KTraits_P::NUM_THREADS,
    KTraits_D::NUM_THREADS)) void PODWithKVCacheTensorKernel(const uint32_t xsize,
                                                             const __grid_constant__ PrefillParams
                                                                 prefill_params,
                                                             const __grid_constant__ DecodeParams
                                                                 decode_params,
                                                             int* tbAssign, int num_SMs) {
  extern __shared__ uint8_t smem[];
  const uint32_t num_kv_heads_p = prefill_params.num_kv_heads;
  const uint32_t num_chunks = prefill_params.partition_kv;
  const uint32_t qo_len = prefill_params.qo_len;

  const uint32_t padded_bsize = decode_params.padded_batch_size;
  const uint32_t num_kv_heads_d = decode_params.paged_kv.num_heads;

  const uint32_t prefill_blocks = num_kv_heads_p * xsize * (PartitionKV_P ? num_chunks : 1);
  const uint32_t decode_blocks = padded_bsize * num_kv_heads_d;

  int op;
  int linear_bid;
  if (threadIdx.x == 0) {
    constexpr int blk_factor_p = 1;
    constexpr int blk_factor_d = 1;

    linear_bid = static_cast<int>(gpu_iface::get_processor_id() % static_cast<uint32_t>(num_SMs));
    const int prefill_slots = (prefill_blocks + blk_factor_p - 1) / blk_factor_p;
    const int decode_slots = (decode_blocks + blk_factor_d - 1) / blk_factor_d;

    if (prefill_slots == 0) {
      op = POD_DECODE;
      linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
    } else if (decode_slots == 0) {
      op = POD_PREFILL;
      linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
    } else if (prefill_slots <= decode_slots) {
      const int total_tags = decode_slots / prefill_slots + 1;
      op = (atomicAdd(&tbAssign[linear_bid], 1) % total_tags);
      if (op > 0) {
        op = 1;
      }
      linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
      // If this CU has exhausted its quota for the chosen op, steal work from the other op.
      if (op == 0 && linear_bid >= prefill_slots) {
        linear_bid = atomicAdd(&tbAssign[num_SMs + 1], 1);
        op = !op;
      } else if (op == 1 && linear_bid >= decode_slots) {
        op = !op;
        linear_bid = atomicAdd(&tbAssign[num_SMs + 0], 1);
      }
    } else {
      const int pref_tags = prefill_slots / decode_slots;
      op = (atomicAdd(&tbAssign[linear_bid], 1) % (pref_tags + 1));
      if (op < pref_tags) {
        op = 0;
      } else {
        op = 1;
      }
      linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
      // If this CU has exhausted its quota for the chosen op, steal work from the other op.
      if (op == 0 && linear_bid >= prefill_slots) {
        linear_bid = atomicAdd(&tbAssign[num_SMs + 1], 1);
        op = !op;
      } else if (op == 1 && linear_bid >= decode_slots) {
        op = !op;
        linear_bid = atomicAdd(&tbAssign[num_SMs + 0], 1);
      }
    }
    ((int*)smem)[0] = linear_bid;
    ((int*)smem)[1] = op;
  }
  __syncthreads();
  linear_bid = ((int*)smem)[0];
  op = ((int*)smem)[1];
  __syncthreads();

  if (op == POD_PREFILL) {
    const uint32_t linear_tid = threadIdx.x;
    if (linear_tid >= WARP_SIZE * KTraits_P::NUM_WARPS_Q * KTraits_P::NUM_WARPS_KV) return;

    const dim3 tid = dim3(linear_tid % WARP_SIZE, (linear_tid / WARP_SIZE) % KTraits_P::NUM_WARPS_Q,
                          (linear_tid / WARP_SIZE) / KTraits_P::NUM_WARPS_Q);
    if (linear_bid >= prefill_blocks) return;

    const uint32_t bx = linear_bid % xsize;
    auto& smem_storage = reinterpret_cast<typename KTraits_P::SharedStorage&>(smem);
    if constexpr (!PartitionKV_P) {
      const uint32_t chunk_idx = 0;
      const uint32_t kv_head_idx = linear_bid / xsize;
      SinglePrefillWithKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, bx, chunk_idx,
                                                kv_head_idx, 1, num_kv_heads_p);
    } else {
      const uint32_t chunk_idx = (linear_bid / xsize) % num_chunks;
      const uint32_t kv_head_idx = linear_bid / (xsize * num_chunks);
      SinglePrefillWithKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, bx, chunk_idx,
                                                kv_head_idx, num_chunks, num_kv_heads_p);
    }
  } else {  // POD_DECODE
    auto& smem_storage = reinterpret_cast<typename KTraits_D::SharedStorage&>(smem);
    if (linear_bid >= decode_blocks) return;

    const uint32_t bx = linear_bid % padded_bsize;
    const uint32_t kv_head_idx = linear_bid / padded_bsize;

    const uint32_t linear_tid = threadIdx.x;
    if (linear_tid >= WARP_SIZE * KTraits_D::NUM_WARPS_Q * KTraits_D::NUM_WARPS_KV) return;

    const dim3 tid = dim3(linear_tid % WARP_SIZE, (linear_tid / WARP_SIZE) % KTraits_D::NUM_WARPS_Q,
                          (linear_tid / WARP_SIZE) / KTraits_D::NUM_WARPS_Q);

    BatchPrefillWithPagedKVCacheDevice<KTraits_D>(decode_params, smem_storage, tid, bx, kv_head_idx,
                                                  num_kv_heads_d);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q_D,
          MaskMode MASK_MODE_D, typename PrefillAttentionVariant, typename DecodeAttentionVariant,
          typename PrefillParams, typename DecodeParams>
gpuError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params,
                                          typename PrefillParams::DTypeO* tmp_p,
                                          DecodeParams decode_params,
                                          typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                          bool /*enable_pdl*/, gpuStream_t stream) {
  static_assert(std::is_same<typename PrefillParams::DTypeQ, typename DecodeParams::DTypeQ>::value);
  static_assert(
      std::is_same<typename PrefillParams::DTypeKV, typename DecodeParams::DTypeKV>::value);
  static_assert(std::is_same<typename PrefillParams::DTypeO, typename DecodeParams::DTypeO>::value);
  assert(prefill_params.num_kv_heads == decode_params.paged_kv.num_heads);
  assert(prefill_params.num_qo_heads == decode_params.num_qo_heads);

  using DTypeQ_P = typename PrefillParams::DTypeQ;
  using DTypeKV_P = typename PrefillParams::DTypeKV;
  using DTypeO_P = typename PrefillParams::DTypeO;
  const uint32_t num_qo_heads = prefill_params.num_qo_heads;
  const uint32_t num_kv_heads = prefill_params.num_kv_heads;
  const uint32_t qo_len = prefill_params.qo_len;
  const uint32_t kv_len = prefill_params.kv_len;
  if (kv_len < qo_len && MASK_MODE_P == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be >= qo_len, got kv_len="
            << kv_len << " qo_len=" << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;

  int64_t unpacked_qo_len = static_cast<int64_t>(qo_len) * group_size;
  uint32_t cta_tile_q_p = FA2DetermineCtaTileQ(unpacked_qo_len, HEAD_DIM_QK);

  using DTypeQ_D = typename DecodeParams::DTypeQ;
  using DTypeKV_D = typename DecodeParams::DTypeKV;
  using DTypeO_D = typename DecodeParams::DTypeO;
  const uint32_t padded_batch_size_d = decode_params.padded_batch_size;
  constexpr uint32_t NUM_MMA_Q_D = get_num_mma_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_Q_D = get_num_warps_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_KV_D = get_num_warps_kv(CTA_TILE_Q_D);

  if (padded_batch_size_d == 0) {
    return gpuSuccess;
  }

  using DTypeQKAccum_D =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_D, half>, half,
                                float>::type;

  int dev_id = 0;
  FI_GPU_CALL(gpuGetDevice(&dev_id));
  int max_smem_per_sm = getMaxSharedMemPerMultiprocessor(dev_id);
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_D) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  constexpr uint32_t max_num_mma_kv_reg_d =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_D == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_D);
  const uint32_t max_num_mma_kv_smem_d =
      (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ_D)) -
       NUM_MMA_Q_D * NUM_WARPS_Q_D) /
      (2 * NUM_WARPS_KV_D);

  DISPATCH_CTA_TILE_Q(cta_tile_q_p, CTA_TILE_Q_P, {
    constexpr uint32_t NUM_WARPS_Q_P = get_num_warps_q(CTA_TILE_Q_P);
    constexpr uint32_t NUM_WARPS_KV_P = get_num_warps_kv(CTA_TILE_Q_P);
    constexpr uint32_t NUM_MMA_Q_P = get_num_mma_q(CTA_TILE_Q_P);

    using DTypeQKAccum_P =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_P, half>, half,
                                  float>::type;

    const int num_ctas_per_sm_p =
        max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_P) * 16) ? 2 : 1;
    const int max_smem_per_threadblock_p = max_smem_per_sm / num_ctas_per_sm_p;

    constexpr uint32_t max_num_mma_kv_reg_p =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q_P == 2 &&
         POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q_P);
    const uint32_t max_num_mma_kv_smem_p =
        (max_smem_per_threadblock_p / (16 * HEAD_DIM_QK * sizeof(DTypeQ_P)) -
         NUM_MMA_Q_P * NUM_WARPS_Q_P) /
        (2 * NUM_WARPS_KV_P);

    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_p, max_num_mma_kv_reg_p), NUM_MMA_KV_P, {
      using KTraits_P =
          KernelTraits<MASK_MODE_P, CTA_TILE_Q_P, NUM_MMA_Q_P, NUM_MMA_KV_P, NUM_MMA_D_QK,
                       NUM_MMA_D_VO, NUM_WARPS_Q_P, NUM_WARPS_KV_P, POS_ENCODING_MODE, DTypeQ_P,
                       DTypeKV_P, DTypeO_P, DTypeQKAccum_P, typename PrefillParams::IdType,
                       PrefillAttentionVariant>;

      if constexpr (KTraits_P::IsInvalid()) {
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid prefill configuration";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_d, max_num_mma_kv_reg_d), NUM_MMA_KV_D, {
          using KTraits_D =
              KernelTraits<MASK_MODE_D, CTA_TILE_Q_D, NUM_MMA_Q_D, NUM_MMA_KV_D, NUM_MMA_D_QK,
                           NUM_MMA_D_VO, NUM_WARPS_Q_D, NUM_WARPS_KV_D, POS_ENCODING_MODE, DTypeQ_D,
                           DTypeKV_D, DTypeO_D, DTypeQKAccum_D, typename DecodeParams::IdType,
                           DecodeAttentionVariant>;
          if constexpr (KTraits_D::IsInvalid()) {
            std::ostringstream err_msg;
            err_msg << "FlashInfer Internal Error: Invalid decode configuration";
            FLASHINFER_ERROR(err_msg.str());
          } else {
            constexpr uint32_t num_threads_p = KTraits_P::NUM_THREADS;
            size_t smem_size_p = sizeof(typename KTraits_P::SharedStorage);
            size_t smem_size_d = sizeof(typename KTraits_D::SharedStorage);

            auto kernel =
                PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, true, PrefillParams, DecodeParams>;

            int num_blocks_per_sm = 0;
            int num_sm = 0;
            FI_GPU_CALL(gpuDeviceGetAttribute(&num_sm, gpuDevAttrMultiProcessorCount, dev_id));
            num_blocks_per_sm = std::max(
                1, std::min((int)(max_smem_per_sm / smem_size_p), (int)(256 / num_threads_p)));
            uint32_t max_num_kv_chunks =
                (num_blocks_per_sm * num_sm) /
                (num_kv_heads * ceil_div(qo_len * group_size, KTraits_P::CTA_TILE_Q));
            uint32_t num_chunks;
            if (max_num_kv_chunks > 0) {
              uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256u);
              num_chunks = ceil_div(kv_len, chunk_size);
            } else {
              num_chunks = 0;
            }

            auto o_p = prefill_params.o;
            auto lse_p = prefill_params.lse;
            float* tmp_lse = (float*)(tmp_p + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
            if (num_chunks <= 1 || tmp_p == nullptr) {
              prefill_params.partition_kv = 0;
              kernel = PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, false, PrefillParams,
                                                  DecodeParams>;
            } else {
              prefill_params.partition_kv = num_chunks;
              prefill_params.o = tmp_p;
              prefill_params.lse = tmp_lse;
              kernel = PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, true, PrefillParams,
                                                  DecodeParams>;
            }

            auto o_d = decode_params.o;
            auto lse_d = decode_params.lse;
            if (tmp_v == nullptr) {
              decode_params.partition_kv = false;
            } else {
              decode_params.partition_kv = true;
              decode_params.o = tmp_v;
              decode_params.lse = tmp_s;
            }
            uint32_t xsize = ceil_div(qo_len * group_size, KTraits_P::CTA_TILE_Q);
            int nblks_p(xsize * (prefill_params.partition_kv ? prefill_params.partition_kv : 1) *
                        num_kv_heads);
            int nthrs_p(KTraits_P::NUM_THREADS);

            int nblks_d = padded_batch_size_d * num_kv_heads;
            int nthrs_d(KTraits_D::NUM_THREADS);

            size_t smem_size = max(smem_size_p, smem_size_d);
            int nblks = nblks_p + nblks_d;
            int nthrs = max(nthrs_p, nthrs_d);

            static int* tbAssign = nullptr;
            if (tbAssign == nullptr) FI_GPU_CALL(gpuMalloc(&tbAssign, sizeof(int) * (num_sm + 2)));
            FI_GPU_CALL(gpuMemset(tbAssign, 0, sizeof(int) * (num_sm + 2)));

            FI_GPU_CALL(
                gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));

            void* args[] = {(void*)&xsize, (void*)&prefill_params, (void*)&decode_params,
                            (void*)&tbAssign, (void*)&num_sm};
            FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

            if (!(num_chunks <= 1 || tmp_p == nullptr)) {
              if constexpr (PrefillAttentionVariant::use_softmax) {
                FI_GPU_CALL(MergeStates(tmp_p, tmp_lse, o_p, lse_p, num_chunks, qo_len,
                                        num_qo_heads, HEAD_DIM_VO, stream));
              } else {
                FI_GPU_CALL(AttentionSum(tmp_p, o_p, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO,
                                         stream));
              }
            }
            if (tmp_v != nullptr) {
              if constexpr (DecodeAttentionVariant::use_softmax) {
                FI_GPU_CALL(VariableLengthMergeStates(tmp_v, tmp_s, decode_params.merge_indptr, o_d,
                                                      lse_d, decode_params.max_total_num_rows,
                                                      decode_params.total_num_rows, num_qo_heads,
                                                      HEAD_DIM_VO, stream));
              } else {
                FI_GPU_CALL(VariableLengthAttentionSum(
                    tmp_v, decode_params.merge_indptr, o_d, decode_params.max_total_num_rows,
                    decode_params.total_num_rows, num_qo_heads, HEAD_DIM_VO, stream));
              }
            }
          }
        });
      }
    });
  });
  return gpuSuccess;
}

}  // namespace flashinfer
