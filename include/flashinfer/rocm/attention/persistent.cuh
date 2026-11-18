// SPDX-FileCopyrightText: 2025 FlashInfer team.
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// Persistent ("holistic") batch attention for CDNA3 / CDNA4 (gfx942, gfx950).
//
// ROCm port of upstream's include/flashinfer/attention/persistent.cuh
// (BlockBatchPagedAttentionPersistent, BlockBatchReductionPersistent,
// BatchPagedAttentionPersistent). The planner is TwoStageHolisticPlan in
// scheduler.cuh.
//
// Control flow follows upstream: every CTA walks its slice of the work list for
// the large-tile (prefill-shaped) task, then for the 16-row (decode-shaped)
// task, writing either the final output or, for split-KV work, a partial
// (o, lse). The merge of the partials follows. The grid is fixed per (device,
// head_dim), so one captured graph serves every batch shape.
//
// Upstream separates attention and merge with grid.sync() inside one
// cooperative launch. Here they are two ordinary launches on the same stream,
// which orders them just the same and is captured into the same graph. A
// cooperative launch from a torch extension crashes the process at exit on
// ROCm 7.2 (any grid-sync kernel, once torch has used the device), and it also
// needs every CTA co-resident, which a GPU shared with another process may
// refuse. The cost is one extra launch.
//
// The tile math differs from upstream (mma.sync m16n8k16 + ldmatrix):
//   * It is MFMA 16x16x16 (f16 / bf16 in, f32 accumulate) on wave64. Lane l holds
//     column l%16 of the B and C fragments and k / row indices 4*(l/16) .. +3.
//   * It computes S^T = K Q^T and O^T = V^T P^T. The C fragment of S^T gives each
//     lane one query and 4 keys, which is exactly the B-operand layout P^T needs
//     in O^T = V^T P^T, so P never goes through LDS, and a softmax row reduction
//     is two cross-lane exchanges (xor 16, xor 32).
//   * KV tiles are loaded with plain 16-byte loads into row-padded LDS, without
//     an async-copy pipeline. Correctness first; tuning is a follow-up.
//   * The partials are fp32 (upstream: DTypeO).
// Not ported: NVFP4 / fp8 KV, the profiler, custom attention variants. The JIT
// generator refuses those before compiling.

#ifndef FLASHINFER_ROCM_ATTENTION_PERSISTENT_CUH_
#define FLASHINFER_ROCM_ATTENTION_PERSISTENT_CUH_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace flashinfer {

/*!
 * \brief Arguments of one task (0: large query tile, 1: 16-row query tile) of the
 *   persistent kernel. Pointers into the workspace come from HolisticPlanInfo.
 */
template <typename DType_, typename IdType_>
struct BatchAttentionPersistentParams {
  using DType = DType_;
  using IdType = IdType_;
  const DType* q;
  const DType* k;
  const DType* v;
  const IdType* kv_indices;
  DType* final_o;    // [num_tokens, num_qo_heads, head_dim], row stride o_stride_n
  float* final_lse;  // [num_tokens, num_qo_heads] contiguous (base-2 LSE), or nullptr
  float* partial_o;
  float* partial_lse;
  const IdType* merge_indptr;
  const IdType* merge_o_indices;
  const IdType* num_packed_qo_len;
  const IdType* len_kv_chunk;  // [2], indexed by task
  // per-task work list
  const IdType* q_indptr;
  const IdType* kv_indptr;
  const IdType* partial_indptr;
  const IdType* q_len;
  const IdType* kv_len;
  const IdType* q_start;
  const IdType* kv_start;
  const IdType* kv_end;
  const IdType* kv_head_idx;
  const IdType* work_indptr;
  uint32_t task;
  uint32_t num_qo_heads, num_kv_heads, gqa_group_size, page_size;
  uint32_t q_stride_n, q_stride_h, o_stride_n;
  uint32_t k_stride_page, k_stride_h, k_stride_n;
  uint32_t v_stride_page, v_stride_h, v_stride_n;
  float sm_scale_log2;            // sm_scale * log2(e); log2(e) * soft_cap with soft cap
  float soft_cap_pre_tanh_scale;  // sm_scale / soft_cap; unused without soft cap
  float v_scale;
};

namespace persistent_rocm {

constexpr uint32_t kWave = 64;
constexpr uint32_t kFrag = 4;                  // elements per lane in a 16x16 fragment
constexpr uint32_t kBlockThreads = 4 * kWave;  // 4 warps, as upstream
constexpr size_t kMaxLdsBytes = 65536;         // gfx942 limit; gfx950 has 160 KB

typedef _Float16 h4_t __attribute__((ext_vector_type(4)));
typedef short s4_t __attribute__((ext_vector_type(4)));
typedef float f4_t __attribute__((ext_vector_type(4)));

// 16-bit element access, independent of the HIP half / bfloat16 class types.
template <typename DType>
struct Elem;

template <>
struct Elem<__half> {
  using vec = h4_t;
  __device__ static inline __half from_float(float f) {
    return __builtin_bit_cast(__half, (_Float16)f);
  }
  __device__ static inline void set(vec& v, int i, const __half& x) {
    v[i] = __builtin_bit_cast(_Float16, x);
  }
  __device__ static inline void set_f(vec& v, int i, float x) { v[i] = (_Float16)x; }
  __device__ static inline f4_t mma(vec a, vec b, f4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  }
};

template <>
struct Elem<__hip_bfloat16> {
  using vec = s4_t;
  __device__ static inline uint16_t f2bf(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);  // keep NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                          // nearest even
    return uint16_t(u >> 16);
  }
  __device__ static inline __hip_bfloat16 from_float(float f) {
    return __builtin_bit_cast(__hip_bfloat16, f2bf(f));
  }
  __device__ static inline void set(vec& v, int i, const __hip_bfloat16& x) {
    v[i] = (short)__builtin_bit_cast(uint16_t, x);
  }
  __device__ static inline void set_f(vec& v, int i, float x) { v[i] = (short)f2bf(x); }
  __device__ static inline f4_t mma(vec a, vec b, f4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  }
};

// Reductions over the four lanes that hold one column (l, l^16, l^32, l^48).
__device__ inline float group_max(float x) {
  x = fmaxf(x, __shfl_xor(x, 16));
  return fmaxf(x, __shfl_xor(x, 32));
}
__device__ inline float group_sum(float x) {
  x += __shfl_xor(x, 16);
  return x + __shfl_xor(x, 32);
}

__device__ inline int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

// NWQ warps split the query tile, NWKV warps split each KV tile.
template <typename DType_, uint32_t D_, uint32_t CTA_TILE_Q_, uint32_t NWQ_, uint32_t NWKV_,
          uint32_t NMKV_, bool CAUSAL_, bool SOFTCAP_>
struct KTraits {
  using DType = DType_;
  static constexpr uint32_t D = D_;
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t NWQ = NWQ_, NWKV = NWKV_, NMKV = NMKV_;
  static constexpr uint32_t NMQ = CTA_TILE_Q / (NWQ * 16);
  static constexpr uint32_t NMD = D / 16;
  static constexpr uint32_t CTA_TILE_KV = NWKV * NMKV * 16;
  static constexpr uint32_t NUM_ACTIVE_WARPS = NWQ * NWKV;
  static constexpr bool CAUSAL = CAUSAL_, SOFTCAP = SOFTCAP_;
  static constexpr uint32_t STRIDE = D + 8;  // LDS row stride in elements (16 B aligned)
  static constexpr size_t Q_BYTES = size_t(CTA_TILE_Q) * STRIDE * sizeof(DType);
  static constexpr size_t KV_BYTES = size_t(CTA_TILE_KV) * STRIDE * sizeof(DType);
  static constexpr size_t MERGE_BYTES = NWKV > 1 ? size_t(NWKV) * CTA_TILE_Q*(D + 2) * 4 : 0;
  static constexpr size_t SMEM_BYTES =
      Q_BYTES + (2 * KV_BYTES > MERGE_BYTES ? 2 * KV_BYTES : MERGE_BYTES);
  static_assert(sizeof(DType) == 2, "16-bit q / kv / o only");
  static_assert(NWQ * NMQ * 16 == CTA_TILE_Q, "query tile must split evenly over warps");
  static_assert(NUM_ACTIVE_WARPS * kWave <= kBlockThreads, "too many warps");
  static_assert(D % kWave == 0, "head_dim must be a multiple of 64");
};

// One task's slice of the work list (upstream: BlockBatchPagedAttentionPersistent).
template <typename KT>
struct AttentionRunner {
  using DType = typename KT::DType;
  using E = Elem<DType>;
  using vec = typename E::vec;

  template <typename Params>
  __device__ static void Run(const Params& p, uint8_t* smem) {
    constexpr uint32_t D = KT::D, STRIDE = KT::STRIDE, NMQ = KT::NMQ, NMKV = KT::NMKV,
                       NMD = KT::NMD, CTA_TILE_Q = KT::CTA_TILE_Q, CTA_TILE_KV = KT::CTA_TILE_KV,
                       NWQ = KT::NWQ, NWKV = KT::NWKV;
    constexpr uint32_t VEC = 8, CHUNKS = D / VEC;  // 16-byte chunks per row

    const uint32_t tid = threadIdx.x, lane = tid % kWave, warp = tid / kWave;
    const bool active = warp < KT::NUM_ACTIVE_WARPS;
    const uint32_t wq = warp % NWQ, wkv = warp / NWQ;
    // col: the query (column of S^T / O^T) this lane holds; fo: its first key / dim row.
    const uint32_t col = lane % 16, grp = lane / 16, fo = kFrag * grp;

    DType* q_s = reinterpret_cast<DType*>(smem);
    DType* k_s = reinterpret_cast<DType*>(smem + KT::Q_BYTES);
    DType* v_s = reinterpret_cast<DType*>(smem + KT::Q_BYTES + KT::KV_BYTES);
    float* mrg = reinterpret_cast<float*>(smem + KT::Q_BYTES);

    const int G = (int)p.gqa_group_size, H = (int)p.num_kv_heads, page_size = (int)p.page_size;
    const int len_kv_chunk = p.len_kv_chunk[p.task];

    for (int w = p.work_indptr[blockIdx.y]; w < p.work_indptr[blockIdx.y + 1]; ++w) {
      const int q_indptr = p.q_indptr[w], kv_indptr = p.kv_indptr[w],
                o_indptr = p.partial_indptr[w], q_len = p.q_len[w], kv_len = p.kv_len[w],
                packed_qo_start = p.q_start[w], kv_start = p.kv_start[w], kv_end = p.kv_end[w],
                kv_head = p.kv_head_idx[w];

      // Same chunk bookkeeping as upstream.
      const int kv_chunk_idx = kv_start / len_kv_chunk;
      const int causal_tile_end = kv_len - q_len + ceil_div_i(packed_qo_start + (int)CTA_TILE_Q, G);
      const int num_kv_chunks =
          ceil_div_i(KT::CAUSAL ? min(causal_tile_end, kv_len) : kv_len, len_kv_chunk);
      const int kv_end_eff = KT::CAUSAL ? min(kv_end, causal_tile_end) : kv_end;

      __syncthreads();  // the previous work item is done with LDS
      for (uint32_t i = tid; i < CTA_TILE_Q * CHUNKS; i += kBlockThreads) {
        const int row = i / CHUNKS, c = i % CHUNKS;
        const int pk = packed_qo_start + row, qi = pk / G, r = pk % G;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (qi < q_len)
          val = *reinterpret_cast<const uint4*>(p.q + size_t(q_indptr + qi) * p.q_stride_n +
                                                size_t(kv_head * G + r) * p.q_stride_h + c * VEC);
        *reinterpret_cast<uint4*>(q_s + row * STRIDE + c * VEC) = val;
      }

      float o[NMQ][NMD][kFrag];
      float m[NMQ], dsum[NMQ];
#pragma unroll
      for (uint32_t mq = 0; mq < NMQ; ++mq) {
        m[mq] = -INFINITY;
        dsum[mq] = 0.f;
#pragma unroll
        for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
          for (uint32_t v = 0; v < kFrag; ++v) o[mq][md][v] = 0.f;
      }

      for (int kv_base = kv_start; kv_base < kv_end_eff; kv_base += CTA_TILE_KV) {
        __syncthreads();
        for (uint32_t i = tid; i < CTA_TILE_KV * CHUNKS; i += kBlockThreads) {
          const int row = i / CHUNKS, c = i % CHUNKS;
          const int pos = kv_base + row;
          uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
          if (pos < kv_end) {
            const int it = kv_indptr * page_size + pos;
            const size_t page = (size_t)p.kv_indices[it / page_size];
            const int e = it % page_size;
            kk = *reinterpret_cast<const uint4*>(p.k + page * p.k_stride_page +
                                                 size_t(kv_head) * p.k_stride_h +
                                                 size_t(e) * p.k_stride_n + c * VEC);
            vv = *reinterpret_cast<const uint4*>(p.v + page * p.v_stride_page +
                                                 size_t(kv_head) * p.v_stride_h +
                                                 size_t(e) * p.v_stride_n + c * VEC);
          }
          *reinterpret_cast<uint4*>(k_s + row * STRIDE + c * VEC) = kk;
          *reinterpret_cast<uint4*>(v_s + row * STRIDE + c * VEC) = vv;
        }
        __syncthreads();
        if (!active) continue;

        // S^T = K Q^T: lane holds query `col`, keys fo .. fo + 3 of each 16-key block.
        float s[NMQ][NMKV][kFrag];
#pragma unroll
        for (uint32_t mq = 0; mq < NMQ; ++mq) {
#pragma unroll
          for (uint32_t mkv = 0; mkv < NMKV; ++mkv) {
            f4_t acc = {0.f, 0.f, 0.f, 0.f};
            const DType* krow = k_s + (wkv * NMKV * 16 + mkv * 16 + col) * STRIDE + fo;
            const DType* qrow = q_s + (wq * NMQ * 16 + mq * 16 + col) * STRIDE + fo;
#pragma unroll
            for (uint32_t kd = 0; kd < NMD; ++kd) {
              vec a = *reinterpret_cast<const vec*>(krow + kd * 16);
              vec b = *reinterpret_cast<const vec*>(qrow + kd * 16);
              acc = E::mma(a, b, acc);
            }
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v) s[mq][mkv][v] = acc[v];
          }
        }

        // Logits transform and mask, then the online softmax (base 2, as upstream).
#pragma unroll
        for (uint32_t mq = 0; mq < NMQ; ++mq) {
          const int pk = packed_qo_start + (int)(wq * NMQ * 16 + mq * 16 + col);
          const int qi = pk / G;
          float mx = -INFINITY;
#pragma unroll
          for (uint32_t mkv = 0; mkv < NMKV; ++mkv) {
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v) {
              const int kv = kv_base + (int)(wkv * NMKV * 16 + mkv * 16 + fo + v);
              float x = s[mq][mkv][v];
              if constexpr (KT::SOFTCAP) x = tanhf(x * p.soft_cap_pre_tanh_scale);
              x *= p.sm_scale_log2;
              bool valid = kv < kv_end;
              if constexpr (KT::CAUSAL) valid = valid && (kv <= kv_len - q_len + qi);
              x = valid ? x : -INFINITY;
              s[mq][mkv][v] = x;
              mx = fmaxf(mx, x);
            }
          }
          mx = group_max(mx);
          const float m_new = fmaxf(m[mq], mx);
          const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
          const float scale = exp2f(m[mq] - m_use);
          m[mq] = m_new;
          dsum[mq] *= scale;
#pragma unroll
          for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v) o[mq][md][v] *= scale;
#pragma unroll
          for (uint32_t mkv = 0; mkv < NMKV; ++mkv)
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v) {
              const float e = exp2f(s[mq][mkv][v] - m_use);
              s[mq][mkv][v] = e;
              dsum[mq] += e;
            }
        }

        // O^T += V^T P^T. P^T is already in the B-operand layout.
#pragma unroll
        for (uint32_t mq = 0; mq < NMQ; ++mq) {
#pragma unroll
          for (uint32_t mkv = 0; mkv < NMKV; ++mkv) {
            vec b;
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v) E::set_f(b, v, s[mq][mkv][v]);
            const DType* vbase = v_s + (wkv * NMKV * 16 + mkv * 16 + fo) * STRIDE + col;
#pragma unroll
            for (uint32_t md = 0; md < NMD; ++md) {
              vec a;
#pragma unroll
              for (uint32_t i = 0; i < kFrag; ++i) E::set(a, i, vbase[i * STRIDE + md * 16]);
              f4_t acc;
#pragma unroll
              for (uint32_t v = 0; v < kFrag; ++v) acc[v] = o[mq][md][v];
              acc = E::mma(a, b, acc);
#pragma unroll
              for (uint32_t v = 0; v < kFrag; ++v) o[mq][md][v] = acc[v];
            }
          }
        }
      }

#pragma unroll
      for (uint32_t mq = 0; mq < NMQ; ++mq) dsum[mq] = group_sum(dsum[mq]);

      // Merge the NWKV warps that split each KV tile (upstream:
      // threadblock_sync_mdo_states).
      if constexpr (NWKV > 1) {
        __syncthreads();  // K / V tiles are dead; the merge scratch overlays them
        if (active) {
#pragma unroll
          for (uint32_t mq = 0; mq < NMQ; ++mq) {
            const uint32_t row = wq * NMQ * 16 + mq * 16 + col;
            float* base = mrg + (size_t(wkv) * CTA_TILE_Q + row) * (D + 2);
#pragma unroll
            for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
              for (uint32_t v = 0; v < kFrag; ++v) base[md * 16 + fo + v] = o[mq][md][v];
            if (grp == 0) {
              base[D] = m[mq];
              base[D + 1] = dsum[mq];
            }
          }
        }
        __syncthreads();
        if (active && wkv == 0) {
#pragma unroll
          for (uint32_t mq = 0; mq < NMQ; ++mq) {
            const uint32_t row = wq * NMQ * 16 + mq * 16 + col;
            float m_tot = -INFINITY;
            for (uint32_t w2 = 0; w2 < NWKV; ++w2)
              m_tot = fmaxf(m_tot, mrg[(size_t(w2) * CTA_TILE_Q + row) * (D + 2) + D]);
            const float m_use = (m_tot == -INFINITY) ? 0.f : m_tot;
            float d_tot = 0.f;
#pragma unroll
            for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
              for (uint32_t v = 0; v < kFrag; ++v) o[mq][md][v] = 0.f;
            for (uint32_t w2 = 0; w2 < NWKV; ++w2) {
              const float* base = mrg + (size_t(w2) * CTA_TILE_Q + row) * (D + 2);
              const float sc = exp2f(base[D] - m_use);
              d_tot += base[D + 1] * sc;
#pragma unroll
              for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
                for (uint32_t v = 0; v < kFrag; ++v) o[mq][md][v] += base[md * 16 + fo + v] * sc;
            }
            m[mq] = m_tot;
            dsum[mq] = d_tot;
          }
        }
      }

      if (!active || wkv != 0) continue;

      // Normalize and write: a partial (split KV) or straight to the final output.
#pragma unroll
      for (uint32_t mq = 0; mq < NMQ; ++mq) {
        const int pk = packed_qo_start + (int)(wq * NMQ * 16 + mq * 16 + col);
        const int qi = pk / G, r = pk % G;
        if (qi >= q_len) continue;
        const float inv = dsum[mq] > 0.f ? p.v_scale / dsum[mq] : 0.f;
        const float lse = dsum[mq] > 0.f ? log2f(dsum[mq]) + m[mq] : -INFINITY;
        if (num_kv_chunks > 1) {
          // partial_o: [packed rows * num_kv_chunks, num_kv_heads, head_dim]
          const size_t idx =
              size_t(o_indptr + (pk - packed_qo_start) * num_kv_chunks + kv_chunk_idx) * H +
              kv_head;
#pragma unroll
          for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v)
              p.partial_o[idx * D + md * 16 + fo + v] = o[mq][md][v] * inv;
          if (grp == 0) p.partial_lse[idx] = lse;
        } else {
          const size_t token = size_t(q_indptr + qi), head = size_t(kv_head) * G + r;
          DType* out = p.final_o + token * p.o_stride_n + head * D;
#pragma unroll
          for (uint32_t md = 0; md < NMD; ++md)
#pragma unroll
            for (uint32_t v = 0; v < kFrag; ++v)
              out[md * 16 + fo + v] = E::from_float(o[mq][md][v] * inv);
          if (grp == 0 && p.final_lse != nullptr) p.final_lse[token * p.num_qo_heads + head] = lse;
        }
      }
    }
  }
};

// Merge of the split-KV partials (upstream: BlockBatchReductionPersistent). One
// warp per (merged row, kv head); each lane owns head_dim / 64 output elements.
template <uint32_t D, typename Params>
__device__ void ReductionRun(const Params& p) {
  using E = Elem<typename Params::DType>;
  constexpr uint32_t NW = kBlockThreads / kWave, PER = D / kWave;
  const uint32_t lane = threadIdx.x % kWave, warp = threadIdx.x / kWave;
  const int n_rows = *p.num_packed_qo_len;
  const int H = (int)p.num_kv_heads, G = (int)p.gqa_group_size, HQ = (int)p.num_qo_heads;
  const int worker = blockIdx.y * NW + warp, num_workers = gridDim.y * NW;
  for (int i = worker; i < n_rows * H; i += num_workers) {
    const int row = i / H, h = i % H;
    const int st = p.merge_indptr[row], n = p.merge_indptr[row + 1] - st;
    if (n <= 1) continue;  // written directly by the attention runner
    float m = -INFINITY, dl = 0.f, acc[PER];
#pragma unroll
    for (uint32_t t = 0; t < PER; ++t) acc[t] = 0.f;
    for (int k = 0; k < n; ++k) {
      const size_t idx = size_t(st + k) * H + h;
      const float s = p.partial_lse[idx];
      const float m_new = fmaxf(m, s);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      const float sc_old = exp2f(m - m_use), sc = exp2f(s - m_use);
      dl = dl * sc_old + sc;
#pragma unroll
      for (uint32_t t = 0; t < PER; ++t)
        acc[t] = acc[t] * sc_old + sc * p.partial_o[idx * D + lane + kWave * t];
      m = m_new;
    }
    // merge_o_indices holds token * num_qo_heads + (kv head 0's) qo head.
    const size_t packed = size_t(p.merge_o_indices[row]) + size_t(h) * G;
    const size_t token = packed / HQ, head = packed % HQ;
    const float inv = dl > 0.f ? 1.f / dl : 0.f;
    typename Params::DType* out = p.final_o + token * p.o_stride_n + head * D;
#pragma unroll
    for (uint32_t t = 0; t < PER; ++t) out[lane + kWave * t] = E::from_float(acc[t] * inv);
    if (lane == 0 && p.final_lse != nullptr)
      p.final_lse[packed] = dl > 0.f ? log2f(dl) + m : -INFINITY;
  }
}

template <typename R0, typename R1, typename Params>
__global__ __launch_bounds__(kBlockThreads) void BatchAttentionPersistentKernel(const Params p0,
                                                                                const Params p1) {
  extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
  R0::Run(p0, smem);
  R1::Run(p1, smem);
}

// Launched after BatchAttentionPersistentKernel on the same stream; the stream
// order is the barrier upstream gets from grid.sync().
template <uint32_t D, typename Params>
__global__ __launch_bounds__(kBlockThreads) void BatchAttentionReductionKernel(const Params p0) {
  ReductionRun<D>(p0);
}

}  // namespace persistent_rocm

/*!
 * \brief Tile configuration and launcher of the persistent kernel.
 *
 * Task 0 (requests whose packed query length exceeds 16) uses a 128-row query
 * tile, task 1 a 16-row tile, as upstream. 64 KB of LDS per workgroup sets the
 * rest: at head_dim 256 a 128 x 256 query tile alone would fill LDS, so task 0
 * uses 64 rows there, and the KV tiles shrink as head_dim grows. The planner
 * must be instantiated with the same CTA_TILE_Q_0 / CTA_TILE_Q_1.
 */
template <typename DType, uint32_t HEAD_DIM, bool CAUSAL, bool SOFTCAP>
struct BatchAttentionPersistent {
  static constexpr uint32_t CTA_TILE_Q_0 = HEAD_DIM >= 256 ? 64 : 128;
  static constexpr uint32_t CTA_TILE_Q_1 = 16;
  using KT0 =
      persistent_rocm::KTraits<DType, HEAD_DIM, CTA_TILE_Q_0, 4, 1,
                               (HEAD_DIM >= 256 ? 1 : (HEAD_DIM >= 128 ? 2 : 4)), CAUSAL, SOFTCAP>;
  using KT1 = persistent_rocm::KTraits<DType, HEAD_DIM, CTA_TILE_Q_1, 1, (HEAD_DIM >= 256 ? 2 : 4),
                                       1, CAUSAL, SOFTCAP>;
  using Params = BatchAttentionPersistentParams<DType, int32_t>;
  static constexpr size_t SMEM_BYTES =
      KT0::SMEM_BYTES > KT1::SMEM_BYTES ? KT0::SMEM_BYTES : KT1::SMEM_BYTES;
  static_assert(SMEM_BYTES <= persistent_rocm::kMaxLdsBytes, "exceeds LDS per workgroup");
  static constexpr uint32_t NUM_THREADS = persistent_rocm::kBlockThreads;

  static constexpr auto kernel = persistent_rocm::BatchAttentionPersistentKernel<
      persistent_rocm::AttentionRunner<KT0>, persistent_rocm::AttentionRunner<KT1>, Params>;
  static constexpr auto reduction_kernel =
      persistent_rocm::BatchAttentionReductionKernel<HEAD_DIM, Params>;

  /*!
   * \brief Persistent CTAs on the current device: upstream's 2 per SM (1 at
   *   head_dim 256), capped by the measured occupancy so every CTA is resident
   *   and the planner's balanced work lists all run at once.
   */
  static hipError_t NumCtas(int* num_ctas) {
    int dev = 0, n_cu = 0, per_cu = 0;
    hipError_t e;
    if ((e = hipGetDevice(&dev)) != hipSuccess) return e;
    if ((e = hipDeviceGetAttribute(&n_cu, hipDeviceAttributeMultiprocessorCount, dev)) !=
        hipSuccess)
      return e;
    if ((e = hipFuncSetAttribute((const void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)SMEM_BYTES)) != hipSuccess)
      return e;
    if ((e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, NUM_THREADS,
                                                          SMEM_BYTES)) != hipSuccess)
      return e;
    if (per_cu < 1) return hipErrorInvalidConfiguration;
    *num_ctas = n_cu * std::min(per_cu, HEAD_DIM >= 256 ? 1 : 2);
    return hipSuccess;
  }

  static hipError_t Launch(const Params& p0, const Params& p1, uint32_t num_blks_y,
                           hipStream_t stream) {
    const dim3 grid(1, num_blks_y), block(NUM_THREADS);
    void* args[] = {(void*)&p0, (void*)&p1};
    hipError_t e = hipLaunchKernel((const void*)kernel, grid, block, args, SMEM_BYTES, stream);
    if (e != hipSuccess) return e;
    void* reduction_args[] = {(void*)&p0};
    return hipLaunchKernel((const void*)reduction_kernel, grid, block, reduction_args, 0, stream);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ROCM_ATTENTION_PERSISTENT_CUH_
