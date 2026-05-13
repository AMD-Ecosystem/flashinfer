// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: HIP kernel wrapper and launch interface with split-KV parallelism.
//
// Templated on TileConfig and IsCausal. One CTA per (Q-block, Q-head, KV-chunk).
// v_mfma_f32_32x32x8f16, TransposedC for both QK and PV GEMMs.
// Double-buffered K + double-buffered K-packed V. sched_group_barrier scheduling.
// Split-KV: blockIdx.y indexes KV chunks; partial O/LSE written to tmp buffer.

#pragma once

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_runtime.h>
#endif

#include "asm_primitives.hpp"
#include "fa3_epilogue.hpp"
#include "fa3_pipeline.hpp"
#include "fa3_tiles.hpp"

namespace flashinfer {
namespace cdna3 {

// ---------------------------------------------------------------------------
// Block index remapping: keep gqa_ratio Q-heads sharing one K-head consecutive
// in launch order so they reuse the K/V tile through L2.
// ---------------------------------------------------------------------------

struct XCDAwareMapping {
  int q_block;
  int head_idx;
  int head_idx_k;
};

__device__ __forceinline__ XCDAwareMapping xcd_aware_remap(int flat_block_id, int num_q_blocks,
                                                           int nhead, int nhead_k,
                                                           int total_blocks) {
  // MI308X CPX mode exposes a single XCD per logical device, so the XCD
  // remap collapses to identity; only the GQA reorder below matters.
  const int logical_block = flat_block_id;

  int gqa_ratio = nhead / nhead_k;
  int gqa_group = logical_block / gqa_ratio;
  int head_in_group = logical_block % gqa_ratio;
  int q_b = gqa_group / nhead_k;
  int h_k = gqa_group % nhead_k;
  int h_q = h_k * gqa_ratio + head_in_group;

  XCDAwareMapping m;
  m.q_block = q_b;
  m.head_idx = h_q;
  m.head_idx_k = h_k;
  return m;
}

// ---------------------------------------------------------------------------
// Templated kernel: Tile selects (kBr, kNumWaves), IsCausal selects mask path.
// When num_kv_chunks == 1: O_out/LSE_out point to final output
// When num_kv_chunks > 1: O_out points to tmp_o, LSE_out to tmp_lse
// ---------------------------------------------------------------------------

// HD=64 fits 3 CTAs/CU under the gfx942 register and LDS budget; HD=128/256
// already saturate one CTA/CU and would spill catastrophically at higher
// occupancy.
template <class Tile, bool IsCausal>
__global__ __launch_bounds__(Tile::kNumThreads, (Tile::kHeadDim == 64 ? 3 : 1))
void fa3_cdna3_prefill_kernel_impl(
    __half* __restrict__ O_out, float* __restrict__ LSE_out, const __half* __restrict__ Q,
    const __half* __restrict__ K, const __half* __restrict__ V, int N_q, int N_kv, int nhead,
    int nhead_k, float scale_log2, int causal_offset, int num_q_blocks, int total_blocks,
    int kv_chunk_size, int num_kv_chunks, int o_row_stride, int lse_row_stride, int lse_head_stride,
    bool base2_lse) {
  int flat_id = blockIdx.x;
  int kv_chunk_idx = blockIdx.y;

  XCDAwareMapping map = xcd_aware_remap(flat_id, num_q_blocks, nhead, nhead_k, total_blocks);

  const int q_block = map.q_block;
  const int head_idx = map.head_idx;
  const int head_idx_k = map.head_idx_k;

  int kv_chunk_start = kv_chunk_idx * kv_chunk_size;
  int kv_chunk_end = kv_chunk_start + kv_chunk_size;
  if (kv_chunk_end > N_kv) kv_chunk_end = N_kv;

  if constexpr (IsCausal) {
    int causal_last_kv = q_block * Tile::kBr + Tile::kBr + causal_offset;
    if (kv_chunk_start >= causal_last_kv) {
      const int wave_q_start = q_block * Tile::kBr + (threadIdx.x / kWaveSize) * Tile::kBrLocal;
      const int lane_id = threadIdx.x % kWaveSize;
      __half* o_chunk = O_out + kv_chunk_idx * nhead * Tile::kHeadDim;
      float* lse_chunk = LSE_out + kv_chunk_idx * nhead;
      fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim> O_zero;
      O_zero.zero();
      if (wave_q_start < N_q) {
        fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
            O_zero, -3.402823466e+38f, 0.0f, scale_log2, o_chunk, lse_chunk, wave_q_start, nhead,
            head_idx, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse, lane_id);
      }
      return;
    }
  }

  extern __shared__ char smem[];

  FA3CDNA3PipelineArgs args;
  args.Q = Q;
  args.K = K;
  args.V = V;
  args.N_q = N_q;
  args.N_kv = N_kv;
  args.nhead = nhead;
  args.nhead_k = nhead_k;
  args.scale_log2 = scale_log2;
  args.causal_offset = causal_offset;
  args.q_block = q_block;
  args.head_idx = head_idx;
  args.head_idx_k = head_idx_k;
  args.kv_chunk_start = kv_chunk_start;
  args.kv_chunk_end = kv_chunk_end;

  const int wave_q_start = q_block * Tile::kBr + (threadIdx.x / kWaveSize) * Tile::kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  __half* o_chunk = O_out + kv_chunk_idx * nhead * Tile::kHeadDim;
  float* lse_chunk = LSE_out + kv_chunk_idx * nhead;

  fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim> O_acc;
  float row_max, row_sum;

  run_fa3_cdna3_pipeline<Tile, Tile::kHeadDim, IsCausal>(args, smem, O_acc, row_max, row_sum);

  if (wave_q_start < N_q) {
    fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
        O_acc, row_max, row_sum, args.scale_log2, o_chunk, lse_chunk, wave_q_start, nhead,
        head_idx, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse, lane_id);
  }
}

// ---------------------------------------------------------------------------
// Host-side predicate
// ---------------------------------------------------------------------------

inline bool can_use_fa3_cdna3(int hdim, int q_len, int kv_len) {
  return (hdim == 64 || hdim == 128 || hdim == 256) && (q_len <= 8192) && (kv_len <= 8192);
}

}  // namespace cdna3
}  // namespace flashinfer
