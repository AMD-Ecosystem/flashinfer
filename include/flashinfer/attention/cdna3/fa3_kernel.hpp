// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: HIP kernel wrapper and launch interface with split-KV parallelism.
//
// Templated on TileConfig (Tile128x128 or Tile64x128), IsCausal, and PairSize.
// PairSize=2 (Phase 2b-7 B.1) collapses 2 sibling Q-heads into one CTA so K/V
// LDS is loaded once per kv-tile. PairSize=1 is the legacy one-CTA-per-Q-head
// path. Host dispatches based on whether gqa_ratio is divisible by 2.
// v_mfma_f32_32x32x8f16, TransposedC for both QK and PV GEMMs.
// Double-buffered K + double-buffered K-packed V. sched_group_barrier scheduling.
// XCD-aware block reordering for GQA LLC reuse.
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
// XCD-aware block index remapping
// ---------------------------------------------------------------------------

struct XCDAwareMapping {
  int q_block;
  int head_idx;
  int head_idx_k;
};

template <int PairSize>
__device__ __forceinline__ XCDAwareMapping xcd_aware_remap(int flat_block_id, int num_q_blocks,
                                                           int nhead, int nhead_k,
                                                           int total_blocks) {
  // MI308X CPX mode: each logical device exposes a single XCD with 20 CUs,
  // so the XCD remap collapses to identity and only the GQA reorder below
  // matters (it keeps gqa_ratio/PairSize Q-head pairs consecutive in launch
  // order, which lets them reuse the K/V tile through the per-XCD L2).
  static constexpr int kNumXCDs = 1;

  int xcd_id = flat_block_id % kNumXCDs;
  int blocks_per_xcd = (total_blocks + kNumXCDs - 1) / kNumXCDs;
  int pos_in_xcd = flat_block_id / kNumXCDs;
  int logical_block = xcd_id * blocks_per_xcd + pos_in_xcd;

  if (logical_block >= total_blocks) {
    logical_block = flat_block_id % total_blocks;
  }

  int gqa_ratio = nhead / nhead_k;
  int pairs_per_kgroup = gqa_ratio / PairSize;
  int gqa_group = logical_block / pairs_per_kgroup;
  int pair_in_group = logical_block % pairs_per_kgroup;
  int q_b = gqa_group / nhead_k;
  int h_k = gqa_group % nhead_k;
  int h_q_lo = h_k * gqa_ratio + pair_in_group * PairSize;

  XCDAwareMapping m;
  m.q_block = q_b;
  m.head_idx = h_q_lo;
  m.head_idx_k = h_k;
  return m;
}

// ---------------------------------------------------------------------------
// Templated kernel: Tile selects (kBr, kNumWaves), IsCausal selects mask path,
// PairSize selects single (=1) vs pair (=2) Q-head per CTA.
// When num_kv_chunks == 1: O_out/LSE_out point to final output
// When num_kv_chunks > 1: O_out points to tmp_o, LSE_out to tmp_lse
// ---------------------------------------------------------------------------

template <class Tile, bool IsCausal, int PairSize>
__global__ __launch_bounds__(Tile::kNumThreads, 1) void fa3_cdna3_prefill_kernel_impl(
    __half* __restrict__ O_out, float* __restrict__ LSE_out, const __half* __restrict__ Q,
    const __half* __restrict__ K, const __half* __restrict__ V, int N_q, int N_kv, int nhead,
    int nhead_k, float scale_log2, int causal_offset, int num_q_blocks, int total_blocks,
    int kv_chunk_size, int num_kv_chunks, int o_row_stride, int lse_row_stride, int lse_head_stride,
    bool base2_lse) {
  int flat_id = blockIdx.x;
  int kv_chunk_idx = blockIdx.y;

  XCDAwareMapping map =
      xcd_aware_remap<PairSize>(flat_id, num_q_blocks, nhead, nhead_k, total_blocks);

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
#pragma unroll
        for (int hh = 0; hh < PairSize; ++hh) {
          fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
              O_zero, -3.402823466e+38f, 0.0f, scale_log2, o_chunk, lse_chunk, wave_q_start, nhead,
              head_idx + hh, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse,
              lane_id);
        }
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
  args.head_idx = head_idx;  // == head_idx_lo in pair mode
  args.head_idx_k = head_idx_k;
  args.kv_chunk_start = kv_chunk_start;
  args.kv_chunk_end = kv_chunk_end;

  const int wave_q_start = q_block * Tile::kBr + (threadIdx.x / kWaveSize) * Tile::kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  __half* o_chunk = O_out + kv_chunk_idx * nhead * Tile::kHeadDim;
  float* lse_chunk = LSE_out + kv_chunk_idx * nhead;

  if constexpr (PairSize == 2) {
    fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim> O_acc_lo;
    fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim> O_acc_hi;
    float row_max_lo, row_max_hi, row_sum_lo, row_sum_hi;

    run_fa3_cdna3_pipeline_pair<Tile, Tile::kHeadDim, IsCausal>(
        args, smem, O_acc_lo, O_acc_hi, row_max_lo, row_max_hi, row_sum_lo, row_sum_hi);

    if (wave_q_start < N_q) {
      fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
          O_acc_lo, row_max_lo, row_sum_lo, args.scale_log2, o_chunk, lse_chunk, wave_q_start,
          nhead, head_idx, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse, lane_id);
      fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
          O_acc_hi, row_max_hi, row_sum_hi, args.scale_log2, o_chunk, lse_chunk, wave_q_start,
          nhead, head_idx + 1, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse,
          lane_id);
    }
  } else {
    fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim> O_acc;
    float row_max, row_sum;

    run_fa3_cdna3_pipeline<Tile, Tile::kHeadDim, IsCausal>(args, smem, O_acc, row_max, row_sum);

    if (wave_q_start < N_q) {
      fa3_cdna3_epilogue<Tile::kBrLocal, Tile::kHeadDim>(
          O_acc, row_max, row_sum, args.scale_log2, o_chunk, lse_chunk, wave_q_start, nhead,
          head_idx, N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse, lane_id);
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side predicate
// ---------------------------------------------------------------------------

inline bool can_use_fa3_cdna3(int hdim, int q_len, int kv_len) {
  return (hdim == 256) && (q_len <= 8192) && (kv_len <= 8192);
}

}  // namespace cdna3
}  // namespace flashinfer
