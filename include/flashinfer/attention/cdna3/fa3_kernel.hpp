// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 V11: HIP kernel wrapper and launch interface with split-KV parallelism.
//
// 4 waves x 64 threads = 256 threads, kBr=128, kBc=128, d=256.
// v_mfma_f32_32x32x8f16, TransposedC for both QK and PV GEMMs.
// Double-buffered K + double-buffered K-packed V. sched_group_barrier scheduling.
// XCD-aware block reordering for GQA LLC reuse.
// Split-KV: blockIdx.y indexes KV chunks; partial O/LSE written to tmp buffer.

#pragma once

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_runtime.h>
#endif
#include <cmath>
#include <cstdint>

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

__device__ __forceinline__ XCDAwareMapping xcd_aware_remap(int flat_block_id, int num_q_blocks,
                                                           int nhead, int nhead_k,
                                                           int total_blocks) {
  static constexpr int kNumXCDs = 8;

  int xcd_id = flat_block_id % kNumXCDs;
  int blocks_per_xcd = (total_blocks + kNumXCDs - 1) / kNumXCDs;
  int pos_in_xcd = flat_block_id / kNumXCDs;
  int logical_block = xcd_id * blocks_per_xcd + pos_in_xcd;

  if (logical_block >= total_blocks) {
    logical_block = flat_block_id % total_blocks;
  }

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
// Causal kernel (no split-KV, single KV chunk covering all of N_kv)
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 1)
    // NOLINTNEXTLINE(misc-definitions-in-headers)
    void fa3_cdna3_prefill_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                  const __half* __restrict__ V, __half* __restrict__ O,
                                  float* __restrict__ LSE, int N_q, int N_kv, int nhead,
                                  int nhead_k, float scale_log2, bool is_causal, int num_q_blocks,
                                  int total_blocks) {
  int flat_id = blockIdx.x;
  XCDAwareMapping map = xcd_aware_remap(flat_id, num_q_blocks, nhead, nhead_k, total_blocks);

  const int q_block = map.q_block;
  const int head_idx = map.head_idx;
  const int head_idx_k = map.head_idx_k;

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
  args.is_causal = is_causal;
  args.q_block = q_block;
  args.head_idx = head_idx;
  args.head_idx_k = head_idx_k;
  args.kv_chunk_start = 0;
  args.kv_chunk_end = N_kv;

  fp32_acc_tile<kBrLocal, kHeadDim> O_acc;
  float row_max, row_sum;

  run_fa3_cdna3_pipeline<kHeadDim>(args, smem, O_acc, row_max, row_sum);

  const int wave_q_start = q_block * kBr + (threadIdx.x / kWaveSize) * kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  if (wave_q_start < N_q) {
    fa3_cdna3_epilogue<kBrLocal, kHeadDim>(O_acc, row_max, row_sum, args.scale_log2, O, LSE,
                                           wave_q_start, nhead, head_idx, N_q, nhead * kHeadDim,
                                           /*lse_row_stride=*/1, /*lse_head_stride=*/N_q,
                                           /*base2_lse=*/false, lane_id);
  }
}

// ---------------------------------------------------------------------------
// Non-causal kernel with split-KV support
// When num_kv_chunks == 1: O_out/LSE_out point to final output (D_stride = nhead*D, LSE_stride =
// N_q) When num_kv_chunks > 1: O_out points to tmp_o with interleaved layout, LSE_out to tmp_lse
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 1)
    // NOLINTNEXTLINE(misc-definitions-in-headers)
    void fa3_cdna3_prefill_kernel_nc(__half* __restrict__ O_out, float* __restrict__ LSE_out,
                                     const __half* __restrict__ Q, const __half* __restrict__ K,
                                     const __half* __restrict__ V, int N_q, int N_kv, int nhead,
                                     int nhead_k, float scale_log2, int num_q_blocks,
                                     int total_blocks, int kv_chunk_size, int num_kv_chunks,
                                     int o_row_stride, int lse_row_stride, int lse_head_stride,
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
  args.is_causal = false;
  args.q_block = q_block;
  args.head_idx = head_idx;
  args.head_idx_k = head_idx_k;
  args.kv_chunk_start = kv_chunk_start;
  args.kv_chunk_end = kv_chunk_end;

  fp32_acc_tile<kBrLocal, kHeadDim> O_acc;
  float row_max, row_sum;

  run_fa3_cdna3_pipeline<kHeadDim>(args, smem, O_acc, row_max, row_sum);

  const int wave_q_start = q_block * kBr + (threadIdx.x / kWaveSize) * kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  // For split-KV: O layout [N_q, num_chunks, nhead, D], LSE layout [N_q, num_chunks, nhead]
  // Offset base pointers by kv_chunk_idx, stride by num_chunks for row access
  __half* o_chunk = O_out + kv_chunk_idx * nhead * kHeadDim;
  float* lse_chunk = LSE_out + kv_chunk_idx * nhead;

  if (wave_q_start < N_q) {
    fa3_cdna3_epilogue<kBrLocal, kHeadDim>(
        O_acc, row_max, row_sum, args.scale_log2, o_chunk, lse_chunk, wave_q_start, nhead, head_idx,
        N_q, o_row_stride, lse_row_stride, lse_head_stride, base2_lse, lane_id);
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
