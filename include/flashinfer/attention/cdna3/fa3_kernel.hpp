// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 V8: HIP kernel wrapper and launch interface.
//
// 4 waves x 64 threads = 256 threads, kBr=128, kBc=128, d=256.
// v_mfma_f32_32x32x8f16, TransposedC for both QK and PV GEMMs.
// Double-buffered K + double-buffered row-major V. sched_group_barrier scheduling.
// XCD-aware block reordering for GQA LLC reuse.

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
// HIP kernel
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 1)
    // NOLINTNEXTLINE(misc-definitions-in-headers)
    void fa3_cdna3_prefill_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                  const __half* __restrict__ V, __half* __restrict__ O,
                                  float* __restrict__ LSE, int N, int nhead, int nhead_k,
                                  float scale_log2, bool is_causal, int num_q_blocks,
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
  args.N = N;
  args.nhead = nhead;
  args.nhead_k = nhead_k;
  args.scale_log2 = scale_log2;
  args.is_causal = is_causal;
  args.q_block = q_block;
  args.head_idx = head_idx;
  args.head_idx_k = head_idx_k;

  fp32_acc_tile<kBrLocal, kHeadDim> O_acc;
  float row_max, row_sum;

  run_fa3_cdna3_pipeline<kHeadDim>(args, smem, O_acc, row_max, row_sum);

  const int wave_q_start = q_block * kBr + (threadIdx.x / kWaveSize) * kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  if (wave_q_start < N) {
    fa3_cdna3_epilogue<kBrLocal, kHeadDim>(O_acc, row_max, row_sum, args.scale_log2, O, LSE,
                                           wave_q_start, nhead, head_idx, N, nhead * kHeadDim,
                                           lane_id);
  }
}

// ---------------------------------------------------------------------------
// Non-causal specialized kernel (avoids runtime is_causal branches)
// Compiled separately to avoid HIP compiler template pathology.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 1)
    // NOLINTNEXTLINE(misc-definitions-in-headers)
    void fa3_cdna3_prefill_kernel_nc(const __half* __restrict__ Q, const __half* __restrict__ K,
                                     const __half* __restrict__ V, __half* __restrict__ O,
                                     float* __restrict__ LSE, int N, int nhead, int nhead_k,
                                     float scale_log2, int num_q_blocks, int total_blocks) {
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
  args.N = N;
  args.nhead = nhead;
  args.nhead_k = nhead_k;
  args.scale_log2 = scale_log2;
  args.is_causal = false;
  args.q_block = q_block;
  args.head_idx = head_idx;
  args.head_idx_k = head_idx_k;

  fp32_acc_tile<kBrLocal, kHeadDim> O_acc;
  float row_max, row_sum;

  run_fa3_cdna3_pipeline<kHeadDim>(args, smem, O_acc, row_max, row_sum);

  const int wave_q_start = q_block * kBr + (threadIdx.x / kWaveSize) * kBrLocal;
  const int lane_id = threadIdx.x % kWaveSize;

  if (wave_q_start < N) {
    fa3_cdna3_epilogue<kBrLocal, kHeadDim>(O_acc, row_max, row_sum, args.scale_log2, O, LSE,
                                           wave_q_start, nhead, head_idx, N, nhead * kHeadDim,
                                           lane_id);
  }
}

// ---------------------------------------------------------------------------
// Host-side predicate
// ---------------------------------------------------------------------------

inline bool can_use_fa3_cdna3(int hdim, int seqlen) { return (hdim == 256) && (seqlen <= 8192); }

}  // namespace cdna3
}  // namespace flashinfer
