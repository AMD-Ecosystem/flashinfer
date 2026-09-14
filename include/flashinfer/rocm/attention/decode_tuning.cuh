// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef FLASHINFER_ROCM_DECODE_TUNING_CUH_
#define FLASHINFER_ROCM_DECODE_TUNING_CUH_

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace decode_tuning {

/*!
 * \brief Per-lane vector width for the batched paged-decode kernel.
 *
 * The launcher and the work estimator must derive this identically, or the
 * estimator sizes the split-KV workspace for a different specialization than
 * the one that runs.
 */
template <typename DTypeKV, uint32_t HEAD_DIM>
constexpr uint32_t BatchDecodeVecSize() {
  // bdx deliberately spans half a wavefront; widening it measures slower on
  // both archs. Do not change without re-benchmarking.
  return std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
}

/*! \brief Threads cooperating on one head, i.e. the width of the in-wave reduction. */
template <typename DTypeKV, uint32_t HEAD_DIM>
constexpr uint32_t BatchDecodeBdx() {
  constexpr uint32_t vec_size = BatchDecodeVecSize<DTypeKV, HEAD_DIM>();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  // The reduction shuffles over bdx lanes, so it must fit within one wavefront.
  static_assert(bdx <= 32);
  // compute_qk butterflies over offsets bdx/2..1, which is an all-reduce only
  // for power-of-two bdx; and the kernel treats bdx * vec_size as the head dim.
  static_assert((bdx & (bdx - 1)) == 0, "head_dim yields a non-power-of-two bdx");
  static_assert(bdx * vec_size == HEAD_DIM, "head_dim is not a multiple of vec_size");
  return bdx;
}

/*! \brief KV chunks a block processes concurrently.
 *
 * Targets a 128-thread block; at 2-byte KV / HEAD_DIM 128 that leaves bdz 1 at
 * GROUP_SIZE 8. Two settings buy bdz 2 there and neither recovers the 64-head
 * collapse: floor 256 is 1.10-1.28x *worse* at those cells (5-28% over all 12
 * gfx942 cells), and floor 256 with NUM_STAGES_SMEM halved is 1.01-1.03x, i.e.
 * flat. Neither isolates bdz -- both also move smem and block size -- so this
 * rules out the settings, not the hypothesis. git log has the numbers.
 */
template <typename DTypeKV, uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
constexpr uint32_t BatchDecodeBdz() {
  constexpr uint32_t plane = BatchDecodeBdx<DTypeKV, HEAD_DIM>() * GROUP_SIZE;
  return std::max(128U, plane) / plane;
}

/*!
 * \brief Threads per block, i.e. what dim3(bdx, bdy, bdz) actually launches.
 *
 * Derived from bdz, not the reverse: bdz truncates for a non-power-of-two
 * GROUP_SIZE, so max(128, bdx*bdy) over-states the block.
 */
template <typename DTypeKV, uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
constexpr uint32_t BatchDecodeNumThreads() {
  return BatchDecodeBdx<DTypeKV, HEAD_DIM>() * GROUP_SIZE *
         BatchDecodeBdz<DTypeKV, HEAD_DIM, GROUP_SIZE>();
}

}  // namespace decode_tuning
}  // namespace flashinfer

#endif  // FLASHINFER_ROCM_DECODE_TUNING_CUH_
