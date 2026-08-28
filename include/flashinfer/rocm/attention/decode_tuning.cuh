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

}  // namespace decode_tuning
}  // namespace flashinfer

#endif  // FLASHINFER_ROCM_DECODE_TUNING_CUH_
