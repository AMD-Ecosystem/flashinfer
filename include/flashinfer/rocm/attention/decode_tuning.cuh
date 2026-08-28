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
  // bdx spans half a wavefront here. Widening it to 64 the way single-decode
  // does measures slower on both gfx942 and gfx950 — 16-byte per-lane loads
  // beat the wider reduction. Do not "fix" without re-measuring.
  return std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
}

/*! \brief Threads cooperating on one head, i.e. the width of the in-wave reduction. */
template <typename DTypeKV, uint32_t HEAD_DIM>
constexpr uint32_t BatchDecodeBdx() {
  constexpr uint32_t bdx = HEAD_DIM / BatchDecodeVecSize<DTypeKV, HEAD_DIM>();
  // The reduction shuffles over bdx lanes, so it must fit within one wavefront.
  static_assert(bdx <= 32);
  return bdx;
}

}  // namespace decode_tuning
}  // namespace flashinfer

#endif  // FLASHINFER_ROCM_DECODE_TUNING_CUH_
