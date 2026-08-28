// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include <optional>

#include "batch_prefill_aiter_config.inc"
#include "pytorch_extension_utils.h"

void batch_ragged_prefill_with_kv_cache_aiter(at::Tensor q, at::Tensor k, at::Tensor v,
                                              at::Tensor qo_indptr, at::Tensor kv_indptr,
                                              at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                              int64_t mask_mode_code, int64_t layout,
                                              int64_t window_left, double logits_soft_cap,
                                              double sm_scale, int64_t max_q_len,
                                              int64_t max_kv_len);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("ragged_run", batch_ragged_prefill_with_kv_cache_aiter);
}
