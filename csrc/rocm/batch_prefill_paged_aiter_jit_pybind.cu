// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include <optional>
#include <string>

#include "batch_prefill_aiter_config.inc"
#include "pytorch_extension_utils.h"

void batch_prefill_with_paged_kv_cache_aiter(
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t window_left,
    double logits_soft_cap, double sm_scale, int64_t page_size, int64_t max_q_len,
    int64_t max_kv_len, std::optional<at::Tensor> aiter_flat_gather_idx,
    std::optional<at::Tensor> aiter_flat_kv_indptr);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("paged_run", batch_prefill_with_paged_kv_cache_aiter);
}
