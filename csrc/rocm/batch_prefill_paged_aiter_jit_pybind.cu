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
    std::optional<at::Tensor> aiter_flat_kv_indptr, std::optional<at::Tensor> maybe_q_descale,
    std::optional<at::Tensor> maybe_k_descale, std::optional<at::Tensor> maybe_v_descale);

// Named so flashinfer/rocm/torch_compile.py can mark the written arguments by name.
// CONSERVATIVE: a named schema otherwise declares this void op side-effect free,
// and torch.jit.trace drops it.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(torch::schema(
            "paged_run(Tensor q, Tensor paged_k_cache, Tensor paged_v_cache, Tensor qo_indptr, "
            "Tensor paged_kv_indptr, Tensor paged_kv_indices, Tensor paged_kv_last_page_len, "
            "Tensor o, Tensor? maybe_lse, int mask_mode_code, int window_left, "
            "float logits_soft_cap, float sm_scale, int page_size, int max_q_len, int max_kv_len, "
            "Tensor? aiter_flat_gather_idx, Tensor? aiter_flat_kv_indptr, Tensor? maybe_q_descale, "
            "Tensor? maybe_k_descale, Tensor? maybe_v_descale) -> ()",
            c10::AliasAnalysisKind::CONSERVATIVE),
        batch_prefill_with_paged_kv_cache_aiter);
}
