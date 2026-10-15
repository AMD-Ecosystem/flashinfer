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

// Named so flashinfer/rocm/torch_compile.py can mark the written arguments by name.
// CONSERVATIVE: a named schema otherwise declares this void op side-effect free,
// and torch.jit.trace drops it.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(
      torch::schema(
          "ragged_run(Tensor q, Tensor k, Tensor v, Tensor qo_indptr, Tensor kv_indptr, Tensor o, "
          "Tensor? maybe_lse, int mask_mode_code, int layout, int window_left, "
          "float logits_soft_cap, float sm_scale, int max_q_len, int max_kv_len) -> ()",
          c10::AliasAnalysisKind::CONSERVATIVE),
      batch_ragged_prefill_with_kv_cache_aiter);
}
