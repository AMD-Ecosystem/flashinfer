// SPDX-FileCopyrightText: 2025 FlashInfer team.
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "batch_attention_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                   at::Tensor int_workspace_buffer,
                                   at::Tensor page_locked_int_workspace_buffer,
                                   at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len,
                                   int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
                                   int64_t head_dim_o, bool causal);

void BatchPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor plan_info_vec, at::Tensor q, at::Tensor k_cache,
                            at::Tensor v_cache, at::Tensor kv_indices, at::Tensor o,
                            std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                            int64_t layout_code, int64_t num_qo_heads, int64_t num_kv_heads,
                            int64_t page_size, double v_scale, double sm_scale,
                            double logits_soft_cap, std::optional<at::Tensor> maybe_k_cache_sf,
                            std::optional<at::Tensor> maybe_v_cache_sf);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Names match upstream's TVM-FFI exports, which flashinfer/attention/_core.py calls.
  m.def("plan", BatchPagedAttentionPlan);
  m.def("run", BatchPagedAttentionRun);
}
