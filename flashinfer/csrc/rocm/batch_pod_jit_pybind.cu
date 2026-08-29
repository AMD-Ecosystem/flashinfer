// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "batch_pod_config.inc"
#include "pytorch_extension_utils.h"

void batch_pod_with_kv_cache_tensor(
    at::Tensor float_workspace_buffer_p, at::Tensor int_workspace_buffer_p,
    at::Tensor plan_info_vec_p, at::Tensor q_p, at::Tensor paged_k_cache_p,
    at::Tensor paged_v_cache_p, at::Tensor qo_indptr_p, at::Tensor paged_kv_indptr_p,
    at::Tensor paged_kv_indices_p, at::Tensor paged_kv_last_page_len_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_mask_indptr_p, std::optional<at::Tensor> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec_d, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl, at::Tensor sm_aware_sched);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("batch_pod_with_kv_cache_tensor", batch_pod_with_kv_cache_tensor);
}
