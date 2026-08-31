// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include <optional>
#include <string>

#include "pytorch_extension_utils.h"

void batch_decode_with_paged_kv_cache_aiter(
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::string so_path, std::string func_name, int64_t max_context_len, int64_t partition_size,
    int64_t sliding_window, double logits_soft_cap, double sm_scale, int64_t max_blocks_per_seq);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("run", batch_decode_with_paged_kv_cache_aiter);
}
