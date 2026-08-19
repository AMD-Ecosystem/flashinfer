// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include "pytorch_extension_utils.h"

void append_paged_kv_cache_aiter(at::Tensor append_key, at::Tensor append_value,
                                 at::Tensor batch_indices, at::Tensor positions,
                                 at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                                 at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor k_scale,
                                 at::Tensor v_scale);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("append_paged_kv_cache_aiter", append_paged_kv_cache_aiter);
}
