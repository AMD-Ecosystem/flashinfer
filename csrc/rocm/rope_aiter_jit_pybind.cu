// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include "pytorch_extension_utils.h"

void apply_rope_pos_ids_cos_sin_cache_aiter(at::Tensor query, at::Tensor key, at::Tensor query_out,
                                            at::Tensor key_out, at::Tensor cos_sin_cache,
                                            at::Tensor positions, int64_t head_size, bool is_neox);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rope_pos_ids_cos_sin_cache_aiter", apply_rope_pos_ids_cos_sin_cache_aiter);
}
