// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include "pytorch_extension_utils.h"

void fused_moe_aiter(at::Tensor out, at::Tensor hidden_states, at::Tensor w1, at::Tensor w2,
                     at::Tensor topk_ids, at::Tensor topk_weights, int64_t block_m,
                     int64_t activation, std::optional<at::Tensor> w1_scale,
                     std::optional<at::Tensor> w2_scale);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("fused_moe_aiter", fused_moe_aiter); }
