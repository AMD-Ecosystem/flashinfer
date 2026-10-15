// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include <optional>

#include "pytorch_extension_utils.h"

void fused_moe_aiter(at::Tensor out, at::Tensor hidden_states, at::Tensor w1, at::Tensor w2,
                     at::Tensor topk_ids, at::Tensor topk_weights, int64_t block_m,
                     int64_t activation, std::optional<at::Tensor> w1_scale,
                     std::optional<at::Tensor> w2_scale);

// Named so flashinfer/rocm/torch_compile.py can mark the written arguments by name.
// CONSERVATIVE: a named schema otherwise declares this void op side-effect free,
// and torch.jit.trace drops it.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(torch::schema("fused_moe_aiter(Tensor out, Tensor hidden_states, Tensor w1, Tensor w2, "
                      "Tensor topk_ids, "
                      "Tensor topk_weights, int block_m, int activation, Tensor? w1_scale, "
                      "Tensor? w2_scale) -> ()",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        fused_moe_aiter);
}
