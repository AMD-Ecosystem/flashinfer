// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pytorch_extension_utils.h"

void fa3_cdna3_single_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                              std::optional<at::Tensor> maybe_lse, bool is_causal, at::Tensor tmp);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("run", fa3_cdna3_single_prefill); }
