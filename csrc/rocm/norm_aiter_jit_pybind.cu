// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include "pytorch_extension_utils.h"

void fused_add_rmsnorm_aiter(at::Tensor input, at::Tensor residual, at::Tensor weight, double eps);
void rmsnorm_aiter(at::Tensor out, at::Tensor input, at::Tensor weight, double eps);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fused_add_rmsnorm_aiter", fused_add_rmsnorm_aiter);
  m.def("rmsnorm_aiter", rmsnorm_aiter);
}
