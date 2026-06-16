// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>

#include "pytorch_extension_utils.h"

void silu_and_mul_aiter(at::Tensor out, at::Tensor input);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("silu_and_mul_aiter", silu_and_mul_aiter); }
