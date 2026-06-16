// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's silu_and_mul C++ API. FlashInfer links the
// symbol-visible AITER module (see flashinfer/jit/aiter_source.py) and calls the
// kernel directly — no Python `import aiter` at runtime.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

// AITER's public header (activation.h) pulls in <torch/extension.h> → full
// pybind11, which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor is
// at::Tensor, so forward-declare the entry point; the linker resolves it against
// the symbol-visible AITER .so.
namespace aiter {
void silu_and_mul(at::Tensor& out, at::Tensor& input);
}  // namespace aiter

void silu_and_mul_aiter(at::Tensor out, at::Tensor input) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  aiter::silu_and_mul(out, input);
}
