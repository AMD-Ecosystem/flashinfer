// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's CK fused-add RMSNorm (rmsnorm2d_with_add).
// FlashInfer links the symbol-visible AITER module (see
// flashinfer/jit/aiter_source.py) and calls the kernel directly — no Python
// `import aiter` at runtime.
//
// FlashInfer's fused_add_rmsnorm is in-place:
//   residual = input + residual; input = rmsnorm(residual) * weight
// AITER's CK kernel writes out / residual_out separately; alias them onto
// input / residual to match the in-place semantics.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

// AITER's public header (rmsnorm.h) pulls in <torch/extension.h> → full pybind11,
// which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor is at::Tensor,
// so forward-declare the entry points; the linker resolves them against the
// symbol-visible AITER .so.
void rmsnorm2d_with_add(at::Tensor& out, at::Tensor& input, at::Tensor& residual_in,
                        at::Tensor& residual_out, at::Tensor& weight, double epsilon,
                        int use_model_sensitive_rmsnorm);
void rms_norm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double epsilon);

void fused_add_rmsnorm_aiter(at::Tensor input, at::Tensor residual, at::Tensor weight, double eps) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  // CK expects weight as [1, n]; FlashInfer passes [n]. reshape (not view) so a
  // non-contiguous weight is handled rather than throwing — weight is read-only,
  // so the copy is harmless.
  at::Tensor weight2d = weight.reshape({1, -1});
  rmsnorm2d_with_add(input, input, residual, residual, weight2d, eps,
                     /*use_model_sensitive_rmsnorm=*/0);
}

void rmsnorm_aiter(at::Tensor out, at::Tensor input, at::Tensor weight, double eps) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  rms_norm(out, input, weight, eps);
}
