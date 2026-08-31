// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry points for AITER's RMSNorm kernels: plain rmsnorm and fused-add
// rmsnorm (aiter::rmsnorm / aiter::add_rmsnorm, from module_rmsnorm_quant).
// FlashInfer links the symbol-visible AITER module (see
// flashinfer/jit/aiter_source.py) and calls the kernels directly — no Python
// `import aiter` at runtime.
//
// FlashInfer's fused_add_rmsnorm is in-place:
//   residual = input + residual; input = rmsnorm(residual) * weight
// The kernel requires its output buffers to be distinct from its inputs, so both
// entry points below stage the result and copy it back rather than aliasing.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/core/GradMode.h>
#include <c10/hip/HIPGuard.h>

// AITER's public header (rmsnorm_quant.h) pulls in <torch/extension.h> → full
// pybind11, which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor is
// at::Tensor, so forward-declare the entry points; the linker resolves them
// against the symbol-visible AITER .so. 0.1.20 renamed these from the global
// rmsnorm2d/rmsnorm2d_with_add and moved them into namespace aiter.
namespace aiter {
void add_rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& residual_in,
                 at::Tensor& residual_out, at::Tensor& weight, double epsilon, bool gemma_norm);
void rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double epsilon,
             bool gemma_norm);
}  // namespace aiter

void fused_add_rmsnorm_aiter(at::Tensor input, at::Tensor residual, at::Tensor weight, double eps) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  // The copy-back below is an autograd-visible in-place op, unlike the native
  // kernel's raw pointer writes; without this a leaf input would start raising.
  const c10::AutoGradMode no_grad(false);
  // CK expects weight as [1, n]; FlashInfer passes [n]. This does NOT fix a
  // strided weight -- reshape returns a view and keeps the stride -- so the
  // Python layer rejects that before we get here.
  at::Tensor weight2d = weight.reshape({1, -1});
  // Aliasing either output onto its input corrupts results silently: CK packs
  // several rows per block for small n, so a write lands on a row still unread.
  at::Tensor out = at::empty_like(input);
  at::Tensor residual_out = at::empty_like(residual);
  aiter::add_rmsnorm(out, input, residual, residual_out, weight2d, eps, /*gemma_norm=*/false);
  input.copy_(out);
  residual.copy_(residual_out);
}

void rmsnorm_aiter(at::Tensor out, at::Tensor input, at::Tensor weight, double eps) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  const c10::AutoGradMode no_grad(false);
  // CK expects weight as [1, n]; FlashInfer passes [n] (see fused-add note above).
  at::Tensor weight2d = weight.reshape({1, -1});
  // rmsnorm(x, w, out=x) is a documented idiom, and that alias corrupts the
  // same way; stage it only when the caller actually aliased.
  if (out.data_ptr() == input.data_ptr()) {
    at::Tensor staged = at::empty_like(out);
    aiter::rmsnorm(staged, input, weight2d, eps, /*gemma_norm=*/false);
    out.copy_(staged);
    return;
  }
  aiter::rmsnorm(out, input, weight2d, eps, /*gemma_norm=*/false);
}
