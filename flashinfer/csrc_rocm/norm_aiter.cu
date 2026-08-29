// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry points for AITER's CK RMSNorm kernels: plain rmsnorm
// (rmsnorm2d) and fused-add rmsnorm (rmsnorm2d_with_add). FlashInfer links the
// symbol-visible AITER module (see flashinfer/jit/aiter_source.py) and calls the
// kernels directly — no Python `import aiter` at runtime.
//
// FlashInfer's fused_add_rmsnorm is in-place:
//   residual = input + residual; input = rmsnorm(residual) * weight
// CK requires its output buffers to be distinct from its inputs, so both entry
// points below stage the result and copy it back rather than aliasing.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/core/GradMode.h>
#include <c10/hip/HIPGuard.h>

// AITER's public header (rmsnorm.h) pulls in <torch/extension.h> → full pybind11,
// which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor is at::Tensor,
// so forward-declare the entry points; the linker resolves them against the
// symbol-visible AITER .so.
void rmsnorm2d_with_add(at::Tensor& out, at::Tensor& input, at::Tensor& residual_in,
                        at::Tensor& residual_out, at::Tensor& weight, double epsilon,
                        int use_model_sensitive_rmsnorm);
// CK 2D forward (the symbol the AITER `rmsnorm2d_fwd` pybind name binds to);
// the in-place overload writes `out` directly.
void rmsnorm2d(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double epsilon,
               int use_model_sensitive_rmsnorm);

void fused_add_rmsnorm_aiter(at::Tensor input, at::Tensor residual, at::Tensor weight, double eps) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  // The copy-back below is an autograd-visible in-place op, unlike the native
  // kernel's raw pointer writes; without this a leaf input would start raising.
  const c10::AutoGradMode no_grad(false);
  // CK expects weight as [1, n]; FlashInfer passes [n]. reshape (not view) so a
  // non-contiguous weight is handled rather than throwing — weight is read-only,
  // so the copy is harmless.
  at::Tensor weight2d = weight.reshape({1, -1});
  // Aliasing either output onto its input corrupts results silently: CK packs
  // several rows per block for small n, so a write lands on a row still unread.
  at::Tensor out = at::empty_like(input);
  at::Tensor residual_out = at::empty_like(residual);
  rmsnorm2d_with_add(out, input, residual, residual_out, weight2d, eps,
                     /*use_model_sensitive_rmsnorm=*/0);
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
    rmsnorm2d(staged, input, weight2d, eps, /*use_model_sensitive_rmsnorm=*/0);
    out.copy_(staged);
    return;
  }
  rmsnorm2d(out, input, weight2d, eps, /*use_model_sensitive_rmsnorm=*/0);
}
