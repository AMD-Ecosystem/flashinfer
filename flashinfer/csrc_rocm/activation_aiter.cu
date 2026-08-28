// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's silu_and_mul C++ API. FlashInfer links the
// symbol-visible AITER module (see flashinfer/jit/aiter_source.py) and calls the
// kernel directly — no Python `import aiter` at runtime.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

// AITER's real header, not a forward declaration: a signature change must be a
// compile error, not a load-time `undefined symbol`. Includable since 0.1.16,
// which dropped <torch/extension.h> and with it the pybind11 clash against
// FlashInfer's -DPy_LIMITED_API.
#include <activation.h>
#include <aiter_stream.h>

#include "aiter_tensor_compat.h"

void silu_and_mul_aiter(at::Tensor out, at::Tensor input) {
  TORCH_CHECK(out.device() == input.device(), "silu_and_mul: out is on ", out.device(),
              " but input is on ", input.device());
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());

  // The kernel indexes linearly, so strides in aiter_tensor_t are not honoured.
  // AITER's torch entry point used to reject this; the POD API cannot.
  TORCH_CHECK(input.is_contiguous(), "silu_and_mul: input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "silu_and_mul: out must be contiguous");

  const aiter_tensor_t out_a = flashinfer::aiter_compat::to_aiter(out);
  const aiter_tensor_t in_a = flashinfer::aiter_compat::to_aiter(input);

  const flashinfer::aiter_compat::StreamGuard stream_guard(at::hip::getCurrentHIPStream());
  // `limit` (new in 0.1.16) gates an optional clamp; 0.0f is AITER's declared
  // default and preserves the previous behaviour.
  aiter::silu_and_mul(out_a, in_a, /*limit=*/0.0f);
}
