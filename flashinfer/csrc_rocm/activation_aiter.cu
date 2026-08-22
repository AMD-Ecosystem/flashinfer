// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's silu_and_mul C++ API. FlashInfer links the
// symbol-visible AITER module (see flashinfer/jit/aiter_source.py) and calls the
// kernel directly — no Python `import aiter` at runtime.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

// AITER's real activation.h is included rather than forward-declared. That
// became possible at 0.1.16: the header now pulls only aiter_tensor.h, not
// <torch/extension.h>, so it no longer drags in pybind11 and no longer clashes
// with FlashInfer's -DPy_LIMITED_API.
//
// Including it is what keeps this shim honest. AITER changed the signature to
// `silu_and_mul(const aiter_tensor_t&, const aiter_tensor_t&, float limit)`, and
// the old hand-written `at::Tensor&` declaration kept compiling happily and then
// failed at load with `undefined symbol`. A real declaration turns the next such
// change into a compile error instead.
#include <activation.h>
#include <aiter_stream.h>

#include "aiter_tensor_compat.h"

void silu_and_mul_aiter(at::Tensor out, at::Tensor input) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());

  // The stream must be propagated explicitly. The old torch-typed entry point
  // read torch's current stream itself; the POD API launches on
  // aiter::getCurrentHIPStream(), a thread_local that defaults to nullptr and is
  // otherwise only set by AITER's Python layer (aiter_stream.h). Without this the
  // kernel silently runs on the default stream while the surrounding torch ops
  // run on another — correct on the default stream, an ordering hazard anywhere
  // else, which is exactly the case tests on the default stream cannot catch.
  // OptionalHIPGuardMasqueradingAsCUDA above restores the device, not the stream.
  aiter::setCurrentHIPStream(at::hip::getCurrentHIPStream());

  const aiter_tensor_t out_a = flashinfer::aiter_compat::to_aiter(out);
  const aiter_tensor_t in_a = flashinfer::aiter_compat::to_aiter(input);
  // `limit` (new in 0.1.16) gates an optional clamp; 0.0f is AITER's own default
  // and preserves the previous behaviour.
  aiter::silu_and_mul(out_a, in_a, /*limit=*/0.0f);
}
