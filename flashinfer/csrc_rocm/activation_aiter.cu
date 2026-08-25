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

namespace {

// The POD API launches on aiter::getCurrentHIPStream(), a thread_local that
// defaults to nullptr and is otherwise set only by AITER's Python layer; the
// old torch-typed entry point read torch's stream itself. Scoped, because the
// value outlives the call otherwise: a caller inside a temporary
// torch.cuda.Stream would strand a freed handle for the next AITER call on
// this thread. c10's device guard restores the device, not the stream.
class AiterStreamGuard {
 public:
  explicit AiterStreamGuard(hipStream_t stream) : prev_(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(stream);
  }
  ~AiterStreamGuard() { aiter::setCurrentHIPStream(prev_); }

  AiterStreamGuard(const AiterStreamGuard&) = delete;
  AiterStreamGuard& operator=(const AiterStreamGuard&) = delete;

 private:
  hipStream_t prev_;
};

}  // namespace

void silu_and_mul_aiter(at::Tensor out, at::Tensor input) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());

  // The kernel indexes linearly, so strides in aiter_tensor_t are not honoured.
  // AITER's torch entry point used to reject this; the POD API cannot.
  TORCH_CHECK(input.is_contiguous(), "silu_and_mul: input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "silu_and_mul: out must be contiguous");

  const aiter_tensor_t out_a = flashinfer::aiter_compat::to_aiter(out);
  const aiter_tensor_t in_a = flashinfer::aiter_compat::to_aiter(input);

  const AiterStreamGuard stream_guard(at::hip::getCurrentHIPStream());
  // `limit` (new in 0.1.16) gates an optional clamp; 0.0f is AITER's declared
  // default and preserves the previous behaviour.
  aiter::silu_and_mul(out_a, in_a, /*limit=*/0.0f);
}
