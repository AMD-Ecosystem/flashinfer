// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// at::Tensor -> aiter_tensor_t adapter, for AITER's POD C++ API (0.1.16+).
//
// Include AITER's real aiter_tensor.h, never a vendored copy: a layout change
// must be a compile error, not wrong numbers. It is safe under
// -DPy_LIMITED_API because it pulls no pybind11.
#pragma once

#include <ATen/ATen.h>
#include <aiter_enum.h>
#include <aiter_stream.h>
#include <aiter_tensor.h>

namespace flashinfer::aiter_compat {

inline AiterDtype to_aiter_dtype(at::ScalarType t) {
  switch (t) {
    case at::kHalf:
      return AITER_DTYPE_fp16;
    case at::kBFloat16:
      return AITER_DTYPE_bf16;
    case at::kFloat:
      return AITER_DTYPE_fp32;
    // OCP (gfx950) and FNUZ (gfx942) e4m3 differ in exponent bias and NaN
    // encoding, but AITER exposes one fp8 enum; the arch picks the meaning.
    case at::kFloat8_e4m3fn:
    case at::kFloat8_e4m3fnuz:
      return AITER_DTYPE_fp8;
    case at::kInt:
      return AITER_DTYPE_i32;
    case at::kShort:
      return AITER_DTYPE_i16;
    case at::kChar:
      return AITER_DTYPE_i8;
    case at::kByte:
      return AITER_DTYPE_u8;
    case at::kLong:
      return AITER_DTYPE_i64;
    default:
      TORCH_CHECK(false, "no aiter_tensor_t dtype for at::ScalarType ", t);
  }
}

// aiter_tensor_t carries fixed shape[8]/strides[8] arrays, matching PyTorch's
// own dimension limit; anything deeper cannot be represented.
inline aiter_tensor_t to_aiter(const at::Tensor& t) {
  TORCH_CHECK(t.dim() <= 8, "aiter_tensor_t supports at most 8 dims, got ", t.dim());

  aiter_tensor_t out{};
  out.ptr = const_cast<void*>(t.data_ptr());
  out.numel_ = static_cast<size_t>(t.numel());
  out.ndim = static_cast<int>(t.dim());
  for (int i = 0; i < out.ndim; ++i) {
    out.shape[i] = t.size(i);
    out.strides[i] = t.stride(i);
  }
  out.dtype_ = to_aiter_dtype(t.scalar_type());
  // Every POD entry point is GPU-only, so reject a host tensor here rather than
  // hand AITER a device_id of -1 and let it fault on a host pointer.
  TORCH_CHECK(t.is_cuda(), "aiter_tensor_t requires a GPU tensor, got ", t.device());
  out.device_id = static_cast<int>(t.device().index());
  return out;
}

// Point AITER's thread_local stream at ours for the duration of a call.
//
// The POD entry points launch on aiter::getCurrentHIPStream(), which defaults
// to nullptr and is otherwise set only by AITER's Python layer. Scoped, because
// the value outlives the call otherwise: a caller inside a temporary
// torch.cuda.Stream would strand a freed handle for the next AITER call on this
// thread. c10's device guard restores the device, not the stream.
class StreamGuard {
 public:
  explicit StreamGuard(hipStream_t stream) : prev_(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(stream);
  }
  ~StreamGuard() { aiter::setCurrentHIPStream(prev_); }

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

 private:
  hipStream_t prev_;
};

}  // namespace flashinfer::aiter_compat
