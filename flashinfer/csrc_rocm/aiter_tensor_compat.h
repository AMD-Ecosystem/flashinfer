// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// at::Tensor -> aiter_tensor_t adapter.
//
// From 0.1.16 AITER's C++ API takes its own POD `aiter_tensor_t` instead of
// `at::Tensor` (15 of 16 public headers migrated). Shims therefore have to
// translate at the boundary.
//
// `aiter_tensor.h` is AITER's *real* header, and it is included deliberately
// rather than vendored: the struct layout is the part that fails silently when
// it drifts (see the mha_fwd_args incident), so taking it from AITER makes any
// future layout change a compile error instead of wrong numbers. The header is
// self-contained -- unlike rope.h and rmsnorm.h it pulls no pybind11, so it is
// safe under FlashInfer's -DPy_LIMITED_API build.
#pragma once

#include <ATen/ATen.h>

#include <aiter_enum.h>
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
  // is_gpu() keys off device_id >= 0, and AITER kernels require device memory.
  out.device_id = t.is_cpu() ? -1 : static_cast<int>(t.device().index());
  return out;
}

}  // namespace flashinfer::aiter_compat
