/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/generic/activation.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }

__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}

__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

void silu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) return;  // empty input → no-op (a 0-sized grid is an invalid launch)

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(out.device());
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);

    dim3 gridDim, blockDim;
    activation::act_and_mul_launch_dims(d, num_tokens, vec_size, out.get_device(), gridDim,
                                        blockDim);

    activation::act_and_mul_kernel<c_type, silu><<<gridDim, blockDim, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "Failed to launch kernel: ", hipGetErrorString(err));

    return true;
  });
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) return;  // empty input → no-op (a 0-sized grid is an invalid launch)

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(out.device());
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 gridDim, blockDim;
    activation::act_and_mul_launch_dims(d, num_tokens, vec_size, out.get_device(), gridDim,
                                        blockDim);

    activation::act_and_mul_kernel<c_type, gelu_tanh><<<gridDim, blockDim, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "Failed to launch kernel: ", hipGetErrorString(err));

    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) return;  // empty input → no-op (a 0-sized grid is an invalid launch)
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(out.device());
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);

    dim3 gridDim, blockDim;
    activation::act_and_mul_launch_dims(d, num_tokens, vec_size, out.get_device(), gridDim,
                                        blockDim);

    activation::act_and_mul_kernel<c_type, gelu><<<gridDim, blockDim, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "Failed to launch kernel: ", hipGetErrorString(err));

    return true;
  });
}
