# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The HIP source template for the act-and-mul kernels."""

activation_templ = r"""
  #include <gpu_iface/platform.hpp>
  #include <flashinfer/rocm/attention/activation.cuh>
  #include "pytorch_extension_utils.h"
  #include <hip/hip_runtime.h>

  {% set func_name = act_func_name ~ '_and_mul' %}

  using namespace flashinfer;

  {{ act_func_def }}

  void {{ func_name }}(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
    int d = input.size(-1) / 2;
    int64_t num_tokens = input.numel() / input.size(-1);
    if (num_tokens == 0) return;  // empty input → no-op (a 0-sized grid is an invalid launch)

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(out.device());
    auto stream = at::hip::getCurrentHIPStream();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
      uint32_t vec_size = 16 / sizeof(c_type);

      dim3 gridDim, blockDim;
      flashinfer::activation::act_and_mul_launch_dims(d, num_tokens, vec_size,
                                                      out.get_device(), gridDim, blockDim);

      auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>;

      hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, stream,
                         static_cast<c_type*>(out.data_ptr()),
                         static_cast<c_type*>(input.data_ptr()), d);

      hipError_t err = hipGetLastError();
      TORCH_CHECK(err == hipSuccess, "Failed to launch kernel: ", hipGetErrorString(err));

      return true;
    });
  }

  TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
    m.def("{{ func_name }}", {{ func_name }});
  }
  """
