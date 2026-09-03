// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <flashinfer/rocm/sampling.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

// v0.6.18 added scratch buffers for its multi-CTA radix kernels and an
// is_deterministic switch for AIR top-p. ROCm keeps the single-CTA
// ternary-search kernels, which need neither and are deterministic already,
// so those parameters are accepted to match the schema and left unused.

// The two top-k ops instantiate their kernels at the caller's dtype rather than
// upcasting the whole tensor. They were fp32-only for a different reason than
// the old comment gave: the kernels are DType-templated, but two of them stored
// through vec_t's float-only `store` overload, so half would not compile.
//
// top_p_renorm_probs stays fp32: sampling.py casts before calling it, so a half
// tensor cannot reach here, and upstream has no dispatch there either.
inline void check_renorm_io(const at::Tensor& in, const at::Tensor& out) {
  CHECK_INPUT(out);
  CHECK_SHAPE(in, out);
  TORCH_CHECK(in.scalar_type() == out.scalar_type(), "input and output dtype must match, got ",
              in.scalar_type(), " and ", out.scalar_type());
  TORCH_CHECK(in.scalar_type() == at::kFloat || in.scalar_type() == at::kHalf ||
                  in.scalar_type() == at::kBFloat16,
              "expected float32, float16 or bfloat16, got ", in.scalar_type());
}

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                        bool is_deterministic, at::Tensor workspace) {
  CHECK_INPUT(probs);
  check_renorm_io(probs, renorm_probs);
  TORCH_CHECK(probs.scalar_type() == at::kFloat,
              "top_p_renorm_probs is fp32 on ROCm; sampling.py casts before calling it");
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  TORCH_CHECK(workspace.numel() == 1,
              "the ternary-search kernel reads no workspace; sampling.py sizes it 1");

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, stream);
  TORCH_CHECK(status == hipSuccess,
              "TopPRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                        at::Tensor row_states_buffer) {
  CHECK_INPUT(probs);
  check_renorm_io(probs, renorm_probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = hipSuccess;
  if (probs.scalar_type() == at::kFloat) {
    status = sampling::TopKRenormProb<float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, stream);
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(probs.scalar_type(), c_type, [&] {
      status = sampling::TopKRenormProb<c_type>(
          static_cast<c_type*>(probs.data_ptr()), static_cast<c_type*>(renorm_probs.data_ptr()),
          has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
          top_k_val, vocab_size, stream);
      return true;
    });
  }

  TORCH_CHECK(status == hipSuccess,
              "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                       at::Tensor row_states_buffer) {
  CHECK_INPUT(logits);
  check_renorm_io(logits, mask_logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = hipSuccess;
  if (logits.scalar_type() == at::kFloat) {
    status = sampling::TopKMaskLogits<float>(
        static_cast<float*>(logits.data_ptr()), static_cast<float*>(mask_logits.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, stream);
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(logits.scalar_type(), c_type, [&] {
      status = sampling::TopKMaskLogits<c_type>(
          static_cast<c_type*>(logits.data_ptr()), static_cast<c_type*>(mask_logits.data_ptr()),
          has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
          top_k_val, vocab_size, stream);
      return true;
    });
  }

  TORCH_CHECK(status == hipSuccess,
              "TopKMaskLogits failed with error code " + std::string(hipGetErrorString(status)));
}
