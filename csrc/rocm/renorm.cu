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

// ROCm's kernels are float32-only, and v0.6.18 stopped casting for these two:
// the wrapper now admits fp16/bf16 and passes the tensor through, so a half
// input would be read at a float stride -- a silent 2x overrun of both buffers.
// Upcast rather than reject; the wrapper itself did this up to 0.5.3.
struct Fp32Pair {
  at::Tensor in, out;
  bool cast_back;
};

inline Fp32Pair as_fp32(const at::Tensor& in, const at::Tensor& out) {
  if (in.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat) {
    return {in, out, false};
  }
  at::Tensor in_f = in.to(at::kFloat);
  // Member form: at::empty_like is not visible in this TU under -xhip.
  return {in_f, in_f.new_empty(in_f.sizes()), true};
}

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                        bool is_deterministic, at::Tensor workspace) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

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
  auto fp32 = as_fp32(probs, renorm_probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(fp32.in.data_ptr()), static_cast<float*>(fp32.out.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == hipSuccess,
              "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
  if (fp32.cast_back) renorm_probs.copy_(fp32.out);
}

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                       at::Tensor row_states_buffer) {
  CHECK_INPUT(logits);
  auto fp32 = as_fp32(logits, mask_logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKMaskLogits<float>(
      static_cast<float*>(fp32.in.data_ptr()), static_cast<float*>(fp32.out.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == hipSuccess,
              "TopKMaskLogits failed with error code " + std::string(hipGetErrorString(status)));
  if (fp32.cast_back) mask_logits.copy_(fp32.out);
}
