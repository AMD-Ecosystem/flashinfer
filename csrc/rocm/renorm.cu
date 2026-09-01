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
// the wrapper now asserts fp32/fp16/bf16 and passes the tensor through. A half
// input would be read at a float stride -- a silent 2x overrun of both buffers.
inline void require_fp32(const at::Tensor& in, const at::Tensor& out, const char* op) {
  TORCH_CHECK(in.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, op,
              " is float32-only on ROCm, got ", in.scalar_type());
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
  require_fp32(probs, renorm_probs, "top_k_renorm_probs");
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == hipSuccess,
              "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                       at::Tensor row_states_buffer) {
  CHECK_INPUT(logits);
  require_fp32(logits, mask_logits, "top_k_mask_logits");
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKMaskLogits<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(mask_logits.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == hipSuccess,
              "TopKMaskLogits failed with error code " + std::string(hipGetErrorString(status)));
}
