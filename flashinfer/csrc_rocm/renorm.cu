// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <flashinfer/rocm/sampling.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  gpuError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, stream);
  TORCH_CHECK(status == gpuSuccess,
              "TopPRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  gpuError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == gpuSuccess,
              "TopKRenormProb failed with error code " + std::string(hipGetErrorString(status)));
}

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  gpuError_t status = sampling::TopKMaskLogits<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(mask_logits.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TORCH_CHECK(status == gpuSuccess,
              "TopKMaskLogits failed with error code " + std::string(hipGetErrorString(status)));
}
