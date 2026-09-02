// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HIP-specific sampling implementation
// Define CUDA types before PyTorch headers for HIP compatibility
#include <hip/hip_runtime.h>
#ifndef cudaStream_t
typedef hipStream_t cudaStream_t;
#endif

#include <ATen/Utils.h>
#include <ATen/core/Generator.h>

// Use HIP generator header if available, otherwise fall back to CUDA header
#if __has_include(<ATen/hip/HIPGeneratorImpl.h>)
#include <ATen/hip/HIPGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <flashinfer/rocm/sampling.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

// v0.6.18 replaced the scalar philox pair with optional per-request tensors and
// added a `valid` output. The ROCm kernels have neither, so both are absorbed
// here rather than in the kernels.

// One scalar seed covers the whole batch; a per-request tensor has no kernel to
// route to, so reject it instead of silently sampling every row from seed_val.
inline void reject_per_request_seed(const std::optional<at::Tensor>& maybe_seed_arr,
                                    const std::optional<at::Tensor>& maybe_offset_arr) {
  TORCH_CHECK(!maybe_seed_arr.has_value() && !maybe_offset_arr.has_value(),
              "per-request seed/offset tensors are not supported on ROCm; "
              "pass int seed/offset instead");
}

// The kernels write `valid` per row now; this only checks the buffer they write
// into. It replaced a fill_(true), which is why the shape checks live here.
inline void check_valid_out(at::Tensor valid, unsigned int batch_size) {
  CHECK_INPUT(valid);
  CHECK_DIM(1, valid);
  CHECK_EQ(valid.size(0), static_cast<int64_t>(batch_size));
  TORCH_CHECK(valid.scalar_type() == at::kBool, "valid must be bool, got ", valid.scalar_type());
}

void softmax(at::Tensor workspace_buffer, at::Tensor logits, at::Tensor output,
             std::optional<at::Tensor> maybe_temperature_arr, double temperature_val,
             bool enable_pdl) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);

  bool has_temperature_arr = maybe_temperature_arr.has_value();

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::OnlineSoftmax<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(output.data_ptr()), batch_size,
      vocab_size,
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr->data_ptr()) : nullptr,
      static_cast<float>(temperature_val), workspace_buffer.data_ptr(),
      workspace_buffer.element_size() * workspace_buffer.size(0), enable_pdl, stream);
  TORCH_CHECK(status == hipSuccess,
              "OnlineSoftmax failed with error code " + std::string(hipGetErrorString(status)));
}

void sampling_from_logits(at::Tensor logits, at::Tensor output,
                          std::optional<at::Tensor> maybe_indices, bool deterministic,
                          std::optional<at::Tensor> maybe_seed_arr, int64_t philox_seed,
                          std::optional<at::Tensor> maybe_offset_arr, int64_t philox_offset) {
  CHECK_INPUT(logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = logits.size(1);

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::SamplingFromLogits(
      static_cast<float*>(logits.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess, "SamplingFromLogits failed with error code " +
                                        std::string(hipGetErrorString(status)));
}

void sampling_from_probs(at::Tensor probs, at::Tensor output, at::Tensor valid,
                         std::optional<at::Tensor> maybe_indices, bool deterministic,
                         std::optional<at::Tensor> maybe_seed_arr, int64_t philox_seed,
                         std::optional<at::Tensor> maybe_offset_arr, int64_t philox_offset) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);
  check_valid_out(valid, batch_size);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::SamplingFromProb(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      valid.data_ptr<bool>(),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess,
              "SamplingFromProbs failed with error code " + std::string(hipGetErrorString(status)));
}

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor output, at::Tensor valid,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, std::optional<at::Tensor> maybe_seed_arr,
                               int64_t philox_seed, std::optional<at::Tensor> maybe_offset_arr,
                               int64_t philox_offset) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);
  check_valid_out(valid, batch_size);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      valid.data_ptr<bool>(),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess, "TopPSamplingFromProbs failed with error code " +
                                        std::string(hipGetErrorString(status)));
}

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor output, at::Tensor valid,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, std::optional<at::Tensor> maybe_seed_arr,
                               int64_t philox_seed, std::optional<at::Tensor> maybe_offset_arr,
                               int64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);
  check_valid_out(valid, batch_size);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      valid.data_ptr<bool>(),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_k_arr ? static_cast<float*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess, "TopKSamplingFromProbs failed with error code " +
                                        std::string(hipGetErrorString(status)));
}

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor output, at::Tensor valid,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, std::optional<at::Tensor> maybe_seed_arr,
                               int64_t philox_seed, std::optional<at::Tensor> maybe_offset_arr,
                               int64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_min_p_arr = maybe_min_p_arr.has_value();

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);
  check_valid_out(valid, batch_size);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::MinPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_min_p_arr ? static_cast<float*>(maybe_min_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()), valid.data_ptr<bool>(),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, min_p_val, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess, "MinPSamplingFromProb failed with error code " +
                                        std::string(hipGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor output, at::Tensor valid,
                                     std::optional<at::Tensor> maybe_indices,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, std::optional<at::Tensor> maybe_seed_arr,
                                     int64_t philox_seed,
                                     std::optional<at::Tensor> maybe_offset_arr,
                                     int64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);
  check_valid_out(valid, batch_size);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()), valid.data_ptr<bool>(),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, top_k_val, top_p_val, vocab_size, deterministic,
      static_cast<uint64_t>(philox_seed), static_cast<uint64_t>(philox_offset), stream);
  TORCH_CHECK(status == hipSuccess, "TopKTopPSamplingFromProbs failed with error code " +
                                        std::string(hipGetErrorString(status)));
}

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor target_probs, at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num, bool deterministic,
                                std::optional<at::Tensor> maybe_seed_arr, int64_t philox_seed,
                                std::optional<at::Tensor> maybe_offset_arr, int64_t philox_offset) {
  CHECK_INPUT(draft_probs);
  CHECK_INPUT(draft_token_ids);
  CHECK_INPUT(target_probs);
  auto device = draft_probs.device();
  CHECK_EQ(draft_token_ids.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(3, draft_probs);      // draft_probs: (batch_size, num_speculate_tokens, vocab_size)
  CHECK_DIM(2, draft_token_ids);  // draft_token_ids: (batch_size, num_speculate_tokens)
  CHECK_DIM(3, target_probs);  // target_probs: (batch_size, num_speculate_tokens + 1, vocab_size)
  unsigned int batch_size = draft_probs.size(0);
  unsigned int num_speculate_tokens = draft_probs.size(1);
  unsigned int vocab_size = draft_probs.size(2);
  CHECK_EQ(batch_size, draft_token_ids.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_speculate_tokens + 1, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, output_accepted_token_num.size(0));
  CHECK_EQ(batch_size, output_emitted_draft_token_num.size(0));

  reject_per_request_seed(maybe_seed_arr, maybe_offset_arr);

  const at::cuda::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = at::cuda::getCurrentHIPStream();
  hipError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()), static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_draft_token_num.data_ptr()), batch_size,
      num_speculate_tokens, vocab_size, deterministic, static_cast<uint64_t>(philox_seed),
      static_cast<uint64_t>(philox_offset), stream);

  TORCH_CHECK(status == hipSuccess, "ChainSpeculativeSampling failed with error code " +
                                        std::string(hipGetErrorString(status)));
}
