// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's reshape_and_cache_flash. FlashInfer links the
// symbol-visible AITER module (see flashinfer/jit/aiter_source.py) and calls the
// kernel directly, bypassing AITER's per-call @compile_ops Python dispatch and its
// lazy build-on-first-call. (It does not avoid `import aiter` altogether — the
// backend-availability probe in aiter_utils still imports the package.)
//
// AITER addresses the paged KV cache with vLLM-style absolute slot indices, while
// FlashInfer describes an append with (batch_indices, positions) plus a page table.
// Translating between them used to be seven tensor ops on the Python side; the
// kernel below does it in one launch. Measured 102 us -> 11 us on gfx942.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <string>

// AITER's public header (cache.h) pulls in <torch/extension.h> → full pybind11,
// which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor is at::Tensor,
// so forward-declare the entry point; the linker resolves it against the
// symbol-visible AITER .so. Unlike the norm/rope/activation entry points, this one
// lives in namespace aiter (verified against the built module's symbol table).
//
// The kernel takes its stream from at::hip::getCurrentHIPStream() internally, so
// the caller only has to set the device.
namespace aiter {
void reshape_and_cache_flash(at::Tensor& key, at::Tensor& value, at::Tensor& key_cache,
                             at::Tensor& value_cache, at::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype, at::Tensor& k_scale,
                             at::Tensor& v_scale);
}  // namespace aiter

namespace {

// slot = kv_indices[kv_indptr[batch] + page_within] * page_size + offset_in_page
__global__ void build_slot_mapping_kernel(const int32_t* __restrict__ batch_indices,
                                          const int32_t* __restrict__ positions,
                                          const int32_t* __restrict__ kv_indices,
                                          const int32_t* __restrict__ kv_indptr,
                                          int64_t* __restrict__ slot_mapping, int64_t nnz,
                                          int32_t page_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nnz) return;
  const int32_t pos = positions[i];
  const int32_t page_within = pos / page_size;
  const int32_t offset_in_page = pos - page_within * page_size;
  const int64_t global_page = kv_indices[kv_indptr[batch_indices[i]] + page_within];
  slot_mapping[i] = global_page * page_size + offset_in_page;
}

}  // namespace

void append_paged_kv_cache_aiter(at::Tensor append_key, at::Tensor append_value,
                                 at::Tensor batch_indices, at::Tensor positions,
                                 at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                                 at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor k_scale,
                                 at::Tensor v_scale) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(paged_k_cache.device());

  TORCH_CHECK(paged_k_cache.dim() == 4,
              "paged_k_cache must be [num_pages, page_size, num_kv_heads, head_dim] (NHD), got ",
              paged_k_cache.dim(), " dims");
  TORCH_CHECK(batch_indices.scalar_type() == at::kInt && positions.scalar_type() == at::kInt &&
                  kv_indices.scalar_type() == at::kInt && kv_indptr.scalar_type() == at::kInt,
              "batch_indices/positions/kv_indices/kv_indptr must be int32");
  TORCH_CHECK(batch_indices.numel() == positions.numel(),
              "batch_indices and positions must have the same length, got ", batch_indices.numel(),
              " vs ", positions.numel());
  // AITER indexes append_key/append_value by slot_mapping position, so a shorter
  // append tensor is read past its end. csrc_rocm/page.cu checks the same.
  TORCH_CHECK(batch_indices.numel() == append_key.size(0),
              "batch_indices length must equal append_key.size(0), got ", batch_indices.numel(),
              " vs ", append_key.size(0));
  TORCH_CHECK(append_key.size(0) == append_value.size(0),
              "append_key and append_value must have the same length, got ", append_key.size(0),
              " vs ", append_value.size(0));

  // The kernel dereferences these as raw device pointers, so a tensor on the wrong
  // device faults (or, with peer access, silently yields wrong indices) instead of
  // erroring. The torch ops this shim replaced raised a device-mismatch error.
  const at::Device device = paged_k_cache.device();
  TORCH_CHECK(batch_indices.device() == device && positions.device() == device &&
                  kv_indices.device() == device && kv_indptr.device() == device &&
                  append_key.device() == device && append_value.device() == device &&
                  paged_v_cache.device() == device,
              "all tensors must be on the same device as paged_k_cache (", device, ")");

  const int64_t nnz = batch_indices.numel();
  const int64_t page_size = paged_k_cache.size(1);
  TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);

  at::Tensor slot_mapping = at::empty({nnz}, paged_k_cache.options().dtype(at::kLong));

  if (nnz > 0) {
    // Bind the contiguous copies to named locals: as temporaries they would be
    // destroyed at the end of the launch statement, while the kernel is still
    // pending.
    at::Tensor batch_indices_c = batch_indices.contiguous();
    at::Tensor positions_c = positions.contiguous();
    at::Tensor kv_indices_c = kv_indices.contiguous();
    at::Tensor kv_indptr_c = kv_indptr.contiguous();

    // Indices are small and read once; a plain 1-D grid is enough.
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((nnz + kThreads - 1) / kThreads);
    const hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(build_slot_mapping_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                       batch_indices_c.data_ptr<int32_t>(), positions_c.data_ptr<int32_t>(),
                       kv_indices_c.data_ptr<int32_t>(), kv_indptr_c.data_ptr<int32_t>(),
                       slot_mapping.data_ptr<int64_t>(), nnz, static_cast<int32_t>(page_size));
    // Without this, a launch failure leaves slot_mapping uninitialized and the
    // scatter below writes the append into garbage slots, silently.
    const hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess,
                "build_slot_mapping_kernel launch failed: ", hipGetErrorString(err));
  }

  // "auto" selects the no-quantization path, for which the scales are ignored;
  // they are still required arguments.
  const std::string kv_cache_dtype = "auto";
  aiter::reshape_and_cache_flash(append_key, append_value, paged_k_cache, paged_v_cache,
                                 slot_mapping, kv_cache_dtype, k_scale, v_scale);
}
