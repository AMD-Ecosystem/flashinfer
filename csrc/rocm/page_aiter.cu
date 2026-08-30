// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's reshape_and_cache_flash, called against the
// symbol-visible AITER module rather than through AITER's per-call Python
// dispatch. build_slot_mapping_kernel translates FlashInfer's (batch_indices,
// positions) + page table into AITER's vLLM-style absolute slot indices.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <string>

// AITER's real header, not a forward declaration: a signature change must be a
// compile error, not a load-time `undefined symbol`. Includable since 0.1.16,
// which moved cache.h off <torch/extension.h> and onto the POD aiter_tensor_t,
// removing the pybind11 clash against -DPy_LIMITED_API.
#include <aiter_stream.h>
#include <cache.h>

#include "aiter_tensor_compat.h"

namespace {

// Every dimension after the first is packed, whatever stride(0) is. This is the
// layout AITER indexes against; it reads stride(0) but nothing below it.
bool inner_dense(const at::Tensor& t) {
  int64_t expected = 1;
  for (int64_t d = t.dim() - 1; d >= 1; --d) {
    if (t.stride(d) != expected) return false;
    expected *= t.size(d);
  }
  return true;
}

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
  // Before the guard, which otherwise fails an INTERNAL ASSERT on CPU tensors.
  TORCH_CHECK(paged_k_cache.is_cuda(), "paged_k_cache must be on a GPU, got ",
              paged_k_cache.device());

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(paged_k_cache.device());

  TORCH_CHECK(paged_k_cache.dim() == 4,
              "paged_k_cache must be [num_pages, page_size, num_kv_heads, head_dim] (NHD), got ",
              paged_k_cache.dim(), " dims");
  // Shape and dtype: the slot indices come from paged_k_cache, so a differently
  // shaped v-cache scatters out of bounds. Dtype matters because strides count
  // elements, so a narrower v-cache has the identical stride tuple.
  TORCH_CHECK(paged_v_cache.sizes() == paged_k_cache.sizes() &&
                  paged_v_cache.scalar_type() == paged_k_cache.scalar_type(),
              "paged_k_cache and paged_v_cache must have the same shape and dtype, got ",
              paged_k_cache.sizes(), " ", paged_k_cache.scalar_type(), " vs ",
              paged_v_cache.sizes(), " ", paged_v_cache.scalar_type());
  // AITER reads stride(0) but assumes every dimension inside it is dense, so an
  // inner-stride difference is a silent wrong write. stride(0) is deliberately
  // left free: a 5-D combined cache unbinds to halves with stride(0) == 2*S*H*D
  // and dense interiors, which AITER handles and which page.py documents.
  // page.cu takes full stride arrays and accepts anything.
  TORCH_CHECK(inner_dense(paged_k_cache) && inner_dense(paged_v_cache) &&
                  paged_k_cache.strides() == paged_v_cache.strides(),
              "backend='aiter' needs paged_k_cache/paged_v_cache dense inside stride(0) and "
              "identically strided; got ",
              paged_k_cache.strides(), " and ", paged_v_cache.strides(),
              ". Use backend='native', which accepts any strides.");
  TORCH_CHECK(inner_dense(append_key) && inner_dense(append_value),
              "backend='aiter' needs append_key/append_value dense inside stride(0); got ",
              append_key.strides(), " and ", append_value.strides(),
              ". Use backend='native', which accepts any strides.");
  TORCH_CHECK(batch_indices.scalar_type() == at::kInt && positions.scalar_type() == at::kInt &&
                  kv_indices.scalar_type() == at::kInt && kv_indptr.scalar_type() == at::kInt,
              "batch_indices/positions/kv_indices/kv_indptr must be int32");
  // Rank only, for parity with page.cu's CHECK_DIM. Element values stay
  // unchecked, as they are there: validating them needs a device read.
  TORCH_CHECK(batch_indices.dim() == 1 && positions.dim() == 1 && kv_indices.dim() == 1 &&
                  kv_indptr.dim() == 1,
              "batch_indices/positions/kv_indices/kv_indptr must be 1-D, got ", batch_indices.dim(),
              "/", positions.dim(), "/", kv_indices.dim(), "/", kv_indptr.dim());
  TORCH_CHECK(batch_indices.numel() == positions.numel(),
              "batch_indices and positions must have the same length, got ", batch_indices.numel(),
              " vs ", positions.numel());
  // AITER derives num_heads/head_size from append_key's trailing dims, so a wrong
  // rank or head_dim copies the wrong element count per slot instead of erroring.
  TORCH_CHECK(append_key.dim() == 3 && append_value.dim() == 3,
              "append_key/append_value must be [nnz, num_kv_heads, head_dim], got ",
              append_key.dim(), " and ", append_value.dim(), " dims");
  TORCH_CHECK(append_key.sizes() == append_value.sizes(),
              "append_key and append_value must have the same shape, got ", append_key.sizes(),
              " vs ", append_value.sizes());
  TORCH_CHECK(batch_indices.numel() == append_key.size(0),
              "batch_indices length must equal append_key.size(0), got ", batch_indices.numel(),
              " vs ", append_key.size(0));
  TORCH_CHECK(append_key.size(1) == paged_k_cache.size(2),
              "append_key.size(1) must equal num_kv_heads, got ", append_key.size(1), " vs ",
              paged_k_cache.size(2));
  TORCH_CHECK(append_key.size(2) == paged_k_cache.size(3),
              "append_key.size(2) must equal head_dim, got ", append_key.size(2), " vs ",
              paged_k_cache.size(3));
  // AITER dispatches on the *source* dtype and casts the cache pointer to it, so
  // a wider append_key writes past the end of a narrower cache.
  TORCH_CHECK(append_key.scalar_type() == paged_k_cache.scalar_type() &&
                  append_value.scalar_type() == paged_k_cache.scalar_type(),
              "append_key/append_value must have the same dtype as the caches, got ",
              append_key.scalar_type(), " and ", append_value.scalar_type(), " vs ",
              paged_k_cache.scalar_type());

  // Raw device pointers: a tensor on the wrong device faults, or silently yields
  // wrong indices under peer access. k_scale/v_scale included, AITER reads them.
  const at::Device device = paged_k_cache.device();
  TORCH_CHECK(batch_indices.device() == device && positions.device() == device &&
                  kv_indices.device() == device && kv_indptr.device() == device &&
                  append_key.device() == device && append_value.device() == device &&
                  paged_v_cache.device() == device && k_scale.device() == device &&
                  v_scale.device() == device,
              "all tensors must be on the same device as paged_k_cache (", device, ")");

  const int64_t nnz = batch_indices.numel();
  const int64_t page_size = paged_k_cache.size(1);
  TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);

  // AITER's host function does dim3 grid(key.size(0)) with no zero guard, so an
  // empty append would launch a zero-block grid and leave the resulting error
  // pending for an unrelated call to report.
  if (nnz == 0) return;

  // Bind the contiguous copies to named locals: as temporaries they would be
  // destroyed at the end of the launch statement, while the kernel is pending.
  at::Tensor batch_indices_c = batch_indices.contiguous();
  at::Tensor positions_c = positions.contiguous();
  at::Tensor kv_indices_c = kv_indices.contiguous();
  at::Tensor kv_indptr_c = kv_indptr.contiguous();

  at::Tensor slot_mapping = at::empty({nnz}, paged_k_cache.options().dtype(at::kLong));

  // Indices are small and read once; a plain 1-D grid is enough.
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((nnz + kThreads - 1) / kThreads);
  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(build_slot_mapping_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                     batch_indices_c.data_ptr<int32_t>(), positions_c.data_ptr<int32_t>(),
                     kv_indices_c.data_ptr<int32_t>(), kv_indptr_c.data_ptr<int32_t>(),
                     slot_mapping.data_ptr<int64_t>(), nnz, static_cast<int32_t>(page_size));
  // Launch-configuration failures only; an async fault surfaces at the next
  // sync. Without this the scatter below would run on an uninitialized mapping.
  const hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess,
              "build_slot_mapping_kernel launch failed: ", hipGetErrorString(err));

  // "auto" selects the no-quantization path; the scales are ignored but required.
  const std::string kv_cache_dtype = "auto";

  namespace compat = flashinfer::aiter_compat;
  aiter_tensor_t key_a = compat::to_aiter(append_key);
  aiter_tensor_t value_a = compat::to_aiter(append_value);
  aiter_tensor_t key_cache_a = compat::to_aiter(paged_k_cache);
  aiter_tensor_t value_cache_a = compat::to_aiter(paged_v_cache);
  aiter_tensor_t slot_mapping_a = compat::to_aiter(slot_mapping);
  aiter_tensor_t k_scale_a = compat::to_aiter(k_scale);
  aiter_tensor_t v_scale_a = compat::to_aiter(v_scale);

  // The POD API launches on AITER's thread_local stream, which only its Python
  // layer otherwise sets; scoped so the value does not outlive this call.
  const flashinfer::aiter_compat::StreamGuard stream_guard(stream);
  aiter::reshape_and_cache_flash(key_a, value_a, key_cache_a, value_cache_a, slot_mapping_a,
                                 kv_cache_dtype, k_scale_a, v_scale_a);
}
