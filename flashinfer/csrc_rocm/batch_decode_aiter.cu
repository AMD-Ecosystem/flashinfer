// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for the AITER PA v1 (paged_attention_v1) C++ harness.
// The variant .so is built lazily by AITER on the Python plan() side via
// aiter.csrc.cpp_itfs.pa.pa_v1.compile(); plan() resolves (so_path, func_name)
// from the returned ctypes function pointer and threads both strings through
// to this run() entry.
//
// FlashInfer's paged KV layout (flat indices + indptr + last_page_len) is
// converted in-line to PA v1's expected dense [num_seqs, max_blocks_per_seq]
// block_tables + per-seq context_lens. This conversion is small (O(batch))
// and runs on GPU via torch ops; correctness-first, optimize later if needed.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/aiter/batch_decode.cuh>
#include <optional>
#include <string>
#include <unordered_map>

#include "pytorch_extension_utils.h"

// AITER PA v1 always dereferences k_scale/v_scale pointers (no nullptr branch in
// the jinja-generated kernel). For non-quant (kv_cache_dtype="auto") paths we
// pass two on-device float[1] sentinels = 1.0f. Cache them per-device so we
// don't allocate every call.
static at::Tensor get_unit_scale(at::Device device) {
  static thread_local std::unordered_map<int, at::Tensor> cache;
  const int idx = device.index() >= 0 ? device.index() : 0;
  auto it = cache.find(idx);
  if (it == cache.end()) {
    it = cache.emplace(idx, at::ones({1}, at::TensorOptions().dtype(at::kFloat).device(device)))
             .first;
  }
  return it->second;
}

// Convert flat (paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len) into:
//   block_tables  [num_seqs, max_blocks_per_seq] int32 (right-padded with 0)
//   context_lens  [num_seqs] int32  = (npages-1) * page_size + last_page_len
// page_size is taken from the paged cache layout (size(1) for NHD).
// One fused kernel, not a chain of ATen ops. The tensors are tiny -- 128 KB at the
// largest shape benchmarked -- so this is launch-bound, and the ATen form cost
// ~11 launches per run() against 1-2 for the attention kernel itself.
static std::pair<at::Tensor, at::Tensor> build_block_tables_and_ctxlens(
    const at::Tensor& paged_kv_indices, const at::Tensor& paged_kv_indptr,
    const at::Tensor& paged_kv_last_page_len, int64_t page_size, int64_t num_seqs,
    int64_t max_blocks_per_seq) {
  auto device = paged_kv_indices.device();
  auto int32_opts = at::TensorOptions().dtype(at::kInt).device(device);

  // The kernel indexes these as flat int32 arrays, so unlike the ATen form it
  // cannot absorb a strided view or a short buffer -- either would read OOB.
  TORCH_CHECK(paged_kv_indptr.is_contiguous() && paged_kv_indices.is_contiguous() &&
                  paged_kv_last_page_len.is_contiguous(),
              "AITER PA v1 requires contiguous paged-KV index tensors");
  TORCH_CHECK(paged_kv_indptr.numel() == num_seqs + 1, "paged_kv_indptr must have num_seqs+1 (",
              num_seqs + 1, ") elements; got ", paged_kv_indptr.numel());
  TORCH_CHECK(paged_kv_last_page_len.numel() == num_seqs,
              "paged_kv_last_page_len must have num_seqs (", num_seqs, ") elements; got ",
              paged_kv_last_page_len.numel());

  // empty, not zeros: the kernel writes every slot including the padding.
  at::Tensor block_tables = at::empty({num_seqs, max_blocks_per_seq}, int32_opts);
  at::Tensor context_lens = at::empty({num_seqs}, int32_opts);

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const hipError_t status = flashinfer::AiterPaV1BuildBlockTables(
      paged_kv_indptr.data_ptr<int32_t>(), paged_kv_indices.data_ptr<int32_t>(),
      paged_kv_last_page_len.data_ptr<int32_t>(), block_tables.data_ptr<int32_t>(),
      context_lens.data_ptr<int32_t>(), static_cast<int>(num_seqs),
      static_cast<int>(max_blocks_per_seq), static_cast<int>(page_size), stream);
  TORCH_CHECK(status == hipSuccess,
              "AiterPaV1BuildBlockTables failed: ", hipGetErrorString(status));

  return {block_tables, context_lens};
}

// Run signature:
//   q                       [num_seqs, num_heads, head_size]
//   paged_k_cache, paged_v_cache  [num_blocks, page_size, num_kv_heads, head_size] (NHD)
//   paged_kv_indptr         [num_seqs+1] int32
//   paged_kv_indices        [total_pages] int32
//   paged_kv_last_page_len  [num_seqs]   int32
//   o                       [num_seqs, num_heads, head_size]
//   so_path, func_name      resolved by Python plan() side from aiter.pa_v1.compile()
//   max_context_len, partition_size, sliding_window — resolved at plan() time
//   logits_soft_cap, sm_scale
//   max_blocks_per_seq      precomputed by plan() to avoid GPU sync at run()
void batch_decode_with_paged_kv_cache_aiter(
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::string so_path, std::string func_name, int64_t max_context_len, int64_t partition_size,
    int64_t sliding_window, double logits_soft_cap, double sm_scale, int64_t max_blocks_per_seq) {
  const auto device = q.device();
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);

  const auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "AITER PA v1 supports fp16/bf16 only; got dtype=", q_dtype);
  TORCH_CHECK(paged_k_cache.scalar_type() == q_dtype && paged_v_cache.scalar_type() == q_dtype,
              "q, k, v must share dtype");
  TORCH_CHECK(o.is_contiguous(), "AITER PA v1 requires a contiguous output tensor");
  TORCH_CHECK(o.scalar_type() == q_dtype, "AITER PA v1 requires output dtype to match input dtype");
  TORCH_CHECK(paged_kv_indptr.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_indices.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_last_page_len.scalar_type() == at::kInt);

  const int64_t num_seqs = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t num_kv_heads = paged_k_cache.size(2);  // NHD: [blocks, page, heads, hd]
  const int64_t page_size = paged_k_cache.size(1);

  auto [block_tables, context_lens] =
      build_block_tables_and_ctxlens(paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
                                     page_size, num_seqs, max_blocks_per_seq);

  const int64_t max_num_partitions = (max_context_len + partition_size - 1) / partition_size;

  // Workspace: exp_sums + max_logits + tmp_out, all on q.device.
  const std::size_t ws_bytes = flashinfer::AiterPaV1WorkspaceBytes(
      static_cast<int>(num_seqs), static_cast<int>(num_heads), static_cast<int>(max_num_partitions),
      static_cast<int>(head_size), q.element_size());
  at::Tensor workspace_buffer = at::empty({static_cast<int64_t>(ws_bytes)},
                                          at::TensorOptions().dtype(at::kByte).device(device));

  // k_scale/v_scale sentinels (= 1.0f) — PA v1 unconditionally dereferences these.
  at::Tensor unit_scale = get_unit_scale(device);

  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  hipError_t status = flashinfer::BatchDecodeAiterPaV1Run(
      so_path, func_name, o.data_ptr(), workspace_buffer.data_ptr(), q.data_ptr(),
      paged_k_cache.data_ptr(), paged_v_cache.data_ptr(),
      static_cast<const int32_t*>(block_tables.data_ptr()),
      /*cu_query_lens_ptr=*/nullptr,  // pure decode (one Q token per seq)
      static_cast<const int32_t*>(context_lens.data_ptr()),
      /*alibi_slopes_ptr=*/nullptr,
      /*q_scale_ptr=*/nullptr, static_cast<const float*>(unit_scale.data_ptr()),
      static_cast<const float*>(unit_scale.data_ptr()),
      /*fp8_out_scale_ptr=*/nullptr, static_cast<float>(sm_scale),
      static_cast<int>(max_blocks_per_seq), static_cast<int>(max_num_partitions),
      static_cast<float>(logits_soft_cap), static_cast<int>(num_seqs),
      static_cast<int>(num_kv_heads), static_cast<int>(num_heads), static_cast<int>(q.stride(0)),
      static_cast<int>(paged_k_cache.stride(0)), static_cast<int>(paged_k_cache.stride(2)),
      static_cast<int>(paged_k_cache.stride(1)), static_cast<int>(sliding_window), stream);

  TORCH_CHECK(status == hipSuccess,
              "AITER PA v1 decode kernel failed: ", hipGetErrorString(status));
}
