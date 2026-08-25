// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic helper that invokes AITER's PA v1 (paged_attention_v1) extern-C
// JIT entry via a cached dlopen+dlsym handle. The .so is per-variant (template params
// baked in by AITER's compile_template_op) — the Python plan() side resolves the
// (so_path, func_name) by calling aiter.csrc.cpp_itfs.pa.pa_v1.compile() and passes
// both strings down here.

#pragma once

#include <flashinfer/attention/aiter/aiter_loader.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace flashinfer {

// The PA v1 jinja template uses plain `int*` for block_tables/context_lens.
// Guard against exotic platforms where int != int32_t (would cause silent ABI mismatch).
static_assert(sizeof(int) == 4 && std::is_same_v<int, int32_t>,
              "AiterPaV1Fn assumes int == int32_t; platform not supported");

// Extern-C signature emitted by aiter/csrc/cpp_itfs/pa/pa_v1.cpp.jinja.
// Pinned to amd-aiter 0.1.10 — regenerate alongside any AITER pa_v1 template change.
using AiterPaV1Fn = void (*)(void* out_ptr, void* workspace_buffer_ptr, void* query_ptr,
                             void* key_cache_ptr, void* value_cache_ptr, int* block_tables_ptr,
                             int* cu_query_lens_ptr, int* context_lens_ptr,
                             const float* alibi_slopes_ptr, const float* q_scale_ptr,
                             const float* k_scale_ptr, const float* v_scale_ptr,
                             const float* fp8_out_scale_ptr, float scale,
                             int max_num_blocks_per_seq, int max_num_partitions,
                             float logits_soft_cap, int num_seqs, int num_kv_heads, int num_heads,
                             int q_stride, int kv_block_stride, int kv_head_stride,
                             int kv_seq_stride, int sliding_window, void* stream);

// Bytes of workspace needed by PA v1:
//   exp_sums   : float [num_seqs, num_heads, max_num_partitions]
//   max_logits : float [num_seqs, num_heads, max_num_partitions]
//   tmp_out    : dtype [num_seqs, num_heads, max_num_partitions, head_size]
inline std::size_t AiterPaV1WorkspaceBytes(int num_seqs, int num_heads, int max_num_partitions,
                                           int head_size, std::size_t dtype_bytes) {
  const std::size_t per_part = static_cast<std::size_t>(num_seqs) * num_heads * max_num_partitions;
  return per_part * (2 * sizeof(float) + dtype_bytes * static_cast<std::size_t>(head_size));
}

// Dense block table + context lengths from FlashInfer's flat paged-KV indexing.
//
//   block_tables[i][j] = j < npages(i) ? indices[indptr[i] + j] : 0
//   context_lens[i]    = npages(i) > 0 ? (npages(i) - 1) * page_size + last_page_len[i] : 0
//
// The npages == 0 arm is not decoration: without it the expression underflows to
// -page_size + last_page_len and hands AITER a negative context length.
//
// One thread per (seq, slot). Grid-stride so the launch config is independent of
// max_blocks_per_seq. `static`, not weak-linked like the templates elsewhere in
// include/ -- a non-template __global__ in a header would collide if a second
// translation unit ever included this.
//
// The iteration space uses slots_per_seq = max(max_blocks_per_seq, 1) so the
// slot-0 threads still run, and still write context_lens, when every sequence
// has zero pages.
static __global__ void AiterPaV1BuildBlockTablesKernel(
    const int32_t* __restrict__ indptr, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ last_page_len, int32_t* __restrict__ block_tables,
    int32_t* __restrict__ context_lens, int num_seqs, int max_blocks_per_seq, int slots_per_seq,
    int page_size, int64_t indices_numel) {
  const int64_t total = static_cast<int64_t>(num_seqs) * slots_per_seq;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int seq = static_cast<int>(idx / slots_per_seq);
    const int slot = static_cast<int>(idx % slots_per_seq);
    const int32_t begin = indptr[seq];
    const int32_t npages = indptr[seq + 1] - begin;

    if (slot < max_blocks_per_seq) {
      // Out-of-range slots read as 0 rather than off the end of indices[]. The
      // ATen index_select this replaced raised instead, but indptr is device
      // memory and checking it host-side would cost a sync every run(); a
      // defined value beats an OOB read, at the cost of masking a bad indptr.
      const int64_t src = static_cast<int64_t>(begin) + slot;
      const bool in_range = slot < npages && src >= 0 && src < indices_numel;
      block_tables[static_cast<int64_t>(seq) * max_blocks_per_seq + slot] =
          in_range ? indices[src] : 0;
    }

    // Fold the context-length pass into the slot-0 threads rather than paying a
    // second launch for num_seqs elements.
    if (slot == 0) {
      context_lens[seq] = npages > 0 ? (npages - 1) * page_size + last_page_len[seq] : 0;
    }
  }
}

// Launches the above. Writes every element of both outputs, so neither needs
// pre-zeroing.
inline hipError_t AiterPaV1BuildBlockTables(const int32_t* indptr, const int32_t* indices,
                                            const int32_t* last_page_len, int32_t* block_tables,
                                            int32_t* context_lens, int num_seqs,
                                            int max_blocks_per_seq, int page_size,
                                            int64_t indices_numel, hipStream_t stream) {
  if (num_seqs == 0) return hipSuccess;
  const int slots_per_seq = max_blocks_per_seq > 0 ? max_blocks_per_seq : 1;
  const int64_t total = static_cast<int64_t>(num_seqs) * slots_per_seq;
  constexpr int kThreads = 256;
  // Cap the grid so a huge batch does not produce an unbounded block count; the
  // grid-stride loop covers the remainder.
  const int64_t want = (total + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 4096 ? want : 4096);
  hipLaunchKernelGGL(AiterPaV1BuildBlockTablesKernel, dim3(blocks), dim3(kThreads), 0, stream,
                     indptr, indices, last_page_len, block_tables, context_lens, num_seqs,
                     max_blocks_per_seq, slots_per_seq, page_size, indices_numel);
  return hipGetLastError();
}

inline hipError_t BatchDecodeAiterPaV1Run(
    const std::string& so_path, const std::string& func_name, void* out_ptr,
    void* workspace_buffer_ptr, const void* query_ptr, const void* key_cache_ptr,
    const void* value_cache_ptr, const int32_t* block_tables_ptr, const int32_t* cu_query_lens_ptr,
    const int32_t* context_lens_ptr, const float* alibi_slopes_ptr, const float* q_scale_ptr,
    const float* k_scale_ptr, const float* v_scale_ptr, const float* fp8_out_scale_ptr, float scale,
    int max_num_blocks_per_seq, int max_num_partitions, float logits_soft_cap, int num_seqs,
    int num_kv_heads, int num_heads, int q_stride, int kv_block_stride, int kv_head_stride,
    int kv_seq_stride, int sliding_window, hipStream_t stream) {
  auto fn = reinterpret_cast<AiterPaV1Fn>(
      flashinfer::aiter::get_aiter_extern_c_handle(so_path, func_name));
  fn(out_ptr, workspace_buffer_ptr, const_cast<void*>(query_ptr), const_cast<void*>(key_cache_ptr),
     const_cast<void*>(value_cache_ptr), const_cast<int*>(block_tables_ptr),
     const_cast<int*>(cu_query_lens_ptr), const_cast<int*>(context_lens_ptr), alibi_slopes_ptr,
     q_scale_ptr, k_scale_ptr, v_scale_ptr, fp8_out_scale_ptr, scale, max_num_blocks_per_seq,
     max_num_partitions, logits_soft_cap, num_seqs, num_kv_heads, num_heads, q_stride,
     kv_block_stride, kv_head_stride, kv_seq_stride, sliding_window, static_cast<void*>(stream));
  return hipGetLastError();
}

}  // namespace flashinfer
