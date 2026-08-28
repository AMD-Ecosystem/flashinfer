// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for the AITER batch-paged-prefill C++ harness.
// Mirrors single_prefill_aiter.cu but for paged KV caches.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/aiter/batch_prefill.cuh>
#include <gpu_iface/enums.hpp>
#include <gpu_iface/layout.cuh>
#include <optional>
#include <string>

#include "batch_prefill_aiter_config.inc"
#include "pytorch_extension_utils.h"

using flashinfer::MaskMode;
using flashinfer::QKVLayout;

// paged_run signature:
//   q                   [total_q, nhead_q, HEAD_DIM_QK]
//   paged_k_cache        [max_pages, page_size, nhead_k, HEAD_DIM_KV]  (NHD linear layout)
//   paged_v_cache        same as paged_k_cache
//   qo_indptr            [batch+1] int32
//   paged_kv_indptr      [batch+1] int32 (prefix-sum into page_indices, SGLang style)
//   paged_kv_indices     [num_total_pages] int32 physical page IDs
//   paged_kv_last_page_len [batch] int32 filled tokens in the last page
//   o                    [total_q, nhead_q, HEAD_DIM_VO] (contiguous)
//   maybe_lse            [total_q, nhead_q] float32 (FlashInfer log2 convention, after conversion)
//   mask_mode_code       int64
//   window_left          int64
//   logits_soft_cap      double
//   sm_scale             double
//   page_size            int64
//   max_q_len            int64  (max per-sequence q length in this batch)
//   max_kv_len           int64  (max per-sequence kv length in this batch)
//   aiter_flat_gather_idx  [total_kv_tokens] int64 (non-native page sizes only; nullopt otherwise)
//   aiter_flat_kv_indptr   [batch+1] int32 (non-native page sizes; cumsum of gathered tokens)
void batch_prefill_with_paged_kv_cache_aiter(
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t window_left,
    double logits_soft_cap, double sm_scale, int64_t page_size, int64_t max_q_len,
    int64_t max_kv_len, std::optional<at::Tensor> aiter_flat_gather_idx,
    std::optional<at::Tensor> aiter_flat_kv_indptr) {
  const auto device = q.device();
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  TORCH_CHECK(mask_mode != MaskMode::kCustom, "AITER backend does not support custom mask");

  const auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "AITER backend supports fp16/bf16 only; got dtype=", q_dtype);
  TORCH_CHECK(paged_k_cache.scalar_type() == q_dtype && paged_v_cache.scalar_type() == q_dtype,
              "q, k, v must share dtype");
  TORCH_CHECK(o.is_contiguous(), "AITER backend requires a contiguous output tensor");
  TORCH_CHECK(o.scalar_type() == q_dtype,
              "AITER backend requires output dtype to match input dtype; got o=", o.scalar_type(),
              " q=", q_dtype);
  TORCH_CHECK(static_cast<int>(HEAD_DIM_QK) == static_cast<int>(HEAD_DIM_VO),
              "AITER backend requires equal head dims; got HEAD_DIM_QK=", HEAD_DIM_QK,
              " HEAD_DIM_VO=", HEAD_DIM_VO);

  TORCH_CHECK(qo_indptr.scalar_type() == at::kInt && qo_indptr.dim() == 1);
  TORCH_CHECK(paged_kv_indptr.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_indices.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_last_page_len.scalar_type() == at::kInt);

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const bool causal = (mask_mode == MaskMode::kCausal);

  const char* dtype_str = (q_dtype == at::kHalf) ? "fp16" : "bf16";
  const auto dtype_enum = (q_dtype == at::kHalf) ? flashinfer::aiter::VariantKey::Dtype::kFp16
                                                 : flashinfer::aiter::VariantKey::Dtype::kBf16;

  const int32_t batch = static_cast<int32_t>(qo_indptr.size(0) - 1);
  const int32_t total_qo = static_cast<int32_t>(q.size(0));
  const int32_t num_qo_heads = static_cast<int32_t>(q.size(1));
  const int32_t num_kv_heads = static_cast<int32_t>(paged_k_cache.size(2));

  // LSE scratch: AITER returns [num_qo_heads, total_qo_len] in natural-log scale.
  // FlashInfer expects [total_qo_len, num_qo_heads] in log2 scale.
  at::Tensor aiter_lse_scratch;
  if (maybe_lse) {
    aiter_lse_scratch = at::empty({num_qo_heads, total_qo}, maybe_lse->options().dtype(at::kFloat));
  }
  float* lse_ptr = maybe_lse ? static_cast<float*>(aiter_lse_scratch.data_ptr()) : nullptr;

  hipError_t status;

  if (aiter_flat_gather_idx.has_value()) {
    // Flat-gather path: gather pages into contiguous k/v then call mha_fwd group-mode.
    TORCH_CHECK(aiter_flat_kv_indptr.has_value(),
                "aiter_flat_kv_indptr must be provided together with aiter_flat_gather_idx");

    const int32_t total_kv = static_cast<int32_t>(aiter_flat_gather_idx->size(0));
    const auto& gather_idx = *aiter_flat_gather_idx;

    // Reshape paged cache to 2D token view: [max_pages * page_size, nhead_k, head_dim]
    auto k_2d = paged_k_cache.contiguous().reshape({-1, num_kv_heads, HEAD_DIM_QK});
    auto v_2d = paged_v_cache.contiguous().reshape({-1, num_kv_heads, HEAD_DIM_VO});

    // GPU gather: k_flat = k_2d[gather_idx], v_flat = v_2d[gather_idx]
    at::Tensor k_flat = k_2d.index_select(0, gather_idx).contiguous();
    at::Tensor v_flat = v_2d.index_select(0, gather_idx).contiguous();

    status = flashinfer::BatchPrefillFlatGatherDispatched<HEAD_DIM_QK, HEAD_DIM_VO, DTypeQ, DTypeKV,
                                                          DTypeO>(
        static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(k_flat.data_ptr()),
        static_cast<DTypeKV*>(v_flat.data_ptr()), static_cast<DTypeO*>(o.data_ptr()), lse_ptr,
        static_cast<const int32_t*>(qo_indptr.data_ptr()),
        static_cast<const int32_t*>(aiter_flat_kv_indptr->data_ptr()), batch, total_qo, total_kv,
        static_cast<int32_t>(max_q_len), num_qo_heads, num_kv_heads,
        static_cast<int32_t>(q.stride(0)),       // q_stride_n
        static_cast<int32_t>(q.stride(1)),       // q_stride_h
        static_cast<int32_t>(k_flat.stride(0)),  // k_stride_n (contiguous after gather)
        static_cast<int32_t>(k_flat.stride(1)),  // k_stride_h
        static_cast<int32_t>(v_flat.stride(0)),  // v_stride_n
        static_cast<int32_t>(v_flat.stride(1)),  // v_stride_h
        static_cast<float>(sm_scale), static_cast<float>(logits_soft_cap),
        static_cast<int32_t>(window_left), causal, dtype_str, dtype_enum, stream);
  } else {
    // Native-paged path: paged KV cache with page_size in {128, 256, 1024}.
    // paged_k_cache layout: [max_pages, page_size, nhead_k, head_dim] (NHD linear).
    const int32_t num_total_pages = static_cast<int32_t>(paged_k_cache.size(0));

    status = flashinfer::BatchPrefillNativePagedDispatched<HEAD_DIM_QK, HEAD_DIM_VO, DTypeQ,
                                                           DTypeKV, DTypeO>(
        static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
        static_cast<DTypeKV*>(paged_v_cache.data_ptr()), static_cast<DTypeO*>(o.data_ptr()),
        lse_ptr, static_cast<const int32_t*>(qo_indptr.data_ptr()),
        static_cast<const int32_t*>(paged_kv_indptr.data_ptr()),
        static_cast<const int32_t*>(paged_kv_indices.data_ptr()),
        static_cast<const int32_t*>(paged_kv_last_page_len.data_ptr()), batch, total_qo,
        static_cast<int32_t>(max_q_len), num_qo_heads, num_kv_heads, num_total_pages,
        static_cast<int32_t>(page_size),
        static_cast<int32_t>(q.stride(0)),  // q_stride_n
        static_cast<int32_t>(q.stride(1)),  // q_stride_h
        // For linear layout [NumBlocks, PageSize, NumHeads, HeadDim]:
        //   stride(1) = NumHeads * HeadDim (within-page token stride)
        //   stride(2) = HeadDim (head stride within a page)
        //   stride(0) = PageSize * NumHeads * HeadDim (cross-page stride)
        static_cast<int32_t>(paged_k_cache.stride(1)),  // k_stride_p (within-page token stride)
        static_cast<int32_t>(paged_k_cache.stride(2)),  // k_stride_h (head stride)
        static_cast<int32_t>(paged_k_cache.stride(0)),  // k_batch_stride (cross-page stride)
        static_cast<int32_t>(paged_v_cache.stride(1)),  // v_stride_p
        static_cast<int32_t>(paged_v_cache.stride(2)),  // v_stride_h
        static_cast<int32_t>(paged_v_cache.stride(0)),  // v_batch_stride
        static_cast<float>(sm_scale), static_cast<float>(logits_soft_cap),
        static_cast<int32_t>(window_left), causal, dtype_enum, stream);
  }

  TORCH_CHECK(status == hipSuccess,
              "AITER batch-prefill kernel failed: ", hipGetErrorString(status));

  if (maybe_lse) {
    // AITER returns LSE in natural-log [nhead, total_q]; convert to log2 and transpose.
    aiter_lse_scratch.div_(std::log(2.0));
    maybe_lse->copy_(aiter_lse_scratch.t());
  }
}
