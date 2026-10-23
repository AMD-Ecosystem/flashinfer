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
#include <flashinfer/rocm/attention/aiter/batch_prefill.cuh>
#include <flashinfer/rocm/enums.hpp>
#include <flashinfer/rocm/layout.cuh>
#include <optional>
#include <string>
#include <utility>

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
//   aiter_flat_gather_idx  [total_kv_tokens] int64 (flat-gather route only; nullopt otherwise)
//   aiter_flat_kv_indptr   [batch+1] int32 (flat-gather route; cumsum of gathered tokens)
//   maybe_q/k/v_descale    [1] float32 per-tensor fp8 descales; required for fp8, else unset
void batch_prefill_with_paged_kv_cache_aiter(
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t window_left,
    double logits_soft_cap, double sm_scale, int64_t page_size, int64_t max_q_len,
    int64_t max_kv_len, std::optional<at::Tensor> aiter_flat_gather_idx,
    std::optional<at::Tensor> aiter_flat_kv_indptr, std::optional<at::Tensor> maybe_q_descale,
    std::optional<at::Tensor> maybe_k_descale, std::optional<at::Tensor> maybe_v_descale) {
  const auto device = q.device();
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  TORCH_CHECK(mask_mode != MaskMode::kCustom, "AITER backend does not support custom mask");

  const auto q_dtype = q.scalar_type();
  const bool is_fp8 = (q_dtype == at::kFloat8_e4m3fnuz || q_dtype == at::kFloat8_e4m3fn);
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16 || is_fp8,
              "AITER backend supports fp16/bf16/fp8 only; got dtype=", q_dtype);
  TORCH_CHECK(paged_k_cache.scalar_type() == q_dtype && paged_v_cache.scalar_type() == q_dtype,
              "q, k, v must share dtype");
  TORCH_CHECK(o.is_contiguous(), "AITER backend requires a contiguous output tensor");
  CHECK_LAST_DIM_CONTIGUOUS(q);
  // fp8 in, bf16 out: AITER ships no fp8-output prefill kernel.
  TORCH_CHECK(o.scalar_type() == (is_fp8 ? at::kBFloat16 : q_dtype),
              "AITER backend requires output dtype ", (is_fp8 ? at::kBFloat16 : q_dtype),
              "; got o=", o.scalar_type(), " q=", q_dtype);

  const float* q_descale_ptr = nullptr;
  const float* k_descale_ptr = nullptr;
  const float* v_descale_ptr = nullptr;
  if (is_fp8) {
    TORCH_CHECK(maybe_q_descale && maybe_k_descale && maybe_v_descale,
                "fp8 prefill requires q/k/v descales: AITER's fp8 kernels have no no-scale "
                "instance, so an unscaled call resolves to no kernel at all.");
    // Per-tensor only. AITER reads element 0 and ignores the rest, so a per-head
    // descale would silently apply head 0's scale to every head.
    auto check_descale = [&device](const at::Tensor& t, const char* name) {
      TORCH_CHECK(t.scalar_type() == at::kFloat && t.numel() == 1, "fp8 ", name,
                  "_descale must be a single float32 (per-tensor); got dtype=", t.scalar_type(),
                  " numel=", t.numel());
      TORCH_CHECK(t.device() == device, "fp8 ", name, "_descale must be on ", device,
                  " (the kernel dereferences it device-side); got ", t.device());
    };
    check_descale(*maybe_q_descale, "q");
    check_descale(*maybe_k_descale, "k");
    check_descale(*maybe_v_descale, "v");
    q_descale_ptr = static_cast<const float*>(maybe_q_descale->data_ptr());
    k_descale_ptr = static_cast<const float*>(maybe_k_descale->data_ptr());
    v_descale_ptr = static_cast<const float*>(maybe_v_descale->data_ptr());
  } else {
    TORCH_CHECK(!maybe_q_descale && !maybe_k_descale && !maybe_v_descale,
                "q/k/v descales are only meaningful for an fp8 query; got dtype=", q_dtype);
  }
  TORCH_CHECK(static_cast<int>(HEAD_DIM_QK) == static_cast<int>(HEAD_DIM_VO),
              "AITER backend requires equal head dims; got HEAD_DIM_QK=", HEAD_DIM_QK,
              " HEAD_DIM_VO=", HEAD_DIM_VO);

  TORCH_CHECK(qo_indptr.scalar_type() == at::kInt && qo_indptr.dim() == 1);
  TORCH_CHECK(paged_kv_indptr.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_indices.scalar_type() == at::kInt);
  TORCH_CHECK(paged_kv_last_page_len.scalar_type() == at::kInt);

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const bool causal = (mask_mode == MaskMode::kCausal);

  const char* dtype_str = is_fp8 ? "fp8bf16" : ((q_dtype == at::kHalf) ? "fp16" : "bf16");
  const auto dtype_enum = is_fp8                   ? flashinfer::aiter::VariantKey::Dtype::kFp8Bf16
                          : (q_dtype == at::kHalf) ? flashinfer::aiter::VariantKey::Dtype::kFp16
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

  TORCH_CHECK(paged_k_cache.size(3) == HEAD_DIM_QK && paged_v_cache.size(3) == HEAD_DIM_VO,
              "paged K/V head dims (", paged_k_cache.size(3), ", ", paged_v_cache.size(3),
              ") do not match the planned ", HEAD_DIM_QK, "/", HEAD_DIM_VO);
  TORCH_CHECK(paged_v_cache.size(1) == paged_k_cache.size(1) &&
                  paged_v_cache.size(2) == paged_k_cache.size(2) && paged_k_cache.size(2) > 0,
              "paged K and V must share a page size and a non-zero head count; got K ",
              paged_k_cache.sizes(), ", V ", paged_v_cache.sizes());

  if (aiter_flat_gather_idx.has_value()) {
    // Flat-gather path: gather pages into contiguous k/v then call mha_fwd group-mode.
    // It dispatches mha_varlen_fwd, which has no fp8 kernel and would ignore the
    // descales checked above, so refuse rather than return unscaled numbers.
    TORCH_CHECK(!is_fp8,
                "fp8 paged prefill requires AITER native paging; this plan resolved to the "
                "flat-gather route, which has no fp8 kernel. Use a natively paged page size "
                "and head dim.");
    TORCH_CHECK(aiter_flat_kv_indptr.has_value(),
                "aiter_flat_kv_indptr must be provided together with aiter_flat_gather_idx");

    const int32_t total_kv = static_cast<int32_t>(aiter_flat_gather_idx->size(0));
    const auto& gather_idx = *aiter_flat_gather_idx;

    // Read only this batch's rows: .contiguous() on the k/v halves of a
    // [pages, 2, page, H, D] pool would copy every page first. View the pool as
    // token rows and remap the page stride (once, as K and V share it).
    std::optional<std::pair<int64_t, at::Tensor>> remapped;
    auto gather_rows = [&](const at::Tensor& cache) -> at::Tensor {
      const int64_t row = cache.size(2) * cache.size(3);
      // A size-1 dim's stride is arbitrary, so it cannot break row contiguity.
      const bool rows_contiguous =
          cache.stride(3) == 1 && (cache.size(2) == 1 || cache.stride(2) == cache.size(3)) &&
          (cache.size(1) == 1 || cache.stride(1) == row) && cache.stride(0) % row == 0;
      if (!rows_contiguous || cache.size(0) == 0) {
        return cache.contiguous()
            .view({-1, cache.size(2), cache.size(3)})
            .index_select(0, gather_idx);
      }
      const int64_t cache_page = cache.size(1);
      const int64_t rows_per_page = cache.stride(0) / row;
      const int64_t num_rows = (cache.size(0) - 1) * rows_per_page + cache_page;
      auto rows =
          cache.as_strided({num_rows, cache.size(2), cache.size(3)}, {row, cache.size(3), 1});
      if (rows_per_page == cache_page) return rows.index_select(0, gather_idx);
      if (!remapped || remapped->first != rows_per_page) {
        auto idx = at::div(gather_idx, cache_page, "floor");
        idx.mul_(rows_per_page - cache_page).add_(gather_idx);
        remapped.emplace(rows_per_page, std::move(idx));
      }
      return rows.index_select(0, remapped->second);
    };
    at::Tensor k_flat = gather_rows(paged_k_cache);
    at::Tensor v_flat = gather_rows(paged_v_cache);

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
    // Native-paged path: page_size in {1, 16, 1024}, NHD [pages, page, H, D].
    // mha_batch_prefill reads past the KV cache at other head dims; plan() routes
    // those to the gather (arch_caps._AITER_NATIVE_PAGED_HEAD_DIMS), this backstops it.
    TORCH_CHECK(HEAD_DIM_QK == 128 || HEAD_DIM_QK == 256,
                "AITER native paged prefill reads past the KV cache at head_dim ", HEAD_DIM_QK);
    // Only the page, token and head strides are passed; head_dim is read packed.
    CHECK_LAST_DIM_CONTIGUOUS(paged_k_cache);
    CHECK_LAST_DIM_CONTIGUOUS(paged_v_cache);
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
        static_cast<int32_t>(window_left), causal, dtype_enum, q_descale_ptr, k_descale_ptr,
        v_descale_ptr, stream);
  }

  TORCH_CHECK(status == hipSuccess,
              "AITER batch-prefill kernel failed: ", hipGetErrorString(status));

  if (maybe_lse) {
    // AITER returns LSE in natural-log [nhead, total_q]; convert to log2 and transpose.
    aiter_lse_scratch.div_(std::log(2.0));
    maybe_lse->copy_(aiter_lse_scratch.t());
  }
}
