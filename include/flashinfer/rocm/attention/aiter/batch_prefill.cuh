// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic (raw pointers + hipStream_t, no at::Tensor) helpers that call
// AITER's C++ symbols directly via cached dlopen function pointers.
//
// Two entry points:
//   BatchPrefillFlatGatherDispatched: gather paged KV then call aiter::mha_fwd group-mode.
//   BatchPrefillNativePagedDispatched: call aiter::mha_batch_prefill with paged KV.

#pragma once

#include <flashinfer/rocm/attention/aiter/aiter_loader.h>
#include <flashinfer/rocm/attention/aiter/mha_batch_prefill_args.h>
#include <flashinfer/rocm/attention/aiter/mha_fwd_args.h>
#include <hip/hip_runtime.h>

#include <ck_tile/host/stream_config.hpp>
#include <cstring>  // memset

namespace flashinfer {

// CK Tile mask_type codes (from CK mask.hpp):
//   no_mask=0, mask_top_left=1, mask_bottom_right=2
// Both flat-gather and native-paged paths use mask_bottom_right for causal to match
// FlashInfer batch_prefill semantics: q[i] attends to k[j <= i + (kv_len - qo_len)],
// allowing the first query token to see all prefix KV when kv_len > qo_len.
inline constexpr int32_t kAiterBatchMaskNone = 0;
inline constexpr int32_t kAiterBatchMaskBottomRight = 2;

// Flat-gather path: k_flat/v_flat are already gathered into [total_kv, nhead_kv, head_dim].
// qo_seqstarts: [batch+1] int32 cumulative Q lengths.
// kv_seqstarts: [batch+1] int32 cumulative KV lengths (flat).
// lse_scratch:  [num_qo_heads, total_qo_len] float32 scratch in nats (nullptr = skip).
// tmp:          unused, accepted for API parity.
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV,
          typename DTypeO>
hipError_t BatchPrefillFlatGatherDispatched(
    const DTypeQ* q, const DTypeKV* k_flat, const DTypeKV* v_flat, DTypeO* o, float* lse_scratch,
    const int32_t* qo_seqstarts, const int32_t* kv_seqstarts, int32_t batch, int32_t total_qo_len,
    int32_t total_kv_len, int32_t max_seqlen_q, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t q_stride_n, int32_t q_stride_h, int32_t k_stride_n, int32_t k_stride_h,
    int32_t v_stride_n, int32_t v_stride_h, float sm_scale, float logits_soft_cap,
    int32_t window_left, bool causal, const char* dtype_str,
    flashinfer::aiter::VariantKey::Dtype dtype_enum, hipStream_t stream) {
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO, "AITER backend requires HEAD_DIM_QK == HEAD_DIM_VO");
  const bool has_lse = (lse_scratch != nullptr);
  const bool has_logits = (logits_soft_cap > 0.0f);
  const bool needs_mask = causal || window_left >= 0;

  const flashinfer::aiter::VariantKey key{
      .dtype = dtype_enum,
      .needs_mask = needs_mask,
      .has_lse = has_lse,
      .has_alibi = false,
      .has_logits_cap = has_logits,
  };

  using mha_fwd_fn = float (*)(::aiter::mha_fwd_args, ::ck_tile::stream_config const&);
  auto fn = reinterpret_cast<mha_fwd_fn>(flashinfer::aiter::get_aiter_mha_varlen_fwd_handle(key));

  ::aiter::mha_fwd_args args{};
  args.use_asm_v3 = false;
  args.v3_api_check = false;
  args.how_v3_bf16_cvt = 0;
  args.data_type = dtype_str;
  args.is_group_mode = true;
  args.bias_type = 0;
  args.has_lse = has_lse;
  args.qscale_type = 0;
  args.has_sink = false;

  args.q_ptr = static_cast<const void*>(q);
  args.k_ptr = static_cast<const void*>(k_flat);
  args.v_ptr = static_cast<const void*>(v_flat);
  args.o_ptr = static_cast<void*>(o);
  args.lse_ptr = static_cast<void*>(lse_scratch);

  args.seqstart_q_ptr = static_cast<const void*>(qo_seqstarts);
  args.seqstart_k_ptr = static_cast<const void*>(kv_seqstarts);

  args.seqlen_q = 0;  // irrelevant in group mode (seqstarts define per-seq lengths)
  args.seqlen_k = 0;
  args.batch = batch;
  args.max_seqlen_q = max_seqlen_q;
  args.hdim_q = static_cast<int32_t>(HEAD_DIM_QK);
  args.hdim_v = static_cast<int32_t>(HEAD_DIM_VO);
  args.nhead_q = num_qo_heads;
  args.nhead_k = num_kv_heads;

  args.scale_s = sm_scale;
  args.logits_soft_cap = logits_soft_cap;

  args.stride_q = q_stride_n;
  args.stride_k = k_stride_n;
  args.stride_v = v_stride_n;
  args.stride_o = static_cast<int32_t>(num_qo_heads * HEAD_DIM_VO);

  args.nhead_stride_q = q_stride_h;
  args.nhead_stride_k = k_stride_h;
  args.nhead_stride_v = v_stride_h;
  args.nhead_stride_lse = total_qo_len;  // lse layout: [nhead, total_q]
  args.nhead_stride_o = static_cast<int32_t>(HEAD_DIM_VO);

  // Batch prefill uses mask_bottom_right so q[0] can attend to all prefix KV tokens
  // (i.e. k[j <= j + kv_len - qo_len]).  mask_top_left would wrongly restrict q[0]
  // to only k[0] when kv_len > qo_len (prefill-with-history / chunked-prefill case).
  args.mask_type = needs_mask ? kAiterBatchMaskBottomRight : kAiterBatchMaskNone;
  args.window_size_left = window_left;
  // right=0 blocks future tokens for causal; -1 saturates to the full extent,
  // leaving the left-bound-only band a non-causal window needs.
  args.window_size_right = causal ? 0 : -1;

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  fn(args, sconfig);
  return hipGetLastError();
}

// Native-paged path: paged_k/paged_v layouts: [max_pages, page_size, nhead_kv, head_dim].
// kv_indptr:          [batch+1] int32, prefix-sum into kv_page_indices.
// kv_page_indices:    [num_total_pages] int32 physical page IDs.
// kv_last_page_lens:  [batch] int32 filled tokens in the last page.
// qo_seqstarts:       [batch+1] int32 cumulative Q lengths.
// lse_scratch:        [num_qo_heads, total_qo_len] float32 scratch in nats (nullptr = skip).
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV,
          typename DTypeO>
hipError_t BatchPrefillNativePagedDispatched(
    const DTypeQ* q, const DTypeKV* paged_k, const DTypeKV* paged_v, DTypeO* o, float* lse_scratch,
    const int32_t* qo_seqstarts, const int32_t* kv_indptr, const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens, int32_t batch, int32_t total_qo_len, int32_t max_seqlen_q,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t num_total_pages, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h,
    // k_stride_p: within-page token stride = k.stride(1) for [NumBlocks, PageSize, Heads, HeadDim]
    // k_stride_h: within-token head stride = k.stride(2)
    // k_batch_stride: cross-page stride = k.stride(0)
    int32_t k_stride_p, int32_t k_stride_h, int32_t k_batch_stride, int32_t v_stride_p,
    int32_t v_stride_h, int32_t v_batch_stride, float sm_scale, float logits_soft_cap,
    int32_t window_left, bool causal, flashinfer::aiter::VariantKey::Dtype dtype_enum,
    hipStream_t stream) {
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO, "AITER backend requires HEAD_DIM_QK == HEAD_DIM_VO");

  const bool has_lse = (lse_scratch != nullptr);
  const bool has_logits = (logits_soft_cap > 0.0f);
  const bool needs_mask = causal || window_left >= 0;

  const flashinfer::aiter::BatchPrefillVariantKey key{
      .dtype = dtype_enum,
      .needs_mask = needs_mask,
      .has_lse = has_lse,
      .has_alibi = false,
      .has_logits_cap = has_logits,
  };

  // mask_enum / bias_enum / quant_scale_enum are declared in CK Tile example headers at
  // global scope.  We can't include those headers here (framework-agnostic rule).  Instead
  // cast the integer values matching the enum constants (verified from mask.hpp/bias.hpp):
  //   mask_enum::no_mask=0, mask_enum::mask_top_left=1
  //   bias_enum::no_bias=0, quant_scale_enum::no_scale=0
  // The function pointer type uses `int` ABI for these enum class parameters.
  using mha_batch_prefill_fn =
      float (*)(::aiter::mha_batch_prefill_args, ::ck_tile::stream_config const&,
                std::string,  // q_dtype_str
                bool,         // is_group_mode
                int,          // mask_enum mask_type
                int,          // bias_enum bias_type
                bool,         // has_lse
                int,          // quant_scale_enum qscale_type
                bool          // use_ext_asm
      );

  auto fn = reinterpret_cast<mha_batch_prefill_fn>(
      flashinfer::aiter::get_aiter_mha_batch_prefill_handle(key));

  const char* dtype_str =
      (dtype_enum == flashinfer::aiter::VariantKey::Dtype::kFp16) ? "fp16" : "bf16";

  ::aiter::mha_batch_prefill_args args{};

  args.q_ptr = static_cast<const void*>(q);
  args.k_ptr = static_cast<const void*>(paged_k);
  args.v_ptr = static_cast<const void*>(paged_v);
  args.bias_ptr = nullptr;
  args.q_descale_ptr = nullptr;
  args.k_descale_ptr = nullptr;
  args.v_descale_ptr = nullptr;
  args.rand_val_ptr = nullptr;
  args.lse_ptr = static_cast<void*>(lse_scratch);
  args.o_ptr = static_cast<void*>(o);

  args.seqstart_q_ptr = static_cast<const void*>(qo_seqstarts);
  args.sink_ptr = nullptr;

  args.seqlen_q = 0;  // irrelevant in group mode (seqstarts define per-seq lengths)
  args.seqlen_k = 0;
  args.batch = batch;
  args.max_seqlen_q = max_seqlen_q;
  args.hdim_q = static_cast<int32_t>(HEAD_DIM_QK);
  args.hdim_v = static_cast<int32_t>(HEAD_DIM_VO);
  args.nhead_q = num_qo_heads;
  args.nhead_k = num_kv_heads;

  args.num_total_pages = num_total_pages;
  args.page_block_size = page_size;
  args.kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
  args.kv_lookup_table = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
  args.kv_indptr = const_cast<void*>(static_cast<const void*>(kv_indptr));
  args.kv_page_indices = const_cast<void*>(static_cast<const void*>(kv_page_indices));
  args.kv_last_page_lens = const_cast<void*>(static_cast<const void*>(kv_last_page_lens));
  args.seqlen_k_ptr = nullptr;
  args.batch_stride_block_table = 0;

  args.scale_s = sm_scale;
  args.scale_p = 1.0f;
  args.scale_o = 1.0f;
  args.logits_soft_cap = logits_soft_cap;

  args.stride_q = q_stride_n;
  args.nhead_stride_q = q_stride_h;
  args.batch_stride_q = 0;  // group mode — batches via seqstarts

  // For LINEAR KV layout: [NumBlocks, PageSize, NumHeads, HeadDim]
  // stride_k   = within-page token stride = k.stride(1) = num_heads * head_dim
  // nhead_stride_k = within-token head stride = k.stride(2) = head_dim
  // batch_stride_k = cross-page stride = k.stride(0) = page_size * num_heads * head_dim
  args.stride_k = k_stride_p;
  args.nhead_stride_k = k_stride_h;
  args.batch_stride_k = k_batch_stride;

  args.stride_v = v_stride_p;
  args.nhead_stride_v = v_stride_h;
  args.batch_stride_v = v_batch_stride;

  args.stride_o = static_cast<int32_t>(num_qo_heads * HEAD_DIM_VO);
  args.nhead_stride_o = static_cast<int32_t>(HEAD_DIM_VO);
  args.batch_stride_o = 0;

  args.nhead_stride_lse = total_qo_len;  // LSE layout: [num_qo_heads, total_qo_len]
  args.batch_stride_lse = 0;

  args.stride_bias = args.nhead_stride_bias = args.batch_stride_bias = 0;
  args.stride_randval = args.nhead_stride_randval = args.batch_stride_randval = 0;

  args.window_size_left = window_left;
  // AITER's own wrapper sets right=0 for causal and leaves it at the caller's value
  // otherwise; -1 saturates to the full extent, which is the left-bound-only band.
  args.window_size_right = causal ? 0 : -1;
  args.sink_size = 0;
  args.mask_type = needs_mask ? kAiterBatchMaskBottomRight : kAiterBatchMaskNone;

  args.p_drop = 0.0f;
  args.s_randval = false;
  // zero-initialize drop_seed_offset (only read when p_drop > 0)
  args.drop_seed_offset =
      std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>(
          std::in_place_index<0>, 0ULL, 0ULL);

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  fn(args, sconfig, dtype_str, /*is_group_mode=*/true, args.mask_type, /*bias_type=*/0, has_lse,
     /*qscale_type=*/0, /*use_ext_asm=*/false);

  return hipGetLastError();
}

}  // namespace flashinfer
