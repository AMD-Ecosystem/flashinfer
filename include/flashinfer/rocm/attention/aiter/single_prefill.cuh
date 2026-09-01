// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic (raw pointers + hipStream_t, no at::Tensor) template that
// calls AITER's mha_fwd C++ symbol directly via a cached dlopen function pointer.

#pragma once

#include <flashinfer/rocm/attention/aiter/aiter_loader.h>
#include <flashinfer/rocm/attention/aiter/mha_fwd_args.h>
#include <hip/hip_runtime.h>

#include <ck_tile/host/stream_config.hpp>

namespace flashinfer {

// CK Tile mask_type codes (from CK example mask.hpp):
//   no_mask=0, mask_top_left=1 (causal), mask_bottom_right=2, window_generic=3
inline constexpr int32_t kAiterMaskNone = 0;
inline constexpr int32_t kAiterMaskTopLeft = 1;  // standard causal (qo_len == kv_len)
inline constexpr int32_t kAiterMaskBottomRight =
    2;  // prefill-with-history causal (kv_len > qo_len)

// AITER's ASM v3 fast path is restricted to the (hdim_q, hdim_v) pairs that ship as
// precompiled .co files in aiter_meta/hsa/gfx9{42,50}/fmha_v3_fwd/. AITER also ships
// hd192/hd128 .co files, but the single-prefill entry point hard-requires
// HEAD_DIM_QK == HEAD_DIM_VO (see static_assert below), so only the equal-hdim pair
// is reachable here.
inline constexpr bool AiterAsmV3HdimSupported(uint32_t hdim_q, uint32_t hdim_v) {
  return hdim_q == 128 && hdim_v == 128;
}

// Returns true iff AITER's mha_fwd dispatcher will hit the ASM v3 pipeline for
// these traits (matches the guard in aiter::fmha_fwd_v3).
inline constexpr bool AiterAsmV3Eligible(uint32_t hdim_q, uint32_t hdim_v,
                                         flashinfer::aiter::VariantKey::Dtype dtype,
                                         bool has_logits_cap, int32_t window_left) {
  return AiterAsmV3HdimSupported(hdim_q, hdim_v) &&
         dtype == flashinfer::aiter::VariantKey::Dtype::kBf16 && !has_logits_cap && window_left < 0;
}

// params.lse: [num_qo_heads, qo_len] float32 scratch in natural-log scale; nullptr to skip.
// tmp: unused; accepted for API parity with the FA2 template.
// cu_seqlens_q / cu_seqlens_k: [0, seqlen] arrays on device; consumed only when
// logits_soft_cap > 0 (which forces the varlen/group-mode .so because the mha_fwd
// template has no _logits arm). May be nullptr otherwise.
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename Params>
hipError_t SinglePrefillWithKVCacheDispatched(Params const& params, bool causal,
                                              const char* dtype_str,
                                              flashinfer::aiter::VariantKey::Dtype dtype_enum,
                                              const int32_t* cu_seqlens_q,
                                              const int32_t* cu_seqlens_k, void* /* tmp */,
                                              hipStream_t stream) {
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO, "AITER backend requires HEAD_DIM_QK == HEAD_DIM_VO");

  const bool has_lse = (params.lse != nullptr);
  const bool has_logits_cap = (params.logits_soft_cap > 0.0f);

  // A window wider than the sequence masks nothing; normalizing it here (as AITER's
  // own torch path does) keeps such a call off the _mask variant and its cold build.
  const int32_t window_left = (params.window_left >= static_cast<int32_t>(params.kv_len))
                                  ? -1
                                  : static_cast<int32_t>(params.window_left);
  const bool needs_mask = causal || window_left >= 0;

  const flashinfer::aiter::VariantKey key{
      .dtype = dtype_enum,
      .needs_mask = needs_mask,
      .has_lse = has_lse,
      .has_alibi = false,
      .has_logits_cap = has_logits_cap,
  };

  // AITER ships two dispatcher .so templates: mha_fwd (batch-mode CK + ASM v3) and
  // mha_varlen_fwd (group-mode CK + ASM v3). Single-prefill is batch=1, so the
  // mha_fwd .so is the natural fit — except it has no _logits arm, so logits_soft_cap
  // forces us onto the varlen path.
  using mha_fwd_fn = float (*)(::aiter::mha_fwd_args, ::ck_tile::stream_config const&);
  auto fn = reinterpret_cast<mha_fwd_fn>(
      has_logits_cap ? flashinfer::aiter::get_aiter_mha_varlen_fwd_handle(key)
                     : flashinfer::aiter::get_aiter_mha_fwd_handle(key));

  // Defensive: the varlen pipeline dereferences seqstart_*_ptr in CK Tile. Surface
  // missing pointers as hipErrorInvalidValue here rather than as a memory fault
  // inside AITER.
  if (has_logits_cap && (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr)) {
    return hipErrorInvalidValue;
  }

  ::aiter::mha_fwd_args args{};
  // ASM v3 path (bf16 + hd128) is ~10-15% faster than CK Tile, gated by the same
  // eligibility predicate as aiter::fmha_fwd_v3.
  args.use_asm_v3 =
      AiterAsmV3Eligible(HEAD_DIM_QK, HEAD_DIM_VO, dtype_enum, has_logits_cap, window_left);
  args.v3_api_check = false;
  args.how_v3_bf16_cvt = 0;
  args.data_type = dtype_str;
  args.is_group_mode = has_logits_cap;
  args.bias_type = 0;  // no bias / no alibi
  args.has_lse = has_lse;
  args.qscale_type = 0;
  args.has_sink = false;

  args.q_ptr = static_cast<const void*>(params.q);
  args.k_ptr = static_cast<const void*>(params.k);
  args.v_ptr = static_cast<const void*>(params.v);
  args.o_ptr = static_cast<void*>(params.o);
  args.lse_ptr = static_cast<void*>(params.lse);

  // Group mode encodes batch=1 as seqstart arrays [0, seqlen] on device.
  // Batch mode addresses tensors via stride/nhead_stride only — batch_stride_*
  // are unused at batch=1 and left at their default 0.
  args.seqstart_q_ptr = args.is_group_mode ? static_cast<const void*>(cu_seqlens_q) : nullptr;
  args.seqstart_k_ptr = args.is_group_mode ? static_cast<const void*>(cu_seqlens_k) : nullptr;

  args.seqlen_q = static_cast<int32_t>(params.qo_len);
  args.seqlen_k = static_cast<int32_t>(params.kv_len);
  args.batch = 1;
  args.max_seqlen_q = static_cast<int32_t>(params.qo_len);
  args.hdim_q = static_cast<int32_t>(HEAD_DIM_QK);
  args.hdim_v = static_cast<int32_t>(HEAD_DIM_VO);
  args.nhead_q = static_cast<int32_t>(params.num_qo_heads);
  args.nhead_k = static_cast<int32_t>(params.num_kv_heads);

  args.scale_s = static_cast<float>(params.sm_scale);
  args.logits_soft_cap = static_cast<float>(params.logits_soft_cap);

  args.stride_q = static_cast<int32_t>(params.q_stride_n);
  args.stride_k = static_cast<int32_t>(params.k_stride_n);
  args.stride_v = static_cast<int32_t>(params.v_stride_n);
  // Output is always contiguous NHD [qo_len, num_qo_heads, HEAD_DIM_VO]
  args.stride_o = static_cast<int32_t>(params.num_qo_heads * HEAD_DIM_VO);

  args.nhead_stride_q = static_cast<int32_t>(params.q_stride_h);
  args.nhead_stride_k = static_cast<int32_t>(params.k_stride_h);
  args.nhead_stride_v = static_cast<int32_t>(params.v_stride_h);
  // LSE layout is [num_qo_heads, qo_len] in natural-log — nhead stride = qo_len
  args.nhead_stride_lse = static_cast<int32_t>(params.qo_len);
  args.nhead_stride_o = static_cast<int32_t>(HEAD_DIM_VO);

  // mask_bottom_right: q[i] attends to kv[kv_len−qo_len+i], correct for prefill-with-history.
  // When qo_len == kv_len, mask_bottom_right degenerates to mask_top_left.
  // A window needs it too: right=-1 saturates to the full extent, leaving the
  // left-bound-only band FlashInfer defines. right=0 is the causal convention.
  args.mask_type = needs_mask ? kAiterMaskBottomRight : kAiterMaskNone;
  args.window_size_left = window_left;
  args.window_size_right = causal ? 0 : -1;

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  fn(args, sconfig);
  return hipGetLastError();
}

}  // namespace flashinfer
