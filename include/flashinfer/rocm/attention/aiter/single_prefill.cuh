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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <flashinfer/rocm/gpu_runtime_compat.hpp>
#include <stdexcept>

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

// Cheap pre-filter for the ASM v3 arm, mirroring the guard in aiter::fmha_fwd_v3.
// AITER's own v3_api_check probe is the authoritative test, but it needs the asm .so
// dlopened and it logs a warning on every rejection, so screen the obvious misses here
// rather than probing for each fp16 or hd64 caller.
inline constexpr bool AiterAsmV3Eligible(uint32_t hdim_q, uint32_t hdim_v,
                                         flashinfer::aiter::VariantKey::Dtype dtype,
                                         bool has_logits_cap, int32_t window_left) {
  return AiterAsmV3HdimSupported(hdim_q, hdim_v) &&
         dtype == flashinfer::aiter::VariantKey::Dtype::kBf16 && !has_logits_cap && window_left < 0;
}

// Smallest qo_len at which the asm arm is worth taking, by architecture. Mirrors
// _AITER_ASM_PREFILL_MIN_QO_LEN in flashinfer/rocm/arch_caps.py, which carries the
// measurement; 0 means never.
//
// These macros are what tests/rocm/test_aiter_asm_routing.py compares against the
// Python table, and they are also what the function returns -- a marker comment
// would let the two drift while the test kept passing.
#define FLASHINFER_AITER_ASM_MIN_QO_LEN_GFX942 0  // never: non-monotonic on CDNA3
#define FLASHINFER_AITER_ASM_MIN_QO_LEN_GFX950 2048

inline uint32_t AiterAsmPrefillMinQoLen(const char* arch) {
  if (arch == nullptr) return 0u;
  if (std::strcmp(arch, "gfx950") == 0) return FLASHINFER_AITER_ASM_MIN_QO_LEN_GFX950;
  if (std::strcmp(arch, "gfx942") == 0) return FLASHINFER_AITER_ASM_MIN_QO_LEN_GFX942;
  return 0u;  // an unrecognised arch is treated as "never", same as a measured loss
}

// One-shot stderr note under FLASHINFER_AITER_ASM_VERBOSE=1, so which arm ran is
// observable. Without it nothing distinguishes "asm served this" from "asm was
// silently unavailable and CK Tile served it" -- the numerics are the same either
// way, so tests and operators both need a signal.
inline void AiterAsmPrefillNote(const char* what) {
  static const bool verbose = [] {
    const char* v = std::getenv("FLASHINFER_AITER_ASM_VERBOSE");
    return v != nullptr && std::strcmp(v, "0") != 0;
  }();
  if (verbose) std::fprintf(stderr, "[flashinfer] aiter asm prefill: %s\n", what);
}

// FLASHINFER_AITER_ASM_PREFILL=0 pins the CK Tile arm. AITER aborts the process on
// several asm failure modes rather than returning, so an operator-side off switch is
// part of the contract, not a convenience.
inline bool AiterAsmPrefillEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("FLASHINFER_AITER_ASM_PREFILL");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  return enabled;
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
  // Set per arm below: the CK Tile .so ignores this field entirely (it is built
  // -DFAV2_ON=1 with no -DFAV3_ON), and the asm .so returns -1 unless it is true.
  args.use_asm_v3 = false;
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
  // Normalize any negative sentinel to -1: AITER's config lookup matches on -1
  // exactly, so -2 would miss and poison the probe cache for later valid calls.
  args.window_size_left = window_left < 0 ? -1 : window_left;
  args.window_size_right = causal ? 0 : -1;

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  // Pure-predicate checks first, so an fp16, hd64, windowed or below-threshold caller
  // reaches CK Tile without paying a HIP runtime query it can never benefit from.
  bool asm_wanted =
      AiterAsmPrefillEnabled() &&
      AiterAsmV3Eligible(HEAD_DIM_QK, HEAD_DIM_VO, dtype_enum, has_logits_cap, window_left);
  if (asm_wanted) {
    int device = 0;
    uint32_t asm_min_qo_len = 0;
    if (hipGetDevice(&device) == hipSuccess) {
      try {
        asm_min_qo_len = AiterAsmPrefillMinQoLen(getGcnArchName(device));
      } catch (const std::exception&) {
        asm_min_qo_len = 0;  // FI_HIP_CALL throws; an unreadable arch stays on CK Tile
      }
    }
    asm_wanted = asm_min_qo_len > 0 && params.qo_len >= asm_min_qo_len;
  }
  if (asm_wanted) {
    // AITER loads its .co lazily on the first asm call, and HIP rejects a module load
    // during stream capture -- which AITER turns into std::abort() rather than an
    // error. Stay on CK Tile while capturing.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    asm_wanted = hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
                 capture_status == hipStreamCaptureStatusNone;
  }

  // A failed load never becomes available later, so skip the block outright rather
  // than re-entering it to throw and immediately catch. Per-thread, so a second
  // thread retries the dlopen once before memoizing its own copy.
  static thread_local bool handle_failed = false;
  if (asm_wanted && !handle_failed) {
    // Only this shim's own errors are catchable. AITER reaches std::abort() through
    // AITER_CHECK for a missing .co, a failed hipModuleLoad and most other internal
    // failures, because the thread_local that would make it throw defaults to false
    // and is per-.so under RTLD_LOCAL. That is why the loader pre-checks
    // AITER_ASM_DIR, why capture is excluded above, and why the kill switch exists:
    // there is no way to recover once AITER is inside one of those paths.
    try {
      auto asm_fn = reinterpret_cast<mha_fwd_fn>(flashinfer::aiter::get_aiter_mha_fwd_asm_handle());

      // Probe once per trait set. v3_api_check resolves AITER's config table and
      // returns without launching, so this costs one lookup per process rather than
      // a wasted dispatch per call. The slot covers needs_mask and the normalized
      // window: head dim is a template parameter, dtype is fixed by the eligibility
      // check above, and is_group_mode is false whenever there is no soft cap.
      static thread_local int probe[2] = {0, 0};  // 0 unknown, 1 supported, -1 not
      const int slot = needs_mask ? 1 : 0;
      if (probe[slot] == 0) {
        ::aiter::mha_fwd_args probe_args = args;
        probe_args.use_asm_v3 = true;
        probe_args.v3_api_check = true;
        // v3_api_check returns a literal 1 when the config table has the instance
        // and -1 when it does not; the real launch below returns the kernel time,
        // hence > 0 here and >= 0 there.
        probe[slot] = asm_fn(probe_args, sconfig) > 0.f ? 1 : -1;
        AiterAsmPrefillNote(probe[slot] == 1 ? "probe: supported" : "probe: unsupported");
      }

      if (probe[slot] == 1) {
        args.use_asm_v3 = true;
        if (asm_fn(args, sconfig) >= 0.f) {
          AiterAsmPrefillNote("launched");
          // The shim consumes hipGetLastError() on its own failure path, so a clean
          // return means it saw none; read it anyway so an async error is reported
          // here rather than charged to whichever op runs next.
          return hipGetLastError();
        }
        // The real call disagreed with the probe, so stop trusting it: otherwise
        // every later call pays a wasted dispatch plus AITER's warning.
        probe[slot] = -1;
        args.use_asm_v3 = false;
        AiterAsmPrefillNote("declined after probe said supported; using CK Tile");
      }
    } catch (const std::exception&) {
      handle_failed = true;
      // asm_fn may have thrown after this was set. Inert today (the CK variant is
      // built without FAV3_ON and ignores it), but not if a variant ever ships both.
      args.use_asm_v3 = false;
      AiterAsmPrefillNote("unavailable; using CK Tile");
    }
  }

  // A negative return means no kernel instance matched and nothing was launched,
  // which would otherwise leave the caller's output buffer untouched and unflagged.
  // Drop anything the abandoned asm attempt left pending, or the CK Tile call below
  // is reported as having failed with an error it did not cause.
  if (asm_wanted) static_cast<void>(hipGetLastError());
  if (fn(args, sconfig) < 0.f) return hipErrorNoBinaryForGpu;
  return hipGetLastError();
}

}  // namespace flashinfer
