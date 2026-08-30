// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace flashinfer::aiter {

struct VariantKey {
  enum class Dtype : uint8_t { kFp16, kBf16 };
  Dtype dtype;
  bool causal;
  bool has_lse;
  bool has_alibi;
  bool has_logits_cap;

  bool operator==(VariantKey const& o) const noexcept {
    return dtype == o.dtype && causal == o.causal && has_lse == o.has_lse &&
           has_alibi == o.has_alibi && has_logits_cap == o.has_logits_cap;
  }
};

struct VariantKeyHash {
  std::size_t operator()(VariantKey const& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.dtype);
    h = h * 31 + k.causal;
    h = h * 31 + k.has_lse;
    h = h * 31 + k.has_alibi;
    h = h * 31 + k.has_logits_cap;
    return h;
  }
};

// Returns the raw dlsym function pointer for aiter::mha_fwd(mha_fwd_args, stream_config const&)
// out of AITER's mha_fwd template .so. The CK Tile kernel instances baked in are batch-mode
// (matching is_group_mode=false); the ASM v3 path is still reachable via use_asm_v3=true.
// This is the preferred loader for single-sequence prefill on shapes that don't need
// logits_soft_cap or min_seqlen_q semantics — it lets us skip group-mode seqstart
// [0, seqlen] plumbing entirely.
//
// has_logits_cap is ignored when building the .so name (the mha_fwd template has no
// _logits/_nlogits arm); callers must verify has_logits_cap==false before invoking,
// otherwise the resulting kernel ignores logits_soft_cap.
void* get_aiter_mha_fwd_handle(VariantKey const& key);

// Returns the raw dlsym function pointer for aiter::mha_fwd out of AITER's mha_varlen_fwd
// template .so. The CK Tile kernel instances baked in are group-mode only. Use this for
// genuinely-varlen call sites, and for single-sequence calls that need logits_soft_cap
// (the non-varlen mha_fwd template does not support that trait).
void* get_aiter_mha_varlen_fwd_handle(VariantKey const& key);

// Batch-prefill variants share the same key fields as mha_fwd variants.
// page_size is NOT in the key — the .so dispatches all native page sizes at runtime.
using BatchPrefillVariantKey = VariantKey;
using BatchPrefillVariantKeyHash = VariantKeyHash;

// Returns the raw dlsym function pointer for aiter::mha_batch_prefill(...).
// The JIT directory is resolved from FLASHINFER_AITER_JIT_DIR / AITER_JIT_DIR env vars at
// runtime; no jit_dir argument is needed. The .so is built lazily by AITER — bootstrap
// by calling aiter.ops.mha.mha_batch_prefill_func once before invoking this function.
// Throws std::runtime_error if the variant .so is not found or the symbol is missing.
void* get_aiter_mha_batch_prefill_handle(BatchPrefillVariantKey const& key);

// Generic helper for AITER's per-variant `extern "C"` JIT outputs (e.g. PA v1, MLA).
// AITER's compile_template_op produces lib.so files keyed by an md5 hash of all
// template params; the Python plan() side calls the AITER compile() helper and
// passes the resolved (so_path, func_name) to the C++ run() entry, which dlopens
// the .so once and caches the dlsym handle. Throws if dlopen or dlsym fails.
void* get_aiter_extern_c_handle(const std::string& so_path, const std::string& func_name);

}  // namespace flashinfer::aiter
