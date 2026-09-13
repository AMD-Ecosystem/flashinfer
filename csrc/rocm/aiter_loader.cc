// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// No AITER or CK Tile headers needed here — we work with void* function pointers.

#include <dlfcn.h>
#include <flashinfer/rocm/attention/aiter/aiter_loader.h>

#include <cstdlib>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace flashinfer::aiter {

namespace {

// Appended to failures on the paths that resolve the pinned mangled symbols
// below. Keep the version in step with _AITER_LAST_VALIDATED in prefill_rocm.py.
constexpr const char* kAbiPinNote =
    "\n  These symbols and .so names are pinned to amd-aiter 0.1.20, which has no stable"
    "\n  C++ ABI. If AITER was upgraded, that is the likely cause. Reinstall the pin:"
    "\n    pip install amd-aiter==0.1.20+rocm10.1.0a20260819.3135022 \\"
    "\n      --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/"
    "\n  or re-pin the symbols in csrc/rocm/aiter_loader.cc.";

// Directories to try, in order, when resolving a variant .so.
//
// FLASHINFER_AITER_VARIANT_DIR is FlashInfer's prebuilt store, exported from
// Python at import time rather than baked in: an AOT-packaged module carries
// whatever -D the *build* machine had, which names a path that does not exist
// on the consumer's. It is second so an operator-set AITER_JIT_DIR still wins,
// which is what the failure message below has always told them to set.
std::vector<std::string> jit_dir_candidates() {
  // Empty is not unset: `export AITER_JIT_DIR=` is a common way to clear a
  // variable in an entrypoint, and taking it would dlopen "/<name>.so" and
  // bury the diagnostic below.
  auto env_dir = [](const char* name) -> const char* {
    const char* value = std::getenv(name);
    return (value && *value) ? value : nullptr;
  };

  std::vector<std::string> dirs;
  // An operator-set AITER_JIT_DIR *replaces* the baked default rather than
  // preceding it. It selects which AITER build to use, so falling through to
  // the pinned install would load a kernel from a different build than the one
  // they chose -- silently, since the mangled symbol still resolves.
  const char* aiter_dir = env_dir("AITER_JIT_DIR");
  if (aiter_dir) dirs.emplace_back(aiter_dir);
  // A ':'-separated list, not one path: several stores can be readable at once
  // (the cache the prebuild driver writes, plus one shipped in the jit-cache
  // wheel), and taking only the first lets a partial store shadow a complete
  // one. Python exports exactly the list it searched.
  if (const char* env = env_dir("FLASHINFER_AITER_VARIANT_DIR")) {
    std::string_view rest(env);
    while (!rest.empty()) {
      const auto sep = rest.find(':');
      const auto part = rest.substr(0, sep);
      if (!part.empty()) dirs.emplace_back(part);
      if (sep == std::string_view::npos) break;
      rest.remove_prefix(sep + 1);
    }
  }
  if (!aiter_dir) {
#ifdef FLASHINFER_AITER_JIT_DIR
    dirs.emplace_back(FLASHINFER_AITER_JIT_DIR);
#endif
  }
  if (dirs.empty()) {
    throw std::runtime_error(
        "AITER_JIT_DIR env var not set and FLASHINFER_AITER_JIT_DIR not compiled in. "
        "Set AITER_JIT_DIR=<path to aiter/jit/> or rebuild the FlashInfer JIT cache "
        "after installing AITER (rm -rf ~/.cache/flashinfer/).");
  }
  return dirs;
}

// Build a variant .so filename from a key.
// Segments: {prefix}{dtype}[_{logits}]_{bias}_{mask}_{lse}{infix}{qscale}{suffix}
// mha_varlen_fwd:    prefix="mha_varlen_fwd_",   include_logits=true,
//                    infix="_ndropout_nskip_", suffix=".so"
// mha_fwd:           prefix="mha_fwd_",          include_logits=false,
//                    infix="_ndropout_", suffix=".so"
// mha_batch_prefill: prefix="mha_batch_prefill_", include_logits=true,
//                    infix="_ndropout_", suffix="_nsink.so"
// The qscale token sits mid-name, so it is a builder segment rather than part
// of a fixed suffix: fp8 spells it "pertensor" where bf16/fp16 spell "nqscale".
const char* dtype_token(VariantKey const& key) {
  switch (key.dtype) {
    case VariantKey::Dtype::kFp16:
      return "fp16";
    case VariantKey::Dtype::kFp8Bf16:
      return "fp8bf16";
    default:
      return "bf16";
  }
}

std::string build_so_name(VariantKey const& key, std::string_view prefix, std::string_view infix,
                          std::string_view suffix, bool include_logits) {
  std::string name(prefix);
  name += dtype_token(key);
  if (include_logits) {
    name += key.has_logits_cap ? "_logits" : "_nlogits";
  }
  name += key.has_alibi ? "_alibi" : "_nbias";
  name += key.needs_mask ? "_mask" : "_nmask";
  name += key.has_lse ? "_lse" : "_nlse";
  name += infix;
  name += key.has_qscale ? "pertensor" : "nqscale";
  name += suffix;
  return name;
}

// dlopen `so_path` and resolve `sym_name`. Returns nullptr when the file could
// not be opened, appending the reason to `tried` so a multi-directory search can
// report every candidate. A file that opens but lacks the symbol throws instead:
// that is an ABI mismatch, and trying the next candidate would only mask it.
void* dlopen_and_sym(const std::string& so_path, const char* sym_name, std::string& tried,
                     const std::function<std::string()>& hint_fn) {
  void* handle = dlopen(so_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!handle) {
    const char* err = dlerror();
    tried += "\n    " + so_path + ": " + (err ? err : "unknown");
    return nullptr;
  }

  dlerror();  // clear any pre-existing error before dlsym
  void* sym = dlsym(handle, sym_name);
  if (!sym) {
    const char* err = dlerror();
    dlclose(handle);
    throw std::runtime_error("dlsym(" + std::string(sym_name) + ") failed in " + so_path + ": " +
                             (err ? err : "unknown") + "\n" + hint_fn());
  }
  return sym;
}

// Double-checked locking cache around `resolve`, which is called at most once
// per key and returns the symbol or throws.
template <typename Key, typename Hash, typename Resolve>
void* cache_sym(std::shared_mutex& mu, std::unordered_map<Key, void*, Hash>& cache, const Key& key,
                Resolve resolve) {
  {
    std::shared_lock rd(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }
  std::unique_lock wr(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  void* sym = resolve();
  cache.emplace(key, sym);
  return sym;
}

// Resolve `so_name` against each candidate directory in turn.
template <typename Key, typename Hash>
void* load_variant_sym(std::shared_mutex& mu, std::unordered_map<Key, void*, Hash>& cache,
                       const Key& key, const std::string& so_name, const char* sym_name,
                       std::function<std::string()> hint_fn) {
  return cache_sym(mu, cache, key, [&]() {
    std::string tried;
    for (const std::string& dir : jit_dir_candidates()) {
      if (void* sym = dlopen_and_sym(dir + "/" + so_name, sym_name, tried, hint_fn)) return sym;
    }
    throw std::runtime_error("AITER .so not found: " + so_name + "\n  tried:" + tried + "\n" +
                             hint_fn());
  });
}

// Resolve an absolute path handed down from Python (the PA v1 decode path).
template <typename Key, typename Hash>
void* load_path_sym(std::shared_mutex& mu, std::unordered_map<Key, void*, Hash>& cache,
                    const Key& key, const std::string& so_path, const char* sym_name,
                    std::function<std::string()> hint_fn) {
  return cache_sym(mu, cache, key, [&]() {
    std::string tried;
    if (void* sym = dlopen_and_sym(so_path, sym_name, tried, hint_fn)) return sym;
    throw std::runtime_error("AITER .so not found: " + so_path + "\n  tried:" + tried + "\n" +
                             hint_fn());
  });
}

// Mangled symbol for aiter::mha_fwd(aiter::mha_fwd_args, ck_tile::stream_config const&).
// Stable across GCC/Clang Itanium ABI; verified by `nm -D` on all shipped variants.
// Both the mha_fwd and mha_varlen_fwd .so files export this same dispatcher symbol.
// Pinned to amd-aiter 0.1.20. Regenerate with: nm -D <variant.so> | grep mha_fwd
constexpr const char* kMhaFwdSymbol =
    "_ZN5aiter7mha_fwdENS_12mha_fwd_argsERKN7ck_tile13stream_configE";

// ----- mha_fwd (non-varlen, batch-mode CK) cache -----

std::string mha_fwd_variant_so_name(VariantKey const& key) {
  return build_so_name(key, "mha_fwd_", "_ndropout_", ".so", /*include_logits=*/false);
}

std::shared_mutex s_mf_mu;
std::unordered_map<VariantKey, void*, VariantKeyHash> s_mf_cache;

// ----- mha_varlen_fwd (varlen, group-mode CK) cache -----

std::string mha_varlen_fwd_variant_so_name(VariantKey const& key) {
  return build_so_name(key, "mha_varlen_fwd_", "_ndropout_nskip_", ".so",
                       /*include_logits=*/true);
}

std::shared_mutex s_vl_mu;
std::unordered_map<VariantKey, void*, VariantKeyHash> s_vl_cache;

// ----- batch-prefill cache -----

std::string batch_prefill_variant_so_name(BatchPrefillVariantKey const& key) {
  // 0.1.20 appended a sink axis to this family only -- mha_fwd and
  // mha_varlen_fwd keep their names. Without it every lookup misses and the
  // paged path degrades to flat-gather instead of using native paging.
  return build_so_name(key, "mha_batch_prefill_", "_ndropout_", "_nsink.so",
                       /*include_logits=*/true);
}

// Itanium-ABI mangled symbol for aiter::mha_batch_prefill(...).
// Pinned to amd-aiter 0.1.20. Regenerate with: nm -D <.so> | grep mha_batch_prefill
// Note: '23' encodes len("fmha_batch_prefill_args") == 23.
constexpr const char* kMhaBatchPrefillSymbol =
    "_ZN5aiter17mha_batch_prefillE23fmha_batch_prefill_argsRKN7ck_tile13stream_configE"
    "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb9mask_enum9bias_enumb"
    "16quant_scale_enumb";

std::shared_mutex s_bp_mu;
std::unordered_map<BatchPrefillVariantKey, void*, BatchPrefillVariantKeyHash> s_bp_cache;

// ----- generic extern "C" JIT cache -----

struct ExternCKey {
  std::string so_path;
  std::string func_name;
  bool operator==(ExternCKey const& o) const noexcept {
    return so_path == o.so_path && func_name == o.func_name;
  }
};

struct ExternCKeyHash {
  std::size_t operator()(ExternCKey const& k) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(k.so_path);
    std::size_t h2 = std::hash<std::string>{}(k.func_name);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

std::shared_mutex s_ec_mu;
std::unordered_map<ExternCKey, void*, ExternCKeyHash> s_ec_cache;

}  // namespace

void* get_aiter_mha_fwd_handle(VariantKey const& key) {
  if (key.has_logits_cap) {
    throw std::runtime_error(
        "get_aiter_mha_fwd_handle called with has_logits_cap=true; the mha_fwd "
        "template has no _logits arm and would silently ignore logits_soft_cap. "
        "Use get_aiter_mha_varlen_fwd_handle for this trait.");
  }
  const std::string so_name = mha_fwd_variant_so_name(key);
  return load_variant_sym(s_mf_mu, s_mf_cache, key, so_name, kMhaFwdSymbol, [&key]() {
    return "  Hint: trigger AITER's lazy JIT build by importing aiter.ops.mha and "
           "calling mha_fwd with matching (q dtype: " +
           std::string(dtype_token(key)) + ", is_causal=" + (key.needs_mask ? "true" : "false") +
           " (window_size_left>=0 selects the same variant)" +
           ", return_softmax_lse=" + (key.has_lse ? "true" : "false") + ")." + kAbiPinNote;
  });
}

void* get_aiter_mha_varlen_fwd_handle(VariantKey const& key) {
  const std::string so_name = mha_varlen_fwd_variant_so_name(key);
  return load_variant_sym(s_vl_mu, s_vl_cache, key, so_name, kMhaFwdSymbol, [&key]() {
    return "  Hint: trigger AITER's lazy JIT build by importing aiter.ops.mha and "
           "calling mha_varlen_fwd with matching (q dtype: " +
           std::string(dtype_token(key)) + ", is_causal=" + (key.needs_mask ? "true" : "false") +
           " (window_size_left>=0 selects the same variant)" +
           ", return_softmax_lse=" + (key.has_lse ? "true" : "false") + ")." + kAbiPinNote;
  });
}

void* get_aiter_mha_batch_prefill_handle(BatchPrefillVariantKey const& key) {
  const std::string so_name = batch_prefill_variant_so_name(key);
  return load_variant_sym(
      s_bp_mu, s_bp_cache, key, so_name, kMhaBatchPrefillSymbol, [&key, &so_name]() {
        return "  Hint: trigger AITER's lazy JIT build by calling "
               "aiter.ops.mha.mha_batch_prefill_func() once with matching (q dtype: " +
               std::string(dtype_token(key)) +
               // mha_batch_prefill_func takes `causal`; mha_fwd/mha_varlen_fwd
               // take `is_causal`.
               ", causal=" + (key.needs_mask ? "true" : "false") +
               " (a window selects the same variant)" +
               ", return_lse=" + (key.has_lse ? "true" : "false") +
               ") before this C++ path.\n"
               "  If the .so exists but dlsym fails, run: nm -D <dir>/" +
               so_name + " | grep mha_batch_prefill" + kAbiPinNote;
      });
}

void* get_aiter_extern_c_handle(const std::string& so_path, const std::string& func_name) {
  ExternCKey key{so_path, func_name};
  return load_path_sym(s_ec_mu, s_ec_cache, key, so_path, func_name.c_str(), [&so_path]() {
    return "  Hint: ensure the AITER compile() helper was invoked from the Python "
           "plan() side to bootstrap this variant.";
  });
}

}  // namespace flashinfer::aiter
