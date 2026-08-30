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

std::string get_jit_dir() {
  if (const char* env = std::getenv("AITER_JIT_DIR")) return env;
#ifdef FLASHINFER_AITER_JIT_DIR
  return FLASHINFER_AITER_JIT_DIR;
#else
  throw std::runtime_error(
      "AITER_JIT_DIR env var not set and FLASHINFER_AITER_JIT_DIR not compiled in. "
      "Set AITER_JIT_DIR=<path to aiter/jit/> or rebuild the FlashInfer JIT cache "
      "after installing AITER (rm -rf ~/.cache/flashinfer/).");
#endif
}

// Build a variant .so filename from a key.
// Segments: {prefix}{dtype}[_{logits}]_{bias}_{mask}_{lse}{suffix}
// mha_varlen_fwd:    prefix="mha_varlen_fwd_",   include_logits=true,
//                    suffix="_ndropout_nskip_nqscale.so"
// mha_fwd:           prefix="mha_fwd_",          include_logits=false,
//                    suffix="_ndropout_nqscale.so"
// mha_batch_prefill: prefix="mha_batch_prefill_", include_logits=true,
//                    suffix="_ndropout_nqscale.so"
std::string build_so_name(VariantKey const& key, std::string_view prefix, std::string_view suffix,
                          bool include_logits) {
  std::string name(prefix);
  name += (key.dtype == VariantKey::Dtype::kFp16) ? "fp16" : "bf16";
  if (include_logits) {
    name += key.has_logits_cap ? "_logits" : "_nlogits";
  }
  name += key.has_alibi ? "_alibi" : "_nbias";
  name += key.causal ? "_mask" : "_nmask";
  name += key.has_lse ? "_lse" : "_nlse";
  name += suffix;
  return name;
}

// Double-checked locking dlopen/dlsym with a shared cache.
// Loads `so_path`, resolves `sym_name`, stores in `cache` under `key`.
// `hint_fn` is called lazily to build the error hint on dlopen failure.
template <typename Key, typename Hash>
void* load_and_cache_sym(std::shared_mutex& mu, std::unordered_map<Key, void*, Hash>& cache,
                         const Key& key, const std::string& so_path, const char* sym_name,
                         std::function<std::string()> hint_fn) {
  {
    std::shared_lock rd(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }
  std::unique_lock wr(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  void* handle = dlopen(so_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!handle) {
    const char* err = dlerror();
    throw std::runtime_error("AITER .so not found: " + so_path +
                             "\n  dlerror: " + (err ? err : "unknown") + "\n" + hint_fn());
  }

  dlerror();  // clear any pre-existing error before dlsym
  void* sym = dlsym(handle, sym_name);
  if (!sym) {
    const char* err = dlerror();
    dlclose(handle);
    throw std::runtime_error("dlsym(" + std::string(sym_name) + ") failed in " + so_path + ": " +
                             (err ? err : "unknown") + "\n" + hint_fn());
  }

  cache.emplace(key, sym);
  return sym;
}

// Mangled symbol for aiter::mha_fwd(aiter::mha_fwd_args, ck_tile::stream_config const&).
// Stable across GCC/Clang Itanium ABI; verified by `nm -D` on all shipped variants.
// Both the mha_fwd and mha_varlen_fwd .so files export this same dispatcher symbol.
// Pinned to amd-aiter 0.1.10. Regenerate with: nm -D <variant.so> | grep mha_fwd
constexpr const char* kMhaFwdSymbol =
    "_ZN5aiter7mha_fwdENS_12mha_fwd_argsERKN7ck_tile13stream_configE";

// ----- mha_fwd (non-varlen, batch-mode CK) cache -----

std::string mha_fwd_variant_so_name(VariantKey const& key) {
  return build_so_name(key, "mha_fwd_", "_ndropout_nqscale.so", /*include_logits=*/false);
}

std::shared_mutex s_mf_mu;
std::unordered_map<VariantKey, void*, VariantKeyHash> s_mf_cache;

// ----- mha_varlen_fwd (varlen, group-mode CK) cache -----

std::string mha_varlen_fwd_variant_so_name(VariantKey const& key) {
  return build_so_name(key, "mha_varlen_fwd_", "_ndropout_nskip_nqscale.so",
                       /*include_logits=*/true);
}

std::shared_mutex s_vl_mu;
std::unordered_map<VariantKey, void*, VariantKeyHash> s_vl_cache;

// ----- batch-prefill cache -----

std::string batch_prefill_variant_so_name(BatchPrefillVariantKey const& key) {
  return build_so_name(key, "mha_batch_prefill_", "_ndropout_nqscale.so",
                       /*include_logits=*/true);
}

// Itanium-ABI mangled symbol for aiter::mha_batch_prefill(...).
// Pinned to amd-aiter 0.1.10. Regenerate with: nm -D <.so> | grep mha_batch_prefill
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
  std::string so_path = get_jit_dir() + "/" + mha_fwd_variant_so_name(key);
  return load_and_cache_sym(s_mf_mu, s_mf_cache, key, so_path, kMhaFwdSymbol, [&key, &so_path]() {
    return "  Hint: trigger AITER's lazy JIT build by importing aiter.ops.mha and "
           "calling mha_fwd with matching (dtype=" +
           std::string(key.dtype == VariantKey::Dtype::kFp16 ? "fp16" : "bf16") +
           ", causal=" + (key.causal ? "true" : "false") +
           ", has_lse=" + (key.has_lse ? "true" : "false") + ")." + kAbiPinNote;
  });
}

void* get_aiter_mha_varlen_fwd_handle(VariantKey const& key) {
  std::string so_path = get_jit_dir() + "/" + mha_varlen_fwd_variant_so_name(key);
  return load_and_cache_sym(s_vl_mu, s_vl_cache, key, so_path, kMhaFwdSymbol, [&key, &so_path]() {
    return "  Hint: trigger AITER's lazy JIT build by importing aiter.ops.mha and "
           "calling mha_varlen_fwd with matching (dtype=" +
           std::string(key.dtype == VariantKey::Dtype::kFp16 ? "fp16" : "bf16") +
           ", causal=" + (key.causal ? "true" : "false") +
           ", has_lse=" + (key.has_lse ? "true" : "false") + ")." + kAbiPinNote;
  });
}

void* get_aiter_mha_batch_prefill_handle(BatchPrefillVariantKey const& key) {
  std::string so_path = get_jit_dir() + "/" + batch_prefill_variant_so_name(key);
  return load_and_cache_sym(
      s_bp_mu, s_bp_cache, key, so_path, kMhaBatchPrefillSymbol, [&key, &so_path]() {
        return "  Hint: trigger AITER's lazy JIT build by calling "
               "aiter.ops.mha.mha_batch_prefill_func() once with matching (dtype=" +
               std::string(key.dtype == VariantKey::Dtype::kFp16 ? "fp16" : "bf16") +
               ", causal=" + (key.causal ? "true" : "false") +
               ", has_lse=" + (key.has_lse ? "true" : "false") +
               ") before this C++ path.\n"
               "  If the .so exists but dlsym fails, run: nm -D " +
               so_path + " | grep mha_batch_prefill" + kAbiPinNote;
      });
}

void* get_aiter_extern_c_handle(const std::string& so_path, const std::string& func_name) {
  ExternCKey key{so_path, func_name};
  return load_and_cache_sym(s_ec_mu, s_ec_cache, key, so_path, func_name.c_str(), [&so_path]() {
    return "  Hint: ensure the AITER compile() helper was invoked from the Python "
           "plan() side to bootstrap this variant.";
  });
}

}  // namespace flashinfer::aiter
