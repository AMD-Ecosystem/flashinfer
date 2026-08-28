// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/aiter/single_prefill.cuh>
#include <gpu_iface/enums.hpp>
#include <gpu_iface/layout.cuh>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "pytorch_extension_utils.h"
#include "single_prefill_config.inc"

using flashinfer::MaskMode;
using flashinfer::QKVLayout;

// Cache of device seqstart tensors keyed by (qo_len, kv_len, device_index).
// The mha_fwd .so handles batch=1 in batch mode (no seqstart needed); only the
// logits_soft_cap path is forced onto the group-mode mha_varlen_fwd .so, which
// requires seqstart [0, seqlen] arrays on device. Caching eliminates per-call
// allocation + H2D copy (~15 µs fixed overhead).
namespace {
struct SeqstartKey {
  uint32_t qo_len, kv_len;
  int device_idx;
  bool operator==(SeqstartKey const& o) const noexcept {
    return qo_len == o.qo_len && kv_len == o.kv_len && device_idx == o.device_idx;
  }
};
struct SeqstartKeyHash {
  size_t operator()(SeqstartKey const& k) const noexcept {
    size_t h = static_cast<size_t>(k.qo_len);
    h = h * 2654435761u ^ static_cast<size_t>(k.kv_len);
    h = h * 2654435761u ^ static_cast<size_t>(k.device_idx);
    return h;
  }
};
std::mutex s_seqstart_mu;
std::unordered_map<SeqstartKey, std::pair<at::Tensor, at::Tensor>, SeqstartKeyHash>
    s_seqstart_cache;

// Return device pointers to [0, qo_len] and [0, kv_len] int32 tensors.
// First call for a given (qo_len, kv_len, device) allocates; subsequent calls are mutex+lookup
// only.
std::pair<const int32_t*, const int32_t*> get_seqstart_ptrs(uint32_t qo_len, uint32_t kv_len,
                                                            at::Device device) {
  SeqstartKey key{qo_len, kv_len, device.index()};
  std::lock_guard<std::mutex> lock(s_seqstart_mu);
  auto [it, inserted] = s_seqstart_cache.try_emplace(key);
  if (inserted) {
    auto opts = at::TensorOptions().dtype(at::kInt).device(device);
    it->second.first = at::tensor({0, static_cast<int32_t>(qo_len)}, opts);
    it->second.second = at::tensor({0, static_cast<int32_t>(kv_len)}, opts);
  }
  return {static_cast<const int32_t*>(it->second.first.data_ptr()),
          static_cast<const int32_t*>(it->second.second.data_ptr())};
}
}  // namespace

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  int64_t mask_mode_code, int64_t layout,
                                  int64_t window_left ADDITIONAL_FUNC_PARAMS) {
  const auto device = q.device();
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);

  const QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  TORCH_CHECK(kv_layout == QKVLayout::kNHD,
              "AITER backend only supports NHD kv_layout; got layout=", layout);

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  TORCH_CHECK(mask_mode != MaskMode::kCustom, "AITER backend does not support custom mask");

  const auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "AITER backend supports fp16/bf16 only; got dtype=", q_dtype);
  TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype, "q, k, v must share dtype");
  TORCH_CHECK(o.scalar_type() == q_dtype,
              "AITER backend requires output dtype to match input dtype; got o=", o.scalar_type(),
              " q=", q_dtype);
  TORCH_CHECK(o.is_contiguous(), "AITER backend requires a contiguous output tensor");

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const bool causal = (mask_mode == MaskMode::kCausal);

  const char* dtype_str = (q_dtype == at::kHalf) ? "fp16" : "bf16";
  const auto dtype_enum = (q_dtype == at::kHalf) ? flashinfer::aiter::VariantKey::Dtype::kFp16
                                                 : flashinfer::aiter::VariantKey::Dtype::kBf16;

  Params params;
  params.q = static_cast<DTypeQ*>(q.data_ptr());
  params.k = static_cast<DTypeKV*>(k.data_ptr());
  params.v = static_cast<DTypeKV*>(v.data_ptr());
  params.o = static_cast<DTypeO*>(o.data_ptr());

  // AITER LSE layout: [num_qo_heads, qo_len] in nats; FlashInfer expects [qo_len, num_qo_heads] in
  // log2.
  at::Tensor aiter_lse_scratch;
  if (maybe_lse) {
    aiter_lse_scratch = at::empty({q.size(1), q.size(0)}, maybe_lse->options().dtype(at::kFloat));
    params.lse = static_cast<float*>(aiter_lse_scratch.data_ptr());
  } else {
    params.lse = nullptr;
  }

  params.num_qo_heads = static_cast<uint32_t>(q.size(1));
  params.num_kv_heads = static_cast<uint32_t>(k.size(1));
  params.qo_len = static_cast<uint32_t>(q.size(0));
  params.kv_len = static_cast<uint32_t>(k.size(0));
  params.q_stride_n = static_cast<uint32_t>(q.stride(0));
  params.q_stride_h = static_cast<uint32_t>(q.stride(1));
  params.k_stride_n = static_cast<uint32_t>(k.stride(0));
  params.k_stride_h = static_cast<uint32_t>(k.stride(1));
  params.v_stride_n = static_cast<uint32_t>(v.stride(0));
  params.v_stride_h = static_cast<uint32_t>(v.stride(1));
  params.window_left = static_cast<int32_t>(window_left);
  ADDITIONAL_PARAMS_SETTER

  // The mha_fwd .so dispatches batch=1 directly via stride/nhead_stride; only the
  // mha_varlen_fwd .so (used when logits_soft_cap > 0) needs the per-(qo_len, kv_len)
  // seqstart [0, len] device tensors.
  const int32_t* cu_seqlens_q_ptr = nullptr;
  const int32_t* cu_seqlens_k_ptr = nullptr;
  if (params.logits_soft_cap > 0.0f) {
    std::tie(cu_seqlens_q_ptr, cu_seqlens_k_ptr) =
        get_seqstart_ptrs(params.qo_len, params.kv_len, device);
  }

  hipError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO>(
      params, causal, dtype_str, dtype_enum, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
      static_cast<DTypeO*>(tmp.data_ptr()), stream);
  TORCH_CHECK(status == hipSuccess,
              "AITER SinglePrefill kernel launch failed: ", hipGetErrorString(status));

  if (maybe_lse) {
    aiter_lse_scratch.div_(std::log(2.0));
    maybe_lse->copy_(aiter_lse_scratch.t());
  }
}
