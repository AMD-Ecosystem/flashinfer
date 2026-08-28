// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER batch-ragged-prefill via the C++ harness.
// Ragged KV is already in flat [total_kv, nhead_k, head_dim] layout, so we
// can call aiter::mha_fwd group-mode directly through the same dlopen/dlsym
// cache as single-prefill — no per-call gather step (unlike the paged path).

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/aiter/batch_prefill.cuh>
#include <gpu_iface/enums.hpp>
#include <gpu_iface/layout.cuh>
#include <optional>

#include "batch_prefill_aiter_config.inc"
#include "pytorch_extension_utils.h"

using flashinfer::MaskMode;
using flashinfer::QKVLayout;

// ragged_run signature:
//   q              [total_q,  nhead_q, HEAD_DIM_QK]
//   k              [total_kv, nhead_k, HEAD_DIM_QK]  (ragged, NHD)
//   v              [total_kv, nhead_k, HEAD_DIM_VO]  (ragged, NHD)
//   qo_indptr      [batch+1] int32 cumulative Q lengths
//   kv_indptr      [batch+1] int32 cumulative KV lengths
//   o              [total_q,  nhead_q, HEAD_DIM_VO] (contiguous)
//   maybe_lse      [total_q,  nhead_q] float32 (FlashInfer log2, after conversion)
void batch_ragged_prefill_with_kv_cache_aiter(at::Tensor q, at::Tensor k, at::Tensor v,
                                              at::Tensor qo_indptr, at::Tensor kv_indptr,
                                              at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                              int64_t mask_mode_code, int64_t layout,
                                              int64_t window_left, double logits_soft_cap,
                                              double sm_scale, int64_t max_q_len,
                                              [[maybe_unused]] int64_t max_kv_len) {
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
  TORCH_CHECK(o.is_contiguous(), "AITER backend requires a contiguous output tensor");
  TORCH_CHECK(o.scalar_type() == q_dtype,
              "AITER backend requires output dtype to match input dtype; got o=", o.scalar_type(),
              " q=", q_dtype);
  TORCH_CHECK(static_cast<int>(HEAD_DIM_QK) == static_cast<int>(HEAD_DIM_VO),
              "AITER backend requires equal head dims; got HEAD_DIM_QK=", HEAD_DIM_QK,
              " HEAD_DIM_VO=", HEAD_DIM_VO);

  TORCH_CHECK(qo_indptr.scalar_type() == at::kInt && qo_indptr.dim() == 1);
  TORCH_CHECK(kv_indptr.scalar_type() == at::kInt && kv_indptr.dim() == 1);
  TORCH_CHECK(qo_indptr.size(0) == kv_indptr.size(0),
              "qo_indptr and kv_indptr must have the same batch+1 length");

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const bool causal = (mask_mode == MaskMode::kCausal);

  const char* dtype_str = (q_dtype == at::kHalf) ? "fp16" : "bf16";
  const auto dtype_enum = (q_dtype == at::kHalf) ? flashinfer::aiter::VariantKey::Dtype::kFp16
                                                 : flashinfer::aiter::VariantKey::Dtype::kBf16;

  const int32_t batch = static_cast<int32_t>(qo_indptr.size(0) - 1);
  const int32_t total_qo = static_cast<int32_t>(q.size(0));
  const int32_t total_kv = static_cast<int32_t>(k.size(0));
  const int32_t num_qo_heads = static_cast<int32_t>(q.size(1));
  const int32_t num_kv_heads = static_cast<int32_t>(k.size(1));

  // AITER LSE layout: [num_qo_heads, total_qo] in nats.
  // FlashInfer expects [total_qo, num_qo_heads] in log2 — convert below.
  at::Tensor aiter_lse_scratch;
  if (maybe_lse) {
    aiter_lse_scratch = at::empty({num_qo_heads, total_qo}, maybe_lse->options().dtype(at::kFloat));
  }
  float* lse_ptr = maybe_lse ? static_cast<float*>(aiter_lse_scratch.data_ptr()) : nullptr;

  hipError_t status = flashinfer::BatchPrefillFlatGatherDispatched<HEAD_DIM_QK, HEAD_DIM_VO, DTypeQ,
                                                                   DTypeKV, DTypeO>(
      static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(k.data_ptr()),
      static_cast<DTypeKV*>(v.data_ptr()), static_cast<DTypeO*>(o.data_ptr()), lse_ptr,
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<const int32_t*>(kv_indptr.data_ptr()), batch, total_qo, total_kv,
      static_cast<int32_t>(max_q_len), num_qo_heads, num_kv_heads,
      static_cast<int32_t>(q.stride(0)),  // q_stride_n
      static_cast<int32_t>(q.stride(1)),  // q_stride_h
      static_cast<int32_t>(k.stride(0)),  // k_stride_n
      static_cast<int32_t>(k.stride(1)),  // k_stride_h
      static_cast<int32_t>(v.stride(0)),  // v_stride_n
      static_cast<int32_t>(v.stride(1)),  // v_stride_h
      static_cast<float>(sm_scale), static_cast<float>(logits_soft_cap),
      static_cast<int32_t>(window_left), causal, dtype_str, dtype_enum, stream);

  TORCH_CHECK(status == hipSuccess,
              "AITER batch-ragged-prefill kernel failed: ", hipGetErrorString(status));

  if (maybe_lse) {
    aiter_lse_scratch.div_(std::log(2.0));
    maybe_lse->copy_(aiter_lse_scratch.t());
  }
}
