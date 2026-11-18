// SPDX-FileCopyrightText: 2025 FlashInfer team.
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// Plan / run of the persistent BatchAttention kernel. ROCm port of upstream's
// csrc/batch_attention.cu (TVM-FFI -> at::Tensor). The argument lists match what
// upstream's flashinfer/attention/_core.py passes, so its BatchAttention class is
// used unchanged.
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <algorithm>
#include <cstdint>
#include <flashinfer/rocm/attention/scheduler.cuh>
#include <optional>

#include "batch_attention_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace {

constexpr int64_t kMaskModeNonCausal = 0, kMaskModeCausal = 1;  // flashinfer.utils.MaskMode
constexpr int64_t kLayoutNHD = 0, kLayoutHND = 1;               // flashinfer.utils.TensorLayout
constexpr float kLog2e = 1.44269504088896340736f;

at::ScalarType expected_scalar_type() {
  if constexpr (std::is_same_v<DTypeQ, __half>) {
    return at::kHalf;
  } else {
    return at::kBFloat16;
  }
}

// The kernel moves rows of q, k and v in 16-byte chunks.
void check_16b_rows(const at::Tensor& t, const char* name, std::initializer_list<int> dims) {
  constexpr int64_t kElems = 16 / sizeof(DTypeQ);
  TORCH_CHECK(t.stride(-1) == 1, name, " must be contiguous in its last dimension");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
  for (int d : dims) {
    TORCH_CHECK(t.stride(d) % kElems == 0, name, ".stride(", d, ") must be a multiple of ", kElems,
                " elements (16 bytes), got ", t.stride(d));
    TORCH_CHECK(t.stride(d) <= int64_t(UINT32_MAX), name, ".stride(", d, ") is too large");
  }
}

}  // namespace

at::Tensor BatchPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                   at::Tensor int_workspace_buffer,
                                   at::Tensor page_locked_int_workspace_buffer,
                                   at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len,
                                   int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
                                   int64_t head_dim_o, bool causal) {
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  TORCH_CHECK(head_dim_o == HEAD_DIM_VO, "BatchAttention: plan() head_dim_vo (", head_dim_o,
              ") does not match the compiled module (", HEAD_DIM_VO, ")");
  for (const at::Tensor* t : {&qo_indptr, &kv_indptr, &kv_len}) {
    TORCH_CHECK(t->device().is_cpu(), "BatchAttention: plan() indptr / kv_len must be on CPU");
    TORCH_CHECK(t->scalar_type() == at::kInt, "BatchAttention: indptr / kv_len must be int32");
    TORCH_CHECK(t->is_contiguous(), "BatchAttention: indptr / kv_len must be contiguous");
  }
  CHECK_GE(qo_indptr.numel(), batch_size + 1);
  CHECK_GE(kv_indptr.numel(), batch_size + 1);
  CHECK_GE(kv_len.numel(), batch_size);

  const size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  const size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(float_workspace_buffer.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  // Both mask instantiations launch on the same grid, so take the smaller residency.
  int num_ctas_nc = 0, num_ctas_c = 0;
  hipError_t status = BatchAttentionKernel<false>::NumCtas(&num_ctas_nc);
  if (status == hipSuccess) status = BatchAttentionKernel<true>::NumCtas(&num_ctas_c);
  TORCH_CHECK(status == hipSuccess,
              "BatchAttention: occupancy query failed: ", hipGetErrorString(status));
  const int num_ctas = std::min(num_ctas_nc, num_ctas_c);

  HolisticPlanInfo<2> plan_info;
  constexpr uint32_t CTA_TILE_Q_0 = BatchAttentionKernel<false>::CTA_TILE_Q_0;
  constexpr uint32_t CTA_TILE_Q_1 = BatchAttentionKernel<false>::CTA_TILE_Q_1;
  status = TwoStageHolisticPlan<CTA_TILE_Q_0, CTA_TILE_Q_1, IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
      kv_indptr.data_ptr<IdType>(), kv_len.data_ptr<IdType>(), batch_size, num_qo_heads,
      num_kv_heads, head_dim_o, causal, num_ctas, stream);
  TORCH_CHECK(status == hipSuccess, "BatchAttention: plan failed: ", hipGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor plan_info_vec, at::Tensor q, at::Tensor k_cache,
                            at::Tensor v_cache, at::Tensor kv_indices, at::Tensor o,
                            std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                            int64_t layout_code, int64_t num_qo_heads, int64_t num_kv_heads,
                            int64_t page_size, double v_scale, double sm_scale,
                            double logits_soft_cap, std::optional<at::Tensor> maybe_k_cache_sf,
                            std::optional<at::Tensor> maybe_v_cache_sf) {
  TORCH_CHECK(!maybe_k_cache_sf.has_value() && !maybe_v_cache_sf.has_value(),
              "BatchAttention: kv_cache_sf (NVFP4 KV cache) is not supported on ROCm");
  TORCH_CHECK(mask_mode_code == kMaskModeNonCausal || mask_mode_code == kMaskModeCausal,
              "BatchAttention: only causal and non-causal masks are supported on ROCm");
  TORCH_CHECK(layout_code == kLayoutNHD || layout_code == kLayoutHND,
              "BatchAttention: unknown kv layout code ", layout_code);
  if constexpr (USE_LOGITS_SOFT_CAP) {
    TORCH_CHECK(logits_soft_cap > 0.0,
                "BatchAttention: the module was planned with logits_soft_cap > 0; pass the same "
                "logits_soft_cap to run()");
  } else {
    TORCH_CHECK(logits_soft_cap <= 0.0,
                "BatchAttention: logits_soft_cap passed to run() but not to plan()");
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  TORCH_CHECK(page_size > 0, "BatchAttention: page_size must be positive");

  const at::ScalarType dtype = expected_scalar_type();
  for (const at::Tensor* tp : {&q, &k_cache, &v_cache, &o}) {
    const at::Tensor& t = *tp;
    CHECK_CUDA(t);
    TORCH_CHECK(t.scalar_type() == dtype, "BatchAttention: q, kv cache and out must be ", dtype,
                " for this module, got ", t.scalar_type());
  }
  CHECK_DIM(3, q);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(4, v_cache);
  TORCH_CHECK(q.size(1) == num_qo_heads && q.size(2) == HEAD_DIM_QK,
              "BatchAttention: q must be [num_tokens, ", num_qo_heads, ", ", HEAD_DIM_QK, "]");
  TORCH_CHECK(o.sizes() == q.sizes(), "BatchAttention: out must have the shape of q");
  TORCH_CHECK(o.stride(2) == 1 && o.stride(1) == HEAD_DIM_VO,
              "BatchAttention: out must be contiguous in its last two dimensions");
  TORCH_CHECK(o.stride(0) <= int64_t(UINT32_MAX), "BatchAttention: out.stride(0) is too large");
  check_16b_rows(q, "q", {0, 1});
  check_16b_rows(k_cache, "k_cache", {0, 1, 2});
  check_16b_rows(v_cache, "v_cache", {0, 1, 2});
  CHECK_CUDA(kv_indices);
  TORCH_CHECK(kv_indices.scalar_type() == at::kInt && kv_indices.is_contiguous(),
              "BatchAttention: kv_indices must be a contiguous int32 tensor");
  const int head_dim_axis = 3, head_axis = layout_code == kLayoutNHD ? 2 : 1,
            token_axis = layout_code == kLayoutNHD ? 1 : 2;
  TORCH_CHECK(k_cache.size(head_dim_axis) == HEAD_DIM_QK &&
                  k_cache.size(head_axis) == num_kv_heads && k_cache.size(token_axis) == page_size,
              "BatchAttention: k_cache shape does not match the layout, heads and page size");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "BatchAttention: v_cache must match k_cache");
  float* lse_ptr = nullptr;
  if (maybe_lse.has_value()) {
    const at::Tensor& lse = maybe_lse.value();
    CHECK_INPUT(lse);
    TORCH_CHECK(lse.scalar_type() == at::kFloat, "BatchAttention: lse must be float32");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == num_qo_heads,
                "BatchAttention: lse must be [num_tokens, num_qo_heads]");
    lse_ptr = lse.data_ptr<float>();
  }

  HolisticPlanInfo<2> plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  void* float_buffer = float_workspace_buffer.data_ptr();
  void* int_buffer = int_workspace_buffer.data_ptr();
  auto I = [&](int64_t offset) { return GetPtrFromBaseOffset<IdType>(int_buffer, offset); };

  using Params = typename BatchAttentionKernel<false>::Params;
  Params params[2];
  for (int i = 0; i < 2; ++i) {
    Params& p = params[i];
    const auto& task = plan_info.tasks[i];
    p.q = static_cast<const DTypeQ*>(q.data_ptr());
    p.k = static_cast<const DTypeKV*>(k_cache.data_ptr());
    p.v = static_cast<const DTypeKV*>(v_cache.data_ptr());
    p.kv_indices = kv_indices.data_ptr<IdType>();
    p.final_o = static_cast<DTypeO*>(o.data_ptr());
    p.final_lse = lse_ptr;
    p.partial_o = GetPtrFromBaseOffset<float>(float_buffer, plan_info.partial_o_offset);
    p.partial_lse = GetPtrFromBaseOffset<float>(float_buffer, plan_info.partial_lse_offset);
    p.merge_indptr = I(plan_info.merge_indptr_offset);
    p.merge_o_indices = I(plan_info.merge_o_indices_offset);
    p.num_packed_qo_len = I(plan_info.num_qo_len_offset);
    p.len_kv_chunk = I(plan_info.len_kv_chunk_offset);
    p.q_indptr = I(task.q_indptr_offset);
    p.kv_indptr = I(task.kv_indptr_offset);
    p.partial_indptr = I(task.partial_indptr_offset);
    p.q_len = I(task.q_len_offset);
    p.kv_len = I(task.kv_len_offset);
    p.q_start = I(task.q_start_offset);
    p.kv_start = I(task.kv_start_offset);
    p.kv_end = I(task.kv_end_offset);
    p.kv_head_idx = I(task.kv_head_idx_offset);
    p.work_indptr = I(task.work_indptr_offset);
    p.task = i;
    p.num_qo_heads = num_qo_heads;
    p.num_kv_heads = num_kv_heads;
    p.gqa_group_size = num_qo_heads / num_kv_heads;
    p.page_size = page_size;
    p.q_stride_n = q.stride(0);
    p.q_stride_h = q.stride(1);
    p.o_stride_n = o.stride(0);
    p.k_stride_page = k_cache.stride(0);
    p.k_stride_h = k_cache.stride(head_axis);
    p.k_stride_n = k_cache.stride(token_axis);
    p.v_stride_page = v_cache.stride(0);
    p.v_stride_h = v_cache.stride(head_axis);
    p.v_stride_n = v_cache.stride(token_axis);
    if constexpr (USE_LOGITS_SOFT_CAP) {
      p.soft_cap_pre_tanh_scale = float(sm_scale / logits_soft_cap);
      p.sm_scale_log2 = float(logits_soft_cap) * kLog2e;
    } else {
      p.soft_cap_pre_tanh_scale = 0.f;
      p.sm_scale_log2 = float(sm_scale) * kLog2e;
    }
    p.v_scale = float(v_scale);
  }

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const hipError_t status =
      mask_mode_code == kMaskModeCausal
          ? BatchAttentionKernel<true>::Launch(params[0], params[1], plan_info.num_blks_y, stream)
          : BatchAttentionKernel<false>::Launch(params[0], params[1], plan_info.num_blks_y, stream);
  TORCH_CHECK(status == hipSuccess,
              "BatchAttention: kernel launch failed: ", hipGetErrorString(status));
}
