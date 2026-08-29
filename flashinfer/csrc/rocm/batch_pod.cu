// SPDX-FileCopyrightText: 2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <flashinfer/rocm/attention/allocator.h>

#include <flashinfer/rocm/attention/pos_enc.cuh>
#include <flashinfer/rocm/attention/scheduler.cuh>
#include <flashinfer/rocm/dispatch.cuh>
#include <flashinfer/rocm/enums.hpp>
#include <flashinfer/rocm/fastdiv.cuh>
#include <optional>

#include "batch_pod_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, uint32_t CTA_TILE_Q_P, MaskMode MASK_MODE_P,
          uint32_t CTA_TILE_Q_D, MaskMode MASK_MODE_D, typename PrefillAttentionVariant,
          typename DecodeAttentionVariant, typename PrefillParams, typename DecodeParams>
hipError_t BatchPODWithKVCacheTensorDispatched(PrefillParams prefill_params,
                                               typename PrefillParams::DTypeO* tmp_v_p,
                                               float* tmp_s_p, DecodeParams decode_params,
                                               typename DecodeParams::DTypeO* tmp_v_d,
                                               float* tmp_s_d, bool enable_pdl, hipStream_t stream,
                                               int* sm_aware_sched);

}  // namespace flashinfer

using namespace flashinfer;

void batch_pod_with_kv_cache_tensor(
    // Prefill params
    at::Tensor float_workspace_buffer_p, at::Tensor int_workspace_buffer_p,
    at::Tensor plan_info_vec_p, at::Tensor q_p, at::Tensor paged_k_cache_p,
    at::Tensor paged_v_cache_p, at::Tensor qo_indptr_p, at::Tensor paged_kv_indptr_p,
    at::Tensor paged_kv_indices_p, at::Tensor paged_kv_last_page_len_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_mask_indptr_p, std::optional<at::Tensor> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec_d, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl, at::Tensor sm_aware_sched) {
  // Prefill setup
  PrefillPlanInfo plan_info_p;
  plan_info_p.FromVector(tensor_to_vec(plan_info_vec_p));
  TORCH_CHECK(!plan_info_p.enable_cuda_graph,
              "CUDA graph mode is not supported with BatchPOD on ROCm");
  QKVLayout kv_layout_p = static_cast<QKVLayout>(layout_p);
  int64_t batch_size_p = paged_kv_indptr_p.size(0) - 1;
  int64_t num_qo_heads = q_p.size(1);

  int64_t num_kv_heads_p, page_size_p;
  if (kv_layout_p == QKVLayout::kHND) {
    num_kv_heads_p = paged_k_cache_p.size(1);
    page_size_p = paged_k_cache_p.size(2);
  } else {
    page_size_p = paged_k_cache_p.size(1);
    num_kv_heads_p = paged_k_cache_p.size(2);
  }

  void* float_buffer_ptr_p = float_workspace_buffer_p.data_ptr();
  void* int_buffer_ptr_p = int_workspace_buffer_p.data_ptr();

  const MaskMode mask_mode_p = static_cast<MaskMode>(mask_mode_code_p);
  const auto q_stride_n_p = q_p.stride(0);
  const auto q_stride_h_p = q_p.stride(1);

  auto k_strides_p = paged_k_cache_p.strides();
  auto v_strides_p = paged_v_cache_p.strides();
  TORCH_CHECK(k_strides_p == v_strides_p, "prefill k/v strides must be identical");
  const int64_t* kv_cache_strides_p = k_strides_p.data();

  // Decode setup
  PrefillPlanInfo plan_info_d;
  plan_info_d.FromVector(tensor_to_vec(plan_info_vec_d));
  TORCH_CHECK(!plan_info_d.enable_cuda_graph,
              "CUDA graph mode is not supported with BatchPOD on ROCm");
  QKVLayout kv_layout_d = static_cast<QKVLayout>(layout_d);
  int64_t batch_size_d = paged_kv_indptr_d.size(0) - 1;
  int64_t num_qo_heads_d = q_d.size(1);

  TORCH_CHECK(num_qo_heads == num_qo_heads_d, "BatchPOD: prefill/decode QO heads must match");

  int64_t num_kv_heads_d, page_size_d;
  if (kv_layout_d == QKVLayout::kHND) {
    num_kv_heads_d = paged_k_cache_d.size(1);
    page_size_d = paged_k_cache_d.size(2);
  } else {
    page_size_d = paged_k_cache_d.size(1);
    num_kv_heads_d = paged_k_cache_d.size(2);
  }

  TORCH_CHECK(num_kv_heads_p == num_kv_heads_d,
              "BatchPOD: prefill/decode KV heads must match; prefill: ", num_kv_heads_p,
              ", decode: ", num_kv_heads_d);

  void* float_buffer_ptr_d = float_workspace_buffer_d.data_ptr();
  void* int_buffer_ptr_d = int_workspace_buffer_d.data_ptr();

  const MaskMode mask_mode_d = static_cast<MaskMode>(mask_mode_code_d);
  const auto q_stride_n_d = q_d.stride(0);
  const auto q_stride_h_d = q_d.stride(1);

  auto k_strides_d = paged_k_cache_d.strides();
  auto v_strides_d = paged_v_cache_d.strides();
  TORCH_CHECK(k_strides_d == v_strides_d, "decode k/v strides must be identical");
  const int64_t* kv_cache_strides_d = k_strides_d.data();

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      float_workspace_buffer_p.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  DISPATCH_context(
      MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK, USE_SLIDING_WINDOW_P,
      USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, [&] {
        PrefillParams prefill_params;
        DTypeO* tmp_v_p = nullptr;
        float* tmp_s_p = nullptr;
        {
          PrefillParams& params = prefill_params;
          params.q = static_cast<DTypeQ*>(q_p.data_ptr());
          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads_p, page_size_p, HEAD_DIM_VO, batch_size_p, kv_layout_p,
              static_cast<DTypeKV*>(paged_k_cache_p.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache_p.data_ptr()), kv_cache_strides_p,
              static_cast<IdType*>(paged_kv_indices_p.data_ptr()),
              static_cast<IdType*>(paged_kv_indptr_p.data_ptr()),
              static_cast<IdType*>(paged_kv_last_page_len_p.data_ptr()));
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr_p.data_ptr());
          params.o = static_cast<DTypeO*>(o_p.data_ptr());
          params.lse = maybe_lse_p ? static_cast<float*>(maybe_lse_p->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / static_cast<uint32_t>(num_kv_heads_p));
          params.q_stride_n = q_stride_n_p;
          params.q_stride_h = q_stride_h_p;
          params.window_left = window_left_p;

          params.request_indices = nullptr;
          params.qo_tile_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.merge_indptr = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.total_num_rows = nullptr;
          params.max_total_num_rows = 0;
          params.padded_batch_size = 0;
          params.partition_kv = false;

          params.maybe_custom_mask = maybe_custom_mask_p
                                         ? static_cast<uint8_t*>(maybe_custom_mask_p->data_ptr())
                                         : nullptr;
          params.maybe_mask_indptr =
              maybe_mask_indptr_p ? static_cast<IdType*>(maybe_mask_indptr_p->data_ptr()) : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_p
                                          ? static_cast<float*>(maybe_alibi_slopes_p->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = static_cast<float>(logits_soft_cap_p);
          params.sm_scale = static_cast<float>(sm_scale_p);
          params.rope_rcp_scale = static_cast<float>(rope_rcp_scale_p);
          params.rope_rcp_theta = static_cast<float>(rope_rcp_theta_p);

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.kv_tile_indices_offset);
          params.o_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.kv_chunk_size_ptr_offset);
          if (plan_info_p.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr_p, plan_info_p.merge_indptr_offset);
            tmp_v_p = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr_p, plan_info_p.v_offset);
            tmp_s_p = GetPtrFromBaseOffset<float>(float_buffer_ptr_p, plan_info_p.s_offset);
            if (plan_info_p.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr_p, plan_info_p.block_valid_mask_offset);
            }
          }
          params.padded_batch_size = plan_info_p.padded_batch_size;
          params.max_total_num_rows = plan_info_p.total_num_rows;
          if (plan_info_p.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr_p, plan_info_p.total_num_rows_offset);
          }
        }

        DecodeParams decode_params;
        DTypeO* tmp_v_d = nullptr;
        float* tmp_s_d = nullptr;
        {
          DecodeParams& params = decode_params;
          params.q = static_cast<DTypeQ*>(q_d.data_ptr());
          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads_d, page_size_d, HEAD_DIM_VO, batch_size_d, kv_layout_d,
              static_cast<DTypeKV*>(paged_k_cache_d.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache_d.data_ptr()), kv_cache_strides_d,
              static_cast<IdType*>(paged_kv_indices_d.data_ptr()),
              static_cast<IdType*>(paged_kv_indptr_d.data_ptr()),
              static_cast<IdType*>(paged_kv_last_page_len_d.data_ptr()));
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr_d.data_ptr());
          params.o = static_cast<DTypeO*>(o_d.data_ptr());
          params.lse = maybe_lse_d ? static_cast<float*>(maybe_lse_d->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / static_cast<uint32_t>(num_kv_heads_d));
          params.q_stride_n = q_stride_n_d;
          params.q_stride_h = q_stride_h_d;
          params.window_left = window_left_d;

          params.request_indices = nullptr;
          params.qo_tile_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.merge_indptr = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.total_num_rows = nullptr;
          params.max_total_num_rows = 0;
          params.padded_batch_size = 0;
          params.partition_kv = false;

          params.maybe_custom_mask = maybe_custom_mask_d
                                         ? static_cast<uint8_t*>(maybe_custom_mask_d->data_ptr())
                                         : nullptr;
          params.maybe_mask_indptr =
              maybe_mask_indptr_d ? static_cast<IdType*>(maybe_mask_indptr_d->data_ptr()) : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_d
                                          ? static_cast<float*>(maybe_alibi_slopes_d->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = static_cast<float>(logits_soft_cap_d);
          params.sm_scale = static_cast<float>(sm_scale_d);
          params.rope_rcp_scale = static_cast<float>(rope_rcp_scale_d);
          params.rope_rcp_theta = static_cast<float>(rope_rcp_theta_d);

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.kv_tile_indices_offset);
          params.o_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.kv_chunk_size_ptr_offset);
          if (plan_info_d.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr_d, plan_info_d.merge_indptr_offset);
            tmp_v_d = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr_d, plan_info_d.v_offset);
            tmp_s_d = GetPtrFromBaseOffset<float>(float_buffer_ptr_d, plan_info_d.s_offset);
            if (plan_info_d.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr_d, plan_info_d.block_valid_mask_offset);
            }
          }
          params.padded_batch_size = plan_info_d.padded_batch_size;
          params.max_total_num_rows = plan_info_d.total_num_rows;
          if (plan_info_d.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr_d, plan_info_d.total_num_rows_offset);
          }
        }

        hipError_t status = hipSuccess;
        DISPATCH_CTA_TILE_Q(plan_info_p.cta_tile_q, CTA_TILE_Q_P, {
          constexpr size_t CTA_TILE_Q_D = 16;
          status = flashinfer::BatchPODWithKVCacheTensorDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, CTA_TILE_Q_P,
              MASK_MODE_P, CTA_TILE_Q_D, MASK_MODE_D, PrefillAttentionVariant,
              DecodeAttentionVariant>(prefill_params, tmp_v_p, tmp_s_p, decode_params, tmp_v_d,
                                      tmp_s_d, enable_pdl, stream,
                                      static_cast<int*>(sm_aware_sched.data_ptr()));
        });

        TORCH_CHECK(status == hipSuccess,
                    "BatchPODWithKVCache kernel launch failed, error: ", hipGetErrorString(status));
        return true;
      });
}
