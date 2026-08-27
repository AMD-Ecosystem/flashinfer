/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/rocm/attention/pos_enc.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;
using flashinfer::QKVLayout;

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  TORCH_CHECK(offsets.scalar_type() == indptr.scalar_type(),
              "offsets and indptr must have the same dtype");

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(indptr.scalar_type(), c_idtype, [&] {
      hipError_t status = BatchQKApplyRotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, stream);
      TORCH_CHECK(status == hipSuccess, "BatchQKApplyRotary failed with error code " +
                                            std::string(hipGetErrorString(status)));
      return true;
    });
  });
}

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      hipError_t status = BatchQKApplyRotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          stream);
      TORCH_CHECK(status == hipSuccess, "BatchQKApplyRotaryPosIds failed with error code " +
                                            std::string(hipGetErrorString(status)));
      return true;
    });
  });
}

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(cos_sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_DIM(2, cos_sin_cache);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int rotary_dim = cos_sin_cache.size(1);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      hipError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()), static_cast<c_idtype*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
          k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, stream);
      TORCH_CHECK(status == hipSuccess,
                  "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " +
                      std::string(hipGetErrorString(status)));
      return true;
    });
  });
}

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  TORCH_CHECK(offsets.scalar_type() == indptr.scalar_type(),
              "offsets and indptr must have the same dtype");

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(indptr.scalar_type(), c_idtype, [&] {
      hipError_t status = BatchQKApplyLlama31Rotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
          old_context_length, stream);
      TORCH_CHECK(status == hipSuccess, "BatchQKApplyLlama31Rotary failed with error code " +
                                            std::string(hipGetErrorString(status)));
      return true;
    });
  });
}

void rope_quantize(at::Tensor q_rope_in, at::Tensor k_rope_in, at::Tensor q_nope_in,
                   at::Tensor k_nope_in, at::Tensor q_rope_out, at::Tensor k_rope_out,
                   at::Tensor q_nope_out, at::Tensor k_nope_out, at::Tensor cos_sin_cache,
                   at::Tensor pos_ids, double quant_scale_q, double quant_scale_kv, bool interleave,
                   bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_out);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope_out);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);

  auto device = q_rope_in.device();
  CHECK_DIM(3, q_rope_in);
  CHECK_DIM(3, q_nope_in);
  CHECK_DIM(3, q_rope_out);
  CHECK_DIM(3, q_nope_out);

  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);
  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);

  uint32_t num_kv_heads;
  uint32_t k_rope_in_stride, k_nope_in_stride, k_rope_out_stride, k_nope_out_stride;
  uint32_t k_rope_in_stride_h, k_nope_in_stride_h, k_rope_out_stride_h, k_nope_out_stride_h;

  if (k_rope_in.dim() == 2) {
    TORCH_CHECK(k_nope_in.dim() == 2 && k_rope_out.dim() == 2 && k_nope_out.dim() == 2,
                "MLA mode expects all K tensors to be 2D");
    TORCH_CHECK(k_rope_in.size(0) == nnz && k_nope_in.size(0) == nnz && k_rope_out.size(0) == nnz &&
                    k_nope_out.size(0) == nnz,
                "K tensor token dim must match q_rope_in.size(0)=", nnz);
    TORCH_CHECK(k_rope_in.size(1) == rope_dim && k_rope_out.size(1) == rope_dim,
                "k_rope_in/out last dim must equal rope_dim=", rope_dim);
    TORCH_CHECK(k_nope_in.size(1) == no_rope_dim && k_nope_out.size(1) == no_rope_dim,
                "k_nope_in/out last dim must equal no_rope_dim=", no_rope_dim);
    num_kv_heads = 1;
    k_rope_in_stride = k_rope_in.stride(0);
    k_nope_in_stride = k_nope_in.stride(0);
    k_rope_out_stride = k_rope_out.stride(0);
    k_nope_out_stride = k_nope_out.stride(0);
    // head stride = batch stride: the kernel still receives a head-stride argument,
    // so we alias it to the token stride to get num_kv_heads=1 behaviour.
    k_rope_in_stride_h = k_rope_in_stride;
    k_nope_in_stride_h = k_nope_in_stride;
    k_rope_out_stride_h = k_rope_out_stride;
    k_nope_out_stride_h = k_nope_out_stride;
  } else {
    CHECK_DIM(3, k_rope_in);
    CHECK_DIM(3, k_nope_in);
    CHECK_DIM(3, k_rope_out);
    CHECK_DIM(3, k_nope_out);
    num_kv_heads = k_rope_in.size(1);
    TORCH_CHECK(k_rope_in.size(0) == nnz && k_nope_in.size(0) == nnz && k_rope_out.size(0) == nnz &&
                    k_nope_out.size(0) == nnz,
                "K tensor token dim must match q_rope_in.size(0)=", nnz);
    TORCH_CHECK(k_nope_in.size(1) == num_kv_heads && k_rope_out.size(1) == num_kv_heads &&
                    k_nope_out.size(1) == num_kv_heads,
                "K tensor head dim must match k_rope_in.size(1)=", num_kv_heads);
    TORCH_CHECK(k_rope_in.size(2) == rope_dim && k_rope_out.size(2) == rope_dim,
                "k_rope_in/out last dim must equal rope_dim=", rope_dim);
    TORCH_CHECK(k_nope_in.size(2) == no_rope_dim && k_nope_out.size(2) == no_rope_dim,
                "k_nope_in/out last dim must equal no_rope_dim=", no_rope_dim);
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in.stride(1);
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in.stride(1);
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out.stride(1);
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out.stride(1);
  }

  TORCH_CHECK(q_rope_in.scalar_type() == at::kHalf || q_rope_in.scalar_type() == at::kBFloat16,
              "Input dtype must be float16 or bfloat16");
  TORCH_CHECK(k_rope_in.scalar_type() == q_rope_in.scalar_type(),
              "k_rope_in dtype must match q_rope_in dtype");
  TORCH_CHECK(q_nope_in.scalar_type() == q_rope_in.scalar_type(),
              "q_nope_in dtype must match q_rope_in dtype");
  TORCH_CHECK(k_nope_in.scalar_type() == q_rope_in.scalar_type(),
              "k_nope_in dtype must match q_rope_in dtype");
  // The kernel tiles no_rope_dim in rope_dim-sized chunks; a non-multiple tail
  // would cause the last chunk to read/write past the end of the K-nope buffer.
  TORCH_CHECK(no_rope_dim == 0 || no_rope_dim % rope_dim == 0,
              "no_rope_dim must be 0 or a multiple of rope_dim (got no_rope_dim=", no_rope_dim,
              ", rope_dim=", rope_dim, ")");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::kFloat, "cos_sin_cache dtype must be float32");
  // pos_ids is int32-only: the RoPE quantize kernel template is instantiated with IdType=int32_t.
  TORCH_CHECK(pos_ids.scalar_type() == at::kInt, "pos_ids dtype must be int32");
  TORCH_CHECK(is_float8_tensor(q_rope_out), "Output dtype must be float8");

  const uint32_t q_rope_in_stride_n = q_rope_in.stride(0);
  const uint32_t q_rope_in_stride_h = q_rope_in.stride(1);
  const uint32_t q_nope_in_stride_n = q_nope_in.stride(0);
  const uint32_t q_nope_in_stride_h = q_nope_in.stride(1);
  const uint32_t q_rope_out_stride_n = q_rope_out.stride(0);
  const uint32_t q_rope_out_stride_h = q_rope_out.stride(1);
  const uint32_t q_nope_out_stride_n = q_nope_out.stride(0);
  const uint32_t q_nope_out_stride_h = q_nope_out.stride(1);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_rope_in.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q_rope_out.scalar_type(), c_quant_type, [&] {
      hipError_t status = RopeQuantize<c_type, int32_t, c_quant_type>(
          static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
          static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
          static_cast<c_quant_type*>(q_rope_out.data_ptr()),
          static_cast<c_quant_type*>(k_rope_out.data_ptr()),
          static_cast<c_quant_type*>(q_nope_out.data_ptr()),
          static_cast<c_quant_type*>(k_nope_out.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()), static_cast<int32_t*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim, q_rope_in_stride_n,
          q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h, q_rope_out_stride_n,
          q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h, k_rope_in_stride,
          k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h, k_rope_out_stride,
          k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
          static_cast<float>(quant_scale_q), static_cast<float>(quant_scale_kv), interleave,
          enable_pdl, stream);
      TORCH_CHECK(status == hipSuccess,
                  "RopeQuantize failed with error code " + std::string(hipGetErrorString(status)));
      return true;
    });
  });
}

void rope_quantize_append_paged_kv_cache(
    at::Tensor q_rope_in, at::Tensor k_rope_in, at::Tensor q_nope_in, at::Tensor k_nope_in,
    at::Tensor v_in, at::Tensor q_rope_out, at::Tensor q_nope_out, at::Tensor cos_sin_cache,
    at::Tensor pos_ids, at::Tensor k_cache, at::Tensor v_cache, at::Tensor ckv_cache,
    at::Tensor kpe_cache, at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor batch_indices,
    at::Tensor positions, int64_t kv_layout_code, int64_t page_size, double quant_scale_q,
    double quant_scale_kv, bool interleave, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_out);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);

  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);
  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);
  uint32_t batch_size = kv_indptr.size(0) - 1;

  TORCH_CHECK(q_rope_in.scalar_type() == at::kHalf || q_rope_in.scalar_type() == at::kBFloat16,
              "Input dtype must be float16 or bfloat16");
  TORCH_CHECK(k_rope_in.scalar_type() == q_rope_in.scalar_type(),
              "k_rope_in dtype must match q_rope_in dtype");
  TORCH_CHECK(q_nope_in.scalar_type() == q_rope_in.scalar_type(),
              "q_nope_in dtype must match q_rope_in dtype");
  TORCH_CHECK(k_nope_in.scalar_type() == q_rope_in.scalar_type(),
              "k_nope_in dtype must match q_rope_in dtype");
  // The kernel tiles no_rope_dim in rope_dim-sized chunks; a non-multiple tail
  // would cause the last chunk to read/write past the end of the K-nope buffer.
  TORCH_CHECK(no_rope_dim == 0 || no_rope_dim % rope_dim == 0,
              "no_rope_dim must be 0 or a multiple of rope_dim (got no_rope_dim=", no_rope_dim,
              ", rope_dim=", rope_dim, ")");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::kFloat, "cos_sin_cache dtype must be float32");
  // pos_ids is intentionally int32-only: paged-cache index arithmetic uses int32 throughout.
  TORCH_CHECK(pos_ids.scalar_type() == at::kInt, "pos_ids dtype must be int32");
  TORCH_CHECK(is_float8_tensor(q_rope_out), "Output dtype must be float8");

  bool has_gqa_caches =
      k_cache.defined() && k_cache.numel() > 0 && v_cache.defined() && v_cache.numel() > 0;
  bool has_mla_caches =
      ckv_cache.defined() && ckv_cache.numel() > 0 && kpe_cache.defined() && kpe_cache.numel() > 0;
  TORCH_CHECK(has_gqa_caches || has_mla_caches,
              "rope_quantize_append_paged_kv_cache requires either GQA caches (k_cache, v_cache) "
              "or MLA caches (ckv_cache, kpe_cache)");
  TORCH_CHECK(!(has_gqa_caches && has_mla_caches),
              "rope_quantize_append_paged_kv_cache: provide either GQA caches (k_cache, v_cache) "
              "or MLA caches (ckv_cache, kpe_cache), not both");
  bool is_mla = has_mla_caches;
  if (has_gqa_caches) {
    CHECK_LAST_DIM_CONTIGUOUS(v_in);
    TORCH_CHECK(v_in.scalar_type() == q_rope_in.scalar_type(),
                "v_in dtype must match q_rope_in dtype (expected float16 or bfloat16)");
  }

  const uint32_t q_rope_in_stride_n = q_rope_in.stride(0);
  const uint32_t q_rope_in_stride_h = q_rope_in.stride(1);
  const uint32_t q_nope_in_stride_n = q_nope_in.stride(0);
  const uint32_t q_nope_in_stride_h = q_nope_in.stride(1);
  const uint32_t q_rope_out_stride_n = q_rope_out.stride(0);
  const uint32_t q_rope_out_stride_h = q_rope_out.stride(1);
  const uint32_t q_nope_out_stride_n = q_nope_out.stride(0);
  const uint32_t q_nope_out_stride_h = q_nope_out.stride(1);
  const uint32_t k_rope_in_stride = k_rope_in.stride(0);
  const uint32_t k_rope_in_stride_h = k_rope_in.stride(1);
  const uint32_t k_nope_in_stride = k_nope_in.stride(0);
  const uint32_t k_nope_in_stride_h = k_nope_in.stride(1);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q_rope_in.device());
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_rope_in.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q_rope_out.scalar_type(), c_quant_type, [&] {
      hipError_t status;

      if (is_mla) {
        TORCH_CHECK(k_rope_in.dim() == 3 && k_rope_in.size(1) == 1,
                    "MLA expects k_rope_in to be 3D with num_kv_heads=1");
        TORCH_CHECK(k_nope_in.dim() == 3 && k_nope_in.size(1) == 1,
                    "MLA expects k_nope_in to be 3D with num_kv_heads=1");
        TORCH_CHECK(k_rope_in.size(0) == nnz && k_nope_in.size(0) == nnz,
                    "k_rope_in/k_nope_in token dim must match q_rope_in.size(0)=", nnz);
        TORCH_CHECK(k_rope_in.size(2) == rope_dim,
                    "k_rope_in last dim must equal rope_dim=", rope_dim);
        TORCH_CHECK(k_nope_in.size(2) == no_rope_dim,
                    "k_nope_in last dim must equal no_rope_dim=", no_rope_dim);
        // MLA cache layout is [num_pages, page_size, dim].
        TORCH_CHECK(ckv_cache.dim() == 3 && kpe_cache.dim() == 3,
                    "ckv_cache and kpe_cache must be 3D [num_pages, page_size, dim]");
        TORCH_CHECK(ckv_cache.size(1) == page_size && kpe_cache.size(1) == page_size,
                    "ckv_cache/kpe_cache page_size dim must equal page_size=", page_size);
        TORCH_CHECK(ckv_cache.size(2) == no_rope_dim,
                    "ckv_cache last dim must equal no_rope_dim=", no_rope_dim);
        TORCH_CHECK(kpe_cache.size(2) == rope_dim,
                    "kpe_cache last dim must equal rope_dim=", rope_dim);

        auto ckv_strides = ckv_cache.strides();
        auto kpe_strides = kpe_cache.strides();
        paged_kv_mla_t<c_quant_type, int32_t> paged_kv_mla(
            page_size, no_rope_dim, rope_dim, batch_size,
            static_cast<c_quant_type*>(ckv_cache.data_ptr()), ckv_strides.data(),
            static_cast<c_quant_type*>(kpe_cache.data_ptr()), kpe_strides.data(),
            static_cast<int32_t*>(kv_indices.data_ptr()),
            static_cast<int32_t*>(kv_indptr.data_ptr()),
            /*last_page_len=*/nullptr);

        status = RopeQuantizeAppendPagedMLACache<c_type, int32_t, c_quant_type>(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()), paged_kv_mla,
            static_cast<int32_t*>(batch_indices.data_ptr()),
            static_cast<int32_t*>(positions.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, rope_dim, no_rope_dim,
            q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
            q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
            k_rope_in_stride, k_nope_in_stride, static_cast<float>(quant_scale_q),
            static_cast<float>(quant_scale_kv), interleave, enable_pdl, stream);
        TORCH_CHECK(status == hipSuccess,
                    "RopeQuantizeAppendPagedMLACache failed with error code " +
                        std::string(hipGetErrorString(status)));

      } else {
        uint32_t num_kv_heads = k_rope_in.size(1);
        uint32_t head_dim = rope_dim + no_rope_dim;
        const uint32_t v_in_stride = v_in.stride(0);
        const uint32_t v_in_stride_h = v_in.stride(1);
        QKVLayout kv_layout = QKVLayout(kv_layout_code);
        auto k_strides = k_cache.strides();

        paged_kv_t<c_quant_type, int32_t> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size, kv_layout,
            static_cast<c_quant_type*>(k_cache.data_ptr()),
            static_cast<c_quant_type*>(v_cache.data_ptr()), k_strides.data(),
            static_cast<int32_t*>(kv_indices.data_ptr()),
            static_cast<int32_t*>(kv_indptr.data_ptr()),
            /*last_page_len=*/nullptr);

        status = RopeQuantizeAppendPagedKVCache<c_type, int32_t, c_quant_type>(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_type*>(v_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()), paged_kv,
            static_cast<int32_t*>(batch_indices.data_ptr()),
            static_cast<int32_t*>(positions.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rope_dim,
            no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
            q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n,
            q_nope_out_stride_h, k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride,
            k_nope_in_stride_h, v_in_stride, v_in_stride_h, static_cast<float>(quant_scale_q),
            static_cast<float>(quant_scale_kv), interleave, enable_pdl, stream);
        TORCH_CHECK(status == hipSuccess, "RopeQuantizeAppendPagedKVCache failed with error code " +
                                              std::string(hipGetErrorString(status)));
      }
      return true;
    });
  });
}

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q.device());
  auto stream = c10::hip::getCurrentHIPStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      hipError_t status = BatchQKApplyLlama31RotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          low_freq_factor, high_freq_factor, old_context_length, stream);
      TORCH_CHECK(status == hipSuccess, "BatchQKApplyLlama31RotaryPosIds failed with error code " +
                                            std::string(hipGetErrorString(status)));
      return true;
    });
  });
}
