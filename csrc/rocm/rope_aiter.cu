// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's cos/sin-cache RoPE C++ API
// (rope_cached_positions_2c_fwd_impl). FlashInfer links the symbol-visible AITER
// module (see flashinfer/jit/aiter_source.py) and calls the kernel directly — no
// Python `import aiter` at runtime.
//
// FlashInfer stores cos_sin_cache as (max_seq_len, rotary_dim) float32 with cosine
// in the first half and sine in the second half. AITER wants two separate
// (max_seq_len, 1, 1, rotary_dim/2) tables (reuse_freqs_front_part=true). Q/K are
// viewed as AITER's SBHD (1, nnz, num_heads, head_dim) layout and only the leading
// rotary_dim slice is rotated (nope_first=false). AITER ships float32-cos/sin
// instances for fp16/bf16 data, so the tables are passed as float32 (better
// precision than downcasting to the query dtype).

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

#include "aiter_tensor_compat.h"

// AITER's public header (rope.h) pulls in <torch/extension.h> → full pybind11,
// which clashes with FlashInfer's -DPy_LIMITED_API, so forward-declare the entry
// point instead; the linker resolves it against the symbol-visible AITER .so.
// 0.1.21 moved it onto the POD aiter_tensor_t API -- declaring it with at::Tensor
// compiles and then fails at dlopen on the mangled name.
void rope_cached_positions_2c_fwd_impl(aiter_tensor_t& output_x, aiter_tensor_t& output_y,
                                       const aiter_tensor_t& input_x, const aiter_tensor_t& input_y,
                                       const aiter_tensor_t& cos, const aiter_tensor_t& sin,
                                       const aiter_tensor_t& positions, const int32_t rotate_style,
                                       const bool reuse_freqs_front_part, const bool nope_first);

void apply_rope_pos_ids_cos_sin_cache_aiter(at::Tensor query, at::Tensor key, at::Tensor query_out,
                                            at::Tensor key_out, at::Tensor cos_sin_cache,
                                            at::Tensor positions, int64_t head_size, bool is_neox) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(query.device());

  TORCH_CHECK(query.scalar_type() == key.scalar_type(),
              "AITER rope requires query and key to share a dtype; got query=", query.scalar_type(),
              ", key=", key.scalar_type());

  const int64_t rotary_dim = cos_sin_cache.size(-1);
  TORCH_CHECK(rotary_dim % 2 == 0, "cos_sin_cache last dim must be even (cos||sin); got ",
              rotary_dim);
  TORCH_CHECK(rotary_dim <= head_size, "rotary_dim (", rotary_dim,
              ") from cos_sin_cache exceeds head_size (", head_size, ")");

  const int64_t nnz = query.size(0);
  const int64_t half = rotary_dim / 2;

  // Split cos||sin into two (max_seq_len, 1, 1, rotary_dim/2) float32 tables.
  at::Tensor cos = cos_sin_cache.slice(/*dim=*/1, 0, half).unsqueeze(1).unsqueeze(1).contiguous();
  at::Tensor sin =
      cos_sin_cache.slice(/*dim=*/1, half, rotary_dim).unsqueeze(1).unsqueeze(1).contiguous();

  // SBHD views (1, nnz, num_heads, head_size).
  at::Tensor q_in = query.view({1, nnz, -1, head_size});
  at::Tensor k_in = key.view({1, nnz, -1, head_size});
  at::Tensor q_out = query_out.view({1, nnz, -1, head_size});
  at::Tensor k_out = key_out.view({1, nnz, -1, head_size});

  // The kernel only writes the rotated [:rotary_dim] slice. For partial rotary
  // into a fresh output, copy the untouched nope tail across first (skipped when
  // the output aliases the input, and when rotary_dim covers the full head_size).
  if (rotary_dim < head_size) {
    if (query_out.data_ptr() != query.data_ptr()) {
      q_out.slice(3, rotary_dim, head_size).copy_(q_in.slice(3, rotary_dim, head_size));
    }
    if (key_out.data_ptr() != key.data_ptr()) {
      k_out.slice(3, rotary_dim, head_size).copy_(k_in.slice(3, rotary_dim, head_size));
    }
  }

  // AITER's kernel requires int64, contiguous positions of shape (1, nnz).
  at::Tensor pos = positions.to(at::kLong).contiguous().view({1, nnz});

  at::Tensor q_out_rot = q_out.slice(3, 0, rotary_dim);
  at::Tensor k_out_rot = k_out.slice(3, 0, rotary_dim);
  at::Tensor q_in_rot = q_in.slice(3, 0, rotary_dim);
  at::Tensor k_in_rot = k_in.slice(3, 0, rotary_dim);

  // Named lvalues: the two outputs are non-const references, so a temporary
  // from to_aiter() would not bind.
  namespace compat = flashinfer::aiter_compat;
  auto a_q_out = compat::to_aiter(q_out_rot);
  auto a_k_out = compat::to_aiter(k_out_rot);
  const compat::StreamGuard stream_guard(at::hip::getCurrentHIPStream());
  rope_cached_positions_2c_fwd_impl(a_q_out, a_k_out, compat::to_aiter(q_in_rot),
                                    compat::to_aiter(k_in_rot), compat::to_aiter(cos),
                                    compat::to_aiter(sin), compat::to_aiter(pos),
                                    /*rotate_style=*/is_neox ? 0 : 1,
                                    /*reuse_freqs_front_part=*/true,
                                    /*nope_first=*/false);
}
