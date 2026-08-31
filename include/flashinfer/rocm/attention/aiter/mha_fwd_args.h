// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored aiter::mha_fwd_args from AITER amd-aiter>=0.1.16.
// Extracted from aiter_meta/csrc/include/mha_fwd.h.
//
// 0.1.16 is a hard minimum, not a recommendation: the block_scale_seqstart_*
// fields below were inserted before sink_ptr, so on 0.1.10 every field from
// sink_ptr on lands at the wrong offset. The struct goes by value through a
// dlsym'd pointer, so the mismatch corrupts silently rather than failing to
// load. flashinfer.aiter_utils enforces the floor at run time.
//
// ABI note: aiter::mha_fwd() is called via dlsym. The struct layout here must
// match the .so exactly. ck_tile::index_t = int32_t (ck_tile/core/numeric/integer.hpp).
// Update this header and bump the amd-aiter version pin if AITER changes the struct.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace aiter {

// mask_enum values from CK Tile's fmha example (mask.hpp):
//   no_mask=0, mask_top_left=1, mask_bottom_right=2, window_generic=3
// bias_enum values: no_bias=0, elementwise_bias=1, alibi=2
// These are only used to document the mask_type / bias_type integer fields below.

struct mha_fwd_args {
  // AITER-specific dispatch fields (control which pipeline is selected)
  bool use_asm_v3 = false;
  bool v3_api_check = false;
  int how_v3_bf16_cvt = 0;

  // Traits baked into the JIT-compiled variant (must match the .so file loaded)
  std::string data_type;  // "fp16" or "bf16"
  bool is_group_mode;     // false = batch mode (seqstart/seqlen ptrs must be nullptr)
  int bias_type = 0;      // 0=no_bias, 1=elementwise, 2=alibi
  bool has_lse;
  int qscale_type = 0;  // 0=no_scale
  bool has_sink = false;

  // Data pointers
  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr = nullptr;
  const void* q_descale_ptr = nullptr;
  const void* k_descale_ptr = nullptr;
  const void* v_descale_ptr = nullptr;
  void* rand_val_ptr = nullptr;
  void* lse_ptr = nullptr;
  void* o_ptr;

  // Sequence-length pointers (batch mode: all nullptr; group mode: use seqstart_*)
  const void* seqstart_q_ptr = nullptr;
  const void* seqstart_k_ptr = nullptr;
  const void* seqlen_q_ptr = nullptr;
  const void* seqlen_k_ptr = nullptr;
  const void* cu_seqlen_q_ptr = nullptr;
  const void* cu_seqlen_k_ptr = nullptr;
  // Added in 0.1.16 — inserted *before* sink_ptr, so every later field shifts.
  const void* block_scale_seqstart_q_ptr = nullptr;
  const void* block_scale_seqstart_k_ptr = nullptr;
  const void* sink_ptr = nullptr;

  // Dimensions (ck_tile::index_t = int32_t)
  int32_t seqlen_q;
  int32_t seqlen_k;
  int32_t batch;
  int32_t max_seqlen_q;
  int32_t hdim_q;
  int32_t hdim_v;
  int32_t nhead_q;
  int32_t nhead_k;

  float scale_s;
  float logits_soft_cap = 0.0f;

  // Strides (in elements)
  int32_t stride_q;
  int32_t stride_k;
  int32_t stride_v;
  int32_t stride_bias = 0;
  int32_t stride_randval = 0;
  int32_t stride_o;

  int32_t nhead_stride_q;
  int32_t nhead_stride_k;
  int32_t nhead_stride_v;
  int32_t nhead_stride_bias = 0;
  int32_t nhead_stride_randval = 0;
  int32_t nhead_stride_lse = 0;
  int32_t nhead_stride_o;
  // Added in 0.1.16.
  int32_t nhead_stride_q_descale = 0;
  int32_t nhead_stride_k_descale = 0;
  int32_t nhead_stride_v_descale = 0;

  int32_t batch_stride_q = 0;
  int32_t batch_stride_k = 0;
  int32_t batch_stride_v = 0;
  int32_t batch_stride_bias = 0;
  int32_t batch_stride_randval = 0;
  int32_t batch_stride_lse = 0;
  int32_t batch_stride_o = 0;
  // Added in 0.1.16.
  int32_t batch_stride_q_descale = 0;
  int32_t batch_stride_k_descale = 0;
  int32_t batch_stride_v_descale = 0;

  int32_t window_size_left = -1;
  int32_t window_size_right = -1;
  int32_t sink_size = 0;
  int32_t mask_type;  // 0=no_mask, 1=mask_top_left (causal), 2=mask_bottom_right, 3=window_generic
  int32_t min_seqlen_q = 0;

  float p_drop = 0.0f;
  bool s_randval = false;

  // Dropout seed/offset (first variant = {0,0} when dropout disabled)
  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>> drop_seed_offset;

  // Added in 0.1.16. Must match what AITER's own arg builders pass, which is
  // an unconditional 128; the pipeline never sees 0 from AITER itself.
  int32_t block_scale_size_q = 128;
  int32_t block_scale_size_kv = 128;
};

}  // namespace aiter
