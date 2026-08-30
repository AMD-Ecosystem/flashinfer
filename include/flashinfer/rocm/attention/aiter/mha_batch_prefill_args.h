// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored fmha_batch_prefill_args from AITER amd-aiter>=0.1.10.
// Extracted from
// aiter_meta/3rdparty/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.hpp:501-594.
//
// ABI note: aiter::mha_batch_prefill() is called via dlsym. The struct layout here must
// match the .so exactly. ck_tile::index_t = int32_t.
// Update this header and the kMhaBatchPrefillSymbol pin if AITER changes the struct.
#pragma once

#include <cstdint>
#include <utility>
#include <variant>

// Vendored from block_attention_kvcache_layout_enum.hpp in CK Tile.
// Enum underlying type is default int (verified: enum class Foo {} uses int on Itanium ABI).
namespace ck_tile {

enum class BlockAttentionKVCacheMemoryLayoutEnum : int {
  VECTORIZED_LAYOUT = 0,  // K: [blocks, heads, hdim/vec, page_size, vec], V: swizzled
  LINEAR_LAYOUT = 1,      // K/V: [blocks, page_size, heads, hdim]
};
static_assert(sizeof(BlockAttentionKVCacheMemoryLayoutEnum) == 4, "enum must be 4 bytes");

enum class BlockAttentionKVCacheLookupTableEnum : int {
  VLLM_BLOCK_TABLE_2D = 0,   // block_table[batch, max_blocks]
  SGLANG_PAGE_TABLE_1D = 1,  // kv_page_indices[kv_indptr[b]..kv_indptr[b+1])
};
static_assert(sizeof(BlockAttentionKVCacheLookupTableEnum) == 4, "enum must be 4 bytes");

}  // namespace ck_tile

// Vendored from fmha_fwd.hpp:501-594.  Global scope (not in any namespace).
// ck_tile::index_t is int32_t.
struct fmha_batch_prefill_args {
  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr;  // alibi_slope or elementwise bias
  const void* q_descale_ptr;
  const void* k_descale_ptr;
  const void* v_descale_ptr;
  void* rand_val_ptr;
  void* lse_ptr;  // [num_qo_heads, total_qo_len] float32, natural-log
  void* o_ptr;

  // Group-mode seqstarts; unused in batch mode (set to nullptr).
  const void* seqstart_q_ptr;
  const void* sink_ptr;

  // Dimensions (ck_tile::index_t == int32_t)
  int32_t seqlen_q;
  int32_t seqlen_k;
  int32_t batch;
  int32_t max_seqlen_q;
  int32_t hdim_q;
  int32_t hdim_v;
  int32_t nhead_q;
  int32_t nhead_k;

  // Paged KV cache fields
  int32_t num_total_pages;
  int32_t page_block_size;
  ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout;
  ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table;
  void* kv_indptr;                   // [batch+1] int32 prefix-sum into kv_page_indices
  void* kv_page_indices;             // [num_total_pages] int32 page ids
  void* kv_last_page_lens;           // [batch] int32 last-page fill lengths
  void* seqlen_k_ptr;                // [batch] int32, vLLM mode only
  int32_t batch_stride_block_table;  // vLLM: row stride of block_table; SGLang: unused

  float scale_s;
  float scale_p;
  float scale_o;
  float logits_soft_cap;

  int32_t stride_q;
  int32_t stride_k;
  int32_t stride_v;
  int32_t stride_bias;
  int32_t stride_randval;
  int32_t stride_o;
  int32_t nhead_stride_q;
  int32_t nhead_stride_k;
  int32_t nhead_stride_v;
  int32_t nhead_stride_bias;
  int32_t nhead_stride_randval;
  int32_t nhead_stride_lse;  // == total_qo_len (lse layout: [nhead, total_q])
  int32_t nhead_stride_o;
  int32_t batch_stride_q;
  int32_t batch_stride_k;
  int32_t batch_stride_v;
  int32_t batch_stride_bias;
  int32_t batch_stride_randval;
  int32_t batch_stride_lse;
  int32_t batch_stride_o;

  int32_t window_size_left;
  int32_t window_size_right;
  int32_t sink_size;
  int32_t mask_type;  // 0=no_mask, 1=mask_top_left, 2=mask_bottom_right (causal), 3=window_generic

  float p_drop;
  bool s_randval;

  // Dropout seed/offset: zero-initialize when dropout disabled (p_drop=0).
  // std::variant layout is libstdc++ ABI-dependent; we always use the first alternative.
  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>> drop_seed_offset;
};

namespace aiter {
using mha_batch_prefill_args = fmha_batch_prefill_args;
}  // namespace aiter
