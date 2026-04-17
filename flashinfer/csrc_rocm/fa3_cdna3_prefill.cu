// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: JIT-compiled host-side entry point with split-KV parallelism
// targeting AMD MI300X (gfx942).
// Tile128x128 (4 waves, 256 threads) for non-causal.
// Tile64x128  (2 waves, 128 threads) for causal — reduced VGPR pressure.
// Split-KV: when CU utilization is low, splits KV dimension across multiple
// thread blocks and merges partial results via MergeStates.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/cdna3/fa3_kernel.hpp>
#include <flashinfer/attention/generic/cascade.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

using namespace flashinfer::cdna3;

namespace {

template <class Tile, bool IsCausal>
void launch_fa3_prefill_kernel(dim3 grid, hipStream_t stream, size_t smem_bytes, dim3 block,
                               __half* O, float* LSE, const __half* q_ptr, const __half* k_ptr,
                               const __half* v_ptr, int N_q, int N_kv, int nhead, int nhead_k,
                               float scale_log2, int causal_offset, int num_q_blocks,
                               int total_q_head_blocks, int kv_chunk_size, int num_kv_chunks,
                               int o_row_stride, int lse_row_stride, int lse_head_stride,
                               bool base2_lse) {
  fa3_cdna3_prefill_kernel_impl<Tile, IsCausal><<<grid, block, smem_bytes, stream>>>(
      O, LSE, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2, causal_offset,
      num_q_blocks, total_q_head_blocks, kv_chunk_size, num_kv_chunks, o_row_stride, lse_row_stride,
      lse_head_stride, base2_lse);
}

template <class Tile>
void do_fa3_single_prefill(bool is_causal, int N_q, int N_kv, int nhead, int nhead_k, int D,
                           float scale_log2, const __half* q_ptr, const __half* k_ptr,
                           const __half* v_ptr, __half* o_ptr, float* lse_ptr, at::Tensor& tmp,
                           hipStream_t stream) {
  const int num_q_blocks = (N_q + Tile::kBr - 1) / Tile::kBr;
  const int total_q_head_blocks = num_q_blocks * nhead;
  const dim3 block(Tile::kNumThreads);
  const size_t smem_bytes = Tile::kLDSBytes;
  const int causal_offset = is_causal ? (N_kv - N_q) : 0;

  static constexpr int kNumCUs = 304;
  int num_kv_chunks = 1;
  if (total_q_head_blocks < kNumCUs && N_kv > static_cast<int>(Tile::kBc)) {
    int max_chunks = kNumCUs / total_q_head_blocks;
    int chunk_size = std::max((N_kv + max_chunks - 1) / max_chunks, static_cast<int>(Tile::kBc));
    chunk_size = ((chunk_size + Tile::kBc - 1) / Tile::kBc) * static_cast<int>(Tile::kBc);
    num_kv_chunks = (N_kv + chunk_size - 1) / chunk_size;
  }

  if (num_kv_chunks <= 1) {
    const dim3 grid(total_q_head_blocks, 1);
    if (is_causal) {
      launch_fa3_prefill_kernel<Tile, true>(grid, stream, smem_bytes, block, o_ptr, lse_ptr, q_ptr,
                                            k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2,
                                            causal_offset, num_q_blocks, total_q_head_blocks, N_kv,
                                            1, nhead * D, 1, N_q, false);
    } else {
      launch_fa3_prefill_kernel<Tile, false>(grid, stream, smem_bytes, block, o_ptr, lse_ptr, q_ptr,
                                             k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2,
                                             causal_offset, num_q_blocks, total_q_head_blocks, N_kv,
                                             1, nhead * D, 1, N_q, false);
    }
  } else {
    int kv_chunk_size = ((N_kv + num_kv_chunks - 1) / num_kv_chunks);
    kv_chunk_size = ((kv_chunk_size + Tile::kBc - 1) / Tile::kBc) * static_cast<int>(Tile::kBc);
    num_kv_chunks = (N_kv + kv_chunk_size - 1) / kv_chunk_size;

    const int64_t tmp_o_elems = static_cast<int64_t>(N_q) * num_kv_chunks * nhead * D;
    const int64_t tmp_lse_elems = static_cast<int64_t>(N_q) * num_kv_chunks * nhead;
    const int64_t tmp_bytes_needed = tmp_o_elems * 2 + tmp_lse_elems * 4;
    TORCH_CHECK(tmp.numel() * tmp.element_size() >= tmp_bytes_needed,
                "FA3-CDNA3 split-KV: tmp buffer too small (need ", tmp_bytes_needed, " bytes)");

    auto* tmp_o_ptr = reinterpret_cast<__half*>(tmp.data_ptr());
    auto* tmp_lse_ptr = reinterpret_cast<float*>(tmp_o_ptr + tmp_o_elems);

    const dim3 grid(total_q_head_blocks, num_kv_chunks);
    const int o_stride = num_kv_chunks * nhead * D;
    const int lse_stride = num_kv_chunks * nhead;
    if (is_causal) {
      launch_fa3_prefill_kernel<Tile, true>(
          grid, stream, smem_bytes, block, tmp_o_ptr, tmp_lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv,
          nhead, nhead_k, scale_log2, causal_offset, num_q_blocks, total_q_head_blocks,
          kv_chunk_size, num_kv_chunks, o_stride, lse_stride, 1, true);
    } else {
      launch_fa3_prefill_kernel<Tile, false>(
          grid, stream, smem_bytes, block, tmp_o_ptr, tmp_lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv,
          nhead, nhead_k, scale_log2, causal_offset, num_q_blocks, total_q_head_blocks,
          kv_chunk_size, num_kv_chunks, o_stride, lse_stride, 1, true);
    }

    flashinfer::MergeStates(tmp_o_ptr, tmp_lse_ptr, o_ptr, lse_ptr,
                            static_cast<uint32_t>(num_kv_chunks), static_cast<uint32_t>(N_q),
                            static_cast<uint32_t>(nhead), static_cast<uint32_t>(D), stream);
  }
}

}  // namespace

void fa3_cdna3_single_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                              std::optional<at::Tensor> maybe_lse, bool is_causal, at::Tensor tmp) {
  TORCH_CHECK(q.dtype() == at::kHalf, "FA3-CDNA3: q must be fp16");
  TORCH_CHECK(k.dtype() == at::kHalf, "FA3-CDNA3: k must be fp16");
  TORCH_CHECK(v.dtype() == at::kHalf, "FA3-CDNA3: v must be fp16");
  TORCH_CHECK(q.dim() == 3, "FA3-CDNA3: q must be 3D [seqlen, nhead, hdim]");

  const int N_q = q.size(0);
  const int N_kv = k.size(0);
  const int nhead = q.size(1);
  const int D = q.size(2);
  const int nhead_k = k.size(1);

  TORCH_CHECK(D == 256, "FA3-CDNA3: head_dim must be 256");
  TORCH_CHECK(N_q <= 8192, "FA3-CDNA3: q_len must be <= 8192");
  TORCH_CHECK(N_kv <= 8192, "FA3-CDNA3: kv_len must be <= 8192");
  TORCH_CHECK(nhead % nhead_k == 0, "FA3-CDNA3: nhead must be divisible by nhead_k (GQA)");

  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  const float scale_s = 1.0f / sqrtf(static_cast<float>(D));
  const float scale_log2 = scale_s * 1.44269504088896340736f;

  float* lse_ptr = nullptr;
  if (maybe_lse.has_value()) {
    lse_ptr = maybe_lse.value().data_ptr<float>();
  }

  const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
  const auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
  const auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr());
  auto* o_ptr = reinterpret_cast<__half*>(o.data_ptr());

  // Both causal and non-causal use Tile128x128. Tile64x128 was tested but causes
  // 3.5x more VGPR spills (117 vs 34) due to 2x more per-thread staging VGPRs
  // from cooperative K/V loads with half the threads.
  do_fa3_single_prefill<Tile128x128>(is_causal, N_q, N_kv, nhead, nhead_k, D, scale_log2, q_ptr,
                                     k_ptr, v_ptr, o_ptr, lse_ptr, tmp, stream);
}
