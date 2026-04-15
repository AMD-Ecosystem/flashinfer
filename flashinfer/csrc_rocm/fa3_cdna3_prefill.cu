// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: JIT-compiled host-side entry point with split-KV parallelism
// targeting AMD MI300X (gfx942). 4 waves, 256 threads,
// kBr=128, kBc=128, d=256, K-packed V LDS for vectorized PV GEMM reads.
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

  const int num_q_blocks = (N_q + kBr - 1) / kBr;
  const int total_q_head_blocks = num_q_blocks * nhead;

  const dim3 block(kNumWaves * kWaveSize);
  const size_t smem_bytes = kLDSBytes;

  float* lse_ptr = nullptr;
  if (maybe_lse.has_value()) {
    lse_ptr = maybe_lse.value().data_ptr<float>();
  }

  const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
  const auto* k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
  const auto* v_ptr = reinterpret_cast<const __half*>(v.data_ptr());
  auto* o_ptr = reinterpret_cast<__half*>(o.data_ptr());

  if (is_causal) {
    const dim3 grid(total_q_head_blocks);
    fa3_cdna3_prefill_kernel<<<grid, block, smem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N_q, N_kv, nhead, nhead_k, scale_log2, is_causal,
        num_q_blocks, total_q_head_blocks);
    return;
  }

  // Non-causal: determine split-KV factor based on CU utilization
  static constexpr int kNumCUs = 304;  // MI300X
  int num_kv_chunks = 1;
  if (total_q_head_blocks < kNumCUs && N_kv > static_cast<int>(kBc)) {
    int max_chunks = kNumCUs / total_q_head_blocks;
    int chunk_size = std::max((N_kv + max_chunks - 1) / max_chunks, static_cast<int>(kBc));
    // Round up to kBc for LDS tile alignment
    chunk_size = ((chunk_size + kBc - 1) / kBc) * kBc;
    num_kv_chunks = (N_kv + chunk_size - 1) / chunk_size;
  }

  if (num_kv_chunks <= 1) {
    // No split -- direct write to output, FlashInfer LSE layout [nhead, N_q]
    const dim3 grid(total_q_head_blocks, 1);
    fa3_cdna3_prefill_kernel_nc<<<grid, block, smem_bytes, stream>>>(
        o_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2, num_q_blocks,
        total_q_head_blocks,
        /*kv_chunk_size=*/N_kv, /*num_kv_chunks=*/1,
        /*o_row_stride=*/nhead * D,
        /*lse_row_stride=*/1, /*lse_head_stride=*/N_q,
        /*base2_lse=*/false);
  } else {
    int kv_chunk_size = ((N_kv + num_kv_chunks - 1) / num_kv_chunks);
    kv_chunk_size = ((kv_chunk_size + kBc - 1) / kBc) * kBc;
    num_kv_chunks = (N_kv + kv_chunk_size - 1) / kv_chunk_size;

    // tmp layout: [N_q, num_kv_chunks, nhead, D] fp16 + [N_q, num_kv_chunks, nhead] fp32
    const int64_t tmp_o_elems = static_cast<int64_t>(N_q) * num_kv_chunks * nhead * D;
    const int64_t tmp_lse_elems = static_cast<int64_t>(N_q) * num_kv_chunks * nhead;
    const int64_t tmp_bytes_needed = tmp_o_elems * 2 + tmp_lse_elems * 4;
    TORCH_CHECK(tmp.numel() * tmp.element_size() >= tmp_bytes_needed,
                "FA3-CDNA3 split-KV: tmp buffer too small (need ", tmp_bytes_needed, " bytes)");

    auto* tmp_o_ptr = reinterpret_cast<__half*>(tmp.data_ptr());
    auto* tmp_lse_ptr = reinterpret_cast<float*>(tmp_o_ptr + tmp_o_elems);

    const dim3 grid(total_q_head_blocks, num_kv_chunks);
    fa3_cdna3_prefill_kernel_nc<<<grid, block, smem_bytes, stream>>>(
        tmp_o_ptr, tmp_lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2,
        num_q_blocks, total_q_head_blocks, kv_chunk_size, num_kv_chunks,
        /*o_row_stride=*/num_kv_chunks * nhead * D,
        /*lse_row_stride=*/num_kv_chunks * nhead,
        /*lse_head_stride=*/1,
        /*base2_lse=*/true);

    // Merge partial results: MergeStates expects v=[N_q, num_chunks, nhead, D], s=[N_q, num_chunks,
    // nhead]
    flashinfer::MergeStates(tmp_o_ptr, tmp_lse_ptr, o_ptr, lse_ptr,
                            static_cast<uint32_t>(num_kv_chunks), static_cast<uint32_t>(N_q),
                            static_cast<uint32_t>(nhead), static_cast<uint32_t>(D), stream);
  }
}
