// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 V7: JIT-compiled host-side entry point for the 32x32 MFMA
// prefill kernel targeting AMD MI300X (gfx942). 4 waves, 256 threads,
// kBr=128, kBc=128, d=256, 2-row-paired V for vectorized PV GEMM reads.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/cdna3/fa3_kernel.hpp>
#include <optional>

#include "pytorch_extension_utils.h"

using namespace flashinfer::cdna3;

void fa3_cdna3_single_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                              std::optional<at::Tensor> maybe_lse, bool is_causal) {
  TORCH_CHECK(q.dtype() == at::kHalf, "FA3-CDNA3: q must be fp16");
  TORCH_CHECK(k.dtype() == at::kHalf, "FA3-CDNA3: k must be fp16");
  TORCH_CHECK(v.dtype() == at::kHalf, "FA3-CDNA3: v must be fp16");
  TORCH_CHECK(q.dim() == 3, "FA3-CDNA3: q must be 3D [seqlen, nhead, hdim]");

  const int N = q.size(0);
  const int nhead = q.size(1);
  const int D = q.size(2);
  const int nhead_k = k.size(1);

  TORCH_CHECK(D == 256, "FA3-CDNA3: head_dim must be 256");
  TORCH_CHECK(N <= 8192, "FA3-CDNA3: seqlen must be <= 8192");
  TORCH_CHECK(nhead % nhead_k == 0, "FA3-CDNA3: nhead must be divisible by nhead_k (GQA)");

  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  const float scale_s = 1.0f / sqrtf(static_cast<float>(D));
  const float scale_log2 = scale_s * 1.44269504088896340736f;

  const int num_q_blocks = (N + kBr - 1) / kBr;
  const int total_blocks = num_q_blocks * nhead;

  const dim3 grid(total_blocks);
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
    fa3_cdna3_prefill_kernel<<<grid, block, smem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N, nhead, nhead_k, scale_log2, is_causal, num_q_blocks,
        total_blocks);
  } else {
    fa3_cdna3_prefill_kernel_nc<<<grid, block, smem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, N, nhead, nhead_k, scale_log2, num_q_blocks,
        total_blocks);
  }
}
