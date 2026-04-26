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

// Cached per-device CU count. MI308X CPX devices expose 20 CUs each; the
// previous hard-coded 304 (MI300X/MI325X full chip) caused a 6-13x grid
// over-subscription on this hardware and forced split-KV to fire on every
// shape, adding a MergeStates kernel per call.
inline int get_device_cu_count() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return 80;  // safe fallback
  static constexpr int kMaxDevices = 64;
  static int cached[kMaxDevices] = {0};
  if (dev < 0 || dev >= kMaxDevices) {
    int n = 0;
    if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
        n <= 0)
      return 80;
    return n;
  }
  int v = cached[dev];
  if (v == 0) {
    int n = 0;
    if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
        n <= 0)
      n = 80;
    cached[dev] = n;
    v = n;
  }
  return v;
}

template <class Tile, bool IsCausal, int PairSize>
void launch_fa3_prefill_kernel(dim3 grid, hipStream_t stream, size_t smem_bytes, dim3 block,
                               __half* O, float* LSE, const __half* q_ptr, const __half* k_ptr,
                               const __half* v_ptr, int N_q, int N_kv, int nhead, int nhead_k,
                               float scale_log2, int causal_offset, int num_q_blocks,
                               int total_q_head_blocks, int kv_chunk_size, int num_kv_chunks,
                               int o_row_stride, int lse_row_stride, int lse_head_stride,
                               bool base2_lse) {
  fa3_cdna3_prefill_kernel_impl<Tile, IsCausal, PairSize><<<grid, block, smem_bytes, stream>>>(
      O, LSE, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead, nhead_k, scale_log2, causal_offset,
      num_q_blocks, total_q_head_blocks, kv_chunk_size, num_kv_chunks, o_row_stride, lse_row_stride,
      lse_head_stride, base2_lse);
}

template <class Tile, int PairSize>
void do_fa3_single_prefill(bool is_causal, int N_q, int N_kv, int nhead, int nhead_k, int D,
                           float scale_log2, const __half* q_ptr, const __half* k_ptr,
                           const __half* v_ptr, __half* o_ptr, float* lse_ptr, at::Tensor& tmp,
                           hipStream_t stream) {
  const int num_q_blocks = (N_q + Tile::kBr - 1) / Tile::kBr;
  // Pair mode (PairSize=2): one CTA per (q_block, k_head, gqa_pair).
  // Single mode (PairSize=1): legacy one-CTA-per-Q-head.
  const int total_q_head_blocks = (num_q_blocks * nhead) / PairSize;
  const dim3 block(Tile::kNumThreads);
  const size_t smem_bytes = Tile::kLDSBytes;
  const int causal_offset = is_causal ? (N_kv - N_q) : 0;

  // Split-KV gate: pick the smallest num_kv_chunks that meaningfully reduces
  // the per-CU wallclock cost vs. running un-split. With 32 (q_block, head_q)
  // blocks on a 20-CU CPX device, no split = ceil(32/20) = 2 wave passes
  // carrying full per-CTA work; split=3 → ceil(96/20)/3 = 1.67 passes worth;
  // split=5 → ceil(160/20)/5 = 1.60. Smaller chunks also mean smaller
  // MergeStates work, so we pick the smallest N that beats no-split by ≥5%.
  const int num_cus = get_device_cu_count();
  static constexpr int kMinChunkBlocks = 4;
  const int min_chunk_size = kMinChunkBlocks * static_cast<int>(Tile::kBc);
  int num_kv_chunks = 1;
  if (N_kv >= 2 * min_chunk_size) {
    const int no_split_batches = (total_q_head_blocks + num_cus - 1) / num_cus;
    double best_cost = static_cast<double>(no_split_batches);
    int best_chunks = 1;
    const int max_possible = std::min(N_kv / min_chunk_size, 8);
    for (int n = 2; n <= max_possible; ++n) {
      const int total_ctas = total_q_head_blocks * n;
      const int batches = (total_ctas + num_cus - 1) / num_cus;
      const double cost = static_cast<double>(batches) / static_cast<double>(n);
      if (cost < best_cost * 0.99) {
        best_cost = cost;
        best_chunks = n;
      }
    }
    num_kv_chunks = best_chunks;
  }

  if (num_kv_chunks <= 1) {
    const dim3 grid(total_q_head_blocks, 1);
    if (is_causal) {
      launch_fa3_prefill_kernel<Tile, true, PairSize>(
          grid, stream, smem_bytes, block, o_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead,
          nhead_k, scale_log2, causal_offset, num_q_blocks, total_q_head_blocks, N_kv, 1,
          nhead * D, 1, N_q, false);
    } else {
      launch_fa3_prefill_kernel<Tile, false, PairSize>(
          grid, stream, smem_bytes, block, o_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv, nhead,
          nhead_k, scale_log2, causal_offset, num_q_blocks, total_q_head_blocks, N_kv, 1,
          nhead * D, 1, N_q, false);
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
      launch_fa3_prefill_kernel<Tile, true, PairSize>(
          grid, stream, smem_bytes, block, tmp_o_ptr, tmp_lse_ptr, q_ptr, k_ptr, v_ptr, N_q, N_kv,
          nhead, nhead_k, scale_log2, causal_offset, num_q_blocks, total_q_head_blocks,
          kv_chunk_size, num_kv_chunks, o_stride, lse_stride, 1, true);
    } else {
      launch_fa3_prefill_kernel<Tile, false, PairSize>(
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
  //
  // Phase 2b-7 B.1 (pair-mode, PairSize=2) was implemented and regressed ~2x
  // wallclock on every shape due to register pressure: Q_reg[2]+O_acc[2]+
  // transient S_acc[2] busts the gfx942 512-entry unified VGPR+AGPR file,
  // causing 909-1237 VGPR spills per kernel (vs 13-27 in single-head). The
  // HBM win from sharing K/V loads is swamped by scratch-memory spill traffic.
  // The PairSize=2 code path is kept in source for future iteration but is
  // not selected at runtime. See project_fa3_phase2b7b1_pair_falsified.md.
  do_fa3_single_prefill<Tile128x128, 1>(is_causal, N_q, N_kv, nhead, nhead_k, D, scale_log2, q_ptr,
                                        k_ptr, v_ptr, o_ptr, lse_ptr, tmp, stream);
}
