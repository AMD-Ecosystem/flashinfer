// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: Split-KV parallelism with chunked prefill for MI300X.
//
// Architecture:
//   - Double-buffered K in LDS.
//   - Double-buffered K-packed (column-major) V in LDS: V_LDS[head][seq].
//     Each MFMA V-operand is a contiguous ds_read_b64 (1 instruction).
//   - sched_group_barrier scheduling for QK GEMM.
//   - TransposedC for both QK and PV GEMMs.
//   - Scalar online softmax + in-register P repack (no LDS round-trip).
//
// LDS layout (56320 bytes, fits in 65536 per CU):
//   [0..10751]      K_LDS[0]    128 rows x 84 B/row   (double-buffered K)
//   [10752..21503]  K_LDS[1]    128 rows x 84 B/row
//   [21504..38911]  V_LDS[0]    256 cols x 68 B/col   (double-buffered V, K-packed)
//   [38912..56319]  V_LDS[1]    256 cols x 68 B/col
//
// Workgroup:
//   4 waves x 64 threads = 256 threads per workgroup
//   kBr = 128 (32 Q-rows per wave), kBc = 128, kHeadDim = 256
//   v_mfma_f32_32x32x8f16 as the compute instruction
//
// TransposedC MFMA register layout (gfx942):
//   QK GEMM (K=A, Q=B): thread t holds S^T[kv_col=mfma_32x32_row(t/32,i)][q_row=t%32]
//   PV GEMM (V=A, P=B): thread t holds O^T[head_dim=mfma_32x32_row(t/32,i)][q_row=t%32]

#pragma once

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
#include "asm_primitives.hpp"
#include "fa3_tiles.hpp"

namespace flashinfer {
namespace cdna3 {

// ===== Tile / LDS constants ==================================================

static constexpr int kBr = 128;
static constexpr int kBc = 128;
static constexpr int kHeadDim = 256;
static constexpr int kNumWaves = 4;
static constexpr int kBrLocal = kBr / kNumWaves;  // 32

// QK micro-tile: kK0=32 along head-dim
static constexpr int kK0 = 32;
static constexpr int kK_Pad = 10;
static constexpr int kK_RowStride = (kK0 + kK_Pad) * 2;  // 84 bytes (21 dwords, gcd(21,32)=1)
static constexpr int kK_LDS_Size = kBc * kK_RowStride;   // 10752 bytes per buffer
static constexpr int k0_loops = kHeadDim / kK0;          // 8

// PV micro-tile: kK1=32 along sequence dim, V K-packed (column-major) in LDS.
// V_LDS layout: V_LDS[head_dim_pos][seq_pos] -- seq positions are contiguous.
// MFMA A-operand needs 4 consecutive seq values at same head_dim -> ds_read_b64.
// Bank = (t*17 + const) % 32. gcd(17,32)=1 -> zero bank conflicts.
static constexpr int kK1 = 32;
static constexpr int k1_loops = kBc / kK1;                   // 4
static constexpr int kV_SeqPad = 2;                          // 2 fp16 padding per head column
static constexpr int kV_ColStride = (kK1 + kV_SeqPad) * 2;   // 68 bytes (17 dwords)
static constexpr int kV_LDS_Size = kHeadDim * kV_ColStride;  // 17408 bytes per buffer

// LDS layout: K[0], K[1] (double-buffered), V[0], V[1] (double-buffered)
static constexpr int kK_LDS_Base0 = 0;
static constexpr int kK_LDS_Base1 = kK_LDS_Size;                          // 10752
static constexpr int kV_LDS_Base0 = 2 * kK_LDS_Size;                      // 21504
static constexpr int kV_LDS_Base1 = 2 * kK_LDS_Size + kV_LDS_Size;        // 38912
static constexpr uint32_t kLDSBytes = 2 * kK_LDS_Size + 2 * kV_LDS_Size;  // 56320

// GEMM iteration counts per micro-tile
static constexpr int kQK_KSteps = kK0 / kMfmaK;  // 4 per k0-strip
static constexpr int kQK_MTiles = kBc / kMfmaM;  // 4 (M-tiles in TransposedC = KV-col groups)
static constexpr int kPV_KSteps = kK1 / kMfmaK;  // 4 per k1-strip
static constexpr int kPV_MTiles =
    kHeadDim / kMfmaM;  // 8 (M-tiles in TransposedC = head-dim groups)

// Register-staged load sizes
static constexpr int kK_LoadsPerThread = (kBc * kK0) / (256 * 8);       // 2
static constexpr int kV_LoadsPerThread = (kK1 * kHeadDim) / (256 * 8);  // 4

// ===== Register-staged GMEM -> LDS loads =====================================

struct k_reg_buf_t {
  uint4 data[kK_LoadsPerThread];
};

struct v_reg_buf_t {
  uint4 data[kV_LoadsPerThread];
};

__device__ __forceinline__ k_reg_buf_t cooperative_load_k_to_regs(
    const __half* __restrict__ gmem_base, int kv_row_stride, int thread_id, int valid_rows) {
  k_reg_buf_t buf;
  const uint4 zero4 = {0, 0, 0, 0};

#pragma unroll
  for (int p = 0; p < kK_LoadsPerThread; ++p) {
    int linear = (thread_id + p * 256) * 8;
    int row = linear / kK0;
    int col = linear % kK0;

    if (row < valid_rows) {
      buf.data[p] = *reinterpret_cast<const uint4*>(gmem_base + row * kv_row_stride + col);
    } else {
      buf.data[p] = zero4;
    }
  }
  return buf;
}

__device__ __forceinline__ void cooperative_store_k_to_lds(const k_reg_buf_t& buf,
                                                           char* __restrict__ smem, int thread_id,
                                                           int k_lds_base) {
#pragma unroll
  for (int p = 0; p < kK_LoadsPerThread; ++p) {
    int linear = (thread_id + p * 256) * 8;
    int row = linear / kK0;
    int col = linear % kK0;

    uint32_t lds_off = static_cast<uint32_t>(k_lds_base) +
                       static_cast<uint32_t>(row) * kK_RowStride + static_cast<uint32_t>(col) * 2;
    *reinterpret_cast<uint4*>(smem + lds_off) = buf.data[p];
  }
}

__device__ __forceinline__ v_reg_buf_t cooperative_load_v_to_regs(
    const __half* __restrict__ gmem_base, int kv_row_stride, int thread_id, int valid_rows) {
  v_reg_buf_t buf;
  const uint4 zero4 = {0, 0, 0, 0};

#pragma unroll
  for (int p = 0; p < kV_LoadsPerThread; ++p) {
    int linear = (thread_id + p * 256) * 8;
    int row = linear / kHeadDim;
    int col = linear % kHeadDim;

    if (row < valid_rows) {
      buf.data[p] = *reinterpret_cast<const uint4*>(gmem_base + row * kv_row_stride + col);
    } else {
      buf.data[p] = zero4;
    }
  }
  return buf;
}

// ===== Store V to K-packed (column-major) LDS ================================
// GMEM load is coalesced row-major (8 consecutive head_dim fp16 per uint4).
// LDS write transposes to V_LDS[head][seq] so MFMA reads are contiguous.

__device__ __forceinline__ void cooperative_store_v_to_lds(const v_reg_buf_t& buf,
                                                           char* __restrict__ smem, int thread_id,
                                                           int v_lds_base) {
#pragma unroll
  for (int p = 0; p < kV_LoadsPerThread; ++p) {
    int linear = (thread_id + p * 256) * 8;
    int seq = linear / kHeadDim;
    int head = linear % kHeadDim;

    uint32_t seq_bytes = static_cast<uint32_t>(seq) * 2;
    const uint16_t* vals = reinterpret_cast<const uint16_t*>(&buf.data[p]);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t lds_off = static_cast<uint32_t>(v_lds_base) +
                         static_cast<uint32_t>(head + i) * kV_ColStride + seq_bytes;
      *reinterpret_cast<uint16_t*>(smem + lds_off) = vals[i];
    }
  }
}

// ===== LDS operand reads =====================================================

// K operand: K[mt*32+t%32, ks*8+(t/32)*4+{0,1,2,3}]
// In TransposedC QK GEMM, K is the A-operand (src0).
__device__ __forceinline__ void load_k_operand(const char* smem, int k_lds_base, int mt, int ks,
                                               int lane_id, uint32_t* out) {
  uint32_t row = static_cast<uint32_t>(mt) * kMfmaM + (lane_id & 31);
  uint32_t col = static_cast<uint32_t>(ks) * kMfmaK + (lane_id >> 5) * 4;
  uint32_t addr = static_cast<uint32_t>(k_lds_base) + row * kK_RowStride + col * 2;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(smem + addr);
  out[0] = src[0];
  out[1] = src[1];
}

// V operand from K-packed V LDS: contiguous ds_read_b64 (1 instruction).
// 4 consecutive seq positions at same head_dim are adjacent -> single 8-byte read.
__device__ __forceinline__ void load_v_operand(const char* smem, int v_lds_base, int ks, int dt,
                                               int lane_id, uint32_t* out) {
  uint32_t head = static_cast<uint32_t>(dt) * kMfmaN + (lane_id & 31);
  uint32_t seq = static_cast<uint32_t>(ks) * kMfmaK + (lane_id >> 5) * 4;
  uint32_t addr = static_cast<uint32_t>(v_lds_base) + head * kV_ColStride + seq * 2;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(smem + addr);
  out[0] = src[0];
  out[1] = src[1];
}

// ===== TransposedC QK GEMM for one kK0=32 strip ==============================
// Issues MTiles=4 x KSteps=4 = 16 MFMA instructions per call.
// K is src0 (A-operand from LDS), Q is src1 (B-operand from registers).

__device__ __forceinline__ void qk_gemm_k0(const fp16_reg_tile<kBrLocal, kHeadDim>& Q_reg,
                                           const char* smem, int k_lds_base,
                                           fp32_s_tile<kBrLocal, kBc>& S_acc, int strip_idx,
                                           int lane_id) {
  using namespace asm_primitives;

  uint32_t k_cur[2], k_next[2];
  const int ks_global_base = strip_idx * kQK_KSteps;

  load_k_operand(smem, k_lds_base, 0, 0, lane_id, k_cur);

  static constexpr int kTotalIters = kQK_KSteps * kQK_MTiles;  // 16

#pragma unroll
  for (int iter = 0; iter < kTotalIters; ++iter) {
    const int ks = iter / kQK_MTiles;
    const int mt = iter % kQK_MTiles;

    if (iter + 1 < kTotalIters) {
      const int next_ks = (iter + 1) / kQK_MTiles;
      const int next_mt = (iter + 1) % kQK_MTiles;
      load_k_operand(smem, k_lds_base, next_mt, next_ks, lane_id, k_next);
    }

    mfma_f32_32x32x8_f16_vec(S_acc.vec(mt),
                             k_cur,                             // K = A (src0)
                             Q_reg.frag(ks_global_base + ks));  // Q = B (src1)

    k_cur[0] = k_next[0];
    k_cur[1] = k_next[1];
  }
}

// ===== QK GEMM scheduling: sched_group_barrier pattern =======================

__device__ __forceinline__ void schedule_gemm0() {
  using namespace asm_primitives;
  sched_group_barrier<0x100, 2>();  // 2 DS reads
  sched_group_barrier<0x008, 2>();  // 2 MFMAs
  sched_group_barrier<0x100, 1>();  // 1 DS read
  sched_group_barrier<0x008, 2>();  // 2 MFMAs
  sched_group_barrier<0x100, 1>();  // 1 DS read
  sched_group_barrier<0x008, 4>();  // 4 MFMAs
}

// ===== TransposedC PV GEMM for one kK1=32 strip ==============================
// V is src0 (A-operand from LDS, K-packed column-major), P is src1 (B-operand).
// Software-pipelined: preload first V operand, then overlap load_v(next)
// with mfma(current) to hide DS read latency.

__device__ __forceinline__ void pv_gemm_k1(const fp16_p_tile<kBrLocal, kBc>& P_f16,
                                           const char* smem, int v_lds_base,
                                           fp32_acc_tile<kBrLocal, kHeadDim>& O_acc, int strip_idx,
                                           int lane_id) {
  using namespace asm_primitives;

  uint32_t v_cur[2], v_next[2];
  const int kf_base = strip_idx * kPV_KSteps;

  load_v_operand(smem, v_lds_base, 0, 0, lane_id, v_cur);

  static constexpr int kTotalIters = kPV_KSteps * kPV_MTiles;  // 32

#pragma unroll
  for (int iter = 0; iter < kTotalIters; ++iter) {
    const int ks = iter / kPV_MTiles;
    const int dt = iter % kPV_MTiles;
    const uint32_t* p_frag = P_f16.frag(kf_base + ks);

    if (iter + 1 < kTotalIters) {
      const int next_ks = (iter + 1) / kPV_MTiles;
      const int next_dt = (iter + 1) % kPV_MTiles;
      load_v_operand(smem, v_lds_base, next_ks, next_dt, lane_id, v_next);
    }

    mfma_f32_32x32x8_f16_vec(O_acc.vec(dt),
                             v_cur,    // V = A (src0)
                             p_frag);  // P = B (src1)

    v_cur[0] = v_next[0];
    v_cur[1] = v_next[1];
  }
}

// ===== TransposedC Online Softmax ============================================
// Each thread handles exactly one Q-row (q_row = t%32).
// All 4*16=64 S values in the thread's registers belong to that single Q-row.
// Only 1 cross-block shuffle needed.

template <int D>
__device__ __forceinline__ void online_softmax(fp32_s_tile<kBrLocal, kBc>& S,
                                               fp32_acc_tile<kBrLocal, D>& O_acc, float& row_max,
                                               float& row_sum, float scale_log2) {
  using namespace asm_primitives;
  static constexpr int kMT = kBc / kMfmaM;  // 4

  float s_max = -3.402823466e+38f;
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      s_max = fmaxf(s_max, S.vec(mt)[i]);
    }
  }
  // Cross-block reduction: combine block 0 and block 1 (same q_row, different kv_cols)
  s_max = fmaxf(s_max, __shfl_xor(s_max, 32, 64));

  float new_max = fmaxf(row_max, s_max);

  // First tile: O_acc and row_sum are still zero; skip rescaling.
  static constexpr int kDBlks = fp32_acc_tile<kBrLocal, D>::kNumDBlks;
  if (row_max > -3.0e+38f) {
    float scale = fast_exp2((row_max - new_max) * scale_log2);
#pragma unroll
    for (int d = 0; d < kDBlks; ++d) {
#pragma unroll
      for (int i = 0; i < kMfmaOutRegs; ++i) {
        O_acc.vec(d)[i] *= scale;
      }
    }
    row_sum *= scale;
  }

#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      S.vec(mt)[i] = fast_exp2((S.vec(mt)[i] - new_max) * scale_log2);
    }
  }

  float local_sum = 0.f;
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      local_sum += S.vec(mt)[i];
    }
  }
  local_sum += __shfl_xor(local_sum, 32, 64);
  row_sum += local_sum;

  row_max = new_max;
}

// ===== In-register P repacking: TransposedC fp32 S -> fp16 P fragments =======
// The TransposedC S_acc layout directly maps to P A-operand fragments:
//   For M-tile mt, register group [ks_local*4..ks_local*4+3] holds the 4 kv_cols
//   needed by PV GEMM K-step (mt*4 + ks_local).

__device__ __forceinline__ void p_register_repack(const fp32_s_tile<kBrLocal, kBc>& S_acc,
                                                  fp16_p_tile<kBrLocal, kBc>& P_f16) {
  static constexpr int kMT = kBc / kMfmaM;               // 4
  static constexpr int kKLocalSteps = kMfmaOutRegs / 4;  // 4

#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int ks_local = 0; ks_local < kKLocalSteps; ++ks_local) {
      int kf = mt * kKLocalSteps + ks_local;
      float v0 = S_acc.vec(mt)[ks_local * 4 + 0];
      float v1 = S_acc.vec(mt)[ks_local * 4 + 1];
      float v2 = S_acc.vec(mt)[ks_local * 4 + 2];
      float v3 = S_acc.vec(mt)[ks_local * 4 + 3];

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
      auto pk01 = __builtin_amdgcn_cvt_pkrtz(v0, v1);
      auto pk23 = __builtin_amdgcn_cvt_pkrtz(v2, v3);
      __builtin_memcpy(&P_f16.frag(kf)[0], &pk01, sizeof(uint32_t));
      __builtin_memcpy(&P_f16.frag(kf)[1], &pk23, sizeof(uint32_t));
#else
      P_f16.frag(kf)[0] = 0;
      P_f16.frag(kf)[1] = 0;
#endif
    }
  }
}

// ===== Causal + OOB masking (TransposedC layout) =============================
// Right-aligned causal: mask when kv_col > q_row + causal_offset, where
// causal_offset = N_kv - N_q. Fused with OOB mask (kv_col >= chunk_kv_len) in
// a single loop with a precomputed per-lane threshold to reduce live-range
// span and avoid the heavy `if (!skip_tile)` guard on compute stages.
// `causal_lane_base` = wave_q_start + (lane_id & 31) + causal_offset - kv_chunk_start
// is loop-invariant (hoisted to the pipeline caller for the causal template).

template <bool IsCausal>
__device__ __forceinline__ void apply_masks(fp32_s_tile<kBrLocal, kBc>& S, int kv_start_local,
                                            int chunk_kv_len, int causal_lane_base, int lane_id) {
  using namespace asm_primitives;
  static constexpr int kMT = kBc / kMfmaM;
  const int oob_limit = chunk_kv_len - 1 - kv_start_local;
  int limit;
  if constexpr (IsCausal) {
    limit = min(causal_lane_base - kv_start_local, oob_limit);
  } else {
    limit = oob_limit;
  }

#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      int kv_col = mt * kMfmaM + mfma_32x32_row(lane_id >> 5, i);
      if (kv_col > limit) {
        S.vec(mt)[i] = -3.402823466e+38f;
      }
    }
  }
}

// ===== Pipeline arguments ====================================================

struct FA3CDNA3PipelineArgs {
  const __half* Q;
  const __half* K;
  const __half* V;
  int N_q;
  int N_kv;
  int nhead;
  int nhead_k;
  float scale_log2;
  int causal_offset;  // N_kv - N_q for right-aligned causal; 0 for non-causal
  int q_block;
  int head_idx;
  int head_idx_k;
  int kv_chunk_start;
  int kv_chunk_end;
};

// One KV tile: QK GEMM, softmax, PV GEMM. IsCausal selects mask path.
// NOTE: we do NOT branch on a runtime `skip_tile` flag. If every element in
// the tile is masked to -inf, softmax produces P=0 and PV adds 0 to O_acc, so
// the tile is mathematically a no-op. Always executing the MFMAs keeps the
// codegen straight-line, which is critical for the causal template's VGPR
// pressure (eliminating the skip_tile branch removed ~20 spills).

template <int D, bool IsCausal>
__device__ __forceinline__ void process_kv_tile(const fp16_reg_tile<kBrLocal, D>& Q_reg,
                                                fp32_acc_tile<kBrLocal, D>& O_acc, float& row_max,
                                                float& row_sum, const FA3CDNA3PipelineArgs& args,
                                                char* smem, int j, const __half* k_head,
                                                const __half* v_head, int kv_row_stride,
                                                int chunk_kv_len, int causal_lane_base, int lane_id,
                                                int thread_id) {
  using namespace asm_primitives;
  constexpr int k_bases[2] = {kK_LDS_Base0, kK_LDS_Base1};

  const int kv_start_local = j * kBc;
  const int kv_valid_rows = min(kBc, chunk_kv_len - kv_start_local);
  const __half* k_tile_base = k_head + kv_start_local * kv_row_stride;

  // --- Stage 1: QK GEMM -- double-buffered K in LDS ---
  fp32_s_tile<kBrLocal, kBc> S_acc;
  S_acc.zero();

  k_reg_buf_t k_regs =
      cooperative_load_k_to_regs(k_tile_base, kv_row_stride, thread_id, kv_valid_rows);
  s_waitcnt_vmcnt(0);
  cooperative_store_k_to_lds(k_regs, smem, thread_id, k_bases[0]);

  if (k0_loops > 1) {
    k_regs = cooperative_load_k_to_regs(k_tile_base + kK0, kv_row_stride, thread_id, kv_valid_rows);
  }

#pragma unroll
  for (int ik = 0; ik < k0_loops - 2; ++ik) {
    int rd = ik & 1;
    int wr = 1 - rd;
    s_barrier();
    qk_gemm_k0(Q_reg, smem, k_bases[rd], S_acc, ik, lane_id);
    schedule_gemm0();
    s_waitcnt_vmcnt(0);
    cooperative_store_k_to_lds(k_regs, smem, thread_id, k_bases[wr]);
    k_regs = cooperative_load_k_to_regs(k_tile_base + (ik + 2) * kK0, kv_row_stride, thread_id,
                                        kv_valid_rows);
  }

  const int v0_valid = min(kK1, chunk_kv_len - kv_start_local);
  v_reg_buf_t v_regs = cooperative_load_v_to_regs(v_head + kv_start_local * kv_row_stride,
                                                  kv_row_stride, thread_id, max(v0_valid, 0));

  {
    const int rd6 = (k0_loops - 2) & 1;
    const int wr7 = 1 - rd6;

    s_barrier();
    qk_gemm_k0(Q_reg, smem, k_bases[rd6], S_acc, k0_loops - 2, lane_id);
    schedule_gemm0();

    s_waitcnt_vmcnt(0);
    cooperative_store_k_to_lds(k_regs, smem, thread_id, k_bases[wr7]);

    s_barrier();
    qk_gemm_k0(Q_reg, smem, k_bases[wr7], S_acc, k0_loops - 1, lane_id);
    schedule_gemm0();
  }

  // --- Stage 2: Fused mask + softmax + P repack ---
  apply_masks<IsCausal>(S_acc, kv_start_local, chunk_kv_len, causal_lane_base, lane_id);

  online_softmax<D>(S_acc, O_acc, row_max, row_sum, args.scale_log2);

  fp16_p_tile<kBrLocal, kBc> P_f16;
  p_register_repack(S_acc, P_f16);

  // --- Stage 3: PV GEMM -- double-buffered K-packed V in LDS ---
  constexpr int v_bases[2] = {kV_LDS_Base0, kV_LDS_Base1};

  s_waitcnt_vmcnt(0);
  cooperative_store_v_to_lds(v_regs, smem, thread_id, v_bases[0]);

  if (k1_loops > 1) {
    const int v1_off = kv_start_local + kK1;
    const int v1_valid = min(kK1, chunk_kv_len - v1_off);
    v_regs = cooperative_load_v_to_regs(v_head + v1_off * kv_row_stride, kv_row_stride, thread_id,
                                        max(v1_valid, 0));
  }

  s_barrier();
  pv_gemm_k1(P_f16, smem, v_bases[0], O_acc, 0, lane_id);

#pragma unroll
  for (int iv = 1; iv < k1_loops; ++iv) {
    const int buf = iv & 1;

    s_waitcnt_vmcnt(0);
    cooperative_store_v_to_lds(v_regs, smem, thread_id, v_bases[buf]);

    if (iv + 1 < k1_loops) {
      const int next_v_off = kv_start_local + (iv + 1) * kK1;
      const int next_v_valid = min(kK1, chunk_kv_len - next_v_off);
      v_regs = cooperative_load_v_to_regs(v_head + next_v_off * kv_row_stride, kv_row_stride,
                                          thread_id, max(next_v_valid, 0));
    }

    s_barrier();
    pv_gemm_k1(P_f16, smem, v_bases[buf], O_acc, iv, lane_id);
  }
}

// Causal: full tiles first (no mask), then edge tiles. Non-causal: single phase.

template <int D, bool IsCausal>
__device__ void run_fa3_cdna3_pipeline(FA3CDNA3PipelineArgs args, char* smem,
                                       fp32_acc_tile<kBrLocal, D>& O_acc, float& row_max_out,
                                       float& row_sum_out) {
  using namespace asm_primitives;

  const int lane_id = threadIdx.x % kWaveSize;
  const int wave_id = threadIdx.x / kWaveSize;
  const int thread_id = threadIdx.x;

  const int wave_q_start = args.q_block * kBr + wave_id * kBrLocal;
  const int kv_row_stride = args.nhead_k * D;

  // Load Q into registers (persistent across all KV-block iterations)
  fp16_reg_tile<kBrLocal, D> Q_reg;
  {
    const __half* q_base = args.Q + wave_q_start * args.nhead * D + args.head_idx * D;

#pragma unroll
    for (int ks = 0; ks < fp16_reg_tile<kBrLocal, D>::kNumKSteps; ++ks) {
      int row = lane_id & 31;
      int col = ks * kMfmaK + (lane_id >> 5) * 4;

      if (wave_q_start + row < args.N_q) {
        const uint32_t* src =
            reinterpret_cast<const uint32_t*>(q_base + row * args.nhead * D + col);
        Q_reg.data[ks][0] = src[0];
        Q_reg.data[ks][1] = src[1];
      } else {
        Q_reg.data[ks][0] = 0;
        Q_reg.data[ks][1] = 0;
      }
    }
  }

  O_acc.zero();
  float row_max = -3.402823466e+38f;
  float row_sum = 0.0f;

  const int chunk_kv_len = args.kv_chunk_end - args.kv_chunk_start;
  const __half* k_head = args.K + args.head_idx_k * D + args.kv_chunk_start * kv_row_stride;
  const __half* v_head = args.V + args.head_idx_k * D + args.kv_chunk_start * kv_row_stride;

  // Hoisted loop-invariant: per-lane causal threshold relative to chunk start.
  // Only used inside process_kv_tile<D, true>; non-causal path ignores it.
  const int causal_lane_base =
      IsCausal ? (wave_q_start + (lane_id & 31) + args.causal_offset - args.kv_chunk_start) : 0;

  const int T = (chunk_kv_len + kBc - 1) / kBc;
  int max_j = T;
  if constexpr (IsCausal) {
    int causal_end = args.q_block * kBr + kBr + args.causal_offset + kBc - 1 - args.kv_chunk_start;
    max_j = min(T, max(causal_end, 0) / kBc);
    max_j = max(max_j, 0);
  }

  int phase1_end;
  if constexpr (IsCausal) {
    int full_end = wave_q_start + args.causal_offset - args.kv_chunk_start + 1;
    phase1_end = max(0, min(max_j, full_end / static_cast<int>(kBc)));
  } else {
    phase1_end = max_j;
  }

  for (int j = 0; j < phase1_end; ++j) {
    process_kv_tile<D, false>(Q_reg, O_acc, row_max, row_sum, args, smem, j, k_head, v_head,
                              kv_row_stride, chunk_kv_len, causal_lane_base, lane_id, thread_id);
  }

  if constexpr (IsCausal) {
    for (int j = phase1_end; j < max_j; ++j) {
      process_kv_tile<D, true>(Q_reg, O_acc, row_max, row_sum, args, smem, j, k_head, v_head,
                               kv_row_stride, chunk_kv_len, causal_lane_base, lane_id, thread_id);
    }
  }

  row_max_out = row_max;
  row_sum_out = row_sum;
}

}  // namespace cdna3
}  // namespace flashinfer
