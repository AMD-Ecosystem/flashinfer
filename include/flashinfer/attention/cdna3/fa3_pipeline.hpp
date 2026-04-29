// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 prefill pipeline for MI300X (gfx942).
//
// Architecture:
//   - Double-buffered K in LDS (kKPack=8 layout, ds_read_b128 paired loads).
//   - Double-buffered V in LDS (kKPack=4 layout, ds_read_b64 per PV K-step).
//   - sched_group_barrier scheduling for QK GEMM.
//   - TransposedC for both QK and PV GEMMs.
//   - Scalar online softmax + in-register P repack (no LDS round-trip).
//
// Workgroup: 4 waves × 64 threads = 256 threads. kBr = 128 (32 Q-rows per
// wave), kBc = 128, kHeadDim = 256. Compute instruction: v_mfma_f32_32x32x8f16.
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

// ===== Tile configuration ====================================================
// Workgroup tile shape (kBr × kBc × kHeadDim) and wave count, plus all derived
// LDS layout constants. The kernel currently uses Tile128x128 only.

template <int Br_, int Bc_, int HeadDim_, int NumWaves_>
struct TileConfig {
  static constexpr int kBr = Br_;
  static constexpr int kBc = Bc_;
  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kNumWaves = NumWaves_;
  static constexpr int kBrLocal = kBr / kNumWaves;
  static constexpr int kNumThreads = kNumWaves * kWaveSize;

  // QK micro-tile: kK0=32 along head-dim
  static constexpr int kK0 = 32;
  // kKPack=8 K LDS layout. Row content per seq position (one 32-head-dim strip):
  //   bytes  0..31  = lane(l_hi=0): 4 ks slots × 8 B (K[row, 0..3, 8..11, 16..19, 24..27])
  //   bytes 32..63  = lane(l_hi=1): 4 ks slots × 8 B (K[row, 4..7, 12..15, 20..23, 28..31])
  //   bytes 64..79  = pad, row stride 80 B = 20 dwords. 16-byte-aligned per row
  //   so ds_read_b128 is well-defined; gcd(20,32)=4 → 4-way bank conflict
  //   (CK accepts the same with their +kKPack pad for hdim=256).
  // Read addr per lane: row * 80 + (lane>>5)*32 + ks*8. ks-paired b128 covers
  // ks_base..ks_base+1 in 16 contiguous bytes per lane.
  // Write side: each thread's gmem uint4 (8 fp16 = K[row, c..c+7]) splits into
  // two ds_write_b64: K[row, c..c+3] → l_hi=0 region, K[row, c+4..c+7] → l_hi=1 region.
  static constexpr int kK_RowStride = 80;
  static constexpr int kK_LDS_Size = kBc * kK_RowStride;
  static constexpr int k0_loops = kHeadDim / kK0;

  // PV micro-tile: kK1=32 along seq dim.
  static constexpr int kK1 = 32;
  static constexpr int k1_loops = kBc / kK1;
  // kKPack=4 V LDS: V_LDS[seq_group=0..7][head=0..255][4_seqs] — 4 consecutive
  // seqs packed contiguously per slot. PV reads are 1 × ds_read_b64 per K-step;
  // writes are ds_write_b64 after an intra-VGPR transpose (the wave-slab gmem
  // load lays out 4 consecutive seqs across one lane's 4 chunks, so the
  // transpose is pure VGPR byte-shuffle, no DS_PERMUTE). Per-row pad of one
  // slot (8 B) per 16 heads makes the WRITE per-lane delta 64+8=72 B →
  // gcd(72/4, 32)=2 banks → 2-way conflict (vs 16-way unpadded).
  static constexpr int kV_HeadStride = 8;  // 4 fp16
  static constexpr int kV_RowHeadCount = 16;
  static constexpr int kV_RowPadBytes = 8;
  static constexpr int kV_RowStrideBytes =
      kV_RowHeadCount * kV_HeadStride + kV_RowPadBytes;             // 136 B
  static constexpr int kV_GroupStride =
      (kHeadDim / kV_RowHeadCount) * kV_RowStrideBytes;             // 2176 B
  static constexpr int kV_LDS_Size = (kK1 / 4) * kV_GroupStride;    // 17408 B

  // LDS layout: K[0], K[1] (double-buffered), V[0], V[1] (double-buffered)
  static constexpr int kK_LDS_Base0 = 0;
  static constexpr int kK_LDS_Base1 = kK_LDS_Size;
  static constexpr int kV_LDS_Base0 = 2 * kK_LDS_Size;
  static constexpr int kV_LDS_Base1 = 2 * kK_LDS_Size + kV_LDS_Size;
  static constexpr uint32_t kLDSBytes = 2 * kK_LDS_Size + 2 * kV_LDS_Size;

  // GEMM iteration counts
  static constexpr int kQK_KSteps = kK0 / kMfmaK;
  static constexpr int kQK_MTiles = kBc / kMfmaM;
  static constexpr int kPV_KSteps = kK1 / kMfmaK;
  static constexpr int kPV_MTiles = kHeadDim / kMfmaM;

  // Register-staged cooperative load sizes (per thread)
  static constexpr int kK_LoadsPerThread = (kBc * kK0) / (kNumThreads * 8);
  static constexpr int kV_LoadsPerThread = (kK1 * kHeadDim) / (kNumThreads * 8);
};

using Tile128x128 = TileConfig<128, 128, 256, 4>;  // 4 waves, 256 threads

static_assert(Tile128x128::kBrLocal == 32, "");

// ===== Register-staged GMEM -> LDS loads (templated on TileConfig) ===========

template <class Tile>
struct k_reg_buf_t {
  uint4 data[Tile::kK_LoadsPerThread];
};

template <class Tile>
struct v_reg_buf_t {
  uint4 data[Tile::kV_LoadsPerThread];
};

template <class Tile>
__device__ __forceinline__ k_reg_buf_t<Tile> cooperative_load_k_to_regs(
    const __half* __restrict__ gmem_base, int kv_row_stride, int thread_id, int valid_rows) {
  k_reg_buf_t<Tile> buf;
  const uint4 zero4 = {0, 0, 0, 0};

#pragma unroll
  for (int p = 0; p < Tile::kK_LoadsPerThread; ++p) {
    int linear = (thread_id + p * Tile::kNumThreads) * 8;
    int row = linear / Tile::kK0;
    int col = linear % Tile::kK0;

    if (row < valid_rows) {
      buf.data[p] = *reinterpret_cast<const uint4*>(gmem_base + row * kv_row_stride + col);
    } else {
      buf.data[p] = zero4;
    }
  }
  return buf;
}

template <class Tile>
__device__ __forceinline__ void cooperative_store_k_to_lds(const k_reg_buf_t<Tile>& buf,
                                                           char* __restrict__ smem, int thread_id,
                                                           int k_lds_base) {
  // kKPack=8: gmem chunk K[row, c..c+7] (8 fp16 per thread, c ∈ {0,8,16,24})
  // splits into two ds_write_b64. ks = c/8 picks the slot in each lane class.
  //   K[row, c..c+3]  → l_hi=0 region, byte off = ks*8
  //   K[row, c+4..c+7]→ l_hi=1 region, byte off = 32 + ks*8
#pragma unroll
  for (int p = 0; p < Tile::kK_LoadsPerThread; ++p) {
    int linear = (thread_id + p * Tile::kNumThreads) * 8;
    int row = linear / Tile::kK0;
    int col = linear % Tile::kK0;
    int ks = col >> 3;  // col / 8

    uint32_t row_base = static_cast<uint32_t>(k_lds_base) +
                        static_cast<uint32_t>(row) * Tile::kK_RowStride;
    uint32_t off_lo = row_base + static_cast<uint32_t>(ks) * 8u;
    uint32_t off_hi = off_lo + 32u;

    uint2 lo = make_uint2(buf.data[p].x, buf.data[p].y);
    uint2 hi = make_uint2(buf.data[p].z, buf.data[p].w);
    *reinterpret_cast<uint2*>(smem + off_lo) = lo;
    *reinterpret_cast<uint2*>(smem + off_hi) = hi;
  }
}

template <class Tile>
__device__ __forceinline__ v_reg_buf_t<Tile> cooperative_load_v_to_regs(
    const __half* __restrict__ gmem_base, int kv_row_stride, int thread_id, int valid_rows) {
  v_reg_buf_t<Tile> buf;
  const uint4 zero4 = {0, 0, 0, 0};

  // Wave-slab load: wave w covers seqs [w*8, w*8+8). Each lane holds 4 chunks
  // at consecutive seqs of one head_block, so the kKPack=4 LDS store needs only
  // an intra-VGPR transpose (no cross-lane shuffle).
  //   chunk p of lane (w, l_hi, l_lo): seq = w*8 + l_hi*4 + p,
  //                                    head_start = l_lo*8 (8 fp16 = 1 uint4)
  static_assert(Tile::kNumWaves == 4 && Tile::kK1 == 32 && Tile::kHeadDim == 256,
                "kKPack=4 V LDS layout requires Tile128x128 shape");
  const int wave_id = thread_id / kWaveSize;
  const int lane = thread_id % kWaveSize;
  const int l_hi = lane >> 5;
  const int l_lo = lane & 31;
#pragma unroll
  for (int p = 0; p < Tile::kV_LoadsPerThread; ++p) {
    int seq = wave_id * 8 + l_hi * 4 + p;
    int head_start = l_lo * 8;
    if (seq < valid_rows) {
      buf.data[p] = *reinterpret_cast<const uint4*>(gmem_base + seq * kv_row_stride + head_start);
    } else {
      buf.data[p] = zero4;
    }
  }
  return buf;
}

template <class Tile>
__device__ __forceinline__ void cooperative_store_v_to_lds(const v_reg_buf_t<Tile>& buf,
                                                           char* __restrict__ smem, int thread_id,
                                                           int v_lds_base) {
  // kKPack=4 store: intra-VGPR transpose + ds_write_b64 per slot.
  // Pre-transpose, lane (w, l_hi, l_lo)'s 4 chunks each hold 8 fp16 at
  //   (seq = w*8 + l_hi*4 + p, heads l_lo*8 .. l_lo*8 + 7).
  // For each j in [0,8) the four chunks share the same head h=l_lo*8+j across
  //   4 consecutive seqs — exactly the kKPack=4 slot. Pack and store:
  //   slot = (chunk[0].fp16[j] | chunk[1].fp16[j]<<16,
  //           chunk[2].fp16[j] | chunk[3].fp16[j]<<16) → 1 ds_write_b64.
  static_assert(Tile::kNumWaves == 4 && Tile::kK1 == 32 && Tile::kHeadDim == 256, "");
  const int wave_id = thread_id / kWaveSize;
  const int lane = thread_id % kWaveSize;
  const int l_hi = lane >> 5;
  const int l_lo = lane & 31;
  const uint32_t group = static_cast<uint32_t>(2 * wave_id + l_hi);
  const uint16_t* c0 = reinterpret_cast<const uint16_t*>(&buf.data[0]);
  const uint16_t* c1 = reinterpret_cast<const uint16_t*>(&buf.data[1]);
  const uint16_t* c2 = reinterpret_cast<const uint16_t*>(&buf.data[2]);
  const uint16_t* c3 = reinterpret_cast<const uint16_t*>(&buf.data[3]);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t head = static_cast<uint32_t>(l_lo * 8 + j);
    const uint32_t row = head >> 4;
    const uint32_t pos = head & 15u;
    const uint32_t lds_off = static_cast<uint32_t>(v_lds_base) + group * Tile::kV_GroupStride +
                             row * Tile::kV_RowStrideBytes + pos * Tile::kV_HeadStride;
    uint2 packed;
    packed.x = static_cast<uint32_t>(c0[j]) | (static_cast<uint32_t>(c1[j]) << 16);
    packed.y = static_cast<uint32_t>(c2[j]) | (static_cast<uint32_t>(c3[j]) << 16);
    *reinterpret_cast<uint2*>(smem + lds_off) = packed;
  }
}

// ===== LDS operand reads =====================================================

template <class Tile>
__device__ __forceinline__ void load_k_operand(const char* smem, int k_lds_base, int mt, int ks,
                                               int lane_id, uint32_t* out) {
  // kKPack=8: addr = base + row*RS + l_hi*32 + ks*8.
  uint32_t row = static_cast<uint32_t>(mt) * kMfmaM + (lane_id & 31);
  uint32_t lane_class_off = static_cast<uint32_t>(lane_id >> 5) * 32u;
  uint32_t ks_off = static_cast<uint32_t>(ks) * 8u;
  uint32_t addr =
      static_cast<uint32_t>(k_lds_base) + row * Tile::kK_RowStride + lane_class_off + ks_off;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(smem + addr);
  out[0] = src[0];
  out[1] = src[1];
}

// Pairs ks_base and ks_base+1 in one ds_read_b128 (16 contiguous bytes per lane).
// out[0..1] = K(mt, ks_base), out[2..3] = K(mt, ks_base+1).
template <class Tile>
__device__ __forceinline__ void load_k_operand_pair(const char* smem, int k_lds_base, int mt,
                                                    int ks_base, int lane_id, uint32_t* out) {
  uint32_t row = static_cast<uint32_t>(mt) * kMfmaM + (lane_id & 31);
  uint32_t lane_class_off = static_cast<uint32_t>(lane_id >> 5) * 32u;
  uint32_t ks_off = static_cast<uint32_t>(ks_base) * 8u;
  uint32_t addr =
      static_cast<uint32_t>(k_lds_base) + row * Tile::kK_RowStride + lane_class_off + ks_off;
  uint4 v = *reinterpret_cast<const uint4*>(smem + addr);
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

template <class Tile>
__device__ __forceinline__ void load_v_operand(const char* smem, int v_lds_base, int ks, int dt,
                                               int lane_id, uint32_t* out) {
  // kKPack=4: 4 consecutive seqs are pre-packed into one 8 B slot at
  //   V_LDS[group=seq/4][head][·]. One ds_read_b64 per MFMA step.
  // seq_base = ks*kMfmaK + (lane_id>>5)*4, kMfmaK=8 → group = ks*2 + (lane_id>>5).
  uint32_t head = static_cast<uint32_t>(dt) * kMfmaN + (lane_id & 31);
  uint32_t row = head >> 4;
  uint32_t pos = head & 15u;
  uint32_t group = static_cast<uint32_t>(ks) * 2u + (static_cast<uint32_t>(lane_id) >> 5);
  uint32_t addr = static_cast<uint32_t>(v_lds_base) + group * Tile::kV_GroupStride +
                  row * Tile::kV_RowStrideBytes + pos * Tile::kV_HeadStride;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(smem + addr);
  out[0] = src[0];
  out[1] = src[1];
}

// ===== TransposedC QK GEMM for one kK0=32 strip ==============================
// Issues MTiles=4 x KSteps=4 = 16 MFMA instructions per call.
// K is src0 (A-operand from LDS), Q is src1 (B-operand from registers).

template <class Tile>
__device__ __forceinline__ void qk_gemm_k0(
    const fp16_reg_tile<Tile::kBrLocal, Tile::kHeadDim>& Q_reg, const char* smem, int k_lds_base,
    fp32_s_tile<Tile::kBrLocal, Tile::kBc>& S_acc, int strip_idx, int lane_id) {
  using namespace asm_primitives;

  static_assert(Tile::kQK_KSteps % 2 == 0, "kKPack=8 ks-pairing requires even kQK_KSteps");
  constexpr int kKPairs = Tile::kQK_KSteps / 2;
  const int ks_global_base = strip_idx * Tile::kQK_KSteps;

  // Outer: ks-pairs (each paired load = 1 ds_read_b128 covering ks=base, base+1).
  // Inner MFMA order: sub-step outer, mt inner — preserves mt-rotated S_acc RAW
  // spacing of 4 (hides 4-cycle MFMA latency).
  uint32_t k_cur[Tile::kQK_MTiles][4];
  uint32_t k_next[Tile::kQK_MTiles][4];

#pragma unroll
  for (int mt = 0; mt < Tile::kQK_MTiles; ++mt) {
    load_k_operand_pair<Tile>(smem, k_lds_base, mt, 0, lane_id, k_cur[mt]);
  }

#pragma unroll
  for (int kp = 0; kp < kKPairs; ++kp) {
    if (kp + 1 < kKPairs) {
#pragma unroll
      for (int mt = 0; mt < Tile::kQK_MTiles; ++mt) {
        load_k_operand_pair<Tile>(smem, k_lds_base, mt, (kp + 1) * 2, lane_id, k_next[mt]);
      }
    }

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int ks = kp * 2 + sub;
#pragma unroll
      for (int mt = 0; mt < Tile::kQK_MTiles; ++mt) {
        mfma_f32_32x32x8_f16_vec(S_acc.vec(mt),
                                 &k_cur[mt][2 * sub],
                                 Q_reg.frag(ks_global_base + ks));
      }
    }

    if (kp + 1 < kKPairs) {
#pragma unroll
      for (int mt = 0; mt < Tile::kQK_MTiles; ++mt) {
        k_cur[mt][0] = k_next[mt][0];
        k_cur[mt][1] = k_next[mt][1];
        k_cur[mt][2] = k_next[mt][2];
        k_cur[mt][3] = k_next[mt][3];
      }
    }
  }
}

// ===== QK GEMM scheduling: sched_group_barrier pattern =======================

__device__ __forceinline__ void schedule_gemm0() {
  using namespace asm_primitives;
  // Post-kKPack=8 stream per qk_gemm_k0(): 8 ds_read_b128 + 16 mfma.
  // Front-load reads under MFMA so DS-read latency hides in QK MFMA throughput.
  sched_group_barrier<0x100, 4>();  // 4 DS reads (k_cur preload)
  sched_group_barrier<0x008, 4>();  // 4 MFMAs (kp=0 sub=0)
  sched_group_barrier<0x100, 2>();  // 2 DS reads (k_next prefetch, half)
  sched_group_barrier<0x008, 4>();  // 4 MFMAs (kp=0 sub=1)
  sched_group_barrier<0x100, 2>();  // 2 DS reads (k_next prefetch, rest)
  sched_group_barrier<0x008, 8>();  // 8 MFMAs (kp=1)
}

// ===== TransposedC PV GEMM for one kK1=32 strip ==============================
// V is src0 (A-operand from LDS), P is src1 (B-operand from registers).
// Software-pipelined: preload first V operand, then overlap load_v(next)
// with mfma(current) to hide DS read latency.
template <class Tile>
__device__ __forceinline__ void pv_gemm_k1(const fp16_p_tile<Tile::kBrLocal, Tile::kBc>& P_f16,
                                           const char* smem, int v_lds_base,
                                           fp32_acc_tile<Tile::kBrLocal, Tile::kHeadDim>& O_acc,
                                           int strip_idx, int lane_id) {
  using namespace asm_primitives;

  uint32_t v_cur[2], v_next[2];
  const int kf_base = strip_idx * Tile::kPV_KSteps;

  load_v_operand<Tile>(smem, v_lds_base, 0, 0, lane_id, v_cur);

  static constexpr int kTotalIters = Tile::kPV_KSteps * Tile::kPV_MTiles;

#pragma unroll
  for (int iter = 0; iter < kTotalIters; ++iter) {
    const int ks = iter / Tile::kPV_MTiles;
    const int dt = iter % Tile::kPV_MTiles;
    const uint32_t* p_frag = P_f16.frag(kf_base + ks);

    if (iter + 1 < kTotalIters) {
      const int next_ks = (iter + 1) / Tile::kPV_MTiles;
      const int next_dt = (iter + 1) % Tile::kPV_MTiles;
      load_v_operand<Tile>(smem, v_lds_base, next_ks, next_dt, lane_id, v_next);
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

template <class Tile, int D>
__device__ __forceinline__ void online_softmax(fp32_s_tile<Tile::kBrLocal, Tile::kBc>& S,
                                               fp32_acc_tile<Tile::kBrLocal, D>& O_acc,
                                               float& row_max, float& row_sum, float scale_log2) {
  using namespace asm_primitives;
  static constexpr int kMT = Tile::kBc / kMfmaM;

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
  static constexpr int kDBlks = fp32_acc_tile<Tile::kBrLocal, D>::kNumDBlks;
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

template <class Tile>
__device__ __forceinline__ void p_register_repack(
    const fp32_s_tile<Tile::kBrLocal, Tile::kBc>& S_acc,
    fp16_p_tile<Tile::kBrLocal, Tile::kBc>& P_f16) {
  static constexpr int kMT = Tile::kBc / kMfmaM;
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
// causal_offset = N_kv - N_q. Fused with the OOB mask (kv_col >= chunk_kv_len)
// into a single per-lane threshold check.
// `causal_lane_base` = wave_q_start + (lane_id & 31) + causal_offset - kv_chunk_start
// is loop-invariant and hoisted to the pipeline caller.

template <class Tile, bool IsCausal>
__device__ __forceinline__ void apply_masks(fp32_s_tile<Tile::kBrLocal, Tile::kBc>& S,
                                            int kv_start_local, int chunk_kv_len,
                                            int causal_lane_base, int lane_id) {
  using namespace asm_primitives;
  static constexpr int kMT = Tile::kBc / kMfmaM;
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

// One KV tile: QK GEMM, softmax, PV GEMM. IsCausal selects the mask path.
// Fully-masked tiles are handled by the mask producing P=0 (no runtime guard),
// which keeps codegen straight-line and avoids the barrier-deadlock hazard of
// early-returning a subset of waves.

template <class Tile, int D, bool IsCausal, bool ApplyMask>
__device__ __forceinline__ void process_kv_tile(const fp16_reg_tile<Tile::kBrLocal, D>& Q_reg,
                                                fp32_acc_tile<Tile::kBrLocal, D>& O_acc,
                                                float& row_max, float& row_sum,
                                                const FA3CDNA3PipelineArgs& args, char* smem, int j,
                                                const __half* k_head, const __half* v_head,
                                                int kv_row_stride, int chunk_kv_len,
                                                int causal_lane_base, int lane_id, int thread_id) {
  using namespace asm_primitives;
  constexpr int k_bases[2] = {Tile::kK_LDS_Base0, Tile::kK_LDS_Base1};

  const int kv_start_local = j * Tile::kBc;
  const int kv_valid_rows = min(Tile::kBc, chunk_kv_len - kv_start_local);
  const __half* k_tile_base = k_head + kv_start_local * kv_row_stride;

  // --- Stage 1: QK GEMM -- double-buffered K in LDS ---
  fp32_s_tile<Tile::kBrLocal, Tile::kBc> S_acc;
  S_acc.zero();

  k_reg_buf_t<Tile> k_regs =
      cooperative_load_k_to_regs<Tile>(k_tile_base, kv_row_stride, thread_id, kv_valid_rows);
  s_waitcnt_vmcnt(0);
  cooperative_store_k_to_lds<Tile>(k_regs, smem, thread_id, k_bases[0]);

  if (Tile::k0_loops > 1) {
    k_regs = cooperative_load_k_to_regs<Tile>(k_tile_base + Tile::kK0, kv_row_stride, thread_id,
                                              kv_valid_rows);
  }

#pragma unroll
  for (int ik = 0; ik < Tile::k0_loops - 2; ++ik) {
    int rd = ik & 1;
    int wr = 1 - rd;
    s_barrier();
    qk_gemm_k0<Tile>(Q_reg, smem, k_bases[rd], S_acc, ik, lane_id);
    schedule_gemm0();
    s_waitcnt_vmcnt(0);
    cooperative_store_k_to_lds<Tile>(k_regs, smem, thread_id, k_bases[wr]);
    k_regs = cooperative_load_k_to_regs<Tile>(k_tile_base + (ik + 2) * Tile::kK0, kv_row_stride,
                                              thread_id, kv_valid_rows);
  }

  const int v0_valid = min(Tile::kK1, chunk_kv_len - kv_start_local);
  v_reg_buf_t<Tile> v_regs = cooperative_load_v_to_regs<Tile>(
      v_head + kv_start_local * kv_row_stride, kv_row_stride, thread_id, max(v0_valid, 0));

  {
    const int rd6 = (Tile::k0_loops - 2) & 1;
    const int wr7 = 1 - rd6;

    s_barrier();
    qk_gemm_k0<Tile>(Q_reg, smem, k_bases[rd6], S_acc, Tile::k0_loops - 2, lane_id);
    schedule_gemm0();

    s_waitcnt_vmcnt(0);
    cooperative_store_k_to_lds<Tile>(k_regs, smem, thread_id, k_bases[wr7]);

    s_barrier();
    qk_gemm_k0<Tile>(Q_reg, smem, k_bases[wr7], S_acc, Tile::k0_loops - 1, lane_id);
    schedule_gemm0();
  }

  // --- Stage 2: Fused mask + softmax + P repack ---
  // Mask is elided on interior tiles where the OOB+causal limit is provably non-binding
  // (callers select ApplyMask=false for non-causal interior tiles and for causal phase-1
  // tiles fully covered by `phase1_end`). For other tiles ApplyMask=true keeps the masking.
  if constexpr (ApplyMask) {
    apply_masks<Tile, IsCausal>(S_acc, kv_start_local, chunk_kv_len, causal_lane_base, lane_id);
  }

  online_softmax<Tile, D>(S_acc, O_acc, row_max, row_sum, args.scale_log2);

  fp16_p_tile<Tile::kBrLocal, Tile::kBc> P_f16;
  p_register_repack<Tile>(S_acc, P_f16);

  // --- Stage 3: PV GEMM -- double-buffered V in LDS ---
  constexpr int v_bases[2] = {Tile::kV_LDS_Base0, Tile::kV_LDS_Base1};

  s_waitcnt_vmcnt(0);
  cooperative_store_v_to_lds<Tile>(v_regs, smem, thread_id, v_bases[0]);

  if (Tile::k1_loops > 1) {
    const int v1_off = kv_start_local + Tile::kK1;
    const int v1_valid = min(Tile::kK1, chunk_kv_len - v1_off);
    v_regs = cooperative_load_v_to_regs<Tile>(v_head + v1_off * kv_row_stride, kv_row_stride,
                                              thread_id, max(v1_valid, 0));
  }

  s_barrier();
  pv_gemm_k1<Tile>(P_f16, smem, v_bases[0], O_acc, 0, lane_id);

#pragma unroll
  for (int iv = 1; iv < Tile::k1_loops; ++iv) {
    const int buf = iv & 1;

    s_waitcnt_vmcnt(0);
    cooperative_store_v_to_lds<Tile>(v_regs, smem, thread_id, v_bases[buf]);

    if (iv + 1 < Tile::k1_loops) {
      const int next_v_off = kv_start_local + (iv + 1) * Tile::kK1;
      const int next_v_valid = min(Tile::kK1, chunk_kv_len - next_v_off);
      v_regs = cooperative_load_v_to_regs<Tile>(v_head + next_v_off * kv_row_stride, kv_row_stride,
                                                thread_id, max(next_v_valid, 0));
    }

    s_barrier();
    pv_gemm_k1<Tile>(P_f16, smem, v_bases[buf], O_acc, iv, lane_id);
  }
}

// Causal: full tiles first (no mask), then edge tiles. Non-causal: single phase.

template <class Tile, int D, bool IsCausal>
__device__ void run_fa3_cdna3_pipeline(FA3CDNA3PipelineArgs args, char* smem,
                                       fp32_acc_tile<Tile::kBrLocal, D>& O_acc, float& row_max_out,
                                       float& row_sum_out) {
  using namespace asm_primitives;

  const int lane_id = threadIdx.x % kWaveSize;
  const int wave_id = threadIdx.x / kWaveSize;
  const int thread_id = threadIdx.x;

  const int wave_q_start = args.q_block * Tile::kBr + wave_id * Tile::kBrLocal;
  const int kv_row_stride = args.nhead_k * D;

  fp16_reg_tile<Tile::kBrLocal, D> Q_reg;
  {
    const __half* q_base = args.Q + wave_q_start * args.nhead * D + args.head_idx * D;

#pragma unroll
    for (int ks = 0; ks < fp16_reg_tile<Tile::kBrLocal, D>::kNumKSteps; ++ks) {
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

  const int causal_lane_base =
      IsCausal ? (wave_q_start + (lane_id & 31) + args.causal_offset - args.kv_chunk_start) : 0;

  const int T = (chunk_kv_len + Tile::kBc - 1) / Tile::kBc;
  int max_j = T;
  int phase1_end = T;
  if constexpr (IsCausal) {
    int causal_end = args.q_block * Tile::kBr + Tile::kBr + args.causal_offset + Tile::kBc - 1 -
                     args.kv_chunk_start;
    max_j = min(T, max(causal_end, 0) / Tile::kBc);
    int full_end = wave_q_start + args.causal_offset - args.kv_chunk_start + 1;
    phase1_end = max(0, min(max_j, full_end / static_cast<int>(Tile::kBc)));
  }

  // Interior-full tile cutoff: tile j has all kv_cols in-bounds iff (j+1)*kBc <= chunk_kv_len.
  // For non-causal these tiles need no mask at all; for causal phase-1 the OOB part is also
  // non-binding (the causal part is non-binding by construction of phase1_end).
  const int T_full = chunk_kv_len / static_cast<int>(Tile::kBc);
  const int phase1_full_end = min(phase1_end, T_full);

  for (int j = 0; j < phase1_full_end; ++j) {
    process_kv_tile<Tile, D, false, false>(Q_reg, O_acc, row_max, row_sum, args, smem, j, k_head,
                                           v_head, kv_row_stride, chunk_kv_len, causal_lane_base,
                                           lane_id, thread_id);
  }
  for (int j = phase1_full_end; j < phase1_end; ++j) {
    process_kv_tile<Tile, D, false, true>(Q_reg, O_acc, row_max, row_sum, args, smem, j, k_head,
                                          v_head, kv_row_stride, chunk_kv_len, causal_lane_base,
                                          lane_id, thread_id);
  }

  if constexpr (IsCausal) {
    for (int j = phase1_end; j < max_j; ++j) {
      process_kv_tile<Tile, D, true, true>(Q_reg, O_acc, row_max, row_sum, args, smem, j, k_head,
                                           v_head, kv_row_stride, chunk_kv_len, causal_lane_base,
                                           lane_id, thread_id);
    }
  }

  row_max_out = row_max;
  row_sum_out = row_sum;
}

}  // namespace cdna3
}  // namespace flashinfer
