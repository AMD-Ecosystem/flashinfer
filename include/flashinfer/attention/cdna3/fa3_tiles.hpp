// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 V5: Tile types for AMD MI300X (gfx942 / CDNA3).
//
// V5 TransposedC design: 4 waves x 64 threads = 256 threads per workgroup.
//   kBr = 128 (32 rows/wave), kBc = 128, kHeadDim = 256
//   v_mfma_f32_32x32x8f16 as the primary compute instruction
//
// 32x32x8 MFMA register layout (gfx942):
//   A-operand: thread t -> row = t%32, k_base = (t/32)*4, owns A[row][k_base+0..3]
//   B-operand: thread t -> col = t%32, k_base = (t/32)*4, owns B[k_base+0..3][col]
//   C-output:  thread t -> col = t%32, rows via mfma_32x32_row(block, v)
//
// TransposedC: MFMA operands are swapped so C is transposed.
//   QK GEMM (K=A, Q=B): S_acc.vec(mt)[i] = S^T[kv_col=mfma_32x32_row(block,i)][q_row=t%32]
//   PV GEMM (V=A, P=B): O_acc.vec(dt)[i] = O^T[head_dim=mfma_32x32_row(block,i)][q_row=t%32]

#pragma once

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
#include <cstdint>
#include <cstring>

#include "asm_primitives.hpp"

namespace flashinfer {
namespace cdna3 {

static constexpr int kWaveSize = 64;
static constexpr int kMfmaM = 32;
static constexpr int kMfmaN = 32;
static constexpr int kMfmaK = 8;
static constexpr int kMfmaOutRegs = 16;  // 16 fp32 per thread per 32x32 MFMA

// ---------------------------------------------------------------------------
// fp16_reg_tile: VGPR-resident Q tile (32 rows x D cols per wave)
// ---------------------------------------------------------------------------

template <int BrLocal, int D>
struct fp16_reg_tile {
  static_assert(D % kMfmaK == 0, "D must be divisible by kMfmaK");

  static constexpr int kNumKSteps = D / kMfmaK;  // 256/8 = 32

  uint32_t data[kNumKSteps][2];

  __device__ __forceinline__ void zero() {
    for (int i = 0; i < kNumKSteps; ++i) {
      data[i][0] = 0;
      data[i][1] = 0;
    }
  }

  __device__ __forceinline__ const uint32_t* frag(int k) const { return data[k]; }
  __device__ __forceinline__ uint32_t* frag(int k) { return data[k]; }
};

// ---------------------------------------------------------------------------
// fp32_acc_tile: O accumulator (BrLocal rows x D cols)
//
// Uses f32x16_t (ext_vector_type(16)) per D-block to guarantee the compiler
// allocates 16 contiguous VGPRs for each MFMA accumulator tile.
//
// TransposedC: vec(dt)[i] = O[dt*32+mfma_32x32_row(block,i)][q_row=t%32]
// ---------------------------------------------------------------------------

template <int BrLocal, int D>
struct fp32_acc_tile {
  static_assert(D % kMfmaN == 0, "D must be divisible by kMfmaN");

  static constexpr int kNumDBlks = D / kMfmaN;  // 8

  asm_primitives::f32x16_t acc[kNumDBlks];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int d = 0; d < kNumDBlks; ++d)
      for (int i = 0; i < 16; ++i) acc[d][i] = 0.f;
  }

  __device__ __forceinline__ float* tile(int d_blk) {
    return reinterpret_cast<float*>(&acc[d_blk]);
  }
  __device__ __forceinline__ const float* tile(int d_blk) const {
    return reinterpret_cast<const float*>(&acc[d_blk]);
  }

  __device__ __forceinline__ asm_primitives::f32x16_t& vec(int d_blk) { return acc[d_blk]; }
  __device__ __forceinline__ const asm_primitives::f32x16_t& vec(int d_blk) const {
    return acc[d_blk];
  }
};

// ---------------------------------------------------------------------------
// fp32_s_tile: S accumulator (BrLocal x Bc) for QK^T result
//
// Uses f32x16_t per tile block for proper MFMA VGPR alignment.
// TransposedC: vec(mt)[i] = S^T[mt*32+mfma_32x32_row(block,i)][q_row=t%32]
//   mt = M-tile index (KV-col group), 4 M-tiles for kBc=128
// ---------------------------------------------------------------------------

template <int BrLocal, int Bc>
struct fp32_s_tile {
  static_assert(Bc % kMfmaN == 0, "Bc must be divisible by kMfmaN");

  static constexpr int kNumTiles = Bc / kMfmaN;

  asm_primitives::f32x16_t data[kNumTiles];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int n = 0; n < kNumTiles; ++n)
      for (int i = 0; i < 16; ++i) data[n][i] = 0.f;
  }

  __device__ __forceinline__ float* tile(int t) { return reinterpret_cast<float*>(&data[t]); }
  __device__ __forceinline__ const float* tile(int t) const {
    return reinterpret_cast<const float*>(&data[t]);
  }

  __device__ __forceinline__ asm_primitives::f32x16_t& vec(int t) { return data[t]; }
  __device__ __forceinline__ const asm_primitives::f32x16_t& vec(int t) const { return data[t]; }
};

// ---------------------------------------------------------------------------
// fp16_p_tile: P in FP16 for PV GEMM B-operand (BrLocal x Bc)
//
// TransposedC: P is the B-operand (src1) of the PV GEMM. Each fragment holds
// 4 fp16 values (2 uint32). Fragment kf corresponds to PV GEMM K-step kf,
// which covers kv_col positions [kf*8 + block*4 .. kf*8 + block*4 + 3].
// ---------------------------------------------------------------------------

template <int BrLocal, int Bc>
struct fp16_p_tile {
  static constexpr int kNumKFrags = Bc / kMfmaK;  // 16

  uint32_t data[kNumKFrags * 2];

  __device__ __forceinline__ void zero() {
    for (int i = 0; i < kNumKFrags * 2; ++i) data[i] = 0;
  }

  __device__ __forceinline__ uint32_t* frag(int k_frag) { return &data[k_frag * 2]; }
  __device__ __forceinline__ const uint32_t* frag(int k_frag) const { return &data[k_frag * 2]; }
};

// ---------------------------------------------------------------------------
// fp32_vec: per-row softmax state (kept for backward compatibility)
// V5 TransposedC uses scalar float for row_max/row_sum (one Q-row per thread),
// but fp32_vec is retained for potential future use.
// ---------------------------------------------------------------------------

template <int BrLocal>
struct fp32_vec {
  static constexpr int kSize = kMfmaOutRegs;  // 16
  float v[kSize];

  __device__ __forceinline__ fp32_vec() {
    for (int i = 0; i < kSize; ++i) v[i] = 0.f;
  }

  __device__ __forceinline__ explicit fp32_vec(float fill) {
    for (int i = 0; i < kSize; ++i) v[i] = fill;
  }

  __device__ __forceinline__ float& operator[](int i) { return v[i]; }
  __device__ __forceinline__ float operator[](int i) const { return v[i]; }
};

}  // namespace cdna3
}  // namespace flashinfer
