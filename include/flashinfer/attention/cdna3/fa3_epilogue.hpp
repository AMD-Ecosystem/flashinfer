// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3 V5: Epilogue -- normalize O accumulator and write to global memory.
//
// TransposedC O_acc register layout (gfx942):
//   Thread t (lane_id 0..63):
//     q_row = t % 32
//     head_dim positions = mfma_32x32_row(t/32, i) for i=0..15, offset by d-block
//   For d-block dt: O_acc.vec(dt)[i] = O[dt*32 + mfma_32x32_row(t/32, i)][q_row]
//   Softmax state (row_max, row_sum) is scalar per thread (one Q-row per thread).

#pragma once

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "asm_primitives.hpp"
#include "fa3_tiles.hpp"

namespace flashinfer {
namespace cdna3 {

template <int BrLocal, int D>
__device__ void fa3_cdna3_epilogue(fp32_acc_tile<BrLocal, D>& O_acc, float row_max, float row_sum,
                                   float scale_log2, __half* O_global, float* LSE_global,
                                   int wave_q_start, int nhead, int head_idx, int N, int D_stride,
                                   int lane_id) {
  float inv_sum = (row_sum > 0.f) ? (1.0f / row_sum) : 0.f;

// Scale O by 1/row_sum (scalar: same scale for all 128 register values)
#pragma unroll
  for (int d = 0; d < fp32_acc_tile<BrLocal, D>::kNumDBlks; ++d) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      O_acc.vec(d)[i] *= inv_sum;
    }
  }

  // Convert FP32 -> FP16 and write to global memory.
  // TransposedC layout: q_row = t%32, head_dim = mfma_32x32_row(block, i)
  const int block = lane_id >> 5;
  const int global_row = wave_q_start + (lane_id & 31);

  if (global_row < N) {
    __half* row_ptr = O_global + global_row * D_stride + head_idx * D;

    for (int dt = 0; dt < fp32_acc_tile<BrLocal, D>::kNumDBlks; ++dt) {
      for (int i = 0; i < kMfmaOutRegs; ++i) {
        int head_dim_col = dt * kMfmaN + asm_primitives::mfma_32x32_row(block, i);
        __half val;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
        auto packed = __builtin_amdgcn_cvt_pkrtz(O_acc.vec(dt)[i], 0.0f);
        __builtin_memcpy(&val, &packed, sizeof(__half));
#else
        val = static_cast<__half>(O_acc.vec(dt)[i]);
#endif
        row_ptr[head_dim_col] = val;
      }
    }
  }

  // Write LSE (optional).
  // With TransposedC, q_row = t%32. Two threads per wave share q_row=0
  // (lane 0 and lane 32). Only one needs to write.
  if (LSE_global != nullptr && global_row < N) {
    if (block == 0) {
      const float kLoge2 = 0.693147180559945309417f;
      float lse_val = row_max / scale_log2 + __log2f(row_sum > 0.f ? row_sum : 1e-38f) * kLoge2;
      LSE_global[head_idx * N + global_row] = lse_val;
    }
  }
}

}  // namespace cdna3
}  // namespace flashinfer
