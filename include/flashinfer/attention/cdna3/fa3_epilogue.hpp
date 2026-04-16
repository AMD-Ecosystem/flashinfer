// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: Epilogue -- normalize O accumulator and write to global memory.
// Supports both direct output and split-KV partial output with strided layout.
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

#include "fa3_tiles.hpp"

namespace flashinfer {
namespace cdna3 {

// base2_lse: if true, writes LSE in log-base-2 format (for MergeStates); else natural-log.
template <int BrLocal, int D>
__device__ void fa3_cdna3_epilogue(fp32_acc_tile<BrLocal, D>& O_acc, float row_max, float row_sum,
                                   float scale_log2, __half* O_global, float* LSE_global,
                                   int wave_q_start, int nhead, int head_idx, int N_q,
                                   int o_row_stride, int lse_row_stride, int lse_head_stride,
                                   bool base2_lse, int lane_id) {
  float inv_sum = (row_sum > 0.f) ? (1.0f / row_sum) : 0.f;

#pragma unroll
  for (int d = 0; d < fp32_acc_tile<BrLocal, D>::kNumDBlks; ++d) {
#pragma unroll
    for (int i = 0; i < kMfmaOutRegs; ++i) {
      O_acc.vec(d)[i] *= inv_sum;
    }
  }

  const int block = lane_id >> 5;
  const int global_row = wave_q_start + (lane_id & 31);

  if (global_row < N_q) {
    __half* row_ptr = O_global + global_row * o_row_stride + head_idx * D;

    // Four consecutive regs per d-block map to four contiguous head_dim columns (uint2 store).
    static constexpr int kGroups = kMfmaOutRegs / 4;
#pragma unroll
    for (int dt = 0; dt < fp32_acc_tile<BrLocal, D>::kNumDBlks; ++dt) {
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        int base_col = dt * kMfmaN + (g << 3) + (block << 2);

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
        auto pk01 = __builtin_amdgcn_cvt_pkrtz(O_acc.vec(dt)[g * 4 + 0], O_acc.vec(dt)[g * 4 + 1]);
        auto pk23 = __builtin_amdgcn_cvt_pkrtz(O_acc.vec(dt)[g * 4 + 2], O_acc.vec(dt)[g * 4 + 3]);
        uint32_t w0, w1;
        __builtin_memcpy(&w0, &pk01, sizeof(uint32_t));
        __builtin_memcpy(&w1, &pk23, sizeof(uint32_t));
        *reinterpret_cast<uint2*>(row_ptr + base_col) = make_uint2(w0, w1);
#else
        row_ptr[base_col + 0] = static_cast<__half>(O_acc.vec(dt)[g * 4 + 0]);
        row_ptr[base_col + 1] = static_cast<__half>(O_acc.vec(dt)[g * 4 + 1]);
        row_ptr[base_col + 2] = static_cast<__half>(O_acc.vec(dt)[g * 4 + 2]);
        row_ptr[base_col + 3] = static_cast<__half>(O_acc.vec(dt)[g * 4 + 3]);
#endif
      }
    }
  }

  if (LSE_global != nullptr && global_row < N_q) {
    if (block == 0) {
      float log2_sum = __log2f(row_sum > 0.f ? row_sum : 1e-38f);
      float lse_val;
      if (base2_lse) {
        lse_val = row_max * scale_log2 + log2_sum;
      } else {
        const float kLoge2 = 0.693147180559945309417f;
        lse_val = row_max / scale_log2 + log2_sum * kLoge2;
      }
      LSE_global[global_row * lse_row_stride + head_idx * lse_head_stride] = lse_val;
    }
  }
}

}  // namespace cdna3
}  // namespace flashinfer
