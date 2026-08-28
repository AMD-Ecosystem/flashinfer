// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#include <algorithm>

#include "gpu_iface/gpu_runtime_compat.hpp"
#include "gpu_iface/math_ops.hpp"
#include "gpu_iface/platform.hpp"
#include "gpu_iface/utils.cuh"
#include "gpu_iface/vec_dtypes.hpp"

namespace flashinfer {
using namespace gpu_iface::vec_dtypes;
namespace activation {

// Adaptive launch config for act_and_mul_kernel. One block per token underfills
// the GPU when num_tokens is small (decode / small batch), so split each row
// across blocks_per_row blocks on gridDim.y until the total block count covers
// the CU array. For large num_tokens this resolves to blocks_per_row == 1, i.e.
// the original one-block-per-token launch. Single definition shared by the AOT
// launcher (flashinfer/csrc/rocm/activation.cu) and the JIT template
// (flashinfer/jit/activation.py) so the two paths cannot drift.
inline void act_and_mul_launch_dims(int d, int64_t num_tokens, uint32_t vec_size, int dev_id,
                                    dim3& grid_dim, dim3& block_dim) {
  uint32_t vecs = std::max(1U, (uint32_t)(d / vec_size));
  uint32_t block_size = std::max(1U, std::min(vecs, 1024U));
  // Oversubscribe CUs by 2x: enough to fill the GPU when num_tokens is small,
  // without splitting rows once num_tokens already covers the CU array (extra
  // splitting only adds launch/tail overhead — empirically bandwidth-neutral).
  const uint32_t target_blocks = (uint32_t)getMultiProcessorCount(dev_id) * 2u;
  const uint32_t max_bpr = ceil_div(vecs, block_size);
  uint32_t blocks_per_row = 1u;
  if ((uint64_t)num_tokens < target_blocks) {
    const uint64_t nt = (uint64_t)std::max<int64_t>(1, num_tokens);
    blocks_per_row =
        std::max(1u, std::min((uint32_t)ceil_div<uint64_t>(target_blocks, nt), max_bpr));
  }
  grid_dim = dim3((unsigned)num_tokens, blocks_per_row, 1);
  block_dim = dim3(block_size, 1, 1);
}

// 2D grid: blockIdx.x selects the token (row), blockIdx.y selects a column-tile
// of that row. Output elements are independent (no cross-element reduction), so a
// row can be split across gridDim.y blocks with no atomics. When gridDim.y == 1
// (e.g. any 1D launch, including the CUDA path) this collapses to one block per
// token with the same memory-access pattern as the original kernel.
template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  // Row-base addresses are 64-bit (token_idx * 2 * d can exceed 2^31); the
  // intra-row column index stays 32-bit to keep the inner-loop address math
  // identical to the original one-block-per-token kernel (no 64-bit multiplies
  // in the hot loop). col_block <= 65535, blockDim.x <= 1024 → products fit u32.
  const int64_t token_idx = blockIdx.x;
  const int64_t offset = token_idx * 2 * d;   // input row base (gate || up)
  const int64_t out_base = token_idx * d;     // output row base
  const uint32_t col_block = blockIdx.y;      // 0 when 1D-equivalent
  const uint32_t num_col_blocks = gridDim.y;  // 1 when 1D-equivalent
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t col_stride = blockDim.x * num_col_blocks;  // == blockDim.x when 1D
  const uint32_t num_vec = d / vec_size;
  const uint32_t vec_start = col_block * blockDim.x + thread_idx;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = vec_start; idx < num_vec; idx += col_stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.cast_store(out + out_base + idx * vec_size);
  }

  // Scalar remainder over [num_vec*vec_size, d), column-tiled the same way.
  // Always empty for the fp16/bf16 dispatch (16-byte alignment forces d % vec_size
  // == 0); kept defensive. Do NOT key this off d % (blockDim.x * vec_size) — that
  // assumes blockDim.x is the global stride, which is false under column-tiling.
  const uint32_t scalar_base = num_vec * vec_size;
  const uint32_t scalar_count = (uint32_t)d - scalar_base;
#pragma unroll 1
  for (uint32_t s = vec_start; s < scalar_count; s += col_stride) {
    const uint32_t e = scalar_base + s;
    out[out_base + e] = Activation((float)input[offset + e]) * (float)input[offset + d + e];
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace activation
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
