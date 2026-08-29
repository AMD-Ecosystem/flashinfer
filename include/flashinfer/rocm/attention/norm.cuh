// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifdef FLASHINFER_NORM_CUH_
#error \
    "include/flashinfer/norm.cuh and include/flashinfer/rocm/attention/norm.cuh both define FLASHINFER_NORM_CUH_; include only one"
#endif

#ifndef FLASHINFER_ROCM_ATTENTION_NORM_CUH_
#define FLASHINFER_ROCM_ATTENTION_NORM_CUH_

#include <numeric>

#include "flashinfer/rocm/dispatch.cuh"
#include "flashinfer/rocm/gpu_runtime_compat.hpp"
#include "flashinfer/rocm/math_hip.h"
#include "flashinfer/rocm/platform.hpp"
#include "flashinfer/rocm/utils.cuh"
#include "flashinfer/rocm/vec_dtypes_hip.h"
namespace flashinfer {

namespace norm {

// Threads per lane group. Must divide the wavefront so a group's shuffle
// reduction never crosses a wave boundary.
constexpr uint32_t kLaneGroupSize = 32;
constexpr uint32_t kMaxBlockSize = 1024;

static_assert(kLaneGroupSize <= static_cast<uint32_t>(kWarpSize) &&
              static_cast<uint32_t>(kWarpSize) % kLaneGroupSize == 0);
// Stage 2 folds num_warps partial sums inside a single lane group. Ceiling
// division, matching how the launchers derive num_warps.
static_assert((kMaxBlockSize + kLaneGroupSize - 1) / kLaneGroupSize <= kLaneGroupSize);

/*!
 * \brief Shared memory for the fused kernels, staging the fp32 row if it fits.
 *
 * Staging costs `d` floats, which exceeds CDNA3's 64 KB above d = 16352. Both
 * fused launchers must decide this identically.
 */
inline uint32_t FusedRMSNormSmemSize(uint32_t num_warps, uint32_t d, bool* stage_x) {
  const uint32_t reduce_bytes = ceil_div(num_warps, 4) * 4 * sizeof(float);
  const uint32_t staged_bytes = reduce_bytes + d * sizeof(float);
  int dev_id = 0;
  FI_GPU_CALL(gpuGetDevice(&dev_id));
  *stage_x = staged_bytes <= static_cast<uint32_t>(getMaxSharedMemPerBlock(dev_id));
  return *stage_x ? staged_bytes : reduce_bytes;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(T* __restrict__ input, T* __restrict__ weight, T* __restrict__ output,
                              const uint32_t d, const uint32_t stride_input,
                              const uint32_t stride_output, float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * kLaneGroupSize;
  const uint32_t num_threads = num_warps * kLaneGroupSize;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = kLaneGroupSize / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = kLaneGroupSize / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<T, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
gpuError_t RMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t d,
                   uint32_t stride_input, uint32_t stride_output, float eps = 1e-5,
                   bool enable_pdl = false, gpuStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(kMaxBlockSize, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, kLaneGroupSize);
  dim3 nblks(batch_size);
  dim3 nthrs(kLaneGroupSize, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    FI_GPU_CALL(
        gpuFuncSetAttribute((void*)kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return gpuSuccess;
}

// \param stage_x Keep the fp32 row in shared memory between the two passes. The
// caller clears it when the row would not fit, and pass 2 then re-reads the row
// from `residual` in global instead, at the cost of a dtype round-trip.
template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(T* __restrict__ input, T* __restrict__ residual,
                                      T* __restrict__ weight, const uint32_t d,
                                      const uint32_t stride_input, const uint32_t stride_residual,
                                      float weight_bias, float eps, bool stage_x) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * kLaneGroupSize;
  const uint32_t num_threads = num_warps * kLaneGroupSize;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = stage_x ? smem + ceil_div(num_warps, 4) * 4 : nullptr;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      if (stage_x) {
        x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      }
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = kLaneGroupSize / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = kLaneGroupSize / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      if (stage_x) {
        x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      } else {
        // Same thread, same address pass 1 stored to; only the dtype round-trip differs.
        vec_t<T, VEC_SIZE> residual_vec;
        residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                          thread_id * VEC_SIZE);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          x_vec[j] = float(residual_vec[j]);
        }
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
gpuError_t FusedAddRMSNorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                           uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                           bool enable_pdl = false, gpuStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(kMaxBlockSize, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, kLaneGroupSize);
  dim3 nblks(batch_size);
  dim3 nthrs(kLaneGroupSize, num_warps);
  bool stage_x = true;
  const uint32_t smem_size = FusedRMSNormSmemSize(num_warps, d, &stage_x);
  float weight_bias = 0.f;
  void* args[] = {&input,           &residual,    &weight, &d,      &stride_input,
                  &stride_residual, &weight_bias, &eps,    &stage_x};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    FI_GPU_CALL(
        gpuFuncSetAttribute((void*)kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return gpuSuccess;
}

template <typename T>
gpuError_t GemmaRMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t d,
                        uint32_t stride_input, uint32_t stride_output, float eps = 1e-5,
                        bool enable_pdl = false, gpuStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(kMaxBlockSize, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, kLaneGroupSize);
  dim3 nblks(batch_size);
  dim3 nthrs(kLaneGroupSize, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 1.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    FI_GPU_CALL(
        gpuFuncSetAttribute((void*)kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return gpuSuccess;
}

template <typename T>
gpuError_t GemmaFusedAddRMSNorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                                uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                                bool enable_pdl = false, gpuStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(kMaxBlockSize, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, kLaneGroupSize);
  dim3 nblks(batch_size);
  dim3 nthrs(kLaneGroupSize, num_warps);
  bool stage_x = true;
  const uint32_t smem_size = FusedRMSNormSmemSize(num_warps, d, &stage_x);
  float weight_bias = 1.f;
  void* args[] = {&input,           &residual,    &weight, &d,      &stride_input,
                  &stride_residual, &weight_bias, &eps,    &stage_x};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    FI_GPU_CALL(
        gpuFuncSetAttribute((void*)kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    FI_GPU_CALL(gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return gpuSuccess;
}

}  // namespace norm

}  // namespace flashinfer

#endif  // FLASHINFER_ROCM_ATTENTION_NORM_CUH_
