// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "memory_ops_hip.h"
#include "memory_types.hpp"
#include "platform.hpp"

namespace flashinfer {
namespace gpu_iface {
namespace memory {

namespace mem_detail = flashinfer::gpu_iface::memory::detail::hip;

/**
 * @brief Commits pending asynchronous memory operations to a group
 */
__device__ __forceinline__ void commit_group() { mem_detail::commit_group(); }

/**
 * @brief Waits until N most recent groups of async operations are complete
 *
 * @tparam N Number of most recent groups to wait for (0-7)
 */
template <size_t N>
__device__ __forceinline__ void wait_group() {
  mem_detail::wait_group<N>();
}

/**
 * @brief Asynchronously loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  mem_detail::load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
}

template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_64b(T* smem_ptr, const T* gmem_ptr) {
#if defined(PLATFORM_HIP_DEVICE)
  mem_detail::load_64b<PrefetchOpt>(smem_ptr, gmem_ptr);
#else
#error "load_64b not implemented for this platform"
#endif
}

/**
 * @brief Conditionally loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  mem_detail::pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
}

template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_64b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#if defined(PLATFORM_HIP_DEVICE)
  mem_detail::pred_load_64b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
#else
#error "pred_load_64b not implemented for this platform"
#endif
}

/**
 * @brief Loads N bits (128 or 256) from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  mem_detail::load<NumBits, PrefetchOpt>(smem_ptr, gmem_ptr);
}

/**
 * @brief Conditionally loads N bits from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  mem_detail::pred_load<NumBits, PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
}

}  // namespace memory
}  // namespace gpu_iface
}  // namespace flashinfer
