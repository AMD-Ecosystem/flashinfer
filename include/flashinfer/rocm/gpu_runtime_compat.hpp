// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
// clang-format off
// macros.hpp first: it raises the "requires a HIP compiler" #error, which is a
// better diagnostic than a missing hip/ header further down.
#include "macros.hpp"
// clang-format on

#include <hip/hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

// HIP error checking macro (replaces FLASHINFER_CUDA_CALL upstream)
#define FI_HIP_CALL(call)                                                                          \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::ostringstream err_msg;                                                                  \
      err_msg << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(err_msg.str());                                                     \
    }                                                                                              \
  } while (0)

/// Returns the number of compute units.
///
/// Cached per device id to avoid repeating the attribute query on every kernel
/// launch. The cache is thread_local so concurrent callers (e.g. multi-threaded
/// Python) never race on it. 0 is treated as "not cached" — a valid count is
/// always > 0, so a device reporting 0 simply isn't memoized rather than
/// poisoning the cache.
///
/// @param dev_id Device ID
/// @return Compute-unit count
inline int getMultiProcessorCount(int dev_id) {
  static thread_local int cache[64] = {0};
  if (dev_id >= 0 && dev_id < 64 && cache[dev_id] > 0) return cache[dev_id];
  int count = 0;
  FI_HIP_CALL(hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, dev_id));
  if (dev_id >= 0 && dev_id < 64 && count > 0) cache[dev_id] = count;
  return count;
}

inline int getMaxSharedMemPerMultiprocessor(int dev_id) {
  int max_smem_per_sm = 0;
  hipDeviceProp_t deviceProp;
  FI_HIP_CALL(hipGetDeviceProperties(&deviceProp, dev_id));
  max_smem_per_sm = deviceProp.sharedMemPerMultiprocessor;

  return max_smem_per_sm;
}

/// Returns the device's gfx architecture with the feature suffix stripped, so
/// `gfx950:sramecc+:xnack-` comes back as `gfx950` — matching what Python's
/// `arch_caps.normalize_arch` produces, since routing policy is keyed on it.
///
/// Cached per device id like the queries above; the returned pointer stays valid
/// for the life of the thread. AITER carries the same logic in
/// `aiter_hip_common.h`, but its headers pull in pybind11 and cannot be included
/// from a `-DPy_LIMITED_API` translation unit.
///
/// @param dev_id Device ID
/// @return NUL-terminated arch name; "" only for a device id outside the cache
///         range. A failed query throws via FI_HIP_CALL, as the accessors above do.
inline const char* getGcnArchName(int dev_id) {
  static thread_local char cache[64][32] = {};
  if (dev_id < 0 || dev_id >= 64) return "";  // out of cache range; "" reads as unknown
  if (cache[dev_id][0] != '\0') return cache[dev_id];

  hipDeviceProp_t deviceProp;
  FI_HIP_CALL(hipGetDeviceProperties(&deviceProp, dev_id));
  const char* full = deviceProp.gcnArchName;
  size_t n = 0;
  while (full[n] != '\0' && full[n] != ':' && n < sizeof(cache[0]) - 1) ++n;
  std::memcpy(cache[dev_id], full, n);
  cache[dev_id][n] = '\0';
  return cache[dev_id];
}

/// Returns the maximum shared memory per thread block
///
/// Cached per device id, on the same rationale as getMultiProcessorCount above:
/// this sits on the per-launch decode/prefill dispatch path, and the underlying
/// query copies out a whole device-properties struct rather than a single
/// attribute. Measured on MI350X/ROCm 7.2 the uncached call costs ~6.4 us once
/// and ~0.17 us thereafter, against ~1.0 us for the kernel launch it precedes.
///
/// @param dev_id Device ID
/// @return Maximum shared memory per block in bytes
inline int getMaxSharedMemPerBlock(int dev_id) {
  static thread_local int cache[64] = {0};
  if (dev_id >= 0 && dev_id < 64 && cache[dev_id] > 0) return cache[dev_id];
  // CDNA3/MI300X: sharedMemPerBlock = 65,536 bytes (64 KB) - the actual per-block limit
  //               sharedMemPerMultiprocessor = 19,922,944 bytes (~19 MB) - total LDS per CU
  hipDeviceProp_t deviceProp;
  FI_HIP_CALL(hipGetDeviceProperties(&deviceProp, dev_id));
  const int max_smem_per_block = static_cast<int>(deviceProp.sharedMemPerBlock);
  if (dev_id >= 0 && dev_id < 64 && max_smem_per_block > 0) cache[dev_id] = max_smem_per_block;
  return max_smem_per_block;
}
