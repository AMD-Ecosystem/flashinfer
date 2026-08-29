// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "platform.hpp"

namespace flashinfer {
namespace gpu_iface {
namespace memory {

/**
 * @brief Control options for shared memory fill behavior
 */
enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

/**
 * @brief Control options for memory prefetch behavior
 */
enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

}  // namespace memory
}  // namespace gpu_iface
}  // namespace flashinfer
