// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "gpu_runtime_compat.hpp"
#include "macros.hpp"

namespace flashinfer {
namespace gpu_iface {

// Platform-agnostic stream type
constexpr int kWarpSize = 64;

}  // namespace gpu_iface
}  // namespace flashinfer
