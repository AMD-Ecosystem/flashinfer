// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "macros.hpp"

// Include platform-specific implementations
#include "math_hip.h"

namespace flashinfer {
namespace gpu_iface {
namespace math {
using flashinfer::math::inf;
using flashinfer::math::log2e;
using flashinfer::math::loge2;
using flashinfer::math::ptx_exp2;
using flashinfer::math::ptx_log2;
using flashinfer::math::ptx_rcp;
using flashinfer::math::rsqrt;
using flashinfer::math::shfl_xor_sync;
using flashinfer::math::tanh;

}  // namespace math
}  // namespace gpu_iface
}  // namespace flashinfer
