// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
// macros.hpp first: it defines PLATFORM_HIP_DEVICE and #errors on a non-HIP
// compiler, which must fire before a hip/ header is reached.
#include <hip/hip_cooperative_groups.h>

#include "macros.hpp"
namespace flashinfer {
namespace gpu_iface {
namespace cg = ::cooperative_groups;
}  // namespace gpu_iface
}  // namespace flashinfer
