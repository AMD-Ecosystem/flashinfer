// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
// clang-format off
// Order matters and clang-format cannot express it: Google style sorts <angle>
// system includes ahead of "quoted" ones, but macros.hpp must come first so its
// non-HIP #error fires before a missing hip/ header does.
#include "macros.hpp"
#include <hip/hip_cooperative_groups.h>
// clang-format on
namespace flashinfer {
namespace gpu_iface {
namespace cg = ::cooperative_groups;
}  // namespace gpu_iface
}  // namespace flashinfer
