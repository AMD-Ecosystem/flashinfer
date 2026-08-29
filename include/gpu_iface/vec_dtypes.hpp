// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <float.h>
#include <math.h>

#include "platform.hpp"

#include "backend/hip/vec_dtypes_hip.h"

namespace flashinfer {
namespace gpu_iface {
namespace vec_dtypes {

namespace vec_t_detail = flashinfer::gpu_iface::vec_dtypes::detail::hip;

// Re-export types and functions from the appropriate backend
// This allows code to use flashinfer::gpu_iface::vec_dtypes::vec_t<float, 4>
using vec_t_detail::vec_cast;
using vec_t_detail::vec_t;

}  // namespace vec_dtypes
}  // namespace gpu_iface
}  // namespace flashinfer
