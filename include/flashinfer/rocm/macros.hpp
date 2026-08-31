// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIP__)
// FIXME: Temporarily setting __forceinline__ to inline as amdclang++ 6.4 throws
// an error when __forceinline__ is used.
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#ifndef __grid_constant__
#define __grid_constant__
#endif

#else
// The CUDA backend was removed; these headers serve HIP only. Fail here with a
// named diagnostic rather than deeper in a missing hip/ header.
#error "flashinfer ROCm requires a HIP compiler (__HIP__ / __HIPCC__)."
#endif
