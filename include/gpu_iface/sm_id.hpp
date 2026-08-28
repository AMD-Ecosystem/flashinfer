// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "macros.hpp"

namespace flashinfer {
namespace gpu_iface {

__device__ __forceinline__ uint32_t get_processor_id() {
  // Read HW_ID (id=4, offset=0, size=32) and extract bits [15:8], which pack
  // CU_ID | SH_ID | SE_ID into 8 bits that uniquely identify a CU across the chip.
  // Encoding of s_getreg arg: (reg_id | (offset << 6) | ((size - 1) << 11)).
  constexpr uint32_t HW_REG_HW_ID = 4;
  uint32_t hw_id = __builtin_amdgcn_s_getreg(HW_REG_HW_ID | (0 << 6) | (31 << 11));
  return (hw_id >> 8) & 0xFF;
}

}  // namespace gpu_iface
}  // namespace flashinfer
