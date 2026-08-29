// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace flashinfer {
namespace mma_hip {

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

}  // namespace mma_hip
}  // namespace flashinfer
