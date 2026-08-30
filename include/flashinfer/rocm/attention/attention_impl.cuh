// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifdef FLASHINFER_ATTENTION_IMPL_CUH_
#error \
    "include/flashinfer/attention_impl.cuh and include/flashinfer/rocm/attention/attention_impl.cuh define the same symbols; include only one"
#endif

#ifndef FLASHINFER_ROCM_ATTENTION_ATTENTION_IMPL_CUH_
#define FLASHINFER_ROCM_ATTENTION_ATTENTION_IMPL_CUH_

#include "cascade.cuh"
#include "decode.cuh"
#include "default_decode_params.cuh"
#include "default_prefill_params.cuh"
#include "prefill.cuh"
#include "variants.cuh"

#endif  // FLASHINFER_ROCM_ATTENTION_ATTENTION_IMPL_CUH_
