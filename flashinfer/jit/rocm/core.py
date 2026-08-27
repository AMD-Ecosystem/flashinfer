# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Arch validation and compiler flags for the ROCm JIT."""

import os
from typing import List, Tuple


def check_rocm_arch() -> None:
    """Validate that this GPU, ROCm and torch can build FlashInfer's kernels.

    Delegates to hip_utils so the ROCm version, the ported-architecture list and
    torch's build architectures are checked in one place.
    """
    import torch.utils.cpp_extension as torch_cpp_ext

    from ...hip_utils import validate_flashinfer_rocm_arch

    try:
        validate_flashinfer_rocm_arch(
            arch_list=None,  # Uses FLASHINFER_ROCM_ARCH_LIST env or defaults to gfx942
            torch_cpp_ext_module=torch_cpp_ext,
            verbose=False,
        )
    except RuntimeError as e:
        raise RuntimeError(f"ROCm architecture validation failed: {e}") from e


def build_flags(compilation_context) -> Tuple[List[str], List[str]]:
    """The (cflags, cuda_cflags) hipcc is invoked with.

    `compilation_context` is left unannotated: annotating it as the HIP
    CompilationContext makes mypy reject the call site, which resolves the CUDA
    one under dual-arm analysis -- the error the moved type: ignore suppressed. Arch flags come from
    `compilation_context`, which resolves FLASHINFER_ROCM_ARCH_LIST."""
    cflags = ["-O3", "-std=c++20", "-Wno-switch-bool"]
    cflags += compilation_context.get_hipcc_flags_list()

    cuda_cflags = [
        "-O3",
        "-std=c++20",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-ffast-math",
        # clang's -ffast-math implies -ffinite-math-only, which breaks kernels
        # using -inf as a sentinel (online-softmax Map+Reduce). CUDA's
        # -use_fast_math does not, hence the explicit re-enable.
        "-fno-finite-math-only",
    ]
    # HIP has no debug-only counterpart to the NVCC flags the CUDA arm adds.
    if os.environ.get("FLASHINFER_JIT_VERBOSE", "0") != "1":
        cuda_cflags += ["-DNDEBUG"]
    return cflags, cuda_cflags
