# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Arch validation and compiler flags for the ROCm JIT."""

import functools
import os
import warnings
from typing import List, Tuple


def check_rocm_arch() -> None:
    """Validate that this GPU and ROCm can build FlashInfer's kernels.

    Delegates to hip_utils so the ROCm version, the ported-architecture list and
    PYTORCH_ROCM_ARCH are checked in one place. Torch's own architecture list is
    only consulted when PYTORCH_ROCM_ARCH is set; unset, it reports the visible
    cards rather than anything about the build.
    """
    import torch.utils.cpp_extension as torch_cpp_ext

    from ...rocm.hip_utils import validate_flashinfer_rocm_arch

    try:
        validate_flashinfer_rocm_arch(
            # FLASHINFER_ROCM_ARCH_LIST, else the detected devices, else gfx942
            arch_list=None,
            torch_cpp_ext_module=torch_cpp_ext,
            verbose=False,
        )
    except RuntimeError as e:
        raise RuntimeError(f"ROCm architecture validation failed: {e}") from e


def jit_debug_enabled() -> bool:
    """FLASHINFER_JIT_DEBUG=1: build -O0 -g with device asserts on."""
    return os.environ.get("FLASHINFER_JIT_DEBUG", "0") == "1"


@functools.cache
def _warn_debug_build() -> None:
    # Ambient, so a jit-cache wheel or benchmark built in the same shell inherits it.
    warnings.warn(
        "FLASHINFER_JIT_DEBUG=1: JIT modules build -O0 -g; prebuilt AOT and "
        "jit-cache modules are loaded as shipped and stay optimised.",
        UserWarning,
        stacklevel=2,
    )


def build_flags(compilation_context) -> Tuple[List[str], List[str]]:
    """The (cflags, cuda_cflags) hipcc is invoked with.

    `compilation_context` is left unannotated: annotating it as the HIP
    CompilationContext makes mypy reject the call site, which resolves the CUDA
    one under dual-arm analysis -- the error the moved type: ignore suppressed. Arch flags come from
    `compilation_context`, which resolves FLASHINFER_ROCM_ARCH_LIST."""
    # FLASHINFER_JIT_DEBUG=1 builds unoptimised with debug info and asserts on.
    # Unlike CUDA, FLASHINFER_JIT_VERBOSE alone does not imply it.
    debug = jit_debug_enabled()
    if debug:
        _warn_debug_build()
    opt = ["-O0", "-g"] if debug else ["-O3"]
    cflags = [*opt, "-std=c++20", "-Wno-switch-bool"]
    cflags += compilation_context.get_hipcc_flags_list()

    cuda_cflags = [
        *opt,
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
    # Device code only; host sources keep their asserts either way.
    if not debug and os.environ.get("FLASHINFER_JIT_VERBOSE", "0") != "1":
        cuda_cflags += ["-DNDEBUG"]
    return cflags, cuda_cflags
