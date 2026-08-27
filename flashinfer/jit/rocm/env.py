# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Where the ROCm JIT puts its AOT and workspace directories."""

import os
import pathlib
import re
from typing import Callable

from ..._version import __version__ as flashinfer_version
from ...arch_caps import normalize_arch


def get_aot_dir(
    package_root: pathlib.Path, has_jit_cache: Callable[[], bool]
) -> pathlib.Path:
    """Prefer the amd-flashinfer-jit-cache wheel, else the in-tree data dir.

    `has_jit_cache` is passed in rather than imported: env.py calls this while
    still executing its own module body, so importing back into it would cycle.
    The wheel's version may carry a ROCm suffix (0.5.3+rocm6.4), hence startswith.
    """
    if not has_jit_cache():
        return package_root / "data" / "aot"

    import amd_flashinfer_jit_cache

    cache_version = amd_flashinfer_jit_cache.__version__
    if not os.getenv(
        "FLASHINFER_DISABLE_VERSION_CHECK"
    ) and not cache_version.startswith(flashinfer_version):
        raise RuntimeError(
            f"amd-flashinfer-jit-cache version ({cache_version}) does not match "
            f"flashinfer version ({flashinfer_version}). "
            "Please install the same version of both packages. "
            "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
        )
    return pathlib.Path(amd_flashinfer_jit_cache.get_jit_cache_dir())


def get_workspace_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    """``<cache_dir>/<version>/<arch>``, or .../noarch when no GPU is visible."""
    try:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # normalize_arch keeps letter suffixes ("gfx90a"); the gfx\d guard only
        # checks the value looks like an arch name before it is trusted.
        arch = normalize_arch(props.gcnArchName)
        if not re.match(r"gfx\d", arch):
            from torch.utils.cpp_extension import _get_rocm_arch_flags

            archs = [
                f.replace("--offload-arch=", "")
                for f in _get_rocm_arch_flags()
                if f.startswith("--offload-arch=")
            ]
            arch = archs[0] if archs else "noarch"

        # Setting the current device is the caller's job; catch a misconfigured
        # one here rather than at kernel launch.
        from ...hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

        if arch != "noarch" and arch not in FLASHINFER_SUPPORTED_ROCM_ARCHS:
            raise RuntimeError(
                f"torch.cuda.current_device() is device {torch.cuda.current_device()} "
                f"with unsupported ROCm architecture '{arch}'. "
                f"Please set the current device to a supported GPU before importing "
                f"flashinfer (e.g. torch.cuda.set_device(<device_index>)). "
                f"Supported architectures: {', '.join(FLASHINFER_SUPPORTED_ROCM_ARCHS)}"
            )
    except RuntimeError:
        raise
    except Exception:
        arch = "noarch"
    return cache_dir / flashinfer_version / arch
