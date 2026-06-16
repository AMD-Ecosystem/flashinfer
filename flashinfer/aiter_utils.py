# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import torch

from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS


@functools.lru_cache(maxsize=8)
def is_aiter_supported(device: torch.device) -> bool:
    """Return True when the given device is an AMD GPU that AITER targets (gfx942/gfx950)."""
    if torch.version.hip is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:
        return False
    return arch in FLASHINFER_SUPPORTED_ROCM_ARCHS


@functools.lru_cache(maxsize=1)
def _aiter_importable() -> bool:
    """True when the AITER source package needed for the C++ backends is importable."""
    import importlib.util

    return (
        importlib.util.find_spec("aiter") is not None
        and importlib.util.find_spec("aiter_meta") is not None
    )


def is_aiter_available(device: torch.device) -> bool:
    """Return True when ``backend="auto"`` may route to AITER for ``device``.

    Combines the arch check with a cheap import probe so ``auto`` falls back to the
    native kernel (rather than raising at build time) when the AITER package is not
    installed. Explicit ``backend="aiter"`` still surfaces a clear error.
    """
    return is_aiter_supported(device) and _aiter_importable()


@functools.cache
def get_aiter_mha_module():
    from aiter.ops import mha as aiter_mha_module

    return aiter_mha_module
