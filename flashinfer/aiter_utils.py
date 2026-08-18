# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import torch

from .arch_caps import normalize_arch
from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS


@functools.lru_cache(maxsize=8)
def is_aiter_supported(device: torch.device) -> bool:
    """Return True when the given device is an AMD GPU that AITER targets (gfx942/gfx950)."""
    if torch.version.hip is None:
        return False
    try:
        arch = normalize_arch(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        return False
    return arch in FLASHINFER_SUPPORTED_ROCM_ARCHS


@functools.lru_cache(maxsize=1)
def _aiter_importable() -> bool:
    """True when the AITER packages needed for the C++ backends actually import.

    Uses a real import (not ``find_spec``) so a broken or partially-installed AITER
    — where the spec exists but importing the compiled extension fails, e.g. missing
    ROCm deps — is reported as unavailable rather than routing ``auto`` into a path
    that raises at build/load time.
    """
    try:
        import aiter  # noqa: F401
        import aiter_meta  # noqa: F401
        from aiter.jit import core as _core  # noqa: F401
    except Exception:
        return False
    return True


def is_aiter_available(device: torch.device) -> bool:
    """Return True when ``backend="auto"`` may route to AITER for ``device``.

    Combines the arch check with an import probe so ``auto`` falls back to the
    native kernel (rather than raising at build time) when the AITER package is not
    installed or not importable. Explicit ``backend="aiter"`` still surfaces a clear
    error via :func:`require_aiter`.
    """
    return is_aiter_supported(device) and _aiter_importable()


def require_aiter(device: torch.device, op: str) -> None:
    """Validate the explicit ``backend="aiter"`` opt-in, raising a clear error.

    Surfaces a ``ValueError`` (rather than a raw ``ImportError`` from the JIT module
    loader) when the device is unsupported or the AITER package is missing/broken,
    matching the public-API docs that say this mode "requires the aiter package".
    """
    if not is_aiter_supported(device):
        raise ValueError(
            f"AITER {op} requires an AMD gfx942/gfx950 device; got {device}. "
            "Use backend='native' instead."
        )
    if not _aiter_importable():
        raise ValueError(
            f"backend='aiter' for {op} requires the aiter package, which is not "
            "installed or failed to import. Install it (see the AITER Support "
            "section in the README) or use backend='native'."
        )


@functools.cache
def get_aiter_mha_module():
    from aiter.ops import mha as aiter_mha_module

    return aiter_mha_module
