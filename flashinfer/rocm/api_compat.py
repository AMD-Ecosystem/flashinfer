# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Accept upstream's CUDA-only arguments, then refuse them by name.

The ROCm twins declare upstream's full parameter list so a caller pinned to the
CUDA API binds correctly and a default-valued argument costs nothing. An
argument that actually asks for a CUDA feature raises here rather than being
ignored, which would silently return unscaled or unmasked attention; the one
exception is ``enable_pdl``, a scheduling hint whose absence changes no result.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

__all__ = ["ignore_enable_pdl", "reject_cuda_only"]


_MISSING = object()


def reject_cuda_only(
    name: str, value: Any, default: Any = None, *, neutral: Any = _MISSING
) -> None:
    """Raise if *value* differs from *default* and from *neutral*.

    *neutral* is a second value that means "not requested" -- ``False`` for a
    tri-state enable flag, ``1.0`` for a calibration scale -- so a caller
    threading a config through unconditionally is not refused for asking that
    the feature be off. ``==`` is applied only to scalars: a tensor compares
    elementwise, and the result has no truth value.
    """
    for accepted in (default, neutral):
        if accepted is _MISSING:
            continue
        if value is accepted:
            return
        if isinstance(value, (bool, int, float, str)) and value == accepted:
            return
    allowed = f"{default!r}" if neutral is _MISSING else f"{default!r} or {neutral!r}"
    raise NotImplementedError(
        f"{name} is a CUDA-only feature and is not supported on ROCm; "
        f"pass {name}={allowed}, or omit it"
    )


def ignore_enable_pdl(enable_pdl: Optional[bool]) -> bool:
    """Resolve ``enable_pdl`` to False, warning (deduplicated) if it was True.

    ROCm has no programmatic dependent launch. The default (None) stays silent:
    upstream resolves it from ``device_support_pdl()``, which is True on gfx9.
    """
    if enable_pdl:
        warnings.warn(
            "enable_pdl is not supported on ROCm and is ignored.",
            UserWarning,
            stacklevel=3,
        )
    return False
