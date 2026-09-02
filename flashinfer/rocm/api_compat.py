# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Accept upstream's CUDA-only arguments, then refuse them by name.

The ROCm twins declare upstream's full parameter list so a caller pinned to the
CUDA API binds correctly and a default-valued argument costs nothing. An
argument that actually asks for a CUDA feature raises here rather than being
ignored, which would silently return unscaled or unmasked attention.
"""

from __future__ import annotations

from typing import Any

__all__ = ["reject_cuda_only"]


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
    raise NotImplementedError(
        f"{name} is a CUDA-only feature and is not supported on ROCm; "
        f"pass {name}={default!r} or omit it"
    )
