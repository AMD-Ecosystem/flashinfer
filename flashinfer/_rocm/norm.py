# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AITER routing for the RMS-norm ops."""

import functools
from typing import Optional

import torch


@functools.cache
def get_norm_aiter_module():
    from ..jit.norm import gen_norm_aiter_module

    return gen_norm_aiter_module().build_and_load()


def _auto_select_norm_backend(input: torch.Tensor, weight: torch.Tensor) -> str:
    # CK rmsnorm2d only accepts 2D fp16/bf16 and reads weight with the input
    # dtype, so anything else routes native.
    from ..aiter_utils import is_aiter_available

    if (
        input.ndim == 2
        and input.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input.dtype
        and is_aiter_available(input.device, "rmsnorm")
    ):
        return "aiter"
    return "native"


def _auto_select_fused_add_rmsnorm_backend(input: torch.Tensor) -> str:
    # auto routes fused_add_rmsnorm to the C++ AITER CK kernel on supported
    # devices and falls back to native everywhere else (incl. when AITER is not
    # installed, so auto never raises). (Shape/precision tuning is deferred to
    # a later performance pass.)
    from ..aiter_utils import is_aiter_available

    return (
        "aiter" if is_aiter_available(input.device, "fused_add_rmsnorm") else "native"
    )


def maybe_rmsnorm(
    out: Optional[torch.Tensor],
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    backend: str,
) -> Optional[torch.Tensor]:
    """Run the AITER rmsnorm if `backend` selects it; None means fall through."""
    resolved = (
        backend if backend != "auto" else _auto_select_norm_backend(input, weight)
    )
    if resolved == "aiter":
        from ..aiter_utils import require_aiter

        require_aiter(input.device, "rmsnorm")
        if input.ndim != 2:
            raise ValueError(
                f"AITER rmsnorm only supports 2D inputs; got {input.ndim}D. "
                "Use backend='native' for 3D inputs."
            )
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"AITER rmsnorm only supports float16/bfloat16 inputs; got {input.dtype}."
            )
        if weight.dtype != input.dtype:
            # CK rmsnorm2d derives a single dtype from input and reads weight
            # bytes with it; a mismatched weight dtype silently yields NaN/garbage.
            raise ValueError(
                f"AITER rmsnorm requires weight.dtype == input.dtype; got "
                f"weight {weight.dtype} vs input {input.dtype}."
            )
        if out is None:
            out = torch.empty_like(input)
        get_norm_aiter_module().rmsnorm_aiter(out, input, weight, eps)
        return out
    if resolved not in ("native",):
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
        )
    return None


def maybe_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    backend: str,
) -> bool:
    """Run the AITER fused_add_rmsnorm if `backend` selects it; False falls through."""
    resolved = (
        backend if backend != "auto" else _auto_select_fused_add_rmsnorm_backend(input)
    )
    if resolved == "aiter":
        from ..aiter_utils import require_aiter

        require_aiter(input.device, "fused_add_rmsnorm")
        get_norm_aiter_module().fused_add_rmsnorm_aiter(input, residual, weight, eps)
        return True
    if resolved != "native":
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
        )
    return False
