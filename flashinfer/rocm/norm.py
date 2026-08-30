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


# Both selectors return "native" unconditionally: measured, not assumed. See
# benchmarks/rocm/bench_norm.py, which is the re-runnable evidence.
# At every 8-aligned hidden_size -- every width a real model uses -- AITER won
# 0 of 230 configs on gfx942 and 0 of 226 on gfx950. `backend="aiter"` stays
# supported and opt-in, as it is for silu_and_mul and rope.


def _aiter_odd_hidden_defect(input: torch.Tensor) -> bool:
    """True when AITER's rmsnorm kernels return wrong results for this shape.

    Odd hidden sizes mis-handle the non-vectorized tail: measured on amd-aiter
    0.1.20 against an fp32 reference, abs error ~1-7 where native is ~0.015.
    Only reachable via an explicit backend="aiter"; auto is native regardless.
    """
    return input.ndim == 2 and input.shape[-1] % 2 != 0


def _auto_select_norm_backend(input: torch.Tensor, weight: torch.Tensor) -> str:
    # A wash on speed (median 1.00 gfx942 / 0.98 gfx950) and native is the more
    # accurate kernel, so there is nothing to trade.
    del input, weight  # signature kept: callers and tests pass them
    return "native"


def _auto_select_fused_add_rmsnorm_backend(input: torch.Tensor) -> str:
    # AITER is 1.6-1.8x slower here: correctness required staging two extra
    # buffers (PR #331), which doubles traffic on a bandwidth-bound kernel.
    del input  # signature kept: callers and tests pass it
    return "native"


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
        # Rank first: stride(-1) on a 0-D weight raises IndexError, not this.
        if weight.ndim != 1 or weight.numel() != input.size(-1):
            # A short weight is an out-of-bounds read inside CK, not an error.
            raise ValueError(
                f"AITER rmsnorm requires a 1-D weight of length "
                f"{input.size(-1)}; got shape {tuple(weight.shape)}."
            )
        if weight.stride(-1) != 1:
            # reshape({1, -1}) in the shim keeps the stride, so CK would read a
            # strided weight as if it were packed.
            raise ValueError(
                f"AITER rmsnorm requires a contiguous weight; got stride "
                f"{weight.stride()}."
            )
        if weight.device != input.device:
            raise ValueError(
                "AITER rmsnorm requires input and weight on the same device; "
                f"got {input.device} and {weight.device}."
            )
        if _aiter_odd_hidden_defect(input):
            raise ValueError(
                f"AITER rmsnorm returns wrong results for odd hidden sizes; got "
                f"{input.shape[-1]}. Use backend='native'."
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


def _check_aiter_fused_add_args(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    """The native kernel enforces these in C++; the AITER path enforced nothing,
    so a mismatch reached CK and came back as garbage rather than an error."""
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"AITER fused_add_rmsnorm only supports float16/bfloat16 inputs; "
            f"got {input.dtype}."
        )
    if weight.dtype != input.dtype:
        # CK derives one dtype from input and reads weight bytes with it.
        raise ValueError(
            f"AITER fused_add_rmsnorm requires weight.dtype == input.dtype; got "
            f"weight {weight.dtype} vs input {input.dtype}."
        )
    if residual.dtype != input.dtype:
        raise ValueError(
            f"AITER fused_add_rmsnorm requires residual.dtype == input.dtype; got "
            f"residual {residual.dtype} vs input {input.dtype}."
        )
    if residual.shape != input.shape:
        raise ValueError(
            f"AITER fused_add_rmsnorm requires residual.shape == input.shape; got "
            f"residual {tuple(residual.shape)} vs input {tuple(input.shape)}."
        )
    if weight.ndim != 1 or weight.numel() != input.size(-1):
        raise ValueError(
            f"AITER fused_add_rmsnorm requires a 1-D weight of length "
            f"{input.size(-1)}; got shape {tuple(weight.shape)}."
        )
    if residual.device != input.device or weight.device != input.device:
        raise ValueError(
            "AITER fused_add_rmsnorm requires input, residual and weight on the "
            f"same device; got {input.device}, {residual.device}, {weight.device}."
        )
    if input.stride(-1) != 1 or residual.stride(-1) != 1:
        raise ValueError(
            "AITER fused_add_rmsnorm requires a contiguous last dimension on "
            "input and residual."
        )
    if weight.stride(-1) != 1:
        # reshape({1, -1}) in the shim keeps the stride, so CK would read a
        # strided weight as if it were packed.
        raise ValueError(
            "AITER fused_add_rmsnorm requires a contiguous weight; got stride "
            f"{weight.stride()}."
        )


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
        _check_aiter_fused_add_args(input, residual, weight)
        if _aiter_odd_hidden_defect(input):
            raise ValueError(
                f"AITER fused_add_rmsnorm returns wrong results for odd hidden "
                f"sizes; got {input.shape[-1]}. Use backend='native'."
            )
        get_norm_aiter_module().fused_add_rmsnorm_aiter(input, residual, weight, eps)
        return True
    if resolved != "native":
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
        )
    return False
