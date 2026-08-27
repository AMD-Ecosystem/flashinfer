# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AITER routing for the activation ops."""

import functools
from typing import Optional

import torch


@functools.cache
def get_silu_and_mul_aiter_module():
    from ..jit.activation import gen_silu_and_mul_aiter_module

    return gen_silu_and_mul_aiter_module().build_and_load()


def _auto_select_silu_and_mul_backend(input: torch.Tensor) -> str:
    # Measured: the in-tree native kernel beats AITER for silu_and_mul, so auto
    # resolves to native. AITER stays reachable via an explicit backend="aiter".
    return "native"


def maybe_silu_and_mul(
    out: torch.Tensor, input: torch.Tensor, backend: str
) -> Optional[torch.Tensor]:
    """Run the AITER kernel if `backend` selects it, else None to fall through."""
    resolved = (
        backend if backend != "auto" else _auto_select_silu_and_mul_backend(input)
    )
    if resolved != "aiter":
        return None
    get_silu_and_mul_aiter_module().silu_and_mul_aiter(out, input)
    return out
