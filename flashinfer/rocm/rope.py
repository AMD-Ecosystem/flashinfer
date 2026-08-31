# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AITER routing for the cos/sin-cache rope ops."""

import functools

import torch


@functools.cache
def get_rope_aiter_module():
    from ..jit.rope import gen_rope_aiter_module

    return gen_rope_aiter_module().build_and_load()


def _auto_select_rope_backend(query: torch.Tensor, key: torch.Tensor) -> str:
    # Measured: the in-tree native kernel beats AITER for the cos/sin-cache
    # rope, so auto resolves to native. AITER stays reachable via backend="aiter".
    return "native"


def _apply_rope_cos_sin_cache_aiter(
    query: torch.Tensor,
    key: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    is_neox: bool,
) -> None:
    r"""Dispatch the cos/sin-cache rope to AITER's C++ rope_cached_positions_2c kernel."""
    from .aiter_utils import require_aiter

    require_aiter(query.device, "rope")
    if key.dtype != query.dtype:
        raise ValueError(
            "AITER rope backend requires query and key to share a dtype; "
            f"got query={query.dtype}, key={key.dtype}. Use backend='native'."
        )
    get_rope_aiter_module().apply_rope_pos_ids_cos_sin_cache_aiter(
        query,
        key,
        query_out,
        key_out,
        cos_sin_cache,
        positions,
        head_size,
        is_neox,
    )


def maybe_apply_rope_cos_sin_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    is_neox: bool,
    backend: str,
) -> bool:
    """Run the AITER rope if `backend` selects it; False means fall through.

    Serves both the out-of-place and in-place call sites: the in-place one
    passes `query`/`key` as their own outputs.
    """
    resolved = backend if backend != "auto" else _auto_select_rope_backend(query, key)
    if resolved != "aiter":
        return False
    _apply_rope_cos_sin_cache_aiter(
        query=query,
        key=key,
        query_out=query_out,
        key_out=key_out,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        head_size=head_size,
        is_neox=is_neox,
    )
    return True
