# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AITER routing for the paged-KV append op."""

import functools
from typing import Optional

import torch

from ..utils import register_custom_op, register_fake_op


@functools.cache
def get_page_aiter_module():
    from ..jit.page import gen_page_aiter_module

    return gen_page_aiter_module().build_and_load()


@functools.cache
def _aiter_unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


def _aiter_kv_append_supported(*, dtype: torch.dtype, kv_layout: str) -> bool:
    """Shape/dtype constraints of AITER's reshape_and_cache_flash.

    Gates the explicit ``backend="aiter"`` opt-in, which is now the only way
    in: the shim reads ``page_size`` from ``paged_k_cache.size(1)``, which is
    ``num_kv_heads`` under HND, so a wrong layout silently writes every token
    to the wrong slot.
    """
    return kv_layout == "NHD" and dtype in (torch.float16, torch.bfloat16)


def _auto_select_kv_append_backend(
    device: torch.device,
    *,
    dtype: torch.dtype,
    kv_layout: str,
) -> str:
    """Always 'native': AITER's append is correct here but slower.

    Not expressed in ``arch_caps``, which answers whether a backend *may*
    run; ``backend="aiter"`` still reaches the shim. Same shape as the
    ``rope`` and ``silu_and_mul`` selectors.
    """
    del device, dtype, kv_layout  # signature kept; every input routes native
    return "native"


@register_custom_op(
    "flashinfer::append_paged_kv_cache_aiter",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _aiter_append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> None:
    """Route append to AITER reshape_and_cache_flash via the compiled shim.

    Cache layout is NHD, ``[num_pages, page_size, num_kv_heads, head_dim]``.
    The custom-op wrapper is required: without it Dynamo traces into the
    TORCH_LIBRARY_FRAGMENT shim, whose ``numel()`` raises on symbolic shapes.
    Indices are narrowed to int32 as the native path does.
    """
    unit = _aiter_unit_scale(paged_k_cache.device)
    get_page_aiter_module().append_paged_kv_cache_aiter(
        append_key,
        append_value,
        batch_indices.int(),
        positions.int(),
        paged_k_cache,
        paged_v_cache,
        kv_indices.int(),
        kv_indptr.int(),
        unit,
        unit,
    )


@register_fake_op("flashinfer::append_paged_kv_cache_aiter")
def _fake_aiter_append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> None:
    pass


def maybe_append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str,
    backend: str,
) -> Optional[bool]:
    """Run the AITER append if `backend` selects it; None means fall through."""
    _backend = (
        _auto_select_kv_append_backend(
            paged_k_cache.device, dtype=paged_k_cache.dtype, kv_layout=kv_layout
        )
        if backend == "auto"
        else backend
    )
    if _backend == "aiter":
        if backend == "aiter":
            # Explicit opt-in skips _auto_select_kv_append_backend, so re-check
            # its constraints here. Without this the shim would read page_size
            # from size(1) -- num_kv_heads under HND -- and scatter every token
            # to the wrong slot with no error.
            from ..aiter_utils import require_aiter

            require_aiter(paged_k_cache.device, "append_paged_kv_cache")
            if not _aiter_kv_append_supported(
                dtype=paged_k_cache.dtype, kv_layout=kv_layout
            ):
                raise ValueError(
                    f"backend='aiter' for append_paged_kv_cache requires "
                    f"kv_layout='NHD' and a float16/bfloat16 cache; got "
                    f"kv_layout={kv_layout!r} and dtype={paged_k_cache.dtype}. "
                    f"Use backend='native'."
                )
        # kv_last_page_len is not forwarded to the shim, so the batch-length
        # invariant native enforces (page.cu's kv_indptr.size(0) == B+1) has
        # nowhere else to live. Without it a short kv_indptr is read past its
        # end inside build_slot_mapping_kernel and scatters silently.
        if kv_indptr.numel() != kv_last_page_len.numel() + 1:
            raise ValueError(
                f"kv_indptr must have kv_last_page_len.numel()+1 entries, got "
                f"{kv_indptr.numel()} vs {kv_last_page_len.numel()}."
            )
        _aiter_append_paged_kv_cache(
            append_key,
            append_value,
            batch_indices,
            positions,
            paged_k_cache,
            paged_v_cache,
            kv_indices,
            kv_indptr,
        )
        return True
    if _backend not in ("native",):
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
        )
    return None
