# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts for ROCm, backed by AITER's CK two-stage kernels.

Routing is the caller's responsibility: pass the selected experts and their
weights, as with upstream's ``cutlass_fused_moe``. Only unquantized bfloat16 and
float16 are wired up so far; the quantized paths are a follow-up.

The expert weights must be pre-shuffled with :func:`shuffle_moe_weight` -- see
that function for why this cannot be checked for you.

The entry point is exported as ``flashinfer.aiter_fused_moe``, backend-prefixed
like upstream's ``cutlass_fused_moe``. Plain ``fused_moe`` is unavailable as a
top-level name: ``flashinfer/fused_moe/`` is a shipped subpackage, and importing
it would rebind the attribute from this function to that module.
"""

import functools
from typing import Optional, Union

import torch

from .aiter_utils import require_aiter
from .jit.fused_moe_rocm import SUPPORTED_DTYPES, gen_fused_moe_aiter_module

__all__ = ["aiter_fused_moe", "shuffle_moe_weight"]

# aiter_enum.h ActivationType, mirrored so callers need no aiter import. The keys
# are also the supported-activation set, here and in the JIT spec.
_ACTIVATION_CODE = {"silu": 0, "gelu": 1}

# CK's stage-1 tile height; the heuristic dispatch enumerates exactly these.
_SUPPORTED_BLOCK_M = (32, 64, 128)

# What fills a stage-1 tile is the tokens routed to *one* expert, so the tile
# height tracks num_tokens * topk / num_experts rather than num_tokens. Measured
# optimum on gfx942 and gfx950 across two expert geometries; see the benchmark's
# --block-m-sweep to regenerate.
_BLOCK_M_THRESHOLDS = ((32, 32), (64, 64))


def _select_block_m(num_tokens: int, topk: int, num_experts: int) -> int:
    """Pick the CK tile height from the average tokens routed to one expert."""
    # max(): a degenerate weight is the shim's error to report, not a
    # ZeroDivisionError from here that would hide it.
    per_expert = num_tokens * topk / max(num_experts, 1)
    for limit, block_m in _BLOCK_M_THRESHOLDS:
        if per_expert < limit:
            return block_m
    return _SUPPORTED_BLOCK_M[-1]


# CK's MFMA tile for the weight operand: 16 rows x 16 columns per instruction.
_SHUFFLE_LAYOUT = (16, 16)


@functools.cache
def _get_module(dtype: torch.dtype, activation: str):
    return gen_fused_moe_aiter_module(dtype, activation).build_and_load()


def shuffle_moe_weight(w: torch.Tensor) -> torch.Tensor:
    """Reorder an expert weight into the layout CK's MoE GEMM reads.

    Call once per weight at model load and keep the result. The permutation
    preserves shape and dtype, so nothing downstream can detect its absence:
    unshuffled weights are silently wrong, not an error.

    Args:
        w: ``[num_experts, n, k]``. ``n`` must be a multiple of 16 and ``k`` a
            multiple of 32.

    Returns:
        A new contiguous tensor with the same shape and dtype as ``w``.
    """
    if w.dim() != 3:
        raise ValueError(f"expected a 3-D [num_experts, n, k] weight, got {w.dim()}-D")

    block_n, inst_k = _SHUFFLE_LAYOUT
    block_k = inst_k * 2
    # Elements per 16-byte lane load, which is what the innermost axis groups.
    lane = 16 // w.element_size()
    n, k = w.shape[-2], w.shape[-1]
    if n % block_n or k % block_k:
        raise ValueError(
            f"weight [..., {n}, {k}] must have n divisible by {block_n} and k "
            f"divisible by {block_k} to be shuffled for CK's MoE GEMM"
        )

    return (
        w.view(-1, n // block_n, block_n, k // block_k, block_k // lane, lane)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(w.shape)
    )


def aiter_fused_moe(
    hidden_states: torch.Tensor,
    w1_shuffled: torch.Tensor,
    w2_shuffled: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str = "silu",
    block_m: Union[int, str] = "auto",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused mixture-of-experts forward pass on ROCm.

    Computes, for each token and each of its selected experts :math:`e`,
    ``act(x @ w1_gate[e].T) * (x @ w1_up[e].T) @ w2[e].T``, scaled by the routing
    weight and summed over the ``topk`` experts.

    Both weights must have been passed through :func:`shuffle_moe_weight`; the
    parameters are named for it because the requirement cannot be checked. The
    first call per ``(dtype, activation)`` builds two AITER modules and takes
    minutes, then caches under ``~/.cache/flashinfer/aiter_libs/``.

    Args:
        hidden_states: ``[num_tokens, model_dim]``, bfloat16 or float16.
        w1_shuffled: ``[num_experts, 2 * inter_dim, model_dim]`` -- the gate and
            up projections concatenated along dim 1, gate first, then shuffled.
        w2_shuffled: ``[num_experts, model_dim, inter_dim]``, the down
            projection, shuffled.
        topk_ids: ``[num_tokens, topk]`` int32, the selected experts. Every
            value must be in ``[0, num_experts)``; a drop marker such as ``-1``,
            or a global id against a local expert-parallel shard, reads out of
            bounds. Validating on device would cost a synchronize per call.
        topk_weights: ``[num_tokens, topk]`` float32, the routing weights.
        activation: ``"silu"`` or ``"gelu"``.
        block_m: CK tile height, one of ``(32, 64, 128)``, or ``"auto"`` to
            pick it from the average tokens per expert. Explicit values are
            honoured unchanged.
        out: Optional ``[num_tokens, model_dim]`` destination. Allocated if
            None. Overwritten, not accumulated into, and it may not overlap any
            input -- it is zero-filled before the activations are read.

    Returns:
        ``[num_tokens, model_dim]``, same dtype as ``hidden_states``.

    Raises:
        ValueError: The device or the aiter install cannot serve this op, or
            ``activation``/``block_m``/the dtype is unsupported.
        RuntimeError: A tensor fails the shim's shape, dtype, device,
            contiguity, or aliasing checks.
    """
    require_aiter(hidden_states.device, "fused_moe")

    if activation not in _ACTIVATION_CODE:
        raise ValueError(
            f"activation must be one of {sorted(_ACTIVATION_CODE)}, got {activation!r}"
        )
    if block_m != "auto" and block_m not in _SUPPORTED_BLOCK_M:
        raise ValueError(
            f'block_m must be "auto" or one of {_SUPPORTED_BLOCK_M}, got {block_m!r}'
        )
    if hidden_states.dtype not in SUPPORTED_DTYPES:
        # Ahead of _get_module: one module is built per dtype, so an unsupported
        # one would otherwise compile for minutes before anything rejected it.
        raise ValueError(
            f"hidden_states must be one of {list(SUPPORTED_DTYPES)}, "
            f"got {hidden_states.dtype}"
        )

    if block_m == "auto":
        # Only when the ranks are sane. Indexing a degenerate shape here would
        # raise IndexError and hide the shim's message naming the real problem.
        if hidden_states.dim() == 2 and topk_ids.dim() == 2 and w1_shuffled.dim() == 3:
            block_m = _select_block_m(
                hidden_states.shape[0], topk_ids.shape[1], w1_shuffled.shape[0]
            )
        else:
            block_m = _SUPPORTED_BLOCK_M[0]

    if out is None:
        # Shaped from hidden_states only when its rank is right. Indexing a
        # degenerate one here raises IndexError before the shim can say
        # "hidden_states must be 2-D"; an empty out reaches that check intact.
        shape = tuple(hidden_states.shape) if hidden_states.dim() == 2 else (0, 0)
        out = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)

    module = _get_module(hidden_states.dtype, activation)
    # Skip torch custom-op dispatch, as the other AITER ROCm paths do: AITER is
    # inference-only here and torch.compile support is not required.
    module.fused_moe_aiter.default(
        out,
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_ids,
        topk_weights,
        block_m,
        _ACTIVATION_CODE[activation],
    )
    return out
