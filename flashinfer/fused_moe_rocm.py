# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts for ROCm, backed by AITER's CK two-stage kernels.

Routing is the caller's responsibility: pass the selected experts and their
weights, as with upstream's ``cutlass_fused_moe``. Expert weights may be
unquantized bfloat16/float16, or fp8 with one scale per output row; the
activations are quantized per token inside the shim.

The expert weights must be pre-shuffled with :func:`shuffle_moe_weight` -- see
that function for why this cannot be checked for you.

The entry point is exported as ``flashinfer.aiter_fused_moe``, backend-prefixed
like upstream's ``cutlass_fused_moe``. Plain ``fused_moe`` is unavailable as a
top-level name: ``flashinfer/fused_moe/`` is a shipped subpackage, and importing
it would rebind the attribute from this function to that module.
"""

import functools
from typing import Optional, Tuple, Union

import torch

from .aiter_utils import require_aiter
from .jit.aiter_source import resolve_aiter_build_arch
from .jit.fused_moe_rocm import SUPPORTED_DTYPES, gen_fused_moe_aiter_module

__all__ = [
    "aiter_fused_moe",
    "moe_fp8_dtype",
    "quantize_moe_weight",
    "shuffle_moe_weight",
]

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

# aiter/utility/dtypes.py: which fp8 encoding the MFMA instructions take. Both
# exist in torch on either machine, so a wrong choice compiles and runs.
_FP8_BY_ARCH = {
    "gfx942": torch.float8_e4m3fnuz,
    "gfx950": torch.float8_e4m3fn,
}
FP8_DTYPES = frozenset(_FP8_BY_ARCH.values())

# CK builds every MoE instance with GemmSpecialization::Default -- no K padding --
# so each GEMM's K must divide its tile exactly. fp8 doubles the tiles over bf16,
# which is why only the quantized path needs checking. Both raise from inside CK.
_FP8_STAGE1_K_TILE = 128


def _select_block_m(num_tokens: int, topk: int, num_experts: int) -> int:
    """Pick the CK tile height from the average tokens routed to one expert."""
    # max(): a degenerate weight is the shim's error to report, not a
    # ZeroDivisionError from here that would hide it.
    per_expert = num_tokens * topk / max(num_experts, 1)
    for limit, block_m in _BLOCK_M_THRESHOLDS:
        if per_expert < limit:
            return block_m
    return _SUPPORTED_BLOCK_M[-1]


def _fp8_stage2_k_tile(arch: str, block_m: int, inter_dim: int) -> int:
    """CK's stage-2 K tile for fp8, mirroring aiter's heuristic dispatch."""
    if arch == "gfx950":
        # gfx950 builds no inter_dim <= 192 instances for fp8, so unlike gfx942
        # there is no narrow tile to fall back to.
        return 256 if block_m < 128 else 128
    return 64 if inter_dim <= 192 else 128


def _fp8_shape_problem(
    arch: str, block_m: int, model_dim: int, inter_dim: int
) -> Optional[str]:
    """Why fp8 cannot serve this shape at this tile, or None if it can."""
    if model_dim % _FP8_STAGE1_K_TILE:
        return (
            f"model_dim must be divisible by {_FP8_STAGE1_K_TILE} for fp8 expert "
            f"weights, got {model_dim}; no block_m helps, since stage 1 steps K "
            f"by {_FP8_STAGE1_K_TILE} on every tile and both architectures"
        )
    k_tile = _fp8_stage2_k_tile(arch, block_m, inter_dim)
    if inter_dim % k_tile:
        return (
            f"inter_dim must be divisible by {k_tile} for fp8 expert weights at "
            f"block_m={block_m} on {arch}, got {inter_dim}"
        )
    return None


# Floor for a weight row's amax, so an all-zero expert does not divide by zero.
_SCALE_EPS = 1e-12

# CK's MFMA tile for the weight operand: 16 rows x 16 columns per instruction.
_SHUFFLE_LAYOUT = (16, 16)


@functools.cache
def _get_module(dtype: torch.dtype, activation: str, weight_dtype: torch.dtype):
    return gen_fused_moe_aiter_module(dtype, activation, weight_dtype).build_and_load()


def moe_fp8_dtype() -> torch.dtype:
    """The fp8 encoding this build's expert weights must use.

    ``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn`` on gfx950; torch has both on
    either machine, so the wrong one runs and is silently wrong. Quantizing on a
    host without the target GPU needs ``FLASHINFER_ROCM_ARCH_LIST`` set, or this
    answers for whatever architecture the shim resolved.
    """
    arch = resolve_aiter_build_arch()
    try:
        return _FP8_BY_ARCH[arch]
    except KeyError:
        raise ValueError(
            f"fp8 fused MoE is not supported on {arch}; "
            f"expected one of {sorted(_FP8_BY_ARCH)}"
        ) from None


def quantize_moe_weight(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an expert weight to fp8 with one scale per output row.

    Produces exactly the ``(weight, scale)`` pair :func:`aiter_fused_moe` takes,
    so callers need not reproduce the scale layout the shim then enforces. Run it
    before :func:`shuffle_moe_weight`, at model load.

    Args:
        w: ``[num_experts, n, k]`` in any float dtype.

    Returns:
        ``(quantized, scale)`` -- ``w.shape`` in :func:`moe_fp8_dtype`, and
        ``[num_experts, n, 1]`` float32.
    """
    if w.dim() != 3:
        raise ValueError(f"expected a 3-D [num_experts, n, k] weight, got {w.dim()}-D")
    fp8 = moe_fp8_dtype()
    flat = w.reshape(-1, w.shape[-1]).float()
    # clamp: an all-zero expert row is legal and would otherwise divide by zero.
    scale = flat.abs().amax(-1, keepdim=True).clamp_min(_SCALE_EPS)
    scale /= torch.finfo(fp8).max
    return (flat / scale).to(fp8).view(w.shape), scale.view(
        w.shape[0], w.shape[1], 1
    ).contiguous()


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
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
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
    first call per configuration builds AITER modules and takes minutes, then
    caches under ``~/.cache/flashinfer/aiter_libs/``.

    fp8 expert weights select CK's per-token path: quantize them offline into
    :func:`moe_fp8_dtype` and pass both scales, and the activations are quantized
    for you, once per stage.

    Args:
        hidden_states: ``[num_tokens, model_dim]``, bfloat16 or float16.
        w1_shuffled: ``[num_experts, 2 * inter_dim, model_dim]`` -- the gate and
            up projections concatenated along dim 1, gate first, then shuffled.
            ``hidden_states.dtype`` or :func:`moe_fp8_dtype`.
        w2_shuffled: ``[num_experts, model_dim, inter_dim]``, the down
            projection, shuffled, same dtype as ``w1_shuffled``.
        topk_ids: ``[num_tokens, topk]`` int32, the selected experts. Every
            value must be in ``[0, num_experts)``; a drop marker such as ``-1``,
            or a global id against a local expert-parallel shard, reads out of
            bounds. Validating on device would cost a synchronize per call.
        topk_weights: ``[num_tokens, topk]`` float32, the routing weights.
        w1_scale: ``[num_experts, 2 * inter_dim, 1]`` float32, one scale per
            output row. Required with fp8 weights and rejected without them --
            CK gates all scaling on the scales arriving together, so a half-set
            pair would run unscaled and return a plausible-looking wrong answer.
        w2_scale: ``[num_experts, model_dim, 1]`` float32, same rule.
        activation: ``"silu"`` or ``"gelu"``.
        block_m: CK tile height, one of ``(32, 64, 128)``, or ``"auto"`` to
            pick it from the average tokens per expert. Explicit values are
            honoured unchanged, except that fp8 on gfx950 rejects 32 and 64
            unless ``inter_dim`` is a multiple of 256 -- CK has no K padding
            there and would raise from inside the GEMM.
        out: Optional ``[num_tokens, model_dim]`` destination. Allocated if
            None. Overwritten, not accumulated into, and it may not overlap any
            input -- it is zero-filled before the activations are read.

    Returns:
        ``[num_tokens, model_dim]``, same dtype as ``hidden_states``.

    Raises:
        ValueError: The device or the aiter install cannot serve this op, or
            ``activation``/``block_m``/a dtype/the scale pair is unsupported.
        RuntimeError: A tensor fails the shim's shape, dtype, device,
            contiguity, or aliasing checks.
    """
    # The fp8 path runs different CK instances and was measured separately, so it
    # gets its own capability row rather than riding on the bf16 evidence.
    require_aiter(
        hidden_states.device,
        "fused_moe_fp8" if w1_shuffled.dtype in FP8_DTYPES else "fused_moe",
    )

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

    weight_dtype = w1_shuffled.dtype
    quantized = weight_dtype in FP8_DTYPES
    # Before _get_module, which keys on w1 alone: a mismatched pair would
    # otherwise compile for minutes and only then fail in the shim.
    if w2_shuffled.dtype != weight_dtype:
        raise ValueError(
            f"w1_shuffled and w2_shuffled must have the same dtype, got "
            f"{weight_dtype} and {w2_shuffled.dtype}"
        )
    if quantized:
        expected = moe_fp8_dtype()
        if weight_dtype != expected:
            raise ValueError(
                f"expert weights must be {expected} on {resolve_aiter_build_arch()}, "
                f"got {weight_dtype}; the two fp8 encodings differ in exponent "
                "bias, so this would run and be silently wrong"
            )
    elif weight_dtype != hidden_states.dtype:
        raise ValueError(
            f"expert weights must be {hidden_states.dtype} or an fp8 dtype, "
            f"got {weight_dtype}"
        )
    # Mirrors the shim's check so it lands before the build, not after it.
    if (w1_scale is not None) + (w2_scale is not None) != (2 if quantized else 0):
        raise ValueError(
            "w1_scale and w2_scale must both be given for fp8 expert weights and "
            f"both omitted otherwise; got weight dtype {weight_dtype} with "
            f"w1_scale={'set' if w1_scale is not None else 'None'} and "
            f"w2_scale={'set' if w2_scale is not None else 'None'}"
        )

    # Degenerate ranks skip the shape reads below so the shim's message names the
    # real problem instead of an IndexError from here.
    shapes_known = (
        hidden_states.dim() == 2 and topk_ids.dim() == 2 and w1_shuffled.dim() == 3
    )
    automatic = block_m == "auto"
    if not automatic:
        tile = int(block_m)
    elif shapes_known:
        tile = _select_block_m(
            hidden_states.shape[0], topk_ids.shape[1], w1_shuffled.shape[0]
        )
    else:
        tile = _SUPPORTED_BLOCK_M[0]

    if quantized and shapes_known and w2_shuffled.dim() == 3:
        arch = resolve_aiter_build_arch()
        model_dim, inter_dim = hidden_states.shape[1], w2_shuffled.shape[2]
        problem = _fp8_shape_problem(arch, tile, model_dim, inter_dim)
        if problem is not None and automatic:
            # Another tile may divide inter_dim even when the chosen one does not;
            # prefer a legal tile over CK's message, which names no dimension.
            for candidate in _SUPPORTED_BLOCK_M:
                if _fp8_shape_problem(arch, candidate, model_dim, inter_dim) is None:
                    tile, problem = candidate, None
                    break
        if problem is not None:
            raise ValueError(problem)

    if out is None:
        # Shaped from hidden_states only when its rank is right. Indexing a
        # degenerate one here raises IndexError before the shim can say
        # "hidden_states must be 2-D"; an empty out reaches that check intact.
        shape = tuple(hidden_states.shape) if hidden_states.dim() == 2 else (0, 0)
        out = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)

    module = _get_module(hidden_states.dtype, activation, weight_dtype)
    # Skip torch custom-op dispatch, as the other AITER ROCm paths do: AITER is
    # inference-only here and torch.compile support is not required.
    module.fused_moe_aiter.default(
        out,
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_ids,
        topk_weights,
        tile,
        _ACTIVATION_CODE[activation],
        w1_scale,
        w2_scale,
    )
    return out
