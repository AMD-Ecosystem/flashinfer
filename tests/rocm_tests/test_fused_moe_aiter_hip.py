# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for flashinfer.aiter_fused_moe, the AITER CK two-stage MoE backend on ROCm.
#
# Note on tolerances: the reference accumulates in float32 while the kernel
# accumulates a K-long and then an inter_dim-long dot product in bf16/fp16
# operands. Relative error against the fp32 reference is ~4e-3 for bf16 and
# ~1e-3 for fp16 across the shapes below; the 2e-2 bound leaves headroom without
# being loose enough to hide a wrong kernel (unshuffled weights, the failure this
# suite exists to catch, land at ~1.3).

import pathlib
import re

import pytest
import torch

import flashinfer
from flashinfer.fused_moe_rocm import (
    _BLOCK_M_THRESHOLDS,
    _FP8_BY_ARCH,
    _fp8_stage2_k_tile,
    _SUPPORTED_BLOCK_M,
    _select_block_m,
    moe_fp8_dtype,
    quantize_moe_weight,
    shuffle_moe_weight,
)
from flashinfer.jit.aiter_source import resolve_aiter_build_arch
from tests.test_helpers.test_helpers import requires_aiter

_ACT = {"silu": torch.nn.functional.silu, "gelu": torch.nn.functional.gelu}

# Bound on fp8 quantization error against the dequantized-weight reference.
# test_fp8_activation_scaling_is_per_token is what pins the *granularity*,
# which this bound is too loose to see.
_FP8_TOL = 0.08


def _moe_ref(x, w1, w2, topk_ids, topk_weights, activation):
    """Float32 reference: gather per expert, gated MLP, routing-weighted sum."""
    act = _ACT[activation]
    xf = x.float()
    acc = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for e in range(w1.shape[0]):
        mask = topk_ids == e
        rows = mask.any(dim=-1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        gate, up = (xf[rows] @ w1[e].float().t()).chunk(2, dim=-1)
        down = (act(gate) * up) @ w2[e].float().t()
        acc.index_add_(
            0, rows, down * (topk_weights * mask).sum(-1)[rows].unsqueeze(-1)
        )
    return acc


def _make_case(M, E, K, I, topk, dtype, device, seed=0xA17E3):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=dtype, device=device) / 8
    w1 = torch.randn(E, 2 * I, K, dtype=dtype, device=device) / 16
    w2 = torch.randn(E, K, I, dtype=dtype, device=device) / 16
    logits = torch.randn(M, E, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    return x, w1, w2, topk_ids.to(torch.int32).contiguous(), topk_weights.contiguous()


def _rel_err(got, ref):
    return (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _dequantize(q, scale):
    return (q.float().view(-1, q.shape[-1]) * scale.view(-1, 1)).view(q.shape)


def _make_fp8_case(M, E, K, I, topk, device, seed=0xA17E3, *, dtype=torch.bfloat16):
    """The unquantized case plus fp8 weights and a dequantized-weight reference."""
    x, w1, w2, ids, weights = _make_case(M, E, K, I, topk, dtype, device, seed)
    w1q, w1s = quantize_moe_weight(w1)
    w2q, w2s = quantize_moe_weight(w2)
    ref = _moe_ref(
        x, _dequantize(w1q, w1s), _dequantize(w2q, w2s), ids, weights, "silu"
    )
    return x, w1q, w1s, w2q, w2s, ids, weights, ref


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M,E,K,I,topk",
    [
        (1, 8, 512, 256, 2),  # single token: below block_m, exercises sort padding
        (7, 8, 512, 256, 2),  # non-power-of-two, below block_m
        (64, 8, 512, 256, 1),  # topk=1
        (256, 32, 1024, 512, 6),
        (1024, 8, 4096, 1408, 2),  # a real MoE layer shape
    ],
)
def test_fused_moe_vs_ref(dtype, M, E, K, I, topk):
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(M, E, K, I, topk, dtype, device)

    # block_m defaults to "auto", so this sweep exercises the selected tile on
    # every shape -- a different tile changes the accumulation order.
    got = flashinfer.aiter_fused_moe(
        x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights
    )

    assert got.shape == (M, K) and got.dtype == dtype
    assert torch.isfinite(got).all()
    assert _rel_err(got, _moe_ref(x, w1, w2, ids, weights, "silu")) < 2e-2


# The measured optimum, gfx942 and gfx950 agreeing at every point:
# (num_tokens, topk, num_experts) -> block_m. Mixtral-8x7B is E=8/topk=2,
# Qwen3-235B is E=128/topk=8. A threshold edit that contradicts this fails.
_MEASURED_BLOCK_M = [
    (1, 2, 8, 32),
    (8, 2, 8, 32),
    (32, 2, 8, 32),
    (128, 2, 8, 64),
    (512, 2, 8, 128),
    (2048, 2, 8, 128),
    (1, 8, 128, 32),
    (8, 8, 128, 32),
    (32, 8, 128, 32),
    (128, 8, 128, 32),
    (512, 8, 128, 64),
    (2048, 8, 128, 128),
    # The two boundary points, which are what the thresholds actually turn on.
    # per_expert=16 still prefers 32 (by 2% on both shapes); per_expert=64 has
    # already flipped to 128 (by 11% and 3%).
    (64, 2, 8, 32),
    (256, 8, 128, 32),
    (256, 2, 8, 128),
    (1024, 8, 128, 128),
]


@pytest.mark.parametrize("num_tokens,topk,num_experts,expected", _MEASURED_BLOCK_M)
def test_select_block_m_matches_measurement(num_tokens, topk, num_experts, expected):
    """Pure arithmetic: no GPU, no aiter."""
    assert _select_block_m(num_tokens, topk, num_experts) == expected


def test_select_block_m_tracks_tokens_per_expert_not_tokens():
    """Same num_tokens, different expert geometry, different tile.

    This is the whole point of the selector: keying on num_tokens alone would
    give Mixtral's answer to Qwen3. The widest divergence is at M=256, two
    steps apart -- and 128 measured 13% slower than 32 on Qwen3 there.
    """
    assert _select_block_m(256, 2, 8) == 128
    assert _select_block_m(256, 8, 128) == 32
    assert _select_block_m(512, 2, 8) == 128
    assert _select_block_m(512, 8, 128) == 64


def test_supported_block_m_matches_the_shim():
    """_SUPPORTED_BLOCK_M must equal the tiles the C++ shim accepts.

    The shim's is_supported_block_m() is the authority -- CK's heuristic
    dispatch TORCH_CHECKs anything else. Growing the Python tuple alone would
    let _select_block_m return a tile that aborts on every large-batch call,
    which no Python-side check can catch.
    """
    shim = (
        pathlib.Path(__file__).parents[2]
        / "flashinfer"
        / "csrc_rocm"
        / "fused_moe_aiter.cu"
    ).read_text()
    body = re.search(r"bool\s+is_supported_block_m\s*\([^)]*\)\s*\{(.*?)\}", shim, re.S)
    assert body, "is_supported_block_m not found in the shim"
    assert tuple(
        int(v) for v in re.findall(r"block_m\s*==\s*(\d+)", body.group(1))
    ) == (_SUPPORTED_BLOCK_M)


def test_select_block_m_only_returns_supported_tiles():
    """Every tile the selector can return is one the shim accepts.

    Paired with the conformance test above, this is what makes the thresholds
    safe to edit: a threshold naming an unsupported tile is a runtime abort,
    not a slow kernel.
    """
    returned = {bm for _, bm in _BLOCK_M_THRESHOLDS} | {_select_block_m(1 << 20, 8, 1)}
    assert returned <= set(_SUPPORTED_BLOCK_M)


def test_select_block_m_never_divides_by_zero():
    """A degenerate weight is the shim's error to report, not ours to crash on."""
    assert _select_block_m(64, 2, 0) in _SUPPORTED_BLOCK_M


@requires_aiter
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_fused_moe_activation_and_block_m(activation, block_m):
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(128, 8, 512, 256, 2, torch.bfloat16, device)

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1),
        shuffle_moe_weight(w2),
        ids,
        weights,
        activation=activation,
        block_m=block_m,
    )

    assert _rel_err(got, _moe_ref(x, w1, w2, ids, weights, activation)) < 2e-2


@requires_aiter
def test_fused_moe_all_tokens_to_one_expert():
    """Degenerate routing: one expert takes everything, the rest take nothing."""
    device = torch.device("cuda:0")
    M, E, K, I = 64, 8, 512, 256
    x, w1, w2, _, _ = _make_case(M, E, K, I, 1, torch.bfloat16, device)
    ids = torch.full((M, 1), 3, dtype=torch.int32, device=device)
    weights = torch.ones(M, 1, dtype=torch.float32, device=device)

    got = flashinfer.aiter_fused_moe(
        x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights
    )

    assert _rel_err(got, _moe_ref(x, w1, w2, ids, weights, "silu")) < 2e-2


@requires_aiter
def test_fused_moe_writes_into_provided_out():
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(64, 8, 512, 256, 2, torch.bfloat16, device)
    out = torch.full((64, 512), 7.0, dtype=torch.bfloat16, device=device)

    got = flashinfer.aiter_fused_moe(
        x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights, out=out
    )

    assert got.data_ptr() == out.data_ptr()
    # Overwritten, not accumulated into: the 7.0 fill must be gone.
    assert _rel_err(out, _moe_ref(x, w1, w2, ids, weights, "silu")) < 2e-2


@requires_aiter
@pytest.mark.parametrize("alias", ["exact", "view", "w1"])
def test_fused_moe_rejects_out_overlapping_an_input(alias):
    """out is zero-filled before the activations are read.

    Aliasing an input therefore feeds stage 1 zeros and returns an all-zero
    result -- correct shape, correct dtype, no error. Measured before the guard:
    100% of output elements zero.
    """
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(64, 8, 512, 256, 2, torch.bfloat16, device)
    w1s, w2s = shuffle_moe_weight(w1), shuffle_moe_weight(w2)
    out = {"exact": x, "view": x[:, :], "w1": w1s.view(-1)[: 64 * 512].view(64, 512)}[
        alias
    ]

    with pytest.raises(RuntimeError, match="must not overlap"):
        flashinfer.aiter_fused_moe(x, w1s, w2s, ids, weights, out=out)


@requires_aiter
def test_fused_moe_rejects_tensors_on_another_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    x, w1, w2, ids, weights = _make_case(
        64, 8, 512, 256, 2, torch.bfloat16, torch.device("cuda:0")
    )
    out = torch.empty(64, 512, dtype=torch.bfloat16, device="cuda:1")

    with pytest.raises(RuntimeError, match="every tensor must be on"):
        flashinfer.aiter_fused_moe(
            x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights, out=out
        )


@requires_aiter
def test_unshuffled_weights_are_wrong():
    """Pin the reason shuffle_moe_weight exists.

    The shuffle preserves shape and dtype, so no layer can detect its absence.
    If a future AITER stops requiring it this test fails loudly, which is the
    point -- the API contract would then be wrong.
    """
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(64, 8, 512, 256, 2, torch.bfloat16, device)

    got = flashinfer.aiter_fused_moe(x, w1, w2, ids, weights)

    assert _rel_err(got, _moe_ref(x, w1, w2, ids, weights, "silu")) > 0.5


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shuffle_moe_weight_is_a_permutation(dtype):
    """Shape and dtype are preserved and no element is lost."""
    device = torch.device("cuda:0")
    w = torch.randn(4, 128, 256, dtype=dtype, device=device)

    s = shuffle_moe_weight(w)

    assert s.shape == w.shape and s.dtype == w.dtype and s.is_contiguous()
    assert torch.equal(s.flatten().sort().values, w.flatten().sort().values)


def test_shuffle_moe_weight_rejects_bad_shapes():
    """Pure tensor-metadata checks: no GPU and no aiter needed."""
    with pytest.raises(ValueError, match="3-D"):
        shuffle_moe_weight(torch.zeros(128, 256, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible"):
        shuffle_moe_weight(torch.zeros(2, 120, 256, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible"):
        shuffle_moe_weight(torch.zeros(2, 128, 250, dtype=torch.bfloat16))


@requires_aiter
@pytest.mark.parametrize(
    "kwargs,cast,match",
    [
        ({"activation": "relu"}, None, "activation must be one of"),
        ({"block_m": 48}, None, 'block_m must be "auto" or one of'),
        ({"block_m": "bogus"}, None, 'block_m must be "auto" or one of'),
        # Rejected in Python, before _get_module: one module is built per dtype,
        # so an unsupported dtype would otherwise trigger a multi-minute compile
        # only to fail afterwards.
        ({}, torch.float32, "hidden_states must be one of"),
    ],
)
def test_fused_moe_rejects_bad_arguments(kwargs, cast, match):
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(8, 8, 512, 256, 2, torch.bfloat16, device)
    if cast is not None:
        x = x.to(cast)
    with pytest.raises(ValueError, match=match):
        flashinfer.aiter_fused_moe(
            x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights, **kwargs
        )


@requires_aiter
def test_fused_moe_rejects_weights_of_the_wrong_dtype():
    """A weight dtype that is neither hidden_states' nor fp8, before the build."""
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(8, 8, 512, 256, 2, torch.bfloat16, device)

    with pytest.raises(ValueError, match="expert weights must be"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1).float(),
            shuffle_moe_weight(w2).float(),
            ids,
            weights,
        )


@requires_aiter
@pytest.mark.parametrize("quantize_w1", [True, False])
def test_fused_moe_rejects_mismatched_weight_dtypes(quantize_w1):
    """Quantizing one weight and not the other, caught before the build.

    _get_module keys on w1 alone, so without this the fp8 case compiles a CK
    module for minutes and only then fails in the shim.
    """
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(8, 8, 512, 256, 2, torch.bfloat16, device)
    w1q, w1s = quantize_moe_weight(w1)
    w2q, w2s = quantize_moe_weight(w2)
    if quantize_w1:
        pair, scales = (w1q, w2), {"w1_scale": w1s, "w2_scale": w2s}
    else:
        pair, scales = (w1, w2q), {}

    with pytest.raises(ValueError, match="must have the same dtype"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(pair[0]),
            shuffle_moe_weight(pair[1]),
            ids,
            weights,
            **scales,
        )


@requires_aiter
@pytest.mark.parametrize(
    "break_scale,match",
    [
        (lambda s: s.half(), "must be float32"),
        (lambda s: s.squeeze(-1), r"must be \["),
        (lambda s: s[:, :-16], r"must be \["),
        (lambda s: s.repeat(1, 1, 2)[..., :1], "must be contiguous"),
    ],
)
def test_fp8_rejects_malformed_weight_scales(break_scale, match):
    """The shim is the only thing that validates scale dtype, shape and layout.

    CK reads the scale buffer at a stride derived from the weight shape, so a
    wrong one is out-of-bounds reads or wrong numbers, not a crash.
    """
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(64, 8, 512, 256, 2, device)

    with pytest.raises(RuntimeError, match=match):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1q),
            shuffle_moe_weight(w2q),
            ids,
            weights,
            w1_scale=break_scale(w1s),
            w2_scale=w2s,
        )


@requires_aiter
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda t: {"ids": t["ids"].to(torch.int64)}, "topk_ids must be int32"),
        (lambda t: {"weights": t["weights"].to(torch.bfloat16)}, "must be float32"),
        (lambda t: {"w2": t["w2"][:, :, : t["w2"].shape[2] // 2]}, "expected w1"),
        (lambda t: {"w1": t["w1"][:-1]}, "experts"),
        (lambda t: {"x": t["x"].unsqueeze(0)}, "must be 2-D"),
        # block_m="auto" reads these shapes; a degenerate rank must still reach
        # the shim's message rather than raising IndexError from the selector.
        (lambda t: {"ids": t["ids"][0, 0].clone()}, "must be 2-D"),
        (lambda t: {"w1": t["w1"][0].clone()}, "must be 3-D"),
        (lambda t: {"x": t["x"][0].clone()}, "must be 2-D"),
        # topk=0 divides by zero inside AITER: without this check the process
        # dies on SIGFPE, which no caller can catch. The upper bound below is
        # the other half of the same range check.
        (
            lambda t: {
                "ids": t["ids"][:, :0].contiguous(),
                "weights": t["weights"][:, :0].contiguous(),
            },
            "topk must be at least 1",
        ),
        (lambda t: {"w1": t["w1"][:1], "w2": t["w2"][:1]}, "exceeds num_experts"),
    ],
)
def test_fused_moe_rejects_bad_tensors(mutate, match):
    """Each bad input must raise a readable error, not abort inside CK."""
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(64, 8, 512, 256, 2, torch.bfloat16, device)
    t = {
        "x": x,
        "w1": shuffle_moe_weight(w1),
        "w2": shuffle_moe_weight(w2),
        "ids": ids,
        "weights": weights,
    }
    t.update(mutate(t))
    with pytest.raises(RuntimeError, match=match):
        flashinfer.aiter_fused_moe(t["x"], t["w1"], t["w2"], t["ids"], t["weights"])


@requires_aiter
@pytest.mark.parametrize(
    "M,E,K,I,topk",
    [
        (1, 8, 512, 256, 2),
        (64, 8, 512, 256, 2),
        (256, 32, 1024, 512, 6),
        # inter_dim=1408 is not a multiple of 256, which gfx950's fp8 stage 2
        # needs at block_m 32 and 64 -- "auto" must step up to 128 rather than
        # raise from inside CK.
        (1024, 8, 4096, 1408, 2),
    ],
)
def test_fp8_moe_vs_dequantized_ref(M, E, K, I, topk):
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, ref = _make_fp8_case(M, E, K, I, topk, device)

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
    )

    assert got.shape == (M, K) and got.dtype == torch.bfloat16
    assert torch.isfinite(got).all()
    assert _rel_err(got, ref) < _FP8_TOL


@requires_aiter
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_fp8_moe_activation_and_block_m(activation, block_m):
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(128, 8, 512, 256, 2, device)
    ref = _moe_ref(
        x, _dequantize(w1q, w1s), _dequantize(w2q, w2s), ids, weights, activation
    )

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
        activation=activation,
        block_m=block_m,
    )

    assert _rel_err(got, ref) < _FP8_TOL


@requires_aiter
@pytest.mark.parametrize("drop", ["w1_scale", "w2_scale", "both"])
def test_fp8_needs_both_scales(drop):
    """CK gates *all* scaling on the scale pointers arriving together, so a
    half-supplied pair silently runs an unscaled GEMM."""
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(64, 8, 512, 256, 2, device)
    scales = {"w1_scale": w1s, "w2_scale": w2s}
    for key in ["w1_scale", "w2_scale"] if drop == "both" else [drop]:
        scales[key] = None

    with pytest.raises(ValueError, match="must both be given"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1q),
            shuffle_moe_weight(w2q),
            ids,
            weights,
            **scales,
        )


@requires_aiter
def test_scales_are_rejected_without_fp8_weights():
    """The other half of the same rule: bf16 weights take no scales."""
    device = torch.device("cuda:0")
    x, w1, w2, ids, weights = _make_case(64, 8, 512, 256, 2, torch.bfloat16, device)
    ones = torch.ones(8, 512, 1, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="must both be given"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1),
            shuffle_moe_weight(w2),
            ids,
            weights,
            w1_scale=ones,
            w2_scale=ones,
        )


@requires_aiter
def test_fp8_rejects_the_other_architectures_encoding():
    """The other board's encoding quantizes, shuffles and runs; only the numbers
    are wrong. Which one is right is a property of the device."""
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(64, 8, 512, 256, 2, device)
    other = next(d for d in _FP8_BY_ARCH.values() if d != moe_fp8_dtype())

    with pytest.raises(ValueError, match="expert weights must be"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1q).view(other),
            shuffle_moe_weight(w2q).view(other),
            ids,
            weights,
            w1_scale=w1s,
            w2_scale=w2s,
        )


# CK's fp8 stage-2 K tile, measured through this shim on both boards by running
# every (inter_dim, block_m) below and recording ok / raised. gfx942 keeps a
# narrow tile for inter_dim <= 192; gfx950 builds no such instance.
_MEASURED_STAGE2_K_TILE = [
    ("gfx942", 32, 192, 64),
    ("gfx942", 32, 320, 128),
    ("gfx942", 128, 1408, 128),
    ("gfx950", 32, 320, 256),
    ("gfx950", 64, 1408, 256),
    ("gfx950", 128, 1408, 128),
    ("gfx950", 128, 192, 128),
]


@pytest.mark.parametrize("arch,block_m,inter_dim,expected", _MEASURED_STAGE2_K_TILE)
def test_fp8_stage2_k_tile_matches_measurement(arch, block_m, inter_dim, expected):
    """Pure arithmetic: no GPU, no aiter."""
    assert _fp8_stage2_k_tile(arch, block_m, inter_dim) == expected


@requires_aiter
@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_fp8_rejects_a_shape_ck_cannot_serve(block_m):
    """inter_dim=1408 is legal on gfx942 at every tile and illegal on gfx950 below 128.

    CK's own rejection names neither dimension, so this asserts the arch-specific
    answer rather than a fixed one.
    """
    device = torch.device("cuda:0")
    inter_dim = 1408
    x, w1q, w1s, w2q, w2s, ids, weights, ref = _make_fp8_case(
        64, 8, 512, inter_dim, 2, device
    )
    call = lambda: flashinfer.aiter_fused_moe(  # noqa: E731
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
        block_m=block_m,
    )

    if inter_dim % _fp8_stage2_k_tile(resolve_aiter_build_arch(), block_m, inter_dim):
        with pytest.raises(ValueError, match="inter_dim must be divisible by"):
            call()
    else:
        assert _rel_err(call(), ref) < _FP8_TOL


@requires_aiter
def test_fp8_rejects_a_model_dim_no_tile_can_serve():
    """Stage 1 steps K by 128 on every tile and both boards, so no tile rescues this.

    Reaching CK instead returns non-finite output on gfx942 rather than raising.
    """
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(64, 8, 544, 512, 2, device)
    assert 544 % 128 and 544 % 32 == 0  # indivisible for CK, still shuffleable

    with pytest.raises(ValueError, match="model_dim must be divisible by 128"):
        flashinfer.aiter_fused_moe(
            x,
            shuffle_moe_weight(w1q),
            shuffle_moe_weight(w2q),
            ids,
            weights,
            w1_scale=w1s,
            w2_scale=w2s,
        )


@requires_aiter
def test_fp8_auto_picks_a_tile_that_can_serve_the_shape():
    """The selector wants 32 here; on gfx950 that tile cannot serve inter_dim=1408.

    "auto" must find a legal tile rather than propagate the rejection, and must
    still be correct on gfx942 where 32 was legal all along.
    """
    device = torch.device("cuda:0")
    assert _select_block_m(64, 2, 8) == 32
    x, w1q, w1s, w2q, w2s, ids, weights, ref = _make_fp8_case(
        64, 8, 512, 1408, 2, device
    )

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
    )

    assert _rel_err(got, ref) < _FP8_TOL


@requires_aiter
def test_fp8_with_float16_activations():
    """fp8 weights against fp16 activations are a distinct set of CK instances.

    Same weight dtype, different C type, so a separate lib and a separate
    dispatch-table entry from the bf16 case every other fp8 test covers.
    """
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, ref = _make_fp8_case(
        64, 8, 512, 256, 2, device, dtype=torch.float16
    )

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
    )

    assert _rel_err(got, ref) < _FP8_TOL


@requires_aiter
def test_fp8_activation_scaling_is_per_token():
    """One token far outside fp8's exponent range, so the rest must not be crushed.

    The outlier has to be this large: e4m3 spans ~2.3e5, so below that a single
    shared scale is nearly as good as per-row and no tolerance separates the two.
    At 1e5 per-token measures 0.06 against per-tensor's 0.75.
    """
    device = torch.device("cuda:0")
    x, w1q, w1s, w2q, w2s, ids, weights, _ = _make_fp8_case(64, 8, 512, 256, 2, device)
    x = x.clone().float()
    x[0] *= 1e5
    x = x.bfloat16()
    ref = _moe_ref(
        x, _dequantize(w1q, w1s), _dequantize(w2q, w2s), ids, weights, "silu"
    )

    got = flashinfer.aiter_fused_moe(
        x,
        shuffle_moe_weight(w1q),
        shuffle_moe_weight(w2q),
        ids,
        weights,
        w1_scale=w1s,
        w2_scale=w2s,
    )

    # Per row: the outlier's own magnitude must not mask the rows it would crush.
    rel = (got.float() - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-9)
    assert rel.max().item() < _FP8_TOL, rel


@requires_aiter
@pytest.mark.parametrize("quantized", [False, True])
def test_zero_tokens_returns_an_empty_result(quantized):
    """An expert-parallel rank can be routed nothing.

    AITER launches a zero-sized grid for that and leaves a sticky HIP error that
    surfaces on a later unrelated op, so the shim has to return before it.
    """
    device = torch.device("cuda:0")
    x, w1, w2, _, _ = _make_case(8, 8, 512, 256, 2, torch.bfloat16, device)
    x = x[:0].contiguous()
    ids = torch.empty(0, 2, dtype=torch.int32, device=device)
    weights = torch.empty(0, 2, dtype=torch.float32, device=device)
    scales = {}
    if quantized:
        w1, w1s = quantize_moe_weight(w1)
        w2, w2s = quantize_moe_weight(w2)
        scales = {"w1_scale": w1s, "w2_scale": w2s}

    got = flashinfer.aiter_fused_moe(
        x, shuffle_moe_weight(w1), shuffle_moe_weight(w2), ids, weights, **scales
    )
    torch.cuda.synchronize()

    assert got.shape == (0, 512)
    # The sticky error the early return exists to prevent lands here, not above.
    torch.zeros(4, device=device).sum().item()


@requires_aiter
def test_fp8_shuffle_matches_aiters_own():
    """The lane width is 16 // element_size, so fp8 takes a different branch than
    the bf16 path shuffle_moe_weight was written for."""
    aiter_shuffle = pytest.importorskip("aiter.ops.shuffle").shuffle_weight
    device = torch.device("cuda:0")
    w = torch.randn(4, 128, 256, dtype=torch.bfloat16, device=device)
    q, _ = quantize_moe_weight(w)

    assert torch.equal(
        shuffle_moe_weight(q).view(torch.uint8),
        aiter_shuffle(q, layout=(16, 16)).view(torch.uint8),
    )
