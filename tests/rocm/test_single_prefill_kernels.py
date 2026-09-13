# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from attention_reference import naive_attention
from jit_utils import gen_prefill_attention_modules

import flashinfer

from flashinfer.jit.core import logger
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.prefill import _aiter_ops_importable
import logging

logger.setLevel(logging.ERROR)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16, torch.bfloat16],  # q_dtypes
            [torch.float16, torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0],  # pos_encoding_modes (NONE)
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reduction_options
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("kv_len", [54, 97, 128, 512, 2048])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 8.0])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "aiter"])
def test_single_prefill_with_kv_cache(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    return_lse: bool,
    backend: str,
):
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    if backend == "aiter" and (
        not is_aiter_supported(torch.device("cuda:0")) or not _aiter_ops_importable()
    ):
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    if backend == "aiter" and kv_layout == "HND":
        pytest.skip("AITER does not support HND layout")

    if causal and qo_len > kv_len:
        pytest.skip("causal attention requires kv_len >= qo_len")

    # A non-zero soft cap forces mha_varlen_fwd, whose CK kernel applies the cap
    # wrongly. Non-causal is unaffected, and
    # mha_batch_prefill is exact on the same inputs. Which architectures are
    # affected comes from the capability table, not a literal.
    from flashinfer.rocm.arch_caps import _device_arch, aiter_softcap_defect_arch

    softcap_defective = aiter_softcap_defect_arch(_device_arch(torch.device("cuda:0")))
    if (
        backend == "aiter"
        and logits_soft_cap > 0
        and causal
        and head_dim == 128
        and softcap_defective
    ):
        pytest.skip("AITER mha_varlen_fwd soft-cap defect (aiter<=0.1.21)")

    if kv_layout == "HND":
        k = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
        v_ref = v.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
    else:  # NHD layout
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        k_ref = k
        v_ref = v

    # Call flashinfer API
    logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend=backend,
        )
        assert lse.shape == (qo_len, num_qo_heads)
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend=backend,
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compute reference in FP32 for better accuracy
    o_ref, lse_ref = naive_attention(
        q.float(),
        k_ref.float(),
        v_ref.float(),
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-3, atol=1e-3)
    if return_lse:
        torch.testing.assert_close(
            lse, lse_ref.to(lse.dtype), rtol=1e-3, atol=1e-3
        )  # lse is in fp32


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_single_prefill_threadblock_sync_mdo_states(
    head_dim: int,
    return_lse: bool,
):
    """
    Test case specifically for threadblock_sync_mdo_states validation.
    This config triggers CTA_TILE_Q=16, NUM_WARPS_KV=4, calling threadblock_sync_mdo_states.
    """
    qo_len = 16
    kv_len = 128
    num_qo_heads = 1
    num_kv_heads = 1
    causal = False
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"
    logits_soft_cap = None

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    # Call flashinfer API
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="fa2",
        )
        assert lse.shape == (qo_len, num_qo_heads)
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="fa2",
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compute reference in FP32 for better accuracy
    o_ref, lse_ref = naive_attention(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-3, atol=1e-3)
    if return_lse:
        torch.testing.assert_close(lse, lse_ref.to(lse.dtype), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("qo_len", [37, 127, 577])
@pytest.mark.parametrize("kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_single_prefill_aiter_bf16(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    return_lse: bool,
):
    """AITER single-prefill with bf16 inputs, all of it on the CK Tile arm.

    Every qo_len here is below the asm routing threshold, so this covers the
    mha_fwd (non-varlen, batch-mode) .so loader path -- which the fp16-only
    matrix above does not reach -- and nothing else. The asm arm is covered by
    tests/rocm/test_aiter_asm_routing.py."""
    if not is_aiter_supported(torch.device("cuda:0")) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    if causal and qo_len > kv_len:
        pytest.skip("causal attention requires kv_len >= qo_len")

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.bfloat16
    )
    k = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.bfloat16
    )
    v = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.bfloat16
    )

    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout="NHD",
            backend="aiter",
        )
        assert lse.shape == (qo_len, num_qo_heads)
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=causal, kv_layout="NHD", backend="aiter"
        )
        lse = None

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    o_ref, lse_ref = naive_attention(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=None,
        return_lse=return_lse,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-2, atol=1e-2)
    if return_lse:
        torch.testing.assert_close(lse, lse_ref.to(lse.dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_auto_backend_selects_aiter(head_dim, return_lse):
    """backend='auto' on gfx942/gfx950 with NHD fp16 should route to AITER and be bit-exact."""
    if not is_aiter_supported(torch.device("cuda:0")) or not _aiter_ops_importable():
        pytest.skip(
            "AITER auto-selection only active on gfx942/gfx950 with aiter installed"
        )

    qo_len, kv_len = 64, 128
    num_qo_heads, num_kv_heads = 8, 8

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    if return_lse:
        o_auto, lse_auto = flashinfer.single_prefill_with_kv_cache_return_lse(
            q, k, v, causal=False, kv_layout="NHD", backend="auto"
        )
        o_aiter, lse_aiter = flashinfer.single_prefill_with_kv_cache_return_lse(
            q, k, v, causal=False, kv_layout="NHD", backend="aiter"
        )
        torch.testing.assert_close(o_auto, o_aiter, rtol=0, atol=0)
        torch.testing.assert_close(lse_auto, lse_aiter, rtol=0, atol=0)
    else:
        o_auto = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, kv_layout="NHD", backend="auto"
        )
        o_aiter = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, kv_layout="NHD", backend="aiter"
        )
        torch.testing.assert_close(o_auto, o_aiter, rtol=0, atol=0)


# (causal, logits_soft_cap, head_dim, kv_len, expect_aiter)
_SOFTCAP_ROUTING = [
    (True, 0.0, 128, 512, True),  # no cap: not gated, exact
    (False, 8.0, 128, 512, True),  # non-causal: exact
    (True, 8.0, 64, 512, True),  # other head dims unaffected
    (True, 8.0, 256, 512, True),
    # The capped causal head_dim=128 cases are arch-dependent: gfx950 is wrong
    # at every length, gfx942 at none. None = derive from the table.
    (True, 8.0, 128, 128, None),
    (True, 8.0, 128, 512, None),
    (True, 8.0, 128, 2048, None),
]


@pytest.mark.parametrize(
    "causal,soft_cap,head_dim,kv_len,expect_aiter", _SOFTCAP_ROUTING
)
def test_auto_backend_avoids_aiter_softcap_defect(
    causal, soft_cap, head_dim, kv_len, expect_aiter
):
    """backend='auto' must not route the miscomputed soft-cap case to AITER.

    Guards the routing directly rather than the numerics, because the wrong
    answer is silent: with AITER selected the call returns plausible values.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    if expect_aiter is None:
        from flashinfer.rocm.arch_caps import _device_arch, aiter_softcap_defect_arch

        expect_aiter = not aiter_softcap_defect_arch(_device_arch(device))

    from flashinfer.rocm.prefill import _auto_select_prefill_backend

    chosen, reason = _auto_select_prefill_backend(
        device,
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        kv_layout="NHD",
        has_custom_mask=False,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        # Capability gates differ per op, so the row asserted here has to be the
        # one single prefill actually consults.
        op="single_prefill",
        causal=causal,
        logits_soft_cap=soft_cap,
        kv_len=kv_len,
    )
    assert chosen == ("aiter" if expect_aiter else "fa2"), reason
    # Assert on the reason too, so a fallback for some unrelated cause cannot
    # masquerade as the soft-cap guard working.
    if not expect_aiter:
        assert reason is not None and "logits_soft_cap" in reason, reason


def test_explicit_aiter_backend_rejects_softcap_defect():
    """An explicit backend='aiter' must fail loudly, not return wrong numbers.

    'auto' silently falls back; asking for AITER by name is a deliberate choice,
    so the defect region has to raise rather than degrade.
    """
    from flashinfer.rocm.arch_caps import _device_arch, aiter_softcap_defect_arch

    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    if not aiter_softcap_defect_arch(_device_arch(device)):
        pytest.skip("this architecture is not affected by the soft-cap defect")

    kv_len, qo_len, num_heads, head_dim = 512, 37, 4, 128
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=device)

    with pytest.raises(ValueError, match="logits_soft_cap"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, logits_soft_cap=8.0, backend="aiter"
        )

    # Same shape without the cap must still be served by AITER, so the guard is
    # not quietly disabling the backend outright.
    flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")


@pytest.mark.parametrize(
    "affected,causal,cap,head_dim,kv_len,expected",
    [
        (True, True, 8.0, 128, 1024, True),  # the gated combination
        (False, True, 8.0, 128, 1024, False),  # unaffected arch: never gated
        (True, False, 8.0, 128, 1024, False),  # non-causal is exact
        (True, True, 0.0, 128, 1024, False),  # no cap: not gated
        # None is what single prefill actually passes on the uncapped path.
        (True, True, None, 128, 1024, False),
        (True, True, 8.0, 64, 1024, False),  # other head dims unaffected
        (True, True, 8.0, 128, None, False),  # kv_len=None disarms (paged route)
    ],
)
def test_softcap_predicate_covers_every_branch(
    monkeypatch, affected, causal, cap, head_dim, kv_len, expected
):
    """Exercise _aiter_softcap_defect's matrix without depending on the host GPU.

    On an unaffected architecture every GPU-backed soft-cap test skips, so the
    guard itself would otherwise only be covered by a gfx950 run.
    """
    from flashinfer.rocm import prefill as rocm_prefill

    # _aiter_softcap_defect imports the accessor inside the function body, so the
    # patch has to land on arch_caps itself, not on a prefill-level alias.
    monkeypatch.setattr(
        "flashinfer.rocm.arch_caps.aiter_softcap_defect_arch", lambda arch: affected
    )
    got = rocm_prefill._aiter_softcap_defect(causal, cap, head_dim, kv_len, None)
    assert got is expected


@pytest.mark.parametrize("affected", [True, False])
def test_auto_declines_softcap_only_on_an_affected_arch(monkeypatch, affected):
    """The router must consume the flag, and name it when it declines.

    Arch-independent on purpose: on an unaffected GPU every table-driven row of
    _SOFTCAP_ROUTING resolves to 'aiter', so the decline branch -- and the
    reason string callers grep for -- is otherwise only reached on gfx950.
    """
    from flashinfer.rocm.prefill import _auto_select_prefill_backend

    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    monkeypatch.setattr(
        "flashinfer.rocm.arch_caps.aiter_softcap_defect_arch", lambda arch: affected
    )
    chosen, reason = _auto_select_prefill_backend(
        device,
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        kv_layout="NHD",
        has_custom_mask=False,
        head_dim_qk=128,
        head_dim_vo=128,
        op="single_prefill",
        causal=True,
        logits_soft_cap=8.0,
        kv_len=1024,
    )
    if affected:
        assert chosen == "fa2"
        assert reason is not None and "logits_soft_cap" in reason
    else:
        assert chosen == "aiter", reason


def _softcap_vs_reference(device, qo_len, kv_len, cap, num_heads=4, head_dim=128):
    """max|AITER - fp32 reference| for one causal soft-capped prefill."""
    torch.manual_seed(0)
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)

    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(head_dim)
    if cap > 0:
        s = cap * torch.tanh(s / cap)
    i = torch.arange(qo_len, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    s = s.masked_fill((j > i + (kv_len - qo_len))[None], float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), v.float())

    got = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=True,
        backend="aiter",
        logits_soft_cap=(cap if cap > 0 else None),
    )
    return float((got.float() - ref).abs().max())


def _softcap_arch_or_skip(device):
    """The declared soft-cap status of this GPU, skipping if there is none."""
    from flashinfer.rocm import arch_caps

    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    # _device_arch answers "unknown" on any read failure, and the table makes no
    # claim there -- skip rather than assert numerics it does not cover.
    arch = arch_caps._device_arch(device)
    if arch not in arch_caps._AITER_SOFTCAP_DEFECT_ARCHS:
        pytest.skip(f"no soft-cap measurement declared for arch {arch!r}")
    return arch_caps.aiter_softcap_defect_arch(arch)


@pytest.mark.parametrize("qo_len,kv_len", [(17, 2048), (512, 512), (2048, 2048)])
@pytest.mark.parametrize("cap", [1.0, 8.0, 30.0, 100.0])
def test_aiter_softcap_is_exact_wherever_the_table_allows_it(qo_len, kv_len, cap):
    """On an ungated arch the cap must not make AITER worse than no cap at all.

    Calibrated against the uncapped error on the same inputs rather than a fixed
    constant: baseline bf16 error at the square shapes (~0.025) already exceeds
    any tolerance tight enough to catch the defect, which starts around 0.04.
    Swept over qo_len and cap as well as kv_len -- a square, single-cap sweep
    cannot separate those variables, and an earlier revision was wrong for
    exactly that reason.
    """
    device = torch.device("cuda:0")
    if _softcap_arch_or_skip(device):
        pytest.skip("this architecture gates soft-capped causal prefill entirely")

    uncapped = _softcap_vs_reference(device, qo_len, kv_len, 0.0)
    capped = _softcap_vs_reference(device, qo_len, kv_len, cap)
    assert capped <= max(2 * uncapped, 2e-2), (
        f"cap={cap} err {capped:.4f} vs uncapped {uncapped:.4f}"
    )


@pytest.mark.parametrize("qo_len,kv_len", [(17, 2048), (512, 512)])
def test_gated_architecture_really_is_defective(qo_len, kv_len):
    """The gate must stay justified: on a gated arch the cap must still be wrong.

    Without this nothing re-checks the gate, and a stale one costs 2-5x -- which
    is exactly what this suite failed to catch on gfx942. A failure here means
    re-measure and consider removing the entry, not that the kernel regressed.
    """
    device = torch.device("cuda:0")
    if not _softcap_arch_or_skip(device):
        pytest.skip("this architecture is not gated")

    uncapped = _softcap_vs_reference(device, qo_len, kv_len, 0.0)
    capped = _softcap_vs_reference(device, qo_len, kv_len, 8.0)
    assert math.isnan(capped) or capped > max(10 * uncapped, 2e-2), (
        f"soft cap looks correct here (err {capped:.4f} vs uncapped "
        f"{uncapped:.4f}); re-measure and consider ungating this architecture"
    )
