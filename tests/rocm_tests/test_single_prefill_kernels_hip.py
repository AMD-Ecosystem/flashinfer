# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from attention_reference import naive_attention
from jit_utils import gen_prefill_attention_modules

import flashinfer

from flashinfer.jit.core import logger
from flashinfer.aiter_utils import is_aiter_supported
from flashinfer.prefill_rocm import _aiter_ops_importable
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

    # A non-zero soft cap disables AITER's asm paths, leaving mha_varlen_fwd's
    # CK kernel, which applies the cap wrongly. Non-causal is unaffected, and
    # mha_batch_prefill is exact on the same inputs.
    if (
        backend == "aiter"
        and logits_soft_cap > 0
        and causal
        and head_dim == 128
        and kv_len >= 512
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
    """AITER single-prefill with bf16 inputs. Exercises the ASM v3 (bf16+hd128)
    fast path inside aiter::mha_fwd in addition to the CK Tile fallback for
    hd64/hd256. logits_soft_cap is held at 0.0 here so this also covers the
    mha_fwd (non-varlen, batch-mode) .so loader path that has no bf16 coverage
    in the fp16-only matrix above."""
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
    (True, 8.0, 128, 512, False),  # the defect region
    (True, 8.0, 128, 2048, False),
    (True, 0.0, 128, 512, True),  # no cap: asm path, exact
    (False, 8.0, 128, 512, True),  # non-causal: exact
    (True, 8.0, 64, 512, True),  # other head dims unaffected
    (True, 8.0, 256, 512, True),
    (True, 8.0, 128, 128, True),  # short kv unaffected
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

    from flashinfer.prefill_rocm import _auto_select_prefill_backend

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
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

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
