# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""fp8 paged prefill: routing, the guards around it, and the numbers.

fp8 reaches AITER only through the paged wrapper's native-paging route. Every
other prefill path lands on fa2, whose kernel rejects 8-bit types with a
static_assert, so the guards here are what keep that from surfacing as a
compiler log.
"""

import math

import pytest
import torch

import flashinfer
import flashinfer.rocm.prefill
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.prefill import (
    FP8_PREFILL_DTYPES,
    _aiter_ops_importable,
    _aiter_paged_route_page_sizes,
)

HEAD_DIM = 128
NHQ, NHKV = 32, 8
PAGE = 16


def fp8_dtype():
    import aiter

    return aiter.dtypes.fp8


def _require_aiter(device):
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")


def _quant(t, fp8):
    scale = (t.abs().amax().clamp(min=1e-6) / 240.0).to(torch.float32)
    return (t / scale).to(fp8), scale.reshape(1)


def _plan_and_run(
    device,
    s_qo,
    s_kv,
    dtype,
    fp8,
    backend="auto",
    page=None,
    causal=True,
    logits_soft_cap=0.0,
    **run_kwargs,
):
    page = PAGE if page is None else page
    npages = s_kv // page
    torch.manual_seed(0)
    q = torch.randn(s_qo, NHQ, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        npages, 2, page, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    ref_k = kv[:, 0].reshape(-1, NHKV, HEAD_DIM)[:s_kv]
    ref_v = kv[:, 1].reshape(-1, NHKV, HEAD_DIM)[:s_kv]

    if dtype in FP8_PREFILL_DTYPES:
        q_in, sq = _quant(q, fp8)
        kv_in, skv = _quant(kv, fp8)
        run_kwargs.setdefault("scale_q", sq)
        run_kwargs.setdefault("scale_k", skv)
        run_kwargs.setdefault("scale_v", skv.clone())
    else:
        q_in, kv_in = q, kv

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(
        torch.tensor([0, s_qo], dtype=torch.int32, device=device),
        torch.tensor([0, npages], dtype=torch.int32, device=device),
        torch.arange(npages, dtype=torch.int32, device=device),
        torch.tensor([page], dtype=torch.int32, device=device),
        NHQ,
        NHKV,
        HEAD_DIM,
        page,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return w, w.run(q_in, kv_in, **run_kwargs), (q, ref_k, ref_v)


def _reference(q, k, v, causal=True):
    rep = q.shape[1] // k.shape[1]
    qs = q.permute(1, 0, 2).float()
    ks = k.repeat_interleave(rep, dim=1).permute(1, 0, 2).float()
    vs = v.repeat_interleave(rep, dim=1).permute(1, 0, 2).float()
    s = (qs @ ks.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    sq, sk = s.shape[-2], s.shape[-1]
    m = torch.ones(sq, sk, dtype=torch.bool, device=s.device)
    if causal:
        m = m.tril(sk - sq)
    return ((s.masked_fill(~m, float("-inf"))).softmax(-1) @ vs).permute(1, 0, 2)


@pytest.mark.parametrize("s_qo,s_kv", [(512, 512), (1024, 1024), (2048, 2048)])
def test_fp8_paged_prefill_matches_fp32_reference(s_qo, s_kv):
    """The whole point: fp8 runs on AITER and the numbers are right.

    Tolerance is set against the bf16 result on the same inputs rather than a
    constant -- fp8 error is dominated by the uncalibrated per-tensor descale,
    so a fixed bound would either pass anything or fail on noise. Note the two
    take different routes at PAGE=16: bf16 flat-gathers, fp8 pages natively.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()

    w8, out8, (q, k, v) = _plan_and_run(device, s_qo, s_kv, fp8, fp8)
    assert w8._backend == "aiter", w8._backend_fallback_reason
    assert out8.dtype == torch.bfloat16, "AITER has no fp8-output prefill kernel"

    ref = _reference(q, k, v)
    err8 = float((out8.float() - ref).abs().max())
    _, out16, _ = _plan_and_run(device, s_qo, s_kv, torch.bfloat16, fp8)
    err16 = float((out16.float() - ref).abs().max())
    # Generous, but far below the ~2.0 a dropped descale produces.
    assert err8 < max(40 * err16, 0.5), f"fp8 err {err8:.4f} vs bf16 {err16:.4f}"


@pytest.mark.parametrize("page", [1, 16, 1024])
def test_every_routed_fp8_page_size_is_numerically_right(page):
    """Page size is an AITER dispatch axis, so set membership is not coverage.

    The routed set is {1, 16, 1024}; a kernel that exists for one of them can
    still be wrong or missing for another, which is the failure mode this whole
    change is about.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    if page not in _aiter_paged_route_page_sizes(fp8):
        pytest.skip(f"page_size={page} is not routed natively on this build")

    s_kv = max(page, 1024)
    w, out, (q, k, v) = _plan_and_run(device, 512, s_kv, fp8, fp8, page=page)
    assert w._backend == "aiter", w._backend_fallback_reason

    ref = _reference(q, k, v)
    assert torch.isfinite(out.float()).all(), "fp8 output has NaN/Inf"
    assert float((out.float() - ref).abs().max()) < 0.5


def test_a_wrapper_that_once_chose_fa2_can_still_reach_fp8():
    """plan() went concrete on the first call and never re-resolved `auto`.

    So a wrapper whose first plan hit any fa2 constraint refused fp8 for the
    rest of its life -- and a served wrapper is re-planned every step.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    s_kv, npages = 512, 512 // PAGE
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="auto")
    args = (
        torch.tensor([0, s_kv], dtype=torch.int32, device=device),
        torch.tensor([0, npages], dtype=torch.int32, device=device),
        torch.arange(npages, dtype=torch.int32, device=device),
        torch.tensor([PAGE], dtype=torch.int32, device=device),
        NHQ,
        NHKV,
        HEAD_DIM,
        PAGE,
    )

    # A custom mask is an AITER constraint, so this plan resolves to fa2.
    wrapper.plan(
        *args,
        custom_mask=torch.ones(s_kv * s_kv, dtype=torch.bool, device=device),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    assert wrapper._backend == "fa2", "test premise: the first plan must pick fa2"

    wrapper.plan(*args, causal=True, q_data_type=fp8, kv_data_type=fp8)

    assert wrapper._backend == "aiter", wrapper._backend_fallback_reason


def test_an_unreadable_fp8_encoding_is_refused_not_guessed(monkeypatch):
    """Failing open here dispatches data under the wrong exponent bias.

    `_aiter_ops_importable()` only proves `aiter.ops` imports, so
    `aiter.dtypes` can be absent while the AITER backend is still selected.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    monkeypatch.setattr(flashinfer.rocm.prefill, "_native_fp8_dtype", lambda: None)

    with pytest.raises(NotImplementedError, match="unreadable"):
        _plan_and_run(device, 512, 512, fp8, fp8)


def test_non_causal_fp8_selects_its_own_variant_and_is_right():
    """`needs_mask` picks a different .so, so causal coverage is not coverage of
    the `_nmask` arm -- a naming or bootstrap defect there would reach a caller
    on their first non-causal call."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()

    w, out, (q, k, v) = _plan_and_run(device, 512, 512, fp8, fp8, causal=False)

    assert w._backend == "aiter", w._backend_fallback_reason
    ref = _reference(q, k, v, causal=False)
    assert torch.isfinite(out.float()).all(), "fp8 non-causal output has NaN/Inf"
    assert float((out.float() - ref).abs().max()) < 0.5


@pytest.mark.parametrize("causal", [True, False])
def test_a_soft_capped_fp8_call_reaches_the_logits_variant(causal):
    """`has_logits_cap` is the other axis in the .so name. Asserted against the
    uncapped result rather than a reference: what matters is that the cap
    reached the kernel, and a dropped cap would return the uncapped answer."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()

    w_cap, capped, _ = _plan_and_run(
        device, 512, 512, fp8, fp8, causal=causal, logits_soft_cap=1.0
    )
    _, uncapped, _ = _plan_and_run(device, 512, 512, fp8, fp8, causal=causal)

    assert w_cap._backend == "aiter", w_cap._backend_fallback_reason
    assert torch.isfinite(capped.float()).all(), "capped fp8 output has NaN/Inf"
    assert not torch.allclose(capped.float(), uncapped.float(), atol=1e-2), (
        "a soft cap of 1.0 changed nothing; the _logits variant is not in use"
    )


def test_fp8_ignoring_descales_would_be_caught():
    """A/B for the test above: wrong descales must fail it.

    AITER silently accepts a descale it does not honour elementwise, so the
    numeric test is only meaningful if a bad scale actually moves the result.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()

    _, good, (q, k, v) = _plan_and_run(device, 512, 512, fp8, fp8)
    wrong = torch.full((1,), 4.0, dtype=torch.float32, device=device)
    _, bad, _ = _plan_and_run(device, 512, 512, fp8, fp8, scale_q=wrong)
    assert not torch.allclose(good.float(), bad.float(), atol=1e-2), (
        "a 4x q_descale changed nothing; the kernel is ignoring it"
    )


def _fp8_kv_with_distinct_scales(device, fp8, s_qo=512, s_kv=512):
    """Quantize K and V separately, so their descales differ by ~8x.

    The shared helper scales the whole KV tensor at once, which makes a swapped
    K/V descale pointer cancel out and stay invisible.
    """
    npages = s_kv // PAGE
    torch.manual_seed(0)
    q = torch.randn(s_qo, NHQ, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(npages, PAGE, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = 8.0 * torch.randn(
        npages, PAGE, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    q8, sq = _quant(q, fp8)
    k8, sk = _quant(k, fp8)
    v8, sv = _quant(v, fp8)
    assert float(sv) > 4 * float(sk), "test premise: the two scales must differ"

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="auto")
    wrapper.plan(
        torch.tensor([0, s_qo], dtype=torch.int32, device=device),
        torch.tensor([0, npages], dtype=torch.int32, device=device),
        torch.arange(npages, dtype=torch.int32, device=device),
        torch.tensor([PAGE], dtype=torch.int32, device=device),
        NHQ,
        NHKV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=fp8,
        kv_data_type=fp8,
    )
    ref = _reference(q, k.reshape(-1, NHKV, HEAD_DIM), v.reshape(-1, NHKV, HEAD_DIM))
    return wrapper, torch.stack((k8, v8), dim=1), q8, (sq, sk, sv), ref


def test_the_k_and_v_descales_reach_their_own_operands():
    """K and V carry different scales here, so swapping them changes the answer
    instead of cancelling -- which is what pins the two pointers in the shim."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    wrapper, kv8, q8, (sq, sk, sv), ref = _fp8_kv_with_distinct_scales(device, fp8)

    good = wrapper.run(q8, kv8, scale_q=sq, scale_k=sk, scale_v=sv)
    swapped = wrapper.run(q8, kv8, scale_q=sq, scale_k=sv, scale_v=sk)

    err = float((good.float() - ref).abs().max())
    assert err < 40 * float(sv), f"correct descales disagree with the reference: {err}"
    assert not torch.allclose(good.float(), swapped.float(), atol=1e-2), (
        "swapping scale_k and scale_v changed nothing; the shim is not "
        "distinguishing the two descale pointers"
    )


@pytest.mark.parametrize("omitted", ["scale_q", "scale_k", "scale_v"])
def test_every_descale_is_required_for_fp8(omitted):
    """AITER's fp8 kernels have no no-scale instance, so a missing descale must
    be refused rather than passed through as a null pointer."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    wrapper, kv8, q8, (sq, sk, sv), _ = _fp8_kv_with_distinct_scales(device, fp8)
    scales = {"scale_q": sq, "scale_k": sk, "scale_v": sv}
    del scales[omitted]

    with pytest.raises(RuntimeError, match="requires q/k/v descales"):
        wrapper.run(q8, kv8, **scales)


def test_descales_are_refused_for_a_non_fp8_query():
    """The other direction: scales on a bf16 call would be silently ignored."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    one = torch.ones(1, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError, match="only meaningful for an fp8 query"):
        _plan_and_run(device, 512, 512, torch.bfloat16, fp8, scale_q=one)


def test_fp8_rejects_per_head_descale():
    """Per-head descales are silently mis-applied by the per-tensor kernel."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    per_head = torch.ones(NHQ, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="per-tensor"):
        _plan_and_run(device, 512, 512, fp8, fp8, scale_q=per_head)


def test_fp8_rejects_return_lse():
    """AITER builds no LSE instance of the fp8 kernel at any page size."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    with pytest.raises(NotImplementedError, match="LSE"):
        _plan_and_run(device, 512, 512, fp8, fp8, return_lse=True)


def test_fp8_on_a_non_routed_page_size_raises_rather_than_running():
    """fp8 has no flat-gather kernel, so a non-routed page size must not run.

    Exercises plan() rather than asserting set membership: the demotion to fa2
    happens after the first fp8 check, so only a real call proves it is caught
    before the kernel's static_assert reaches the user as a compiler log.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    assert 32 not in _aiter_paged_route_page_sizes(fp8), "test premise"
    # NotImplementedError specifically: a bare RuntimeError is what the AITER
    # bootstrap and the ninja path raise, so accepting those would let the
    # regression this guards against pass.
    with pytest.raises(NotImplementedError, match="fp8 prefill"):
        _plan_and_run(device, 512, 512, fp8, fp8, page=32)


@pytest.mark.parametrize("s_qo", [4, 512])
@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_a_non_routed_page_size_is_refused_on_either_backend(backend, s_qo):
    """An explicit `aiter` is not demotable, so it reaches AITER's own bootstrap.

    Before the guard that answered `RuntimeError: invalid argument for fmha_fwd`
    -- not the ninja log the earlier tests chase, but not the documented
    contract either.
    """
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    # s_qo=4 also crosses the short-query gate, which picks fa2 first; the
    # message must still name the page size rather than fa2's missing kernel.
    with pytest.raises(NotImplementedError, match="flat-gather"):
        _plan_and_run(device, s_qo, 1024, fp8, fp8, backend=backend, page=32)


@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_ragged_fp8_is_refused_on_either_backend(backend):
    """Ragged is mha_varlen_fwd on both backends, and it has no fp8 kernel."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", backend=backend
    )

    with pytest.raises(NotImplementedError, match="ragged batch prefill"):
        wrapper.plan(
            torch.tensor([0, 64], dtype=torch.int32, device=device),
            torch.tensor([0, 64], dtype=torch.int32, device=device),
            NHQ,
            NHKV,
            HEAD_DIM,
            causal=True,
            q_data_type=fp8,
            kv_data_type=fp8,
        )


def test_single_prefill_fp8_raises_instead_of_a_ninja_log():
    """fa2 rejects 8-bit types in a static_assert; that must not reach the user."""
    device = torch.device("cuda:0")
    _require_aiter(device)
    fp8 = fp8_dtype()
    q = torch.randn(128, NHQ, HEAD_DIM, dtype=torch.bfloat16, device=device).to(fp8)
    k = torch.randn(128, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device).to(fp8)
    v = torch.randn(128, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device).to(fp8)
    with pytest.raises(NotImplementedError, match="fp8"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")


def test_routing_keeps_fp16_and_bf16_on_their_existing_route():
    """Correcting the capability set must not re-route the dtypes it measured.

    No GPU or aiter import needed: this is a property of the routing table.
    """
    for dtype in (torch.float16, torch.bfloat16):
        assert 16 not in _aiter_paged_route_page_sizes(dtype)
    assert 16 in _aiter_paged_route_page_sizes(FP8_PREFILL_DTYPES[0])


def test_non_native_fp8_encoding_is_rejected():
    """The other 8-bit encoding is read under the wrong bias and returns NaN."""
    from flashinfer.rocm.prefill import _native_fp8_dtype, _require_native_fp8_dtype

    device = torch.device("cuda:0")
    _require_aiter(device)
    native = _native_fp8_dtype()
    if native is None:
        pytest.skip("aiter not importable")
    other = next(d for d in FP8_PREFILL_DTYPES if d != native)
    _require_native_fp8_dtype(native)  # must not raise
    with pytest.raises(NotImplementedError, match="exponent bias"):
        _require_native_fp8_dtype(other)
