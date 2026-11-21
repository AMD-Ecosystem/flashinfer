# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# How `auto` routes tensor-core decode: AITER at one query row per request,
# fa2 wherever AITER cannot serve, with the reason recorded.

import logging

import pytest
import torch

import flashinfer
from flashinfer.jit.core import logger
from flashinfer.rocm import decode as decode_rocm
from tests.test_helpers.test_helpers import requires_aiter

logger.setLevel(logging.ERROR)

_DTYPE, _QO, _KV, _HD = torch.bfloat16, 32, 8, 128


def _paged(batch, kv_len, page, device):
    npages = (kv_len + page - 1) // page
    kv = torch.randn(batch * npages, 2, page, _KV, _HD, dtype=_DTYPE, device=device)
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * npages
    indices = torch.arange(batch * npages, dtype=torch.int32, device=device)
    last = torch.full(
        (batch,), kv_len - (npages - 1) * page, dtype=torch.int32, device=device
    )
    return kv, indptr, indices, last


def _wrapper(device, **kw):
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", **kw)


def _plan(w, indptr, indices, last, page, q_len_per_req=1):
    w.plan(
        indptr,
        indices,
        last,
        _QO,
        _KV,
        _HD,
        page,
        q_data_type=_DTYPE,
        kv_data_type=_DTYPE,
        q_len_per_req=q_len_per_req,
    )


@requires_aiter
def test_auto_tensor_core_decode_resolves_per_plan():
    """q=1 takes AITER and matches fa2's tensor-core kernel; q=4 stays on fa2
    with a reason; back at q=1 the same eager wrapper returns to AITER."""
    device = torch.device("cuda:0")
    batch, page = 3, 16
    kv, indptr, indices, last = _paged(batch, 700, page, device)
    q = torch.randn(batch, _QO, _HD, dtype=_DTYPE, device=device)

    ref = _wrapper(device, backend="fa2", use_tensor_cores=True)
    _plan(ref, indptr, indices, last, page)
    o_ref = ref.run(q, kv)

    w = _wrapper(device, use_tensor_cores=True)
    _plan(w, indptr, indices, last, page)
    assert w.backend == "aiter", w.backend_fallback_reason
    torch.testing.assert_close(w.run(q, kv), o_ref, rtol=2e-2, atol=2e-2)

    _plan(w, indptr, indices, last, page, q_len_per_req=4)
    assert w.backend == "fa2"
    assert "q_len_per_req=4" in w.backend_fallback_reason
    w.run(torch.randn(batch * 4, _QO, _HD, dtype=_DTYPE, device=device), kv)

    _plan(w, indptr, indices, last, page)
    assert w.backend == "aiter", w.backend_fallback_reason
    assert w.backend_fallback_reason is None
    torch.testing.assert_close(w.run(q, kv), o_ref, rtol=2e-2, atol=2e-2)


@requires_aiter
def test_explicit_aiter_serves_tensor_core_decode_at_one_row():
    device = torch.device("cuda:0")
    kv, indptr, indices, last = _paged(2, 300, 16, device)
    q = torch.randn(2, _QO, _HD, dtype=_DTYPE, device=device)
    w = _wrapper(device, backend="aiter", use_tensor_cores=True)
    _plan(w, indptr, indices, last, 16)
    ref = _wrapper(device, backend="aiter")
    _plan(ref, indptr, indices, last, 16)
    torch.testing.assert_close(w.run(q, kv), ref.run(q, kv))


@requires_aiter
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_auto_declines_a_page_size_pa_v1_cannot_build(monkeypatch, use_tensor_cores):
    """Above AITER's page ceiling auto names the reason without attempting the
    doomed pa_v1 build."""

    def _no_build(*a, **k):
        raise AssertionError("pa_v1 build attempted above its page ceiling")

    monkeypatch.setattr(decode_rocm, "_aiter_pa_v1_available", _no_build)
    device = torch.device("cuda:0")
    kv, indptr, indices, last = _paged(2, 300, 64, device)
    w = _wrapper(device, use_tensor_cores=use_tensor_cores)
    _plan(w, indptr, indices, last, 64)
    assert w.backend == "fa2"
    assert "page_size 64" in w.backend_fallback_reason


@requires_aiter
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_return_lse_run_moves_only_tensor_core_wrappers_to_fa2(use_tensor_cores):
    """A tensor-core wrapper returns to the fa2 kernel it ran before AITER served
    it; a plain wrapper keeps AITER, as plain fa2 decode can be far slower."""
    device = torch.device("cuda:0")
    kv, indptr, indices, last = _paged(2, 500, 16, device)
    q = torch.randn(2, _QO, _HD, dtype=_DTYPE, device=device)
    ref = _wrapper(device, backend="fa2", use_tensor_cores=True)
    _plan(ref, indptr, indices, last, 16)
    o_ref, lse_ref = ref.run(q, kv, return_lse=True)

    w = _wrapper(device, use_tensor_cores=use_tensor_cores)
    for step in range(2):
        _plan(w, indptr, indices, last, 16)
        demoted = use_tensor_cores and step == 1
        assert w.backend == ("fa2" if demoted else "aiter"), w.backend_fallback_reason
        if demoted:
            assert "return_lse" in w.backend_fallback_reason
        o, lse = w.run(q, kv, return_lse=True)
        torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(lse, lse_ref, rtol=2e-2, atol=2e-2)


@requires_aiter
def test_declines_do_not_consult_the_selector(monkeypatch):
    """Plans AITER decode can never serve must not reach the selector: it raises
    under FLASHINFER_AITER_STRICT=1 on a box where AITER is unusable."""

    def _strict(*a, **k):
        raise AssertionError("selector consulted for a plan AITER cannot serve")

    monkeypatch.setattr(decode_rocm, "_auto_select_prefill_backend", _strict)
    device = torch.device("cuda:0")
    kv, indptr, indices, last = _paged(2, 300, 16, device)
    w = _wrapper(device, use_tensor_cores=True)
    _plan(w, indptr, indices, last, 16, q_len_per_req=4)
    assert "q_len_per_req=4" in w.backend_fallback_reason
    kv, indptr, indices, last = _paged(2, 300, 64, device)
    _plan(w, indptr, indices, last, 64)
    assert "page_size 64" in w.backend_fallback_reason
    kv, indptr, indices, last = _paged(2, 300, 16, device)
    for kw, reason in (
        ({}, "max_seq_len"),
        ({"use_tensor_cores": True}, "tensor-core"),
    ):
        g = _wrapper(
            device,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.empty_like(indptr),
            paged_kv_indices_buffer=torch.empty_like(indices),
            paged_kv_last_page_len_buffer=torch.empty_like(last),
            max_seq_len=None if not kw else 4096,
            **kw,
        )
        _plan(g, indptr, indices, last, 16)
        assert reason in g.backend_fallback_reason


@requires_aiter
@pytest.mark.parametrize("head_dim", [80, 96, 112, 160])
def test_head_dims_pa_v1_gets_wrong_never_reach_aiter(head_dim):
    """PA v1 returns wrong trailing columns off multiples of 64. fa2 builds none
    of these either, so auto declines and explicit AITER raises."""
    w = _wrapper(torch.device("cuda:0"), use_tensor_cores=True)
    reason = w._aiter_decode_decline_reason(1, 16, 4, head_dim)
    assert reason is not None and f"head_dim {head_dim}" in reason
    device = torch.device("cuda:0")
    npages = 19
    indptr = torch.tensor([0, npages], dtype=torch.int32, device=device)
    indices = torch.arange(npages, dtype=torch.int32, device=device)
    last = torch.tensor([16], dtype=torch.int32, device=device)
    a = _wrapper(device, backend="aiter")
    with pytest.raises(ValueError, match=f"head_dim={head_dim}"):
        a.plan(indptr, indices, last, _QO, _KV, head_dim, 16, q_data_type=_DTYPE)


@requires_aiter
def test_explicit_aiter_rejects_a_page_above_the_ceiling(monkeypatch):
    def _no_build(*a, **k):
        raise AssertionError("pa_v1 build attempted above its page ceiling")

    monkeypatch.setattr(decode_rocm, "_aiter_pa_v1_available", _no_build)
    device = torch.device("cuda:0")
    kv, indptr, indices, last = _paged(2, 300, 64, device)
    w = _wrapper(device, backend="aiter")
    with pytest.raises(ValueError, match="page_size=64"):
        _plan(w, indptr, indices, last, 64)


@requires_aiter
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_explicit_aiter_rejects_a_jit_args_variant(use_tensor_cores):
    """PA v1 has no variant hook, so it would silently drop the variant."""
    ws = torch.empty(1024, dtype=torch.uint8, device="cuda:0")
    with pytest.raises(ValueError, match="jit_args"):
        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws,
            "NHD",
            use_tensor_cores=use_tensor_cores,
            backend="aiter",
            jit_args=("unbuilt_variant",),
        )


def test_jit_args_variant_is_never_routed_to_aiter():
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8, device="cuda:0"),
        "NHD",
        use_tensor_cores=True,
    )
    w._jit_module = object()
    reason = w._aiter_decode_decline_reason(1, 16, 4, 128)
    assert reason is not None and "jit_args" in reason


@requires_aiter
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("qo,kv_heads", [(64, 2), (40, 2), (71, 1), (128, 4)])
def test_gqa_above_16_never_reaches_aiter(use_tensor_cores, qo, kv_heads):
    """PA v1 leaves query heads unwritten above GQA 16. auto serves those groups
    on fa2's tensor-core kernel (plain decode lacks them); explicit AITER raises."""
    torch.manual_seed(0)
    device, page, hd = torch.device("cuda:0"), 16, 128
    npages = [(n + page - 1) // page for n in (700, 300)]
    kv = torch.randn(sum(npages), 2, page, kv_heads, hd, dtype=_DTYPE, device=device)
    indptr = torch.tensor([0, npages[0], sum(npages)], dtype=torch.int32, device=device)
    indices = torch.arange(sum(npages), dtype=torch.int32, device=device)
    last = torch.tensor([700 - (npages[0] - 1) * page, 300 - (npages[1] - 1) * page],
                        dtype=torch.int32, device=device)  # fmt: skip
    q = torch.randn(2, qo, hd, dtype=_DTYPE, device=device)

    def planned(**kw):
        w = _wrapper(device, **kw)
        w.plan(indptr, indices, last, qo, kv_heads, hd, page, q_data_type=_DTYPE)
        return w

    o_ref = planned(backend="fa2", use_tensor_cores=True).run(q, kv)
    w = planned(use_tensor_cores=use_tensor_cores)
    assert w.backend == "fa2"
    assert "GQA group" in w.backend_fallback_reason
    torch.testing.assert_close(w.run(q, kv), o_ref, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="GQA groups up to 16"):
        planned(backend="aiter", use_tensor_cores=use_tensor_cores)


@requires_aiter
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_fa2_tensor_core_kernel_serves_an_empty_batch(use_tensor_cores):
    """GQA 32 sends every wrapper to fa2's tensor-core kernel, whose planner
    divides by the batch size; an idle rank's empty batch must still plan and run."""
    device = torch.device("cuda:0")
    empty = torch.zeros(0, dtype=torch.int32, device=device)
    w = _wrapper(device, use_tensor_cores=use_tensor_cores)
    w.plan(torch.zeros(1, dtype=torch.int32, device=device), empty, empty,
           64, 2, 128, 16, q_data_type=_DTYPE)  # fmt: skip
    assert w.backend == "fa2"
    kv = torch.zeros(1, 2, 16, 2, 128, dtype=_DTYPE, device=device)
    o, lse = w.run(torch.empty(0, 64, 128, dtype=_DTYPE, device=device), kv,
                   return_lse=True)  # fmt: skip
    assert o.shape == (0, 64, 128) and lse.shape == (0, 64)
