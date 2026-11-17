# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Causal requests with qo_len > kv_len on the batch-prefill routes.

fa2's causal bounds are unsigned and wrap on such a request, so fa2 refuses it;
a request with no KV is fine and AITER computes the shape. These cases build
and run kernels, unlike the argument guards in ``test_prefill_guards.py``.
"""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

import flashinfer

_WORKSPACE_BYTES = 16 * 1024 * 1024


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("the wrapper allocates its scratch on the workspace's device")
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda:0")


@pytest.fixture(scope="module")
def aiter_workspace(workspace):
    from flashinfer.rocm.aiter_utils import _aiter_importable, is_aiter_supported

    if not (is_aiter_supported(workspace.device) and _aiter_importable()):
        pytest.skip("needs the aiter runtime")
    return workspace


def _indptr(values, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


_H, _D, _PAGE = 4, 128, 16
_ROUTES = ["paged", "ragged"]


def _batch(device, qo_lens=(4, 5), kv_lens=(4, 3)):
    """One request per entry, each non-empty KV in a single page; by default
    request 1 is causal-short (qo_len 5 > kv_len 3)."""
    torch.manual_seed(0)
    f16 = dict(dtype=torch.float16, device=device)
    q = torch.randn(sum(qo_lens), _H, _D, **f16)
    k = torch.randn(sum(kv_lens), _H, _D, **f16)
    v = torch.randn_like(k)
    cache = torch.zeros(len(kv_lens), 2, _PAGE, _H, _D, **f16)
    kv_indptr = [0, *itertools.accumulate(kv_lens)]
    for i, n in enumerate(kv_lens):
        cache[i, 0, :n] = k[kv_indptr[i] : kv_indptr[i + 1]]
        cache[i, 1, :n] = v[kv_indptr[i] : kv_indptr[i + 1]]
    return SimpleNamespace(
        qo_lens=qo_lens,
        kv_lens=kv_lens,
        q=q,
        k=k,
        v=v,
        cache=cache,
        qo_indptr=_indptr([0, *itertools.accumulate(qo_lens)], device),
        kv_indptr=_indptr(kv_indptr, device),
        page_indptr=_indptr([0, *itertools.accumulate(n > 0 for n in kv_lens)], device),
        page_indices=_indptr([i for i, n in enumerate(kv_lens) if n > 0], device),
        last_page_len=_indptr(list(kv_lens), device),
    )


def _wrapper(route, workspace, backend, **kw):
    cls = (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper
        if route == "paged"
        else flashinfer.BatchPrefillWithRaggedKVCacheWrapper
    )
    return cls(workspace, "NHD", backend=backend, **kw)


def _plan(w, route, b, causal, **kw):
    if route == "paged":
        w.plan(
            b.qo_indptr, b.page_indptr, b.page_indices, b.last_page_len,
            _H, _H, _D, _PAGE, causal=causal, **kw,
        )  # fmt: skip
    else:
        w.plan(b.qo_indptr, b.kv_indptr, _H, _H, _D, causal=causal, **kw)


def _inputs(route, b):
    return (b.q, b.cache) if route == "paged" else (b.q, b.k, b.v)


def _causal_reference(b):
    """Bottom-right causal attention; a row with no visible key is 0."""
    out, qs, ks = [], 0, 0
    for nq, nk in zip(b.qo_lens, b.kv_lens, strict=True):
        q, k, v = (
            t.float() for t in (b.q[qs : qs + nq], b.k[ks : ks + nk], b.v[ks : ks + nk])
        )
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(_D)
        row = torch.arange(nq, device=q.device)[:, None] + (nk - nq)
        s = s.masked_fill(
            torch.arange(nk, device=q.device)[None, :] > row, float("-inf")
        )
        out.append(
            torch.einsum("hqk,khd->qhd", torch.softmax(s, -1).nan_to_num(0.0), v)
        )
        qs, ks = qs + nq, ks + nk
    return torch.cat(out)


class TestCausalShortKV:
    """fa2's causal bounds are unsigned and wrap on a request with qo_len > kv_len,
    returning wrong output for it, so fa2 refuses one. AITER computes it, so it runs.
    """

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_refuses_it_at_plan(self, workspace, route):
        w = _wrapper(route, workspace, "fa2")
        with pytest.raises(ValueError, match="request 1 has qo_len=5 and kv_len=3"):
            _plan(w, route, _batch(workspace.device), causal=True)

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_refuses_forward_causal_after_a_non_causal_plan(self, workspace, route):
        b = _batch(workspace.device)
        w = _wrapper(route, workspace, "fa2")
        _plan(w, route, b, causal=False)
        with pytest.raises(ValueError, match="kv_len >= qo_len"):
            w.forward(*_inputs(route, b), causal=True)

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_runs_it_without_causal(self, workspace, route):
        b = _batch(workspace.device)
        w = _wrapper(route, workspace, "fa2")
        _plan(w, route, b, causal=False)
        assert torch.isfinite(w.run(*_inputs(route, b))).all()

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_runs_qo_len_equal_to_kv_len(self, workspace, route):
        b = _batch(workspace.device, qo_lens=(4, 3), kv_lens=(4, 3))
        w = _wrapper(route, workspace, "fa2")
        _plan(w, route, b, causal=True)
        torch.testing.assert_close(
            w.run(*_inputs(route, b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_does_not_refuse_a_custom_mask(self, workspace, route):
        """run() applies the mask, not causal, so the causal flag must not refuse."""
        b = _batch(workspace.device)
        mask = torch.ones(
            sum(q * k for q, k in zip(b.qo_lens, b.kv_lens, strict=True)),
            dtype=torch.bool,
            device=workspace.device,
        )
        w = _wrapper(route, workspace, "fa2")
        _plan(w, route, b, causal=True, custom_mask=mask)
        assert torch.isfinite(w.run(*_inputs(route, b))).all()

    @pytest.mark.parametrize("route", _ROUTES)
    def test_aiter_computes_it(self, aiter_workspace, route):
        b = _batch(aiter_workspace.device)
        w = _wrapper(route, aiter_workspace, "aiter")
        _plan(w, route, b, causal=True)
        torch.testing.assert_close(
            w.run(*_inputs(route, b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize("route", _ROUTES)
    def test_fa2_runs_a_request_with_no_kv(self, workspace, route):
        """fa2 writes zero rows for a request with no KV, so it is not refused."""
        b = _batch(workspace.device, qo_lens=(4, 5), kv_lens=(4, 0))
        w = _wrapper(route, workspace, "fa2")
        _plan(w, route, b, causal=True)
        torch.testing.assert_close(
            w.run(*_inputs(route, b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_a_stale_last_page_len_does_not_count_as_kv(self, workspace):
        """A zero-page request has no KV, so last_page_len 2 < qo_len 4 is not short."""
        b = _batch(workspace.device, qo_lens=(4, 4), kv_lens=(4, 0))
        b.last_page_len[1] = 2
        w = _wrapper("paged", workspace, "fa2")
        _plan(w, "paged", b, causal=True)
        torch.testing.assert_close(
            w.run(*_inputs("paged", b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_a_ragged_prefix_len_ptr_does_not_hide_it(self, workspace):
        """Ragged run() ignores prefix_len_ptr and stays causal, so plan() refuses."""
        b = _batch(workspace.device)
        w = _wrapper("ragged", workspace, "fa2")
        with pytest.raises(ValueError, match="request 1 has qo_len=5 and kv_len=3"):
            _plan(
                w, "ragged", b, causal=True, prefix_len_ptr=_indptr([0, 0], b.q.device)
            )

    def test_an_overstated_seq_lens_does_not_hide_it(self, workspace):
        b = _batch(workspace.device)
        w = _wrapper("paged", workspace, "fa2")
        with pytest.raises(ValueError, match="request 1 has qo_len=5 and kv_len=3"):
            _plan(w, "paged", b, causal=True, seq_lens=_indptr([4, 16], b.q.device))

    def test_an_understated_seq_lens_does_not_refuse_it(self, workspace):
        """fa2 reads KV lengths from the page tables, so seq_lens must not refuse."""
        b = _batch(workspace.device, qo_lens=(4, 3), kv_lens=(4, 3))
        w = _wrapper("paged", workspace, "fa2")
        _plan(w, "paged", b, causal=True, seq_lens=_indptr([4, 1], b.q.device))
        torch.testing.assert_close(
            w.run(*_inputs("paged", b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.parametrize("batch", [1, 2])
    def test_tensor_core_decode_accepts_uint32_seq_lens(self, workspace, batch):
        """seq_lens is documented as uint32; the planner reads int32/int64."""
        device = workspace.device
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=True
        )
        w.plan(
            _indptr(list(range(batch + 1)), device), _indptr(list(range(batch)), device),
            _indptr([7] * batch, device), _H, _H, _D, _PAGE,
            seq_lens=torch.full((batch,), 7, dtype=torch.uint32),
        )  # fmt: skip

    def test_paged_prefill_accepts_uint32_seq_lens(self, workspace):
        b = _batch(workspace.device, qo_lens=(4, 3), kv_lens=(4, 3))
        w = _wrapper("paged", workspace, "fa2")
        seq_lens = torch.tensor([4, 3], dtype=torch.uint32)
        _plan(w, "paged", b, causal=True, seq_lens=seq_lens)
        torch.testing.assert_close(
            w.run(*_inputs("paged", b)).float(),
            _causal_reference(b),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_a_refused_plan_leaves_run_refusing_until_replanned(self, workspace):
        good = _batch(workspace.device, qo_lens=(4, 3), kv_lens=(4, 3))
        w = _wrapper("paged", workspace, "fa2")
        _plan(w, "paged", good, causal=True)
        before = w.run(*_inputs("paged", good))
        with pytest.raises(ValueError, match="kv_len >= qo_len"):
            _plan(w, "paged", _batch(workspace.device), causal=True)
        with pytest.raises(ValueError, match=r"call plan\(\) again"):
            w.run(*_inputs("paged", good))
        _plan(w, "paged", good, causal=True)
        torch.testing.assert_close(w.run(*_inputs("paged", good)), before)

    @pytest.mark.parametrize("route", _ROUTES)
    def test_a_graph_wrapper_refuses_the_bad_replan(self, workspace, route):
        """Replay skips run(), so plan() is the only check a captured graph gets,
        and a refused plan must leave the captured inputs as they were."""
        device = workspace.device
        good = _batch(device, qo_lens=(5, 4), kv_lens=(5, 4))
        i32 = dict(dtype=torch.int32, device=device)
        names = ["qo_indptr_buf"] + (
            [
                "paged_kv_indptr_buf",
                "paged_kv_indices_buf",
                "paged_kv_last_page_len_buf",
            ]
            if route == "paged"
            else ["kv_indptr_buf"]
        )
        bufs = {
            n: torch.empty(2 if "indices" in n or "last" in n else 3, **i32)
            for n in names
        }
        w = _wrapper(route, workspace, "fa2", use_cuda_graph=True, **bufs)
        _plan(w, route, good, causal=True)
        before = {n: b.clone() for n, b in bufs.items()}
        with pytest.raises(ValueError, match="request 1 has qo_len=5 and kv_len=3"):
            _plan(w, route, _batch(device), causal=True)
        for n, b in bufs.items():
            torch.testing.assert_close(b, before[n], msg=n)

    @pytest.mark.parametrize("route", _ROUTES)
    def test_a_refused_first_graph_plan_does_not_pin_the_rows(self, workspace, route):
        device = workspace.device
        i32 = dict(dtype=torch.int32, device=device)
        names = ["qo_indptr_buf"] + (
            [
                "paged_kv_indptr_buf",
                "paged_kv_indices_buf",
                "paged_kv_last_page_len_buf",
            ]
            if route == "paged"
            else ["kv_indptr_buf"]
        )
        bufs = {
            n: torch.empty(2 if "indices" in n or "last" in n else 3, **i32)
            for n in names
        }
        w = _wrapper(route, workspace, "fa2", use_cuda_graph=True, **bufs)
        with pytest.raises(ValueError, match="request 1 has qo_len=5 and kv_len=3"):
            _plan(w, route, _batch(device), causal=True)  # 9 rows
        _plan(w, route, _batch(device, qo_lens=(5, 8), kv_lens=(5, 8)), causal=True)


class TestBlockSparseCausalShortKV:
    """Block-sparse plans through the shared fa2 planner, one request per block row.

    ``sparse.py`` is upstream's and passes ``causal`` even with a mask, so the
    refusal comes from run(), and only when run() applies the causal mask.
    """

    _R, _C, _N = 16, 4, 16  # R * group size >= 4: the prefill kernel, not decode

    def _qkv(self, device, rows=_R):
        torch.manual_seed(0)
        f16 = dict(dtype=torch.float16, device=device)
        return (torch.randn(rows, _H, _D, **f16),
                torch.randn(self._N, _H, _D, **f16),
                torch.randn(self._N, _H, _D, **f16))  # fmt: skip

    def _plan(self, workspace, columns, causal, mask=None, indptr=None):
        w = flashinfer.BlockSparseAttentionWrapper(workspace)
        device = workspace.device
        indptr = indptr or [0, len(columns)]
        w.plan(
            _indptr(indptr, device), _indptr(columns, device),
            self._R * (len(indptr) - 1), self._N, self._R, self._C, _H, _H, _D,
            mask=mask, causal=causal, q_data_type=torch.float16,
        )  # fmt: skip
        return w

    @staticmethod
    def _attend(q, k, v, visible):
        s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(_D)
        s = s.masked_fill(~visible[None], float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        return torch.einsum("hqk,khd->qhd", p, v.float())

    def test_a_short_causal_block_row_is_refused(self, workspace):
        q, k, v = self._qkv(workspace.device)
        w = self._plan(workspace, [0, 2], causal=True)
        with pytest.raises(ValueError, match="kv_len >= qo_len"):
            w.run(q, k, v)

    def test_a_custom_mask_replaces_causal(self, workspace):
        q, k, v = self._qkv(workspace.device)
        mask = torch.rand(2, self._R, self._C, device=workspace.device) > 0.3
        mask[..., 0] = True
        w = self._plan(workspace, [0, 2], causal=True, mask=mask)
        visible = torch.zeros(self._R, self._N, dtype=torch.bool, device=q.device)
        visible[:, 0:4], visible[:, 8:12] = mask[0], mask[1]
        torch.testing.assert_close(
            w.run(q, k, v).float(), self._attend(q, k, v, visible), rtol=1e-2, atol=1e-2
        )

    def test_an_empty_causal_block_row_is_zero(self, workspace):
        q, k, v = self._qkv(workspace.device, rows=2 * self._R)
        w = self._plan(workspace, [0, 1, 2, 3], causal=True, indptr=[0, 0, 4])
        o = w.run(q, k, v).float()
        assert (o[: self._R] == 0).all()
        visible = torch.ones(self._R, self._N, dtype=torch.bool, device=q.device)
        torch.testing.assert_close(
            o[self._R :],
            self._attend(q[self._R :], k, v, visible.tril(self._N - self._R)),
            rtol=1e-2,
            atol=1e-2,
        )
