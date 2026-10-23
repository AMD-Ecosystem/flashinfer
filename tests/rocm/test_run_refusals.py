# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""run()-time arguments ROCm cannot honour: refused when wrong, quiet when default.

``sinks`` and descale tensors have no ROCm kernel on these routes, so accepting
them returns wrong numbers. ``enable_pdl`` has no ROCm effect, so its default
must not warn on every call.
"""

import warnings

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter

PAGE_SIZE = 16
HEADS = 8
HEAD_DIM = 128


def _paged_inputs(device):
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)
    kv = torch.randn(
        1, 2, PAGE_SIZE, HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    return kv_indptr, kv_indices, last_page_len, kv


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def workspace(device):
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)


@pytest.fixture
def paged_prefill(device, workspace):
    kv_indptr, kv_indices, last_page_len, kv = _paged_inputs(device)
    qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        HEADS,
        HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
    )
    q = torch.randn(4, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    return wrapper, q, kv


@pytest.fixture
def ragged_prefill(device, workspace):
    indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper.plan(indptr, indptr, HEADS, HEADS, HEAD_DIM, causal=True)
    q, k, v = (
        torch.randn(4, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
        for _ in range(3)
    )
    return wrapper, q, k, v


@pytest.fixture
def decode(device, workspace):
    kv_indptr, kv_indices, last_page_len, kv = _paged_inputs(device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper.plan(
        kv_indptr, kv_indices, last_page_len, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    q = torch.randn(1, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    return wrapper, q, kv


def _sinks(device):
    return torch.zeros(HEADS, dtype=torch.float32, device=device)


class TestSinksAreRefused:
    def test_paged_prefill(self, paged_prefill, device):
        wrapper, q, kv = paged_prefill
        with pytest.raises(NotImplementedError, match="sinks"):
            wrapper.run(q, kv, sinks=_sinks(device))

    def test_paged_prefill_return_lse(self, paged_prefill, device):
        wrapper, q, kv = paged_prefill
        with pytest.raises(NotImplementedError, match="sinks"):
            wrapper.run_return_lse(q, kv, sinks=_sinks(device))

    def test_decode(self, decode, device):
        wrapper, q, kv = decode
        with pytest.raises(NotImplementedError, match="sinks"):
            wrapper.run(q, kv, sinks=_sinks(device))

    def test_ragged_prefill_wrapper_attribute(self, ragged_prefill, device):
        """vLLM stores sinks on the wrapper rather than passing them to run()."""
        wrapper, q, k, v = ragged_prefill
        wrapper._sinks = _sinks(device)
        with pytest.raises(NotImplementedError, match="_sinks"):
            wrapper.run(q, k, v)

    def test_none_still_runs(self, paged_prefill, decode, ragged_prefill):
        wrapper, q, kv = paged_prefill
        wrapper.run(q, kv, sinks=None)
        wrapper, q, kv = decode
        wrapper.run(q, kv, sinks=None)
        wrapper, q, k, v = ragged_prefill
        wrapper._sinks = None
        wrapper.run(q, k, v)


def _pdl_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [w for w in caught if "enable_pdl" in str(w.message)]


class TestPdlDefaultIsQuiet:
    """The default is silent on every wrapper; an explicit True warns, as a UserWarning."""

    def test_paged_prefill(self, paged_prefill):
        wrapper, q, kv = paged_prefill
        assert _pdl_warnings(lambda: wrapper.run(q, kv)) == []
        assert len(_pdl_warnings(lambda: wrapper.run(q, kv, enable_pdl=True))) == 1

    def test_ragged_prefill(self, ragged_prefill):
        wrapper, q, k, v = ragged_prefill
        assert _pdl_warnings(lambda: wrapper.run(q, k, v)) == []
        assert len(_pdl_warnings(lambda: wrapper.run(q, k, v, enable_pdl=True))) == 1

    def test_decode(self, decode):
        wrapper, q, kv = decode
        assert _pdl_warnings(lambda: wrapper.run(q, kv)) == []
        assert len(_pdl_warnings(lambda: wrapper.run(q, kv, enable_pdl=True))) == 1


@pytest.mark.parametrize("name", ["scale_q", "scale_k", "scale_v"])
def test_single_prefill_refuses_descale_tensors(device, name):
    """No single-prefill kernel applies a descale tensor, so passing one is refused."""
    q, k, v = (
        torch.randn(8, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
        for _ in range(3)
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="scale_q/scale_k/scale_v"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, **{name: scale})


def test_block_sparse_refuses_descale_tensors(device):
    """The block-sparse wrapper reaches the shared fa2 paged_run, which drops them."""
    R = C = 16
    M = N = 64
    indptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BlockSparseAttentionWrapper(ws)
    wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        HEADS,
        HEADS,
        HEAD_DIM,
        q_data_type=torch.float16,
    )
    q = torch.randn(M, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(N, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(N, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    wrapper.run(q, k, v)
    with pytest.raises(NotImplementedError, match="scale_q/scale_k/scale_v"):
        wrapper.run(
            q, k, v, scale_k=torch.ones(HEADS, dtype=torch.float32, device=device)
        )


def test_paged_fa2_refuses_descale_tensors(paged_prefill, device):
    wrapper, q, kv = paged_prefill
    scale = torch.ones(1, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="resolved to fa2"):
        wrapper.run(q, kv, scale_q=scale)


def test_paged_custom_variant_refuses_descale_tensors(paged_prefill, device):
    """A custom variant takes its tensors through run(*args), never the descale slots."""
    wrapper, q, kv = paged_prefill
    # Stands in for jit_args: only the presence of a custom module is under test.
    wrapper._jit_module = object()
    scale = torch.ones(1, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="custom variant"):
        wrapper.run(q, kv, scale_q=scale)


# Without the check these run silently or kill the process (SIGFPE, an HSA
# exception), so A/B them one test per process.
_SHAPE_MISMATCHES = {
    "q heads": dict(q_heads=HEADS // 2),
    "k heads": dict(kv_heads=HEADS // 2),
    "q head_dim": dict(head_dim=HEAD_DIM // 2),
    "k page_size": dict(page_size=PAGE_SIZE // 2),
}
_PLAN_ROUTES = [
    "decode-fa2",
    "decode-fa2-tc",
    pytest.param("decode-aiter", marks=requires_aiter),
    "paged-fa2",
    pytest.param("paged-aiter", marks=requires_aiter),
    "ragged-fa2",
    pytest.param("ragged-aiter", marks=requires_aiter),
]


def _planned_wrapper_and_inputs(
    route, workspace, device, q_heads, kv_heads, head_dim, page_size
):
    kind, backend, *tc = route.split("-")
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    rand = lambda *s: torch.randn(*s, dtype=torch.float16, device=device)  # noqa: E731
    paged = (i32(0, 1), i32(0), i32(PAGE_SIZE))
    if kind == "decode":
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", backend=backend, use_tensor_cores=bool(tc)
        )
        w.plan(*paged, HEADS, HEADS, HEAD_DIM, PAGE_SIZE, q_data_type=torch.float16)
        return w, (
            rand(1, q_heads, head_dim),
            rand(1, 2, page_size, kv_heads, head_dim),
        )
    if kind == "paged":
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend=backend
        )
        w.plan(i32(0, 4), *paged, HEADS, HEADS, HEAD_DIM, PAGE_SIZE, causal=True)
        return w, (
            rand(4, q_heads, head_dim),
            rand(1, 2, page_size, kv_heads, head_dim),
        )
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    w.plan(i32(0, 4), i32(0, 4), HEADS, HEADS, HEAD_DIM, causal=True)
    return w, (
        rand(4, q_heads, head_dim),
        rand(4, kv_heads, head_dim),
        rand(4, kv_heads, head_dim),
    )


@pytest.mark.parametrize("route", _PLAN_ROUTES)
@pytest.mark.parametrize("mismatch", list(_SHAPE_MISMATCHES))
def test_run_refuses_shapes_plan_did_not_build_for(route, mismatch, workspace, device):
    """The kernels are built for the planned shapes and would read others out of bounds."""
    if mismatch == "k page_size" and route.startswith("ragged"):
        pytest.skip("ragged KV has no pages")
    dims = dict(q_heads=HEADS, kv_heads=HEADS, head_dim=HEAD_DIM, page_size=PAGE_SIZE)
    dims.update(_SHAPE_MISMATCHES[mismatch])
    wrapper, inputs = _planned_wrapper_and_inputs(route, workspace, device, **dims)
    with pytest.raises(ValueError, match=f"run\\(\\) got {mismatch}="):
        wrapper.run(*inputs)


@pytest.mark.parametrize("route", _PLAN_ROUTES)
def test_run_with_planned_shapes_still_runs(route, workspace, device):
    wrapper, inputs = _planned_wrapper_and_inputs(
        route, workspace, device, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    wrapper.run(*inputs)


@requires_aiter
def test_a_late_aiter_plan_failure_leaves_run_refusing(workspace, device):
    """The AITER gather route's indptr bound is checked after the wrapper is rewritten."""
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    w.plan(
        i32(0, 4), i32(0, 1), i32(0), i32(PAGE_SIZE), HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    with pytest.raises(ValueError, match="paged_kv_indptr ends at"):
        w.plan(
            i32(0, 4),
            i32(0, 2),
            i32(0),
            i32(PAGE_SIZE),
            HEADS // 2,
            HEADS // 2,
            HEAD_DIM,
            PAGE_SIZE,
        )
    q = torch.randn(4, HEADS // 2, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(
        2, 2, PAGE_SIZE, HEADS // 2, HEAD_DIM, dtype=torch.float16, device=device
    )
    with pytest.raises(ValueError, match=r"call plan\(\) again"):
        w.run(q, kv)


@pytest.mark.parametrize("route", ["paged-fa2", "ragged-fa2"])
def test_run_refuses_fewer_tokens_than_planned(route, workspace, device):
    """out is sized from q, so a short q would be written past its end."""
    wrapper, (q, *rest) = _planned_wrapper_and_inputs(
        route, workspace, device, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    with pytest.raises(
        ValueError, match=r"run\(\) got q tokens=2 but plan\(\) covers 4"
    ):
        wrapper.run(q[:2], *rest)
    if route.startswith("ragged"):
        k, v = rest
        with pytest.raises(
            ValueError, match=r"run\(\) got k tokens=2 but plan\(\) covers 4"
        ):
            wrapper.run(q, k[:2], v[:2])
    wrapper.run(torch.cat([q, q]), *rest)  # a padded q is allowed


def test_run_after_a_plan_that_raised_part_way_is_refused(workspace, device):
    """A plan() that raises after rewriting the wrapper leaves it half-planned."""
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w.plan(
        i32(0, 4), i32(0, 1), i32(0), i32(PAGE_SIZE), HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    q = torch.randn(4, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(
        1, 2, PAGE_SIZE, HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    w.run(q, kv)
    # An empty seq_lens passes the up-front checks and raises after the rewrite.
    with pytest.raises(ValueError):
        w.plan(
            i32(0, 4),
            i32(0, 1),
            i32(0),
            i32(PAGE_SIZE),
            HEADS,
            HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            seq_lens=i32(),
        )
    with pytest.raises(ValueError, match=r"call plan\(\) again"):
        w.run(q, kv)


@pytest.mark.parametrize("route", _PLAN_ROUTES)
def test_run_refuses_tensors_of_the_wrong_rank(route, workspace, device):
    """Every shim reads heads at q.size(1), so an extra leading dim must not pass."""
    wrapper, (q, *rest) = _planned_wrapper_and_inputs(
        route, workspace, device, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    with pytest.raises(ValueError, match=r"run\(\) got a 4-D q; expected 3-D"):
        wrapper.run(q.unsqueeze(1), *rest)
    with pytest.raises(ValueError, match=r"run\(\) got a 2-D q; expected 3-D"):
        wrapper.run(q[0], *rest)  # decode would read it as a multi-token q
    if route.startswith("ragged"):
        k, v = rest
        with pytest.raises(ValueError, match=r"run\(\) got a 2-D k; expected 3-D"):
            wrapper.run(q, k.flatten(1), v)


def test_a_rejected_ragged_plan_leaves_the_last_plan_usable(workspace, device):
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w, (q, k, v) = _planned_wrapper_and_inputs(
        "ragged-fa2", workspace, device, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    expected = w.run(q, k, v)
    with pytest.raises(ValueError, match="kv_indptr length"):
        w.plan(i32(0, 4), i32(0, 2, 4), HEADS, HEADS, HEAD_DIM, causal=True)
    torch.testing.assert_close(w.run(q, k, v), expected)


def test_a_rejected_cuda_graph_decode_plan_leaves_the_last_plan_usable(
    workspace, device
):
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=i32(0, 0),
        paged_kv_indices_buffer=i32(0),
        paged_kv_last_page_len_buffer=i32(0),
        backend="fa2",
    )
    w.plan(i32(0, 1), i32(0), i32(PAGE_SIZE), HEADS, HEADS, HEAD_DIM, PAGE_SIZE)
    q = torch.randn(1, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(
        1, 2, PAGE_SIZE, HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    expected = w.run(q, kv)
    with pytest.raises(ValueError, match="batch size should be fixed"):
        w.plan(
            i32(0, 1, 2),
            i32(0, 0),
            i32(PAGE_SIZE, PAGE_SIZE),
            HEADS,
            HEADS,
            HEAD_DIM,
            PAGE_SIZE,
        )
    torch.testing.assert_close(w.run(q, kv), expected)


def test_a_rejected_cuda_graph_paged_plan_leaves_the_last_plan_usable(
    workspace, device
):
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=i32(0, 0),
        paged_kv_indptr_buf=i32(0, 0),
        paged_kv_indices_buf=i32(0),
        paged_kv_last_page_len_buf=i32(0),
        backend="fa2",
    )
    w.plan(
        i32(0, 4), i32(0, 1), i32(0), i32(PAGE_SIZE), HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    q = torch.randn(4, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(
        1, 2, PAGE_SIZE, HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    expected = w.run(q, kv)
    with pytest.raises(ValueError, match="batch size should be fixed"):
        w.plan(
            i32(0, 2, 4),
            i32(0, 1, 1),
            i32(0),
            i32(PAGE_SIZE, PAGE_SIZE),
            HEADS,
            HEADS,
            HEAD_DIM,
            PAGE_SIZE,
        )
    with pytest.raises(ValueError, match="paged_kv_indices exceeds"):
        w.plan(
            i32(0, 4),
            i32(0, 2),
            i32(0, 0),
            i32(PAGE_SIZE),
            HEADS,
            HEADS,
            HEAD_DIM,
            PAGE_SIZE,
        )
    torch.testing.assert_close(w.run(q, kv), expected)


def test_cuda_graph_row_capacity_is_not_the_callers_buffer(workspace):
    """A reused CPU qo_indptr must not move the capacity the graph was sized for."""
    qo_indptr = torch.tensor([0, 4], dtype=torch.int32)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.zeros(2, dtype=torch.int32, device=workspace.device),
        kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=workspace.device),
        backend="fa2",
    )
    w.plan(qo_indptr, qo_indptr.clone(), HEADS, HEADS, HEAD_DIM, causal=True)
    qo_indptr[1] = 8
    with pytest.raises(ValueError, match="cannot exceed"):
        w.plan(qo_indptr, qo_indptr.clone(), HEADS, HEADS, HEAD_DIM, causal=True)


@requires_aiter
def test_a_rejected_aiter_decode_plan_leaves_the_last_plan_usable(workspace, device):
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    w, (q, kv) = _planned_wrapper_and_inputs(
        "decode-aiter", workspace, device, HEADS, HEADS, HEAD_DIM, PAGE_SIZE
    )
    expected = w.run(q, kv)
    with pytest.raises(ValueError, match="pos_encoding_mode='NONE'"):
        w.plan(
            i32(0, 1),
            i32(0),
            i32(PAGE_SIZE),
            HEADS,
            HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            pos_encoding_mode="ROPE_LLAMA",
            q_data_type=torch.float16,
        )
    torch.testing.assert_close(w.run(q, kv), expected)
