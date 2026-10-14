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
