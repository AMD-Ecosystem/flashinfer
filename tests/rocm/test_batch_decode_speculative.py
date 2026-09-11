# SPDX-FileCopyrightText : 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier : Apache-2.0

"""Multi-token decode (``q_len_per_req > 1``), the speculative-decode verify step.

The oracle throughout is the paged-prefill wrapper with ``causal=True`` over a
``qo_indptr`` of stride ``q_len_per_req``. That is the same kernel the
tensor-core decode path plans through, so a correct adapter reproduces it
bitwise; the tests therefore catch a wrong mask mode, a wrong ``qo_indptr``
stride or a wrong ``total_num_rows`` rather than kernel arithmetic.
"""

import logging

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

DTYPE = torch.float16
HEAD_DIM = 128
PAGE_SIZE = 16
WORKSPACE = 128 * 1024 * 1024


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [DTYPE], [DTYPE], [HEAD_DIM], [0], [False], [False], [False]
        ),
        verbose=False,
    )
    yield


def _paged_inputs(batch_size, kv_len, q_len, num_qo_heads, num_kv_heads, device):
    pages_per_seq = kv_len // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        batch_size * q_len,
        num_qo_heads,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
        generator=gen,
    )
    kv = torch.randn(
        total_pages,
        2,
        PAGE_SIZE,
        num_kv_heads,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
        generator=gen,
    )
    indptr = (
        torch.arange(batch_size + 1, device=device, dtype=torch.int32) * pages_per_seq
    )
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full(
        (batch_size,), PAGE_SIZE, device=device, dtype=torch.int32
    )
    return q, kv, indptr, indices, last_page_len


def _prefill_reference(
    q,
    kv,
    indptr,
    indices,
    last_page_len,
    batch_size,
    q_len,
    num_qo_heads,
    num_kv_heads,
    device,
    return_lse=False,
    backend="fa2",
):
    workspace = torch.empty(WORKSPACE, dtype=torch.int8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    qo_indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * q_len
    wrapper.plan(
        qo_indptr,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    return wrapper.run(q, kv, return_lse=return_lse)


def _decode_wrapper(device, use_tensor_cores=True, **kwargs):
    workspace = torch.empty(WORKSPACE, dtype=torch.int8, device=device)
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=use_tensor_cores, **kwargs
    )


@pytest.mark.parametrize("q_len_per_req", [1, 2, 3, 4, 8])
# 32/8 and 64/8 straddle the cta_tile_q step at q_len * gqa_group_size > 16
# (rocm/utils.cuh:100), which is where the cost -- and any tiling bug -- changes.
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (64, 8)])
@pytest.mark.parametrize("kv_len", [256, 1024])
def test_multi_token_decode_matches_causal_prefill(
    q_len_per_req, num_qo_heads, num_kv_heads, kv_len
):
    device = torch.device("cuda:0")
    batch_size = 4
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len_per_req, num_qo_heads, num_kv_heads, device
    )

    wrapper = _decode_wrapper(device)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=q_len_per_req,
    )
    out = wrapper.run(q, kv)

    reference = _prefill_reference(
        q,
        kv,
        indptr,
        indices,
        last_page_len,
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
        device,
    )

    assert out.shape == (batch_size * q_len_per_req, num_qo_heads, HEAD_DIM)
    torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)


def test_multi_token_decode_matches_aiter_reference():
    """Cross-backend arm: the fa2-vs-fa2 comparison above cannot catch a fault
    shared by both plan paths, since they reach the same module."""
    device = torch.device("cuda:0")
    from flashinfer.rocm.aiter_utils import is_aiter_supported

    if not is_aiter_supported(device):
        pytest.skip("AITER requires gfx942/gfx950 and the aiter package")

    batch_size, kv_len, q_len = 4, 1024, 4
    num_qo_heads, num_kv_heads = 32, 8
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len, num_qo_heads, num_kv_heads, device
    )

    wrapper = _decode_wrapper(device)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=q_len,
    )
    out = wrapper.run(q, kv)

    reference = _prefill_reference(
        q,
        kv,
        indptr,
        indices,
        last_page_len,
        batch_size,
        q_len,
        num_qo_heads,
        num_kv_heads,
        device,
        backend="aiter",
    )
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("q_len_per_req", [1, 4])
def test_multi_token_decode_return_lse(q_len_per_req):
    """kv_len is long enough to force the split-kv path, whose merge_indptr is
    sized from total_num_rows -- so LSE depends on the total_num_rows change."""
    device = torch.device("cuda:0")
    batch_size, kv_len = 2, 8192
    num_qo_heads, num_kv_heads = 32, 8
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len_per_req, num_qo_heads, num_kv_heads, device
    )

    wrapper = _decode_wrapper(device)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=q_len_per_req,
    )
    out, lse = wrapper.run(q, kv, return_lse=True)

    ref_out, ref_lse = _prefill_reference(
        q,
        kv,
        indptr,
        indices,
        last_page_len,
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
        device,
        return_lse=True,
    )
    assert lse.shape == (batch_size * q_len_per_req, num_qo_heads)
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)


def test_rejects_multi_token_without_tensor_cores():
    device = torch.device("cuda:0")
    _, _, indptr, indices, last_page_len = _paged_inputs(4, 256, 1, 32, 8, device)
    wrapper = _decode_wrapper(device, use_tensor_cores=False)
    with pytest.raises(ValueError, match="use_tensor_cores"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            32,
            8,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            q_len_per_req=2,
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_q_len(bad):
    device = torch.device("cuda:0")
    _, _, indptr, indices, last_page_len = _paged_inputs(4, 256, 1, 32, 8, device)
    wrapper = _decode_wrapper(device)
    with pytest.raises(ValueError, match="q_len_per_req must be >= 1"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            32,
            8,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            q_len_per_req=bad,
        )


def test_rejects_kv_shorter_than_q_len():
    """The draft tokens must already be in the KV cache; otherwise the earlier
    query rows would attend to nothing and the C++ dispatch aborts."""
    device = torch.device("cuda:0")
    # One page of 16 tokens per request, asking to verify 32.
    _, _, indptr, indices, last_page_len = _paged_inputs(2, PAGE_SIZE, 1, 32, 8, device)
    wrapper = _decode_wrapper(device)
    with pytest.raises(ValueError, match="empty KV range"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            32,
            8,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            q_len_per_req=32,
        )


def test_run_rejects_q_len_disagreeing_with_plan():
    device = torch.device("cuda:0")
    batch_size, kv_len = 4, 256
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, 4, 32, 8, device
    )
    wrapper = _decode_wrapper(device)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        32,
        8,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=2,
    )
    # q carries 4 rows per request but the plan promised 2.
    with pytest.raises(ValueError, match="q_len_per_req"):
        wrapper.run(q, kv)


def test_rejected_plan_leaves_wrapper_replayable():
    """_plan_impl promises a rejected plan does not disturb a wrapper that is
    still being replayed. The per-request KV check is the newest way to trip it."""
    device = torch.device("cuda:0")
    batch_size, kv_len, q_len = 4, 256, 4
    num_qo_heads, num_kv_heads = 32, 8
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len, num_qo_heads, num_kv_heads, device
    )

    wrapper = _decode_wrapper(device)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=q_len,
    )
    expected = wrapper.run(q, kv)

    with pytest.raises(ValueError, match="empty KV range"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            q_len_per_req=kv_len + 1,
        )

    torch.testing.assert_close(wrapper.run(q, kv), expected, rtol=1e-3, atol=1e-3)


def test_cudagraph_replay_matches_eager():
    """Guards a silent failure: if the captured _qo_indptr_buf keeps its stride-1
    values, replay attends with q=1 offsets and returns plausible numbers."""
    device = torch.device("cuda:0")
    batch_size, kv_len, q_len = 4, 256, 4
    num_qo_heads, num_kv_heads = 32, 8
    pages_per_seq = kv_len // PAGE_SIZE
    total_pages = batch_size * pages_per_seq

    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len, num_qo_heads, num_kv_heads, device
    )
    eager = _prefill_reference(
        q,
        kv,
        indptr,
        indices,
        last_page_len,
        batch_size,
        q_len,
        num_qo_heads,
        num_kv_heads,
        device,
    )

    workspace = torch.empty(WORKSPACE, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=torch.empty_like(indptr),
        paged_kv_indices_buffer=torch.empty(
            total_pages, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.empty_like(last_page_len),
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        q_len_per_req=q_len,
    )

    # The buffer the captured graph will read must carry the scaled offsets.
    expected_qo = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * q_len
    torch.testing.assert_close(wrapper._qo_indptr_buf, expected_qo)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=1e-3, atol=1e-3)


def test_cudagraph_freezes_q_len_per_req():
    device = torch.device("cuda:0")
    batch_size, kv_len = 4, 256
    pages_per_seq = kv_len // PAGE_SIZE
    _, _, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, 1, 32, 8, device
    )
    workspace = torch.empty(WORKSPACE, dtype=torch.int8, device=device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=torch.empty_like(indptr),
        paged_kv_indices_buffer=torch.empty(
            batch_size * pages_per_seq, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.empty_like(last_page_len),
    )
    plan_kwargs = dict(
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        32,
        8,
        HEAD_DIM,
        PAGE_SIZE,
        q_len_per_req=4,
        **plan_kwargs,
    )
    with pytest.raises(ValueError, match="frozen cudagraph shape"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            32,
            8,
            HEAD_DIM,
            PAGE_SIZE,
            q_len_per_req=2,
            **plan_kwargs,
        )
