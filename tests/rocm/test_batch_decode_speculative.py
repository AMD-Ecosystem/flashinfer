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
            # use_sliding_window covers both: the window test would otherwise
            # compile a fresh module inside its own body, twice.
            [DTYPE],
            [DTYPE],
            [HEAD_DIM],
            [0],
            [False, True],
            [False],
            [False],
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
# (FA2DetermineCtaTileQ), which is where the cost -- and any tiling bug -- changes.
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


# Both sides of the cta_tile_q step at q_len * gqa_group_size > 16: (4, 32/8)
# is tile 16, (8, 32/8) and (4, 64/8) are tile 64. The fa2-vs-fa2 test cannot
# see a fault shared by both plan paths, so this independent oracle has to cover
# both tilings rather than only the one.
@pytest.mark.parametrize(
    "q_len,num_qo_heads,num_kv_heads", [(4, 32, 8), (8, 32, 8), (4, 64, 8)]
)
def test_multi_token_decode_matches_aiter_reference(q_len, num_qo_heads, num_kv_heads):
    """Cross-backend arm: the fa2-vs-fa2 comparison above cannot catch a fault
    shared by both plan paths, since they reach the same module."""
    device = torch.device("cuda:0")
    from flashinfer.rocm.aiter_utils import is_aiter_supported
    from flashinfer.rocm.arch_caps import capability_available, capability_reason

    if not is_aiter_supported(device):
        pytest.skip("AITER requires gfx942/gfx950 and the aiter package")
    # is_aiter_supported only checks the arch; the capability table can still
    # gate this (op, backend, arch) and the wrapper would raise, not skip.
    if not capability_available(device, "batch_prefill", "aiter"):
        pytest.skip(capability_reason(device, "batch_prefill", "aiter"))

    batch_size, kv_len = 4, 1024
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


def test_run_warns_and_validates_explicit_q_len_per_req():
    """The explicit-argument branch of run(): upstream soft-deprecates it in
    favour of plan(), and it validates q exactly rather than by inference."""
    device = torch.device("cuda:0")
    batch_size, kv_len, q_len = 4, 256, 4
    q, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, q_len, 32, 8, device
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
        q_len_per_req=q_len,
    )
    with pytest.warns(DeprecationWarning, match="q_len_per_req"):
        wrapper.run(q, kv, q_len_per_req=q_len)
    # Explicit value that does not match q's rows must be rejected outright.
    with pytest.raises(ValueError, match="does not match batch_size"):
        wrapper.run(q, kv, q_len_per_req=q_len + 1)


def test_rejected_plan_leaves_wrapper_replayable():
    """A plan rejected *before* any wrapper state is written leaves the previous
    plan usable. Scope note: this covers the pre-write raises only. The paged-KV
    buffers are still written ahead of the C++ plan() -- pre-existing, and not
    something this test should be read as certifying."""
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

    # Distinguishing input: the rejected plan carries a *different* page
    # mapping. Passing the accepted one back would make the buffers
    # byte-identical whether or not the writes ran, so the test could not tell
    # a correct ordering from a broken one.
    other_indices = torch.flip(indices, dims=(0,)).contiguous()
    assert not torch.equal(other_indices, indices)

    with pytest.raises(ValueError, match="empty KV range"):
        wrapper.plan(
            indptr,
            other_indices,
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

    # Warm up outside the graph: a first call allocates into module-global
    # caches, and capturing that puts them in the graph's private pool.
    wrapper.run(q, kv)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=1e-3, atol=1e-3)


def test_rejected_plan_does_not_touch_the_captured_qo_indptr():
    """A plan that raises after the buffer-write point must leave the captured
    _qo_indptr_buf alone -- an already-captured graph is still reading it, and a
    stride it did not capture with produces plausible numbers, not an error.

    backend="aiter" with use_tensor_cores=True raises well after the write site,
    which is what makes it the useful probe here.
    """
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
        backend="aiter",
        paged_kv_indptr_buffer=torch.empty_like(indptr),
        paged_kv_indices_buffer=torch.empty(
            batch_size * pages_per_seq, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.empty_like(last_page_len),
    )
    before = wrapper._qo_indptr_buf.clone()
    with pytest.raises(ValueError):
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
            q_len_per_req=4,
        )
    torch.testing.assert_close(wrapper._qo_indptr_buf, before)
    assert getattr(wrapper, "_q_len_per_req", None) in (None, 1), (
        "a rejected plan committed q_len_per_req"
    )


@pytest.mark.parametrize("q_len_per_req", [1, 4])
def test_multi_token_decode_with_sliding_window(q_len_per_req):
    """The decode adapter must forward window_left and the scaled qo_indptr
    together. Both arms run the same fa2 kernel, so this cannot catch an error
    inside prefill.cuh's window derivation -- that needs an independent oracle."""
    device = torch.device("cuda:0")
    batch_size, kv_len, window_left = 4, 512, 64
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
        window_left=window_left,
    )
    out = wrapper.run(q, kv)

    workspace = torch.empty(WORKSPACE, dtype=torch.int8, device=device)
    ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    qo_indptr = (
        torch.arange(batch_size + 1, device=device, dtype=torch.int32) * q_len_per_req
    )
    ref.plan(
        qo_indptr,
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        window_left=window_left,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    torch.testing.assert_close(out, ref.run(q, kv), rtol=1e-3, atol=1e-3)


def test_cudagraph_freezes_q_len_per_req_downward():
    """The dangerous direction. Replanning a captured wrapper back to the
    default q_len_per_req=1 would rewrite _qo_indptr_buf to stride 1, and
    replay would then read 4x the rows it attends to -- plausible numbers, no
    error. Guarding only the upward change leaves exactly that open."""
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
    plan_kwargs = dict(q_data_type=DTYPE, kv_data_type=DTYPE)
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
    # Omitting q_len_per_req entirely is the realistic way to hit this.
    with pytest.raises(ValueError, match="frozen cudagraph shape"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            32,
            8,
            HEAD_DIM,
            PAGE_SIZE,
            **plan_kwargs,
        )


def test_run_rejects_q_rows_not_a_multiple_of_batch():
    """Floor division would accept this: batch 3 with 5 rows floors to 1, equals
    the planned value, and the two trailing rows of `out` come back
    uninitialised because the kernel only writes 3."""
    device = torch.device("cuda:0")
    batch_size, kv_len = 3, 256
    _, kv, indptr, indices, last_page_len = _paged_inputs(
        batch_size, kv_len, 1, 32, 8, device
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
    )
    q = torch.randn(5, 32, HEAD_DIM, device=device, dtype=DTYPE)
    with pytest.raises(ValueError, match="not a multiple of batch_size"):
        wrapper.run(q, kv)


def test_run_before_plan_says_to_call_plan():
    """run() guards on _plan_info, not on the paged buffers: under cudagraph
    those are allocated in __init__ and so are not None before the first
    plan(). Removing the _plan_info check would break that case."""
    device = torch.device("cuda:0")
    q = torch.randn(4, 32, HEAD_DIM, device=device, dtype=DTYPE)
    kv = torch.randn(4, 2, PAGE_SIZE, 8, HEAD_DIM, device=device, dtype=DTYPE)
    with pytest.raises(ValueError, match="call plan\\(\\) first"):
        _decode_wrapper(device).run(q, kv)

    # The cudagraph wrapper is the case that matters: __init__ pre-allocates the
    # paged buffers, so a guard keyed on those would pass straight through.
    _, _, indptr, indices, last_page_len = _paged_inputs(4, 256, 1, 32, 8, device)
    graph_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE, dtype=torch.int8, device=device),
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=torch.empty_like(indptr),
        paged_kv_indices_buffer=torch.empty_like(indices),
        paged_kv_last_page_len_buffer=torch.empty_like(last_page_len),
    )
    with pytest.raises(ValueError, match="call plan\\(\\) first"):
        graph_wrapper.run(q, kv)


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
