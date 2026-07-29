# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the AITER paged_attention_v1 decode backend vs the FA2 reference.

import logging

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)


def _build_paged_kv(
    batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
):
    """Build a NHD paged KV cache + indptr/indices/last_page_len for the given per-seq lengths.

    KV cache shape: [num_pages_total, page_size, num_kv_heads, head_dim].
    """
    total_pages = sum((L + page_size - 1) // page_size for L in kv_lens) + 4
    k = (
        torch.randn(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        * 0.1
    ).contiguous()
    v = (
        torch.randn(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        * 0.1
    ).contiguous()

    indptr = [0]
    indices = []
    last_page_len = []
    pg = 0
    for L in kv_lens:
        rem = L % page_size
        n = (L // page_size) + (1 if rem > 0 else 0)
        indices.extend(range(pg, pg + n))
        pg += n
        indptr.append(len(indices))
        last_page_len.append(rem if rem > 0 else page_size)

    return (
        k,
        v,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_page_len, dtype=torch.int32, device=device),
    )


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 8), (16, 4), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_kv_len", [64, 1024, 2048])
def test_batch_decode_aiter_vs_fa2(
    dtype,
    batch_size,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    max_kv_len,
):
    torch.manual_seed(0xA17E2)
    device = torch.device("cuda:0")

    # Mix of short/long sequences within the batch.
    kv_lens = [max(1, max_kv_len - 23 * (i % 7)) for i in range(batch_size)]

    k_cache, v_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _build_paged_kv(
            batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
        )
    )
    q = (
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        * 0.1
    ).contiguous()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_ref = ref.run(q, (k_cache, v_cache))

    cand = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    cand.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_cand = cand.run(q, (k_cache, v_cache))

    rtol, atol = (5e-3, 5e-3) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(o_cand.float(), o_ref.float(), rtol=rtol, atol=atol)


@requires_aiter
def test_batch_decode_aiter_rejects_invalid_config():
    """plan() should reject unsupported configs with a clear error."""
    device = torch.device("cuda:0")
    workspace = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indices = torch.tensor([0], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([1], dtype=torch.int32, device=device)

    # HND layout not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", backend="aiter")
    with pytest.raises(ValueError, match="NHD"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # ROPE not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="aiter")
    with pytest.raises(ValueError, match="pos_encoding_mode"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            pos_encoding_mode="ROPE_LLAMA",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # tensor cores not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="aiter"
    )
    with pytest.raises(ValueError, match="use_tensor_cores"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # NOTE: CUDA-graph capture with backend="aiter" is now supported (the grid
    # and .so variant are fixed at capture-time shapes; the kernel early-exits
    # per-seq on context_lens). Positive coverage lives in
    # test_batch_decode_aiter_cuda_graph_replay below.


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_kv_len", [256, 1024])
@pytest.mark.parametrize("window_left", [0, 31, 127, 1023])
def test_batch_decode_aiter_sliding_window_vs_fa2(
    dtype,
    batch_size,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    max_kv_len,
    window_left,
):
    """AITER PA v1 with sliding_window=window_left+1 must match FA2 with window_left.

    Covers (a) the convention mapping fix in decode_rocm.py and (b) the in-kernel
    masking logic in AITER's pa_kernels.cuh. Includes the saturation regime
    (window_left >= max_kv_len-1) to exercise the no-op branch.
    """
    torch.manual_seed(0xA17E3)
    device = torch.device("cuda:0")

    kv_lens = [max(1, max_kv_len - 17 * (i % 5)) for i in range(batch_size)]
    k_cache, v_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _build_paged_kv(
            batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
        )
    )
    q = (
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        * 0.1
    ).contiguous()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_ref = ref.run(q, (k_cache, v_cache))

    cand = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    cand.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_cand = cand.run(q, (k_cache, v_cache))

    rtol, atol = (5e-3, 5e-3) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(o_cand.float(), o_ref.float(), rtol=rtol, atol=atol)


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window_left", [-1, 31])
def test_batch_decode_aiter_return_lse_via_fa2(dtype, window_left):
    """run(return_lse=True) on an AITER-planned wrapper must transparently dispatch
    through the pre-built FA2 shadow plan and produce (output, lse) matching the
    pure-FA2 reference. Covers window_left=-1 and a sliding-window setting.
    """
    torch.manual_seed(0xA17E4)
    device = torch.device("cuda:0")
    batch_size = 4
    page_size = 16
    num_qo_heads = 16
    num_kv_heads = 4
    head_dim = 128
    kv_lens = [128, 256, 511, 1024]

    k_cache, v_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _build_paged_kv(
            batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
        )
    )
    q = (
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        * 0.1
    ).contiguous()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_ref, lse_ref = ref.run(q, (k_cache, v_cache), return_lse=True)

    cand = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    cand.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # Output-only call still goes to AITER and must match FA2.
    o_aiter = cand.run(q, (k_cache, v_cache))
    rtol, atol = (5e-3, 5e-3) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(o_aiter.float(), o_ref.float(), rtol=rtol, atol=atol)

    # LSE call falls back to the FA2 shadow plan; both output and LSE must match.
    o_cand, lse_cand = cand.run(q, (k_cache, v_cache), return_lse=True)
    torch.testing.assert_close(o_cand.float(), o_ref.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(lse_cand, lse_ref, rtol=1e-3, atol=1e-3)


@requires_aiter
def test_batch_decode_auto_routes_cuda_graph_to_fa2():
    """backend='auto' with use_cuda_graph=True routes to fa2. fa2's graph path is
    capacity-based (correct regardless of capture-vs-replay sizes); AITER decode
    under graph is opt-in via backend='aiter' (capture-at-max contract), so auto
    does not select it silently."""
    device = torch.device("cuda:0")
    workspace = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr_buf = torch.empty(2, dtype=torch.int32, device=device)
    indices_buf = torch.empty(8, dtype=torch.int32, device=device)
    last_page_len_buf = torch.empty(1, dtype=torch.int32, device=device)

    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_len_buf,
        backend="auto",
    )
    indptr_buf.copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
    indices_buf[:1].copy_(torch.tensor([0], dtype=torch.int32, device=device))
    last_page_len_buf.copy_(torch.tensor([1], dtype=torch.int32, device=device))
    w.plan(
        indptr_buf,
        indices_buf[:1],
        last_page_len_buf,
        8,
        8,
        128,
        16,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    assert w._backend == "fa2"


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_decode_aiter_cuda_graph_replay(dtype):
    """Opt-in AITER decode under CUDA graph (explicit backend='aiter'): capture
    once at a maximum sequence length, then replay for a shorter sequence; the
    result must match an eager AITER run (the kernel early-exits per-seq on
    context_lens). Captures at cap_seq and replays shorter — the supported
    capture-at-max usage."""
    device = torch.device("cuda:0")
    batch, page, num_qo, num_kv, hd = 4, 16, 8, 8, 128
    cap_seq, replay_seq = 2048, 512
    cap_pages = (cap_seq + page - 1) // page
    total_pages = batch * cap_pages

    kv = torch.randn(total_pages, 2, page, num_kv, hd, dtype=dtype, device=device)
    q = torch.randn(batch, num_qo, hd, dtype=dtype, device=device)

    def layout(seq_len):
        npages = (seq_len + page - 1) // page
        last = seq_len - (npages - 1) * page
        indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * npages
        # Each sequence keeps a stable reserved block of cap_pages in the fixed
        # pool; a shorter seq_len just uses the first `npages` of its block. This
        # models real paged-KV (stable per-seq page pool) and exercises the
        # capture-at-max contract faithfully.
        base = (torch.arange(batch, device=device) * cap_pages).view(-1, 1)
        offs = torch.arange(npages, device=device).view(1, -1)
        indices = (base + offs).reshape(-1).to(torch.int32)
        last_page = torch.full((batch,), last, dtype=torch.int32, device=device)
        return indptr, indices, last_page

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr_buf = torch.empty(batch + 1, dtype=torch.int32, device=device)
    indices_buf = torch.empty(total_pages, dtype=torch.int32, device=device)
    last_page_buf = torch.empty(batch, dtype=torch.int32, device=device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        use_cuda_graph=True,
        backend="aiter",
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_buf,
    )
    # plan + capture at capacity
    ip, ix, lp = layout(cap_seq)
    w.plan(
        ip,
        ix,
        lp,
        num_qo,
        num_kv,
        hd,
        page,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert w._backend == "aiter"
    for _ in range(3):
        w.run(q, kv)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = w.run(q, kv)

    # replay at a shorter sequence
    q.copy_(torch.randn(batch, num_qo, hd, dtype=dtype, device=device))
    ip, ix, lp = layout(replay_seq)
    w.plan(
        ip,
        ix,
        lp,
        num_qo,
        num_kv,
        hd,
        page,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    g.replay()
    torch.cuda.synchronize()

    # eager AITER reference on identical inputs
    ws2 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref_w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws2, "NHD", backend="aiter")
    ip, ix, lp = layout(replay_seq)
    ref_w.plan(
        ip,
        ix,
        lp,
        num_qo,
        num_kv,
        hd,
        page,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref = ref_w.run(q, kv)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
