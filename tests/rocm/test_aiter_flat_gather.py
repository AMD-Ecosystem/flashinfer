# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AITER's flat-gather paged prefill: the gather it plans, and what run() reads."""

import pytest
import torch

import flashinfer
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.prefill import _aiter_ops_importable, _flat_gather_indices


def _reference(indptr, indices, last_len, page_size):
    rows, counts = [], []
    for i in range(len(indptr) - 1):
        pages = indices[indptr[i] : indptr[i + 1]]
        n = 0
        for j, p in enumerate(pages):
            take = last_len[i] if j == len(pages) - 1 else page_size
            rows += [p * page_size + t for t in range(take)]
            n += take
        counts.append(n)
    flat = [0]
    for c in counts:
        flat.append(flat[-1] + c)
    return rows, flat


@pytest.mark.parametrize(
    "page_size,indptr,indices,last_len",
    [
        (16, [0, 3, 5], [7, 2, 9, 0, 4], [5, 16]),
        (1, [0, 2, 3], [4, 1, 0], [1, 1]),
        (1024, [0, 1, 3], [1, 0, 2], [1024, 1]),
        # An empty request between two others, and indices longer than indptr[-1]
        # (a graph-mode buffer).
        (16, [0, 2, 2, 3], [3, 1, 0, 99, 99], [7, 16, 1]),
        # indptr starting past 0: vLLM's prefill tail of a mixed decode batch.
        (4, [2, 3, 5], [8, 9, 7, 2, 9], [3, 4]),
        (16, [3, 3, 5, 6], [4, 4, 4, 1, 0, 2, 99], [16, 9, 2]),
        # last_page_len padded past the batch, as a reused buffer is.
        (16, [0, 2, 3], [5, 6, 7], [4, 16, 0, 0]),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs a GPU"
            ),
        ),
    ],
)
def test_gather_indices_match_the_per_token_definition(
    page_size, indptr, indices, last_len, device
):
    rows, flat = _flat_gather_indices(
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_len, dtype=torch.int32, device=device),
        page_size,
        torch.device(device),
    )
    want_rows, want_flat = _reference(
        indptr, indices, last_len[: len(indptr) - 1], page_size
    )
    assert rows.tolist() == want_rows
    assert flat.dtype == torch.int32 and flat.tolist() == want_flat


def _require_aiter_prefill(device):
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    from flashinfer.rocm.arch_caps import capability_available, capability_reason

    if not capability_available(device, "batch_prefill", "aiter"):
        pytest.skip(capability_reason(device, "batch_prefill", "aiter"))


def _plan_run(kv_cache, ref_cache, pool_pages, device, check_memory):
    page_size, heads, head_dim, dtype = 16, 8, 128, torch.float16
    batch, kv_len, qo_len = 2, 256, 64
    npages = kv_len // page_size
    plan_args = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * qo_len,
        torch.arange(batch + 1, dtype=torch.int32, device=device) * npages,
        torch.randperm(pool_pages, device=device)[: batch * npages].int(),
        torch.full((batch,), page_size, dtype=torch.int32, device=device),
        heads,
        heads,
        head_dim,
        page_size,
    )
    q = torch.randn(batch * qo_len, heads, head_dim, device=device, dtype=dtype)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    outs = {}
    for backend, cache in (("aiter", kv_cache), ("fa2", ref_cache)):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
        w.plan(*plan_args, causal=True, q_data_type=dtype)
        if backend == "aiter":
            assert w._aiter_flat_gather_idx is not None, "test premise: flat gather"
            if check_memory:
                w.run(q, cache)  # warm the module
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                base = torch.cuda.memory_allocated(device)
        outs[backend] = w.run(q, cache)
        if backend == "aiter" and check_memory:
            torch.cuda.synchronize()
            extra = torch.cuda.max_memory_allocated(device) - base
            pool_bytes = kv_cache.numel() * kv_cache.element_size()
            assert extra < pool_bytes // 8, (
                f"run() allocated {extra >> 20} MiB against a {pool_bytes >> 20} MiB pool"
            )
    torch.testing.assert_close(outs["aiter"], outs["fa2"], rtol=2e-2, atol=2e-2)


def test_run_reads_only_the_batch_from_a_combined_pool():
    """The k/v halves of a [pages, 2, ...] pool are non-contiguous; run() must not copy it whole."""
    device = torch.device("cuda:0")
    _require_aiter_prefill(device)
    pool_pages = 32768  # 2 GiB pool, 1 MiB read
    pool = torch.randn(pool_pages, 2, 16, 8, 128, device=device, dtype=torch.float16)
    try:
        _plan_run(pool, pool, pool_pages, device, check_memory=True)
    finally:
        del pool
        torch.cuda.empty_cache()


def test_run_gathers_from_a_layout_without_contiguous_rows():
    """A head-major cache viewed as NHD has no contiguous token rows: the copy fallback."""
    device = torch.device("cuda:0")
    _require_aiter_prefill(device)
    pool_pages = 256
    k, v = (
        torch.randn(
            pool_pages, 8, 16, 128, device=device, dtype=torch.float16
        ).transpose(1, 2)
        for _ in range(2)
    )
    _plan_run((k, v), (k.contiguous(), v.contiguous()), pool_pages, device, False)
