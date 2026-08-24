# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the AITER reshape_and_cache_flash KV-append backend vs the
# native flashinfer-ROCm append_paged_kv_cache kernel.

import logging
import re
from contextlib import nullcontext

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)


def _build_append_inputs(append_lens, page_size, num_kv_heads, head_dim, dtype, device):
    nnz = sum(append_lens)
    indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(append_lens), 0)),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(append_lens, dtype=torch.int32, device=device)
    num_pages_per = [(L + page_size - 1) // page_size for L in append_lens]
    num_pages = sum(num_pages_per) + 4  # slack
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(num_pages_per), 0)),
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.tensor(
        list(range(sum(num_pages_per))), dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [
            L - (n - 1) * page_size
            for L, n in zip(append_lens, num_pages_per, strict=True)
        ],
        dtype=torch.int32,
        device=device,
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        indptr, seq_lens, nnz
    )
    k = (
        torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
    ).contiguous()
    v = (
        torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
    ).contiguous()
    return (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    )


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("num_kv_heads,head_dim", [(4, 64), (8, 128), (16, 128)])
@pytest.mark.parametrize(
    "append_lens",
    [
        [37, 100, 5, 64, 23],
        [1, 1, 1, 1],
        [128, 256, 512],
        [1024],
    ],
)
def test_append_paged_kv_cache_aiter_vs_native(
    dtype, page_size, num_kv_heads, head_dim, append_lens
):
    torch.manual_seed(0xA17E2)
    device = torch.device("cuda:0")

    (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    ) = _build_append_inputs(
        append_lens, page_size, num_kv_heads, head_dim, dtype, device
    )

    k_native = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_native = torch.zeros_like(k_native)
    k_aiter = torch.zeros_like(k_native)
    v_aiter = torch.zeros_like(v_native)

    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        (k_native, v_native),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        backend="native",
    )
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        (k_aiter, v_aiter),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        backend="aiter",
    )

    # Bit-exact: both backends just do a memcpy/scatter; no FP arithmetic.
    torch.testing.assert_close(k_aiter, k_native, rtol=0, atol=0)
    torch.testing.assert_close(v_aiter, v_native, rtol=0, atol=0)


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_append_paged_kv_cache_auto_always_routes_native(dtype, kv_layout):
    """auto picks native for every input, including the ones AITER supports.

    AITER's reshape_and_cache_flash is correct but slower than the in-tree kernel
    at every size measured on gfx942 (2.86 vs 3.62 TB/s), so the fp16/bf16 + NHD
    combination it does support is routed native too. backend='aiter' remains the
    way to reach it.
    """
    from flashinfer.page import _auto_select_kv_append_backend

    device = torch.device("cuda:0")
    assert (
        _auto_select_kv_append_backend(device, dtype=dtype, kv_layout=kv_layout)
        == "native"
    )


@requires_aiter
def test_append_paged_kv_cache_aiter_honors_current_stream():
    """The AITER append must run on torch's current stream, not the default one.

    The shim sets only the device; reshape_and_cache_flash picks up the stream
    itself. In amd-aiter 0.1.10 it calls at::hip::getCurrentHIPStream(), so this
    holds. Newer AITER moved these ops to a thread-local stream set from Python,
    which would silently put the append on the null stream instead -- and because
    torch's default stream *is* stream 0, every other test here would still pass.
    This is the one that would not.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_kv_heads, head_dim = 16, 8, 128

    (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    ) = _build_append_inputs(
        [37, 5, 64], page_size, num_kv_heads, head_dim, dtype, device
    )

    def _run_append(stream):
        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_cache = torch.zeros_like(k_cache)
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            flashinfer.append_paged_kv_cache(
                k,
                v,
                batch_indices,
                positions,
                (k_cache, v_cache),
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                backend="aiter",
            )
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return k_cache, v_cache

    k_default, v_default = _run_append(None)
    k_side, v_side = _run_append(torch.cuda.Stream())

    torch.testing.assert_close(k_side, k_default, rtol=0, atol=0)
    torch.testing.assert_close(v_side, v_default, rtol=0, atol=0)


@requires_aiter
def test_append_auto_never_builds_the_aiter_shim(monkeypatch):
    """'auto' routes to native without so much as loading the AITER module.

    Routing is a static decision (AITER's append is slower), so 'auto' must not
    pay AITER's build cost to discover that. Breaking get_page_aiter_module is
    what makes this fail if the routing ever consults it again -- asserting only
    on the returned string would not.
    """
    from flashinfer import page as page_mod

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_kv_heads, head_dim = 16, 8, 128

    (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    ) = _build_append_inputs([12, 3], page_size, num_kv_heads, head_dim, dtype, device)

    def _append(cache_pair, backend):
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            cache_pair,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            backend=backend,
        )

    def _zeros():
        c = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        return c, torch.zeros_like(c)

    # Reference, taken before the shim is broken.
    k_ref, v_ref = _zeros()
    _append((k_ref, v_ref), "native")
    assert (k_ref != 0).any(), "reference append wrote nothing"

    def _boom():
        raise AssertionError("auto must not load the AITER shim")

    monkeypatch.setattr(page_mod, "get_page_aiter_module", _boom)

    assert (
        page_mod._auto_select_kv_append_backend(device, dtype=dtype, kv_layout="NHD")
        == "native"
    )
    k_cache, v_cache = _zeros()
    _append((k_cache, v_cache), "auto")

    # Assert the native path actually appended; isfinite() on a zero cache would
    # pass even if it silently no-oped.
    torch.testing.assert_close(k_cache, k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_ref, rtol=0, atol=0)


@requires_aiter
@pytest.mark.parametrize(
    "kv_layout,dtype,expected",
    [("HND", torch.bfloat16, "HND"), ("NHD", torch.float32, "float32")],
)
def test_append_explicit_aiter_rejects_unsupported_layout_and_dtype(
    kv_layout, dtype, expected
):
    """Explicit backend='aiter' must enforce the gates 'auto' applies.

    The shim reads page_size from paged_k_cache.size(1), which is num_kv_heads
    under HND — so an unchecked HND cache would scatter every token to the wrong
    slot with no error at all.
    """
    device = torch.device("cuda:0")
    page_size, num_kv_heads, head_dim = 16, 8, 128
    (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    ) = _build_append_inputs([5], page_size, num_kv_heads, head_dim, dtype, device)

    if kv_layout == "HND":
        shape = (num_pages, num_kv_heads, page_size, head_dim)
    else:
        shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.zeros(*shape, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError, match=re.escape(expected)):
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            (k_cache, v_cache),
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout=kv_layout,
            backend="aiter",
        )


@requires_aiter
def test_append_aiter_shim_rejects_mismatched_device_shape_dtype_and_length():
    """The shim is a directly-callable torch op, so it validates its own arguments.

    A host-resident kv_indptr (the page table is often built on the CPU) would
    otherwise be dereferenced from device code, and a dtype mismatch reinterprets
    memory rather than erroring.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_kv_heads, head_dim = 16, 8, 128
    (
        k,
        v,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_pages,
    ) = _build_append_inputs([9], page_size, num_kv_heads, head_dim, dtype, device)
    k_cache = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    mod = flashinfer.page.get_page_aiter_module()
    unit = flashinfer.page._aiter_unit_scale(device)

    with pytest.raises(RuntimeError, match="same device"):
        mod.append_paged_kv_cache_aiter(
            k,
            v,
            batch_indices,
            positions,
            k_cache,
            v_cache,
            kv_indices,
            kv_indptr.cpu(),
            unit,
            unit,
        )

    # Both caches are written through slot indices derived from paged_k_cache's
    # page_size, so a shorter value cache is scattered out of bounds. Note the
    # mismatch has to preserve stride(0) — AITER has its own stride(0) equality
    # check, so e.g. a doubled head_dim would be caught there and this test would
    # pass with our guard removed. Fewer pages is the case only we catch.
    with pytest.raises(RuntimeError, match="paged_k_cache and paged_v_cache"):
        mod.append_paged_kv_cache_aiter(
            k,
            v,
            batch_indices,
            positions,
            k_cache,
            torch.zeros(
                num_pages - 1,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ),
            kv_indices,
            kv_indptr,
            unit,
            unit,
        )

    # Strides are counted in elements, so a narrower v-cache has the identical
    # stride tuple and slips past AITER's stride(0) check while the kernel writes
    # wider elements into it.
    with pytest.raises(RuntimeError, match="paged_k_cache and paged_v_cache"):
        mod.append_paged_kv_cache_aiter(
            k,
            v,
            batch_indices,
            positions,
            k_cache,
            v_cache.to(torch.float16),
            kv_indices,
            kv_indptr,
            unit,
            unit,
        )

    # AITER dispatches on the source dtype, so a wider append_key writes past the
    # end of the cache. The Python gate only inspects paged_k_cache.
    with pytest.raises(RuntimeError, match="append_key/append_value"):
        mod.append_paged_kv_cache_aiter(
            k.to(torch.float32),
            v.to(torch.float32),
            batch_indices,
            positions,
            k_cache,
            v_cache,
            kv_indices,
            kv_indptr,
            unit,
            unit,
        )

    # batch_indices longer than append_key would read past the end of append_key.
    with pytest.raises(RuntimeError, match="append_key.size"):
        mod.append_paged_kv_cache_aiter(
            k,
            v,
            torch.cat([batch_indices, batch_indices[:1]]),
            torch.cat([positions, positions[:1]]),
            k_cache,
            v_cache,
            kv_indices,
            kv_indptr,
            unit,
            unit,
        )
