# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the AITER reshape_and_cache_flash KV-append backend vs the
# native flashinfer-ROCm append_paged_kv_cache kernel.

import logging
import re

import pytest
import torch

import flashinfer
from flashinfer.rocm.aiter_utils import is_aiter_supported
from tests.test_helpers.test_helpers import requires_aiter
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

# @requires_aiter was this module's only GPU/arch gate until two routing tests
# dropped it (they must run without the AITER package). This keeps the arch gate
# for the whole file; the routing tests still do not require aiter to be
# importable.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_aiter_supported(torch.device("cuda:0")),
    reason="requires a gfx942/gfx950 GPU",
)


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


# Deliberately NOT @requires_aiter: the point is that routing does not depend on
# AITER at all. Gating on the package would skip exactly the environments where a
# reintroduced AITER probe would go unnoticed.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_append_paged_kv_cache_auto_always_routes_native(dtype, kv_layout):
    """auto picks native for every input, including the ones AITER supports.

    AITER's reshape_and_cache_flash is correct but slower than the in-tree kernel
    at every size measured on gfx942 (2.86 vs 3.62 TB/s), so the fp16/bf16 + NHD
    combination it does support is routed native too. backend='aiter' remains the
    way to reach it.
    """
    from flashinfer.rocm.page import _auto_select_kv_append_backend

    device = torch.device("cuda:0")
    assert (
        _auto_select_kv_append_backend(device, dtype=dtype, kv_layout=kv_layout)
        == "native"
    )


@requires_aiter
def test_append_explicit_aiter_accepts_int64_indices():
    """int64 index tensors must work, as they do on the native path.

    The shim requires int32, so without normalization in _aiter_append_paged_kv_cache
    an int64 page table -- which append_paged_kv_cache otherwise accepts, and which
    the native kernel narrows with .int() -- would fail only when AITER is selected.
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

    def _run(cast):
        kc = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        vc = torch.zeros_like(kc)
        flashinfer.append_paged_kv_cache(
            k,
            v,
            cast(batch_indices),
            cast(positions),
            (kc, vc),
            cast(kv_indices),
            cast(kv_indptr),
            cast(kv_last_page_len),
            backend="aiter",
        )
        return kc, vc

    k32, v32 = _run(lambda t: t)
    k64, v64 = _run(lambda t: t.long())
    # Without this, a shim that silently no-oped would leave both runs zero and
    # the comparison below would pass.
    assert (k32 != 0).any(), "the int32 run wrote nothing"
    torch.testing.assert_close(k64, k32, rtol=0, atol=0)
    torch.testing.assert_close(v64, v32, rtol=0, atol=0)


@requires_aiter
def test_append_aiter_rejects_noncontiguous_inputs():
    """A fused KV projection yields non-contiguous halves; AITER cannot take them.

    AITER reads stride(0) but assumes the dimensions below it are packed, so a
    strided append_key writes the wrong bytes for every head past the first --
    silently. The native kernel takes full stride arrays and handles the same
    input, so the shim must refuse rather than corrupt.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    nnz, num_kv_heads, head_dim, page_size, num_pages = 8, 4, 64, 16, 4

    torch.manual_seed(0xC0FFEE)
    fused = torch.randn(nnz, num_kv_heads, 2 * head_dim, dtype=dtype, device=device)
    k, v = fused[..., :head_dim], fused[..., head_dim:]
    assert not k.is_contiguous()

    indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([nnz], dtype=torch.int32, device=device)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        indptr, seq_lens, nnz
    )
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([nnz], dtype=torch.int32, device=device)

    def _caches():
        c = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        return c, torch.zeros_like(c)

    # native accepts strided input and is the reference.
    k_native, v_native = _caches()
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
    assert (k_native != 0).any(), "reference append wrote nothing"

    with pytest.raises(RuntimeError, match="dense inside stride"):
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            _caches(),
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            backend="aiter",
        )

    # Made contiguous, aiter must agree with native bit for bit.
    k_aiter, v_aiter = _caches()
    flashinfer.append_paged_kv_cache(
        k.contiguous(),
        v.contiguous(),
        batch_indices,
        positions,
        (k_aiter, v_aiter),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        backend="aiter",
    )
    torch.testing.assert_close(k_aiter, k_native, rtol=0, atol=0)
    torch.testing.assert_close(v_aiter, v_native, rtol=0, atol=0)


def test_append_native_accepts_the_5d_combined_cache():
    """Same shape as the aiter case below, but reachable without amd-aiter installed.

    Every other 5-D case is @requires_aiter, so on a box without the package this
    is the only thing that launches the native kernel on the combined layout.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    nnz, num_kv_heads, head_dim, page_size, num_pages = 8, 4, 64, 16, 4

    torch.manual_seed(0x5D)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([nnz], dtype=torch.int32, device=device)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        indptr, seq_lens, nnz
    )
    combined = torch.zeros(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        combined,
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.tensor([nnz], dtype=torch.int32, device=device),
        backend="native",
    )
    torch.testing.assert_close(
        combined[0, 0, :nnz], k.view(nnz, num_kv_heads, head_dim)
    )
    torch.testing.assert_close(
        combined[0, 1, :nnz], v.view(nnz, num_kv_heads, head_dim)
    )


@requires_aiter
def test_append_aiter_accepts_the_5d_combined_cache():
    """The documented 5-D [P, 2, S, H, D] cache must still reach AITER.

    _unpack_paged_kv_cache unbinds it into halves whose stride(0) is 2*S*H*D with
    dense interiors -- not contiguous, but exactly what AITER indexes against,
    since it reads stride(0) and assumes only the dimensions below it are packed.
    A blanket is_contiguous() check rejects this form.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    nnz, num_kv_heads, head_dim, page_size, num_pages = 8, 4, 64, 16, 4

    torch.manual_seed(0x5D)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([nnz], dtype=torch.int32, device=device)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        indptr, seq_lens, nnz
    )
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([nnz], dtype=torch.int32, device=device)

    def _run(backend):
        combined = torch.zeros(
            num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            combined,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            backend=backend,
        )
        return combined

    native = _run("native")
    assert (native != 0).any(), "reference append wrote nothing"
    torch.testing.assert_close(_run("aiter"), native, rtol=0, atol=0)


@requires_aiter
def test_append_empty_batch_is_a_noop_under_aiter():
    """An empty append step is a no-op, not a launch failure.

    A scheduler that drains to zero requests issues this routinely. AITER's host
    function does dim3 grid(key.size(0)) with no zero guard, so the shim returns
    before dispatching.

    backend='native' is deliberately not exercised: it raises
    hipErrorInvalidConfiguration for the same input, because
    include/flashinfer/rocm/attention/page.cuh:398 computes nblks(0). That is
    upstream CUDA code this port does not modify, so the shim is simply better
    here rather than at parity.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_kv_heads, head_dim, page_size, num_pages = 4, 64, 16, 4

    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    k = torch.empty(0, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.empty_like(k)
    kv_indptr = torch.tensor([0], dtype=torch.int32, device=device)

    kc = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    vc = torch.zeros_like(kc)
    flashinfer.append_paged_kv_cache(
        k,
        v,
        empty_i32,
        empty_i32,
        (kc, vc),
        empty_i32,
        kv_indptr,
        empty_i32,
        backend="aiter",
    )
    torch.cuda.synchronize()  # a deferred launch failure would surface here
    assert not bool(kc.any().item()), "empty append wrote to the cache"


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

    k_ref = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_ref = torch.zeros_like(k_ref)
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        (k_ref, v_ref),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        backend="native",
    )
    torch.cuda.synchronize()

    # Queue a long blocker on a side stream, then the append behind it, and read
    # the cache from the host *while the side stream is still busy*. An append
    # that honours the current stream cannot have run yet, so the cache is still
    # zero; one that ignores it and lands on the null stream is not behind the
    # blocker, completes in ~11 us, and the read sees data.
    #
    # Checking contents after any sync cannot detect this -- the null-stream
    # append finishes long before the blocker either way. That is the flaw in
    # the obvious version of this test.
    side = torch.cuda.Stream()
    k_cache = torch.zeros_like(k_ref)
    v_cache = torch.zeros_like(v_ref)
    # Warm the AITER path first. Its first call loads the kernel module, and
    # hipModuleLoad synchronizes the device -- inside the block below that would
    # drain the blocker before the observation.
    warm_k, warm_v = torch.zeros_like(k_ref), torch.zeros_like(v_ref)
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        (warm_k, warm_v),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        backend="aiter",
    )
    torch.cuda.synchronize()

    # Preallocated out= buffers: allocating per iteration exhausts the caching
    # allocator, which synchronizes to free and drains the very queue this test
    # depends on staying full.
    burn_a = torch.randn(4096, 4096, dtype=torch.float32, device=device)
    burn_b = torch.empty_like(burn_a)

    with torch.cuda.stream(side):
        for _ in range(100):
            torch.mm(burn_a, burn_a, out=burn_b)
            torch.mm(burn_b, burn_b, out=burn_a)
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
        done = torch.cuda.Event()
        done.record(side)

    # If the blocker already drained, the observation below proves nothing. Fail
    # with a clear reason rather than passing vacuously. Measured margin on
    # gfx942: the blocker still has ~140 ms left at this point.
    assert not done.query(), (
        "side stream drained before the cache was read; the blocker is too small "
        "for this GPU and the test cannot observe stream ordering"
    )
    # No sleep needed: .item() issues on the default stream, which is the stream a
    # rogue append would have landed on, so this read is ordered after it.
    still_empty = not bool(k_cache.any().item())
    done.synchronize()

    assert still_empty, (
        "the AITER append completed while the side stream it was issued on was "
        "still blocked, so it ran on another stream (likely the null stream)"
    )
    torch.testing.assert_close(k_cache, k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_ref, rtol=0, atol=0)


# Also not @requires_aiter, for the same reason: breaking get_page_aiter_module
# is what this asserts, and that does not need the package installed.
def test_append_auto_never_builds_the_aiter_shim(monkeypatch):
    """'auto' routes to native without so much as loading the AITER module.

    Routing is a static decision (AITER's append is slower), so 'auto' must not
    pay AITER's build cost to discover that. Breaking get_page_aiter_module is
    what makes this fail if the routing ever consults it again -- asserting only
    on the returned string would not.
    """
    from flashinfer.rocm import page as page_mod

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
    from flashinfer.rocm import page as page_mod

    mod = page_mod.get_page_aiter_module()
    unit = page_mod._aiter_unit_scale(device)

    with pytest.raises(RuntimeError, match="must be on a GPU"):
        mod.append_paged_kv_cache_aiter(
            k.cpu(),
            v.cpu(),
            batch_indices.cpu(),
            positions.cpu(),
            k_cache.cpu(),
            v_cache.cpu(),
            kv_indices.cpu(),
            kv_indptr.cpu(),
            unit.cpu(),
            unit.cpu(),
        )

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
