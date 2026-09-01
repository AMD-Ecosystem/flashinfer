# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Sliding-window prefill, with the non-causal case the tree never covered.

Upstream's window tests are all causal, and tests/rocm/test_sliding_window.py is
decode-only, which is how AITER shipped silently returning the *unwindowed*
result for causal=False: the variant key drove the .so mask axis off causal, so
the request loaded a _nmask binary with no masking compiled in.
"""

import logging

import pytest
import torch

import flashinfer
from flashinfer.jit.core import logger
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.prefill import _aiter_needs_mask, _aiter_ops_importable

logger.setLevel(logging.ERROR)

HEAD_DIM = 128
NUM_HEADS = 4


def _skip_unless_aiter(backend: str, device: torch.device, op: str) -> None:
    """Stand down when AITER cannot serve ``op`` here.

    The arch check alone is not enough: a gated toolchain (gfx950 on ROCm 7.2.x)
    raises ArchCapabilityError from the wrapper constructor, which would error
    these tests rather than skip them.
    """
    if backend != "aiter":
        return
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    from flashinfer.rocm.arch_caps import capability_available, capability_reason

    if not capability_available(device, op, "aiter"):
        pytest.skip(capability_reason(device, op, "aiter"))


def _reference(q, k, v, causal: bool, window_left: int) -> torch.Tensor:
    """fp32 attention with FlashInfer's window semantics.

    The band is bottom-right anchored and left-bounded only, applied
    independently of ``causal`` -- see include/flashinfer/attention/variants.cuh.
    """
    qo_len, kv_len = q.shape[0], k.shape[0]
    qs, ks, vs = (t.transpose(0, 1).float() for t in (q, k, v))
    logits = (qs @ ks.transpose(-1, -2)) * q.shape[-1] ** -0.5

    row = torch.arange(qo_len, device=q.device)[:, None]
    col = torch.arange(kv_len, device=q.device)[None, :]
    query_positions = kv_len - qo_len + row

    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device)
    if causal:
        mask &= query_positions >= col
    if window_left >= 0:
        mask &= query_positions - window_left <= col

    p = torch.softmax(logits.masked_fill(~mask[None], float("-inf")), dim=-1)
    return (p @ vs).transpose(0, 1)


def _randn(shape, dtype, device):
    return torch.randn(*shape, dtype=dtype, device=device)


def _tolerance(dtype: torch.dtype) -> float:
    return 2e-2 if dtype is torch.bfloat16 else 1e-2


# window_left=0 is the tightest band -- one token per row -- and the case most
# likely to expose an off-by-one in the bottom-right anchoring.
WINDOWS = [-1, 0, 31, 127]
DTYPES = [torch.float16, torch.bfloat16]
BACKENDS = ["fa2", "aiter"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window_left", WINDOWS)
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_window(backend, dtype, window_left, causal):
    device = torch.device("cuda:0")
    _skip_unless_aiter(backend, device, "single_prefill")

    qo_len, kv_len = 37, 512
    torch.manual_seed(0)
    q = _randn((qo_len, NUM_HEADS, HEAD_DIM), dtype, device)
    k = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)
    v = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)

    out = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, window_left=window_left, backend=backend
    ).float()
    ref = _reference(q, k, v, causal, window_left)
    torch.testing.assert_close(out, ref, rtol=_tolerance(dtype), atol=_tolerance(dtype))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window_left", WINDOWS)
@pytest.mark.parametrize("causal", [False, True])
def test_ragged_prefill_window(backend, dtype, window_left, causal):
    device = torch.device("cuda:0")
    _skip_unless_aiter(backend, device, "batch_prefill")

    qo_len, kv_len = 37, 512
    torch.manual_seed(0)
    q = _randn((qo_len, NUM_HEADS, HEAD_DIM), dtype, device)
    k = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)
    v = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", backend=backend
    )
    wrapper.plan(
        torch.tensor([0, qo_len], dtype=torch.int32, device=device),
        torch.tensor([0, kv_len], dtype=torch.int32, device=device),
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = wrapper.run(q, k, v).float()
    ref = _reference(q, k, v, causal, window_left)
    torch.testing.assert_close(out, ref, rtol=_tolerance(dtype), atol=_tolerance(dtype))


# 1024 is a native AITER page size and takes mha_batch_prefill; 16 is not and
# degrades to the flat-gather kernel. Those are two different args sites.
@pytest.mark.parametrize("page_size", [1024, 16])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window_left", WINDOWS)
@pytest.mark.parametrize("causal", [False, True])
def test_paged_prefill_window(page_size, backend, dtype, window_left, causal):
    device = torch.device("cuda:0")
    _skip_unless_aiter(backend, device, "batch_prefill")

    qo_len, kv_len = 37, 2048
    num_pages = (kv_len + page_size - 1) // page_size
    torch.manual_seed(0)
    q = _randn((qo_len, NUM_HEADS, HEAD_DIM), dtype, device)
    kv = _randn((num_pages, 2, page_size, NUM_HEADS, HEAD_DIM), dtype, device)
    k = kv[:, 0].reshape(-1, NUM_HEADS, HEAD_DIM)[:kv_len]
    v = kv[:, 1].reshape(-1, NUM_HEADS, HEAD_DIM)[:kv_len]

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    wrapper.plan(
        torch.tensor([0, qo_len], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([(kv_len - 1) % page_size + 1], dtype=torch.int32, device=device),
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        page_size,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # Without this the test can silently narrow: if the native-paging probe
    # degrades, both page sizes take flat-gather and the native args site goes
    # untested while the suite stays green.
    if backend == "aiter" and page_size == 1024:
        assert wrapper._aiter_flat_gather_idx is None, (
            "expected the native-paged kernel at page_size=1024; AITER degraded "
            "to flat-gather, so this case no longer covers batch_prefill.cuh's "
            "native args site"
        )

    out = wrapper.run(q, kv).float()
    ref = _reference(q, k, v, causal, window_left)
    torch.testing.assert_close(out, ref, rtol=_tolerance(dtype), atol=_tolerance(dtype))


@pytest.mark.parametrize(
    "causal,window_left,kv_len,expected",
    [
        # The bug: non-causal + a real window still needs the _mask variant.
        (False, 31, 512, True),
        (True, -1, 512, True),
        (False, -1, 512, False),
        # A window at least as wide as the sequence masks nothing, so it must not
        # pull an unwindowed call onto _mask and its cold CK build.
        (False, 512, 512, False),
        (False, 1024, 512, False),
        (True, 512, 512, True),
        # Without kv_len the caller cannot normalize, so the window stands.
        (False, 512, None, True),
    ],
)
def test_needs_mask_table(causal, window_left, kv_len, expected):
    assert _aiter_needs_mask(causal, window_left, kv_len) is expected


@pytest.mark.parametrize("dtype", DTYPES)
def test_window_wider_than_sequence_matches_no_window(dtype):
    """The normalization must be numerically invisible, not just cheaper."""
    device = torch.device("cuda:0")
    _skip_unless_aiter("aiter", device, "single_prefill")

    qo_len, kv_len = 37, 256
    torch.manual_seed(0)
    q = _randn((qo_len, NUM_HEADS, HEAD_DIM), dtype, device)
    k = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)
    v = _randn((kv_len, NUM_HEADS, HEAD_DIM), dtype, device)

    wide = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=False, window_left=kv_len, backend="aiter"
    ).float()
    none = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=False, window_left=-1, backend="aiter"
    ).float()
    torch.testing.assert_close(wide, none, rtol=0, atol=0)


def test_aiter_and_fa2_agree_on_a_non_causal_window():
    """The regression itself: fa2 was right and AITER silently was not."""
    device = torch.device("cuda:0")
    _skip_unless_aiter("aiter", device, "single_prefill")

    qo_len, kv_len, window_left = 37, 512, 64
    torch.manual_seed(0)
    q = _randn((qo_len, NUM_HEADS, HEAD_DIM), torch.float16, device)
    k = _randn((kv_len, NUM_HEADS, HEAD_DIM), torch.float16, device)
    v = _randn((kv_len, NUM_HEADS, HEAD_DIM), torch.float16, device)

    kwargs = dict(causal=False, window_left=window_left)
    aiter_out = flashinfer.single_prefill_with_kv_cache(
        q, k, v, backend="aiter", **kwargs
    ).float()
    fa2_out = flashinfer.single_prefill_with_kv_cache(
        q, k, v, backend="fa2", **kwargs
    ).float()

    # Guard against the test passing because both ignore the window: the
    # unwindowed answer must be measurably different from the windowed one.
    unwindowed = _reference(q, k, v, causal=False, window_left=-1)
    windowed = _reference(q, k, v, causal=False, window_left=window_left)
    assert (unwindowed - windowed).abs().max() > 0.1

    torch.testing.assert_close(aiter_out, fa2_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(aiter_out, windowed, rtol=1e-2, atol=1e-2)
