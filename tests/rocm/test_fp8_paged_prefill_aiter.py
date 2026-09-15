# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""fp8 paged prefill, the only route that serves fp8 at all.

The in-tree fa2 kernel rejects 8-bit types at compile time and the flat-gather
route runs mha_varlen_fwd, which has no fp8 arm -- so AITER's native paged
kernel is the whole of fp8 prefill support, and nothing else here exercised it.

It is also the only end-to-end check on the eight fp8 variants
docker/prebuild_aiter_attention.py composes: a CK ``--filter`` that matched no
instances still links, still passes the driver's own --check, and fails only
here, at dispatch.
"""

import pytest
import torch

import flashinfer
from flashinfer.rocm.prefill import (
    FP8_PREFILL_OUT_DTYPE,
    _aiter_paged_route_page_sizes,
    _native_fp8_dtype,
)
from tests.test_helpers.test_helpers import requires_aiter

pytestmark = requires_aiter

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 2, 128
PAGE, KV_LEN, Q_LEN, BATCH = 16, 512, 64, 2


def _fp8_dtype():
    """This GPU's fp8 encoding, from the same probe routing uses.

    The two encodings are indistinguishable by shape or by the .so name, and the
    wrong one is read under the wrong exponent bias and returns NaN.
    """
    pytest.importorskip("aiter")
    dtype = _native_fp8_dtype()
    if dtype is None:
        pytest.skip("aiter.dtypes.fp8 is unreadable, so the encoding is unknown")
    return dtype


@pytest.fixture(scope="module")
def workspace():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")


def _plan_and_run(workspace, q, kv, backend, **kw):
    pages_per_seq = KV_LEN // PAGE
    dev = q.device
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    w.plan(
        torch.arange(0, (BATCH + 1) * Q_LEN, Q_LEN, device=dev, dtype=torch.int32),
        torch.arange(
            0, (BATCH + 1) * pages_per_seq, pages_per_seq, device=dev, dtype=torch.int32
        ),
        torch.arange(BATCH * pages_per_seq, device=dev, dtype=torch.int32),
        torch.full((BATCH,), PAGE, device=dev, dtype=torch.int32),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=kv.dtype,
    )
    return w.run(q, kv, **kw)


def test_fp8_paged_prefill_matches_a_bf16_reference(workspace):
    fp8 = _fp8_dtype()
    if PAGE not in _aiter_paged_route_page_sizes(fp8):
        pytest.skip(f"page_size={PAGE} does not route to the native paged kernel")

    torch.manual_seed(0)
    dev = "cuda:0"
    q = torch.randn(
        BATCH * Q_LEN, NUM_QO_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16
    )
    kv = torch.randn(
        BATCH * (KV_LEN // PAGE),
        2,
        PAGE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=dev,
        dtype=torch.bfloat16,
    )

    ref = _plan_and_run(workspace, q, kv, "fa2")

    # From the dtype, not a constant: e4m3fnuz maxes at 240 and e4m3fn at 448,
    # so a fixed divisor wastes half the range on one of the two architectures.
    fp8_max = torch.finfo(fp8).max
    sq = q.abs().amax().float() / fp8_max
    sk = kv.abs().amax().float() / fp8_max
    got = _plan_and_run(
        workspace,
        (q.float() / sq).to(fp8),
        (kv.float() / sk).to(fp8),
        "aiter",
        scale_q=sq.reshape(1),
        scale_k=sk.reshape(1),
        scale_v=sk.reshape(1),
    )

    assert got.dtype == FP8_PREFILL_OUT_DTYPE
    assert not torch.isnan(got).any(), "fp8 prefill returned NaN"
    rel = (got.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    # Loose on purpose: this asserts the kernel ran and is not garbage, not that
    # fp8 quantisation is accurate.
    assert rel < 0.15, f"fp8 output differs from the bf16 reference by rel={rel:.4f}"


def test_fp8_prefill_is_refused_on_fa2(workspace):
    """fa2 has no fp8 kernel, so the refusal must be an error, not a fallback."""
    fp8 = _fp8_dtype()
    dev = "cuda:0"
    q = torch.zeros(
        BATCH * Q_LEN, NUM_QO_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16
    )
    kv = torch.zeros(
        BATCH * (KV_LEN // PAGE),
        2,
        PAGE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=dev,
        dtype=torch.bfloat16,
    )

    with pytest.raises(NotImplementedError, match="fp8"):
        _plan_and_run(workspace, q.to(fp8), kv.to(fp8), "fa2")
