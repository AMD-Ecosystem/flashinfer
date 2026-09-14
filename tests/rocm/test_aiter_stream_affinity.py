# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""AITER POD-API shims must launch on the caller's HIP stream.

The POD entry points read `aiter::getCurrentHIPStream()`, a thread-local that
defaults to nullptr and is set only by AITER's Python layer. A shim that omits
`aiter_compat::StreamGuard` enqueues its kernel on the null stream while the
surrounding `at::empty_like` and `copy_` run on the caller's -- a silent race
that every existing test misses, because they all run on the default stream
where the two happen to coincide.

vLLM and SGLang both drive a rank inside `with torch.cuda.stream(s)`, so this is
the production configuration, not an exotic one.
"""

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter

# requires_aiter, not is_aiter_supported: the latter answers for the
# architecture only, so on a box whose AITER is below the ABI floor these ran
# and failed inside require_aiter instead of skipping.
pytestmark = requires_aiter

DT = torch.float16


def _on_side_stream(fn):
    """Run `fn` on a non-default stream with real work queued ahead of it.

    The preceding work matters: it is what makes a null-stream launch observably
    early rather than merely differently ordered.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ballast = torch.randn(4096, 4096, device="cuda", dtype=DT)
        for _ in range(8):
            ballast = ballast @ ballast.T / 64.0
        out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("hidden", [128, 1024, 4096])
def test_rmsnorm_aiter_is_correct_on_a_side_stream(hidden):
    x = torch.randn(256, hidden, device="cuda", dtype=DT)
    w = torch.randn(hidden, device="cuda", dtype=DT)
    expected = flashinfer.rmsnorm(x, w, backend="aiter")

    got = _on_side_stream(lambda: flashinfer.rmsnorm(x, w, backend="aiter"))

    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_fused_add_rmsnorm_aiter_is_correct_on_a_side_stream():
    """Covers the in-place path, which stages into a fresh buffer and copies back.

    Measured: this case still passes with the guard removed. rmsnorm at 4096 and
    the rope case are the two that actually detect a missing guard.
    """
    hidden = 1024
    x = torch.randn(256, hidden, device="cuda", dtype=DT)
    res = torch.randn(256, hidden, device="cuda", dtype=DT)
    w = torch.randn(hidden, device="cuda", dtype=DT)

    x_ref, res_ref = x.clone(), res.clone()
    flashinfer.fused_add_rmsnorm(x_ref, res_ref, w, backend="aiter")

    x_got, res_got = x.clone(), res.clone()
    _on_side_stream(
        lambda: flashinfer.fused_add_rmsnorm(x_got, res_got, w, backend="aiter")
    )

    torch.testing.assert_close(x_got, x_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(res_got, res_ref, rtol=1e-2, atol=1e-2)


def test_rope_aiter_is_correct_on_a_side_stream():
    nnz, heads, head_dim = 512, 8, 128
    q = torch.randn(nnz, heads * head_dim, device="cuda", dtype=DT)
    k = torch.randn(nnz, heads * head_dim, device="cuda", dtype=DT)
    pos = torch.arange(nnz, device="cuda", dtype=torch.int32)
    cache = torch.randn(nnz, head_dim, device="cuda", dtype=torch.float32)

    def call():
        return flashinfer.apply_rope_with_cos_sin_cache(
            pos, q.clone(), k.clone(), head_dim, cache, is_neox=True, backend="aiter"
        )

    expected = call()
    got = _on_side_stream(call)

    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
