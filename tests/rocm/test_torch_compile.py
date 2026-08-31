# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``FLASHINFER_USE_TORCH_CUSTOM_OPS`` and ``torch.compile`` on ROCm.

Because the flag is evaluated at import time, each test that
needs a different env-var value runs in a subprocess so the module is freshly
imported with the desired setting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch


pytestmark = [
    pytest.mark.skipif(
        not hasattr(torch.version, "hip") or torch.version.hip is None,
        reason="HIP not available",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPU available",
    ),
]


_PREAMBLE = textwrap.dedent(
    """\
    import torch
    from flashinfer.page import append_paged_kv_cache
    import flashinfer

    B, PAGE_SIZE, KV_HEADS, HEAD_DIM, NUM_TOKENS = 2, 16, 1, 128, 8
    DEVICE = "cuda"

    pages = B * 2
    k_cache = torch.zeros(pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)

    indptr = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * 2
    indices = torch.arange(pages, dtype=torch.int32, device=DEVICE)
    last_page_len = torch.full((B,), PAGE_SIZE, dtype=torch.int32, device=DEVICE)

    batch_idx = torch.arange(B, device=DEVICE, dtype=torch.int32).repeat_interleave(NUM_TOKENS)
    positions = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32).repeat(B)

    k = torch.randn(B * NUM_TOKENS, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def append(k, v):
        append_paged_kv_cache(k, v, batch_idx, positions, (k_cache, v_cache), indices, indptr, last_page_len)
        return k

    def append_aiter(k, v):
        append_paged_kv_cache(k, v, batch_idx, positions, (k_cache, v_cache), indices, indptr, last_page_len, backend="aiter")
        return k
"""
)


def _run_snippet(
    snippet: str, env_override: dict[str, str] | None = None, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **(env_override or {})}
    return subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_eager_without_custom_ops():
    """append_paged_kv_cache works in eager mode with custom ops disabled."""
    snippet = _PREAMBLE + textwrap.dedent(
        """\
        assert not flashinfer.use_torch_custom_ops_enabled()
        append(k, v)
        print("OK")
    """
    )
    result = _run_snippet(snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "0"})
    assert result.returncode == 0, f"eager (custom ops off) failed:\n{result.stderr}"


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)
def test_eager_with_custom_ops():
    """append_paged_kv_cache works in eager mode with custom ops enabled."""
    snippet = _PREAMBLE + textwrap.dedent(
        """\
        assert flashinfer.use_torch_custom_ops_enabled()
        append(k, v)
        print("OK")
    """
    )
    result = _run_snippet(snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "1"})
    assert result.returncode == 0, f"eager (custom ops on) failed:\n{result.stderr}"


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)
def test_torch_compile_with_custom_ops():
    """torch.compile succeeds when FLASHINFER_USE_TORCH_CUSTOM_OPS=1."""
    snippet = _PREAMBLE + textwrap.dedent(
        """\
        assert flashinfer.use_torch_custom_ops_enabled()
        compiled = torch.compile(append, dynamic=True)
        compiled(k, v)
        print("OK")
    """
    )
    result = _run_snippet(snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "1"})
    assert result.returncode == 0, f"torch.compile failed:\n{result.stderr}"


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)
def test_torch_compile_with_aiter_backend():
    """torch.compile succeeds through the AITER shim's custom-op wrapper.

    backend='auto' routes to the native kernel, so the shim is reachable only
    via an explicit opt-in -- and without this test nothing compiles it.

    What it guards is the @register_custom_op wrapper: the shim is registered
    with TORCH_LIBRARY_FRAGMENT, so without a wrapper Dynamo traces into it and
    its numel()/size() calls raise on symbolic shapes. Verified by removing the
    wrapper, which fails this test. The paired fake is *not* what saves it --
    a None-returning custom op needs no separate fake impl -- so removing that
    alone leaves the test green.
    """
    snippet = _PREAMBLE + textwrap.dedent(
        """\
        import sys
        from flashinfer.rocm.aiter_utils import is_aiter_available
        if not is_aiter_available(k_cache.device, "append_paged_kv_cache"):
            sys.exit(99)  # sentinel: no AITER here, report a skip rather than a pass
        compiled = torch.compile(append_aiter, dynamic=True)
        compiled(k, v)
        assert (k_cache != 0).any(), "compiled aiter append wrote nothing"
        print("OK")
    """
    )
    # Longer than the default: this is the only test here whose child may have to
    # build the AITER module and hipcc the shim from a cold ~/.cache/flashinfer,
    # and it can additionally block on another xdist worker holding the JIT lock.
    result = _run_snippet(
        snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "1"}, timeout=900
    )
    if result.returncode == 99:
        pytest.skip("aiter package not available for the append backend")
    assert result.returncode == 0, f"torch.compile (aiter) failed:\n{result.stderr}"


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)
def test_torch_compile_mla_append():
    """torch.compile succeeds through append_paged_mla_kv_cache.

    That op is newly exported on HIP in this branch, but nothing compiled it --
    a registration or signature regression would have kept the suite green.

    As with the AITER case above, the custom-op wrapper is what this guards, not
    the fake: removing @register_custom_op fails this test, removing the fake
    alone does not.
    """
    snippet = textwrap.dedent(
        """\
        import torch
        from flashinfer.page import append_paged_mla_kv_cache
        import flashinfer

        B, PAGE_SIZE, CKV, KPE, NUM_TOKENS = 2, 16, 512, 64, 8
        DEVICE = "cuda"
        pages = B * 2

        ckv_cache = torch.zeros(pages, PAGE_SIZE, CKV, device=DEVICE, dtype=torch.bfloat16)
        kpe_cache = torch.zeros(pages, PAGE_SIZE, KPE, device=DEVICE, dtype=torch.bfloat16)

        indptr = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * 2
        indices = torch.arange(pages, dtype=torch.int32, device=DEVICE)
        last_page_len = torch.full((B,), PAGE_SIZE, dtype=torch.int32, device=DEVICE)
        batch_idx = torch.arange(B, device=DEVICE, dtype=torch.int32).repeat_interleave(NUM_TOKENS)
        positions = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32).repeat(B)

        ckv = torch.randn(B * NUM_TOKENS, CKV, device=DEVICE, dtype=torch.bfloat16)
        kpe = torch.randn(B * NUM_TOKENS, KPE, device=DEVICE, dtype=torch.bfloat16)

        def mla_append(ckv, kpe):
            append_paged_mla_kv_cache(ckv, kpe, batch_idx, positions, ckv_cache, kpe_cache,
                                      indices, indptr, last_page_len)
            return ckv

        assert flashinfer.use_torch_custom_ops_enabled()
        compiled = torch.compile(mla_append, dynamic=True)
        compiled(ckv, kpe)
        assert (ckv_cache != 0).any(), "compiled MLA append wrote nothing"
        print("OK")
    """
    )
    result = _run_snippet(snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "1"})
    assert result.returncode == 0, (
        f"torch.compile (MLA append) failed:\n{result.stderr}"
    )


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)
def test_torch_compile_without_custom_ops_fails():
    """torch.compile fails when custom ops are disabled."""
    snippet = _PREAMBLE + textwrap.dedent(
        """\
        assert not flashinfer.use_torch_custom_ops_enabled()
        compiled = torch.compile(append, dynamic=True)
        try:
            compiled(k, v)
        except Exception:
            print("OK: torch.compile raised as expected")
        else:
            raise AssertionError("Expected error but torch.compile succeeded")
    """
    )
    result = _run_snippet(snippet, {"FLASHINFER_USE_TORCH_CUSTOM_OPS": "0"})
    assert result.returncode == 0, f"unexpected failure:\n{result.stderr}"
