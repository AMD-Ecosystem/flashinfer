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
import uuid
from typing import Optional

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


_AITER_PREAMBLE = textwrap.dedent(
    """\
    import sys
    import torch
    import flashinfer
    from flashinfer.rocm.aiter_utils import is_aiter_supported
    from flashinfer.rocm.prefill import _aiter_ops_importable

    dev = torch.device("cuda:0")
    if not is_aiter_supported(dev) or not _aiter_ops_importable():
        sys.exit(99)
    dt = torch.bfloat16
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)
    torch.manual_seed(0)

    def check(fn, *args):
        \"\"\"Compiled == eager, and the AITER op is a node in the graph.\"\"\"
        ref = fn(*args)
        targets = []

        def record(gm, example_inputs):
            targets.extend(str(n.target) for n in gm.graph.nodes)
            return gm.forward

        torch._dynamo.reset()
        torch.compile(fn, backend=record, dynamic=True)(*args)
        assert any("flashinfer" in t and "aiter" in t for t in targets), targets
        torch._dynamo.reset()
        out = torch.compile(fn, dynamic=True)(*args)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    # run's output and lse, read back from a zeros buffer made here: an undeclared
    # write reads back as zeros under compile, which flashinfer's own
    # torch.empty lse would hide.
    def with_lse(run, rows, *args):
        lse = torch.zeros(rows, 8, device=dev, dtype=torch.float32)
        o, _ = run(*args, lse=lse, return_lse=True)
        return torch.cat([o.float().flatten(), (lse + 1).flatten()])

    def paged(page_size, head_dim, kv_len=2048):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="aiter")
        pages = kv_len // page_size
        w.plan(i32(0, 64), i32(0, pages), torch.arange(pages, dtype=torch.int32, device=dev),
               i32(page_size), 8, 8, head_dim, page_size, causal=True, q_data_type=dt)
        assert w.backend == "aiter"
        q = torch.randn(64, 8, head_dim, device=dev, dtype=dt)
        kv = torch.randn(pages, 2, page_size, 8, head_dim, device=dev, dtype=dt)
        return w, q, kv
"""
)


def _run_aiter_snippet(
    body: str, custom_ops: str = "1", fresh_inductor: bool = False
) -> None:
    # Cold, the child builds the AITER shims; see test_torch_compile_with_aiter_backend.
    env = {"FLASHINFER_USE_TORCH_CUSTOM_OPS": custom_ops}
    if fresh_inductor:
        # Inductor's on-disk caches ignore op alias schemas, so a graph compiled
        # by another checkout could replay here.
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    result = _run_snippet(_AITER_PREAMBLE + textwrap.dedent(body), env, timeout=1800)
    if result.returncode == 99:
        pytest.skip("AITER prefill unavailable here")
    assert result.returncode == 0, f"failed:\n{result.stderr[-4000:]}"


_needs_custom_ops = pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.4"),
    reason="torch.compile custom ops require torch >= 2.4",
)


@_needs_custom_ops
@pytest.mark.parametrize("backend", ["aiter", "auto"])
def test_torch_compile_aiter_single_prefill(backend):
    """auto resolves to AITER here, so this is the default route under compile.

    The two sliding-window settings share one AITER URI but not one module-cache
    key, so both module getters hand back the same registered op.
    """
    _run_aiter_snippet(
        f"""\
        q = torch.randn(64, 8, 128, device=dev, dtype=dt)
        k, v = torch.randn_like(q), torch.randn_like(q)
        for window_left in (-1, 32):
            check(lambda q, k, v: flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=True, window_left=window_left, backend="{backend}"), q, k, v)
        # The public API allocates lse itself; drive the module so a caller-owned
        # zeros buffer shows an undeclared lse write.
        from flashinfer.rocm.prefill import get_single_prefill_module
        run = get_single_prefill_module(
            "aiter", dt, dt, dt, 128, 128, 0, False, False, False).run
        # Builds the has_lse variant the raw run() below needs.
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, backend="aiter", return_lse=True)
        tmp = torch.empty(1, device=dev)

        def single_lse(q, k, v):
            o, lse = torch.empty_like(q), torch.zeros(64, 8, device=dev)
            run(q, k, v, tmp, o, lse, 1, 0, -1, None, None, 0.0, 128 ** -0.5,
                None, None, None, 1.0, 1e4)
            return torch.cat([o.float().flatten(), (lse + 1).flatten()])

        check(single_lse, q, k, v)
        """
    )


def test_customized_aiter_single_prefill_registers():
    """A caller's own parameter list reaches the named schema, flag off.

    The list covers every type the schema tail maps: Tensor?, Tensor, float, int, bool.
    """
    _run_aiter_snippet(
        """\
        from flashinfer.jit.rocm.modules import gen_customize_single_prefill_module
        module = gen_customize_single_prefill_module(
            "aiter", "single_prefill_aiter_mixed_params_bf16_hd128", dt, dt, dt, 128, 128,
            ["maybe_bias", "t"], ["float", "float"],
            ["logits_soft_cap", "sm_scale", "n", "flag"], ["double", "double", "int64_t", "bool"],
            "DefaultAttention<use_custom_mask, false, false, false>",
            "#include<flashinfer/rocm/attention/variants.cuh>",
        ).build_and_load()
        q = torch.randn(64, 8, 128, device=dev, dtype=dt)
        k, v = torch.randn_like(q), torch.randn_like(q)
        o = torch.empty_like(q)
        module.run(q, k, v, torch.empty(1, device=dev), o, None, 1, 0, -1,
                   None, torch.zeros(1, device=dev), 0.0, 128 ** -0.5, 3, True)
        ref = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")
        torch.testing.assert_close(o, ref, rtol=2e-2, atol=2e-2)
        """,
        custom_ops="0",
    )


@_needs_custom_ops
@pytest.mark.parametrize(
    "page_size, head_dim, gathers",
    [(1024, 128, False), (32, 128, True), (1024, 64, True)],
    ids=["native", "gather-page", "gather-head-dim"],
)
def test_torch_compile_aiter_paged_prefill(page_size, head_dim, gathers):
    _run_aiter_snippet(
        f"""\
        w, q, kv = paged({page_size}, {head_dim})
        assert (w._aiter_flat_gather_idx is not None) is {gathers}
        check(lambda q, kv: w.run(q, kv), q, kv)
        check(lambda q, kv: with_lse(w.run, 64, q, kv), q, kv)
        """
    )


@_needs_custom_ops
def test_torch_compile_aiter_paged_prefill_replanned():
    """One compiled function across two plans, short query then long.

    max_q_len reaches the op as an int the kernel reads; a graph that kept the
    first plan's value would under-size the second launch.
    """
    _run_aiter_snippet(
        """\
        w, q, kv = paged(1024, 128, kv_len=2048)
        # Two requests over the same 64 rows: only the split, and so max_q_len,
        # moves between plans, so no tensor shape forces a recompile.
        for qo_indptr in ((0, 32, 64), (0, 8, 64)):
            w.plan(i32(*qo_indptr), i32(0, 1, 2), i32(0, 1), i32(1024, 1024), 8, 8,
                   128, 1024, causal=True, q_data_type=dt)
            if qo_indptr[1] == 32:
                f = torch.compile(lambda q, kv: w.run(q, kv))
                f(q, kv)
        torch.testing.assert_close(f(q, kv), w.run(q, kv), rtol=2e-2, atol=2e-2)
        """
    )


@_needs_custom_ops
def test_torch_compile_aiter_ragged_prefill():
    _run_aiter_snippet(
        """\
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="aiter")
        w.plan(i32(0, 64), i32(0, 256), 8, 8, 128, causal=True, q_data_type=dt)
        assert w.backend == "aiter"
        q = torch.randn(64, 8, 128, device=dev, dtype=dt)
        k = torch.randn(256, 8, 128, device=dev, dtype=dt)
        v = torch.randn_like(k)
        check(lambda q, k, v: w.run(q, k, v), q, k, v)
        check(lambda q, k, v: with_lse(w.run, 64, q, k, v), q, k, v)
        """
    )


# AITER decode builds its return_lse shadow plan on the first such call after
# plan(), so each case compiles before any eager call; references use fa2.
_DECODE = """\
ws_ref = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

def replan(w, batch, kv_len, page=16):
    per = kv_len // page
    w.plan(i32(*range(0, batch * per + 1, per)),
           torch.arange(batch * per, dtype=torch.int32, device=dev),
           i32(*[page] * batch), 8, 2, 128, page, q_data_type=dt)

def decode(backend, workspace, batch, kv_len):
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend=backend)
    replan(w, batch, kv_len)
    return w

def inputs(batch, kv_len, page=16):
    q = torch.randn(batch, 8, 128, device=dev, dtype=dt)
    shape = (batch * kv_len // page, page, 2, 128)
    return q, (torch.randn(shape, device=dev, dtype=dt), torch.randn(shape, device=dev, dtype=dt))

def close(a, b):
    torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
"""


@_needs_custom_ops
@pytest.mark.parametrize("backend", ["aiter", "auto"])
def test_torch_compile_aiter_batch_decode(backend):
    """auto resolves to AITER here, so this is the default route under compile."""
    _run_aiter_snippet(
        _DECODE
        + f"""\
q, kv = inputs(4, 512)
w = decode({backend!r}, ws, 4, 512)
assert w.backend == "aiter"
ref = decode("fa2", ws_ref, 4, 512)
close(torch.compile(lambda q, kv: with_lse(w.run, 4, q, kv), dynamic=True)(q, kv),
      with_lse(ref.run, 4, q, kv))
replan(w, 4, 512)
torch._dynamo.reset()
o, lse = torch.compile(lambda q, kv: w.forward_return_lse(q, kv), dynamic=True)(q, kv)
ref_o, ref_lse = ref.run(q, kv, return_lse=True)
close(o, ref_o)
close(lse, ref_lse)
check(lambda q, kv: w.run(q, kv), q, kv)
"""
    )


@_needs_custom_ops
def test_torch_compile_aiter_batch_decode_replanned():
    """One compiled function across two split-KV plans: a stale shadow plan would
    under-cover the longer KV."""
    _run_aiter_snippet(
        _DECODE
        + """\
q1, kv1 = inputs(1, 2048)
q2, kv2 = inputs(1, 16384)
w = decode("aiter", ws, 1, 2048)
assert w.backend == "aiter"
f = torch.compile(lambda q, kv: with_lse(w.run, 1, q, kv), dynamic=True)
close(f(q1, kv1), with_lse(decode("fa2", ws_ref, 1, 2048).run, 1, q1, kv1))
first = w._fa2_lse_plan_info
replan(w, 1, 16384)
out = f(q2, kv2)
second = w._fa2_lse_plan_info
assert int(first[9]) == 1 and int(second[9]) == 1, (first, second)
assert not torch.equal(first, second), (first, second)
close(out, with_lse(decode("fa2", ws_ref, 1, 16384).run, 1, q2, kv2))
"""
    )


@_needs_custom_ops
@pytest.mark.parametrize(
    "route", ["decode fa2", "decode fa2 tensor cores", "decode aiter", "prefill fa2"]
)
def test_torch_compile_kv_cache_views(route):
    """Compiled attention over K/V views matches eager and leaves the cache
    unchanged; the reference is taken before compiling."""
    _run_aiter_snippet(
        f"""\
        route = {route!r}
        pages, page = 32, 16
        indptr = i32(*range(0, pages + 1, pages // 4))
        indices = torch.arange(pages, dtype=torch.int32, device=dev)
        if route.startswith("decode"):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", backend=route.split()[1], use_tensor_cores="tensor" in route)
            w.plan(indptr, indices, i32(*[page] * 4), 8, 2, 128, page, q_data_type=dt)
            q = torch.randn(4, 8, 128, device=dev, dtype=dt)
        else:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            w.plan(i32(0, 16, 32, 48, 64), indptr, indices, i32(*[page] * 4),
                   8, 2, 128, page, causal=True, q_data_type=dt)
            q = torch.randn(64, 8, 128, device=dev, dtype=dt)
        assert w.backend == route.split()[1], w.backend
        stacked = torch.randn(2, pages, page, 2, 128, device=dev, dtype=dt)
        interleaved = stacked.transpose(0, 1).contiguous()
        pool = torch.stack([torch.zeros_like(interleaved), interleaved])
        cases = (  # layer-1 last: before the fix it faults the GPU
            ("[2, pages] views", (stacked[0], stacked[1]), None),
            ("interleaved tensor", interleaved, True),
            ("layer 1 of a pool", pool[1], None),
            ("layer 1 of a pool", pool[1], True),
        )
        failures = []
        for name, kv, dynamic in cases:
            name = f"{{name}}, dynamic={{dynamic}}"
            ref = w.run(q, kv)
            before = [t.clone() for t in (pool, stacked, interleaved)]
            torch._dynamo.reset()
            run = torch.compile(lambda q, kv: w.run(q, kv), dynamic=dynamic, fullgraph=True)
            out = run(q, kv)
            torch.cuda.synchronize()
            status = "ok"
            try:
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
                assert all(torch.equal(t, b) for t, b in zip((pool, stacked, interleaved), before)), "cache changed"
            except AssertionError as e:
                failures.append(f"{{name}}: {{e}}")
                status = "FAILED"
            print(name, status, flush=True)
        assert not failures, "\\n".join(failures)
        """,
        fresh_inductor=True,
    )


@_needs_custom_ops
@pytest.mark.parametrize("route", ["decode fa2", "decode aiter", "prefill fa2"])
def test_torch_compile_append_and_attention_order(route):
    """In one compiled graph, attention reads the cache after an earlier append
    and before a later one."""
    _run_aiter_snippet(
        f"""\
        from flashinfer.page import append_paged_kv_cache
        route = {route!r}
        pages, page = 8, 16
        indptr, indices = i32(0, 4, 8), torch.arange(pages, dtype=torch.int32, device=dev)
        last = i32(page, page)
        if route.startswith("decode"):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend=route.split()[1])
            w.plan(indptr, indices, last, 8, 2, 128, page, q_data_type=dt)
            q = torch.randn(2, 8, 128, device=dev, dtype=dt)
        else:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            w.plan(i32(0, 16, 32), indptr, indices, last, 8, 2, 128, page, causal=True, q_data_type=dt)
            q = torch.randn(32, 8, 128, device=dev, dtype=dt)
        assert w.backend == route.split()[1], w.backend
        # Half of each request's positions, so untouched slots must still read
        # the original cache.
        k_new = torch.randn(2 * 2 * page, 2, 128, device=dev, dtype=dt)
        v_new = torch.randn_like(k_new)
        batch = torch.arange(2, device=dev, dtype=torch.int32).repeat_interleave(2 * page)
        pos = torch.arange(2 * page, device=dev, dtype=torch.int32).repeat(2)

        def append(k_cache, v_cache):
            append_paged_kv_cache(k_new, v_new, batch, pos, (k_cache, v_cache), indices, indptr, last)

        def append_then_attend(q, k_cache, v_cache):
            append(k_cache, v_cache)
            return w.run(q, (k_cache, v_cache))

        def attend_then_append(q, k_cache, v_cache):
            o = w.run(q, (k_cache, v_cache))
            append(k_cache, v_cache)
            return o

        separate = lambda: tuple(torch.randn(pages, page, 2, 128, device=dev, dtype=dt) for _ in range(2))
        def views():  # K and V as views of one interleaved tensor
            kv = torch.randn(pages, 2, page, 2, 128, device=dev, dtype=dt)
            return kv[:, 0], kv[:, 1]
        for caches, dynamic in ((separate, None), (views, True)):
            for step in (append_then_attend, attend_then_append):
                torch.manual_seed(1)
                ref_caches = caches()
                ref = step(q, *ref_caches)
                torch.manual_seed(1)
                got_caches = caches()
                torch._dynamo.reset()
                out = torch.compile(step, fullgraph=True, dynamic=dynamic)(q, *got_caches)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
                for got, want in zip(got_caches, ref_caches):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
        """,
        fresh_inductor=True,
    )


@_needs_custom_ops
def test_torch_compile_aiter_decode_reduce_overhead_rerecords():
    """AITER decode's first call inside a reduce-overhead warm-up must leave no
    untracked allocation in the CUDA-graph pool, across re-records."""
    _run_aiter_snippet(
        """\
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="aiter")
        w.plan(i32(*range(0, 129, 16)), torch.arange(128, dtype=torch.int32, device=dev),
               i32(*[16] * 8), 32, 8, 128, 16, q_data_type=dt)
        assert w.backend == "aiter", w.backend

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("kv", torch.randn(128, 2, 16, 8, 128, device=dev, dtype=dt))

            def forward(self, x, q):
                return x * 2, w.run(q, self.kv)

        m = Attn()
        q = torch.randn(8, 32, 256, device=dev, dtype=dt)[..., :128]  # the shim copies it
        # Reference from fa2 before compiling: an eager AITER call would create
        # the shim's state outside the graph pool.
        ref_w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(64 << 20, dtype=torch.uint8, device=dev), "NHD", backend="fa2")
        ref_w.plan(i32(*range(0, 129, 16)), torch.arange(128, dtype=torch.int32, device=dev),
                   i32(*[16] * 8), 32, 8, 128, 16, q_data_type=dt)
        ref = ref_w.run(q, m.kv)
        kv_before = m.kv.clone()
        f = torch.compile(m, mode="reduce-overhead")
        for _ in range(3):  # the first AITER call of this process is the warm-up
            f(torch.randn(64, device=dev), q)
        for n in (128, 256):  # new shapes: new tree paths
            f(torch.randn(n, device=dev), q)
            f(torch.randn(n, device=dev), q)
        out = f(torch.randn(64, device=dev), q)[1].clone()
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        assert torch.equal(m.kv, kv_before), "cache changed"
        """,
        fresh_inductor=True,
    )


_needs_disable_reason = pytest.mark.skipif(
    torch.torch_version.TorchVersion(torch.__version__)
    < torch.torch_version.TorchVersion("2.8"),
    reason="torch.compiler.disable takes reason= from torch 2.8",
)


@_needs_custom_ops
@_needs_disable_reason
def test_aiter_batch_decode_return_lse_fullgraph_names_the_reason():
    _run_aiter_snippet(
        _DECODE
        + """\
import torch._dynamo  # plan()'s prepare() needs it; do not rely on aiter loading it
q, kv = inputs(4, 512)
w = decode("aiter", ws, 4, 512)
assert w.backend == "aiter"
try:
    torch.compile(lambda q, kv: w.run(q, kv, return_lse=True), fullgraph=True)(q, kv)
except Exception as e:
    assert "AITER PA v1 emits no LSE" in str(e), str(e)[-2000:]
else:
    raise AssertionError("fullgraph traced the fa2 shadow plan")
ref = decode("fa2", ws_ref, 4, 512)  # the advice the reason gives
fa2 = torch.compile(lambda q, kv: with_lse(ref.run, 4, q, kv), fullgraph=True, dynamic=True)
close(fa2(q, kv), with_lse(ref.run, 4, q, kv))
"""
    )


def test_jit_trace_keeps_the_aiter_ops():
    """Flag off, the default: a traced AITER call must still run its kernel.

    A named schema with default alias analysis declares the void op side-effect
    free, and jit.trace's dead-code pass then returns the uninitialised output.
    Single prefill is absent: it builds its module from traced shapes.
    """
    _run_aiter_snippet(
        """\
        assert not flashinfer.use_torch_custom_ops_enabled()
        from flashinfer.rocm.fused_moe import shuffle_moe_weight
        w, q, kv = paged(1024, 128)
        M, E, K, I, topk = 7, 8, 512, 256, 2
        x = torch.randn(M, K, device=dev, dtype=dt) / 10
        w1 = shuffle_moe_weight(torch.randn(E, 2 * I, K, device=dev, dtype=dt) / 10)
        w2 = shuffle_moe_weight(torch.randn(E, K, I, device=dev, dtype=dt) / 10)
        ids = torch.stack([torch.randperm(E, device=dev)[:topk] for _ in range(M)]).int()
        weights = torch.softmax(torch.randn(M, topk, device=dev), -1)
        # Outputs are zeros made inside the traced function: a dropped op then
        # returns zeros, where torch.empty could hand back a stale correct result.
        for fn, args in (
            (lambda q, kv: w.run(q, kv, out=torch.zeros_like(q)), (q, kv)),
            (lambda x: flashinfer.aiter_fused_moe(
                x, w1, w2, ids, weights, out=torch.zeros_like(x)), (x,)),
        ):
            traced = torch.jit.trace(fn, args, check_trace=False)
            torch.testing.assert_close(traced(*args), fn(*args), rtol=2e-2, atol=2e-2)
        """,
        custom_ops="0",
    )


@_needs_custom_ops
def test_torch_compile_aiter_fused_moe():
    _run_aiter_snippet(
        """\
        from flashinfer.rocm.fused_moe import shuffle_moe_weight
        M, E, K, I, topk = 7, 8, 512, 256, 2
        x = torch.randn(M, K, device=dev, dtype=dt) / 10
        w1 = shuffle_moe_weight(torch.randn(E, 2 * I, K, device=dev, dtype=dt) / 10)
        w2 = shuffle_moe_weight(torch.randn(E, K, I, device=dev, dtype=dt) / 10)
        ids = torch.stack([torch.randperm(E, device=dev)[:topk] for _ in range(M)]).int()
        weights = torch.softmax(torch.randn(M, topk, device=dev), -1)
        check(lambda x: flashinfer.aiter_fused_moe(x, w1, w2, ids, weights), x)
        """
    )


class TestRegistrationInProcess:
    """The decorators themselves, with the module-level flag patched.

    The subprocess cases above prove the env var is honoured at import; they
    cannot reach the decorator bodies, because the flag is off in the parent and
    the subprocess's coverage is its own. Patching the already-imported constant
    is what exercises the arms a serving process with the flag on would take.
    """

    @staticmethod
    def _tc():
        from flashinfer.rocm import torch_compile as tc

        return tc

    @staticmethod
    def _registered(name):
        """Resolve through torch.ops rather than torch._C._dispatch_has_kernel.

        The public surface answers both directions -- AttributeError before
        registration, an OpOverloadPacket after -- with no private API.
        """
        namespace, _, stem = name.partition("::")
        return hasattr(getattr(torch.ops, namespace), stem)

    @staticmethod
    def _unique(stem):
        """`torch.library` registration is process-global and permanent.

        CLAUDE.md's documented command is `pytest -n auto --reruns 2`, so a
        flake re-runs this test in the same process; a fixed name would then
        raise "already registered" and replace the real failure.
        """
        return f"flashinfer_test::{stem}_{uuid.uuid4().hex[:8]}"

    def test_a_traced_call_is_refused_while_registration_is_off(self, monkeypatch):
        """Silently entering the extension under torch.compile is the failure
        the guard exists to prevent."""
        tc = self._tc()
        wrapped = tc._guard_compile(lambda x: x + 1, "flashinfer::demo")

        assert wrapped(1) == 2
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(RuntimeError, match="custom ops are not enabled"):
            wrapped(1)

    def test_with_the_flag_on_an_inferable_signature_registers(self, monkeypatch):
        tc = self._tc()
        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)
        name = self._unique("inferable")

        @tc.register_custom_op(name, mutates_args=())
        def op(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        assert self._registered(name)

    def test_a_signature_torch_cannot_infer_falls_back_to_the_guard(self, monkeypatch):
        """`Optional[torch.Generator]` is the real case: every sampling op takes
        one, which is why their fake ops never register."""
        tc = self._tc()
        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)

        with pytest.warns(UserWarning, match="falling back to compile guard"):

            @tc.register_custom_op(self._unique("ungeneratable"), mutates_args=())
            def op(
                x: torch.Tensor, generator: Optional[torch.Generator]
            ) -> torch.Tensor:
                return x + 1

        assert op(torch.zeros(1), None).item() == 1
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(RuntimeError, match="custom ops are not enabled"):
            op(torch.zeros(1), None)

    def test_register_fake_op_is_inert_while_the_flag_is_off(self):
        tc = self._tc()
        sentinel = object()
        assert tc.register_fake_op("flashinfer_test::absent")(sentinel) is sentinel

    def test_register_fake_op_swallows_a_failure_with_the_flag_on(self, monkeypatch):
        """An op that never registered has no fake to attach; that must not end
        the import."""
        tc = self._tc()
        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)
        sentinel = object()

        assert tc.register_fake_op("flashinfer_test::no_such_op")(sentinel) is sentinel

    def test_the_decorator_accepts_a_function_directly(self, monkeypatch):
        """Both call shapes are used in-tree: bare and with parentheses.

        The flag has to be on: with it off both calls take the _guard_compile
        passthrough and the torch.library arm this covers is never entered.
        """
        tc = self._tc()
        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)

        # Annotated: an unannotated parameter fails infer_schema, which
        # register_custom_op catches, so the op would silently not register.
        def f(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        name = self._unique("direct")
        assert tc.register_custom_op(name, f, mutates_args=())
        assert self._registered(name), "the op did not register"
        assert tc.register_fake_op(name, f) is f

        # register_fake_op returns f on success, on a suppressed failure, and
        # if `return decorator(fn)` regressed to `return fn`, so the identity
        # above proves nothing on its own. opcheck drives the registered fake.
        namespace, _, stem = name.partition("::")
        op = getattr(getattr(torch.ops, namespace), stem)
        torch.library.opcheck(
            op,
            (torch.randn(4, device="cuda"),),
            test_utils=("test_faketensor",),
        )


class TestExtensionCustomOp:
    """``extension_custom_op`` on a stand-in for a void extension op, on CPU."""

    @staticmethod
    def _void_op():
        ns = f"flashinfer_test_{uuid.uuid4().hex[:8]}"
        lib = torch.library.Library(ns, "DEF")
        lib.define("write(Tensor x, Tensor out, Tensor? lse) -> ()")

        def write(x, out, lse):
            out.copy_(x + 1)
            if lse is not None:
                lse.fill_(7)

        lib.impl("write", write, "CompositeExplicitAutograd")
        return lib, getattr(torch.ops, ns).write.default

    @pytest.fixture
    def flag_on(self, monkeypatch):
        from flashinfer.rocm import torch_compile as tc

        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)

    def test_flag_off_returns_the_op_unchanged(self, monkeypatch):
        """The default eager path must not pay custom-op dispatch."""
        from flashinfer.rocm import torch_compile as tc
        from flashinfer.rocm.torch_compile import extension_custom_op

        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", False)

        lib, c_op = self._void_op()
        assert extension_custom_op(c_op, ("out",)) is c_op

    def test_written_arguments_are_annotated_by_name(self, flag_on):
        from flashinfer.rocm.torch_compile import extension_custom_op

        lib, c_op = self._void_op()
        op = extension_custom_op(c_op, ("out", "lse"))
        schema = str(op._opoverload._schema)
        assert "Tensor(a!) out" in schema and "Tensor(b!)? lse" in schema, schema
        assert "Tensor x" in schema

        x, out, lse = torch.zeros(3), torch.empty(3), torch.empty(1)
        op(x, out, lse)
        assert out.tolist() == [1.0, 1.0, 1.0] and lse.item() == 7

        def f(x):
            o = torch.empty_like(x)
            op(x, o, None)
            return o * 2

        torch._dynamo.reset()
        assert torch.compile(f, fullgraph=True)(x).tolist() == [2.0, 2.0, 2.0]
        assert extension_custom_op(c_op, ("out", "lse")) is op, "registered twice"

    def test_an_unknown_argument_name_raises(self, flag_on):
        """A renamed C++ argument must fail loudly, not leave a write undeclared."""
        from flashinfer.rocm.torch_compile import extension_custom_op

        lib, c_op = self._void_op()
        with pytest.raises(ValueError, match="no Tensor argument named 'o'"):
            extension_custom_op(c_op, ("o",))

    def test_a_second_caller_cannot_declare_different_writes(self, flag_on):
        """Whichever caller registered first would otherwise win silently."""
        from flashinfer.rocm.torch_compile import extension_custom_op

        lib, c_op = self._void_op()
        extension_custom_op(c_op, ("out", "lse"))
        with pytest.raises(ValueError, match="already declares writes"):
            extension_custom_op(c_op, ("out",))


class TestUntraced:
    """``untraced`` on a CPU stand-in for the decode shadow-plan build."""

    @pytest.fixture(autouse=True)
    def flag_on(self, monkeypatch):
        from flashinfer.rocm import torch_compile as tc

        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", True)
        yield
        torch._dynamo.reset()

    @staticmethod
    def _stand_in(calls):
        def build(x):
            calls.append(torch.compiler.is_compiling())
            return x.sum().item()  # host sync: untraceable under fullgraph

        return build

    def test_flag_off_returns_the_function_unchanged(self, monkeypatch):
        from flashinfer.rocm import torch_compile as tc

        monkeypatch.setattr(tc, "_USE_TORCH_CUSTOM_OPS", False)
        fn = self._stand_in([])
        assert tc.untraced(fn, "why") is fn

    def test_a_compiled_caller_runs_it_eagerly(self):
        from flashinfer.rocm.torch_compile import untraced

        calls = []
        build = untraced(self._stand_in(calls), "why")
        x = torch.ones(4)
        assert build(x) == 4.0 and calls == [False]
        f = torch.compile(lambda x: x * build(x), backend="eager")
        assert f(x).tolist() == [4.0] * 4 and calls[1:] == [False]

    @_needs_disable_reason
    def test_after_prepare_fullgraph_names_the_reason(self):
        import torch._dynamo  # prepare() is a no-op until Dynamo is loaded

        from flashinfer.rocm.torch_compile import untraced

        build = untraced(self._stand_in([]), "UNTRACED-REASON")
        build.prepare()  # no trace has built the boundary yet
        with pytest.raises(Exception, match="UNTRACED-REASON"):
            torch.compile(lambda x: x * build(x), backend="eager", fullgraph=True)(
                torch.ones(4)
            )
