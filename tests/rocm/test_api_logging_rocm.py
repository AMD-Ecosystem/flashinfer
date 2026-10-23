# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FLASHINFER_LOGLEVEL reaches the ROCm attention and MoE entry points.

The level is read at import, so each case runs in its own interpreter.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tests.test_helpers.test_helpers import requires_aiter

_ENTRY_POINTS = [
    "single_prefill_with_kv_cache",
    "single_decode_with_kv_cache",
    "BatchPrefillWithPagedKVCacheWrapper.__init__",
    "BatchPrefillWithPagedKVCacheWrapper.plan",
    "BatchPrefillWithPagedKVCacheWrapper.run",
    "BatchPrefillWithRaggedKVCacheWrapper.__init__",
    "BatchPrefillWithRaggedKVCacheWrapper.plan",
    "BatchPrefillWithRaggedKVCacheWrapper.run",
    "BatchDecodeWithPagedKVCacheWrapper.__init__",
    "BatchDecodeWithPagedKVCacheWrapper.plan",
    "BatchDecodeWithPagedKVCacheWrapper.run",
    "BatchMLAPagedAttentionWrapper.__init__",
    "BatchMLAPagedAttentionWrapper.plan",
    "BatchMLAPagedAttentionWrapper.run",
    "aiter_fused_moe",
]

_CALLS = """
import torch, flashinfer
dev, dt = torch.device("cuda:0"), torch.float16
i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
H, D, P = 8, 128, 16
q, k, v = (torch.randn(16, H, D, dtype=dt, device=dev) for _ in range(3))
flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")
flashinfer.single_decode_with_kv_cache(q[0], k, v)
kv = torch.randn(1, 2, P, H, D, dtype=dt, device=dev)
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
w.plan(i32(0, 16), i32(0, 1), i32(0), i32(P), H, H, D, P, causal=True)
w.run(q, kv)
w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
w.plan(i32(0, 16), i32(0, 16), H, H, D, causal=True)
w.run(q, k, v)
w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
w.plan(i32(0, 1), i32(0), i32(P), H, H, D, P, q_data_type=dt)
w.run(q[:1], kv)
"""


def _run(code, level, **env):
    env = {
        **os.environ,
        "FLASHINFER_LOGLEVEL": str(level),
        "FLASHINFER_LOGDEST": "stderr",
        **env,
    }
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stderr


def test_level_zero_leaves_entry_points_unwrapped():
    """Level 0 must cost nothing: the decorator hands back the function itself."""
    code = f"""
    import flashinfer
    wrapped = []
    for path in {_ENTRY_POINTS!r}:
        obj = flashinfer
        for part in path.split("."):
            obj = getattr(obj, part)
        if hasattr(obj, "__wrapped__"):
            wrapped.append(path)
    assert not wrapped, wrapped
    """
    _run(code, 0)


@pytest.mark.parametrize("level", [1, 3, 5, 10])
def test_attention_calls_are_logged(level, tmp_path):
    env = {}
    if level >= 10:
        # The include filter also filters the log at level >= 3; it keeps
        # __init__ from dumping each 128 MB workspace.
        env = dict(FLASHINFER_DUMP_DIR=str(tmp_path), FLASHINFER_DUMP_INCLUDE="*.run")
    log = _run(_CALLS, level, **env)
    for name in _ENTRY_POINTS:
        if name.startswith(("BatchMLA", "aiter_fused_moe")):
            continue
        if level >= 10 and not name.endswith(".run"):
            continue
        assert f"FlashInfer API Call: {name}" in log, name
    if level >= 3:
        assert "stride=" in log
    if level >= 10:
        assert any(tmp_path.rglob("inputs.pt"))


@requires_aiter
def test_mla_and_moe_calls_are_logged():
    code = """
    import torch, flashinfer
    dev, dt = torch.device("cuda:0"), torch.bfloat16
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchMLAPagedAttentionWrapper(ws)
    w.plan(i32(0, 1), i32(0, 64), torch.arange(64, dtype=torch.int32, device=dev),
           i32(64), 16, 512, 64, 1, False, 1 / 24.0, dt, dt)
    w.run(torch.randn(1, 16, 512, dtype=dt, device=dev), torch.randn(1, 16, 64, dtype=dt, device=dev),
          torch.randn(64, 1, 512, dtype=dt, device=dev), torch.randn(64, 1, 64, dtype=dt, device=dev))
    M, E, K, I = 4, 8, 512, 256
    w1 = flashinfer.shuffle_moe_weight(torch.randn(E, 2 * I, K, dtype=dt, device=dev) / 10)
    w2 = flashinfer.shuffle_moe_weight(torch.randn(E, K, I, dtype=dt, device=dev) / 10)
    ids = torch.stack([torch.randperm(E, device=dev)[:2] for _ in range(M)]).to(torch.int32)
    flashinfer.aiter_fused_moe(torch.randn(M, K, dtype=dt, device=dev), w1, w2, ids,
                               torch.softmax(torch.randn(M, 2, device=dev), -1))
    """
    log = _run(code, 1)
    for name in _ENTRY_POINTS:
        if name.startswith(("BatchMLA", "aiter_fused_moe")):
            assert f"FlashInfer API Call: {name}" in log, name


def test_logging_inside_torch_compile():
    """The logging wrapper is untraceable, so a logged call breaks the graph
    but must still compile and match eager."""
    code = """
    import torch, flashinfer
    dev, dt = torch.device("cuda:0"), torch.float16
    i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(i32(0, 1), i32(0), i32(16), 8, 8, 128, 16, q_data_type=dt)
    q = torch.randn(1, 8, 128, dtype=dt, device=dev)
    kv = torch.randn(1, 2, 16, 8, 128, dtype=dt, device=dev)
    torch.testing.assert_close(torch.compile(w.run)(q, kv), w.run(q, kv))
    """
    _run(code, 1, FLASHINFER_USE_TORCH_CUSTOM_OPS="1")
