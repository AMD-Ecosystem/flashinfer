# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Routing to AITER's asm prefill arm, and the numerics once it is reached.

The arm was unreachable before this test existed: the variant .so the loader
opens is built -DFAV2_ON=1 with no -DFAV3_ON, so `args.use_asm_v3` selected code
that was not in the binary. Nothing here is a regression test for that -- it is
first coverage.

`return_lse` is the case worth watching. The asm kernel writes LSE from a
different code path than CK Tile, and single_prefill_aiter.cu divides whatever
comes back by log(2) unconditionally, so a differing base would corrupt every
LSE silently rather than failing loudly.
"""

import logging
import pathlib
import re
import subprocess
import sys

import pytest
import torch
from attention_reference import naive_attention

import flashinfer
from flashinfer.jit.core import logger
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.arch_caps import (
    _AITER_ASM_PREFILL_MIN_QO_LEN,
    _device_arch,
    aiter_asm_prefill_min_qo_len,
)
from flashinfer.rocm.prefill import _aiter_ops_importable

logger.setLevel(logging.ERROR)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CUH = _REPO_ROOT / "include/flashinfer/rocm/attention/aiter/single_prefill.cuh"

HEAD_DIM = 128
# 8/2 keeps GQA=4 while holding the fp32 reference at 4096 to ~1.6 GB;
# 32 heads would be ~6.3 GB live on a card several xdist workers share.
NUM_QO_HEADS = 8
NUM_KV_HEADS = 2


def _skip_unless_aiter(device: torch.device) -> None:
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    from flashinfer.rocm.arch_caps import capability_available, capability_reason

    if not capability_available(device, "single_prefill", "aiter"):
        pytest.skip(capability_reason(device, "single_prefill", "aiter"))


# --------------------------------------------------------------------------
# Policy. No GPU needed.
# --------------------------------------------------------------------------


def test_cuh_mirrors_arch_caps():
    """The C++ gate carries its own copy of the table; drift would be silent.

    Binds each arch to the macro its own branch returns, so swapping the two
    branches fails here rather than silently reversing the routing policy.
    """
    text = _CUH.read_text()
    for arch, value in _AITER_ASM_PREFILL_MIN_QO_LEN.items():
        macro = f"FLASHINFER_AITER_ASM_MIN_QO_LEN_{arch.upper()}"
        expected = 0 if value is None else value
        assert re.search(rf"^#define {macro} {expected}\b", text, re.M), (
            f"{macro} in single_prefill.cuh does not equal {expected}; "
            "arch_caps.py and the C++ gate have drifted."
        )
        assert re.search(
            rf'strcmp\(arch, "{re.escape(arch)}"\) == 0\)\s*return {macro};', text
        ), f"the {arch} branch does not return {macro}"


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("gfx942", None),
        ("gfx950", 2048),
        ("gfx950:sramecc+:xnack-", 2048),
        ("gfx90a", None),
        ("unknown", None),
    ],
)
def test_accessor(arch, expected):
    assert aiter_asm_prefill_min_qo_len(arch) == expected


def test_gfx942_never_routes_to_asm():
    """Not an oversight: asm wins and loses non-monotonically on CDNA3, so no
    threshold holds there. Pinning it stops a later edit reintroducing one
    without a fresh sweep."""
    assert _AITER_ASM_PREFILL_MIN_QO_LEN["gfx942"] is None


# --------------------------------------------------------------------------
# Numerics. Needs a GPU.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qo_len,kv_len", [(512, 512), (2048, 2048), (4096, 4096), (512, 4096)]
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_asm_gate_numerics(qo_len, kv_len, causal, return_lse):
    """Both sides of the threshold against an fp32 reference.

    Shapes are chosen to straddle it: 512 is always CK Tile, 2048 and 4096 take
    asm on gfx950, and (512, 4096) checks the gate reads qo_len rather than
    kv_len -- that shape measured 0.73x, so routing it to asm would be a
    regression.
    """
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)
    if causal and qo_len > kv_len:
        pytest.skip("causal requires qo_len <= kv_len")

    torch.manual_seed(7)
    q = torch.randn(qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # .float() matters: naive_attention does not upcast, so passing bf16 would
    # compute the reference in the dtype under test and hide the kernel's error.
    ref_o, ref_lse = naive_attention(
        q.float(), k.float(), v.float(), causal=causal, return_lse=return_lse
    )
    res = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="aiter", return_lse=return_lse
    )
    out, lse = res if return_lse else (res, None)

    torch.testing.assert_close(out.float(), ref_o.float(), rtol=2e-2, atol=2e-2)
    if return_lse:
        assert torch.isfinite(lse).all(), "asm arm returned a non-finite LSE"
        torch.testing.assert_close(lse.float(), ref_lse.float(), rtol=5e-2, atol=5e-2)


_PROBE = """
import os, torch, flashinfer

d = torch.device("cuda:0")
torch.manual_seed(7)
qo = int(os.environ["PROBE_QO_LEN"])
causal = os.environ["PROBE_CAUSAL"] == "1"
return_lse = os.environ["PROBE_LSE"] == "1"
q = torch.randn(qo, 8, 128, dtype=torch.bfloat16, device=d)
k = torch.randn(qo, 2, 128, dtype=torch.bfloat16, device=d)
v = torch.randn(qo, 2, 128, dtype=torch.bfloat16, device=d)


def call():
    return flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="aiter", return_lse=return_lse
    )


if os.environ.get("PROBE_CAPTURE") == "1":
    # Cold on purpose: the guard exists because AITER loads its .co on the first
    # asm call, and a module load inside capture aborts the process.
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    print("CAPTURE_OK")
else:
    res = call()
    out = res[0] if return_lse else res
    torch.save(out.float().cpu(), os.environ["PROBE_OUT"])
"""


def _run_probe(
    tmp_path, qo_len, causal=True, return_lse=False, capture=False, **env_overrides
):
    """Run one prefill in a fresh process; return (output tensor or None, stderr).

    A fresh process per case is not fastidiousness: the kill switch and the verbose
    flag are both read once into C++ statics, and a broken capture guard aborts
    rather than raising, which only a subprocess can survive.
    """
    import os

    tag = (
        f"{qo_len}-{int(causal)}{int(return_lse)}{int(capture)}{''.join(env_overrides)}"
    )
    out_path = tmp_path / f"o{tag}.pt"
    env = dict(os.environ)
    env.update(
        PROBE_QO_LEN=str(qo_len),
        PROBE_CAUSAL="1" if causal else "0",
        PROBE_LSE="1" if return_lse else "0",
        PROBE_CAPTURE="1" if capture else "0",
        PROBE_OUT=str(out_path),
        FLASHINFER_AITER_ASM_VERBOSE="1",
    )
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    assert proc.returncode == 0, f"probe failed ({tag}): {proc.stderr[-2000:]}"
    out = torch.load(out_path) if out_path.exists() else None
    return out, proc.stderr


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_asm_arm_is_actually_reached(tmp_path, causal, return_lse):
    """Which arm ran has to be observable for every trait the gate admits.

    If the asm .so is missing or AITER_ASM_DIR is unset, the C++ swallows the
    failure and CK Tile serves the call with identical numerics -- so without this,
    test_asm_gate_numerics would pass whether or not the arm was ever reached.
    """
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)
    threshold = aiter_asm_prefill_min_qo_len(_device_arch(device))
    if not threshold:
        pytest.skip(f"{_device_arch(device)} never routes to asm")

    _, above = _run_probe(tmp_path, threshold, causal=causal, return_lse=return_lse)
    _, below = _run_probe(tmp_path, 512, causal=causal, return_lse=return_lse)
    assert "aiter asm prefill: launched" in above, (
        f"qo_len={threshold} did not reach the asm arm. stderr:\n{above[-2000:]}"
    )
    assert "aiter asm prefill: launched" not in below, (
        f"qo_len=512 is below the threshold but took the asm arm. stderr:\n{below[-2000:]}"
    )


def test_graph_capture_stays_on_ck_tile(tmp_path):
    """A cold above-threshold call captured into a graph must not reach the asm arm.

    AITER loads its .co lazily on first use and turns HIP's rejection of a module
    load during capture into std::abort(), so a broken guard kills the process --
    which is why this runs in a subprocess and asserts on the exit code at all.
    """
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)
    threshold = aiter_asm_prefill_min_qo_len(_device_arch(device))
    if not threshold:
        pytest.skip(f"{_device_arch(device)} never routes to asm")

    _, err = _run_probe(tmp_path, threshold, capture=True)
    assert "CAPTURE_OK" not in err  # stdout, not stderr
    assert "aiter asm prefill: launched" not in err, (
        f"asm arm was entered during graph capture. stderr:\n{err[-2000:]}"
    )


def test_kill_switch_pins_ck_tile(tmp_path):
    """FLASHINFER_AITER_ASM_PREFILL=0 must change which kernel runs, not the answer."""
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)
    threshold = aiter_asm_prefill_min_qo_len(_device_arch(device))
    if not threshold:
        pytest.skip(f"{_device_arch(device)} never routes to asm")

    on_out, on_err = _run_probe(tmp_path, threshold, FLASHINFER_AITER_ASM_PREFILL=None)
    off_out, off_err = _run_probe(tmp_path, threshold, FLASHINFER_AITER_ASM_PREFILL="0")

    assert "aiter asm prefill: launched" in on_err
    assert "aiter asm prefill" not in off_err, "the kill switch did not disable the arm"
    # Elementwise, not a sum: ~8M near-zero-mean terms cancel, so a sum would pass
    # even with a whole block of the output wrong.
    assert (on_out - off_out).abs().max().item() < 2e-2, "asm and CK Tile disagree"
