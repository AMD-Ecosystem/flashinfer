# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An installed, supported amd-aiter must actually import on gfx942/gfx950.

If it does not, every AITER test skips and ``auto`` serves fa2, so the suite
reads green while measuring none of AITER. These tests fail instead.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

import flashinfer
from flashinfer.rocm import aiter_utils, prefill
from flashinfer.rocm.aiter_utils import is_aiter_supported


def _skip_unless_supported_aiter():
    if not torch.cuda.is_available() or not is_aiter_supported(torch.device("cuda:0")):
        pytest.skip("AITER targets gfx942/gfx950 only")
    if aiter_utils._aiter_installed_version() is None:
        pytest.skip("amd-aiter is not installed")
    if not aiter_utils._aiter_version_supported():
        pytest.skip("below AITER_MIN_VERSION; test_aiter_version_gate owns that")


def test_an_installed_aiter_imports():
    _skip_unless_supported_aiter()
    assert aiter_utils._aiter_import_error() is None
    assert aiter_utils._aiter_importable()
    assert prefill._aiter_ops_importable()


def test_explicit_aiter_decode_works_as_the_first_aiter_use():
    """A fresh process: nothing has imported aiter, which is what puts AITER's
    csrc on sys.path for the decode kernel's codegen. This process already has."""
    _skip_unless_supported_aiter()
    probe = textwrap.dedent(
        """
        import pathlib, sys, torch, flashinfer
        assert "aiter" not in sys.modules
        print(f"PROBE_FLASHINFER={pathlib.Path(flashinfer.__file__).resolve()}")
        dev = torch.device("cuda:0")
        B, H, HKV, D, P, NP = 2, 8, 2, 128, 16, 4
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(64 << 20, dtype=torch.uint8, device=dev), "NHD", backend="aiter"
        )
        w.plan(
            torch.arange(B + 1, dtype=torch.int32, device=dev) * NP,
            torch.arange(B * NP, dtype=torch.int32, device=dev),
            torch.full((B,), P, dtype=torch.int32, device=dev),
            H, HKV, D, P, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
        q = torch.randn(B, H, D, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(B * NP, 2, P, HKV, D, dtype=torch.bfloat16, device=dev)
        assert torch.isfinite(w.run(q, kv).float()).all()
        """
    )
    # Without GPU_ARCHS, which this process has exported, so the child must set it.
    env = {k: v for k, v in os.environ.items() if k != "GPU_ARCHS"}
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    # The child resolves flashinfer without pytest's rootdir insertion.
    here = pathlib.Path(flashinfer.__file__).resolve()
    assert f"PROBE_FLASHINFER={here}" in done.stdout, done.stdout[-500:]
