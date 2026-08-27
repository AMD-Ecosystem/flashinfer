# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that importing flashinfer does not require a live GPU.

``flashinfer/jit/env.py`` resolves the JIT workspace directory at import time,
and on HIP it did so through ``torch.cuda.current_device()`` -- so any process
without a visible device died on ``import flashinfer``. Subprocesses spawned by
the test suite are the common victims, but a GPU-free build host is the same
case.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not hasattr(torch.version, "hip") or torch.version.hip is None,
    reason="HIP not available",
)

# No host has this many cards, so it selects nothing on any ROCm version.
_NO_SUCH_DEVICE = "999999"


def test_import_succeeds_with_no_visible_device():
    """A device index that names no card hides them all; the import must survive.

    The child checks its own device count before importing, so the test cannot
    pass because the sentinel failed to hide anything. Those checks are raises
    rather than asserts: the parent's env is inherited, and PYTHONOPTIMIZE
    would strip an assert and leave the test passing vacuously.
    """
    snippet = textwrap.dedent(
        """\
        import torch

        if torch.cuda.device_count() != 0:
            raise SystemExit("sentinel failed: child still sees a GPU")

        import flashinfer
        from flashinfer.jit import env as jit_env

        name = jit_env.FLASHINFER_WORKSPACE_DIR.name
        if not name.startswith("gfx"):
            raise SystemExit(f"workspace dir dropped the arch: {name}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        # Both layers: ROCR filters before HIP does, and an out-of-range index is
        # unambiguous where an empty string is read as "unset" on some stacks.
        env={
            **os.environ,
            "ROCR_VISIBLE_DEVICES": _NO_SUCH_DEVICE,
            "HIP_VISIBLE_DEVICES": _NO_SUCH_DEVICE,
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"GPU-free import failed:\n{result.stderr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_workspace_dir_still_names_the_device_arch():
    """With a device present the path is unchanged -- existing caches keep working.

    Read through current_device(), which is what env.py resolved the name from;
    device 0 is a different card once a worker has been pinned.
    """
    from flashinfer.arch_caps import normalize_arch
    from flashinfer.jit import env as jit_env

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected = normalize_arch(props.gcnArchName)
    assert jit_env.FLASHINFER_WORKSPACE_DIR.name == expected
