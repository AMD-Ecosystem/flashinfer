# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the ROCm import gate in flashinfer/comm/__init__.py.
#
# On ROCm the comm package must:
#   1. Import successfully and expose the backend-agnostic re-exports
#      (Mapping, pack_strided_memory).
#   2. Raise ImportError — not AssertionError — for every CUDA-only submodule,
#      via both `import X` and `from X import Y`. vLLM's
#      flashinfer_all_reduce.py guards on ImportError; anything else escapes.

import importlib
import subprocess
import sys
import textwrap

import pytest

from flashinfer.rocm import CUDA_ONLY_MODULES
from flashinfer.rocm.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not IS_HIP, reason="comm import gate is only installed on ROCm"
)


# Read from the registry rather than restated, so a module added there cannot
# silently go uncovered here.
_CUDA_ONLY_SUBMODULES = sorted(CUDA_ONLY_MODULES)


def test_comm_package_imports_and_exposes_backend_agnostic_symbols():
    import flashinfer.comm as comm

    assert hasattr(comm, "Mapping")
    assert hasattr(comm, "pack_strided_memory")


@pytest.mark.parametrize("modname", _CUDA_ONLY_SUBMODULES)
def test_cuda_only_submodule_import_raises_importerror(modname):
    with pytest.raises(ImportError):
        importlib.import_module(modname)


@pytest.mark.parametrize("modname", _CUDA_ONLY_SUBMODULES)
def test_cuda_only_submodule_from_import_raises_importerror(modname):
    # `from X import Y` exercises a different code path than plain
    # `import X` — the loader must raise ImportError for both.
    with pytest.raises(ImportError):
        exec(f"from {modname} import _anything", {})


def test_the_gate_is_up_after_importing_flashinfer_alone():
    """A subprocess, because every other gate test imports flashinfer.comm first.

    That import is what used to install the gate, so in-process the gate looked
    present no matter where the call site was -- and under `-n auto` the tests
    need not even share a worker.
    """
    snippet = textwrap.dedent(
        """\
        import flashinfer, importlib, sys

        if "flashinfer.comm" in sys.modules:
            raise SystemExit("flashinfer.comm imported; the test proves nothing")
        try:
            importlib.import_module("flashinfer.quantization.fp4_quantization")
        except ImportError as e:
            if "CUDA-only" not in str(e):
                raise SystemExit(f"wrong ImportError, gate is not up: {e}")
        else:
            raise SystemExit("import succeeded; the gate is not installed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
