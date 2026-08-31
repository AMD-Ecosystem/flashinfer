# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Conformance for the ROCm dispatch registry in ``flashinfer.rocm``.

The CUDA-only import gate is covered by ``test_comm_import_gate.py``; this
file covers the shadow table and the gate's install/widen behaviour.
"""

import importlib
import sys

import pytest

from flashinfer.rocm import (
    CUDA_ONLY_MODULES,
    SHADOW_MODULES,
    gate_cuda_only_modules,
)
from flashinfer.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not IS_HIP, reason="ROCm dispatch registry is only installed on HIP builds"
)


@pytest.fixture
def restore_gate():
    """Narrow the gate back to the real module set after a test widens it.

    Must not remove the finder: flashinfer.comm is cached in sys.modules by
    then, so nothing would reinstall it and every later gate test would pass a
    CUDA-only import straight through.
    """
    yield
    for finder in sys.meta_path:
        if getattr(finder, "_is_flashinfer_cuda_only_finder", False):
            finder.names = frozenset(CUDA_ONLY_MODULES)


@pytest.mark.parametrize("upstream,rocm", sorted(SHADOW_MODULES.items()))
def test_shadowed_module_resolves_to_rocm_twin(upstream, rocm):
    """Both lookup paths must reach the twin: sys.modules and the attribute."""
    import flashinfer

    attr = upstream.rsplit(".", 1)[1]
    assert importlib.import_module(upstream).__name__ == rocm
    assert sys.modules[upstream].__name__ == rocm
    # `flashinfer.mla.X` goes through the parent attribute, not sys.modules.
    assert getattr(flashinfer, attr) is sys.modules[upstream]


def test_gate_installs_on_demand_and_widens_in_place(restore_gate):
    import flashinfer  # noqa: F401

    def finders():
        return [
            f
            for f in sys.meta_path
            if getattr(f, "_is_flashinfer_cuda_only_finder", False)
        ]

    # Not asserted as pre-installed: nothing on the HIP path imports
    # flashinfer.comm, so the gate may legitimately not be up yet.
    gate_cuda_only_modules({"flashinfer.comm._not_a_real_module"})
    assert len(finders()) == 1
    with pytest.raises(ImportError, match="CUDA-only"):
        importlib.import_module("flashinfer.comm._not_a_real_module")

    # A second call must widen the existing finder, not add another.
    gate_cuda_only_modules({"flashinfer.comm._also_not_real"})
    assert len(finders()) == 1
    with pytest.raises(ImportError, match="CUDA-only"):
        importlib.import_module("flashinfer.comm._also_not_real")


def test_gate_tolerates_a_foreign_marker(restore_gate):
    """A marker-bearing object we cannot widen must not break gating."""
    import flashinfer  # noqa: F401

    class Impostor:
        _is_flashinfer_cuda_only_finder = True

        def find_spec(self, *args, **kwargs):
            return None

    sys.meta_path.insert(0, Impostor())
    try:
        gate_cuda_only_modules({"flashinfer.comm._foreign_marker_probe"})
        with pytest.raises(ImportError, match="CUDA-only"):
            importlib.import_module("flashinfer.comm._foreign_marker_probe")
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, Impostor)]
