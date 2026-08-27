# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ROCm JIT halves in :mod:`flashinfer.jit.rocm`.

No GPU and no build: these cover the directory resolution and hipcc flag
construction that every JIT-compiled kernel depends on.

What is being protected. These functions decide where compiled kernels are
cached and which flags they are built with, and nothing else asserts on them --
a wrong answer does not fail loudly, it silently caches under the wrong
directory or drops a define. ``-fno-finite-math-only`` in particular must
survive: clang's ``-ffast-math`` implies ``-ffinite-math-only``, which breaks
kernels using ``-inf`` as a sentinel (the online-softmax Map+Reduce path), and
CUDA's ``-use_fast_math`` does not have that implication. Dropping it produces
wrong attention output, not a build error.
"""

import pathlib

import pytest

from flashinfer.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not IS_HIP, reason="flashinfer.jit.rocm is only importable on HIP builds"
)


def test_aot_dir_falls_back_to_the_package_data_dir():
    """Without the jit-cache wheel, the AOT dir is <package>/data/aot."""
    from flashinfer.jit.rocm.env import get_aot_dir

    root = pathlib.Path("/nonexistent/pkg")
    assert get_aot_dir(root, lambda: False) == root / "data" / "aot"


def test_aot_dir_version_mismatch_is_reported(monkeypatch):
    """A jit-cache wheel whose version disagrees must raise, not be used."""
    import sys
    import types

    from flashinfer.jit.rocm.env import get_aot_dir

    fake = types.ModuleType("amd_flashinfer_jit_cache")
    fake.__version__ = "0.0.1-definitely-not-the-flashinfer-version"
    fake.get_jit_cache_dir = lambda: "/tmp/should-not-be-reached"
    monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", fake)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)

    with pytest.raises(RuntimeError, match="does not match"):
        get_aot_dir(pathlib.Path("/unused"), lambda: True)


def test_aot_dir_version_check_can_be_bypassed(monkeypatch):
    """FLASHINFER_DISABLE_VERSION_CHECK is the documented escape hatch."""
    import sys
    import types

    from flashinfer.jit.rocm.env import get_aot_dir

    fake = types.ModuleType("amd_flashinfer_jit_cache")
    fake.__version__ = "0.0.1-mismatched"
    fake.get_jit_cache_dir = lambda: "/cache/from/wheel"
    monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", fake)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == pathlib.Path(
        "/cache/from/wheel"
    )


def test_workspace_dir_is_keyed_by_version_and_arch():
    """<cache>/<version>/<arch> -- the arch segment keeps gfx942 and gfx950 apart.

    Sharing one directory across architectures would let a kernel built for one
    be loaded on the other.
    """
    from flashinfer._version import __version__
    from flashinfer.jit.rocm.env import get_workspace_dir

    got = get_workspace_dir(pathlib.Path("/cache"))
    assert got.parent.parent == pathlib.Path("/cache")
    assert got.parent.name == __version__
    assert got.name.startswith("gfx") or got.name == "noarch"


def test_hipcc_flags_keep_inf_sentinels_alive():
    """-ffast-math must always be paired with -fno-finite-math-only."""
    from flashinfer.jit.core import current_compilation_context
    from flashinfer.jit.rocm.core import build_flags

    _, cuda_cflags = build_flags(current_compilation_context)
    assert "-ffast-math" in cuda_cflags
    assert cuda_cflags.index("-fno-finite-math-only") > cuda_cflags.index("-ffast-math")


def test_hipcc_flags_carry_the_dtype_defines():
    """Every dtype the ROCm kernels instantiate needs its define present."""
    from flashinfer.jit.core import current_compilation_context
    from flashinfer.jit.rocm.core import build_flags

    _, cuda_cflags = build_flags(current_compilation_context)
    for define in (
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ):
        assert define in cuda_cflags


def test_hipcc_flags_include_the_requested_offload_archs(monkeypatch):
    """cflags must carry --offload-arch for each arch, or the build targets the wrong GPU."""
    from flashinfer.compilation_context_hip import CompilationContext

    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942,gfx950")
    from flashinfer.jit.rocm.core import build_flags

    cflags, _ = build_flags(CompilationContext())
    assert "--offload-arch=gfx942" in cflags
    assert "--offload-arch=gfx950" in cflags


def test_verbose_drops_ndebug(monkeypatch):
    """FLASHINFER_JIT_VERBOSE=1 must not leave -DNDEBUG in the flags."""
    from flashinfer.jit.core import current_compilation_context
    from flashinfer.jit.rocm.core import build_flags

    monkeypatch.setenv("FLASHINFER_JIT_VERBOSE", "1")
    _, verbose_flags = build_flags(current_compilation_context)
    monkeypatch.setenv("FLASHINFER_JIT_VERBOSE", "0")
    _, quiet_flags = build_flags(current_compilation_context)

    assert "-DNDEBUG" not in verbose_flags
    assert "-DNDEBUG" in quiet_flags


def test_activation_template_targets_the_forked_rocm_header():
    """The template must include the ROCm fork, not upstream's activation.cuh."""
    from flashinfer.jit.rocm.activation import activation_templ

    assert "flashinfer/rocm/attention/activation.cuh" in activation_templ
    assert "hipLaunchKernelGGL" in activation_templ
