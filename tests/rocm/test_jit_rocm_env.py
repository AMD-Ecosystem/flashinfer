# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ROCm JIT halves in :mod:`flashinfer.jit.rocm`.

No GPU work and no kernel build. These cover where kernels are cached and which
flags they are built with -- neither fails loudly when wrong.
"""

import pathlib

import pytest
import torch

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


def _wheel_with_manifest(monkeypatch, tmp_path, manifest_text, device_arch):
    """A fake jit-cache wheel whose kernels were built for `manifest_text`.

    `device_arch` is the GPU the guard should believe it is running on, patched
    so the test does not depend on the GPU it runs on -- the guard has to work
    on both. Patching the device (not FLASHINFER_ROCM_ARCH_LIST) is the point:
    the check must key on hardware, not on a build-time variable.
    """
    import sys
    import types

    from flashinfer._version import __version__
    from flashinfer.jit.rocm.env import AOT_MANIFEST_NAME

    cache = tmp_path / "wheel_cache"
    cache.mkdir()
    if manifest_text is not None:
        (cache / AOT_MANIFEST_NAME).write_text(manifest_text)

    fake = types.ModuleType("amd_flashinfer_jit_cache")
    fake.__version__ = __version__
    fake.get_jit_cache_dir = lambda: str(cache)
    monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", fake)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_AOT_ARCH_CHECK", raising=False)
    monkeypatch.setattr(
        "flashinfer.jit.rocm.env._live_device_arch", lambda: device_arch
    )
    return cache


def test_matching_arch_uses_the_prebuilt_kernels(monkeypatch, tmp_path):
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', "gfx942"
    )

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_mismatched_arch_falls_back_to_jit_rather_than_serving_wrong_kernels(
    monkeypatch, tmp_path
):
    """A gfx942 wheel on a gfx950 box must not be used.

    JitSpec.is_aot is a bare exists() check, so returning the wheel dir here is
    what would load gfx942 code on CDNA4. The returned path must not exist, so
    is_aot goes False and every module JITs.
    """
    from flashinfer.jit.rocm.env import get_aot_dir

    _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', "gfx950"
    )

    with pytest.warns(UserWarning, match="gfx942.*this GPU is gfx950"):
        got = get_aot_dir(tmp_path / "pkg", lambda: True)

    assert not got.exists(), "the fallback must not resolve to a populated tree"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_the_build_arch_env_var_does_not_decide_which_wheel_is_usable(
    monkeypatch, tmp_path
):
    """The guard must key on hardware, not on FLASHINFER_ROCM_ARCH_LIST.

    That variable is build intent: CLAUDE.md tells developers to export
    "gfx942,gfx950", Dockerfiles set it, and an AOT build publishes it into the
    environment itself. Resolving the running architecture through it -- as
    hip_utils.resolve_target_archs does, env var before hardware -- discards a
    wheel that matches the actual GPU, and JIT-compiles everything the wheel
    exists to supply. Deliberately does not patch the device probe: the real
    GPU is the whole point of the test.
    """
    import json
    import sys
    import types

    from flashinfer._version import __version__
    from flashinfer.arch_caps import normalize_arch
    from flashinfer.jit.rocm.env import AOT_MANIFEST_NAME, get_aot_dir

    arch = normalize_arch(
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    )
    cache = tmp_path / "wheel_cache"
    cache.mkdir()
    (cache / AOT_MANIFEST_NAME).write_text(json.dumps({"rocm_arch_list": arch}))

    fake = types.ModuleType("amd_flashinfer_jit_cache")
    fake.__version__ = __version__
    fake.get_jit_cache_dir = lambda: str(cache)
    monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", fake)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_AOT_ARCH_CHECK", raising=False)
    # The value CLAUDE.md's Essential Commands table tells developers to export.
    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942,gfx950")

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_the_fallback_is_not_the_in_tree_aot_dir(monkeypatch, tmp_path):
    """`aot_hip --out-dir flashinfer/data/aot` populates the in-tree dir.

    Falling back to it on a mismatch would swap one unchecked kernel tree for
    another, which is the bug this guard exists to prevent.
    """
    from flashinfer.jit.rocm.env import get_aot_dir

    _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', "gfx950"
    )
    root = tmp_path / "pkg"
    (root / "data" / "aot" / "norm").mkdir(parents=True)
    (root / "data" / "aot" / "norm" / "norm.so").write_bytes(b"\x7fELF gfx942")

    with pytest.warns(UserWarning):
        got = get_aot_dir(root, lambda: True)

    assert got != root / "data" / "aot"
    assert not (got / "norm" / "norm.so").exists()


def test_a_fat_wheel_covers_this_gpu(monkeypatch, tmp_path):
    """Superset is a match -- otherwise a gfx942+gfx950 wheel serves nobody."""
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942,gfx950"}', "gfx950"
    )

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_a_semicolon_separated_manifest_is_still_understood(monkeypatch, tmp_path):
    """`;` is documented for this same arch list in jit/aiter_source.py."""
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942;gfx950"}', "gfx950"
    )

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_an_invisible_gpu_leaves_the_kernels_alone(monkeypatch, tmp_path):
    """Import must not require a device, and "unknown" is not "mismatched"."""
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', None
    )

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_a_wheel_without_a_manifest_keeps_working(monkeypatch, tmp_path):
    """Wheels built before this check must not break on upgrade."""
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(monkeypatch, tmp_path, None, "gfx950")

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


@pytest.mark.parametrize(
    "manifest_text",
    [
        "not json{",  # JSONDecodeError -> ValueError
        '{"rocm_arch_list": null}',  # AttributeError on .split
        '{"rocm_arch_list": ["gfx942"]}',  # AttributeError on a list
        '["gfx942"]',  # TypeError: list is not subscriptable by str
        '{"other_key": "gfx942"}',  # KeyError
    ],
)
def test_a_malformed_manifest_warns_but_does_not_break_the_install(
    monkeypatch, tmp_path, manifest_text
):
    """get_aot_dir runs in jit/env.py's module body -- a raise here is an
    unimportable package, so every shape of bad manifest must degrade."""
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(monkeypatch, tmp_path, manifest_text, "gfx950")

    with pytest.warns(UserWarning, match="Could not read"):
        assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_the_arch_check_has_its_own_bypass(monkeypatch, tmp_path):
    from flashinfer.jit.rocm.env import get_aot_dir

    cache = _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', "gfx950"
    )
    monkeypatch.setenv("FLASHINFER_DISABLE_AOT_ARCH_CHECK", "1")

    assert get_aot_dir(pathlib.Path("/unused"), lambda: True) == cache


def test_the_version_bypass_does_not_also_disable_the_arch_check(monkeypatch, tmp_path):
    """Two independent failures must not share one off switch.

    FLASHINFER_DISABLE_VERSION_CHECK is documented for version skew, and three
    build backends set it process-wide; if it also silenced the ISA check, any
    of them would forfeit wrong-ISA protection as a side effect.
    """
    from flashinfer.jit.rocm.env import get_aot_dir

    _wheel_with_manifest(
        monkeypatch, tmp_path, '{"rocm_arch_list": "gfx942"}', "gfx950"
    )
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    with pytest.warns(UserWarning, match="this GPU is gfx950"):
        got = get_aot_dir(tmp_path / "pkg", lambda: True)

    assert not got.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_workspace_dir_is_keyed_by_version_and_arch():
    """<cache>/<version>/<arch> -- the arch segment keeps gfx942 and gfx950 apart.

    Sharing one directory across architectures would let a kernel built for one
    be loaded on the other.
    """
    from flashinfer._version import __version__
    from flashinfer.jit.rocm.env import get_workspace_dir

    from flashinfer.arch_caps import normalize_arch

    got = get_workspace_dir(pathlib.Path("/cache"))
    assert got.parent.parent == pathlib.Path("/cache")
    assert got.parent.name == __version__
    # The exact arch, not "gfx-something": a regression that collapses detection
    # to the noarch fallback would satisfy a looser assertion while making
    # gfx942 and gfx950 share one cache directory.
    expected = normalize_arch(
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    )
    assert got.name == expected


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


def test_hipcc_flags_pass_through_the_contexts_arch_flags():
    """build_flags must forward get_hipcc_flags_list() verbatim into cflags.

    A stub context, not a real one: constructing CompilationContext validates the
    request against the installed torch, so an arch-specific torch build would
    fail this for a reason unrelated to build_flags.
    """
    from flashinfer.jit.rocm.core import build_flags

    class _StubContext:
        def get_hipcc_flags_list(self):
            return ["--offload-arch=gfx942", "--offload-arch=gfx950"]

    cflags, _ = build_flags(_StubContext())
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


def test_workspace_dir_without_a_device_uses_the_requested_archs(monkeypatch):
    """With no GPU, the arch comes from FLASHINFER_ROCM_ARCH_LIST (#316).

    Importing flashinfer must not need a device, but the arch has to stay in the
    path: build.ninja is written only when absent, so one shared directory would
    serve two ISAs the same objects.
    """
    from flashinfer._version import __version__
    from flashinfer.jit.rocm import env as rocm_env

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950,gfx942")

    got = rocm_env.get_workspace_dir(pathlib.Path("/cache"))
    # Sorted and deduplicated, so the directory does not depend on list order.
    assert got == pathlib.Path("/cache") / __version__ / "gfx942_gfx950"
