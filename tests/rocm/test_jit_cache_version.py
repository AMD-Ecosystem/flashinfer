# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The amd-flashinfer-jit-cache wheel names its architecture in its version.

Nothing else in a wheel filename does: the tag is ``cp310-abi3-linux_x86_64``
for every architecture, so two arch builds otherwise collide on an index.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "amd-flashinfer-jit-cache" / "build_backend.py"
# Spelled out so the fixture can clear it before the module body runs.
_PRETEND_VERSION_NAME = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_FLASHINFER_JIT_CACHE"

pytest.importorskip("wheel", reason="build_backend imports wheel.bdist_wheel")


@pytest.fixture
def backend(monkeypatch):
    """The build backend, imported without running a build.

    Loaded by path rather than imported: it lives outside the flashinfer
    package and is only ever on sys.path as a PEP 517 ``backend-path``. Its
    module body sets FLASHINFER_DISABLE_VERSION_CHECK and prepends the repo
    root to sys.path -- neither may outlive the test, or the rest of this
    xdist worker runs with the real jit-cache skew check switched off.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Restored by hand: monkeypatch.delenv records no undo for a name that was
    # absent, so a value written afterwards outlives the test.
    touched = ("FLASHINFER_DISABLE_VERSION_CHECK", _PRETEND_VERSION_NAME)
    saved = {name: os.environ.pop(name, None) for name in touched}

    try:
        spec = importlib.util.spec_from_file_location("_jit_cache_backend", _BACKEND)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._PRETEND_VERSION == _PRETEND_VERSION_NAME
        yield module
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.parametrize(
    "base,archs,expected",
    [
        ("0.6.18+amd.1", "gfx942", "0.6.18+amd.1.gfx942"),
        ("0.6.18+amd.1", "gfx950", "0.6.18+amd.1.gfx950"),
        ("0.6.18+amd.1.dev7", "gfx942", "0.6.18+amd.1.dev7.gfx942"),
        # A comma is not a legal local-segment separator.
        ("0.6.18+amd.1", "gfx942,gfx950", "0.6.18+amd.1.gfx942.gfx950"),
        # A bare upstream tag has no local segment to extend.
        ("0.6.18", "gfx942", "0.6.18+gfx942"),
    ],
)
def test_version_carries_the_target_architecture(
    backend, monkeypatch, base, archs, expected
):
    monkeypatch.setattr(backend, "_scm_version", lambda: base)
    monkeypatch.setattr(backend, "_target_arch_list", lambda: archs)
    assert backend._pin_arch_tagged_version() == (expected, archs)
    # setuptools_scm reads it from the environment, not from the return value.
    assert os.environ[backend._PRETEND_VERSION] == expected


def test_the_two_architectures_do_not_collide(backend, monkeypatch):
    """The whole point: same base version, different wheel filename."""
    monkeypatch.setattr(backend, "_scm_version", lambda: "0.6.18+amd.1")
    versions = set()
    for arch in ("gfx942", "gfx950"):
        monkeypatch.setattr(backend, "_target_arch_list", lambda arch=arch: arch)
        versions.add(backend._pin_arch_tagged_version()[0])
    assert len(versions) == 2


def test_a_preset_version_does_not_survive(backend, monkeypatch):
    """FLASHINFER_ROCM_ARCH_LIST is the only control; a stale pretend loses.

    Honouring one would let the label disagree with the kernels, which is the
    failure this whole module exists to prevent.
    """
    monkeypatch.setenv(backend._PRETEND_VERSION, "9.9.9+amd.9")
    monkeypatch.setattr(backend, "_scm_version", lambda: "0.6.18+amd.1")
    monkeypatch.setattr(backend, "_target_arch_list", lambda: "gfx950")
    assert backend._pin_arch_tagged_version() == ("0.6.18+amd.1.gfx950", "gfx950")


def _write_manifest(tmp_path: pathlib.Path, backend, archs: str) -> None:
    """Point the backend at a throwaway jit_cache tree holding just a manifest."""
    from flashinfer.jit.rocm.env import AOT_MANIFEST_NAME

    cache = tmp_path / "jit_cache"
    cache.mkdir(parents=True)
    (cache / AOT_MANIFEST_NAME).write_text(json.dumps({"rocm_arch_list": archs}))
    backend._JIT_CACHE_DIR = cache


@pytest.mark.parametrize(
    "archs,built",
    [
        ("gfx942", "gfx942"),
        ("gfx942,gfx950", "gfx942,gfx950"),
        # Order is a preference, not a difference in what was compiled.
        ("gfx942,gfx950", "gfx950,gfx942"),
    ],
)
def test_matching_kernels_pass_the_check(backend, tmp_path, archs, built):
    _write_manifest(tmp_path, backend, built)
    backend._check_kernels_match(archs)


@pytest.mark.parametrize("built", ["gfx942", "gfx950"])
def test_a_dropped_architecture_fails_the_build(backend, tmp_path, built):
    """Either half of a two-arch request can be the one that goes missing.

    Shipping the wider name would put an architecture in the version of a wheel
    with none of its kernels, which the consumer-side guard cannot detect: the
    manifest it reads would name only what is there, and be believed.
    """
    _write_manifest(tmp_path, backend, built)
    with pytest.raises(RuntimeError, match="the version names"):
        backend._check_kernels_match("gfx942,gfx950")


@pytest.mark.parametrize("text", [None, "{}", "not json at all"])
def test_an_unusable_manifest_is_reported_not_ignored(backend, tmp_path, text):
    """copy_built_kernels skips the manifest entirely when the arch list is empty.

    A bare FileNotFoundError or KeyError out of the guard would read as a build
    crash rather than the mismatch it exists to name.
    """
    from flashinfer.jit.rocm.env import AOT_MANIFEST_NAME

    _write_manifest(tmp_path, backend, "gfx942")
    manifest = backend._JIT_CACHE_DIR / AOT_MANIFEST_NAME
    if text is None:
        manifest.unlink()
    else:
        manifest.write_text(text)

    with pytest.raises(RuntimeError, match="cannot confirm"):
        backend._check_kernels_match("gfx942")
