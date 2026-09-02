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

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "amd-flashinfer-jit-cache" / "build_backend.py"

pytest.importorskip("wheel", reason="build_backend imports wheel.bdist_wheel")


@pytest.fixture
def backend(monkeypatch):
    """The build backend, imported without running a build.

    Loaded by path rather than imported: it lives outside the flashinfer
    package and is only ever on sys.path as a PEP 517 ``backend-path``.
    """
    spec = importlib.util.spec_from_file_location("_jit_cache_backend", _BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv(module._PRETEND_VERSION, raising=False)
    return module


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
    assert backend._pin_arch_tagged_version() == expected
    # setuptools_scm reads it from the environment, not from the return value.
    assert os.environ[backend._PRETEND_VERSION] == expected


def test_the_two_architectures_do_not_collide(backend, monkeypatch):
    """The whole point: same base version, different wheel filename."""
    monkeypatch.setattr(backend, "_scm_version", lambda: "0.6.18+amd.1")
    versions = set()
    for arch in ("gfx942", "gfx950"):
        monkeypatch.setattr(backend, "_target_arch_list", lambda arch=arch: arch)
        monkeypatch.delenv(backend._PRETEND_VERSION, raising=False)
        versions.add(backend._pin_arch_tagged_version())
    assert len(versions) == 2


def test_an_explicit_pretend_version_wins(backend, monkeypatch):
    """CI must be able to name the version without a working GPU resolver."""

    def boom():
        raise AssertionError("resolver must not run when the version is pinned")

    monkeypatch.setattr(backend, "_scm_version", boom)
    monkeypatch.setattr(backend, "_target_arch_list", boom)
    monkeypatch.setenv(backend._PRETEND_VERSION, "9.9.9+amd.9.gfx942")
    assert backend._pin_arch_tagged_version() == "9.9.9+amd.9.gfx942"


def _write_manifest(tmp_path: pathlib.Path, backend, archs: str) -> None:
    from flashinfer.jit.rocm.env import AOT_MANIFEST_NAME

    cache = tmp_path / "amd_flashinfer_jit_cache" / "jit_cache"
    cache.mkdir(parents=True)
    (cache / AOT_MANIFEST_NAME).write_text(json.dumps({"rocm_arch_list": archs}))
    backend._HERE = tmp_path


@pytest.mark.parametrize(
    "version,built",
    [
        ("0.6.18+amd.1.gfx942", "gfx942"),
        ("0.6.18+amd.1.gfx942.gfx950", "gfx942,gfx950"),
        ("0.6.18+gfx950", "gfx950"),
    ],
)
def test_matching_kernels_pass_the_check(backend, tmp_path, version, built):
    _write_manifest(tmp_path, backend, built)
    backend._check_version_matches_kernels(version)


def test_a_dropped_architecture_fails_the_build(backend, tmp_path):
    """validate_flashinfer_rocm_arch only warns when it narrows the target set.

    Shipping the wider name would put gfx950 in the version of a wheel holding
    gfx942 kernels, which the consumer-side arch guard cannot detect: the
    manifest it reads would say gfx942 and be believed.
    """
    _write_manifest(tmp_path, backend, "gfx942")
    with pytest.raises(RuntimeError, match="gfx942"):
        backend._check_version_matches_kernels("0.6.18+amd.1.gfx942.gfx950")
