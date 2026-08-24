# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Postconditions for the in-tree PEP 517 backend's ``flashinfer/include`` handling.

Guards the regressions in ``_restoring_pkg_include``: a wheel or sdist build
must never leave a header copy in the checkout, because ``get_include()`` would
resolve it later and shadow edits under ``include/``.

The assertions are filesystem-only and need no GPU, but collection still pulls
in the suite's torch-importing conftest — there is no CPU-only lane to put this
in, and ``tests/rocm_tests`` is the only directory ``testpaths`` covers.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend():
    spec = importlib.util.spec_from_file_location(
        "_fi_build_backend", _REPO_ROOT / "build_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(bb, state):
    """Put ``_pkg_include`` into ``state`` and return it."""
    bb._clear(bb._pkg_include)
    if state == "symlink":
        bb._pkg_include.symlink_to(Path("..") / "include", target_is_directory=True)
    elif state == "realdir":
        shutil.copytree(bb._src_include, bb._pkg_include)
    elif state == "abslink":
        bb._pkg_include.symlink_to(bb._src_include, target_is_directory=True)
    elif state == "file":
        bb._pkg_include.write_text("not a directory")
    elif state != "absent":
        raise ValueError(state)


def _describe(path):
    if path.is_symlink():
        return f"symlink:{path.readlink()}"
    if path.is_dir():
        return "realdir"
    return "file" if path.is_file() else "absent"


@pytest.fixture
def backend(tmp_path):
    """Backend module rebound to a scratch tree, so the real checkout is untouched."""
    bb = _load_backend()
    src = tmp_path / "include"
    (src / "flashinfer").mkdir(parents=True)
    for name in ("attention.cuh", "utils.h", "traits.hpp", "notes.txt", "gen.jinja"):
        (src / "flashinfer" / name).write_text("// x\n")
    pkg = tmp_path / "flashinfer"
    pkg.mkdir()
    bb._root, bb._src_include, bb._pkg_include = tmp_path, src, pkg / "include"
    return bb


@pytest.mark.parametrize("prepare", ["_prepare_for_wheel", "_prepare_for_sdist"])
@pytest.mark.parametrize(
    "state, expected",
    [
        ("symlink", "symlink:../include"),
        ("abslink", "symlink:../include"),
        ("realdir", "absent"),
        ("file", "absent"),
        ("absent", "absent"),
    ],
)
def test_restore_leaves_symlink_or_nothing(backend, prepare, state, expected):
    """A symlink survives a build; every other prior state is cleared, not left stale."""
    _seed(backend, state)
    with backend._restoring_pkg_include():
        getattr(backend, prepare)()
    assert _describe(backend._pkg_include) == expected


@pytest.mark.parametrize("state", ["symlink", "realdir", "absent"])
def test_restore_runs_when_the_build_raises(backend, state):
    expected = "symlink:../include" if state == "symlink" else "absent"
    _seed(backend, state)
    with (
        pytest.raises(RuntimeError, match="build failed"),
        backend._restoring_pkg_include(),
    ):
        backend._prepare_for_wheel()
        raise RuntimeError("build failed")
    assert _describe(backend._pkg_include) == expected


def test_wheel_copy_is_real_and_header_filtered(backend):
    """Inside the build the copy must be real (symlinks are not followed into a wheel)."""
    _seed(backend, "symlink")
    with backend._restoring_pkg_include():
        backend._prepare_for_wheel()
        pkg = backend._pkg_include
        assert not pkg.is_symlink() and pkg.is_dir()
        names = {p.name for p in pkg.rglob("*") if p.is_file()}
        assert names == {"attention.cuh", "utils.h", "traits.hpp"}, names


def test_editable_symlink_is_relative(backend):
    """An absolute link would break under a container bind mount."""
    _seed(backend, "absent")
    backend._prepare_for_editable()
    assert backend._pkg_include.readlink() == Path("../include")


def test_find_patterns_exclude_sibling_projects():
    """``flashinfer*`` also matched flashinfer-cubin/ and shipped it in the wheel."""
    try:
        import tomllib  # Python >= 3.11
    except ImportError:
        tomllib = pytest.importorskip("tomli")  # requires-python allows 3.10
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    patterns = cfg["tool"]["setuptools"]["packages"]["find"]["include"]
    from fnmatch import fnmatch

    for sibling in ("flashinfer-cubin", "flashinfer-jit-cache"):
        assert not any(fnmatch(sibling, p) for p in patterns), sibling
    for wanted in ("flashinfer", "flashinfer.jit", "flashinfer.jit.attention"):
        assert any(fnmatch(wanted, p) for p in patterns), wanted
