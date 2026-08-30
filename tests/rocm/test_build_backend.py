# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Postconditions for the in-tree PEP 517 backend's ``flashinfer/include`` handling.

Guards the regressions in ``_restoring_pkg_trees``: a wheel or sdist build
must never leave a header copy in the checkout, because ``get_include()`` would
resolve it later and shadow edits under ``include/``.

The assertions are filesystem-only and need no GPU, but collection still pulls
in the suite's torch-importing conftest — there is no CPU-only lane to put this
in, and ``tests/rocm`` is the only directory ``testpaths`` covers.
"""

import importlib.util
import shutil
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend():
    spec = importlib.util.spec_from_file_location(
        "_fi_build_backend_rocm", _REPO_ROOT / "build_backend_rocm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inc(bb):
    """(source, destination) of the include tree — the first entry in _trees()."""
    src, dst, _suffixes = bb._trees()[0]
    return src, dst


def _seed(bb, state):
    """Put the include tree's destination into ``state``."""
    src, dst = _inc(bb)
    bb._clear(dst)
    if state == "symlink":
        dst.symlink_to(Path("..") / "include", target_is_directory=True)
    elif state == "realdir":
        shutil.copytree(src, dst)
    elif state == "abslink":
        dst.symlink_to(src, target_is_directory=True)
    elif state == "file":
        dst.write_text("not a directory")
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
    # The csrc tree must exist too: _prepare_for_wheel materializes both, and a
    # missing source raises before the include assertions are ever reached.
    csrc = tmp_path / "csrc" / "rocm"
    csrc.mkdir(parents=True)
    (csrc / "op.cu").write_text("// x\n")
    (tmp_path / "flashinfer").mkdir()
    bb._root = tmp_path
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
    with backend._restoring_pkg_trees():
        getattr(backend, prepare)()
    assert _describe(_inc(backend)[1]) == expected


@pytest.mark.parametrize("state", ["symlink", "abslink", "realdir", "absent"])
def test_restore_runs_when_the_build_raises(backend, state):
    expected = "symlink:../include" if state.endswith("link") else "absent"
    _seed(backend, state)
    with (
        pytest.raises(RuntimeError, match="build failed"),
        backend._restoring_pkg_trees(),
    ):
        backend._prepare_for_wheel()
        raise RuntimeError("build failed")
    assert _describe(_inc(backend)[1]) == expected


def test_restore_survives_a_vanished_source_tree(backend):
    """The finally must not raise: that would mask whatever failed the build."""
    _seed(backend, "symlink")
    with backend._restoring_pkg_trees():
        shutil.rmtree(_inc(backend)[0])
    assert _describe(_inc(backend)[1]) == "symlink:../include"


def test_wheel_copy_is_real_and_header_filtered(backend):
    """Inside the build the copy must be real (symlinks are not followed into a wheel)."""
    _seed(backend, "symlink")
    with backend._restoring_pkg_trees():
        backend._prepare_for_wheel()
        pkg = _inc(backend)[1]
        assert not pkg.is_symlink() and pkg.is_dir()
        names = {p.name for p in pkg.rglob("*") if p.is_file()}
        assert names == {"attention.cuh", "utils.h", "traits.hpp"}, names


@pytest.mark.parametrize(
    "index, target",
    [(0, "../include"), (1, "../../csrc/rocm")],
    ids=["include", "csrc"],
)
def test_editable_symlink_is_relative(backend, index, target):
    """An absolute link would break under a container bind mount.

    Both trees, because csrc/rocm sits a level deeper and its target is
    derived rather than literal.
    """
    _seed(backend, "absent")
    backend._prepare_for_editable()
    assert backend._trees()[index][1].readlink() == Path(target)


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


# --------------------------------------------------------- artifact integration


def _isolated_project(dest):
    """Copy the source layout a build needs into ``dest``, without .git."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "build_backend.py",
        "build_backend_rocm.py",
        "build_utils.py",
        "LICENSE",
        "NOTICE",
        "README.md",
    ):
        shutil.copy2(_REPO_ROOT / name, dest / name)
    shutil.copytree(_REPO_ROOT / "include", dest / "include")
    # Both source trees, or the inventory assertions below compare 0 to 0.
    shutil.copytree(_REPO_ROOT / "csrc" / "rocm", dest / "csrc" / "rocm")
    shutil.copytree(
        _REPO_ROOT / "flashinfer",
        dest / "flashinfer",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "include", "csrc"),
    )
    # The sibling project roots must be present or the packages.find assertion
    # below is vacuous — an unanchored glob is exactly what swept them in.
    for sibling in ("flashinfer-cubin", "flashinfer-jit-cache"):
        pkg = sibling.replace("-", "_")
        (dest / sibling / pkg).mkdir(parents=True)
        (dest / sibling / pkg / "__init__.py").write_text("")
        (dest / sibling / "pyproject.toml").write_text("")
    return dest


def test_wheel_carries_the_paths_the_jit_resolves(tmp_path, monkeypatch):
    """Assert on the artifact: the helpers above do not prove what setuptools ships.

    csrc/rocm lives at the repo root and reaches the wheel only because
    build_backend_rocm.py materializes it into flashinfer/csrc/rocm, so this
    also covers that step.
    """
    build = pytest.importorskip("build")
    project = _isolated_project(tmp_path / "src")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.1")

    out = tmp_path / "dist"
    # No isolation: build deps are already present, and this must not hit the network.
    name = build.ProjectBuilder(project).build("wheel", str(out))

    with zipfile.ZipFile(name) as z:
        names = z.namelist()
        top = z.read(next(f for f in names if f.endswith("top_level.txt"))).decode()

    shipped = {f for f in names if f.startswith("flashinfer/include/")}
    for suffix in (".cuh", ".h", ".hpp"):
        assert any(f.endswith(suffix) for f in shipped), suffix
    assert len(shipped) == sum(
        1
        for f in (_REPO_ROOT / "include").rglob("*")
        if f.suffix in {".cuh", ".h", ".hpp"}
    )
    csrc = {f for f in names if f.startswith("flashinfer/csrc/rocm/")}
    assert csrc, "csrc/rocm did not reach the wheel at all"
    assert len(csrc) == sum(
        1
        for f in (_REPO_ROOT / "csrc" / "rocm").rglob("*")
        if f.suffix in {".cu", ".cc", ".h", ".jinja"}
    )
    # The sibling projects leaked in once via an unanchored packages.find glob.
    assert top.split() == ["flashinfer"], top
    assert not [f for f in names if f.startswith(("flashinfer-", "amd-flashinfer-"))]


class _Recorder:
    """Stands in for setuptools.build_meta, recording the delegated call."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _hook(*args):
            self.calls.append((name, args))
            return f"{name}-result"

        return _hook


@pytest.fixture
def hooks(backend, monkeypatch):
    """The backend with setuptools swapped out, so no real build runs."""
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_orig", recorder)
    return backend, recorder


@pytest.mark.parametrize("index", [0, 1], ids=["include", "csrc"])
def test_missing_source_tree_is_named(backend, index):
    """Either missing source must name itself, not fail later as a missing file."""
    src = backend._trees()[index][0]
    shutil.rmtree(src)
    with pytest.raises(RuntimeError, match=f"missing source tree: {src}"):
        backend._prepare_for_editable()


@pytest.mark.parametrize(
    "hook, args, delegated",
    [
        ("get_requires_for_build_wheel", (), (None,)),
        ("get_requires_for_build_sdist", (), (None,)),
        ("get_requires_for_build_editable", (), (None,)),
        ("prepare_metadata_for_build_wheel", ("md",), ("md", None)),
    ],
)
def test_passthrough_hooks_delegate_unchanged(hooks, hook, args, delegated):
    """These must not touch flashinfer/include; metadata comes from [project]."""
    backend, recorder = hooks

    assert getattr(backend, hook)(*args) == f"{hook}-result"

    assert recorder.calls == [(hook, delegated)]
    assert not _inc(backend)[1].exists()


def test_editable_metadata_materializes_a_symlink_first(hooks):
    backend, recorder = hooks

    result = backend.prepare_metadata_for_build_editable("md")

    assert result == "prepare_metadata_for_build_editable-result"
    assert recorder.calls == [("prepare_metadata_for_build_editable", ("md", None))]
    assert _describe(_inc(backend)[1]) == "symlink:../include"


def test_build_editable_materializes_a_symlink_and_leaves_it(hooks):
    backend, recorder = hooks

    assert backend.build_editable("wd") == "build_editable-result"

    assert recorder.calls == [("build_editable", ("wd", None, None))]
    assert _describe(_inc(backend)[1]) == "symlink:../include"


def test_build_wheel_sees_a_real_copy_and_leaves_nothing(hooks):
    backend, recorder = hooks
    seen = {}
    backend._orig.build_wheel = lambda *a: seen.setdefault(
        "state", _describe(_inc(backend)[1])
    )

    backend.build_wheel("wd")

    # Real directory during the build; gone once the context manager unwinds.
    assert seen["state"] == "realdir"
    assert _describe(_inc(backend)[1]) == "absent"


def test_build_sdist_sees_no_copy_and_leaves_nothing(hooks):
    backend, _ = hooks
    _seed(backend, "realdir")
    seen = {}
    backend._orig.build_sdist = lambda *a: seen.setdefault(
        "state", _describe(_inc(backend)[1])
    )

    backend.build_sdist("sd")

    assert seen["state"] == "absent"
    assert _describe(_inc(backend)[1]) == "absent"
