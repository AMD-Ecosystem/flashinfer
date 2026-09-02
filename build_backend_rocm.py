# SPDX-FileCopyrightText: 2023 FlashInfer team.
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
In-tree PEP 517 backend for ``amd-flashinfer``.

Kept separate from upstream's ``build_backend.py``, which stays byte-identical
and unused, so rebases never conflict on it. ``pyproject.toml`` selects this one.

This is a thin wrapper around ``setuptools.build_meta`` whose only extra job is
materializing two generated trees inside the package, from sources that live at
the repository root: ``flashinfer/include`` from ``include/``, and
``flashinfer/csrc/rocm`` from ``csrc/rocm/``.

Why that matters: ``flashinfer/get_include_paths.py`` resolves both as
``Path(__file__).parent / ...``, which become ``FLASHINFER_INCLUDE_DIR`` and
``FLASHINFER_CSRC_DIR`` in ``flashinfer/jit/env.py``. One is passed as
``-isystem`` on every HIP JIT compile and the other is where the ``.cu`` sources
are read from, so without them nothing builds at *runtime*, not just at build
time. Keeping the destinations unchanged is what lets the sources move to the
root without touching the packaging globs or the public path helpers.

Three materialization modes, and the differences are load-bearing:

- editable -> a relative symlink, so edits under the source tree are picked up
  with no rebuild (and it matches the manual worktree setup in CLAUDE.md).
- wheel -> a real recursive copy, because a symlink is not followed into a wheel
  and would ship a dangling link.
- sdist -> cleared; the tarball carries the root trees, ``include/`` via
  MANIFEST.in and ``csrc/rocm/`` via setuptools-scm's tracked-file finder.

Wheel and sdist build in the checkout, so neither leaves a copy behind: a
symlink is put back, and any other generated tree is deleted rather than left to
shadow the source (see ``_restoring_pkg_trees``).

The per-tree suffix filters mirror the ``package-data`` globs in
``pyproject.toml``, so wheel contents do not change with a source move.

Versioning is handled entirely by setuptools-scm via ``[tool.setuptools_scm]``
(which writes ``flashinfer/_version.py``). This backend deliberately does not
implement the upstream ``version.txt`` / ``_build_meta.py`` scheme.
"""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from setuptools import build_meta as _orig

_root = Path(__file__).parent.resolve()


def _trees():
    """(source at the repo root, destination in the package, suffixes to copy).

    Computed per call rather than at import so rebinding ``_root`` redirects
    every tree at once; the link target follows the destination's depth.
    """
    return (
        # Header suffixes match the old CMake rule: REGEX "\\.(cuh|h|hpp)$".
        (_root / "include", _root / "flashinfer" / "include", {".cuh", ".h", ".hpp"}),
        (
            _root / "csrc" / "rocm",
            _root / "flashinfer" / "csrc" / "rocm",
            {".cu", ".cc", ".h", ".jinja"},
        ),
    )


# Headers the wheel ships, as paths under include/. Everything else in that tree
# is upstream CUDA that nothing on ROCm compiles: 287 headers in, 47 out. Note
# MANIFEST.in grafts include/ wholesale, so the sdist carries all 287.
# fp16.h is the one upstream header a ROCm source still reaches for
# (rocm/attention/prefill.cuh). Editable installs are not filtered -- they
# symlink the whole tree, so a developer keeps the upstream headers to read.
_WHEEL_HEADER_DIRS = ("flashinfer/rocm",)
_WHEEL_HEADER_FILES = ("flashinfer/fp16.h",)


def _wanted_in_wheel(relative: Path) -> bool:
    posix = relative.as_posix()
    return posix in _WHEEL_HEADER_FILES or any(
        posix.startswith(d + "/") for d in _WHEEL_HEADER_DIRS
    )


def _clear(path: Path) -> None:
    """Remove ``path`` whether it is a symlink, a directory, or a file.

    ``is_symlink()`` is checked first: for a symlink to a directory ``is_dir()``
    is also true, and ``rmtree`` on it would fail.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_target(src: Path, dst: Path) -> Path:
    """Relative path from ``dst``'s parent back to ``src``."""
    return Path(os.path.relpath(src, dst.parent))


def _materialize(src: Path, dst: Path, suffixes: set, use_symlink: bool) -> None:
    if not src.is_dir():
        raise RuntimeError(f"missing source tree: {src}")

    _clear(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if use_symlink:
        # Relative, so the link stays valid if the checkout is moved or bind
        # mounted at a different path inside a container.
        dst.symlink_to(_link_target(src, dst), target_is_directory=True)
        return

    shutil.copytree(
        src,
        dst,
        symlinks=False,
        ignore=lambda _dir, names: [
            n
            for n in names
            if not (Path(_dir) / n).is_dir() and Path(n).suffix not in suffixes
        ],
    )


def _prepare_for_editable() -> None:
    for src, dst, suffixes in _trees():
        _materialize(src, dst, suffixes, use_symlink=True)


def _prepare_for_wheel() -> None:
    for src, dst, suffixes in _trees():
        _materialize(src, dst, suffixes, use_symlink=False)
    _prune_unshipped_headers(_root / "flashinfer" / "include")


def _prune_unshipped_headers(include_dir: Path) -> None:
    """Drop the upstream CUDA headers from the copied include tree.

    Runs after the copy rather than as a copytree filter so the kept set is
    stated once, positively, in _wanted_in_wheel.
    """
    kept = 0
    for path in sorted(include_dir.rglob("*"), reverse=True):
        if path.is_dir():
            if not any(path.iterdir()):
                path.rmdir()
        elif _wanted_in_wheel(path.relative_to(include_dir)):
            kept += 1
        else:
            path.unlink()
    if kept == 0:
        raise RuntimeError(f"pruned every header under {include_dir}")


def _prepare_for_sdist() -> None:
    """Clear the generated copies; the sdist ships the root trees instead.

    A real copy would duplicate them in the tarball, a symlink would dangle, and
    a wheel built from the sdist re-materializes them anyway.
    """
    for _src, dst, _suffixes in _trees():
        _clear(dst)


@contextmanager
def _restoring_pkg_trees():
    """Leave each generated tree a symlink, or leave it absent.

    A generated copy left in the checkout is what ``get_include()`` and
    ``get_csrc_dir()`` resolve later, shadowing edits under the real source.
    Absent fails loudly; stale does not. A restored link is re-made relative,
    since the retired CMake hook wrote absolute ones and those break under a
    bind mount.
    """
    trees = _trees()
    had_link = [dst.is_symlink() for _src, dst, _suffixes in trees]
    try:
        yield
    finally:
        for (src, dst, _suffixes), was_link in zip(trees, had_link, strict=True):
            _clear(dst)
            if was_link:
                # Linked directly, not via _prepare_for_editable: that validates
                # the source and would raise out of this finally, masking the
                # build's own error and leaving nothing behind.
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(_link_target(src, dst), target_is_directory=True)


# --------------------------------------------------------------- PEP 517 hooks


def get_requires_for_build_wheel(config_settings=None):
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    # No headers needed: metadata comes from [project], and build_wheel
    # materializes the tree itself.
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    return _orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _restoring_pkg_trees():
        _prepare_for_wheel()
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    with _restoring_pkg_trees():
        _prepare_for_sdist()
        return _orig.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
