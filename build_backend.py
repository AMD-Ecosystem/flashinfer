# SPDX-FileCopyrightText: 2023 FlashInfer team.
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
In-tree PEP 517 backend for ``amd-flashinfer``.

This is a thin wrapper around ``setuptools.build_meta`` whose only extra job is
to materialize ``flashinfer/include`` from the top-level ``include/`` directory.

Why that matters: ``flashinfer/get_include_paths.py`` resolves headers as
``Path(__file__).parent / "include"``, which becomes ``FLASHINFER_INCLUDE_DIR``
in ``flashinfer/jit/env.py`` and ends up as the ``-I`` flag on every HIP JIT
compile. Without it, nothing builds at *runtime*, not just at build time.

Three materialization modes, and the differences are load-bearing:

- editable -> a relative symlink, so edits under ``include/`` are picked up with
  no rebuild (and it matches the manual worktree setup documented in CLAUDE.md).
- wheel -> a real recursive copy, because a symlink is not followed into a wheel
  and would ship a dangling link.
- sdist -> cleared. The tarball carries the source layout (top-level
  ``include/``, via MANIFEST.in) and a wheel built from it re-creates the copy;
  see ``_prepare_for_sdist``.

The header filter mirrors what the retired CMake ``install(DIRECTORY ...)`` rule
did, so wheel contents do not change with this backend swap.

Versioning is handled entirely by setuptools-scm via ``[tool.setuptools_scm]``
(which writes ``flashinfer/_version.py``). This backend deliberately does not
implement the upstream ``version.txt`` / ``_build_meta.py`` scheme.
"""

import shutil
from pathlib import Path

from setuptools import build_meta as _orig

_root = Path(__file__).parent.resolve()
_src_include = _root / "include"
_pkg_include = _root / "flashinfer" / "include"

# Matches the old CMake rule: FILES_MATCHING REGEX "\\.(cuh|h|hpp)$".
_HEADER_SUFFIXES = {".cuh", ".h", ".hpp"}


def _clear(path: Path) -> None:
    """Remove ``path`` whether it is a symlink, a directory, or a file.

    ``is_symlink()`` is checked first: for a symlink to a directory ``is_dir()``
    is also true, and ``rmtree`` on it would fail.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _materialize_include(use_symlink: bool) -> None:
    if not _src_include.is_dir():
        raise RuntimeError(f"missing source header tree: {_src_include}")

    _clear(_pkg_include)

    if use_symlink:
        # Relative, so the link stays valid if the checkout is moved or bind
        # mounted at a different path inside a container.
        _pkg_include.symlink_to(Path("..") / "include", target_is_directory=True)
        return

    shutil.copytree(
        _src_include,
        _pkg_include,
        symlinks=False,
        ignore=lambda _dir, names: [
            n
            for n in names
            if not (Path(_dir) / n).is_dir() and Path(n).suffix not in _HEADER_SUFFIXES
        ],
    )


def _prepare_for_editable() -> None:
    _materialize_include(use_symlink=True)


def _prepare_for_wheel() -> None:
    _materialize_include(use_symlink=False)


def _prepare_for_sdist() -> None:
    """Clear the generated copy; the sdist ships the source layout instead.

    An sdist carries top-level ``include/`` (see MANIFEST.in), and building a
    wheel from that sdist re-runs ``build_wheel`` and re-materializes the copy.
    Leaving a real copy here would duplicate the whole header tree in the
    tarball, and leaving a symlink would ship a dangling one.
    """
    _clear(_pkg_include)


# --------------------------------------------------------------- PEP 517 hooks


def get_requires_for_build_wheel(config_settings=None):
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_for_wheel()
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    return _orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_wheel()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _prepare_for_sdist()
    return _orig.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
