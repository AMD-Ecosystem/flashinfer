# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Every relative import in :mod:`flashinfer` must name a module that exists.

Function-local imports are invisible to ``import flashinfer`` and to a pkgutil
walk, so moving a module can leave one dangling with nothing failing until the
line runs -- and when it runs inside a ``try``, never.
"""

import ast
import pathlib

_PKG = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"

# setuptools-scm writes these at build time; they are absent in a source tree.
_GENERATED = {"flashinfer._build_meta", "flashinfer._version"}


def _module_exists(dotted: str) -> bool:
    path = _PKG.parent / dotted.replace(".", "/")
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


def _relative_imports():
    """(source file, lineno, resolved dotted name) for every `from .x import y`."""
    for file in sorted(_PKG.rglob("*.py")):
        if "__pycache__" in file.parts:
            continue
        package = list(file.relative_to(_PKG.parent).with_suffix("").parts)[:-1]
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            # `from . import x` is skipped: x may be a symbol, not a module.
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if not node.module or len(package) - (node.level - 1) < 1:
                continue
            base = package[: len(package) - (node.level - 1)]
            yield file, node.lineno, f"{'.'.join(base)}.{node.module}"


def test_no_relative_import_points_at_a_missing_module():
    dangling = [
        f"{f.relative_to(_PKG.parent)}:{line} -> {target}"
        for f, line, target in _relative_imports()
        if target not in _GENERATED and not _module_exists(target)
    ]

    assert not dangling, "relative imports resolve to nothing:\n" + "\n".join(dangling)


def test_the_sweep_reaches_function_local_imports():
    """A sweep blind to these would pass while the AOT arch guard was broken.

    jit/rocm/env.py imports the arch canonicaliser inside a function, and that
    is exactly the import the move to flashinfer/rocm/ left dangling.
    """
    found = {
        (str(f.relative_to(_PKG.parent)), target)
        for f, _, target in _relative_imports()
    }

    assert ("flashinfer/jit/rocm/env.py", "flashinfer.rocm.hip_utils") in found
