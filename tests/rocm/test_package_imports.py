# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Every relative import in :mod:`flashinfer` must resolve to something real.

Function-local imports are invisible to ``import flashinfer`` and to a pkgutil
walk, so moving a module can leave one dangling with nothing failing until the
line runs -- and when it runs inside a ``try``, never.
"""

import ast
import functools
import pathlib

_PKG = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"

# setuptools-scm writes these at build time; they are absent in a source tree.
_GENERATED = {"flashinfer._build_meta", "flashinfer._version"}


def _module_exists(dotted: str) -> bool:
    path = _PKG.parent / dotted.replace(".", "/")
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


@functools.lru_cache(maxsize=None)
def _names_bound_in_init(package: str) -> frozenset:
    """Top-level names an ``__init__.py`` defines, so `from . import x` resolves.

    ``x`` is usually a submodule but can be a re-export, which no filesystem
    check would find.
    """
    init = _PKG.parent / package.replace(".", "/") / "__init__.py"
    if not init.exists():
        return frozenset()

    names = set()
    for node in ast.walk(ast.parse(init.read_text(), filename=str(init))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(names)


def _relative_imports():
    """(source file, lineno, dotted target) for every relative import.

    `from .x import y` yields the module `.x`; `from . import y` yields `.y`,
    which resolves as a submodule or as a name the package re-exports.
    """
    for file in sorted(_PKG.rglob("*.py")):
        if "__pycache__" in file.parts:
            continue
        package = list(file.relative_to(_PKG.parent).with_suffix("").parts)[:-1]
        for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if len(package) - (node.level - 1) < 1:
                continue
            base = ".".join(package[: len(package) - (node.level - 1)])
            if node.module:
                yield file, node.lineno, f"{base}.{node.module}"
                continue
            for alias in node.names:
                if alias.name != "*":
                    yield file, node.lineno, f"{base}.{alias.name}"


def _resolves(target: str) -> bool:
    if target in _GENERATED or _module_exists(target):
        return True
    package, _, name = target.rpartition(".")
    return name in _names_bound_in_init(package)


def test_no_relative_import_points_at_a_missing_module():
    dangling = [
        f"{f.relative_to(_PKG.parent)}:{line} -> {target}"
        for f, line, target in _relative_imports()
        if not _resolves(target)
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
    # `from . import x` must be in scope too, or the sweep has a blind side.
    assert ("flashinfer/rocm/api.py", "flashinfer.rocm.install_shadow_modules") in found
