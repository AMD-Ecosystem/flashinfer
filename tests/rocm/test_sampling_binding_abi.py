# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pins the ROCm sampling bindings to the arity upstream's Python calls with.

``flashinfer/sampling.py`` is upstream's and calls ``module.<op>(...)``
positionally. When upstream widens a signature -- v0.6.18 added per-request
seed tensors, a ``valid`` output and multi-CTA scratch buffers -- the ROCm
declarations silently fall behind and every call raises at runtime, deep in the
suite. This reads both sides statically, so the mismatch surfaces without a GPU.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PY = _ROOT / "flashinfer" / "sampling.py"
_BINDING = _ROOT / "csrc" / "rocm" / "flashinfer_sampling_binding.cu"


def _python_call_arities() -> dict[str, int]:
    """Positional arg count of each ``module.<op>(...)`` in sampling.py."""
    tree = ast.parse(_PY.read_text())
    calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "module"
        ):
            assert not node.keywords, f"{node.func.attr} is called with keywords"
            calls[node.func.attr] = len(node.args)
    return calls


def _split_params(params: str) -> list[str]:
    """Top-level comma split, so std::optional<at::Tensor> stays one parameter."""
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _binding_arities() -> dict[str, int]:
    """Parameter count of each forward declaration in the binding TU."""
    src = _BINDING.read_text()
    decls = {}
    for m in re.finditer(r"^void (\w+)\(((?:[^;])*?)\);", src, re.M | re.S):
        decls[m.group(1)] = len(_split_params(m.group(2)))
    return decls


def test_the_binding_declares_every_op_the_python_calls():
    missing = sorted(set(_python_call_arities()) - set(_binding_arities()))
    assert not missing, (
        f"declared in sampling.py but not in the ROCm binding: {missing}"
    )


@pytest.mark.parametrize("op", sorted(_python_call_arities()))
def test_binding_arity_matches_the_python_call(op):
    py = _python_call_arities()[op]
    cpp = _binding_arities()[op]
    assert cpp == py, (
        f"{op}: sampling.py passes {py} args, csrc/rocm declares {cpp}. "
        "Upstream widened the signature; update csrc/rocm/{sampling,renorm}.cu."
    )


def test_the_definitions_agree_with_the_declarations():
    """A declaration the .cu files do not define links, then fails at call time."""
    defined = {}
    for path in ("sampling.cu", "renorm.cu"):
        src = (_ROOT / "csrc" / "rocm" / path).read_text()
        for m in re.finditer(r"^void (\w+)\(((?:[^;{])*?)\) \{", src, re.M | re.S):
            defined[m.group(1)] = len(_split_params(m.group(2)))
    for op, arity in _binding_arities().items():
        assert op in defined, f"{op} is declared but never defined in csrc/rocm/"
        assert defined[op] == arity, (
            f"{op}: declared with {arity} params, defined with {defined[op]}"
        )
