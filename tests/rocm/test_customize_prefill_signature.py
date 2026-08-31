# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Signature parity for the ROCm custom-variant prefill JIT path.

rocm/prefill.py forwards arguments positionally into these generators and run
wrappers, so a parameter missing on the HIP side is a TypeError at call time.
Upstream is read with ast because modules.py cannot be imported on ROCm.
"""

import ast
import functools
import inspect
import pathlib

import pytest

import flashinfer
from flashinfer.rocm.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not IS_HIP, reason="modules_hip is only importable on ROCm"
)

_PKG_DIR = pathlib.Path(flashinfer.__file__).parent

CUSTOMIZE_PREFILL_GENERATORS = [
    "gen_customize_single_prefill_module",
    "gen_customize_batch_prefill_module",
]

# The JIT-path run wrappers must accept the same fixed argument prefix that the
# non-JIT ones do, up to and including enable_pdl; everything after it is *args.
JIT_RUN_WRAPPERS = ["ragged_run", "_fake_ragged_run", "paged_run", "_fake_paged_run"]


@functools.cache
def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _params(node: ast.FunctionDef) -> list[str]:
    a = node.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _positional(node: ast.FunctionDef) -> list[str]:
    """Names callers may pass positionally — everything before any ``*``."""
    a = node.args
    return [p.arg for p in (*a.posonlyargs, *a.args)]


def _find(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _generator(module_path: pathlib.Path, func_name: str) -> ast.FunctionDef:
    return _find(_parse(module_path), func_name)


def _wrapper(outer: str, inner: str) -> ast.FunctionDef:
    return _find(_find(_parse(_PKG_DIR / "rocm" / "prefill.py"), outer), inner)


@pytest.mark.parametrize("func_name", CUSTOMIZE_PREFILL_GENERATORS)
def test_hip_generator_matches_upstream_signature(func_name):
    hip = _generator(_PKG_DIR / "jit" / "rocm" / "modules.py", func_name)
    upstream = _generator(_PKG_DIR / "jit" / "attention" / "modules.py", func_name)

    assert _params(hip) == _params(upstream)
    # Names alone would still pass if HIP moved a parameter behind a `*`;
    # rocm/prefill.py forwards all of them positionally.
    assert _positional(hip) == _positional(upstream)


def test_rocm_wrapper_forwards_exactly_what_the_generator_accepts():
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    from flashinfer.rocm.prefill import get_customize_batch_prefill_module

    # get_customize_batch_prefill_module is wrapped by make_hashable_cache,
    # which uses functools.wraps, so signature() sees the underlying function.
    assert list(inspect.signature(get_customize_batch_prefill_module).parameters) == (
        list(inspect.signature(gen_customize_batch_prefill_module).parameters)
    )


@pytest.mark.parametrize("inner", JIT_RUN_WRAPPERS)
def test_jit_run_wrapper_takes_the_same_fixed_prefix_as_the_non_jit_one(inner):
    node = _wrapper("get_batch_prefill_jit_module", inner)
    jit = _positional(node)
    non_jit = _positional(_wrapper("get_batch_prefill_module", inner))

    assert jit[-1] == "enable_pdl", f"{inner} would absorb enable_pdl into *args"
    assert jit == non_jit[: len(jit)]
    # enable_pdl must stay positional and *args must survive to carry the
    # variant's additional tensors/scalars; either alone passes the name check.
    assert not node.args.kwonlyargs
    assert node.args.vararg is not None


@pytest.mark.parametrize("func_name", CUSTOMIZE_PREFILL_GENERATORS)
def test_fp8_enabled_is_rejected(func_name):
    from flashinfer.jit import attention as jit_attention

    generator = getattr(jit_attention, func_name)
    n_required = sum(
        p.default is inspect.Parameter.empty
        for p in inspect.signature(generator).parameters.values()
    )
    # The guard runs before any argument is inspected, so placeholders suffice.
    with pytest.raises(ValueError, match="fp8"):
        generator(*[None] * n_required, fp8_enabled=True)
