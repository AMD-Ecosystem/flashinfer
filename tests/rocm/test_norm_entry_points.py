# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every ``get_norm_module().<name>`` call site must resolve, or be declared.

``flashinfer/norm/__init__.py`` forces ``_USE_CUDA_NORM`` on HIP, so on ROCm
every norm entry point dispatches through ``get_norm_module()``. Five of the
nine names it reaches are not bound by ``csrc/rocm/flashinfer_norm_binding.cu``,
and the call fails with a bare ``AttributeError`` naming neither ROCm nor the op.

``test_kernel_source_coverage.py`` cannot catch this: it checks that a named
``.cu`` *file* exists, and here the file exists and the *symbol* does not.

The expected-missing set is asserted in both directions on purpose. A name that
stops resolving is a regression; a name that starts resolving means someone
ported the op and must move it out of the set, which is also how the arch
support matrix and ``docs/rocm/backends.md`` get updated.
"""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_NORM_SRC = Path(__file__).resolve().parents[2] / "flashinfer" / "norm" / "__init__.py"

# Not bound by csrc/rocm/flashinfer_norm_binding.cu, which binds exactly
# rmsnorm, fused_add_rmsnorm, gemma_rmsnorm and gemma_fused_add_rmsnorm.
_UNBOUND_ON_ROCM = {
    "fused_add_rmsnorm_quant",
    "fused_dit_layernorm",
    "layernorm",
    "layernorm_quant",
    "rmsnorm_quant",
}


def _call_sites() -> set[str]:
    """Names reached as ``get_norm_module().<name>``, read out of the source.

    AST rather than ``dir()``: the point is to catch a call site upstream adds,
    which no amount of introspecting the built module would reveal.
    """
    names = set()
    for node in ast.walk(ast.parse(_NORM_SRC.read_text())):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "get_norm_module"
        ):
            names.add(node.attr)
    return names


def test_call_sites_are_found():
    """Guard the guard: an upstream refactor that renames the accessor would
    otherwise make every assertion below vacuously true."""
    assert "rmsnorm" in _call_sites()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm device")
def test_norm_entry_points_resolve_or_are_declared():
    from flashinfer.norm import get_norm_module

    module = get_norm_module()
    sites = _call_sites()
    missing = {name for name in sites if not hasattr(module, name)}

    assert missing == _UNBOUND_ON_ROCM & sites, (
        "the set of unresolvable norm entry points moved.\n"
        f"  missing now: {sorted(missing)}\n"
        f"  declared   : {sorted(_UNBOUND_ON_ROCM & sites)}\n"
        "If an op was ported, drop it from _UNBOUND_ON_ROCM and add its row to "
        "flashinfer/rocm/arch_caps.py. If one regressed, the binding lost a symbol."
    )


def test_declared_missing_names_are_still_call_sites():
    """A stale entry in the set would silently weaken the assertion above."""
    stale = _UNBOUND_ON_ROCM - _call_sites()
    assert not stale, f"_UNBOUND_ON_ROCM names nothing reachable: {sorted(stale)}"
