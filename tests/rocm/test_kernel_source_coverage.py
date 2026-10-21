# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every op that names a kernel source must have that source on ROCm.

A module whose ``gen_*`` JIT spec lists a file absent from ``csrc/rocm`` imports
fine and dies in ninja on first call -- the failure upstream syncs keep
introducing, and the one ``CUDA_ONLY_MODULES`` exists to convert into a
statement about the backend. ``test_comm_import_gate.py`` asserts every gated
module raises; this asserts the complement, so a newly vendored op cannot land
unclassified.

Pure AST, no torch and no GPU: it has to run in the hardware-less lane, which
installs neither, and under ``--noconftest``.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG = _REPO_ROOT / "flashinfer"
# The JIT resolves FLASHINFER_CSRC_DIR here on ROCm (get_include_paths.py).
_CSRC = _REPO_ROOT / "csrc" / "rocm"
_JIT_ROCM = _PKG / "jit" / "rocm"

# Ops with no ROCm kernel that are deliberately left importable, because
# gating them would break a module that does work: topk_varlen imports topk at
# module scope, and tests/trace/template_registry.py registers four of the
# five. (sampling's top-k-first path also imports topk, but behind a
# `not IS_HIP` guard, so it is not a reason.) Their ninja error already names
# the missing file, which is a fair report for an op nobody has ported.
_UNPORTED = {
    # Keyed by (importing module, generator file): flashinfer.norm reaches
    # both jit/norm.py, which is supported, and jit/rmsnorm_silu.py, which is
    # not. Excusing the module would stop guarding the supported half.
    ("flashinfer.concat_ops", "dsv3_optimizations.py"): "no concat_mla kernel",
    ("flashinfer.mhc", "mhc.py"): "no mhc kernel",
    ("flashinfer.nvfp4_attention_sm120", "nvfp4_attention_sm120.py"): "SM120 NVFP4",
    ("flashinfer.topk", "topk.py"): "no topk kernel",
    ("flashinfer.xqa", "xqa.py"): "SM90+ XQA kernels",
    # Two logging side paths, both lazily built and neither reached by any
    # in-tree caller: set_log_level() in utils, and the opt-in GPU stats
    # counter in api_logging. Porting them is a separate change.
    ("flashinfer.utils", "spdlog.py"): "spdlog logging.cc",
    ("flashinfer.api_logging", "api_log_stats.py"): "api_log_stats.cu",
    # comm submodules that are not themselves gated but whose only JIT spec is
    # the CUDA comm module; importing one already fails transitively through a
    # gated dependency.
    ("flashinfer.comm.dcp_alltoall", "comm.py"): "CUDA comm kernels",
    ("flashinfer.comm.trtllm_moe_alltoall", "comm.py"): "CUDA comm kernels",
    ("flashinfer.comm.ulysses", "comm.py"): "CUDA comm kernels",
    # Mamba/SSM kernels arrived with v0.6.18 and are unported. ssd_combined is
    # gated instead -- it imports cutlass eagerly.
    ("flashinfer.mamba.checkpointing_ssu", "checkpointing_ssu.py"): "Mamba SSM",
    (
        "flashinfer.mamba.selective_state_update",
        "selective_state_update.py",
    ): "Mamba SSM",
    # norm itself is supported; only its fused rmsnorm+silu variant has no
    # ROCm source, and nothing in tree calls it.
    ("flashinfer.norm", "rmsnorm_silu.py"): "rmsnorm_silu.cu",
    # TensorRT-LLM host utilities (nv_internal).
    ("flashinfer.tllm_utils", "tllm_utils.py"): "nv_internal TensorRT-LLM sources",
    # KDA: csrc/rocm has no kda/ tree. These import on ROCm, unlike
    # flashinfer.kda itself, which stops at a missing tvm_ffi.
    ("flashinfer.kda_prefill", "flash_kda.py"): "no kda/ sources",
    ("flashinfer.kda_kernels.recurrent_kda", "flash_kda_decode.py"): "no kda/ sources",
    ("flashinfer.kda_kernels.cake_packed_kda_decode", "cake_flash_kda_packed_t1.py"): (
        "no kda/ sources"
    ),
    ("flashinfer.kda_kernels.cake_packed_kda_decode", "cake_kda_packed_t1.py"): (
        "no kda/ sources"
    ),
    # Both reach the nv_internal FP4 quantization sources.
    (
        "flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge",
        "fp4_quantization.py",
    ): ("nv_internal FP4 sources"),
    (
        "flashinfer.trace.templates.gemm",
        "fp4_quantization.py",
    ): "nv_internal FP4 sources",
    # moe_ep's SM90 push-style MegaMoE shim, on the nv_internal DeepGEMM tree.
    (
        "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe",
        "gemm.py",
    ): "SM90 MegaMoE",
    (
        "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim",
        "gemm.py",
    ): "SM90 MegaMoE",
    (
        "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.runner",
        "gemm.py",
    ): "SM90 MegaMoE",
}


def _load_registry():
    """flashinfer/rocm/__init__.py without importing flashinfer (needs torch)."""
    name = "_fi_rocm_registry"
    spec = importlib.util.spec_from_file_location(name, _PKG / "rocm" / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_registry = _load_registry()


def _dotted(path: Path) -> str:
    parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str | None, level: int, package: str) -> str:
    """Absolute name for an import, as importlib would resolve it."""
    if level == 0:  # already absolute: `from flashinfer.jit import ...`
        return module or ""
    base = package.rsplit(".", level - 1)[0] if level > 1 else package
    return f"{base}.{module}" if module else base


def _parameter_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Parameters bound to the string literals this file's call sites pass.

    The ROCm POD generators share one body and name every source off a
    `prefix` argument -- "pod" from one caller, "batch_pod" from the other --
    so without this the guard sees no sources for a supported op at all.
    """
    params = {
        node.name: [a.arg for a in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    bound: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        names = params.get(node.func.id)
        if names is None:
            continue
        pairs = list(zip(names, node.args, strict=False)) + [
            (kw.arg, kw.value) for kw in node.keywords if kw.arg
        ]
        for name, value in pairs:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                bound.setdefault(name, set()).add(value.value)
    return bound


def _string_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Names bound to string values, following literals, loops and f-strings.

    `for filename in [f"{prefix}.cu", ...]: ... CSRC_DIR / filename` is as
    common in the generators as a plain literal, and a check that only saw
    literals would stay green with those sources deleted.
    """
    bound: dict[str, set[str]] = _parameter_bindings(tree)

    def literals(node):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return {v for elt in node.elts for v in literals(elt)}
        return _values(node, bound)

    # Two passes: an f-string can interpolate a name bound earlier in the file.
    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                values, targets = literals(node.value), node.targets
            elif isinstance(node, (ast.For, ast.comprehension)):
                values, targets = literals(node.iter), [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and values:
                    bound.setdefault(target.id, set()).update(values)
    return bound


def _values(node, bound: dict[str, set[str]]) -> set[str]:
    """The strings this expression can evaluate to, empty when not knowable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(bound.get(node.id, ()))
    if isinstance(node, ast.JoinedStr):
        out = {""}
        for part in node.values:
            pieces = (
                {part.value}
                if isinstance(part, ast.Constant)
                else _values(part.value, bound)
                if isinstance(part, ast.FormattedValue)
                else set()
            )
            if not pieces:  # one unknown interpolation makes the whole unknown
                return set()
            out = {prefix + piece for prefix in out for piece in pieces}
        return out
    return set()


def _csrc_names(jit_file: Path) -> tuple[set[str], set[str]]:
    """Every source path built from FLASHINFER_CSRC_DIR in this generator.

    Follows one level of aliasing -- both `csrc_dir = FLASHINFER_CSRC_DIR`
    on the left and a string-bound name on the right -- which is as far as the
    generators go today.
    """
    tree = ast.parse(jit_file.read_text())
    bound = _string_bindings(tree)
    # Alias name -> the relative prefixes it stands for. An alias is not always
    # the bare root: cute_sm120_mxfp8_groupwise.py binds
    # `csrc_dir = FLASHINFER_CSRC_DIR / "cute_sm120_mxfp8_groupwise"` and then
    # appends six filenames to it.
    aliases: dict[str, set[str]] = {}
    # Rooted expressions whose filename this analysis cannot read. Distinct
    # from "no sources at all": an empty result for both would let a generator
    # using an unsupported dynamic expression pass unexamined.
    unresolved: set[str] = set()
    # Helper name -> the prefixes it returns. monomoe, bgmv_moe and flash_kda
    # all bind `csrc_dir = _get_..._csrc_dir()`, and a Call the analysis cannot
    # see through yields neither sources nor an unresolved marker.
    helpers: dict[str, set[str]] = {}

    def components(node) -> set[str] | None:
        """Relative paths this expression builds, or None if not rooted here."""
        if isinstance(node, (ast.Name, ast.Attribute)):
            if ast.unparse(node).endswith("FLASHINFER_CSRC_DIR"):
                return {""}
            if isinstance(node, ast.Name) and node.id in aliases:
                return set(aliases[node.id])
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in helpers:
                return set(helpers[node.func.id])
            return None
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            return None
        prefixes = components(node.left)
        if prefixes is None:
            return None
        tails = _values(node.right, bound)
        if not tails:
            unresolved.add(ast.unparse(node))
        # An unreadable tail makes the whole path unknown; keeping the prefix
        # would check a directory and call the file present.
        return {f"{p}/{t}" if p else t for p in prefixes for t in tails} or None

    # Three passes: a helper feeds an alias, which can feed another alias.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                returned = set()
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and inner.value is not None:
                        returned |= components(inner.value) or set()
                if returned:
                    helpers.setdefault(node.name, set()).update(returned)
                continue
            if not isinstance(node, ast.Assign):
                continue
            prefixes = components(node.value)
            if not prefixes:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.setdefault(target.id, set()).update(prefixes)

    names: set[str] = set()
    for node in ast.walk(tree):
        # Chained: CSRC_DIR / "fused_moe" / "hash_topk.cu". Only the outermost
        # node carries the whole path; the inner one names a directory that
        # exists whether or not the source does.
        for path in components(node) or ():
            if path:
                names.add(path)
    return names, unresolved


def _exempt(dotted: str) -> bool:
    """True when the module, or a package above it, is gated or shadowed."""
    parts = dotted.split(".")
    prefixes = {".".join(parts[: i + 1]) for i in range(len(parts))}
    return bool(
        prefixes & set(_registry.CUDA_ONLY_MODULES)
        or prefixes & set(_registry.SHADOW_MODULES)
    )


def _definition_sites() -> dict[str, set[Path]]:
    """gen_* name -> the file(s) under flashinfer/jit that define it.

    Resolving by definition rather than by the imported path is what makes
    re-exports work: `from .jit import gen_pod_module` names the package, and
    `from ..jit.gemm import ...` names a package directory, so a path built
    from the import alone points at no file in either case.
    """
    sites: dict[str, set[Path]] = {}
    # Package-wide, not just flashinfer/jit: moe_ep keeps its generators under
    # kernel_src/.../shim/, and a generator the index misses is a module the
    # guard never scores.
    for path in _PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Wrappers count: kda_prefill imports get_flash_kda_prefill_module,
            # which calls the generators beside it, so indexing only gen_*
            # leaves that whole source tree unscored.
            calls_generator = any(
                isinstance(inner, ast.Name) and inner.id.startswith("gen_")
                for inner in ast.walk(node)
            )
            if node.name.startswith("gen_") or calls_generator:
                sites.setdefault(node.name, set()).add(path)
    return sites


_SITES = _definition_sites()
# Files that define at least one generator, so an import of any name from one
# is an import of its kernel sources.
_GENERATOR_FILES = {
    path for name, paths in _SITES.items() if name.startswith("gen_") for path in paths
}


def _cuda_only_line_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans of `if IS_CUDA:` bodies -- code ROCm never executes.

    flashinfer/comm/__init__.py imports its generators there, so without this
    the guard scores a package that imports fine on ROCm against CUDA sources.
    """
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "IS_CUDA" in ast.unparse(node.test):
            body = node.body
            spans.append((body[0].lineno, body[-1].end_lineno or body[-1].lineno))
    return spans


def _gen_importers():
    """(importing module, generator file) for each `import gen_*` an op makes.

    Function-scope imports count -- concat_ops defers its generator, and a
    deferred import fails just as hard on the first call. Generators importing
    each other inside flashinfer.jit are skipped: they describe kernels rather
    than offering an op.
    """
    for path in sorted(_PKG.rglob("*.py")):
        dotted = _dotted(path)
        if dotted == "flashinfer.jit" or dotted.startswith("flashinfer.jit."):
            continue
        package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text())
        cuda_only = _cuda_only_line_ranges(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if any(lo <= node.lineno <= hi for lo, hi in cuda_only):
                continue
            target = _resolve(node.module, node.level, package)
            if not target.startswith("flashinfer."):
                continue
            # An import naming a generator module reaches that file's sources
            # whatever it pulls out of it: kda_prefill imports
            # get_flash_kda_prefill_module, a two-hop wrapper over the
            # generators beside it.
            for candidate in (
                _REPO_ROOT / (target.replace(".", "/") + ".py"),
                _REPO_ROOT / target.replace(".", "/") / "__init__.py",
            ):
                if candidate.exists() and candidate in _GENERATOR_FILES:
                    yield dotted, candidate
            for alias in node.names:
                sites = _SITES.get(alias.name, set())
                # An import naming a module outright means that file, even when
                # a ROCm twin defines the same generator.
                named = {p for p in sites if _dotted(p) == target}
                # Otherwise the import went through a package re-export, which
                # resolves to the ROCm definition at runtime:
                # flashinfer/jit/__init__.py star-imports .rocm.api on IS_HIP.
                # Scoring the CUDA twin would fail every supported op.
                rocm = {p for p in sites if _JIT_ROCM in p.parents}
                for jit_file in named or rocm or sites:
                    yield dotted, jit_file


_CASES = sorted({(dotted, str(jit)) for dotted, jit in _gen_importers()})


@pytest.mark.parametrize("dotted,jit_file", _CASES, ids=[c[0] for c in _CASES])
def test_kernel_sources_present_or_module_classified(dotted, jit_file):
    if _exempt(dotted) or (dotted, Path(jit_file).name) in _UNPORTED:
        pytest.skip(f"{dotted} is gated, shadowed, or a known unported op")
    names, unresolved = _csrc_names(Path(jit_file))
    missing = sorted(n for n in names if not (_CSRC / n).exists())
    assert not missing, (
        f"{dotted} builds kernels that do not exist under csrc/rocm: "
        f"{', '.join(missing)}. Port them, add the module to CUDA_ONLY_MODULES "
        f"in flashinfer/rocm/__init__.py, or record it in _UNPORTED here."
    )
    # Fail closed: a rooted path this analysis cannot read is exactly how a
    # newly vendored generator would slip through unexamined.
    assert not unresolved, (
        f"{dotted} builds kernel paths this test cannot read: "
        f"{', '.join(sorted(unresolved))}. Teach _values() the expression, or "
        f"classify the module as gated or _UNPORTED."
    )


def test_extractor_reads_the_shapes_the_generators_actually_use(tmp_path):
    """Each clause here is a shape that once slipped past the extractor."""
    source = tmp_path / "gen.py"
    source.write_text(
        "from . import env as jit_env\n"
        "def _body(prefix):\n"
        "    csrc = jit_env.FLASHINFER_CSRC_DIR\n"
        '    open(csrc / f"{prefix}_customize_config.jinja")\n'
        '    for filename in [f"{prefix}.cu", "plain.cu"]:\n'
        "        open(csrc / filename)\n"
        '    open(jit_env.FLASHINFER_CSRC_DIR / "nested" / "deep.cu")\n'
        '    subdir = jit_env.FLASHINFER_CSRC_DIR / "sub"\n'
        '    open(subdir / "under_alias.cu")\n'
        '    open(_root() / "via_helper.cu")\n'
        "    open(jit_env.FLASHINFER_CSRC_DIR / unknowable)\n"
        "def _root():\n"
        '    return jit_env.FLASHINFER_CSRC_DIR / "helper"\n'
        "def gen_x():\n"
        '    return _body("pod")\n'
    )
    names, unresolved = _csrc_names(source)
    assert unresolved == {"jit_env.FLASHINFER_CSRC_DIR / unknowable"}
    assert names == {
        "pod_customize_config.jinja",  # f-string over a call-site argument
        "pod.cu",  # f-string inside a loop iterable
        "plain.cu",  # plain literal in the same iterable
        "nested/deep.cu",  # chained, not just the "nested" prefix
        "nested",  # the inner node of that chain, harmlessly
        "sub/under_alias.cu",  # alias bound to a subdirectory, not the root
        "sub",
        "helper/via_helper.cu",  # root returned by a local helper call
        "helper",
    }


def test_pod_kernel_sources_are_resolved_and_present():
    """POD is supported on ROCm and names every source through an f-string.

    A regression in the extractor shows up here as an empty set rather than as
    a failure somewhere else, which is what makes the guard's silence safe.
    """
    names, _ = _csrc_names(_PKG / "jit" / "rocm" / "modules.py")
    expected = {
        f"{p}{s}" for p in ("pod", "batch_pod") for s in (".cu", "_jit_pybind.cu")
    }
    assert expected <= names
    assert all((_CSRC / name).exists() for name in expected)


def test_unported_allowlist_has_no_stale_entries():
    """An entry whose sources all landed, or that is now gated, must go.

    Presence in _CASES is not the test -- a ported module stays in _CASES
    forever. What retires an entry is having nothing left to miss.
    """
    covered = {(dotted, Path(jit).name) for dotted, jit in _CASES}
    unproven = set()
    for dotted, jit_file in _CASES:
        sources, unreadable = _csrc_names(Path(jit_file))
        # No literal to read -- nvfp4_attention_sm120 takes its filename as a
        # parameter -- is unknown, not ported. Retiring on that would drop the
        # entry for a kernel that is still missing.
        if unreadable or not sources or any(not (_CSRC / n).exists() for n in sources):
            unproven.add((dotted, Path(jit_file).name))
    stale = sorted(
        f"{module} ({generator})"
        for module, generator in _UNPORTED
        if (module, generator) not in covered
        or _exempt(module)
        or (module, generator) not in unproven
    )
    assert not stale, f"remove from _UNPORTED: {', '.join(stale)}"


def test_api_parity_covers_every_shadowed_module():
    """A shadow with no PAIRS entry is a twin whose signatures nothing audits."""
    pairs = ast.literal_eval(
        next(
            ast.unparse(node.value)
            for node in ast.parse(
                (_REPO_ROOT / "scripts" / "rocm_api_parity.py").read_text()
            ).body
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "PAIRS"
        )
    )
    audited = {rocm_path for rocm_path, _ in pairs}
    missing = sorted(
        rocm
        for rocm in _registry.SHADOW_MODULES.values()
        if rocm.replace(".", "/") + ".py" not in audited
        and (rocm.replace(".", "/") + "/_core.py") not in audited
    )
    assert not missing, (
        f"shadowed but absent from scripts/rocm_api_parity.py PAIRS: "
        f"{', '.join(missing)}"
    )
