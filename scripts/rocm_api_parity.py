#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Report where the ROCm twins diverge from the upstream modules they shadow.

``flashinfer/rocm/{decode,prefill,mla}.py`` replace their upstream namesakes at
import (see ``flashinfer/rocm/__init__.py``), so a caller written against the
CUDA API reaches them unchanged. Upstream syncs never touch these files, so
every added parameter is a silent gap until someone runs a diff by hand.

Usage::

    python3 scripts/rocm_api_parity.py            # human-readable report
    python3 scripts/rocm_api_parity.py --json     # machine-readable

Exit 0 clean, 1 on unallowlisted divergence, 2 if the tool could not run.
Static only -- no torch, no GPU, no git.

Out of the guard's reach: ``flashinfer/mla/__init__.py``'s star re-export and
its ``__getattr__`` lazy PrimTS names (the shadow replaces the whole package on
ROCm, so those are unreachable there anyway), and ``@overload`` stubs -- only
the binding implementation is compared.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# The parameter model is upstream's, from the release API-diff tooling: it keeps
# positional-only / positional-or-keyword / keyword-only apart, which is what
# makes a mid-signature insertion visible rather than looking like a rename.
from check_pr_api_diff import ApiParameter, parameters  # noqa: E402

EXIT_OK, EXIT_DIVERGED, EXIT_ERROR = 0, 1, 2

# ROCm twin -> the upstream module it shadows.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("flashinfer/rocm/decode.py", "flashinfer/decode.py"),
    ("flashinfer/rocm/prefill.py", "flashinfer/prefill.py"),
    ("flashinfer/rocm/mla.py", "flashinfer/mla/_core.py"),
)

# Upstream symbols with no ROCm analogue. Substring match on the qualified name;
# every entry states why, because that reason is the whole record of the choice.
CUDA_ONLY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("trtllm", "TensorRT-LLM generated kernels; NVIDIA-only"),
    ("xqa", "XQA kernels; NVIDIA-only"),
    ("fmha_v2", "FMHA v2 cubins; NVIDIA-only"),
    ("cute_dsl", "CuTe DSL backend; NVIDIA-only"),
    ("CuteDsl", "CuTe DSL backend; NVIDIA-only"),
    ("sm120", "SM120-specific path; NVIDIA-only"),
    ("nvfp4", "NVFP4 quantisation; NVIDIA-only"),
    ("cudnn", "cuDNN backend; NVIDIA-only"),
    ("sage_attention", "SageAttention quantisation; NVIDIA-only"),
)

# Upstream symbols no name pattern catches. Same rule: a reason per entry.
CUDA_ONLY_SYMBOLS: Dict[str, Dict[str, str]] = {
    "flashinfer/decode.py": {
        "get_batch_decode_mla_module": "JIT generator for the CUDA MLA decode kernel",
        "BatchDecodeMlaWithPagedKVCacheWrapper": "wraps the CUDA MLA decode kernel",
        "fast_decode_plan": "CUDAGraph plan fast-path bound to the CUDA scheduler",
    },
    "flashinfer/prefill.py": {
        "get_fmha_module": "JIT generator for the CUDA FMHA kernel",
        "fmha_varlen": "CUDA FMHA varlen entry point",
        "fmha_varlen_plan": "CUDA FMHA varlen entry point",
    },
    "flashinfer/mla/_core.py": {
        "get_mla_module": "JIT generator for the CUDA MLA kernel",
        "get_batch_mla_module": "JIT generator for the CUDA MLA kernel",
        "MLAHeadDimensions": "descriptor consumed only by CUDA MLA runners",
        "MLALayerDimensions": "descriptor consumed only by CUDA MLA runners",
        "DSV4HCAMetadata": "DSV4 sparse-MLA metadata; NVIDIA-only",
        "convert_compressed_page_aligned_sparse_indices_to_hca_metadata": (
            "DSV4 sparse-MLA metadata; NVIDIA-only"
        ),
    },
}

# Parameters the ROCm twin declares -- so a caller's kwarg binds -- but refuses
# rather than honours. They must still be *present*, so this is not an exemption
# from the signature check: it is the record of which arguments raise, and the
# table the runtime test enumerates. Anything absent from here is expected to
# work; q/k/v_scale are the notable case, folded into sm_scale and the output.
CUDA_ONLY_PARAMS: Dict[str, str] = {
    "kv_cache_sf": "NVFP4 KV-cache scale factors",
    "skip_softmax_threshold_scale_factor": "trtllm-gen skip-softmax",
    "use_fp16_softmax": "trtllm-gen SM107 feature",
    "uses_spcompress": "trtllm-gen SM107 feature",
    "fixed_split_size": "CUDA split-KV scheduler knob; no slot in the ROCm plan binding",
    "disable_split_kv": "CUDA split-KV scheduler knob; no slot in the ROCm plan binding",
    "o_scale": "fp8 output calibration; ROCm attention has no fp8 output path",
    "ckv_scale": "fp8 MLA scaling; AITER MLA takes no scale",
    "ckv_scale_arr": "fp8 MLA scaling; AITER MLA takes no scale",
    "kpe_scale": "fp8 MLA scaling; AITER MLA takes no scale",
    "profiler_buffer": "CUDA in-kernel profiler",
    "return_lse_base_on_e": "CUDA MLA kernel variant",
    "window_right": "cute-dsl backend only",
    "v_indptr": "CUDA ragged-prefill scheduler input",
    "o_indptr": "CUDA ragged-prefill scheduler input",
}

# Methods a ROCm twin may leave out. Keyed "<upstream path>::<Class>.<method>".
CUDA_ONLY_METHODS: Dict[str, str] = {}


@dataclass(frozen=True)
class Sig:
    """A function's parameters, order preserved."""

    name: str
    positional: Tuple[ApiParameter, ...]
    vararg: Optional[ApiParameter]
    keyword_only: Tuple[ApiParameter, ...]
    kwarg: Optional[ApiParameter]
    line: int

    @property
    def all_names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.positional + self.keyword_only)


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str
    where: str
    detail: str


class ToolError(Exception):
    """The tool could not produce a report -- distinct from a divergence."""


# Private, but the real signature: upstream's public ``plan`` is a thin
# ``*args/**kwargs`` deprecation shim, so comparing only ``plan`` would compare
# nothing once the ROCm twin mirrors that shape.
PRIVATE_COMPARE = frozenset({"_plan_impl"})


def _is_public(name: str) -> bool:
    if name in PRIVATE_COMPARE:
        return True
    return not name.startswith("_") or name in ("__init__", "__call__")


def _implementation(nodes: Sequence[ast.FunctionDef]) -> ast.FunctionDef:
    """Pick the binding definition from a `@overload` group.

    The overload stubs never run; the last undecorated definition is what a
    caller actually binds against, so it is the only one worth comparing.
    """
    real = [n for n in nodes if not _has_decorator(n, "overload")]
    return (real or list(nodes))[-1]


def _has_decorator(node: ast.FunctionDef, name: str) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == name:
            return True
        if isinstance(target, ast.Attribute) and target.attr == name:
            return True
    return False


def _signature(node: ast.FunctionDef, qualname: str) -> Sig:
    posonly, poskw, vararg, kwonly, kwarg = parameters(node)
    return Sig(
        name=qualname,
        positional=posonly + poskw,
        vararg=vararg,
        keyword_only=kwonly,
        kwarg=kwarg,
        line=node.lineno,
    )


def _collect(path: Path) -> Tuple[Dict[str, Sig], Dict[str, None]]:
    """Return the module's public callables by qualified name, plus its classes."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ToolError(f"cannot parse {path}: {exc}") from exc

    sigs: Dict[str, Sig] = {}
    classes: Dict[str, None] = {}

    def group(body: Iterable[ast.stmt], prefix: str) -> None:
        by_name: Dict[str, List[ast.FunctionDef]] = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
                node.name
            ):
                by_name.setdefault(node.name, []).append(node)  # type: ignore[arg-type]
        for name, nodes in by_name.items():
            impl = _implementation(nodes)
            sigs[f"{prefix}{name}"] = _signature(impl, f"{prefix}{name}")

    group(tree.body, "")
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            classes[node.name] = None
            group(node.body, f"{node.name}.")
    return sigs, classes


def _allowlisted(upstream_path: str, qualname: str) -> Optional[str]:
    symbol = qualname.split(".", 1)[0]
    for entry in (qualname, symbol):
        reason = CUDA_ONLY_SYMBOLS.get(upstream_path, {}).get(entry)
        if reason:
            return reason
    reason = CUDA_ONLY_METHODS.get(f"{upstream_path}::{qualname}")
    if reason:
        return reason
    lowered = qualname.lower()
    for pattern, why in CUDA_ONLY_PATTERNS:
        if pattern.lower() in lowered:
            return why
    return None


def _compare_signature(up: Sig, ro: Sig, where: str) -> List[Finding]:
    findings: List[Finding] = []

    # Order first: a parameter that binds to a different name than upstream is
    # worse than a missing one, because the call succeeds and does the wrong
    # thing. Compare the shared positional prefix name-for-name.
    up_pos = [p.name for p in up.positional]
    ro_pos = [p.name for p in ro.positional]
    for index, (a, b) in enumerate(zip(up_pos, ro_pos, strict=False)):
        if a != b:
            findings.append(
                Finding(
                    "misbind",
                    "error",
                    where,
                    f"positional #{index} is {a!r} upstream but {b!r} on ROCm "
                    f"(ROCm line {ro.line})",
                )
            )
            break

    # ROCm accepting *args/**kwargs only counts where upstream does too;
    # otherwise a swallowed kwarg is a silent no-op, not parity.
    absorbs = ro.kwarg is not None and up.kwarg is not None
    missing = [
        n
        for n in up.all_names
        if n not in ro.all_names and not (absorbs and n != "self")
    ]
    if missing:
        findings.append(
            Finding(
                "missing-param",
                "error",
                where,
                f"ROCm does not accept {', '.join(sorted(missing))}",
            )
        )

    # A ROCm-only parameter is fine appended or keyword-only; inserted among
    # upstream's positionals it shifts every later argument.
    inserted = [
        p.name
        for index, p in enumerate(ro.positional)
        if p.name not in up.all_names and index < len(up.positional)
    ]
    if inserted:
        findings.append(
            Finding(
                "inserted-extra",
                "error",
                where,
                f"ROCm-only positional(s) {', '.join(inserted)} sit inside "
                "upstream's positional range",
            )
        )

    # A default that drifts is as silent as a mis-bind: every caller that omits
    # the argument gets the other behaviour, and no signature name changes.
    ro_by_name = {p.name: p for p in ro.positional + ro.keyword_only}
    ro_kind = {p.name: "positional" for p in ro.positional}
    ro_kind.update({p.name: "keyword-only" for p in ro.keyword_only})
    for kind, group in (
        ("positional", up.positional),
        ("keyword-only", up.keyword_only),
    ):
        for param in group:
            mine = ro_by_name.get(param.name)
            if mine is None:
                continue
            if mine.default != param.default:
                findings.append(
                    Finding(
                        "default-drift",
                        "error",
                        where,
                        f"{param.name} defaults to {param.default} upstream but "
                        f"{mine.default} on ROCm",
                    )
                )
            if ro_kind[param.name] != kind:
                findings.append(
                    Finding(
                        "kind-drift",
                        "error",
                        where,
                        f"{param.name} is {kind} upstream but "
                        f"{ro_kind[param.name]} on ROCm",
                    )
                )

    if up.vararg is not None and ro.vararg is None:
        findings.append(
            Finding(
                "missing-param",
                "error",
                where,
                f"ROCm drops upstream's *{up.vararg.name}",
            )
        )
    if up.kwarg is not None and ro.kwarg is None:
        findings.append(
            Finding(
                "missing-param",
                "error",
                where,
                f"ROCm drops upstream's **{up.kwarg.name}",
            )
        )
    return findings


def _legacy_positional_tuple(path: Path, name: str) -> Optional[Tuple[str, ...]]:
    """Read a module-level tuple-of-strings constant, or None if absent."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ToolError(f"cannot parse {path}: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            return None
        if isinstance(value, tuple) and all(isinstance(x, str) for x in value):
            return value
    return None


def audit(root: Path) -> Tuple[List[Finding], List[str]]:
    findings: List[Finding] = []
    used_allowlist: List[str] = []

    for rocm_rel, up_rel in PAIRS:
        rocm_sigs, rocm_classes = _collect(root / rocm_rel)
        up_sigs, up_classes = _collect(root / up_rel)

        # Classes with no public methods (upstream's dataclass descriptors) have
        # no entry in *_sigs at all, so presence is checked separately.
        for cls in up_classes:
            reason = _allowlisted(up_rel, cls)
            if cls in rocm_classes:
                continue
            if reason:
                used_allowlist.append(f"{up_rel}::{cls}")
            else:
                findings.append(
                    Finding(
                        "missing-symbol",
                        "error",
                        f"{rocm_rel}::{cls}",
                        f"class present upstream ({up_rel}), absent on ROCm",
                    )
                )

        for qualname, up_sig in up_sigs.items():
            reason = _allowlisted(up_rel, qualname)
            if qualname not in rocm_sigs:
                if reason:
                    used_allowlist.append(f"{up_rel}::{qualname}")
                else:
                    findings.append(
                        Finding(
                            "missing-symbol",
                            "error",
                            f"{rocm_rel}::{qualname}",
                            f"present upstream ({up_rel}:{up_sig.line}), absent on ROCm",
                        )
                    )
                continue
            if reason:
                used_allowlist.append(f"{up_rel}::{qualname}")
            findings.extend(
                _compare_signature(
                    up_sig, rocm_sigs[qualname], f"{rocm_rel}::{qualname}"
                )
            )

    # The ROCm decode twin copies upstream's legacy-positional tuple because
    # flashinfer/decode.py is unimportable under HIP. Nothing else would notice
    # the copy going stale.
    const = "_BATCH_DECODE_PLAN_LEGACY_POS_ARGS"
    up_tuple = _legacy_positional_tuple(root / "flashinfer/decode.py", const)
    ro_tuple = _legacy_positional_tuple(root / "flashinfer/rocm/decode.py", const)
    if up_tuple is None:
        raise ToolError(f"{const} not found in flashinfer/decode.py")
    if ro_tuple is None:
        findings.append(
            Finding(
                "missing-symbol",
                "error",
                f"flashinfer/rocm/decode.py::{const}",
                "the copied legacy-positional tuple is missing",
            )
        )
    elif ro_tuple != up_tuple:
        findings.append(
            Finding(
                "stale-copy",
                "error",
                f"flashinfer/rocm/decode.py::{const}",
                f"copy has drifted: upstream {up_tuple}, ROCm {ro_tuple}",
            )
        )

    return findings, used_allowlist


def stale_allowlist_entries(root: Path) -> List[str]:
    """Allowlisted names upstream no longer defines -- dead entries to delete."""
    stale: List[str] = []
    for _, up_rel in PAIRS:
        up_sigs, up_classes = _collect(root / up_rel)
        known = set(up_sigs) | set(up_classes)
        for symbol in CUDA_ONLY_SYMBOLS.get(up_rel, {}):
            if symbol not in known:
                stale.append(f"{up_rel}::{symbol}")
    for key in CUDA_ONLY_METHODS:
        up_rel, qualname = key.split("::", 1)
        up_sigs, _ = _collect(root / up_rel)
        if qualname not in up_sigs:
            stale.append(key)

    # A refusal entry for a parameter no ROCm twin declares any more is dead: the
    # runtime test would stop covering it and nothing would say so.
    declared: Set[str] = set()
    for rocm_rel, _ in PAIRS:
        for sig in _collect(root / rocm_rel)[0].values():
            declared.update(sig.all_names)
    for param in CUDA_ONLY_PARAMS:
        if param not in declared:
            stale.append(f"CUDA_ONLY_PARAMS::{param}")
    return stale


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--root", type=Path, default=_REPO_ROOT, help="repository root to audit"
    )
    args = parser.parse_args(argv)

    try:
        findings, _ = audit(args.root)
        stale = stale_allowlist_entries(args.root)
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    for entry in stale:
        findings.append(
            Finding(
                "stale-allowlist",
                "error",
                entry,
                "allowlisted but upstream no longer defines it",
            )
        )

    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    elif not findings:
        print("ROCm API parity: clean")
    else:
        for f in findings:
            print(f"{f.severity}: [{f.kind}] {f.where}\n    {f.detail}")
        print(f"\n{len(findings)} divergence(s)")

    return EXIT_DIVERGED if findings else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
