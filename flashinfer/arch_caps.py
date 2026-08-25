# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-architecture capability knowledge for the ROCm backends.

This module owns *which architectures are validated for which operations*.
That is deliberately distinct from :mod:`flashinfer.hip_utils`, which owns
*which architectures we compile for*, and from :mod:`flashinfer.aiter_utils`,
which owns *whether the AITER package is importable*.

.. important::
   Do not add a module-level ``import torch``; keep every torch import
   function-local.

   ``hip_utils`` imports this module at module scope, and ``hip_utils`` is used
   very early -- ``tests/conftest.py`` calls it to pick a GPU *before* pinning
   ``HIP_VISIBLE_DEVICES``, and it must do so without touching the HIP runtime
   (hence its use of ``rocminfo`` rather than ``torch.cuda``). Staying torch-free
   is what keeps this module usable on that path.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

__all__ = [
    "ArchCapabilityError",
    "ArchSupport",
    "CAPABILITIES",
    "Capability",
    "KnownBad",
    "Support",
    "capability_available",
    "capability_reason",
    "normalize_arch",
    "require_capability",
]


def normalize_arch(gcn_arch_name: str) -> str:
    """Strip ROCm feature qualifiers from a GPU architecture name.

    ROCm reports architectures with trailing feature flags, e.g. torch's
    ``gcnArchName`` yields ``"gfx942:sramecc+:xnack-"``. Comparisons against
    :data:`~flashinfer.hip_utils.FLASHINFER_SUPPORTED_ROCM_ARCHS` need the bare
    ``"gfx942"``.

    This is the single place that transformation happens. It replaces several
    open-coded variants that did not agree; notably a ``re.match(r"(gfx\\d+)")``
    form that truncated letter-suffixed architectures (``gfx90a`` -> ``gfx90``),
    naming an architecture that does not exist.

    Args:
        gcn_arch_name: An architecture name, with or without qualifiers.

    Returns:
        The architecture without qualifiers, e.g. ``"gfx942"``. Input that
        contains no qualifier is returned unchanged (modulo surrounding
        whitespace), so this is safe to apply to already-normalized values.
    """
    return gcn_arch_name.split(":", 1)[0].strip()


class ArchCapabilityError(ValueError, RuntimeError):
    """Raised when an op is not usable on the running GPU architecture.

    Both bases keep an existing caller working, so routing the three divergent
    ROCm arch checks through one type churns no call sites:

    - ``ValueError`` -- ``aiter_utils.require_aiter`` raised ``ValueError``, and
      ``tests/rocm_tests/test_activation_aiter_hip.py:67`` asserts on it.
    - ``RuntimeError`` -- ``prefill_rocm`` and ``mla_rocm`` raised
      ``RuntimeError``, and
      ``tests/rocm_tests/test_batch_prefill_bf16_custom_mask_hip.py:157``
      catches it.

    Deliberately *not* an ``ImportError``: a missing ``aiter`` package is a
    different condition from an unsupported architecture, and the import probes
    that raise ``ImportError`` are left alone.

    Deliberately *not* a :class:`flashinfer.utils.GPUArchitectureError` either,
    though that would be the tidier hierarchy. That class lives in
    ``flashinfer/utils.py``, which imports torch at module scope, and this module
    must stay torch-free -- ``hip_utils`` imports it at module scope, and
    ``tests/conftest.py`` uses ``hip_utils`` to choose a GPU *before* pinning
    ``HIP_VISIBLE_DEVICES``. Inheriting would drag torch onto that path and
    silently defeat the pinning. Re-declaring the base locally was considered and
    rejected: a same-named copy is a different class, so ``except
    GPUArchitectureError`` would not catch it -- worse than not claiming the
    relationship at all. No ROCm path raised ``GPUArchitectureError`` before this
    change, so nothing regresses; ``skip_on_gpu_arch_error`` remains CUDA-only.
    """


class Support(Enum):
    """Whether an (op, backend) pair may run on an architecture at all."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


def _version_tuple(text: str) -> Tuple[int, ...]:
    """``"7.2.4"`` -> ``(7, 2, 4)``; non-numeric trailing parts are dropped."""
    parts = []
    for chunk in text.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _compare(left: str, right: str) -> int:
    """Three-way compare two version strings, zero-padding absent components.

    ``"7.2"`` and ``"7.2.0"`` name the same release and must compare equal.
    Comparing raw tuples would make ``(7, 2) < (7, 2, 0)``, so a window written
    as ``rocm_min="7.2.0"`` would silently fail to gate a machine reporting
    ``"7.2"`` -- a reachable state, not a hypothetical:
    ``hip_utils.get_system_rocm_version_from_hipconfig`` matches
    ``\\d+\\.\\d+(?:\\.\\d+)?``, so the patch component is optional, and on
    TheRock builds it is the *only* detection method consulted.

    The current table writes ``rocm_min="7.2"`` and is unaffected either way;
    this keeps the next window from having to know about the quirk.
    """
    a, b = _version_tuple(left), _version_tuple(right)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    return (a > b) - (a < b)


@dataclass(frozen=True)
class KnownBad:
    """A toolchain window in which an otherwise-supported op is broken.

    Support is not purely a property of ``(op, backend, arch)``: the one CDNA4
    defect we have is a *compiler* bug, correct on ROCm 7.1 and wrong on 7.2.x
    with everything else held constant. Bounds are literal version strings
    rather than a predicate so the window is inspectable, renderable into docs,
    and testable without a GPU.

    Bounds are half-open: ``rocm_min`` inclusive, ``rocm_max`` exclusive.
    """

    rocm_min: Optional[str] = None
    rocm_max: Optional[str] = None
    aiter_min: Optional[str] = None
    aiter_max: Optional[str] = None
    detail: str = ""
    url: str = ""

    def matches(self, rocm: Optional[str], aiter: Optional[str]) -> bool:
        """True when the live toolchain falls inside this window.

        An unknown version does **not** match: refusing to route on a version we
        could not read would break machines that are probably fine, and the
        failure mode we are guarding against is already loud in the header.
        """
        for value, low, high in (
            (rocm, self.rocm_min, self.rocm_max),
            (aiter, self.aiter_min, self.aiter_max),
        ):
            if low is None and high is None:
                continue
            if value is None:
                return False
            if low is not None and _compare(value, low) < 0:
                return False
            if high is not None and _compare(value, high) >= 0:
                return False
        return True


@dataclass(frozen=True)
class ArchSupport:
    """How one ``(op, backend)`` behaves on one architecture.

    ``evidence`` records *what was actually run* -- board, ROCm, AITER, date --
    rather than a bare "validated" flag. An empty string means the row is a
    declaration nobody has measured, which is a fact worth being able to render.
    """

    support: Support
    evidence: str = ""
    caveat: str = ""
    known_bad: Tuple[KnownBad, ...] = ()


@dataclass(frozen=True)
class Capability:
    """One ``(op, backend)`` row of the table.

    ``frozen=True`` only stops the *fields* being rebound; a plain dict in
    ``archs`` would still let a caller edit the global table in place
    (``CAPABILITIES[0].archs["gfx942"] = ...``). ``__post_init__`` wraps it in a
    :class:`~types.MappingProxyType` so the whole structure is read-only:
    ``ArchSupport`` and ``KnownBad`` are frozen with tuple fields, so nothing
    below this point is mutable either.

    The coercion lives here rather than in the construction helper so it holds
    for every row however it was built.
    """

    op: str
    backend: str
    archs: Mapping[str, ArchSupport] = field(default_factory=dict)
    note: str = ""
    # The public ``backend=`` string to suggest when this row is unavailable.
    # Declared per row because it is not derivable: the table keys backends as
    # "aiter"/"hip", but the user-facing argument spells the HIP path "native"
    # for rope/norm/activation/page-append and "fa2" for prefill and decode.
    # Empty means no alternative exists -- ``mla`` accepts only 'auto'/'aiter',
    # so suggesting anything there would send the user into a ValueError.
    fallback: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "archs", MappingProxyType(dict(self.archs)))


# --------------------------------------------------------------------------
# The table.
#
# Keyed by (op, backend) because every call site already passes both halves
# separately -- `require_aiter(input.device, "rmsnorm")` (norm.py:116) -- so
# routing them through here changes no signatures.
#
# `evidence` is only filled in where a suite was actually run on that board. An
# empty string means "declared, nobody measured it", which is a distinct and
# useful thing for the docs to be able to say. The AITER rows carry evidence
# from the runs below; the HIP rows deliberately do not yet.
#
#   gfx950  MI350X   27517 passed / 3585 skipped / 3 failed   (49a8cdd8)
#   gfx942  MI300X   27520 passed / 3585 skipped / 0 failed   (49a8cdd8)
#
# both on torch 2.9.1+rocm7.2.0, HIP 7.2.26015, amd-aiter 0.1.10.
# --------------------------------------------------------------------------

_MEASURED_950 = "MI350X / rocm 7.2.0 / aiter 0.1.10 / torch 2.9.1 / 2026-08-19"
_MEASURED_942 = "MI300X / rocm 7.2.0 / aiter 0.1.10 / torch 2.9.1 / 2026-08-19"

# Separate from the suite strings above, which predate the MLA tests.
_MEASURED_950_MLA = (
    "mla: MI350X / rocm 7.2.0 / aiter 0.1.10 / torch 2.9.1 / 2026-08-24 "
    "(decode + prefill, heads 16/128)"
)

# The one defect measured so far. Upstream calls it a compiler bug and gates
# their own test skips on exactly (major, minor) == (7, 2) with gfx950
# (aiter op_tests/test_batch_prefill.py::should_skip_rocm72_issue, still present
# at v0.1.20 -- so no amd-aiter upgrade avoids it).
#
# Confirmed here rather than taken on trust, torch and aiter held constant:
#   ROCm 7.2.0  max_abs_err vs fp32 reference 0.595268   3/12 tests fail
#   ROCm 7.2.4  max_abs_err vs fp32 reference 0.595268   (bit-identical)
#   ROCm 7.1    max_abs_err vs fp32 reference 0.000250   12/12 tests pass
_ROCM72_CAUSAL_PREFILL = KnownBad(
    rocm_min="7.2",
    rocm_max="7.3",
    detail=(
        "ROCm 7.2.x miscompiles AITER's causal batch-prefill kernel on gfx950: "
        "causal=True with logits_soft_cap=0.0 returns wrong numbers (not an "
        "error), 97.6% of elements off. Use ROCm 7.1, or backend='fa2'"
    ),
    url="https://github.com/ROCm/aiter/blob/main/op_tests/test_batch_prefill.py",
)


def _archs(gfx942: ArchSupport, gfx950: ArchSupport) -> Mapping[str, ArchSupport]:
    """Positional shorthand for the two architectures every row must declare."""
    return {"gfx942": gfx942, "gfx950": gfx950}


_OK_942 = ArchSupport(Support.SUPPORTED, evidence=_MEASURED_942)
_OK_950 = ArchSupport(Support.SUPPORTED, evidence=_MEASURED_950)
# Fused MoE has its own runs, so they name themselves: the README lists evidence
# per architecture, and two bare gfx950 strings differing only by date are not
# attributable to an op.
_OK_942_MOE = ArchSupport(
    Support.SUPPORTED,
    evidence="fused_moe: MI300X / rocm 7.2.0 / aiter 0.1.10 / torch 2.9.1 / 2026-08-24",
)
_OK_950_MOE = ArchSupport(
    Support.SUPPORTED,
    evidence="fused_moe: MI350X / rocm 7.2.0 / aiter 0.1.10 / torch 2.9.1 / 2026-08-24",
)
# HIP rows: declared, not yet individually attributed. The suites above cover
# them, but no per-op HIP measurement has been recorded, so the evidence field
# stays empty rather than borrowing the AITER runs' credibility.
_HIP_942 = ArchSupport(Support.SUPPORTED)
_HIP_950 = ArchSupport(Support.SUPPORTED)

CAPABILITIES: Tuple[Capability, ...] = (
    # --- AITER backends: measured on both architectures --------------------
    Capability("batch_decode", "aiter", _archs(_OK_942, _OK_950), fallback="fa2"),
    Capability("single_prefill", "aiter", _archs(_OK_942, _OK_950), fallback="fa2"),
    Capability(
        "batch_prefill",
        "aiter",
        _archs(
            _OK_942,
            ArchSupport(
                Support.SUPPORTED,
                evidence=_MEASURED_950,
                caveat=("causal=True is miscompiled on ROCm 7.2.x; correct on 7.1"),
                known_bad=(_ROCM72_CAUSAL_PREFILL,),
            ),
        ),
        fallback="fa2",
    ),
    # No alternative backend -- hence no fallback=.
    Capability(
        "mla",
        "aiter",
        _archs(_OK_942, ArchSupport(Support.SUPPORTED, evidence=_MEASURED_950_MLA)),
    ),
    Capability("rope", "aiter", _archs(_OK_942, _OK_950), fallback="native"),
    Capability(
        "append_paged_kv_cache", "aiter", _archs(_OK_942, _OK_950), fallback="native"
    ),
    Capability("rmsnorm", "aiter", _archs(_OK_942, _OK_950), fallback="native"),
    Capability(
        "fused_add_rmsnorm", "aiter", _archs(_OK_942, _OK_950), fallback="native"
    ),
    Capability("silu_and_mul", "aiter", _archs(_OK_942, _OK_950), fallback="native"),
    # No HIP MoE kernel, so no fallback.
    Capability("fused_moe", "aiter", _archs(_OK_942_MOE, _OK_950_MOE)),
    # --- HIP backends: declared; per-op evidence not yet recorded ----------
    Capability("single_decode", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("batch_decode", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("single_prefill", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("batch_prefill", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("cascade", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("pod", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("rope", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("append_paged_kv_cache", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("rmsnorm", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("fused_add_rmsnorm", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("layernorm", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("sampling", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("logits_processor", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("silu_and_mul", "hip", _archs(_HIP_942, _HIP_950)),
    Capability("quantization", "hip", _archs(_HIP_942, _HIP_950)),
)


def _index(caps: Tuple[Capability, ...]) -> Mapping[Tuple[str, str], Capability]:
    """Index rows by ``(op, backend)``, refusing duplicates.

    A dict comprehension lets the later of two contradictory rows win silently,
    which is exactly the unearned claim this table exists to remove.
    ``test_keys_are_unique`` covers it, but raising at import puts the error next
    to the edit that caused it instead of in a later CI run.
    """
    index = {}
    for cap in caps:
        key = (cap.op, cap.backend)
        if key in index:
            raise ValueError(f"duplicate capability row for {key} in CAPABILITIES")
        index[key] = cap
    return MappingProxyType(index)


_BY_KEY = _index(CAPABILITIES)


@lru_cache(maxsize=1)
def _live_versions() -> Tuple[Optional[str], Optional[str]]:
    """``(rocm_version, aiter_version)``, either ``None`` when undetectable.

    Cached for the process lifetime because it is not cheap:
    ``get_system_rocm_version`` walks up to four detection methods, three of
    which shell out (``amd-smi``, ``dpkg``, ``hipconfig``) with timeouts. Neither
    version can change under a running process, so probing once is sufficient
    even though ``_blocking_reason`` may be called per routing decision.
    """
    rocm = aiter = None
    try:
        import contextlib
        import io

        from .hip_utils import get_system_rocm_version

        with contextlib.redirect_stdout(io.StringIO()):
            rocm = get_system_rocm_version()
    except Exception:
        pass
    try:
        import importlib.metadata as _md

        aiter = _md.version("amd-aiter")
    except Exception:
        pass
    return rocm, aiter


def _lookup(op: str, backend: str, arch: str) -> Optional[ArchSupport]:
    cap = _BY_KEY.get((op, backend))
    if cap is None:
        return None
    return cap.archs.get(arch)


def _blocking_reason(op: str, backend: str, arch: str) -> Optional[str]:
    """Why this combination must not run, or ``None`` if it may.

    An architecture absent from a row resolves to unsupported: adding an entry
    to ``FLASHINFER_SUPPORTED_ROCM_ARCHS`` then grants nothing until someone
    declares it here, which closes the "any new arch is AITER-supported for
    free" hole.
    """
    entry = _lookup(op, backend, arch)
    if entry is None:
        cap = _BY_KEY.get((op, backend))
        if cap is None:
            return f"{backend} {op} is not a declared capability"
        # Name the architectures that would work. The common way to land here is
        # a CPU tensor or a non-ROCm device, where "not declared for unknown"
        # would be accurate and useless.
        supported = "/".join(sorted(cap.archs))
        msg = f"{backend} {op} requires an AMD {supported} device; got {arch!r}"
        # Only suggest a fallback the op actually accepts. A blanket
        # "use backend='native'" is wrong for prefill and decode (they take
        # 'fa2') and impossible for mla, which accepts only 'auto'/'aiter' --
        # following it would trade this error for an "Unknown backend" one.
        if cap.fallback:
            msg += f". Use backend={cap.fallback!r} instead"
        return msg
    if entry.support is Support.UNSUPPORTED:
        return f"{backend} {op} is not supported on {arch}"
    if not entry.known_bad:
        # The common case: 23 of 24 rows have no window, so they never pay for
        # version detection at all.
        return None

    rocm, aiter = _live_versions()
    for bad in entry.known_bad:
        if bad.matches(rocm, aiter):
            where = f"rocm={rocm or 'unknown'} aiter={aiter or 'unknown'}"
            detail = f": {bad.detail}" if bad.detail else ""
            link = f" ({bad.url})" if bad.url else ""
            if os.environ.get("FLASHINFER_ARCH_ALLOW_KNOWN_BAD", "0") == "1":
                return None
            return (
                f"{backend} {op} is known-broken on {arch} with {where}{detail}"
                f"{link}. Set FLASHINFER_ARCH_ALLOW_KNOWN_BAD=1 to run anyway"
            )
    return None


def capability_reason(device, op: str, backend: str) -> Optional[str]:
    """Why ``backend`` cannot serve ``op`` on ``device``, or ``None`` if it can.

    The ``auto`` selectors already build a human-readable ``reason`` string per
    unmet constraint and warn once per ``(device, reason)``. Returning the reason
    rather than a bare bool lets a capability gate slot into that machinery as
    one more reason, so a user whose batch prefill silently dropped to ``fa2``
    can find out why.
    """
    return _blocking_reason(op, backend, _device_arch(device))


def capability_available(device, op: str, backend: str) -> bool:
    """Whether ``auto`` may route ``op`` to ``backend`` on ``device``.

    ``device`` leads to match the gating helpers this backs
    (``aiter_utils.is_aiter_available(device, op)``,
    ``aiter_utils.require_aiter(device, op)``), so those delegate by appending
    ``backend`` rather than reordering. ``op`` and ``backend`` are both ``str``,
    so an argument-order slip between them would be silent -- keeping the shared
    prefix identical is what stops one happening.
    """
    return _blocking_reason(op, backend, _device_arch(device)) is None


def require_capability(device, op: str, backend: str) -> None:
    """Raise :class:`ArchCapabilityError` if ``backend`` cannot serve ``op`` here.

    Mirrors ``aiter_utils.require_aiter(device, op)`` -- same information plus
    the backend, one exception type instead of three.
    """
    reason = _blocking_reason(op, backend, _device_arch(device))
    if reason is not None:
        raise ArchCapabilityError(reason)


def _device_arch(device) -> str:
    """Normalized architecture of ``device``, or ``"unknown"``.

    torch is imported lazily; see the module docstring.
    """
    try:
        import torch

        return normalize_arch(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        return "unknown"
