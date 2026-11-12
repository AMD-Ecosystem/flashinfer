# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Optional

import torch

from .arch_caps import (
    ArchCapabilityError,
    aiter_fallback_backend,
    capability_available,
    normalize_arch,
    require_capability,
)
from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS
from ..jit.core import MissingJITCacheError, logger


@functools.lru_cache(maxsize=8)
def is_aiter_supported(device: torch.device) -> bool:
    """Return True when the given device is an AMD GPU that AITER targets (gfx942/gfx950)."""
    if torch.version.hip is None:
        return False
    try:
        arch = normalize_arch(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        return False
    return arch in FLASHINFER_SUPPORTED_ROCM_ARCHS


def _ensure_aiter_gpu_archs() -> None:
    """Give AITER's JIT a GPU_ARCHS, since from 0.1.16 it requires one.

    Unset reaches AITER's validator as ``['']`` and every JIT build asserts. Our
    own shim build sets this for its own scope, but AITER's Python ops (decode,
    paged-append, fused MoE) build outside it. Only fills a missing value, so an
    operator-set GPU_ARCHS still wins.
    """
    if os.environ.get("GPU_ARCHS"):
        return
    # Imported lazily: flashinfer.jit pulls in the compilation context, and
    # importing it at module scope here would be circular.
    from ..jit.rocm.aiter_source import resolve_aiter_build_arch

    arch = resolve_aiter_build_arch()
    if arch:
        os.environ["GPU_ARCHS"] = arch


# 0.1.21 moved rmsnorm and the cos/sin-cache rope from at::Tensor to the POD
# aiter_tensor_t, so csrc/rocm/{norm,rope}_aiter.cu cannot resolve their mangled
# symbols against anything older, and the vendored arg structs assume this layout.
AITER_MIN_VERSION = "0.1.21"


@functools.lru_cache(maxsize=1)
def _aiter_installed_version() -> Optional[str]:
    """The installed amd-aiter version, or None when it is not installed.

    Cached like the import probe, since the auto selector's fallback asks on
    every call."""
    try:
        import importlib.metadata as _md

        return _md.version("amd-aiter")
    except Exception:
        return None


def _aiter_version_supported() -> bool:
    """True when amd-aiter is installed and new enough for our vendored ABI."""
    installed = _aiter_installed_version()
    if installed is None:
        return False
    try:
        from packaging.version import Version

        # Straight PEP 440 comparison, which already sorts the cases that matter:
        # 0.1.16.post3.dev0+g<sha> (the nightly) is a dev build *of post3* and
        # sorts above 0.1.16, while 0.1.16.dev0 is a pre-release of 0.1.16 and
        # sorts below it. base_version would strip both segments and wrongly
        # admit the latter.
        return Version(installed) >= Version(AITER_MIN_VERSION)
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _aiter_import_error() -> Optional[str]:
    """What importing AITER's Python packages raised, one line, or None if they import.

    An installed amd-aiter that fails to import (e.g. a FlyDSL that will not load)
    is warned about once, so the fallback to fa2/native is never silent.
    """
    try:
        _ensure_aiter_gpu_archs()
        import aiter.ops  # noqa: F401
        from aiter.jit import core as _core  # noqa: F401

        # Deliberately no header-tree check: this answers "can AITER run?", and
        # the shim may already be built or AOT-cached. The build path raises with
        # the real cause when the headers are the thing that is missing.
    except Exception as exc:
        error = one_line(exc)
        installed = _aiter_installed_version()
        if installed is not None:
            logger.warning(
                "amd-aiter %s is installed but failed to import, so every AITER "
                "backend is unavailable: %s",
                installed,
                error,
            )
        return error
    return None


def one_line(exc: BaseException) -> str:
    """``"<Type>: <message>"`` on one bounded line.

    These strings reach the benchmark CSV's backend_fallback_reason column, where
    an embedded traceback or newline would corrupt the row.
    """
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"


def aiter_unavailable_reason() -> Optional[str]:
    """Why an *installed* amd-aiter cannot run, or None if it can or is absent.

    The one ranking every caller reports: below the ABI floor, then a failed import.
    """
    installed = _aiter_installed_version()
    if installed is None:
        return None
    if not _aiter_version_supported():
        return (
            f"amd-aiter {installed} is below the {AITER_MIN_VERSION} ABI floor; the "
            "vendored struct layouts do not match older releases"
        )
    error = _aiter_import_error()
    if error is None:
        return None
    return f"amd-aiter {installed} is installed but failed to import: {error}"


@functools.lru_cache(maxsize=1)
def _aiter_importable() -> bool:
    """True when the AITER packages needed for the C++ backends actually import.

    Uses a real import (not ``find_spec``) so a broken or partially-installed AITER
    is reported as unavailable rather than routing ``auto`` into a path that raises
    at build/load time. Too-old AITER counts as unavailable for the same reason:
    ``auto`` must not route into an ABI it would corrupt.
    """
    return _aiter_version_supported() and _aiter_import_error() is None


def is_aiter_available(device: torch.device, op: str) -> bool:
    """Return True when ``backend="auto"`` may route to AITER for ``device``.

    Combines the arch check with an import probe so ``auto`` falls back to the
    native kernel (rather than raising at build time) when the AITER package is not
    installed or not importable. Explicit ``backend="aiter"`` still surfaces a clear
    error via :func:`require_aiter`.

    Args:
        device: The device the op would run on.
        op: The capability-table op name, e.g. ``"rmsnorm"``. Required, and it
            changes the answer: support is per ``(op, arch)``, so one op can be
            gated on a toolchain where another is fine.
            The name must match a row in :data:`flashinfer.rocm.arch_caps.CAPABILITIES`
            or the lookup treats it as undeclared and returns False.
    """
    return capability_available(device, op, "aiter") and _aiter_importable()


def require_aiter(device: torch.device, op: str) -> None:
    """Validate the explicit ``backend="aiter"`` opt-in, raising a clear error.

    The architecture half is delegated to the capability table, so this op can be
    gated on more than "is the arch in the allowlist" -- e.g. a known_bad window
    for a toolchain that miscompiles the kernel (no row carries one today).
    ``ArchCapabilityError`` subclasses ``ValueError``, so the previous contract is
    unchanged for callers.

    The package check stays here: a missing ``aiter`` is a different condition
    from an unusable architecture, and it keeps raising a ``ValueError`` rather
    than a raw ``ImportError`` from the JIT loader, matching the public-API docs
    that say this mode "requires the aiter package".
    """
    require_capability(device, op, "aiter")
    if not _aiter_importable():
        # The usable fallback is per-op -- "fa2" for attention, "native" for the
        # rest -- so take it from the table rather than naming one that this op
        # does not accept.
        alt = aiter_fallback_backend(op)
        advice = f"use backend={alt!r}" if alt else "use a non-AITER backend"
        reason = aiter_unavailable_reason()
        if reason is not None:
            raise ValueError(
                f"backend='aiter' for {op} requires amd-aiter >= {AITER_MIN_VERSION} "
                f"that imports: {reason}. Fix amd-aiter or {advice}."
            )
        raise ValueError(
            f"backend='aiter' for {op} requires the aiter package, which is not "
            f"installed or failed to import. Install it (see docs/rocm/backends.md) "
            f"or {advice}."
        )


# Failures that are never AITER's to own, so ``auto`` must not demote on them:
# a wrong-numerics arch gate, our own missing AOT cache, a caller contract
# violation, and a transient OOM that would otherwise cache as "unsupported".
_AITER_PROBE_FATAL = (
    ArchCapabilityError,
    MissingJITCacheError,
    ValueError,
    torch.cuda.OutOfMemoryError,
)


def handle_aiter_probe_failure(exc: BaseException, *, op: str) -> str:
    """Classify a failure raised while probing whether AITER can serve ``op``.

    Re-raises when the failure is not a "this AITER install cannot build this
    variant" condition; otherwise warns and returns the reason string for
    ``backend_fallback_reason``. Callers must be ``lru_cache``d, since that —
    not this function — is what stops a failing build being retried.
    """
    if isinstance(exc, _AITER_PROBE_FATAL):
        raise exc
    if os.environ.get("FLASHINFER_AITER_STRICT", "0") == "1":
        raise exc

    reason = (
        f"aiter {op} kernel bootstrap failed on this install: {one_line(exc)} "
        "(set FLASHINFER_AITER_STRICT=1 to raise instead)"
    )
    logger.warning("auto backend falling back to fa2: %s", reason)
    return reason
