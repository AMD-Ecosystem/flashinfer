# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Optional

import torch

from .arch_caps import (
    ArchCapabilityError,
    capability_available,
    normalize_arch,
    require_capability,
)
from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS
from .jit.core import MissingJITCacheError, logger


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
    from .jit.aiter_source import resolve_aiter_build_arch

    arch = resolve_aiter_build_arch()
    if arch:
        os.environ["GPU_ARCHS"] = arch


# The vendored structs in include/flashinfer/attention/aiter/ follow the 0.1.16
# layout. They travel by value through dlsym'd pointers, so an older AITER
# mismatches offsets silently instead of failing to load -- hence a hard floor
# rather than a warning.
AITER_MIN_VERSION = "0.1.16"


def _aiter_installed_version() -> Optional[str]:
    """The installed amd-aiter version, or None when it is not installed."""
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

        # Compare on base_version: the nightly wheels carry a local '+g<sha>'
        # segment, and a '.dev0' pre-release segment that PEP 440 sorts *below*
        # the release it is built from -- 0.1.16.post3.dev0 must still pass a
        # 0.1.16 floor. Re-wrap in Version; base_version is a str.
        return Version(Version(installed).base_version) >= Version(AITER_MIN_VERSION)
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _aiter_importable() -> bool:
    """True when the AITER packages needed for the C++ backends actually import.

    Uses a real import (not ``find_spec``) so a broken or partially-installed AITER
    — where the spec exists but importing the compiled extension fails, e.g. missing
    ROCm deps — is reported as unavailable rather than routing ``auto`` into a path
    that raises at build/load time. Too-old AITER counts as unavailable for the same
    reason: ``auto`` must not route into an ABI it would corrupt.
    """
    try:
        _ensure_aiter_gpu_archs()
        import aiter  # noqa: F401
        import aiter_meta  # noqa: F401
        from aiter.jit import core as _core  # noqa: F401
    except Exception:
        return False
    return _aiter_version_supported()


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
            gated on a toolchain where another is fine -- AITER batch prefill is
            gated on gfx950 under ROCm 7.2.x while every other op stays open.
            The name must match a row in :data:`flashinfer.arch_caps.CAPABILITIES`
            or the lookup treats it as undeclared and returns False.
    """
    return capability_available(device, op, "aiter") and _aiter_importable()


def require_aiter(device: torch.device, op: str) -> None:
    """Validate the explicit ``backend="aiter"`` opt-in, raising a clear error.

    The architecture half is delegated to the capability table, so this op can be
    gated on more than "is the arch in the allowlist" -- e.g. a toolchain in which
    the kernel is known to be miscompiled. ``ArchCapabilityError`` subclasses
    ``ValueError``, so the previous contract is unchanged for callers.

    The package check stays here: a missing ``aiter`` is a different condition
    from an unusable architecture, and it keeps raising a ``ValueError`` rather
    than a raw ``ImportError`` from the JIT loader, matching the public-API docs
    that say this mode "requires the aiter package".
    """
    require_capability(device, op, "aiter")
    if not _aiter_importable():
        installed = _aiter_installed_version()
        if installed is not None and not _aiter_version_supported():
            raise ValueError(
                f"backend='aiter' for {op} requires amd-aiter >= {AITER_MIN_VERSION}, "
                f"but {installed} is installed; the vendored struct layouts do not "
                "match older releases and would corrupt arguments silently. Upgrade "
                "amd-aiter or use backend='native'."
            )
        raise ValueError(
            f"backend='aiter' for {op} requires the aiter package, which is not "
            "installed or failed to import. Install it (see the AITER Support "
            "section in the README) or use backend='native'."
        )


@functools.cache
def get_aiter_mha_module():
    from aiter.ops import mha as aiter_mha_module

    return aiter_mha_module


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

    # Single-line and bounded: this string reaches a CSV column in the benchmark
    # harness, where an embedded traceback would corrupt the row.
    detail = " ".join(str(exc).split())[:200]
    reason = (
        f"aiter {op} kernel bootstrap failed on this install: "
        f"{type(exc).__name__}: {detail} "
        "(set FLASHINFER_AITER_STRICT=1 to raise instead)"
    )
    logger.warning("auto backend falling back to fa2: %s", reason)
    return reason
