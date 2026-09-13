# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The AITER attention variants FlashInfer ``dlopen``s, and where they live.

The amd-aiter wheel prebuilds almost none of them, so AITER's lazy JIT builds
them at ``plan()`` time. This module enumerates the set so it can be built ahead
of time instead.

**Variants and builds are not one-to-one**: two bootstraps loop over
``return_lse`` internally and emit two files per call, so driving this table by
filename would run those builds twice into one directory -- hence :func:`builds`
alongside :func:`reachable_variants`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from .. import env as jit_env
from .aiter_source import resolve_aiter_build_arch

__all__ = [
    "Family",
    "VariantKey",
    "builds",
    "reachable_variants",
    "so_name",
    "variant_store_dir",
]


class Family(Enum):
    """The three ``.so`` families, keyed by their loader in aiter_loader.cc.

    ``prefix``/``suffix``/``include_logits`` mirror the ``build_so_name`` call
    sites there exactly; ``tests/rocm/test_aiter_variants.py`` asserts the
    spelling against the C++ source rather than trusting this copy.
    """

    MHA_FWD = ("mha_fwd_", "_ndropout_nqscale.so", False, True)
    MHA_VARLEN_FWD = ("mha_varlen_fwd_", "_ndropout_nskip_nqscale.so", True, True)
    MHA_BATCH_PREFILL = (
        "mha_batch_prefill_",
        "_ndropout_nqscale_nsink.so",
        True,
        False,
    )

    def __init__(
        self, prefix: str, suffix: str, include_logits: bool, servable: bool
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.include_logits = include_logits
        # Can a store hit actually spare the AITER build? Only where the
        # bootstrap exists to produce the .so. _aiter_bootstrap_batch_prefill is
        # also the page_size capability probe, and the filename has no page-size
        # axis, so it must run either way -- prebuilding that family fills the
        # store with files nothing ever saves time on.
        self.servable_from_store = servable


# Deliberately not order=True: `family` is an Enum, so the generated __lt__
# raises rather than sorting. Sort by so_name() when an order is needed.
@dataclass(frozen=True)
class VariantKey:
    """One ``.so``. Mirrors ``VariantKey`` / ``BatchPrefillVariantKey`` in C++.

    ``has_alibi`` is absent on purpose: all three construction sites in
    ``include/flashinfer/rocm/attention/aiter/`` hard-code it to false, so only
    the ``_nbias`` arm is reachable and enumerating the other doubles the set
    for nothing.
    """

    family: Family
    dtype: str  # "fp16" or "bf16"
    has_logits_cap: bool
    needs_mask: bool
    has_lse: bool


def so_name(key: VariantKey) -> str:
    """The filename ``aiter_loader.cc`` will ask ``dlopen`` for."""
    name = key.family.prefix + key.dtype
    if key.family.include_logits:
        name += "_logits" if key.has_logits_cap else "_nlogits"
    name += "_nbias"
    name += "_mask" if key.needs_mask else "_nmask"
    name += "_lse" if key.has_lse else "_nlse"
    return name + key.family.suffix


_DTYPES = ("bf16", "fp16")


def reachable_variants() -> Tuple[VariantKey, ...]:
    """Every ``.so`` the loader can ask for: 8 + 16 + 16 = 40 per architecture.

    ``mha_fwd`` contributes only 8 because ``get_aiter_mha_fwd_handle`` refuses
    ``has_logits_cap=true`` -- that template has no ``_logits`` arm, and a
    soft-capped call is routed to the varlen family instead.
    """
    out = []
    for family in Family:
        logits_values = (False, True) if family.include_logits else (False,)
        for dtype in _DTYPES:
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    for has_lse in (False, True):
                        out.append(
                            VariantKey(
                                family, dtype, has_logits_cap, needs_mask, has_lse
                            )
                        )
    return tuple(out)


@dataclass(frozen=True)
class BuildSpec:
    """One AITER build, and the variants it produces.

    ``emits_both_lse`` records the asymmetry: the varlen bootstraps loop over
    ``return_lse`` and yield two files per call, so a driver must not schedule
    the ``_lse`` and ``_nlse`` variants as separate builds.
    """

    family: Family
    dtype: str
    has_logits_cap: bool
    needs_mask: bool
    has_lse: Optional[bool]  # None when the build emits both arms
    produces: Tuple[VariantKey, ...]

    @property
    def emits_both_lse(self) -> bool:
        return self.has_lse is None


def builds() -> Tuple[BuildSpec, ...]:
    """The 32 AITER builds that cover all 40 variants.

    ``mha_varlen_fwd`` needs 8 builds for 16 files because
    ``_aiter_bootstrap_batch_ragged_prefill`` loops ``return_lse`` internally.
    """
    by_key = {
        (v.family, v.dtype, v.has_logits_cap, v.needs_mask, v.has_lse): v
        for v in reachable_variants()
    }
    out = []
    for family in Family:
        logits_values = (False, True) if family.include_logits else (False,)
        for dtype in _DTYPES:
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    if family is Family.MHA_VARLEN_FWD:
                        produces = tuple(
                            by_key[(family, dtype, has_logits_cap, needs_mask, lse)]
                            for lse in (False, True)
                        )
                        out.append(
                            BuildSpec(
                                family,
                                dtype,
                                has_logits_cap,
                                needs_mask,
                                None,
                                produces,
                            )
                        )
                        continue
                    for has_lse in (False, True):
                        key = by_key[
                            (family, dtype, has_logits_cap, needs_mask, has_lse)
                        ]
                        out.append(
                            BuildSpec(
                                family,
                                dtype,
                                has_logits_cap,
                                needs_mask,
                                has_lse,
                                (key,),
                            )
                        )
    return tuple(out)


MANIFEST_NAME = "variants_manifest.json"


def _rocm_version() -> str:
    """The ROCm/HIP version the artifacts were compiled against.

    Part of the store tag because these are CK-tile objects that ship between
    machines in a wheel, unlike the ``aiter_libs`` cache, which never leaves the
    box that built it.
    """
    try:
        import torch

        return torch.version.hip or "unknown"
    except Exception:
        return "unknown"


def variant_store_dir(arch: Optional[str] = None) -> Path:
    """``<cache>/aiter_variants/<arch>__aiter-<ver>__rocm-<ver>/``.

    Keyed hard enough that staleness is structural rather than something to
    detect: bump AITER or ROCm and the tag names a different directory, so the
    old contents are simply never found and the lookup misses into a rebuild.
    A mismatched artifact is never loaded.
    """
    try:
        import importlib.metadata as _md

        aiter_version = _md.version("amd-aiter")
    except Exception:
        aiter_version = "unknown"
    tag = f"{arch or resolve_aiter_build_arch()}__aiter-{aiter_version}__rocm-{_rocm_version()}"
    # The tag becomes a directory name; refuse anything that is not one
    # component, the same guard _aiter_cache_tag applies.
    if not tag or tag != Path(tag).name or tag.startswith("."):
        raise ValueError(f"refusing to build a cache directory name from {tag!r}")
    return jit_env.FLASHINFER_CACHE_DIR / "aiter_variants" / tag


def _explicit_stores() -> List[Path]:
    """Operator-set ``FLASHINFER_AITER_VARIANT_DIR``, as a ``os.pathsep`` list.

    Non-existent entries are dropped rather than trusted: an exported path that
    is not there only lengthens the loader's candidate list.
    """
    raw = os.environ.get("FLASHINFER_AITER_VARIANT_DIR", "")
    return [p for p in (Path(x) for x in raw.split(os.pathsep) if x) if p.is_dir()]


def _wheel_store() -> Optional[Path]:
    """The store this install's tag names inside the jit-cache wheel, if any.

    The wheel holds one subdirectory per ``<arch>__aiter-<ver>__rocm-<ver>``, so
    a multi-arch wheel can carry several and the consumer picks its own. That
    also settles the mismatch between the AOT manifest, which is a comma-joined
    multi-arch list, and a store, which is always single-arch: the tag *is* the
    check, so no cross-read of the manifest is needed.

    ``getattr`` rather than a plain attribute: an older jit-cache wheel exports
    only ``get_jit_cache_dir``, and ``FLASHINFER_DISABLE_VERSION_CHECK`` can get
    one past the version gate, so this must degrade rather than raise at import.
    """
    try:
        import amd_flashinfer_jit_cache
    except Exception:
        return None
    accessor = getattr(amd_flashinfer_jit_cache, "get_aiter_variant_dir", None)
    if accessor is None:
        return None
    try:
        candidate = Path(accessor()) / variant_store_dir().name
    except Exception:
        return None
    return candidate if candidate.is_dir() else None


# Set by flashinfer.rocm.prebuild_aiter_variants while it builds.
_SKIP_STORE_LOOKUP = False


def active_stores() -> List[Path]:
    """Every store this process may read, in priority order.

    A list rather than one directory: a jit-cache wheel can ship a partial
    store, and returning only that would permanently shadow the cache store the
    prebuild driver writes. :func:`find_variant` and :func:`export_variant_store`
    share this list so lookup and dlopen cannot disagree.

    An operator-set ``AITER_JIT_DIR`` suppresses the auto-discovered stores: it
    selects which AITER build to use, and a store built against the pinned
    install would load a kernel from a different build, silently, since the
    mangled symbol still resolves. An explicit variant dir is their own choice
    and still honoured.
    """
    explicit = _explicit_stores()
    if explicit:
        return explicit
    if os.environ.get("AITER_JIT_DIR"):
        return []
    stores = []
    try:
        cache = variant_store_dir()
    except Exception:
        # variant_store_dir raises on an unusable arch tag. Reached from all
        # four bootstraps; "no store" is the honest answer, and the caller then
        # builds exactly as it did before.
        cache = None
    if cache is not None and cache.is_dir():
        stores.append(cache)
    wheel = _wheel_store()
    if wheel is not None and wheel not in stores:
        stores.append(wheel)
    return stores


def _is_loadable(path: Path) -> bool:
    """A plausible ELF, not merely a name that exists.

    A truncated or half-copied artifact would otherwise satisfy the lookup, skip
    the build, and then fail at dlopen with no fallback left to take.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def find_variant(key: VariantKey) -> Optional[Path]:
    """The prebuilt ``.so`` for ``key``, or None if this install has none."""
    if _SKIP_STORE_LOOKUP:
        # The prebuild driver sets this: its whole job is to run the bootstraps,
        # and a store hit -- including one from a jit-cache wheel it is not
        # writing to -- would make every build a silent no-op.
        return None
    name = so_name(key)
    for root in active_stores():
        candidate = root / name
        if _is_loadable(candidate):
            return candidate
    return None


def dtype_tag(dtype) -> Optional[str]:
    """``torch.bfloat16`` -> ``"bf16"``. None for a dtype with no variant."""
    import torch

    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype)


def prebuilt(
    family: Family,
    dtype,
    *,
    has_logits_cap: bool = False,
    needs_mask: bool,
    has_lse: Optional[bool] = None,
) -> bool:
    """Is every ``.so`` this bootstrap would produce already in the store?

    ``has_lse=None`` asks about a varlen bootstrap, which emits both arms from
    one call -- so both must be present, or the call still has work to do.

    Only the store is consulted, not AITER's own JIT directory: a variant
    already there makes the bootstrap a no-op anyway (measured at 0.0s), so
    checking it would buy nothing and would couple this to AITER's layout.
    """
    tag = dtype_tag(dtype)
    if tag is None:
        return False
    lse_values = (False, True) if has_lse is None else (has_lse,)
    return all(
        find_variant(VariantKey(family, tag, has_logits_cap, needs_mask, lse))
        is not None
        for lse in lse_values
    )


def export_variant_store() -> List[Path]:
    """Publish the stores to ``FLASHINFER_AITER_VARIANT_DIR`` for the C++ loader.

    Resolved here and passed through the environment rather than baked in as a
    ``-D``: an AOT-packaged module carries whatever path its *build* machine
    had, which does not exist on the consumer's.

    Exports the same list :func:`find_variant` searches, so a variant that
    skipped a build is always one the loader can also find. An empty list leaves
    the variable alone rather than clearing an operator's value.
    """
    stores = active_stores()
    if stores:
        os.environ["FLASHINFER_AITER_VARIANT_DIR"] = os.pathsep.join(
            str(p) for p in stores
        )
    return stores
