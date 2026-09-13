# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the AITER attention variants ahead of time, into FlashInfer's store.

Without this, the first ``plan()`` needing a missing variant pays an AITER
CK-tile compile -- 74-360s on gfx942 depending on the family, once per variant
(one ``.so`` serves every head dimension), and again after every container
restart that does not persist site-packages. On a read-only install it
does not stall, it fails, and ``backend="auto"`` silently drops to fa2.

    python -m flashinfer.rocm.prebuild_aiter_variants --list
    python -m flashinfer.rocm.prebuild_aiter_variants --arch gfx942
    python -m flashinfer.rocm.prebuild_aiter_variants --only mha_batch_prefill

**This needs a GPU.** The only supported way to make AITER emit a variant is to
call the op, which launches a kernel -- so it cannot be a ``docker build`` step
and has to run as a GPU-attached job whose output the image copies in.

One build at a time, deliberately. ``_aiter_env_scope`` mutates process-global
environment that AITER reads, so two concurrent builds in one process have the
first to finish restore the environment under the second. Parallelism belongs
across *processes*, each with its own environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..jit.rocm.aiter_source import (
    _aiter_env_scope,
    _BUILD_LOCK,
    resolve_aiter_build_arch,
)
from ..jit.rocm import aiter_variants as _variants
from ..jit.rocm.aiter_variants import (
    MANIFEST_NAME,
    BuildSpec,
    Family,
    VariantKey,
    builds,
    so_name,
    variant_store_dir,
)


def _select_device(device_idx: int) -> None:
    """Make ``--device`` current before anything resolves the architecture.

    ``resolve_aiter_build_arch()`` is ``lru_cache``d and reads
    ``torch.cuda.current_device()``, so the first caller decides the store tag
    and ``GPU_ARCHS`` for the whole process. No-op without a GPU, so ``--list``
    still works on a build box -- but a bad index raises rather than silently
    leaving the previous device current under a tag naming the wrong arch.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(device_idx)


def _dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _run_build(spec: BuildSpec, device_idx: int, head_dim: int) -> None:
    """Invoke the bootstrap that produces ``spec``.

    Calls the real bootstraps rather than reimplementing AITER's naming: they
    already encode which (dtype, mask, lse, logits) maps to which call, they are
    what plan() uses, and they are covered by the existing tests. Only the output
    directory differs, and that comes from the env scope the caller opened.
    """
    from . import prefill as _prefill

    dtype = _dtype(spec.dtype)
    if spec.family is Family.MHA_FWD:
        _prefill._aiter_bootstrap_single_prefill_mha_fwd(
            dtype, spec.needs_mask, bool(spec.has_lse), head_dim, device_idx
        )
    elif spec.family is Family.MHA_VARLEN_FWD:
        # Loops return_lse internally and emits both arms.
        _prefill._aiter_bootstrap_batch_ragged_prefill(
            dtype, spec.has_logits_cap, spec.needs_mask, head_dim, device_idx
        )
    elif spec.family is Family.MHA_BATCH_PREFILL:
        _build_batch_prefill(spec, dtype, head_dim, device_idx)
    else:  # pragma: no cover - Family is closed
        raise AssertionError(f"unhandled family {spec.family}")


def _build_batch_prefill(
    spec: BuildSpec, dtype, head_dim: int, device_idx: int
) -> None:
    """Build the paged variant, trying each page size the predicate offers.

    ``_aiter_native_page_sizes()`` is a version predicate, and an installed build
    can reject a size it names with "no matching kernel found" -- which is the
    whole reason ``_aiter_native_paging_available`` probes at runtime. Stopping
    at the smallest would fail the entire family when a larger size would have
    built. The variant filename has no page-size axis, so any working size
    produces the artifact the loader wants.
    """
    from . import prefill as _prefill

    sizes = sorted(_prefill._aiter_native_page_sizes()) or [16]
    errors = []
    for page_size in sizes:
        try:
            _prefill._aiter_bootstrap_batch_prefill(
                dtype,
                spec.has_logits_cap,
                spec.needs_mask,
                bool(spec.has_lse),
                page_size,
                head_dim,
                device_idx,
            )
            return
        except Exception as exc:  # noqa: BLE001 - try the next size
            errors.append(f"page_size={page_size}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "no page size produced a paged kernel:\n  " + "\n  ".join(errors)
    )


def _publish(produced: Iterable[Path], store: Path) -> List[str]:
    """Copy artifacts into the store, atomically. Returns the names published."""
    store.mkdir(parents=True, exist_ok=True)
    names = []
    for src in produced:
        dst = store / src.name
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        names.append(src.name)
    return names


def _locate(keys: Sequence[VariantKey], search: Sequence[Path]) -> List[Path]:
    """Find each variant's artifact. Missing keys are simply absent from the result.

    The recursive fallback prefers the shallowest match: AITER stages a module
    under ``<jit_dir>/build/<md_name>/`` before installing it alongside the
    others, and taking the first hit in path order would publish the staged copy.
    """
    found = []
    for key in keys:
        name = so_name(key)
        for root in search:
            candidate = root / name
            if candidate.is_file():
                found.append(candidate)
                break
            hits = (
                sorted(root.rglob(name), key=lambda p: len(p.parts))
                if root.exists()
                else []
            )
            if hits:
                found.append(hits[0])
                break
    return found


def prebuild(
    specs: Sequence[BuildSpec],
    *,
    arch: Optional[str] = None,
    device_idx: int = 0,
    head_dim: int = 128,
    force: bool = False,
) -> Tuple[int, int, List[str]]:
    """Build ``specs`` into the variant store. Returns (built, skipped, failures).

    A spec whose every output is already in the store is skipped, so a rerun
    after a partial failure resumes rather than restarting.
    """
    from . import prefill as _prefill

    # _aiter_env_scope(None, ...) preserves an ambient AITER_JIT_DIR so the
    # bootstraps can import what they build. That would publish a *different*
    # AITER build's artifacts into a store tagged with the installed version,
    # which a later process auto-discovers -- the isolation active_stores()
    # enforces on the read side, defeated on the write side.
    # _aiter_env_scope(None, ...) preserves an ambient AITER_JIT_DIR so the
    # bootstraps can import what they build. That would publish a *different*
    # AITER build's artifacts into a store tagged with the installed version,
    # which a later process auto-discovers -- the isolation active_stores()
    # enforces on the read side, defeated on the write side.
    if os.environ.get("AITER_JIT_DIR"):
        raise RuntimeError(
            "AITER_JIT_DIR is set. The store tag comes from the installed "
            "amd-aiter version, so publishing artifacts from another AITER "
            "build would mislabel them and they would be auto-loaded by a "
            "process without the override. Unset it to prebuild for the "
            "installed AITER."
        )

    # First, ahead of the probe below: it reaches _ensure_aiter_gpu_archs() and
    # so the cached resolve_aiter_build_arch(), which would freeze the *current*
    # device's arch and leave --device naming the store for the wrong card.
    _select_device(device_idx)

    # The varlen bootstraps reach torch.ops.aiter.mha_varlen_fwd, which is not
    # registered until aiter.ops is imported. plan() always passes through this
    # probe first; the driver does not, so call it explicitly rather than rely
    # on an import another path happened to do. It also applies the ABI floor.
    if not _prefill._aiter_ops_importable():
        raise RuntimeError(
            "aiter.ops is not importable, or amd-aiter is below the supported "
            "floor; nothing can be prebuilt."
        )

    store = variant_store_dir(arch)
    store.mkdir(parents=True, exist_ok=True)

    # The bootstraps skip themselves on a store hit, and the store they consult
    # may not be the one being written -- a jit-cache wheel or an operator
    # FLASHINFER_AITER_VARIANT_DIR both satisfy it. Without this every build
    # would return immediately and then fail "AITER produced 0 of N".
    previous = _variants._SKIP_STORE_LOOKUP
    _variants._SKIP_STORE_LOOKUP = True
    try:
        return _prebuild_specs(specs, store, arch, device_idx, head_dim, force)
    finally:
        _variants._SKIP_STORE_LOOKUP = previous


def _prebuild_specs(specs, store, arch, device_idx, head_dim, force):
    import torch

    built = skipped = 0
    failures: List[str] = []
    for index, spec in enumerate(specs, start=1):
        wanted = [so_name(key) for key in spec.produces]
        if force:
            # The bootstraps short-circuit on a store hit, so leaving the old
            # copies in place would make --force a no-op. Removing them first
            # re-arms both that guard and this one; AITER still skips its own
            # compile if it holds the artifact, which is what "re-publish"
            # should mean.
            for name in wanted:
                (store / name).unlink(missing_ok=True)
        elif all(_variants._is_loadable(store / name) for name in wanted):
            skipped += 1
            continue

        label = ", ".join(wanted)
        print(f"[{index}/{len(specs)}] building {label}", flush=True)
        started = time.time()
        try:
            # build_dir=None on purpose: AITER imports the variant it just built,
            # and only puts AITER_JIT_DIR on sys.path at its own import time, so
            # redirecting it here yields ModuleNotFoundError for the new module.
            # Let AITER build where it wants and copy the result into the store.
            with _BUILD_LOCK, _aiter_env_scope(None, symbol_visible=False):
                _run_build(spec, device_idx, head_dim)
                # The bootstraps launch kernels asynchronously. Without this a
                # device-side fault surfaces on the *next* spec's launch, which
                # is then blamed for it while this artifact is published as good.
                torch.cuda.synchronize(device_idx)
                from aiter.jit import core as aiter_core

                search = [Path(aiter_core.get_user_jit_dir())]
                produced = _locate(spec.produces, search)
            # Every output, not merely one: a varlen spec emits two files, and
            # publishing half of it would record a store the caller believes is
            # complete.
            if len(produced) != len(spec.produces):
                got = {p.name for p in produced}
                raise RuntimeError(
                    f"AITER produced {len(produced)} of {len(spec.produces)}; "
                    f"missing {sorted(set(wanted) - got)}. "
                    f"Searched {[str(p) for p in search]}."
                )
            names = _publish(produced, store)
            built += 1
            print(
                f"    -> {len(names)} file(s) in {time.time() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad spec must not end the run
            # 32 builds is a one-to-three hour job; aborting it on the first
            # failure would also skip the manifest for everything that worked.
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"    !! FAILED: {type(exc).__name__}: {exc}", flush=True)

    try:
        _write_manifest(store, arch)
    except Exception as exc:  # noqa: BLE001 - see the abort note above
        # Same reason the per-spec failures are collected: losing a multi-hour
        # run's counts to a manifest write is the outcome that note rejects.
        failures.append(f"manifest: {type(exc).__name__}: {exc}")
        print(f"!! manifest write FAILED: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        print(f"\n{len(failures)} spec(s) failed:", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
    return built, skipped, failures


def _write_manifest(store: Path, arch: Optional[str]) -> None:
    try:
        import importlib.metadata as _md

        aiter_version = _md.version("amd-aiter")
    except Exception:
        aiter_version = "unknown"
    try:
        import torch

        rocm = torch.version.hip or "unknown"
    except Exception:
        rocm = "unknown"
    present = sorted(p.name for p in store.glob("*.so"))
    (store / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "rocm_arch_list": arch or resolve_aiter_build_arch(),
                "aiter_version": aiter_version,
                "rocm_version": rocm,
                "variants": present,
            },
            indent=2,
        )
        + "\n"
    )


def prune(*, arch: Optional[str] = None, apply: bool = False) -> List[Path]:
    """Stores whose tag does not match this install. Deletes only when ``apply``.

    Each (arch, aiter, rocm) tuple gets its own directory and nothing removes
    the old ones, so every upgrade leaves ~165 MB behind -- on shared nodes,
    indefinitely.

    Dry-run unless ``apply``, scoped to the running architecture, and limited to
    directories this uid owns. FLASHINFER_CACHE_DIR is shared on these nodes, so
    arch alone is not enough: a colleague on the same card with a different AITER
    pin has a same-arch store, and deleting it mid-run breaks their dlopens.
    """
    current_dir = variant_store_dir(arch)
    root = current_dir.parent
    arch = current_dir.name.split("__", 1)[0]
    uid = os.getuid()

    def mine(path: Path) -> bool:
        try:
            return path.stat().st_uid == uid
        except OSError:
            return False

    stale = (
        sorted(
            d
            for d in root.glob(f"{arch}__*")
            if d.is_dir() and d.name != current_dir.name and mine(d)
        )
        if root.is_dir()
        else []
    )

    for path in stale:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{'removing' if apply else 'stale'}   {path}  ({size / 1e6:.0f} MB)")
        if apply:
            shutil.rmtree(path)
    if not stale:
        print(f"no stale stores under {root}")
    elif not apply:
        print(f"{len(stale)} stale store(s); re-run with --prune --yes to delete")
    return stale


def _select(only: Optional[str]) -> List[BuildSpec]:
    """The specs to build: by default only the families a store hit can serve.

    ``mha_batch_prefill`` is excluded unless asked for by name. Its bootstrap
    doubles as the page_size capability probe, so it runs whether or not the
    artifact is in the store -- building those 16 costs ~32 min on gfx942 and
    ~68 min on gfx950 and saves nothing. ``--only mha_batch_prefill`` still
    builds them, for when that probe learns to read the store.
    """
    specs = list(builds())
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        known = {f.name.lower() for f in Family}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"unknown family {sorted(unknown)}; expected {sorted(known)}"
            )
        return [s for s in specs if s.family.name.lower() in wanted]
    return [s for s in specs if s.family.servable_from_store]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", help="override the store arch (default: this device)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="a cache key only; one .so serves every head dim",
    )
    parser.add_argument("--only", help="comma-separated families, e.g. mha_fwd")
    parser.add_argument(
        "--force", action="store_true", help="re-publish present variants"
    )
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="report this arch's stores from another aiter/rocm pin",
    )
    parser.add_argument("--yes", action="store_true", help="with --prune, delete them")
    args = parser.parse_args(argv)

    # Before the --arch check and the store lookup below, both of which resolve
    # the architecture -- and resolve_aiter_build_arch is lru_cached, so whoever
    # calls it first freezes it for the process.
    _select_device(args.device)

    if args.arch:
        resolved = resolve_aiter_build_arch()
        if args.arch != resolved:
            # --arch only renames the store; GPU_ARCHS comes from the env
            # scope and the bootstraps launch on the local device, so this
            # would file this box's objects under another arch's tag and
            # fault when loaded there.
            raise SystemExit(
                f"--arch {args.arch} does not match the resolved build arch "
                f"{resolved}; this would mislabel the artifacts. Set "
                "FLASHINFER_ROCM_ARCH_LIST and run on that device instead."
            )

    if args.prune:
        try:
            prune(arch=args.arch, apply=args.yes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return 0

    specs = _select(args.only)
    store = variant_store_dir(args.arch)
    if args.list:
        print(f"store: {store}")
        n_variants = sum(len(spec.produces) for spec in specs)
        print(f"{n_variants} variants from {len(specs)} builds")
        for spec in specs:
            state = (
                "present"
                if all(
                    _variants._is_loadable(store / so_name(k)) for k in spec.produces
                )
                else "missing"
            )
            print(f"  {state:8s} {', '.join(so_name(k) for k in spec.produces)}")
        return 0

    built, skipped, failures = prebuild(
        specs,
        arch=args.arch,
        device_idx=args.device,
        head_dim=args.head_dim,
        force=args.force,
    )
    print(
        f"built {built}, skipped {skipped} already present, {len(failures)} failed "
        f"-> {store}",
        flush=True,
    )
    # Non-zero on any failure: an image build that produced an empty store must
    # not look like a success, or every consumer silently pays the per-shape
    # compile this exists to remove.
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
