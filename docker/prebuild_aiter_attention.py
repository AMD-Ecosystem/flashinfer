#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build AITER's attention variant ``.so`` at image-build time, with no GPU.

FlashInfer ``dlopen``s these by composed filename. The wheel prebuilds almost
none of them, so AITER's lazy JIT compiles each one *inside the serving process*
on the first request that needs it -- minutes of stall, possibly hours into the
process's life, under a process-global build lock.

Runs during ``docker build``, where there is no device. Everything here is
therefore stdlib-only and reaches AITER through the flat ``sys.path`` import its
own ``setup.py`` uses; ``import aiter`` would run arch detection and a triton
import that both need a GPU. The companion GPU-attached driver
(``flashinfer.rocm.prebuild_aiter_variants``) produces variants by *calling the
op*, which is why that one cannot run here.

Artifacts land in AITER's own ``jit/`` directory, so AITER itself finds them
already built and the bootstraps return without compiling. That is what makes
``mha_batch_prefill`` worth building by this route and not by the store route:
its bootstrap doubles as the page-size probe and runs either way, so only a hit
AITER can see spares the compile. Measured: 101.4 s cold against 0.2 s prebuilt.

    python3 docker/prebuild_aiter_attention.py --list
    python3 docker/prebuild_aiter_attention.py --jobs 2
    python3 docker/prebuild_aiter_attention.py --check

This file deliberately carries its own copy of the variant table: importing
``flashinfer.jit.rocm.aiter_variants`` would pull in torch.
``tests/rocm/test_prebuild_aiter_attention.py`` asserts the two agree, and
asserts the receipts and token spellings against the installed ``aiter/ops/mha.py``
-- the only detector for an upstream recipe change.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

DTYPES = ("bf16", "fp16")

# fp8 is batch-prefill only: prefill.py sends every fp8 query down the native
# paged route, because the flat-gather route runs varlen, which has no fp8 arm.
_FAMILY_DTYPES = {
    "mha_fwd": DTYPES,
    "mha_varlen_fwd": DTYPES,
    "mha_batch_prefill": DTYPES + ("fp8bf16",),
}

# Receipts select which CK-tile instance set the codegen emits. They differ per
# family and a wrong one ships a *different kernel* under the right filename,
# with nothing downstream to detect it.
RECEIPT_MHA_FWD = 100
RECEIPT_VARLEN = 200
RECEIPT_BATCH_PREFILL = 200

GENERATE_PY = "example/ck_tile/01_fmha/generate.py"

# Whole AITER modules aiter_loader.cc dlopens by name, with no variant axes. The
# wheel shipped these prebuilt; a PREBUILD_KERNELS=0 source install ships none,
# so the image has to build them or the load throws.
# tests/rocm/test_prebuild_aiter_attention.py checks this against the loader.
LOADER_MODULES = ("module_fmha_v3_fwd",)


class Variant(NamedTuple):
    """One ``.so``. Mirrors ``VariantKey`` in flashinfer/jit/rocm/aiter_variants.py.

    ``has_alibi`` is absent for the same reason it is absent there: all three C++
    construction sites hard-code it false, so the other arm is unreachable.
    """

    family: str  # "mha_fwd" | "mha_varlen_fwd" | "mha_batch_prefill"
    dtype: str
    has_logits_cap: bool
    needs_mask: bool
    has_lse: bool

    @property
    def md_name(self) -> str:
        return _MD_NAME[self.family](self)

    @property
    def so_name(self) -> str:
        return f"{self.md_name}.so"

    @property
    def qscale_token(self) -> str:
        """fp8 always arrives with per-tensor descales; bf16/fp16 never do."""
        return "pertensor" if self.dtype == "fp8bf16" else "nqscale"


def _md_mha_fwd(v: Variant) -> str:
    """Mirrors aiter/ops/mha.py::mha_fwd. No logits or skip token in this family."""
    return (
        "mha_fwd"
        + f"_{v.dtype}"
        + "_nbias"
        + ("_mask" if v.needs_mask else "_nmask")
        + ("_lse" if v.has_lse else "_nlse")
        + "_ndropout"
        + "_nqscale"
    )


def _md_varlen(v: Variant) -> str:
    """Mirrors aiter/ops/mha.py::mha_varlen_fwd. Carries logits *and* skip tokens."""
    return (
        "mha_varlen_fwd"
        + f"_{v.dtype}"
        + ("_logits" if v.has_logits_cap else "_nlogits")
        + "_nbias"
        + ("_mask" if v.needs_mask else "_nmask")
        + ("_lse" if v.has_lse else "_nlse")
        + "_ndropout"
        + "_nskip"
        + "_nqscale"
    )


def _md_batch_prefill(v: Variant) -> str:
    """Mirrors aiter/ops/mha.py::mha_batch_prefill. Adds the sink axis."""
    return (
        "mha_batch_prefill"
        + f"_{v.dtype}"
        + ("_logits" if v.has_logits_cap else "_nlogits")
        + "_nbias"
        + ("_mask" if v.needs_mask else "_nmask")
        + ("_lse" if v.has_lse else "_nlse")
        + "_ndropout"
        + f"_{v.qscale_token}"
        + "_nsink"
    )


_MD_NAME = {
    "mha_fwd": _md_mha_fwd,
    "mha_varlen_fwd": _md_varlen,
    "mha_batch_prefill": _md_batch_prefill,
}

# Families that carry a logits-cap axis. mha_fwd has no _logits arm at all --
# get_aiter_mha_fwd_handle refuses has_logits_cap, routing those calls to varlen.
_HAS_LOGITS_AXIS = {
    "mha_fwd": False,
    "mha_varlen_fwd": True,
    "mha_batch_prefill": True,
}


def _filter_mha_fwd(v: Variant) -> str:
    """The CK ``--filter`` for mha_fwd, mirroring aiter/ops/mha.py.

    Two spellings differ from every other family and both are load-bearing: the
    dtype token takes a leading underscore, and the mask token is ``_m*``, not
    ``_mask*``.
    """
    return (
        "*"
        + f"_{v.dtype}*"
        + "_nbias*"
        + ("_m*" if v.needs_mask else "_nmask*")
        + ("_lse*" if v.has_lse else "_nlse*")
        + "_ndropout*"
        + "_nqscale*"
    )


def _filter_batch_prefill(v: Variant) -> str:
    """The CK ``--filter`` for mha_batch_prefill, mirroring aiter/ops/mha.py.

    The dtype token has **no** leading underscore here, unlike mha_fwd.
    """
    return (
        "*"
        + f"{v.dtype}*"
        + ("_logits*" if v.has_logits_cap else "_nlogits*")
        + "_nbias*"
        + ("_mask*" if v.needs_mask else "_nmask*")
        + ("_lse*" if v.has_lse else "_nlse*")
        + "_ndropout*"
        + f"_{v.qscale_token}*"
        + "_nsink*"
    )


def reachable_variants() -> Tuple[Variant, ...]:
    """Every ``.so`` the loader can ask for: 8 + 16 + 24 = 48 per architecture."""
    out: List[Variant] = []
    for family, has_logits_axis in _HAS_LOGITS_AXIS.items():
        logits_values = (False, True) if has_logits_axis else (False,)
        for dtype in _FAMILY_DTYPES[family]:
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    for has_lse in (False, True):
                        out.append(
                            Variant(family, dtype, has_logits_cap, needs_mask, has_lse)
                        )
    return tuple(out)


def selected_variants(only: Optional[Sequence[str]] = None) -> Tuple[Variant, ...]:
    """Every reachable variant; ``only`` narrows to named families.

    Nothing is trimmed: the whole set costs minutes, a missing arm costs a stall.
    """
    variants = reachable_variants()
    if only:
        variants = tuple(v for v in variants if v.family in set(only))
    return variants


# --------------------------------------------------------------------------
# Everything below needs the installed aiter; nothing above does, so the module
# stays importable (and testable) on a box with no aiter and no GPU.
# --------------------------------------------------------------------------


def _aiter_jit_core():
    """Flat-import AITER's builder without running ``aiter/__init__.py``.

    ``aiter/setup.py`` does exactly this. Going through ``aiter.jit.core`` instead
    runs the package __init__, which calls ``get_gfx_runtime()`` and imports
    triton -- both fatal with no GPU attached, and with no env-var escape.
    """
    import importlib.util

    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("amd-aiter is not installed")
    aiter_dir = Path(list(spec.submodule_search_locations)[0])
    if str(aiter_dir) not in sys.path:
        sys.path.insert(0, str(aiter_dir))
    from jit import core  # type: ignore[import-not-found]

    return core


def build_recipe(v: Variant, core) -> Dict[str, object]:
    """``{md_name, blob_gen_cmd}`` for one variant.

    varlen goes through AITER's own helper so those builds are byte-identical to
    the lazy ones; the other two families have no helper and are composed here.
    """
    ck_dir = core.CK_DIR
    if v.family == "mha_varlen_fwd":
        from jit.utils.mha_recipes import (  # type: ignore[import-not-found]
            get_mha_varlen_prebuild_variants_by_names,
        )

        recipes = get_mha_varlen_prebuild_variants_by_names(
            [v.md_name], ck_dir, receipt=RECEIPT_VARLEN
        )
        if len(recipes) != 1 or recipes[0]["md_name"] != v.md_name:
            raise RuntimeError(
                f"aiter's varlen recipe helper returned {recipes!r} for {v.md_name}; "
                "its md_name composition has changed"
            )
        return recipes[0]

    if v.family == "mha_fwd":
        direction, receipt, filt = "fwd", RECEIPT_MHA_FWD, _filter_mha_fwd(v)
    else:
        direction, receipt = "batch_prefill", RECEIPT_BATCH_PREFILL
        filt = _filter_batch_prefill(v)
    return {
        "md_name": v.md_name,
        "blob_gen_cmd": [
            f"{ck_dir}/{GENERATE_PY} -d {direction} "
            f"--receipt {receipt} --filter {filt} --output_dir {{}}"
        ],
    }


def jit_dir(core) -> Path:
    return Path(core.get_user_jit_dir())


def _args_of_build(core, family: str) -> Dict[str, object]:
    """AITER's compile flags for a family, keyed as ``compile_ops`` keys them.

    The ``module_`` prefix matters: without it ``get_args_of_build`` warns and
    returns empty args rather than raising, so assert rather than trust.
    """
    args = core.get_args_of_build(f"module_{family}")
    if not args.get("srcs") or not args.get("extra_include"):
        raise RuntimeError(
            f"aiter's optCompilerConfig.json gave no srcs/includes for "
            f"module_{family}; the config key has been renamed"
        )
    return args


def build_module_by_name(md_name: str) -> Path:
    """Build one whole AITER module, e.g. the asm prefill arm."""
    core = _aiter_jit_core()
    args = core.get_args_of_build(md_name)
    if not args.get("srcs"):
        raise RuntimeError(f"aiter's config gave no srcs for {md_name}")
    core.build_module(
        md_name,
        args["srcs"],
        args["flags_extra_cc"],
        args["flags_extra_hip"],
        args["blob_gen_cmd"],
        args["extra_include"],
        args["extra_ldflags"],
        args["verbose"],
        args["is_python_module"],
        args["is_standalone"],
        args.get("torch_exclude", False),
        args.get("third_party", []),
    )
    out = jit_dir(core) / f"{md_name}.so"
    if not out.is_file():
        raise RuntimeError(f"build of {md_name} produced no {out}")
    shutil.rmtree(jit_dir(core) / "build" / md_name, ignore_errors=True)
    return out


def build_one(v: Variant) -> Path:
    """Build a single variant. Returns the artifact path; raises if absent after."""
    core = _aiter_jit_core()
    recipe = build_recipe(v, core)
    args = _args_of_build(core, v.family)
    core.build_module(
        recipe["md_name"],
        args["srcs"],
        args["flags_extra_cc"],
        args["flags_extra_hip"],
        recipe["blob_gen_cmd"],
        args["extra_include"],
        args["extra_ldflags"],
        args["verbose"],
        args["is_python_module"],
        args["is_standalone"],
        args.get("torch_exclude", False),
        args.get("third_party", []),
    )
    out = jit_dir(core) / v.so_name
    if not out.is_file():
        raise RuntimeError(f"build of {v.md_name} produced no {out}")
    # Drop the CK blob and object tree: ~15 MB per variant, 603 MB over the set,
    # and nothing reads it again -- AITER decides "already built" from the .so.
    # Kept on failure, where it is the only diagnostic.
    shutil.rmtree(jit_dir(core) / "build" / v.md_name, ignore_errors=True)
    return out


def _run_jobs(variants: Sequence[Variant], jobs: int, extra: Sequence[str] = ()) -> int:
    """Build in parallel across *processes*.

    Never threads: AITER's build mutates process-global environment and takes a
    process-global lock, so two concurrent builds in one process have the first
    to finish restore the environment under the second.
    """
    pending = list(variants) + list(extra)
    # Child output goes to a file, never a pipe: hipcc is verbose enough to fill
    # a pipe buffer, and a child blocked on a full pipe never exits, so poll()
    # would spin forever. Only read on failure, and only the tail.
    running: List[Tuple[Variant, subprocess.Popen, float, str]] = []
    failed: List[str] = []
    total = len(pending)

    try:
        _drain(pending, running, failed, jobs, total)
    finally:
        for _, proc, _, _ in running:
            proc.kill()
    if failed:
        print(f"\n{len(failed)} build(s) failed:\n  " + "\n  ".join(failed))
    return 1 if failed else 0


def _drain(pending, running, failed, jobs, total) -> None:
    done = 0
    while pending or running:
        while pending and len(running) < jobs:
            v = pending.pop(0)
            name = v if isinstance(v, str) else v.md_name
            fd, log_path = tempfile.mkstemp(prefix=f"prebuild-{name}-", suffix=".log")
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--build-one",
                        v if isinstance(v, str) else _encode(v),
                    ],
                    stdout=fd,
                    stderr=subprocess.STDOUT,
                )
            finally:
                os.close(fd)  # Popen dup'd it; holding ours would exhaust the table
            running.append((v, proc, time.time(), log_path))
        time.sleep(1.0)
        for entry in list(running):
            v, proc, started, log_path = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            done += 1
            label = f"{v}.so" if isinstance(v, str) else v.so_name
            if proc.returncode == 0:
                print(
                    f"[{done}/{total}] ok   {label}  {time.time() - started:.0f}s",
                    flush=True,
                )
                os.unlink(log_path)
            else:
                failed.append(label)
                tail = Path(log_path).read_text(errors="replace").splitlines()[-15:]
                print(
                    f"[{done}/{total}] FAIL {label}  {time.time() - started:.0f}s"
                    f"  (full log: {log_path})\n" + "\n".join(tail),
                    flush=True,
                )


def _encode(v: Variant) -> str:
    return "|".join(
        [
            v.family,
            v.dtype,
            str(int(v.has_logits_cap)),
            str(int(v.needs_mask)),
            str(int(v.has_lse)),
        ]
    )


def _decode(s: str) -> Variant:
    family, dtype, lc, mask, lse = s.split("|")
    return Variant(family, dtype, bool(int(lc)), bool(int(mask)), bool(int(lse)))


def _check(variants: Sequence[Variant]) -> int:
    """Assert every selected artifact is present and non-empty."""
    core = _aiter_jit_core()
    d = jit_dir(core)
    names = [v.so_name for v in variants] + [f"{m}.so" for m in LOADER_MODULES]
    missing = [n for n in names if not (d / n).is_file() or (d / n).stat().st_size == 0]
    print(f"{len(names) - len(missing)}/{len(names)} present in {d}")
    if missing:
        print("missing or empty:\n  " + "\n  ".join(missing))
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--list", action="store_true", help="print the selected set and exit"
    )
    p.add_argument(
        "--check", action="store_true", help="verify artifacts, build nothing"
    )
    p.add_argument(
        "--only",
        action="append",
        choices=sorted(_MD_NAME),
        help="restrict to a family (repeatable)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("AITER_PREBUILD_JOBS", "1")),
        help="concurrent build processes",
    )
    p.add_argument("--build-one", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.build_one:
        if "|" in a.build_one:
            build_one(_decode(a.build_one))
        else:
            build_module_by_name(a.build_one)
        return 0

    variants = selected_variants(a.only)
    if a.list:
        for v in variants:
            print(v.so_name)
        print(f"\n{len(variants)} variant(s)")
        return 0
    if a.check:
        return _check(variants)

    if a.jobs < 1:
        p.error("--jobs must be >= 1")
    t0 = time.time()
    rc = _run_jobs(variants, a.jobs, extra=list(LOADER_MODULES) if not a.only else [])
    print(
        f"\n{len(variants)} variant(s) in {(time.time() - t0) / 60:.0f} min "
        f"at --jobs {a.jobs}"
    )
    return rc or _check(variants)


if __name__ == "__main__":
    raise SystemExit(main())
