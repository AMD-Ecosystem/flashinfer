#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build AITER's attention variant ``.so`` at image-build time, with no GPU.

Without this, AITER's lazy JIT compiles each variant inside the serving process
on the first request that needs it. Artifacts land in AITER's own ``jit/``
directory so its bootstraps find them built -- the store route cannot serve
``mha_batch_prefill``, whose bootstrap doubles as the page-size probe.

Stdlib-only, and reaches AITER by the flat ``sys.path`` import its own
``setup.py`` uses: ``import aiter`` needs a device. The table here is a
deliberate copy of ``flashinfer.jit.rocm.aiter_variants``, which pulls torch;
``tests/rocm/test_prebuild_aiter_attention.py`` holds the two in step.

    GPU_ARCHS="gfx942;gfx950" python3 docker/prebuild_aiter_attention.py --jobs 2
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

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

# Whole AITER modules, no variant axes. A PREBUILD_KERNELS=0 source install ships
# none, so the image builds them or the first caller pays: fmha_v3_fwd throws at
# dlopen, mla_asm and aiter_core only stall.
LOADER_MODULES = ("module_fmha_v3_fwd", "module_aiter_core", "module_mla_asm")


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
    """Every ``.so`` the loader can ask for: 8 + 16 + 20 = 44 per architecture.

    fp8 contributes 4, not 8: AITER ships no LSE instance of the fp8 kernel at
    any page size, and prefill.py raises on the combination rather than
    dispatching it.
    """
    out: List[Variant] = []
    for family, has_logits_axis in _HAS_LOGITS_AXIS.items():
        logits_values = (False, True) if has_logits_axis else (False,)
        for dtype in _FAMILY_DTYPES[family]:
            lse_values = (False,) if dtype == "fp8bf16" else (False, True)
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    for has_lse in lse_values:
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


@contextlib.contextmanager
def _hip_clang_path(args: Dict) -> Iterator[None]:
    """Export ``HIP_CLANG_PATH`` the way AITER's ``compile_ops`` does.

    Only ``compile_ops`` sets it; calling ``build_module`` directly would leave
    the mha families on a different clang from their lazy builds.
    """
    path = args.get("hip_clang_path")
    if not path:
        yield
        return
    prev = os.environ.get("HIP_CLANG_PATH")
    os.environ["HIP_CLANG_PATH"] = path
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HIP_CLANG_PATH", None)
        else:
            os.environ["HIP_CLANG_PATH"] = prev


def build_module_by_name(md_name: str) -> Path:
    """Build one whole AITER module, e.g. the asm prefill arm."""
    core = _aiter_jit_core()
    args = core.get_args_of_build(md_name)
    if not args.get("srcs"):
        raise RuntimeError(f"aiter's config gave no srcs for {md_name}")
    with _hip_clang_path(args):
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
    with _hip_clang_path(args):
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
    pending = list(extra) + list(variants)
    # Child output goes to a file, never a pipe: hipcc is verbose enough to fill
    # a pipe buffer, and a child blocked on a full pipe never exits, so poll()
    # would spin forever. Only read on failure, and only the tail.
    running: List[Tuple[Variant, subprocess.Popen, float, str]] = []
    failed: List[str] = []
    total = len(pending)

    try:
        _drain(pending, running, failed, jobs, total)
    finally:
        for _, proc, _, log_path in running:
            proc.kill()
            # wait(), not just kill(): AITER's baton is only stale when
            # kill(pid, 0) fails, which a zombie answers successfully, so an
            # unreaped child makes the next build for that name hang.
            proc.wait()
            os.unlink(log_path)
    if failed:
        print(f"\n{len(failed)} build(s) failed:\n  " + "\n  ".join(failed))
    return 1 if failed else 0


def _child_env(jobs: int) -> Dict[str, str]:
    """Divide ninja's parallelism across the children rather than per child.

    Left alone, each child sets MAX_JOBS from ~80% of *all* host CPUs and a
    free-memory snapshot it took independently, so --jobs N oversubscribes the
    box N-fold. AITER honours MAX_JOBS when it is already set.
    """
    env = dict(os.environ)
    try:
        budget = int(env["MAX_JOBS"])
    except (KeyError, ValueError):
        budget = int((os.cpu_count() or 1) * 0.8)
    env["MAX_JOBS"] = str(max(1, budget // max(1, jobs)))
    return env


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
                    env=_child_env(jobs),
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


_MIN_CK_MARKERS = 200


def _has_ck_instances(path: Path) -> bool:
    """Does this artifact carry CK-tile kernel instances, or only a dispatcher?

    Only meaningful for the CK variants: LOADER_MODULES are built -DENABLE_CK=0.
    """
    marker, seen, tail = b"ck_tile", 0, b""
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            seen += (tail + chunk).count(marker)
            if seen >= _MIN_CK_MARKERS:
                return True
            tail = chunk[-(len(marker) - 1) :]
    return False


def _check(variants: Sequence[Variant], extra: Sequence[str]) -> int:
    """Assert every selected artifact is present and non-empty.

    ``extra`` must be the module set the same invocation builds, or a ``--only``
    run fails its own check on artifacts it was never asked to produce.
    """
    core = _aiter_jit_core()
    d = jit_dir(core)
    names = [v.so_name for v in variants] + [f"{m}.so" for m in extra]
    missing = [n for n in names if not (d / n).is_file() or (d / n).stat().st_size == 0]
    print(f"{len(names) - len(missing)}/{len(names)} present in {d}")
    if missing:
        print("missing or empty:\n  " + "\n  ".join(missing))
        return 1
    # A --filter that selected no instances still compiles and links, so presence
    # is not enough: the result is a dispatcher with no kernel behind it.
    hollow = [v.so_name for v in variants if not _has_ck_instances(d / v.so_name)]
    if hollow:
        print(
            f"{len(hollow)} artifact(s) carry no CK instances -- the filter "
            "selected nothing:\n  " + "\n  ".join(hollow)
        )
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
        # str, validated in main(): an int() here runs while the parser is being
        # built, so a bad AITER_PREBUILD_JOBS would crash before --help works.
        default=os.environ.get("AITER_PREBUILD_JOBS", "1"),
        help="concurrent build processes",
    )
    p.add_argument(
        "--print-jit-dir",
        action="store_true",
        help="print the directory artifacts are written to, and exit",
    )
    p.add_argument("--build-one", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.print_jit_dir:
        print(jit_dir(_aiter_jit_core()))
        return 0

    if a.build_one:
        if "|" in a.build_one:
            build_one(_decode(a.build_one))
        else:
            build_module_by_name(a.build_one)
        return 0

    variants = selected_variants(a.only)
    # The whole-module set has no family axis, so --only excludes it.
    extra = [] if a.only else list(LOADER_MODULES)
    if a.list:
        for name in [v.so_name for v in variants] + [f"{m}.so" for m in extra]:
            print(name)
        print(f"\n{len(variants)} variant(s) + {len(extra)} module(s)")
        return 0
    if a.check:
        return _check(variants, extra)

    try:
        a.jobs = int(a.jobs)
    except ValueError:
        p.error(f"--jobs/AITER_PREBUILD_JOBS must be an integer, got {a.jobs!r}")
    if a.jobs < 1:
        p.error("--jobs must be >= 1")
    t0 = time.time()
    rc = _run_jobs(variants, a.jobs, extra=extra)
    print(
        f"\n{len(variants)} variant(s) + {len(extra)} module(s) in "
        f"{(time.time() - t0) / 60:.0f} min at --jobs {a.jobs}"
    )
    # Both, not `rc or`: a run that lost one variant should still report every
    # other artifact that linked hollow, rather than one finding per rebuild.
    return _check(variants, extra) or rc


if __name__ == "__main__":
    raise SystemExit(main())
