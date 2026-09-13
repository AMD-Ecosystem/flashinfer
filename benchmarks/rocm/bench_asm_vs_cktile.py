"""
Copyright (c) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AITER asm-vs-CK-Tile prefill sweep: the evidence behind
``_AITER_ASM_PREFILL_MIN_QO_LEN`` in ``flashinfer/rocm/arch_caps.py``.

Measures AITER's two forward kernels against each other directly rather than
through ``single_prefill_with_kv_cache``. The shim's overhead is identical on
both arms, and the routing threshold is a statement about the kernels, so this
is the unit the decision is actually made in. It also sidesteps the fact that
``FLASHINFER_AITER_ASM_PREFILL`` is read once into a C++ static and so cannot be
toggled inside one process.

The sweep is two-dimensional on purpose. asm needs enough q-side parallelism to
fill the device, so a seqlen-only sweep at one batch/head count finds a
threshold that does not generalise -- batch 1 with 16 heads and batch 4 with 64
heads sit at opposite ends of the same effect.

``--aa`` runs CK Tile against itself to establish the noise floor. Read it before
believing any ratio: a margin inside the A/A spread is not a result. Since the
allocator-cache fix the floor is roughly +/-1.3%, tight enough that every
per-seqlen geomean in the sweep now sits outside it.

Run:
    python benchmarks/rocm/bench_asm_vs_cktile.py --aa
    python benchmarks/rocm/bench_asm_vs_cktile.py --csv asm-gfx950.csv
    python benchmarks/rocm/bench_asm_vs_cktile.py --accuracy
"""

import argparse
import csv
import logging
import math
import statistics
import subprocess
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.rocm.aiter_utils import is_aiter_available
from flashinfer.testing import bench_gpu_time

_jit_logger.setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixed traits: the asm arm only exists for bf16 at head_dim 128 with no bias,
# no dropout and no soft cap, so varying them would only add skipped rows.
_HEAD_DIM = 128
_GQA_RATIO = 4
_SEQLENS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144]
_BATCHES = [1, 4]
_QO_HEADS = [16, 32, 64]
# Large enough to keep b4/hq64/s6144 in: pruning it would make this script
# sweep 59 cells while the recorded evidence cites 60.
_MAX_ELEMS = 256 * 1024 * 1024


def _assert_import_provenance() -> Path:
    """The editable install and a pinned worktree are easy to confuse, and the
    wrong one rebuilds cleanly -- so fail loudly rather than measure it."""
    got = Path(flashinfer.__file__).resolve()
    if _REPO_ROOT not in got.parents:
        raise SystemExit(
            f"flashinfer resolved to {got}, which is outside this checkout "
            f"({_REPO_ROOT}). Set PYTHONPATH to the tree you mean to measure."
        )
    return got


def _git_describe() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _provenance() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        import importlib.metadata as md

        aiter_ver = md.version("amd-aiter")
    except Exception:  # noqa: BLE001 - absent or unreadable is a valid answer
        aiter_ver = "absent"
    from flashinfer.rocm.arch_caps import aiter_asm_prefill_min_qo_len, normalize_arch

    arch = normalize_arch(props.gcnArchName)
    return {
        "flashinfer": str(_assert_import_provenance()),
        "git": _git_describe(),
        "arch": props.gcnArchName,
        "cu_count": props.multi_processor_count,
        "torch": torch.__version__,
        # Kernel selection and timing are both ROCm-version sensitive, so two runs
        # are only comparable when this matches.
        "hip": torch.version.hip or "none",
        "aiter": aiter_ver,
        "shipping_threshold": aiter_asm_prefill_min_qo_len(arch),
    }


def _time_us(fn, dry_run_iters: int, repeat_iters: int) -> tuple[float, float]:
    """Median and p95-p05 spread in microseconds."""
    times = bench_gpu_time(fn, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters)
    times = sorted(float(t) * 1000.0 for t in times)
    lo = times[max(0, int(0.05 * len(times)) - 1)]
    hi = times[min(len(times) - 1, int(0.95 * len(times)))]
    return statistics.median(times), hi - lo


def _make_case(batch: int, seqlen: int, hq: int, causal: bool, arm: str):
    """Zero-arg closure for one (shape, arm). q/k/v are bshd, as both ops expect."""
    from aiter.ops.mha import fmha_v3_fwd, mha_fwd

    hk = max(1, hq // _GQA_RATIO)
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(
        batch, seqlen, hq, _HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    k = torch.randn(
        batch, seqlen, hk, _HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    v = torch.randn(
        batch, seqlen, hk, _HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scale = _HEAD_DIM**-0.5

    if arm == "asm":
        # how_v3_bf16_cvt is 0 to match what single_prefill.cuh sends. It selects a
        # different .co on gfx942 and is ignored on gfx950, so it is not a free knob.
        return lambda: fmha_v3_fwd(
            q, k, v, 0.0, scale, causal, -1, -1, False, False, 0
        )[0]
    return lambda: mha_fwd(q, k, v, 0.0, scale, causal, -1, -1, 0, False, False)[0]


def _shapes():
    for batch in _BATCHES:
        for hq in _QO_HEADS:
            for seqlen in _SEQLENS:
                if batch * seqlen * hq * _HEAD_DIM <= _MAX_ELEMS:
                    yield batch, seqlen, hq


def _accuracy(causal: bool) -> None:
    """Max abs error of each arm against an fp32 reference.

    Worth its own mode: the arms do not agree bit-for-bit, and on gfx942 the asm
    kernel is the more accurate of the two, which a timing table hides.
    """
    print(f"{'shape':<22}{'ck err':>12}{'asm err':>12}")
    for batch, seqlen, hq in _shapes():
        # The reference materializes [batch, hq, seqlen, seqlen] fp32 three times
        # over; _MAX_ELEMS bounds q, not this, so cap it separately.
        if batch * hq * seqlen * seqlen > 2 * 1024**3 // 4:
            print(
                f"b{batch} hq{hq} s{seqlen}".ljust(22)
                + f"{'skipped: ref too large':>24}"
            )
            continue
        hk = max(1, hq // _GQA_RATIO)
        g = torch.Generator(device="cuda").manual_seed(7)
        shape_q = (batch, seqlen, hq, _HEAD_DIM)
        shape_kv = (batch, seqlen, hk, _HEAD_DIM)
        q = torch.randn(*shape_q, dtype=torch.bfloat16, device="cuda", generator=g)
        k = torch.randn(*shape_kv, dtype=torch.bfloat16, device="cuda", generator=g)
        v = torch.randn(*shape_kv, dtype=torch.bfloat16, device="cuda", generator=g)
        rep = hq // hk
        qf = q.float().permute(0, 2, 1, 3)
        kf = k.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)
        vf = v.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)
        s = torch.matmul(qf, kf.transpose(-1, -2)) / (_HEAD_DIM**0.5)
        if causal:
            idx = torch.arange(seqlen, device=q.device)
            s = s.masked_fill(idx.unsqueeze(0) > idx.unsqueeze(1), float("-inf"))
        ref = torch.matmul(torch.softmax(s, dim=-1), vf).permute(0, 2, 1, 3)
        cells = []
        for arm in ("ck", "asm"):
            try:
                out = _make_case(batch, seqlen, hq, causal, arm)()
                cells.append(f"{(out.float() - ref).abs().max().item():>12.5f}")
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                cells.append(f"{type(exc).__name__:>12s}")
        print(f"b{batch} hq{hq} s{seqlen}".ljust(22) + "".join(cells), flush=True)


def _sweep(causal: bool, dry_run_iters: int, repeat_iters: int, aa: bool) -> list[dict]:
    arms = ["ck", "asm"]
    rows = []
    for batch, seqlen, hq in _shapes():
        rec = {"batch": batch, "seqlen": seqlen, "qo_heads": hq, "causal": causal}
        # Warm BOTH arms before timing EITHER: timing them in order otherwise
        # measures the first on a colder clock, which in A/A showed as a
        # systematic penalty to whichever ran first.
        fns = {}
        for arm in arms:
            real_arm = "ck" if aa else arm
            try:
                fns[arm] = _make_case(batch, seqlen, hq, causal, real_arm)
                for _ in range(dry_run_iters):
                    fns[arm]()
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                rec[f"{arm}_err"] = f"{type(exc).__name__}: {exc}"[:160]
        torch.cuda.synchronize()

        for arm in arms:
            if arm not in fns:
                rec[f"{arm}_us"] = None
                continue
            try:
                med, spread = _time_us(fns[arm], dry_run_iters, repeat_iters)
                rec[f"{arm}_us"] = round(med, 3)
                rec[f"{arm}_spread_us"] = round(spread, 3)
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                rec[f"{arm}_us"] = None
                rec[f"{arm}_err"] = f"{type(exc).__name__}: {exc}"[:160]
            finally:
                torch.cuda.synchronize()

        # Between shapes, never between arms: releasing segments mid-shape makes the
        # second arm re-acquire through hipMalloc and pay for it in the ratio.
        torch.cuda.empty_cache()

        ck, asm = rec.get("ck_us"), rec.get("asm_us")
        rec["speedup"] = round(ck / asm, 4) if ck and asm else None
        rows.append(rec)
        errs = [rec[k] for k in rec if k.endswith("_err")]
        note = f"  !! {errs[0]}" if errs else ""
        print(
            f"b{batch} hq{hq:<3d} s{seqlen:<5d} causal={int(causal)}  "
            f"ck={ck} asm={asm} speedup={rec['speedup']}{note}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aa",
        action="store_true",
        help="CK against itself: the noise floor any ratio must clear",
    )
    ap.add_argument(
        "--accuracy", action="store_true", help="max abs error vs fp32, then exit"
    )
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--dry-run-iters", type=int, default=25)
    ap.add_argument("--repeat-iters", type=int, default=200)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    if not is_aiter_available(dev, "single_prefill"):
        raise SystemExit(
            "AITER cannot serve single_prefill here; both arms are AITER kernels."
        )

    for key, value in _provenance().items():
        print(f"# {key}: {value}")

    if args.accuracy:
        _accuracy(args.causal)
        return

    rows = _sweep(args.causal, args.dry_run_iters, args.repeat_iters, args.aa)

    if args.csv:
        fieldnames = sorted({k for r in rows for k in r})
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"# wrote {args.csv}")

    ratios = sorted(r["speedup"] for r in rows if r["speedup"])
    if not ratios:
        return
    label = "A/A spread" if args.aa else "asm/ck"
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    print(
        f"# {label}: n={len(ratios)} geomean={geo:.3f} min={ratios[0]:.3f} max={ratios[-1]:.3f}"
    )
    print(
        f"#   p05={ratios[int(0.05 * len(ratios))]:.3f} median={statistics.median(ratios):.3f} "
        f"p95={ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]:.3f}"
    )
    regressions = [r for r in rows if r["speedup"] and r["speedup"] < 1.0]
    print(f"# cells below 1.00: {len(regressions)}/{len(ratios)}")
    for r in sorted(regressions, key=lambda r: r["speedup"])[:10]:
        print(f"#   b{r['batch']} hq{r['qo_heads']} s{r['seqlen']}: {r['speedup']}")


if __name__ == "__main__":
    main()
