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

RMSNorm native-vs-AITER sweep: the evidence behind the ``backend="auto"`` policy
in ``flashinfer/rocm/norm.py``.

Covers ``fused_add_rmsnorm`` and ``rmsnorm`` in both its ``out=None`` and
``out=x`` forms -- since PR #331 those are separate code paths, because CK cannot
alias its output onto its input and the shim only stages when the caller aliased.

Two shapes are deliberately in the sweep and easy to drop by accident:
ill-aligned ``hidden_size`` (the native kernel's ``vec_size`` is
``gcd(16/sizeof(T), d)``, so d=111 goes fully scalar and d=500 drops to 4), and
the device-derived shared-memory staging pair, above which the *native* fused
kernel re-reads from global instead of staging the row.

``--aa`` runs native against itself to establish the noise floor. Read it before
believing any ratio: a margin inside the A/A spread is not a result.

Run:
    python benchmarks/rocm_benchmarks/bench_norm.py --aa
    python benchmarks/rocm_benchmarks/bench_norm.py --csv norm-gfx942.csv
    python benchmarks/rocm_benchmarks/bench_norm.py --accuracy
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
from flashinfer.aiter_utils import is_aiter_available
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.testing import bench_gpu_time

_jit_logger.setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ROWS = [19, 128, 989, 2048, 8192, 65536]
# 111 and 500 collapse the native kernel's vec_size (gcd(16/sizeof(T), d)) to 1
# and 4; a powers-of-two-only sweep hides that and flatters native.
_HIDDEN = [111, 500, 3072, 3584, 4096, 5120, 8192, 11008, 14336]
_DTYPES = [(torch.float16, "f16"), (torch.bfloat16, "bf16")]
# Skip corners that would allocate tens of GB; AITER needs two extra full-size
# buffers on top of the caller's.
_MAX_ELEMS = 128 * 1024 * 1024


def _assert_import_provenance() -> Path:
    """The editable install and the pinned worktree are easy to confuse, and the
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


def _staging_pair(dtype: torch.dtype) -> tuple[int, int]:
    """The largest hidden_size the native fused kernel stages in LDS, and the
    next one it cannot. 16352/16360 on CDNA3, 40928/40936 on CDNA4.

    Mirrors FusedRMSNormSmemSize in include/flashinfer/rocm/attention/norm.cuh.
    """
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block
    vec = math.gcd(16 // torch.tensor([], dtype=dtype).element_size(), 8)
    # At the cliff d is large, so block_size saturates at kMaxBlockSize and
    # num_warps is 1024/32; the reduction scratch is then a constant 128 B.
    num_warps = 1024 // 32
    reduce_bytes = -(-num_warps // 4) * 4 * 4
    fits = (smem - reduce_bytes) // 4
    fits -= fits % vec  # stay vec-aligned, or vec_size changes too
    return fits, fits + vec


def _provenance() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        import importlib.metadata as md

        aiter_ver = md.version("amd-aiter")
    except Exception:  # noqa: BLE001 - absent or unreadable is a valid answer
        aiter_ver = "absent"
    return {
        "flashinfer": str(_assert_import_provenance()),
        "git": _git_describe(),
        "arch": props.gcnArchName,
        "smem_per_block": props.shared_memory_per_block,
        "torch": torch.__version__,
        "aiter": aiter_ver,
    }


def _time_us(fn, dry_run_iters: int, repeat_iters: int) -> tuple[float, float]:
    """Median and p95-p05 spread in microseconds.

    ``fn`` is the bare op: refreshing inputs inside it would land in the timed
    window and cost about as much as the op. fused_add_rmsnorm accumulates into
    ``residual`` across reps, which is a self-limiting random walk -- run --aa to
    confirm that does not move the timing before trusting a ratio.
    """
    times = bench_gpu_time(fn, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters)
    times = sorted(float(t) * 1000.0 for t in times)
    lo = times[max(0, int(0.05 * len(times)) - 1)]
    hi = times[min(len(times) - 1, int(0.95 * len(times)))]
    return statistics.median(times), hi - lo


def _make_case(op: str, rows: int, hidden: int, dtype: torch.dtype, backend: str):
    """Return a zero-arg closure for one (op, shape, dtype, backend)."""
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    if op == "fused_add_rmsnorm":
        residual = torch.randn_like(x)
        return lambda: flashinfer.fused_add_rmsnorm(
            x, residual, w, 1e-6, backend=backend
        )
    if op == "rmsnorm_out_none":
        return lambda: flashinfer.rmsnorm(x, w, 1e-6, backend=backend)
    if op == "rmsnorm_out_aliased":
        return lambda: flashinfer.rmsnorm(x, w, 1e-6, out=x, backend=backend)
    raise ValueError(f"unknown op {op!r}")


def _shapes(op: str, dtype: torch.dtype) -> list[tuple[int, int]]:
    hidden = list(_HIDDEN)
    if op == "fused_add_rmsnorm":
        # Only the fused kernels stage the row, so the cliff exists only here.
        hidden += list(_staging_pair(dtype))
    out = []
    for h in sorted(set(hidden)):
        for r in _ROWS:
            if r * h <= _MAX_ELEMS:
                out.append((r, h))
    return out


def _warmup(ops, backends, aa: bool) -> None:
    """Force every JIT module to build before anything is timed. The first call
    on a cold cache compiles, and that lands in the first timed arm."""
    for op in ops:
        for backend in backends:
            real = "native" if aa else backend
            try:
                _make_case(op, 128, 4096, torch.float16, real)()
                _make_case(op, 128, 4096, torch.bfloat16, real)()
            except Exception:  # noqa: BLE001 - unsupported combos fail later, visibly
                pass
    torch.cuda.synchronize()


def _sweep(ops, backends, dry_run_iters, repeat_iters, aa: bool) -> list[dict]:
    rows = []
    _warmup(ops, backends, aa)
    for op in ops:
        for dtype, dt_name in _DTYPES:
            for r, h in _shapes(op, dtype):
                rec = {"op": op, "dtype": dt_name, "rows": r, "hidden": h}
                # Warm BOTH arms before timing EITHER. Timing them in order
                # otherwise measures arm 1 on a colder clock: in A/A that showed
                # as a systematic ~16% penalty to whichever ran first, which
                # would flatter the second backend.
                fns = {}
                for backend in backends:
                    real_backend = "native" if aa else backend
                    try:
                        fns[backend] = _make_case(op, r, h, dtype, real_backend)
                        for _ in range(dry_run_iters):
                            fns[backend]()
                    except Exception as exc:  # noqa: BLE001 - a refusal is a result
                        rec[f"{backend}_err"] = f"{type(exc).__name__}: {exc}"[:160]
                torch.cuda.synchronize()

                for backend in backends:
                    if backend not in fns:
                        rec[f"{backend}_us"] = None
                        continue
                    try:
                        med, spread = _time_us(
                            fns[backend], dry_run_iters, repeat_iters
                        )
                        rec[f"{backend}_us"] = round(med, 3)
                        rec[f"{backend}_spread_us"] = round(spread, 3)
                    except Exception as exc:  # noqa: BLE001 - a refusal is a result
                        rec[f"{backend}_us"] = None
                        rec[f"{backend}_err"] = f"{type(exc).__name__}: {exc}"[:160]
                    finally:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                a, b = rec.get(f"{backends[0]}_us"), rec.get(f"{backends[1]}_us")
                rec["ratio"] = round(b / a, 4) if a and b else None
                rows.append(rec)
                errs = [rec[k] for k in rec if k.endswith("_err")]
                note = f"  !! {errs[0]}" if errs else ""
                print(
                    f"{op:22s} {rec['dtype']:4s} {r:6d}x{h:<6d} "
                    f"{backends[0]}={a} {backends[1]}={b} ratio={rec['ratio']}{note}",
                    flush=True,
                )
    return rows


def _accuracy(ops) -> None:
    """Max abs error of AITER against a float32 reference, per shape."""
    for op in ops:
        for dtype, dt_name in _DTYPES:
            for r, h in _shapes(op, dtype):
                x = torch.randn(r, h, dtype=dtype, device="cuda")
                w = torch.randn(h, dtype=dtype, device="cuda")
                f32 = x.float()
                if op == "fused_add_rmsnorm":
                    res = torch.randn_like(x)
                    f32 = f32 + res.float()
                var = f32.pow(2).mean(-1, keepdim=True)
                ref = (f32 * torch.rsqrt(var + 1e-6) * w.float()).to(dtype)
                xa = x.clone()
                try:
                    if op == "fused_add_rmsnorm":
                        flashinfer.fused_add_rmsnorm(
                            xa, res.clone(), w, 1e-6, backend="aiter"
                        )
                        got = xa
                    elif op == "rmsnorm_out_aliased":
                        flashinfer.rmsnorm(xa, w, 1e-6, out=xa, backend="aiter")
                        got = xa
                    else:
                        got = flashinfer.rmsnorm(xa, w, 1e-6, backend="aiter")
                    err = (got.float() - ref.float()).abs().max().item()
                    print(f"{op:22s} {dt_name:4s} {r:6d}x{h:<6d} max_abs_err={err:.5f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{op:22s} {dt_name:4s} {r:6d}x{h:<6d} FAILED: {exc}")
                finally:
                    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ops",
        nargs="+",
        default=["fused_add_rmsnorm", "rmsnorm_out_none", "rmsnorm_out_aliased"],
    )
    ap.add_argument("--dry-run-iters", type=int, default=25)
    ap.add_argument("--repeat-iters", type=int, default=200)
    ap.add_argument(
        "--aa",
        action="store_true",
        help="native vs native: the noise floor any ratio must clear",
    )
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    prov = _provenance()
    for k, v in prov.items():
        print(f"# {k}: {v}")

    # --accuracy is AITER-only, so it needs the same guard as the A/B sweep.
    # Capability is per-op, so ask about the ops actually selected.
    caps = {"fused_add_rmsnorm": "fused_add_rmsnorm"}
    dev = torch.device("cuda:0")
    if not args.aa and not any(
        is_aiter_available(dev, caps.get(op, "rmsnorm")) for op in args.ops
    ):
        raise SystemExit(
            f"AITER unavailable for {args.ops}; nothing to compare. Use --aa."
        )

    if args.accuracy:
        _accuracy(args.ops)
        return

    backends = ["native", "native2"] if args.aa else ["native", "aiter"]
    rows = _sweep(args.ops, backends, args.dry_run_iters, args.repeat_iters, args.aa)

    if args.csv:
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f"# wrote {args.csv} ({len(rows)} rows)")

    ratios = sorted(r["ratio"] for r in rows if r["ratio"])
    if ratios:
        label = "A/A spread" if args.aa else "aiter/native"
        p = lambda q: ratios[min(len(ratios) - 1, int(q * len(ratios)))]  # noqa: E731
        print(
            f"# {label}: n={len(ratios)} median={statistics.median(ratios):.4f} "
            f"p05={p(0.05):.4f} p95={p(0.95):.4f} min={ratios[0]:.4f} "
            f"max={ratios[-1]:.4f}"
        )
        # The decision rule keys off this: anything inside it is not a result.
        outliers = [r for r in rows if r["ratio"] and abs(r["ratio"] - 1.0) > 0.10]
        for r in sorted(outliers, key=lambda r: -abs(r["ratio"] - 1.0))[:10]:
            print(
                f"#   outlier {r['op']:22s} {r['dtype']:4s} "
                f"{r['rows']:6d}x{r['hidden']:<6d} ratio={r['ratio']}"
            )


if __name__ == "__main__":
    main()
