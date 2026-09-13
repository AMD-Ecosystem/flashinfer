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

POD attention against running prefill and decode separately.

This is the continuous-batching question: a serving step holds one long prefill
plus many one-token decodes, and POD fuses them into a single kernel launch so
the decodes ride along in the prefill's idle CUs. The upstream
``benchmarks/bench_mixed_attention.py`` compares three arms; the third is
``flashinfer.BatchAttention``, which is gated CUDA-only here, so this keeps the
two that exist on ROCm.

The sweep is over the decode batch size at a fixed prefill, because that is what
moves: the prefill arm is constant in it, the decode arm is linear, and POD is
supposed to be flat until the decodes stop fitting alongside. Reading a single
mix tells you nothing about whether fusing was worth it.

Run:
    python benchmarks/rocm/bench_mixed_attention.py
    python benchmarks/rocm/bench_mixed_attention.py --csv pod-gfx942.csv
    python benchmarks/rocm/bench_mixed_attention.py --accuracy
"""

import argparse
import csv
import logging
import statistics
import subprocess
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.testing import bench_gpu_time

_jit_logger.setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DECODE_BATCHES = [1, 8, 32, 128]
_PREFILL_LENS = [1024, 4096]
_DECODE_KV_LEN = 4096
_PAGE_SIZE = 16
_HEAD_DIM = 128
_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 8
_DTYPE = torch.bfloat16
_WORKSPACE_BYTES = 256 * 1024 * 1024


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


def _provenance() -> dict:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        import importlib.metadata as md

        aiter_ver = md.version("amd-aiter")
    except Exception:  # noqa: BLE001 - absent or unreadable is a valid answer
        aiter_ver = "absent"
    return {
        "flashinfer": str(_assert_import_provenance()),
        "git": _git_describe(),
        "arch": props.gcnArchName,
        "rocm": torch.version.hip,
        "torch": torch.__version__,
        # --decode-backend auto can route the split arm to AITER, so its version
        # is part of what makes a number reproducible.
        "aiter": aiter_ver,
    }


def _time_us(fn, dry_run_iters: int, repeat_iters: int) -> tuple[float, float]:
    """Median and p95-p05 spread in microseconds."""
    times = bench_gpu_time(fn, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters)
    times = sorted(float(t) * 1000.0 for t in times)
    lo = times[max(0, int(0.05 * len(times)) - 1)]
    hi = times[min(len(times) - 1, int(0.95 * len(times)))]
    return statistics.median(times), hi - lo


def _build(decode_bs: int, prefill_len: int, causal: bool, decode_backend: str):
    """Tensors and both arms' wrappers for one (decode_bs, prefill_len) mix.

    Each stream carries its own KV tensor and its own 0-based page indices: the
    wrapper indexes the prefill indices into the prefill tensor, so sharing one
    cache and offsetting reads past the end and faults the GPU.
    """
    dev = torch.device("cuda")
    d_pages = decode_bs * (_DECODE_KV_LEN // _PAGE_SIZE)
    p_pages = prefill_len // _PAGE_SIZE
    shape = (2, _PAGE_SIZE, _NUM_KV_HEADS, _HEAD_DIM)
    kv_d = torch.randn(d_pages, *shape, dtype=_DTYPE, device=dev)
    kv_p = torch.randn(p_pages, *shape, dtype=_DTYPE, device=dev)
    q_d = torch.randn(decode_bs, _NUM_QO_HEADS, _HEAD_DIM, dtype=_DTYPE, device=dev)
    q_p = torch.randn(prefill_len, _NUM_QO_HEADS, _HEAD_DIM, dtype=_DTYPE, device=dev)

    d_q_indptr = torch.arange(decode_bs + 1, dtype=torch.int32, device=dev)
    d_kv_indptr = d_q_indptr * (_DECODE_KV_LEN // _PAGE_SIZE)
    d_kv_indices = torch.arange(d_pages, dtype=torch.int32, device=dev)
    d_last = torch.full((decode_bs,), _PAGE_SIZE, dtype=torch.int32, device=dev)

    p_q_indptr = torch.tensor([0, prefill_len], dtype=torch.int32, device=dev)
    p_kv_indptr = torch.tensor([0, p_pages], dtype=torch.int32, device=dev)
    p_kv_indices = torch.arange(p_pages, dtype=torch.int32, device=dev)
    p_last = torch.full((1,), _PAGE_SIZE, dtype=torch.int32, device=dev)

    common = dict(
        num_qo_heads=_NUM_QO_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        page_size=_PAGE_SIZE,
        q_data_type=_DTYPE,
        kv_data_type=_DTYPE,
    )

    pod_ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    pod = flashinfer.BatchPODWithPagedKVCacheWrapper(pod_ws, kv_layout="NHD")
    pod.plan(
        p_q_indptr,
        p_kv_indptr,
        p_kv_indices,
        p_last,
        d_q_indptr,
        d_kv_indptr,
        d_kv_indices,
        d_last,
        **common,
    )

    # Separate arms get their own workspaces: sharing one would serialise them
    # on the planning buffers and measure the sharing, not the split.
    pre_ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        pre_ws, "NHD", backend="fa2"
    )
    # The prefill wrapper spells head_dim head_dim_qk, so it cannot share the
    # POD kwargs above.
    prefill.plan(
        p_q_indptr,
        p_kv_indptr,
        p_kv_indices,
        p_last,
        num_qo_heads=_NUM_QO_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim_qk=_HEAD_DIM,
        page_size=_PAGE_SIZE,
        causal=causal,
        q_data_type=_DTYPE,
        kv_data_type=_DTYPE,
    )
    dec_ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        dec_ws, "NHD", backend=decode_backend
    )
    decode.plan(d_kv_indptr, d_kv_indices, d_last, **common)

    def run_pod():
        return pod.run(q_p, kv_p, q_d, kv_d, causal_p=causal)

    def run_split():
        return prefill.run(q_p, kv_p), decode.run(q_d, kv_d)

    # POD runs the HIP kernel for both halves, so a split arm that resolved to
    # AITER is measuring backend choice as well as fusion. Report which ran.
    return {"pod": run_pod, "split": run_split, "split_decode": decode.backend}


def _sweep(
    dry_run_iters: int, repeat_iters: int, causal: bool, decode_backend: str
) -> list:
    rows = []
    for prefill_len in _PREFILL_LENS:
        for decode_bs in _DECODE_BATCHES:
            rec = {
                "prefill_len": prefill_len,
                "decode_bs": decode_bs,
                "decode_kv_len": _DECODE_KV_LEN,
                "causal": causal,
                "decode_backend": decode_backend,
            }
            try:
                fns = _build(decode_bs, prefill_len, causal, decode_backend)
                rec["split_decode_resolved"] = fns["split_decode"]
                for arm in ("pod", "split"):
                    med, spread = _time_us(fns[arm], dry_run_iters, repeat_iters)
                    rec[f"{arm}_us"] = round(med, 3)
                    rec[f"{arm}_spread_us"] = round(spread, 3)
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                rec["err"] = f"{type(exc).__name__}: {exc}"[:160]
            finally:
                # A hard fault surfaces here, outside the try above, and would
                # otherwise discard every row measured so far.
                try:
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    rec.setdefault("err", f"{type(exc).__name__}: {exc}"[:160])
                # Drop the closures first: they own this case's KV cache and
                # three workspaces, so empty_cache() before this frees none of
                # it and the peak holds two cases at once.
                fns = None
                torch.cuda.empty_cache()
            p, s = rec.get("pod_us"), rec.get("split_us")
            rec["speedup"] = round(s / p, 4) if p and s else None
            rows.append(rec)
            note = f"  !! {rec['err']}" if "err" in rec else ""
            print(
                f"prefill={prefill_len:<6d} decode_bs={decode_bs:<4d} "
                f"pod={p} split={s} speedup={rec['speedup']} "
                f"split_decode={rec.get('split_decode_resolved')}{note}",
                flush=True,
            )
    return rows


def _accuracy(causal: bool, decode_backend: str) -> None:
    """POD's two outputs against the separate wrappers', per mix.

    Only a reference at --decode-backend fa2, where both arms call the same HIP
    kernels and a mismatch is POD's scheduling. Under auto the split decode may
    run AITER, so the difference is cross-backend and says nothing about POD.
    """
    for prefill_len in _PREFILL_LENS:
        for decode_bs in _DECODE_BATCHES:
            try:
                fns = _build(decode_bs, prefill_len, causal, decode_backend)
                o_p, o_d = fns["pod"]()
                r_p, r_d = fns["split"]()
                err_p = (o_p.float() - r_p.float()).abs().max().item()
                err_d = (o_d.float() - r_d.float()).abs().max().item()
                print(
                    f"prefill={prefill_len:<6d} decode_bs={decode_bs:<4d} "
                    f"prefill_err={err_p:.5f} decode_err={err_d:.5f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"prefill={prefill_len:<6d} decode_bs={decode_bs:<4d} FAILED: {exc}"
                )
            finally:
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run-iters", type=int, default=25)
    ap.add_argument("--repeat-iters", type=int, default=100)
    # Causal by default: a prefill in a serving step is. --no-causal is the
    # off switch; there is deliberately no --causal, which could never do anything.
    ap.add_argument("--no-causal", dest="causal", action="store_false", default=True)
    ap.add_argument(
        "--decode-backend",
        choices=["fa2", "auto"],
        default="fa2",
        help="Split arm's decode backend. fa2 matches what POD runs internally, "
        "so the ratio isolates fusion; auto answers the production question "
        "of whether to use POD at all.",
    )
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    for key, value in _provenance().items():
        print(f"# {key}: {value}")

    if args.accuracy:
        if args.decode_backend != "fa2":
            print(
                "# NOTE --accuracy compares across backends unless --decode-backend fa2"
            )
        _accuracy(args.causal, args.decode_backend)
        return

    rows = _sweep(
        args.dry_run_iters, args.repeat_iters, args.causal, args.decode_backend
    )

    if args.csv:
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"# wrote {args.csv} ({len(rows)} rows)")

    speedups = [r["speedup"] for r in rows if r.get("speedup")]
    if speedups:
        print(
            f"# pod/split speedup: n={len(speedups)} "
            f"median={statistics.median(speedups):.4f} min={min(speedups):.4f} "
            f"max={max(speedups):.4f}"
        )


if __name__ == "__main__":
    main()
