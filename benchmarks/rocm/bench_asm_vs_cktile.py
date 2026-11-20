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
believing any ratio, and compare each cell against its own A/A control rather
than one global band -- the per-shape floors differ and a margin inside one is
not a result.

Only the ``batch=1`` rows are reachable through ``single_prefill``, which pins
``args.batch = 1``. The ``batch=4`` rows characterise no shipping path -- batched
prefill runs group mode, which ``--mode varlen`` measures, or native-paged
``mha_batch_prefill``, which nothing here measures -- and are kept only to
reproduce the mixed aggregate published before the threshold was rescoped.

Run:
    python benchmarks/rocm/bench_asm_vs_cktile.py --aa
    python benchmarks/rocm/bench_asm_vs_cktile.py --csv asm-gfx950.csv
    python benchmarks/rocm/bench_asm_vs_cktile.py --accuracy
    python benchmarks/rocm/bench_asm_vs_cktile.py --mode varlen --csv varlen.csv
    python benchmarks/rocm/bench_asm_vs_cktile.py --mode varlen --holdout 200
    python benchmarks/rocm/bench_asm_vs_cktile.py --mode varlen --serving
    python benchmarks/rocm/bench_asm_vs_cktile.py --mode varlen --edge
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


# --- group mode (varlen): ragged and paged flat-gather prefill -------------------
# asm is called as fmha_v3_varlen_fwd, which compiles the same mha_fwd.cu, flags and
# .co set as the module_fmha_v3_fwd.so the shim loads; how_v3_bf16_cvt=1 (rtna)
# matches CK's ties-away bf16 conversion on gfx942 and is ignored on gfx950.
_VL_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
_VL_QO = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_VL_KV_EXTRA = [0, 4096]
_VL_HEADS = [(8, 1), (16, 2), (28, 4), (32, 8), (32, 32), (64, 8)]
# (hd_qk, hd_vo, heads override or None, arch restriction or None)
_VL_HEAD_DIMS = [
    (128, 128, None, None),
    (192, 128, [(16, 16), (128, 128)], "gfx942"),  # gfx950 serves this on OPUS
    (256, 256, [(8, 8), (16, 16)], "gfx950"),
]
_VL_MAX_TOKENS = 128 * 1024
_VL_ARMS = ("ck", "ck2", "asm", "fa2")


def _vl_shapes(arch: str):
    for hd_qk, hd_vo, heads, only in _VL_HEAD_DIMS:
        if only and only != arch:
            continue
        for hq, hk in heads or _VL_HEADS:
            for batch in _VL_BATCHES:
                for qo in _VL_QO:
                    for extra in _VL_KV_EXTRA:
                        if batch * (qo + extra) > _VL_MAX_TOKENS:
                            continue
                        for causal in (True, False):
                            yield (
                                [qo] * batch,
                                [qo + extra] * batch,
                                hq,
                                hk,
                                hd_qk,
                                hd_vo,
                                causal,
                            )


def _vl_holdout(n: int, seed: int = 11):
    """Ragged batches, log-normal prompts or extend-like (short new tokens on a prefix),
    2/3 causal. Batches over _VL_MAX_TOKENS are dropped, which removes most large
    extend batches."""
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n):
        batch = int(torch.randint(1, 65, (1,), generator=rng))
        prompt, causal, heads = (
            int(torch.randint(0, m, (1,), generator=rng))
            for m in (2, 3, len(_VL_HEADS))
        )
        if prompt:
            qo = (
                torch.exp(torch.randn(batch, generator=rng) * 1.2 + 6.0)
                .clamp(1, 8192)
                .int()
            )
            kv = qo.clone()
        else:
            qo = torch.randint(1, 513, (batch,), generator=rng).int()
            kv = qo + torch.randint(0, 8193, (batch,), generator=rng).int()
        if int(kv.sum()) > _VL_MAX_TOKENS:
            continue
        hq, hk = _VL_HEADS[heads]
        yield (qo.tolist(), kv.tolist(), hq, hk, 128, 128, bool(causal))


def _vl_serving():
    """Serving-size batches, causal: spec-decode verify (2-8 new tokens) and chunked prefill."""
    for hq, hk in [(8, 1), (32, 8), (64, 8)]:
        for batch in (32, 128, 256, 512):
            for qo in (2, 4, 8):
                for kv in (1024, 4096):
                    yield [qo] * batch, [kv] * batch, hq, hk, 128, 128, True
        for batch, qo, kv in (
            (256, 64, 2048),
            (512, 16, 1024),
            (192, 128, 4096),
            (64, 512, 512),
        ):
            yield [qo] * batch, [kv] * batch, hq, hk, 128, 128, True


def _vl_cu(lens) -> torch.Tensor:
    out = [0]
    for n in lens:
        out.append(out[-1] + int(n))
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def _vl_case(qo, kv, hq, hk, hd_qk, hd_vo, causal, arm):
    """(closure, out) for one (shape, arm); every arm gets the same packed thd inputs."""
    from aiter.ops.mha import fmha_v3_varlen_fwd, mha_varlen_fwd

    g = torch.Generator(device="cuda").manual_seed(7)
    tq, tk = sum(qo), sum(kv)
    q = torch.randn(tq, hq, hd_qk, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(tk, hk, hd_qk, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(tk, hk, hd_vo, dtype=torch.bfloat16, device="cuda", generator=g)
    cq, ck = _vl_cu(qo), _vl_cu(kv)
    mq, mk, scale = max(qo), max(kv), hd_qk**-0.5
    out = torch.empty(tq, hq, hd_vo, dtype=torch.bfloat16, device="cuda")
    # min_seqlen_q must stay 0: CK treats a positive value as a skip threshold.
    common = (q, k, v, cq, ck, mq, mk, 0, 0.0, scale, 0.0, False, causal, -1, -1)
    if arm == "asm":
        fn = lambda: fmha_v3_varlen_fwd(*common, False, False, 1, out=out)[0]
    elif arm in ("ck", "ck2"):
        fn = lambda: mha_varlen_fwd(*common, 0, False, False, out=out)[0]
    else:
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(
            cq,
            ck,
            hq,
            hk,
            hd_qk,
            head_dim_vo=hd_vo,
            causal=causal,
            q_data_type=torch.bfloat16,
        )
        fn = lambda: w.run(q, k, v, out=out)
    return fn, out


def _vl_features(qo, kv, hq, hk, hd_qk, hd_vo, causal) -> dict:
    return {
        "batch": len(qo),
        "total_q": sum(qo),
        "total_k": sum(kv),
        "max_q": max(qo),
        "max_k": max(kv),
        "mean_q": round(sum(qo) / len(qo), 1),
        "hq": hq,
        "hk": hk,
        "hd_qk": hd_qk,
        "hd_vo": hd_vo,
        "causal": causal,
    }


def _vl_flops(qo, kv, hq, hd_qk, hd_vo, causal) -> float:
    pairs = 0
    for a, b in zip(qo, kv, strict=True):
        pairs += (
            a * b - (a * (a - 1) // 2 if causal else 0)
            if a <= b
            else (b * (b + 1) // 2 if causal else a * b)
        )
    return 2.0 * pairs * hq * (hd_qk + hd_vo)


def _vl_sweep(cases, arms, dry_run_iters, repeat_iters, peak) -> list[dict]:
    rows = []
    for n, (qo, kv, hq, hk, hd_qk, hd_vo, causal) in enumerate(cases):
        rec = _vl_features(qo, kv, hq, hk, hd_qk, hd_vo, causal)
        fns, outs = {}, {}
        for arm in arms:
            try:
                fns[arm], outs[arm] = _vl_case(
                    qo, kv, hq, hk, hd_qk, hd_vo, causal, arm
                )
                for _ in range(dry_run_iters):
                    fns[arm]()
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                fns.pop(arm, None)
                rec[f"{arm}_err"] = f"{type(exc).__name__}: {exc}"[:160]
        torch.cuda.synchronize()
        # A fast arm that skipped work would otherwise pass as a speedup.
        if "ck" in fns:
            for arm in fns:
                if arm != "ck":
                    diff = (outs[arm].float() - outs["ck"].float()).abs().max().item()
                    rec[f"{arm}_maxdiff"] = round(diff, 4)
        # Rotate arm order per cell so no arm always runs first on a cooler clock.
        order = list(fns)[n % max(1, len(fns)) :] + list(fns)[: n % max(1, len(fns))]
        for arm in order:
            try:
                med, spread = _time_us(fns[arm], dry_run_iters, repeat_iters)
                rec[f"{arm}_us"], rec[f"{arm}_spread_us"] = (
                    round(med, 3),
                    round(spread, 3),
                )
            except Exception as exc:  # noqa: BLE001
                rec[f"{arm}_err"] = f"{type(exc).__name__}: {exc}"[:160]
            torch.cuda.synchronize()
        del fns, outs
        torch.cuda.empty_cache()
        f = _vl_flops(qo, kv, hq, hd_qk, hd_vo, causal)
        for arm in arms:
            us = rec.get(f"{arm}_us")
            if us:
                rec[f"{arm}_roof_pct"] = round(100 * f / (us * 1e-6) / peak, 2)
        ck, ck2, asm = rec.get("ck_us"), rec.get("ck2_us"), rec.get("asm_us")
        rec["asm_over_ck"] = round(asm / ck, 4) if ck and asm else None
        rec["aa"] = round(ck / ck2, 4) if ck and ck2 else None
        rows.append(rec)
        print(
            " ".join(f"{k}={rec[k]}" for k in rec if not k.endswith("spread_us")),
            flush=True,
        )
    return rows


def _vl_summary(rows) -> None:
    """asm/CK time, summed and geomean, split verify (max_q <= 8) vs the rest."""
    for name, keep in (("verify", True), ("other", False)):
        sub = [r for r in rows if r.get("asm_over_ck") and (r["max_q"] <= 8) == keep]
        if not sub:
            continue
        tw = sum(r["asm_us"] for r in sub) / sum(r["ck_us"] for r in sub)
        geo = statistics.geometric_mean([r["asm_over_ck"] for r in sub])
        wins = sum(r["asm_over_ck"] < 0.95 for r in sub)
        aa = max((abs(r["aa"] - 1) for r in sub if r.get("aa")), default=float("nan"))
        md = max(r.get("asm_maxdiff", float("nan")) for r in sub)
        print(
            f"# {name}: {len(sub)} cells, asm/ck time-weighted {tw:.3f} geomean "
            f"{geo:.3f}, asm faster (<0.95) in {wins}, A/A max deviation {aa:.3f}, "
            f"asm vs ck max |diff| {md:.4f}"
        )


def _vl_reference(q, k, v, qo, kv, causal):
    """fp32 output and natural-log LSE; rows with no keys give 0 and -inf."""
    hq, hk = q.shape[1], k.shape[1]
    outs, lses, oq, ok = [], [], 0, 0
    for a, b in zip(qo, kv, strict=True):
        qs = q[oq : oq + a].float().transpose(0, 1)
        ks = k[ok : ok + b].float().repeat_interleave(hq // hk, 1).transpose(0, 1)
        vs = v[ok : ok + b].float().repeat_interleave(hq // hk, 1).transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * q.shape[-1] ** -0.5
        if causal:
            row = torch.arange(a, device=q.device)[:, None] + (b - a)
            s = s.masked_fill(
                torch.arange(b, device=q.device)[None, :] > row, float("-inf")
            )
        lse = torch.logsumexp(s, -1)
        p = torch.nan_to_num(torch.exp(s - lse[..., None]), nan=0.0)
        outs.append((p @ vs).transpose(0, 1))
        lses.append(lse)
        oq, ok = oq + a, ok + b
    return torch.cat(outs), torch.cat(lses, dim=1)


def _vl_edge() -> None:
    """Edge numerics for both arms: empty sequences, empty KV, qo > kv rows, LSE."""
    from aiter.ops.mha import fmha_v3_varlen_fwd, mha_varlen_fwd

    cases = [
        ("qo_i=0", [0, 300, 700], [256, 300, 700], True),
        ("kv_i=0", [64, 300, 700], [0, 300, 700], False),
        ("causal qo>kv", [512, 300], [128, 900], True),
        ("1-token seqs", [1, 1, 1, 777], [1, 50, 999, 777], True),
        ("tune_opt total_q>16384", [9000, 9000], [9000, 9000], True),
    ]
    print(f"{'case':<26}{'arm':>5}{'out err':>11}{'lse err':>11}  note")
    for name, qo, kv, causal in cases:
        g = torch.Generator(device="cuda").manual_seed(3)
        hq, hk, hd = 32, 8, 128
        q = torch.randn(
            sum(qo), hq, hd, dtype=torch.bfloat16, device="cuda", generator=g
        )
        k = torch.randn(
            sum(kv), hk, hd, dtype=torch.bfloat16, device="cuda", generator=g
        )
        v = torch.randn(
            sum(kv), hk, hd, dtype=torch.bfloat16, device="cuda", generator=g
        )
        ref, ref_lse = _vl_reference(q, k, v, qo, kv, causal)
        common = (
            q,
            k,
            v,
            _vl_cu(qo),
            _vl_cu(kv),
            max(qo),
            max(kv),
            0,
            0.0,
            hd**-0.5,
            0.0,
            False,
            causal,
            -1,
            -1,
        )
        for arm in ("ck", "asm"):
            try:
                if arm == "asm":
                    o, lse = fmha_v3_varlen_fwd(*common, True, False, 1)[:2]
                else:
                    o, lse = mha_varlen_fwd(*common, 0, True, False)[:2]
                torch.cuda.synchronize()
                err = (o.float() - ref).abs().max().item()
                fin = torch.isfinite(ref_lse)
                lerr = (
                    (lse[fin] - ref_lse[fin]).abs().max().item() if fin.any() else 0.0
                )
                bad = int((torch.isfinite(lse) != fin).sum())
                bad += int((lse[~fin] != ref_lse[~fin]).sum()) if not bad else 0
                note = f"{bad} rows not finite/-inf as fp32" if bad else ""
                print(
                    f"{name:<26}{arm:>5}{err:>11.5f}{lerr:>11.2e}  {note}", flush=True
                )
            except Exception as exc:  # noqa: BLE001 - a refusal is a result
                print(
                    f"{name:<26}{arm:>5}  {type(exc).__name__}: {str(exc)[:80]}",
                    flush=True,
                )


def _vl_main(args) -> None:
    from flashinfer.rocm.arch_caps import normalize_arch

    props = torch.cuda.get_device_properties(0)
    arch = normalize_arch(props.gcnArchName)
    # bf16 dense MFMA peak per CU per clock x CUs x clock; the clock is the spec boost.
    peak = {
        "gfx942": 1.307e15 * props.multi_processor_count / 304,
        "gfx950": 2.5e15 * props.multi_processor_count / 256,
    }[arch]
    if args.edge:
        _vl_edge()
        return
    unknown = set(args.arms.split(",")) - set(_VL_ARMS)
    if unknown:
        raise SystemExit(f"--arms: unknown {sorted(unknown)}; choose from {_VL_ARMS}")
    if args.holdout:
        cases = list(_vl_holdout(args.holdout))
    else:
        cases = list(_vl_serving() if args.serving else _vl_shapes(arch))
    i, n = (int(x) for x in args.shard.split("/"))
    cases = cases[i::n]
    print(f"# varlen cells: {len(cases)} (shard {args.shard}), peak {peak:.3e}")
    rows = _vl_sweep(
        cases, args.arms.split(","), args.dry_run_iters, args.repeat_iters, peak
    )
    _vl_summary(rows)
    if args.csv:
        fieldnames = sorted({k for r in rows for k in r})
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"# wrote {args.csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("single", "varlen"), default="single")
    ap.add_argument("--edge", action="store_true", help="varlen: edge-case numerics")
    ap.add_argument(
        "--holdout", type=int, default=0, help="varlen: N random ragged batches"
    )
    ap.add_argument(
        "--serving", action="store_true", help="varlen: serving-size batches"
    )
    ap.add_argument("--shard", default="0/1", help="varlen: run cells i::n as 'i/n'")
    ap.add_argument("--arms", default=",".join(_VL_ARMS), help="varlen arms")
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

    if args.mode == "varlen":
        _vl_main(args)
        return

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
