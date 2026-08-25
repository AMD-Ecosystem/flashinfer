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

Fused MoE benchmark: sweeps num_tokens across the decode -> prefill range for
Mixtral-8x7B and Qwen3-235B expert shapes.

At low token counts only a few rows land in each expert, so the two GEMMs read
far more weight than activation and the roofline sits on HBM: the headline
metric is expert-weight bytes/sec. Past roughly block_m tokens per expert the
GEMMs fill their tiles and it becomes compute bound.

`aiter.fused_moe` runs alongside as the baseline: it selects block_m, ksplit and
kernel names from a tuned CSV where the shim uses `_select_block_m` and AITER's
heuristic dispatch. That comparison is what decides whether the CSV lookup is
worth a dependency; it is skipped with a message if aiter is not importable.

`--block-m-sweep` times 32/64/128 per shape instead, which is what regenerates
the table behind `_select_block_m`'s thresholds -- so they stay checkable rather
than folklore.

`--fp8` swaps the expert weights for per-token-quantized fp8. The activations are
quantized inside the shim, so that cost is in the measurement; the number to read
is the speedup over the same shape in bf16, which is what justifies the path.

Run:
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py                  # full pipeline
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --timing-only    # no profiling
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --fp8            # fp8 weights
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --block-m-sweep  # tile-size table
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --replot         # regenerate plot
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --counters       # PMC passes
"""

import functools
import logging
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.fused_moe_rocm import (
    _SUPPORTED_BLOCK_M,
    moe_fp8_dtype,
    quantize_moe_weight,
    shuffle_moe_weight,
)
from flashinfer.jit.core import logger as _jit_logger

_jit_logger.setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

_OUTPUT_DIR = str(Path(__file__).parent)

# Crosses the tile-fill boundary: 1-32 tokens leaves most experts with a partial
# block_m tile, 1024+ saturates.
_NUM_TOKENS = [1, 8, 32, 128, 512, 2048]

# (label, num_experts, model_dim, inter_dim, topk)
_SHAPES = [
    ("mixtral8x7b", 8, 4096, 14336, 2),
    ("qwen3-235b", 128, 4096, 1536, 8),
]

_DTYPE = torch.bfloat16


@functools.cache
def _aiter_baseline():
    """aiter.fused_moe, or None. Imported lazily and once.

    Importing aiter drives its JIT loader, so this stays out of module scope:
    --replot and --list-presets build no configs and must not pay for it. The
    except is broad because a half-installed aiter raises more than ImportError
    -- same reason as aiter_utils._aiter_importable.
    """
    try:
        from aiter.fused_moe import fused_moe

        return fused_moe
    except Exception:  # noqa: BLE001
        print("note: aiter not importable, running without the tuned-CSV baseline")
        return None


@torch.inference_mode()
def _make_configs(block_m_sweep: bool = False, fp8: bool = False) -> list[KernelConfig]:
    itemsize = (moe_fp8_dtype() if fp8 else _DTYPE).itemsize
    configs = []
    for label, E, K, I, topk in _SHAPES:
        w1 = torch.randn(E, 2 * I, K, device="cuda", dtype=_DTYPE) / 16
        w2 = torch.randn(E, K, I, device="cuda", dtype=_DTYPE) / 16
        scales = {}
        if fp8:
            w1, w1_scale = quantize_moe_weight(w1)
            w2, w2_scale = quantize_moe_weight(w2)
            scales = {"w1_scale": w1_scale, "w2_scale": w2_scale}
        w1 = shuffle_moe_weight(w1)
        w2 = shuffle_moe_weight(w2)
        for nt in _NUM_TOKENS:
            x = torch.randn(nt, K, device="cuda", dtype=_DTYPE) / 8
            logits = torch.randn(nt, E, device="cuda", dtype=torch.float32)
            weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
            ids = ids.to(torch.int32).contiguous()
            weights = weights.contiguous()

            # Two GEMMs per (token, expert): [1,K]x[K,2I] and [1,I]x[I,K].
            theo_flops = 2 * nt * topk * (K * 2 * I + I * K)
            # Weights dominate until every expert is hit by many tokens; count the
            # experts actually touched rather than all E. Only the weight term is
            # fp8 -- activations reach the kernel in bf16 either way.
            experts_hit = min(E, nt * topk)
            theo_bytes = (
                experts_hit * (2 * I * K + K * I) * itemsize
                + nt * (K + topk * I + K) * _DTYPE.itemsize
            )

            def add(name, fn, suffix=""):
                configs.append(
                    KernelConfig(
                        name=name,
                        run_fn=torch.inference_mode()(fn),
                        theoretical_flops=theo_flops,
                        theoretical_bytes=theo_bytes,
                        num_tokens=nt,
                        label=f"{label} nt={nt:>5d} E={E:>3d} topk={topk}{suffix}",
                    )
                )

            if block_m_sweep:
                # Only the sweep reuses a buffer; the comparison path below
                # deliberately lets both sides allocate.
                out = torch.empty(nt, K, device="cuda", dtype=_DTYPE)
                for bm in _SUPPORTED_BLOCK_M:
                    add(
                        f"moe_{label}_nt{nt}_bm{bm}",
                        lambda x=x,
                        w1=w1,
                        w2=w2,
                        ids=ids,
                        w=weights,
                        o=out,
                        bm=bm,
                        s=scales: (
                            flashinfer.aiter_fused_moe(
                                x, w1, w2, ids, w, out=o, block_m=bm, **s
                            )
                        ),
                        suffix=f" bm={bm}",
                    )
                continue

            # No out= here: aiter.fused_moe allocates its own output and has no
            # way not to, so passing one only to the shim would charge the
            # baseline an allocation and bias the very number this comparison
            # exists to produce.
            add(
                f"moe_{label}_nt{nt}",
                lambda x=x, w1=w1, w2=w2, ids=ids, w=weights, s=scales: (
                    flashinfer.aiter_fused_moe(x, w1, w2, ids, w, **s)
                ),
            )
            baseline = _aiter_baseline()
            if baseline is not None:
                add(
                    f"moe_{label}_nt{nt}_aiter",
                    lambda x=x,
                    w1=w1,
                    w2=w2,
                    ids=ids,
                    w=weights,
                    f=baseline,
                    s=scales: (f(x, w1, w2, w, ids, **_aiter_quant_kwargs(s))),
                    suffix=" [aiter baseline]",
                )
    return configs


def _aiter_quant_kwargs(scales):
    """Translate our scale kwargs into aiter.fused_moe's own spelling."""
    if not scales:
        return {}
    from aiter.ops.enum import QuantType

    return {"quant_type": QuantType.per_Token, **scales}


if __name__ == "__main__":
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    _sweep = "--block-m-sweep" in sys.argv
    _fp8 = "--fp8" in sys.argv
    _label = "fused_moe_aiter_block_m" if _sweep else "fused_moe_aiter"
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(_sweep, _fp8),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters="roofline",
        kernel_name_regex="moe|gemm|quant",
        output_dir=_OUTPUT_DIR,
        label=f"{_label}_fp8" if _fp8 else _label,
        roofline=True,
    )
    profiler.run()
