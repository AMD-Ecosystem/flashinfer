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

Two numbers this exists to produce, both open questions in the shim:

  1. The gap to `aiter.fused_moe`, which selects kernels from a tuned CSV while
     the shim passes an empty kernelName and takes AITER's heuristic. If the gap
     is small the CSV lookup is not worth building.
  2. The cost of allocating the five sorting buffers per call, visible as the
     shim's overhead over the same work at large M where allocation is amortized.

Run:
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py                 # full pipeline
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --timing-only   # no profiling
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --replot        # regenerate plot
    python benchmarks/rocm_benchmarks/bench_fused_moe_aiter.py --counters      # PMC passes
"""

import logging
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.fused_moe_rocm import shuffle_moe_weight
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


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    itemsize = torch.tensor([], dtype=_DTYPE).element_size()
    configs = []
    for label, E, K, I, topk in _SHAPES:
        w1 = shuffle_moe_weight(
            torch.randn(E, 2 * I, K, device="cuda", dtype=_DTYPE) / 16
        )
        w2 = shuffle_moe_weight(torch.randn(E, K, I, device="cuda", dtype=_DTYPE) / 16)
        for nt in _NUM_TOKENS:
            x = torch.randn(nt, K, device="cuda", dtype=_DTYPE) / 8
            logits = torch.randn(nt, E, device="cuda", dtype=torch.float32)
            weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
            ids = ids.to(torch.int32).contiguous()
            weights = weights.contiguous()
            out = torch.empty(nt, K, device="cuda", dtype=_DTYPE)

            # Two GEMMs per (token, expert): [1,K]x[K,2I] and [1,I]x[I,K].
            theo_flops = 2 * nt * topk * (K * 2 * I + I * K)
            # Weights dominate until every expert is hit by many tokens; count the
            # experts actually touched rather than all E.
            experts_hit = min(E, nt * topk)
            theo_bytes = (
                experts_hit * (2 * I * K + K * I) * itemsize
                + nt * (K + topk * I + K) * itemsize
            )
            configs.append(
                KernelConfig(
                    name=f"moe_{label}_nt{nt}",
                    run_fn=torch.inference_mode()(
                        lambda x=x, w1=w1, w2=w2, ids=ids, weights=weights, out=out: (
                            flashinfer.aiter_fused_moe(x, w1, w2, ids, weights, out=out)
                        )
                    ),
                    theoretical_flops=theo_flops,
                    theoretical_bytes=theo_bytes,
                    num_tokens=nt,
                    label=f"{label} nt={nt:>5d} E={E:>3d} topk={topk}",
                )
            )
    return configs


if __name__ == "__main__":
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters="roofline",
        kernel_name_regex="moe|gemm",
        output_dir=_OUTPUT_DIR,
        label="fused_moe_aiter",
        roofline=True,
    )
    profiler.run()
