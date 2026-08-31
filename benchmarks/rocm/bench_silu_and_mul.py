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

silu_and_mul benchmark: sweeps num_tokens x d x dtype to expose the small-batch
(decode) regime where one-block-per-token underfills the GPU, alongside the
large-batch (prefill) regime that is already saturated.

This kernel is memory-bandwidth bound (reads 2*d, writes d per token), so the
roofline sits on the HBM ceiling and tokens/sec is the headline metric.

Run:
    python benchmarks/rocm/bench_silu_and_mul.py                 # full pipeline
    python benchmarks/rocm/bench_silu_and_mul.py --timing-only   # no profiling
    python benchmarks/rocm/bench_silu_and_mul.py --replot        # regenerate plot
"""

import logging
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger as _jit_logger

_jit_logger.setLevel(logging.WARNING)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "profiler" / "rocm")
)
from rocm_profiler import KernelConfig, RocmProfiler

_OUTPUT_DIR = str(Path(__file__).parent)

# num_tokens sweep crosses the CU-fill boundary: tiny (decode), small batch,
# up to prefill-scale where blocks_per_row resolves to 1.
_NUM_TOKENS = [1, 8, 32, 64, 128, 256, 1024, 4096]
# d = hidden_size // 2 for representative MLP intermediate sizes.
_DIMS = [4096, 14336]
_DTYPES = [(torch.float16, "f16"), (torch.bfloat16, "bf16")]


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    configs = []
    for dtype, dt_name in _DTYPES:
        itemsize = torch.tensor([], dtype=dtype).element_size()
        for d in _DIMS:
            for nt in _NUM_TOKENS:
                x = torch.randn(nt, 2 * d, device="cuda", dtype=dtype)
                out = torch.empty(nt, d, device="cuda", dtype=dtype)
                # Bandwidth-bound: read gate+up (2*d) and write (d) per token.
                theo_bytes = nt * 3 * d * itemsize
                # One mul per output element; FLOPs are not the bottleneck but the
                # profiler needs a nonzero value for arithmetic intensity.
                theo_flops = nt * d
                configs.append(
                    KernelConfig(
                        name=f"silu_{dt_name}_nt{nt}_d{d}",
                        run_fn=torch.inference_mode()(
                            lambda x=x, out=out: flashinfer.activation.silu_and_mul(
                                x, out=out
                            )
                        ),
                        theoretical_flops=theo_flops,
                        theoretical_bytes=theo_bytes,
                        num_tokens=nt,
                        label=f"{dt_name} nt={nt:>5d} d={d:>5d}",
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
        kernel_name_regex="act_and_mul_kernel",
        output_dir=_OUTPUT_DIR,
        label="silu_and_mul",
        roofline=True,
    )
    profiler.run()
