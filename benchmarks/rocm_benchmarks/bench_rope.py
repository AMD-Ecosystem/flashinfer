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

cos/sin-cache RoPE benchmark: native (in-tree HIP JIT) vs AITER, sweeping nnz
(number of tokens) across the decode -> prefill range so the small-batch regime
where native's lower launch overhead wins is visible alongside the large-batch
regime where AITER's kernel pulls ahead. This is the comparison behind the
``backend="auto"`` policy in ``flashinfer/rope.py`` (see PR #252 / #259); unlike
the upstream ``benchmarks/bench_rope.py`` it has no vLLM dependency and adds an
accuracy axis, since the auto policy turns partly on bf16 numerical error.

Like ``apply_rope_with_cos_sin_cache_inplace``, this kernel is memory-bandwidth
bound (reads + writes q and k per token), so the roofline sits on the HBM
ceiling and tokens/sec is the headline metric.

Run:
    python benchmarks/rocm_benchmarks/bench_rope.py                  # full pipeline (both op modes)
    python benchmarks/rocm_benchmarks/bench_rope.py --timing-only    # no profiling
    python benchmarks/rocm_benchmarks/bench_rope.py --op outplace    # out-of-place only (inplace|outplace)
    python benchmarks/rocm_benchmarks/bench_rope.py --accuracy       # native-vs-aiter error table, no profiling
    python benchmarks/rocm_benchmarks/bench_rope.py --replot         # regenerate plot
"""

import logging
import sys
from pathlib import Path

import torch

from flashinfer.aiter_utils import is_aiter_available
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.rope import (
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
)

_jit_logger.setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

_OUTPUT_DIR = str(Path(__file__).parent)

# nnz sweep crosses the native->aiter crossover: tiny (decode), small batch, up
# through prefill-scale where AITER's higher throughput dominates launch cost.
_NNZ = [8, 32, 64, 128, 256, 512, 1024, 2048, 8192, 32768]
# Llama-3 8B attention shape: 32 q heads / 8 kv heads, head_size 128, full rotary.
_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_SIZE = 128
_ROTARY_DIM = 128
_MAX_SEQ_LEN = 65536
_DTYPES = [(torch.float16, "f16"), (torch.bfloat16, "bf16")]
_BACKENDS = ["native", "aiter"]
# Both op modes: inplace rotates q/k in place; out-of-place returns fresh
# tensors. They differ for AITER — the out-of-place path uses the zero-copy
# `_impl` entry point (no q/k copy), which is where AITER's largest wins land
# (see PR #252). The default `--op` runs both; restrict with --op inplace|outplace.
_OP_MODES = ["inplace", "outplace"]


_COS_SIN_CACHE: torch.Tensor | None = None


def _shared_cos_sin_cache() -> torch.Tensor:
    # cos||sin cache is float32 on the HIP path (see rope_aiter.cu) and is
    # read-only, so a single ~32 MiB tensor is shared across all configs
    # rather than reallocated per config.
    global _COS_SIN_CACHE
    if _COS_SIN_CACHE is None:
        _COS_SIN_CACHE = torch.randn(
            _MAX_SEQ_LEN, _ROTARY_DIM, device="cuda", dtype=torch.float32
        )
    return _COS_SIN_CACHE


def _make_inputs(nnz: int, dtype: torch.dtype):
    positions = torch.arange(nnz, device="cuda", dtype=torch.int64) % _MAX_SEQ_LEN
    query = torch.randn(nnz, _NUM_Q_HEADS * _HEAD_SIZE, device="cuda", dtype=dtype)
    key = torch.randn(nnz, _NUM_KV_HEADS * _HEAD_SIZE, device="cuda", dtype=dtype)
    return positions, query, key, _shared_cos_sin_cache()


def _run_fn(op_mode, positions, query, key, cos_sin_cache, backend):
    if op_mode == "inplace":
        return lambda: apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=_HEAD_SIZE,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
            backend=backend,
        )
    return lambda: apply_rope_with_cos_sin_cache(
        positions=positions,
        query=query,
        key=key,
        head_size=_HEAD_SIZE,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        backend=backend,
    )


@torch.inference_mode()
def _make_configs(op_modes: list[str]) -> list[KernelConfig]:
    aiter_ok = is_aiter_available(torch.device("cuda"))
    if not aiter_ok:
        print(
            "[bench_rope] AITER unavailable on this device; benchmarking native only."
        )
    configs = []
    for op_mode in op_modes:
        for dtype, dt_name in _DTYPES:
            itemsize = torch.tensor([], dtype=dtype).element_size()
            for nnz in _NNZ:
                # Bandwidth-bound: read + write q and k per token.
                rows = nnz * (_NUM_Q_HEADS + _NUM_KV_HEADS) * _HEAD_SIZE
                theo_bytes = 2 * rows * itemsize
                # One sincos rotate per rotary element; FLOPs are not the
                # bottleneck but the profiler needs a nonzero value for
                # arithmetic intensity.
                theo_flops = nnz * (_NUM_Q_HEADS + _NUM_KV_HEADS) * _ROTARY_DIM
                for backend in _BACKENDS:
                    if backend == "aiter" and not aiter_ok:
                        continue
                    positions, query, key, cos_sin_cache = _make_inputs(nnz, dtype)
                    configs.append(
                        KernelConfig(
                            name=f"rope_{op_mode}_{backend}_{dt_name}_nnz{nnz}",
                            run_fn=torch.inference_mode()(
                                _run_fn(
                                    op_mode,
                                    positions,
                                    query,
                                    key,
                                    cos_sin_cache,
                                    backend,
                                )
                            ),
                            theoretical_flops=theo_flops,
                            theoretical_bytes=theo_bytes,
                            num_tokens=nnz,
                            label=f"{op_mode:>8s} {backend:>6s} {dt_name} nnz={nnz:>5d}",
                        )
                    )
    return configs


@torch.inference_mode()
def _accuracy() -> None:
    """Print max abs error of AITER vs the native kernel (the auto policy's
    precision axis): bf16 sits near the tolerance edge, fp16 is comfortably safe."""
    if not is_aiter_available(torch.device("cuda")):
        print("[bench_rope] AITER unavailable; cannot run accuracy comparison.")
        return
    print(f"{'dtype':>6s} {'nnz':>7s} {'max_abs_err':>14s}")
    for dtype, dt_name in _DTYPES:
        for nnz in _NNZ:
            positions, query, key, cos_sin_cache = _make_inputs(nnz, dtype)
            q_ref, k_ref = query.clone(), key.clone()
            apply_rope_with_cos_sin_cache_inplace(
                positions,
                q_ref,
                k_ref,
                _HEAD_SIZE,
                cos_sin_cache,
                is_neox=True,
                backend="native",
            )
            q_ait, k_ait = query.clone(), key.clone()
            apply_rope_with_cos_sin_cache_inplace(
                positions,
                q_ait,
                k_ait,
                _HEAD_SIZE,
                cos_sin_cache,
                is_neox=True,
                backend="aiter",
            )
            err = max(
                (q_ait.float() - q_ref.float()).abs().max().item(),
                (k_ait.float() - k_ref.float()).abs().max().item(),
            )
            print(f"{dt_name:>6s} {nnz:>7d} {err:>14.2e}")


if __name__ == "__main__":
    if "--accuracy" in sys.argv:
        _accuracy()
        sys.exit(0)
    op_modes = _OP_MODES
    if "--op" in sys.argv:
        idx = sys.argv.index("--op") + 1
        choice = sys.argv[idx] if idx < len(sys.argv) else None
        if choice not in _OP_MODES:
            sys.exit(f"--op must be one of {_OP_MODES}, got {choice!r}")
        op_modes = [choice]
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(op_modes),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters="roofline",
        kernel_name_regex="Rotary|rope",
        output_dir=_OUTPUT_DIR,
        label="rope",
        roofline=True,
    )
    profiler.run()
