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

Batch paged-decode benchmark: fa2 (in-tree HIP) vs aiter (PA v1), eager.

Baseline for the "HIP-graph-safe fast decode" work item
(docs/rocm_library_optimization_plan.md §3 #1): today AITER decode falls back
to fa2 under CUDA-graph capture, so the fast path never runs in the serving
hot loop. This bench measures the eager fa2-vs-aiter gap (the win we forfeit
under graphs) across a serving-shaped batch x kv_len sweep. Once the
capacity-sized-grid change lands, add a --graph path here to compare
aiter-under-graph vs fa2-under-graph and to measure the idle-block
over-provisioning cost (the one open perf item in the plan).

Shapes: decode (q_len=1), GQA 32/8, HD=128, bf16, page_size=16.
Sweep: batch x kv_len.

Run:
    python benchmarks/rocm_benchmarks/bench_batch_decode_hip.py               # full pipeline
    python benchmarks/rocm_benchmarks/bench_batch_decode_hip.py --timing-only # no profiling
    python benchmarks/rocm_benchmarks/bench_batch_decode_hip.py --backend fa2
    python benchmarks/rocm_benchmarks/bench_batch_decode_hip.py --backend aiter
    python benchmarks/rocm_benchmarks/bench_batch_decode_hip.py --counters stall

Design note: bench flags are parsed at module level because rocprofv3
re-executes this script as a subprocess per PMC pass with the same sys.argv.
Module-level parsing ensures the subprocess builds identical configs to the
outer timing run.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.aiter_utils import is_aiter_available
from flashinfer.jit.core import logger as _jit_logger

# Suppress routine JIT INFO/DEBUG output; WARNING still surfaces compile errors.
_jit_logger.setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing (see design note above)
# ---------------------------------------------------------------------------
_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument(
    "--counters",
    default="roofline",
    metavar="PRESET_OR_FILE",
    help=(
        "Counter preset name ('roofline', 'occupancy', 'stall', 'compute', "
        "'memory', 'basic') or path to a rocprofv3 YAML file. Default: roofline."
    ),
)
_bench_parser.add_argument(
    "--label",
    default=None,
    metavar="PREFIX",
    help="Output-file label prefix (default: 'decode' for roofline, 'decode_<preset>' otherwise).",
)
_bench_parser.add_argument(
    "--backend",
    default="both",
    choices=["fa2", "aiter", "both"],
    help="Which decode backend(s) to sweep. Default: both.",
)
_bench_args, _ = _bench_parser.parse_known_args()

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("decode" if _counters == "roofline" else f"decode_{_counters}")
)

# ---------------------------------------------------------------------------
# Sweep configuration — decode is q_len=1; shapes ~ Llama-70B TP1 decode.
# ---------------------------------------------------------------------------
_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_DTYPE = torch.bfloat16
_PAGE_SIZE = 16
_BATCHES = [1, 8, 32, 128, 256]
_KV_LENS = [1024, 2048, 4096, 8192]

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(kv_len: int, num_qo_heads: int, head_dim: int) -> int:
    # decode: q_len=1 → attended = kv_len; QK^T + PV ≈ 2 matmuls → factor 4.
    return kv_len * num_qo_heads * head_dim * 4


def _bytes(kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int) -> int:
    # Dominated by reading K and V (q_len=1). 2 bytes/elem (bf16).
    return 2 * head_dim * (2 * num_qo_heads + 2 * kv_len * num_kv_heads)


def _build_paged_kv(batch, kv_len, page_size, num_kv_heads, head_dim, dtype, device):
    """Build batch-paged decode KV + query tensors for one (batch, kv_len)."""
    num_full_pages, last_tokens = divmod(kv_len, page_size)
    if last_tokens == 0:
        last_tokens = page_size
    else:
        num_full_pages += 1
    total_pages = num_full_pages * batch

    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    kv_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_full_pages
    )
    _rng = torch.Generator(device=device).manual_seed(42)
    kv_indices = torch.randperm(
        total_pages, dtype=torch.int32, device=device, generator=_rng
    )
    kv_last_page_len = torch.full(
        (batch,), last_tokens, dtype=torch.int32, device=device
    )
    q = torch.randn(batch, _NUM_QO_HEADS, head_dim, dtype=dtype, device=device)
    return q, kv_data, kv_indptr, kv_indices, kv_last_page_len


def _make_wrapper_config(backend, batch, kv_len):
    device = torch.device("cuda")
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch, kv_len, _PAGE_SIZE, _NUM_KV_HEADS, _HEAD_DIM, _DTYPE, device
    )
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=_DTYPE,
        kv_data_type=_DTYPE,
    )
    return KernelConfig(
        name=f"{backend}_b{batch}_kv{kv_len}",
        run_fn=torch.inference_mode()(lambda q=q, kv=kv_data, w=wrapper: w.run(q, kv)),
        theoretical_flops=batch * _flops(kv_len, _NUM_QO_HEADS, _HEAD_DIM),
        theoretical_bytes=batch
        * _bytes(kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM),
        num_tokens=batch,  # one decoded token per sequence
        label=f"{backend:<5s} b={batch:>3d}  kv={kv_len:>5d}",
    )


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    backends = (
        ["fa2", "aiter"] if _bench_args.backend == "both" else [_bench_args.backend]
    )
    if "aiter" in backends and not is_aiter_available(
        torch.device("cuda:0"), "batch_decode"
    ):
        print("AITER not supported on this device; dropping the aiter sweep.")
        backends = [b for b in backends if b != "aiter"]

    configs: list[KernelConfig] = []
    for backend in backends:
        for batch in _BATCHES:
            for kv_len in _KV_LENS:
                configs.append(_make_wrapper_config(backend, batch, kv_len))
    return configs


if __name__ == "__main__":
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters=_counters,
        kernel_name_regex="",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
