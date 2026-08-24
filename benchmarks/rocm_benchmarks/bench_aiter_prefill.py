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

AITER prefill benchmark: single-prefill and batch-paged via backend="aiter".

Shapes: q=256, GQA 16/4, HD=64, causal, kv ∈ {512, 1024, 2048, 3072, 4096, 8192}.
Paged regimes: page_size=256 (native-paged) and page_size=16 (flat-gather).

Run:
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py               # full pipeline
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --timing-only # no profiling
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --replot      # regenerate plot
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --batch 0    # single only
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --counters stall
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --list-presets

Design note: bench flags are parsed at module level because rocprofv3 re-executes this
script as a subprocess per PMC pass with the same sys.argv.  Module-level parsing ensures
the subprocess builds identical configs to the outer timing run.
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

# vLLM max_num_seqs=256 with mixed prefill+decode → prefill burst of ~8 seqs is mid-range.
_DEFAULT_BATCH = 8

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
        "'memory', 'basic') or path to a rocprofv3 YAML file. "
        "Default: roofline."
    ),
)
_bench_parser.add_argument(
    "--label",
    default=None,
    metavar="PREFIX",
    help="Output-file label prefix (default: 'aiter' for roofline, 'aiter_<preset>' otherwise).",
)
_bench_parser.add_argument(
    "--batch",
    type=int,
    default=_DEFAULT_BATCH,
    metavar="N",
    help=(
        "batch_size for the paged-KV sections (0 = disable paged sections). "
        f"Default: {_DEFAULT_BATCH}."
    ),
)
_bench_args, _ = _bench_parser.parse_known_args()

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("aiter" if _counters == "roofline" else f"aiter_{_counters}")
)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
_Q_LEN = 256
_KV_LENS = [512, 1024, 2048, 3072, 4096, 8192]
_NUM_QO_HEADS = 16
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_DTYPE = torch.float16
_CAUSAL = True
_BATCH = _bench_args.batch

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(
    q_len: int, kv_len: int, num_qo_heads: int, head_dim: int, causal: bool
) -> int:
    # For causal with q_len < kv_len the mask removes only the last q*(q-1)/2
    # entries of the attended set, not half the matrix (which factor=2 assumed).
    attended = q_len * kv_len - q_len * (q_len - 1) // 2 if causal else q_len * kv_len
    return attended * num_qo_heads * head_dim * 4


def _bytes(
    q_len: int, kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    return 2 * head_dim * (2 * q_len * num_qo_heads + 2 * kv_len * num_kv_heads)


def _build_paged_kv(batch, kv_len, page_size, num_kv_heads, head_dim, dtype, device):
    """Build batch-paged KV tensors for one (batch, kv_len, page_size) combination."""
    num_full_pages, last_tokens = divmod(kv_len, page_size)
    if last_tokens == 0:
        last_tokens = page_size
    else:
        num_full_pages += 1
    total_pages = num_full_pages * batch

    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * _Q_LEN
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
    q = torch.randn(batch * _Q_LEN, _NUM_QO_HEADS, head_dim, dtype=dtype, device=device)

    return q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    from flashinfer.prefill_rocm import _aiter_native_page_sizes

    native_pages = _aiter_native_page_sizes()
    # Pick the largest available native page size ≤ 256 (prefer 256 for the sweep)
    native_page = max(p for p in native_pages if p <= 256) if native_pages else 128
    flat_gather_page = 16  # deliberately not in native_pages

    configs: list[KernelConfig] = []

    # ── Single-prefill ───────────────────────────────────────────────────────
    for kv_len in _KV_LENS:
        q = torch.randn(_Q_LEN, _NUM_QO_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        k = torch.randn(kv_len, _NUM_KV_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        v = torch.randn(kv_len, _NUM_KV_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")

        flops = _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _bytes(_Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM)

        configs.append(
            KernelConfig(
                name=f"single_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q,
                    k=k,
                    v=v: flashinfer.single_prefill_with_kv_cache_return_lse(
                        q, k, v, kv_layout="NHD", causal=_CAUSAL, backend="aiter"
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=_Q_LEN,
                label=f"single  kv={kv_len:>5d}",
            )
        )

    # ── Batch-paged prefill — native-paged path ──────────────────────────────
    if _BATCH == 0:
        return configs

    for kv_len in _KV_LENS:
        q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len = (
            _build_paged_kv(
                _BATCH,
                kv_len,
                native_page,
                _NUM_KV_HEADS,
                _HEAD_DIM,
                _DTYPE,
                torch.device("cuda"),
            )
        )
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="aiter"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            _NUM_QO_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            native_page,
            causal=_CAUSAL,
            q_data_type=_DTYPE,
        )

        flops = _BATCH * _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _BATCH * _bytes(
            _Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM
        )

        configs.append(
            KernelConfig(
                name=f"batch_native_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, kv=kv_data, w=wrapper: w.run(q, kv, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=_BATCH * _Q_LEN,
                label=f"batch(native,p={native_page:>3d})  kv={kv_len:>5d}",
            )
        )

    # ── Batch-paged prefill — flat-gather path ───────────────────────────────
    for kv_len in _KV_LENS:
        q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len = (
            _build_paged_kv(
                _BATCH,
                kv_len,
                flat_gather_page,
                _NUM_KV_HEADS,
                _HEAD_DIM,
                _DTYPE,
                torch.device("cuda"),
            )
        )
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="aiter"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            _NUM_QO_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            flat_gather_page,
            causal=_CAUSAL,
            q_data_type=_DTYPE,
        )

        flops = _BATCH * _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _BATCH * _bytes(
            _Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM
        )

        configs.append(
            KernelConfig(
                name=f"batch_flat_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, kv=kv_data, w=wrapper: w.run(q, kv, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=_BATCH * _Q_LEN,
                label=f"batch(flat,  p={flat_gather_page:>3d})  kv={kv_len:>5d}",
            )
        )

    return configs


if __name__ == "__main__":
    if not is_aiter_available(torch.device("cuda:0"), "batch_prefill"):
        print("AITER backend not supported on this device. Exiting.")
        sys.exit(0)

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
