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

FA2 prefill benchmark: single-prefill, batch-ragged, and batch-paged via backend="fa2".

Run:
    python benchmarks/rocm/bench_fa2_prefill.py                        # full pipeline
    python benchmarks/rocm/bench_fa2_prefill.py --timing-only          # no profiling
    python benchmarks/rocm/bench_fa2_prefill.py --replot               # regenerate plot
    python benchmarks/rocm/bench_fa2_prefill.py --model llama3-8b      # GQA shape
    python benchmarks/rocm/bench_fa2_prefill.py --batch 0 --paged 0   # single only
    python benchmarks/rocm/bench_fa2_prefill.py --counters stall       # stall preset
    python benchmarks/rocm/bench_fa2_prefill.py --list-presets         # preset names

Design note: bench flags are parsed at module level because rocprofv3 re-executes this
script as a subprocess per PMC pass with the same sys.argv.  Module-level parsing ensures
the subprocess builds identical configs to the outer timing run.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger as _jit_logger

# Suppress routine JIT INFO/DEBUG output; WARNING still surfaces compile errors.
_jit_logger.setLevel(logging.WARNING)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "profiler" / "rocm")
)
from rocm_profiler import KernelConfig, RocmProfiler

# vLLM max_num_batched_tokens=2048 at seq_len≈512 → ~4 seqs; paged burst mid-range → 8.
_DEFAULT_BATCH_RAGGED = 4
_DEFAULT_BATCH_PAGED = 8
_DEFAULT_BATCH_Q_LEN = (
    256  # matches AITER bench _Q_LEN; represents a chunked-prefill burst
)

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
    help=(
        "Output-file label prefix (default: 'fa2' for the roofline preset, "
        "'fa2_<preset>' for all others)."
    ),
)
_bench_parser.add_argument(
    "--model",
    default="intensity",
    choices=["intensity", "llama3-8b", "llama3-70b", "mistral-7b"],
    help=(
        "Shape preset. 'intensity' = MHA 32/32 hd=128 sweep (default, good for "
        "roofline). GQA presets match real model architectures: "
        "llama3-8b (32/8 hd=128), llama3-70b (64/8 hd=128), mistral-7b (32/8 hd=128)."
    ),
)
_bench_parser.add_argument(
    "--q-len",
    type=int,
    default=0,
    metavar="Q",
    help=(
        "When > 0, adds asymmetric single-prefill configs with q_len=Q and "
        "kv_len swept over the standard sequence lengths (causal only). "
        "Models chunked prefill / decode-step scenarios. Default: 0 (disabled)."
    ),
)
_bench_parser.add_argument(
    "--batch",
    type=int,
    default=_DEFAULT_BATCH_RAGGED,
    metavar="N",
    help=(
        "batch_size for ragged-KV batch-prefill configs (0 = disabled). "
        f"Default: {_DEFAULT_BATCH_RAGGED} (~vLLM max_num_batched_tokens=2048 "
        "at seq_len=512)."
    ),
)
_bench_parser.add_argument(
    "--paged",
    type=int,
    default=_DEFAULT_BATCH_PAGED,
    metavar="N",
    help=(
        "batch_size for paged-KV batch-prefill configs (0 = disabled). "
        "Runs page_size=16 and page_size=64 by default; override with --page-size. "
        f"Default: {_DEFAULT_BATCH_PAGED} (mid-range prefill burst in a "
        "mixed prefill+decode paged system)."
    ),
)
_bench_parser.add_argument(
    "--page-size",
    type=int,
    default=0,
    metavar="P",
    help=(
        "Override page size for --paged mode. When 0 (default), runs both "
        "page_size=16 and page_size=64 automatically."
    ),
)
_bench_parser.add_argument(
    "--batch-q-len",
    type=int,
    default=_DEFAULT_BATCH_Q_LEN,
    metavar="Q",
    help=(
        "Fixed query length per sequence for --batch and --paged configs. "
        "Each batch sequence has q_len=Q attending to kv_len swept over the "
        "standard range (causal only, kv_len > Q). "
        f"Default: {_DEFAULT_BATCH_Q_LEN} (chunked-prefill burst, matches AITER bench)."
    ),
)
_bench_args, _ = _bench_parser.parse_known_args()

_counters = _bench_args.counters
_model: str = _bench_args.model
_q_len_asym: int = _bench_args.q_len
_batch_size: int = _bench_args.batch
_paged_size: int = _bench_args.paged
_batch_q_len: int = _bench_args.batch_q_len
_page_sizes: list[int] = (
    [_bench_args.page_size] if _bench_args.page_size > 0 else [16, 64]
)
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("fa2" if _counters == "roofline" else f"fa2_{_counters}")
)

# ---------------------------------------------------------------------------
# Shape presets (seq_len, num_qo_heads, num_kv_heads, head_dim, causal)
# ---------------------------------------------------------------------------
_MODEL_CONFIGS: dict[str, list[tuple]] = {
    # MHA sweep — maximises arithmetic intensity range for roofline analysis.
    # Crosses the MI300X ridge point (~247 FLOPs/B) near seq_len=1024.
    "intensity": [
        (512, 32, 32, 128, True),
        (1024, 32, 32, 128, True),
        (2048, 32, 32, 128, True),
        (4096, 32, 32, 128, True),
        (8192, 32, 32, 128, True),
        (512, 32, 32, 128, False),
        (1024, 32, 32, 128, False),
        (2048, 32, 32, 128, False),
        (4096, 32, 32, 128, False),
    ],
    # Llama-3-8B: GQA 32Q/8KV, hd=128 — causal only
    "llama3-8b": [(s, 32, 8, 128, True) for s in [512, 1024, 2048, 4096, 8192]],
    # Llama-3-70B: GQA 64Q/8KV, hd=128 — causal only
    "llama3-70b": [(s, 64, 8, 128, True) for s in [512, 1024, 2048, 4096, 8192]],
    # Mistral-7B: GQA 32Q/8KV, hd=128 — causal only
    "mistral-7b": [(s, 32, 8, 128, True) for s in [512, 1024, 2048, 4096, 8192]],
}
_CONFIGS = _MODEL_CONFIGS[_model]

_OUTPUT_DIR = str(Path(__file__).parent)


# ---------------------------------------------------------------------------
# FLOPs / bytes helpers
# ---------------------------------------------------------------------------


def _flops(
    q_len: int, kv_len: int, num_qo_heads: int, head_dim: int, causal: bool
) -> int:
    # For causal with q_len < kv_len the mask removes only the last q*(q-1)/2
    # entries of the attended set, not half the matrix (which factor=2 assumes).
    attended = q_len * kv_len - q_len * (q_len - 1) // 2 if causal else q_len * kv_len
    return attended * num_qo_heads * head_dim * 4


def _bytes(
    q_len: int, kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    return 2 * head_dim * (2 * q_len * num_qo_heads + 2 * kv_len * num_kv_heads)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    configs = []

    for seq_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        q = torch.randn(
            seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )

        flops = _flops(seq_len, seq_len, num_qo_heads, head_dim, causal)
        theo_bytes = _bytes(seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
        causal_str = "causal" if causal else "nc"
        configs.append(
            KernelConfig(
                name=f"s{seq_len}_{causal_str}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v, c=causal: (
                        flashinfer.single_prefill_with_kv_cache_return_lse(
                            q, k, v, causal=c, backend="fa2"
                        )
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=seq_len,
                label=f"seq={seq_len:>5d}  {'causal' if causal else 'non-causal'}",
            )
        )

    if _q_len_asym > 0:
        for seq_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
            if not causal or seq_len <= _q_len_asym:
                continue  # asymmetric is causal-only; skip if kv not larger than q
            kv_len = seq_len
            q = torch.randn(
                _q_len_asym, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
            )
            k = torch.randn(
                kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
            )
            v = torch.randn(
                kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
            )

            flops = _flops(_q_len_asym, kv_len, num_qo_heads, head_dim, causal=True)
            theo_bytes = _bytes(
                _q_len_asym, kv_len, num_qo_heads, num_kv_heads, head_dim
            )
            configs.append(
                KernelConfig(
                    name=f"q{_q_len_asym}_kv{kv_len}_causal",
                    run_fn=torch.inference_mode()(
                        lambda q=q, k=k, v=v: (
                            flashinfer.single_prefill_with_kv_cache_return_lse(
                                q, k, v, causal=True, backend="fa2"
                            )
                        )
                    ),
                    theoretical_flops=flops,
                    theoretical_bytes=theo_bytes,
                    num_tokens=_q_len_asym,
                    label=f"q={_q_len_asym:>4d} kv={kv_len:>5d}  causal",
                )
            )

    return configs


@torch.inference_mode()
def _make_batch_configs(batch_size: int, q_len: int) -> list[KernelConfig]:
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    configs = []
    for seq_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        kv_len = seq_len
        if not causal or kv_len <= q_len:
            continue  # asymmetric (q < kv) is causal-only; need kv_len > q_len
        q = torch.randn(
            batch_size * q_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
        )
        k = torch.randn(
            batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )
        v = torch.randn(
            batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )
        qo_indptr = torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device="cuda"
        )
        kv_indptr = torch.arange(
            0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device="cuda"
        )

        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim, causal=True
        )

        flops = batch_size * _flops(q_len, kv_len, num_qo_heads, head_dim, causal=True)
        theo_bytes = batch_size * _bytes(
            q_len, kv_len, num_qo_heads, num_kv_heads, head_dim
        )
        configs.append(
            KernelConfig(
                name=f"ragged_bs{batch_size}_q{q_len}_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v, w=wrapper: w.run(q, k, v, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=batch_size * q_len,
                label=f"ragged bs={batch_size} q={q_len:>4d} kv={kv_len:>5d}",
            )
        )
    return configs


@torch.inference_mode()
def _make_paged_configs(
    batch_size: int, page_size: int, q_len: int
) -> list[KernelConfig]:
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    configs = []
    for seq_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        kv_len = seq_len
        if not causal or kv_len <= q_len:
            continue  # asymmetric (q < kv) is causal-only; need kv_len > q_len

        pages_per_seq = math.ceil(kv_len / page_size)
        last_page_len = kv_len - (pages_per_seq - 1) * page_size
        total_pages = batch_size * pages_per_seq

        q = torch.randn(
            batch_size * q_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
        )
        kv_cache = torch.randn(
            total_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.half,
            device="cuda",
        )

        qo_indptr = torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device="cuda"
        )
        paged_kv_indptr = torch.arange(
            0,
            (batch_size + 1) * pages_per_seq,
            pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
        _rng = torch.Generator(device="cuda").manual_seed(42)
        paged_kv_indices = torch.randperm(
            total_pages, dtype=torch.int32, device="cuda", generator=_rng
        )
        paged_kv_last_page_len = torch.full(
            (batch_size,), last_page_len, dtype=torch.int32, device="cuda"
        )

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
        )

        flops = batch_size * _flops(q_len, kv_len, num_qo_heads, head_dim, causal=True)
        theo_bytes = batch_size * _bytes(
            q_len, kv_len, num_qo_heads, num_kv_heads, head_dim
        )
        configs.append(
            KernelConfig(
                name=f"paged{page_size}_bs{batch_size}_q{q_len}_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, kv=kv_cache, w=wrapper: w.run(q, kv, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                num_tokens=batch_size * q_len,
                label=f"paged(ps={page_size:>2d}) bs={batch_size} q={q_len:>4d} kv={kv_len:>5d}",
            )
        )
    return configs


if __name__ == "__main__":
    # Defer GPU tensor allocation: --replot and --list-presets don't need a CUDA device.
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv

    def _build_configs() -> list[KernelConfig]:
        cfgs = _make_configs()
        if _batch_size > 0:
            cfgs += _make_batch_configs(_batch_size, _batch_q_len)
        if _paged_size > 0:
            for ps in _page_sizes:
                cfgs += _make_paged_configs(_paged_size, ps, _batch_q_len)
        return cfgs

    # Widen regex when any batch/paged mode is active so counter collection
    # covers BatchPrefillWithRaggedKVCacheKernel and BatchPrefillWithPagedKVCacheKernel.
    _kernel_regex = (
        "PrefillWith.*KVCacheKernel"
        if (_batch_size > 0 or _paged_size > 0)
        else "SinglePrefillWithKVCacheKernel"
    )
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _build_configs(),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters=_counters,
        kernel_name_regex=_kernel_regex,
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
