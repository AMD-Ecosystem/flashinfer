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

DeepSeek MLA decode benchmark (AITER backend), with cost attribution.

Four modes, so the number is attributable rather than opaque:

  attn  — wrapper.run(...) end to end.
  cat   — the two torch.cat calls alone, so `attn - cat` under --separate
          estimates what a combined-buffer layout leaves behind.
  plan  — wrapper.plan(...) and its host round-trips.
  pool  — run() against KV *pool* size at a fixed active set. Excluded from
          --mode all: it allocates several GB per config.

Shapes are DeepSeek-V3/R1 MLA: ckv=512, kpe=64, page_size=1 (what vLLM's AITER
MLA requires), bf16, num_heads 16 (TP8) and 128 (TP1).

The KV cache is allocated combined by default, which run() consumes as a
zero-copy view; --separate forces the two-allocation layout and its
concatenation fallback.

Run:
    python benchmarks/rocm_benchmarks/bench_mla_hip.py --timing-only
    python benchmarks/rocm_benchmarks/bench_mla_hip.py --timing-only --separate
    python benchmarks/rocm_benchmarks/bench_mla_hip.py --mode pool --timing-only
    python benchmarks/rocm_benchmarks/bench_mla_hip.py --heads 16,128
    python benchmarks/rocm_benchmarks/bench_mla_hip.py                 # + roofline

Bench flags are parsed at module level because rocprofv3 re-executes this script
per PMC pass with the same sys.argv; the subprocess must build identical configs.
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
    help="Output-file label prefix (default: 'mla' for roofline, 'mla_<preset>' otherwise).",
)
_bench_parser.add_argument(
    "--mode",
    default="all",
    choices=["attn", "cat", "plan", "pool", "all"],
    help="Which cost(s) to sweep. Default: all ('pool' is separate, see below).",
)
_bench_parser.add_argument(
    "--heads",
    default="16",
    help="Comma-separated num_heads per GPU (AITER MLA: multiples of 16, <=128). Default: 16.",
)
_bench_parser.add_argument(
    "--separate",
    action="store_true",
    help=(
        "Allocate ckv/kpe as two separate tensors (the pre-fix layout) instead of "
        "splitting one combined buffer. Forces run()'s concatenation fallback, so "
        "this is how you reproduce the 'before' numbers."
    ),
)
_bench_args, _ = _bench_parser.parse_known_args()

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("mla" if _counters == "roofline" else f"mla_{_counters}")
)

# ---------------------------------------------------------------------------
# Sweep configuration — DeepSeek-V3/R1 MLA decode (q_len=1).
# ---------------------------------------------------------------------------
_HEAD_DIM_CKV = 512  # kv_lora_rank
_HEAD_DIM_KPE = 64  # qk_rope_head_dim
_QK_HEAD_DIM = _HEAD_DIM_CKV + _HEAD_DIM_KPE  # 576
_DTYPE = torch.bfloat16
_PAGE_SIZE = 1  # what vLLM's AITER MLA backend requires
_BATCHES = [1, 8, 32]
_KV_LENS = [1024, 8192, 32768]

# --mode pool: fixed (small) active set, growing page pool. Not in --mode all
# because it allocates several GB per config.
_POOL_BATCH = 8
_POOL_KV_LEN = 1024
_POOL_PAGES = [8192, 262144, 1048576, 4194304]  # 0.01 .. 4.5 GB at 576 x bf16

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(batch: int, kv_len: int, num_heads: int) -> int:
    # decode q_len=1: QK^T over 576 dims + PV over 512 dims, 2 flops per MAC.
    return batch * num_heads * kv_len * (_QK_HEAD_DIM + _HEAD_DIM_CKV) * 2


def _kv_bytes(batch: int, kv_len: int) -> int:
    # MLA is MQA: the 576-dim latent KV is read once per sequence, not per head.
    return batch * kv_len * _QK_HEAD_DIM * 2  # bf16


def _attn_bytes(batch: int, kv_len: int, num_heads: int) -> int:
    # KV dominates; add q in and o out.
    return _kv_bytes(batch, kv_len) + batch * num_heads * (
        _QK_HEAD_DIM * 2 + _HEAD_DIM_CKV * 2
    )


def _alloc_kv(num_pages, device):
    """Allocate the paged KV cache in the combined (default) or separate layout.

    Combined is what production does and what run() turns into a zero-copy view;
    --separate reproduces the two-allocation layout that forces the concatenation.
    """
    if _bench_args.separate:
        ckv = torch.randn(
            num_pages, _PAGE_SIZE, _HEAD_DIM_CKV, dtype=_DTYPE, device=device
        )
        kpe = torch.randn(
            num_pages, _PAGE_SIZE, _HEAD_DIM_KPE, dtype=_DTYPE, device=device
        )
        return ckv, kpe
    cache = torch.randn(
        num_pages, _PAGE_SIZE, _QK_HEAD_DIM, dtype=_DTYPE, device=device
    )
    # Views keep `cache` alive, so returning only the halves is safe.
    return cache.split([_HEAD_DIM_CKV, _HEAD_DIM_KPE], dim=-1)


def _build_mla_inputs(batch, kv_len, num_heads, device):
    """Build DeepSeek MLA decode tensors for one (batch, kv_len, num_heads)."""
    num_pages = batch * kv_len  # page_size == 1
    q_nope = torch.randn(batch, num_heads, _HEAD_DIM_CKV, dtype=_DTYPE, device=device)
    q_pe = torch.randn(batch, num_heads, _HEAD_DIM_KPE, dtype=_DTYPE, device=device)
    ckv, kpe = _alloc_kv(num_pages, device)
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_len
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    return q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices, kv_lens


def _make_wrapper(device):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchMLAPagedAttentionWrapper(ws)


def _make_attn_config(batch, kv_len, num_heads):
    device = torch.device("cuda")
    q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices, kv_lens = (
        _build_mla_inputs(batch, kv_len, num_heads, device)
    )
    wrapper = _make_wrapper(device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        _HEAD_DIM_CKV,
        _HEAD_DIM_KPE,
        _PAGE_SIZE,
        False,  # causal — ignored by the ROCm wrapper, passed for API parity
        1.0 / (_QK_HEAD_DIM**0.5),
        _DTYPE,
        _DTYPE,
    )
    return KernelConfig(
        name=f"attn_h{num_heads}_b{batch}_kv{kv_len}",
        run_fn=torch.inference_mode()(
            lambda w=wrapper, a=q_nope, b=q_pe, c=ckv, d=kpe: w.run(a, b, c, d)
        ),
        theoretical_flops=_flops(batch, kv_len, num_heads),
        theoretical_bytes=_attn_bytes(batch, kv_len, num_heads),
        num_tokens=batch,  # one decoded token per sequence
        label=f"attn  h={num_heads:>3d} b={batch:>3d} kv={kv_len:>6d}",
    )


def _make_cat_config(batch, kv_len, num_heads):
    """Isolate the per-call torch.cat that run() performs on the whole KV cache."""
    device = torch.device("cuda")
    q_nope, q_pe, ckv, kpe, *_ = _build_mla_inputs(batch, kv_len, num_heads, device)

    def _cat(a=q_nope, b=q_pe, c=ckv, d=kpe):
        torch.cat([a, b], dim=-1)
        torch.cat([c.unsqueeze(2), d.unsqueeze(2)], dim=-1)

    return KernelConfig(
        name=f"cat_h{num_heads}_b{batch}_kv{kv_len}",
        run_fn=torch.inference_mode()(_cat),
        # Pure data movement: read ckv+kpe and q_nope+q_pe, write both results.
        theoretical_flops=0,
        theoretical_bytes=2
        * (_kv_bytes(batch, kv_len) + batch * num_heads * _QK_HEAD_DIM * 2),
        num_tokens=batch,
        label=f"cat   h={num_heads:>3d} b={batch:>3d} kv={kv_len:>6d}",
    )


def _make_plan_config(batch, kv_len, num_heads):
    """Isolate plan(): the .item() host syncs in _kv_lens_to_last_page_len_cpu."""
    device = torch.device("cuda")
    _, _, _, _, qo_indptr, kv_indptr, kv_indices, kv_lens = _build_mla_inputs(
        batch, kv_len, num_heads, device
    )
    wrapper = _make_wrapper(device)

    def _plan(w=wrapper):
        w.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            num_heads,
            _HEAD_DIM_CKV,
            _HEAD_DIM_KPE,
            _PAGE_SIZE,
            False,
            1.0 / (_QK_HEAD_DIM**0.5),
            _DTYPE,
            _DTYPE,
        )

    return KernelConfig(
        name=f"plan_h{num_heads}_b{batch}_kv{kv_len}",
        run_fn=torch.inference_mode()(_plan),
        theoretical_flops=0,
        theoretical_bytes=batch * 4 * 2,  # kv_indptr + kv_lens reads; host-bound
        num_tokens=batch,
        label=f"plan  h={num_heads:>3d} b={batch:>3d} kv={kv_len:>6d}",
    )


def _make_pool_config(pool_pages, batch, kv_len, num_heads):
    """run() cost vs KV *pool* size at a fixed active set.

    run() cats `ckv_cache` in full, so its cost tracks the allocated pool rather
    than the live working set. Production sizes the pool to fill HBM, so a bench
    that allocates exactly batch*kv_len pages measures the best case and misses
    this entirely. Holding the active set fixed and growing only the pool isolates
    it: every row does identical attention work.
    """
    device = torch.device("cuda")
    q_nope, q_pe, _, _, qo_indptr, kv_indptr, kv_indices, kv_lens = _build_mla_inputs(
        batch, kv_len, num_heads, device
    )
    # Pool is larger than the active set; kv_indices still addresses only the
    # first batch*kv_len pages.
    ckv, kpe = _alloc_kv(pool_pages, device)
    wrapper = _make_wrapper(device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        _HEAD_DIM_CKV,
        _HEAD_DIM_KPE,
        _PAGE_SIZE,
        False,
        1.0 / (_QK_HEAD_DIM**0.5),
        _DTYPE,
        _DTYPE,
    )
    pool_gb = pool_pages * _QK_HEAD_DIM * 2 / 2**30
    # Traffic depends on which path run() takes. With adjacent halves (the
    # default) there is no copy at all and the only traffic is the attention
    # read of the live KV; the pool-sized cat exists only under --separate, and
    # counting it unconditionally would attribute GBs of nonexistent traffic to
    # the zero-copy path and make it appear to scale with pool capacity.
    attn_bytes = batch * kv_len * _QK_HEAD_DIM * 2
    cat_bytes = 2 * pool_pages * _QK_HEAD_DIM * 2 if _bench_args.separate else 0
    return KernelConfig(
        name=f"pool_h{num_heads}_b{batch}_kv{kv_len}_p{pool_pages}",
        run_fn=torch.inference_mode()(
            lambda w=wrapper, a=q_nope, b=q_pe, c=ckv, d=kpe: w.run(a, b, c, d)
        ),
        theoretical_flops=_flops(batch, kv_len, num_heads),
        theoretical_bytes=attn_bytes + cat_bytes,
        num_tokens=batch,
        label=f"pool  h={num_heads:>3d} b={batch:>3d} kv={kv_len:>6d} pool={pool_gb:>6.2f}GB",
    )


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    device = torch.device("cuda:0")
    if not is_aiter_available(device, "mla"):
        # is_aiter_supported also returns False on a CUDA build and with no
        # visible GPU, and gcnArchName does not exist in either case -- so the
        # lookup would raise in exactly the situations this message is for.
        try:
            arch = torch.cuda.get_device_properties(device).gcnArchName
        except Exception:
            arch = "unknown"
        print(
            f"AITER MLA requires gfx942/gfx950; this device is {arch!r}. Nothing to run."
        )
        return []

    heads_list = [int(h) for h in _bench_args.heads.split(",") if h.strip()]
    modes = ["attn", "cat", "plan"] if _bench_args.mode == "all" else [_bench_args.mode]
    builders = {
        "attn": _make_attn_config,
        "cat": _make_cat_config,
        "plan": _make_plan_config,
    }

    configs: list[KernelConfig] = []
    for mode in modes:
        if mode == "pool":
            for num_heads in heads_list:
                for pool_pages in _POOL_PAGES:
                    configs.append(
                        _make_pool_config(
                            pool_pages, _POOL_BATCH, _POOL_KV_LEN, num_heads
                        )
                    )
            continue
        for num_heads in heads_list:
            for batch in _BATCHES:
                for kv_len in _KV_LENS:
                    configs.append(builders[mode](batch, kv_len, num_heads))
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
