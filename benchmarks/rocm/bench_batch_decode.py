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

Measures the eager fa2-vs-aiter gap across a serving-shaped batch x kv_len
sweep. Graph-captured AITER decode (capture-at-max and a declared max_seq_len)
is timed by ``benchmarks/rocm/bench_decode_graph.py``.

Shapes: decode (q_len=1), 8 KV heads, 16-bit query. Sweep: batch x kv_len x
qo_heads x page_size x head_dim, each overridable on the command line.

Arms (``--backend``, comma-separated): ``fa2`` and ``fa2_tc`` (the
``use_tensor_cores=True`` kernel), ``aiter``, ``auto`` and ``auto_tc`` (labelled
with what ``auto`` resolved to), and ``engine`` -- AITER's ``pa_fwd_asm`` on its
own shuffled cache layout, the call vLLM makes; hd 128 bf16 only. ``--step``
times ``plan()`` + ``run()``, since a serving loop re-plans every step.

Run:
    python benchmarks/rocm/bench_batch_decode.py               # full pipeline
    python benchmarks/rocm/bench_batch_decode.py --timing-only # no profiling
    python benchmarks/rocm/bench_batch_decode.py --backend fa2
    python benchmarks/rocm/bench_batch_decode.py --backend aiter
    python benchmarks/rocm/bench_batch_decode.py --counters stall
    # The short-KV / multi-head grid, in passes: every config is built before
    # the first run, so in one go it is ~67 GiB resident against ~23 and
    # ~45 GiB for the two passes below.
    # Distinct --label per pass; the timing CSV is opened "w", so a shared
    # label makes the second pass overwrite the first.
    python benchmarks/rocm/bench_batch_decode.py --timing-only --output-dir /out \
        --label short --batches 1,8,32,128,256 --kv-lens 128,256,512,1024 --qo-heads 32,64
    python benchmarks/rocm/bench_batch_decode.py --timing-only --output-dir /out \
        --label mid --batches 1,8,32,128,256 --kv-lens 2048,4096 --qo-heads 32,64

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
from flashinfer.rocm.aiter_utils import is_aiter_available
from flashinfer.jit.core import logger as _jit_logger

# Suppress routine JIT INFO/DEBUG output; WARNING still surfaces compile errors.
_jit_logger.setLevel(logging.WARNING)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "profiler" / "rocm")
)
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
_ARMS = ("fa2", "fa2_tc", "aiter", "auto", "auto_tc", "engine")


def _arm_list(raw: str) -> list[str]:
    arms = ["fa2", "aiter"] if raw == "both" else raw.split(",")
    bad = [a for a in arms if a not in _ARMS]
    if bad or not arms:
        raise argparse.ArgumentTypeError(f"unknown arm(s) {bad}; choose from {_ARMS}")
    return arms


_bench_parser.add_argument(
    "--backend",
    type=_arm_list,
    default=["fa2", "aiter"],
    metavar="ARM,ARM,...",
    help=f"Arms to sweep, from {', '.join(_ARMS)}; 'both' is fa2,aiter (the default).",
)
_bench_parser.add_argument(
    "--kv-dtype",
    default="bf16",
    choices=["bf16", "fp16", "fp8", "fp8-fnuz"],
    help="KV cache dtype; the query matches it, or is bf16 under an fp8 cache. "
    "fp8 is this GPU's native e4m3. Default: bf16.",
)
_bench_parser.add_argument(
    "--step",
    action="store_true",
    help="Time plan() + run() per call instead of run() alone.",
)


def _int_list(raw: str) -> list[int]:
    # Strict: an empty list is falsy and would fall back to the default grid,
    # and a dropped field ("1,,8") would silently sweep a different one. Either
    # way the run is not the grid that was asked for.
    tokens = raw.split(",")
    if not tokens or any(not tok.strip() for tok in tokens):
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {raw!r}")
    values = [int(tok) for tok in tokens]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive ints, got {raw!r}")
    return values


# Every config is built before the first run, so the whole grid is resident at
# once: ~55 GiB at the defaults below, ~116 GiB if kv_lens is widened to 16384.
# Override these to run a wide sweep in passes on a shared board.
_bench_parser.add_argument(
    "--batches",
    type=_int_list,
    default=None,
    metavar="N,N,...",
    help="Batch sizes to sweep. Default: 1,8,32,128,256.",
)
_bench_parser.add_argument(
    "--kv-lens",
    type=_int_list,
    default=None,
    metavar="N,N,...",
    help="KV lengths to sweep. Default: 1024,2048,4096,8192.",
)
_bench_parser.add_argument(
    "--page-sizes",
    type=_int_list,
    default=None,
    metavar="N,N,...",
    help="Page sizes to sweep. Default: 16.",
)
_bench_parser.add_argument(
    "--head-dims",
    type=_int_list,
    default=None,
    metavar="N,N,...",
    help="Head dims to sweep. Default: 128.",
)
_bench_parser.add_argument(
    "--qo-heads",
    type=_int_list,
    default=None,
    metavar="N,N,...",
    help="Query head counts to sweep, at a fixed 8 KV heads. Default: 32.",
)
# Defaults beside the script, which is inside the source tree -- so a run
# against a pinned checkout needs somewhere else to write.
_bench_parser.add_argument(
    "--output-dir",
    default=None,
    metavar="DIR",
    help="Where to write CSVs and plots. Default: this script's directory.",
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
_NUM_KV_HEADS = 8
_HEAD_DIMS = _bench_args.head_dims or [128]
_PAGE_SIZES = _bench_args.page_sizes or [16]


def _kv_dtype() -> torch.dtype:
    if _bench_args.kv_dtype == "bf16":
        return torch.bfloat16
    if _bench_args.kv_dtype == "fp16":
        return torch.float16
    if _bench_args.kv_dtype == "fp8-fnuz":
        return torch.float8_e4m3fnuz
    from flashinfer.rocm.prefill import _native_fp8_dtype

    native = _native_fp8_dtype()
    if native is None:
        raise SystemExit("--kv-dtype fp8: aiter.dtypes.fp8 is unreadable")
    return native


_KV_DTYPE = _kv_dtype()
_Q_DTYPE = torch.float16 if _KV_DTYPE is torch.float16 else torch.bfloat16
# Defaults are unchanged by the short-KV work: reaching 16384 would take the
# resident set from ~55 to ~116 GiB, which fits an idle MI300X and not a shared
# one. Widen with the flags, in passes -- see the module docstring.
_BATCHES = _bench_args.batches or [1, 8, 32, 128, 256]
_KV_LENS = _bench_args.kv_lens or [1024, 2048, 4096, 8192]
_QO_HEADS = _bench_args.qo_heads or [32]
if any(h % _NUM_KV_HEADS for h in _QO_HEADS):
    raise SystemExit(
        f"--qo-heads must be multiples of the fixed {_NUM_KV_HEADS} KV heads; "
        f"got {_QO_HEADS}"
    )

_OUTPUT_DIR = _bench_args.output_dir or str(Path(__file__).parent)


def _flops(kv_len: int, num_qo_heads: int, head_dim: int) -> int:
    # decode: q_len=1 → attended = kv_len; QK^T + PV ≈ 2 matmuls → factor 4.
    return kv_len * num_qo_heads * head_dim * 4


def _bytes(kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int) -> int:
    # Dominated by reading K and V (q_len=1); q and o are 2-byte.
    kv_bytes = _KV_DTYPE.itemsize
    return head_dim * (2 * 2 * num_qo_heads + kv_bytes * 2 * kv_len * num_kv_heads)


def _build_paged_kv(
    batch, kv_len, page_size, num_qo_heads, num_kv_heads, head_dim, dtype, device
):
    """Build batch-paged decode KV + query tensors for one (batch, kv_len)."""
    num_full_pages, last_tokens = divmod(kv_len, page_size)
    if last_tokens == 0:
        last_tokens = page_size
    else:
        num_full_pages += 1
    total_pages = num_full_pages * batch

    # Seeded so every arm of one shape sees the same tensors. randn fits e4m3
    # at unit scale, so an fp8 cache is a plain cast.
    torch.manual_seed(0)
    kv_data = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)
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
    q = torch.randn(batch, num_qo_heads, head_dim, dtype=_Q_DTYPE, device=device)
    return q, kv_data, kv_indptr, kv_indices, kv_last_page_len


def _engine_run_fn(q, kv_data, kv_indices, batch, kv_len, page_size):
    """AITER pa_fwd_asm on vLLM's layout: K [P,H,D/x,ps,x], V [P,H,ps/x,D,x]."""
    import aiter

    pages, _, ps, h, d = kv_data.shape
    x = 16 // kv_data.element_size()
    k5 = kv_data[:, 0].view(pages, ps, h, d // x, x).permute(0, 2, 3, 1, 4).contiguous()
    v5 = kv_data[:, 1].view(pages, ps // x, x, h, d).permute(0, 3, 1, 4, 2).contiguous()
    block_tables = kv_indices.view(batch, -1).contiguous()
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)

    def run():
        return aiter.pa_fwd_asm(
            q, k5, v5, block_tables, seq_lens, block_tables.stride(0), out_=out
        )

    return run


def _make_wrapper_config(arm, batch, kv_len, num_qo_heads, page_size, head_dim):
    device = torch.device("cuda")
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch,
        kv_len,
        page_size,
        num_qo_heads,
        _NUM_KV_HEADS,
        head_dim,
        _KV_DTYPE,
        device,
    )
    shape = f"b{batch}_kv{kv_len}_h{num_qo_heads}_p{page_size}_d{head_dim}"
    if arm == "engine":
        if head_dim != 128 or _Q_DTYPE is not torch.bfloat16 or _KV_DTYPE.itemsize != 2:
            raise ValueError("pa_fwd_asm: hd 128 with a bf16 cache only here")
        run_fn = _engine_run_fn(q, kv_data, kv_indices, batch, kv_len, page_size)
        resolved = "engine"
    else:
        backend = {"fa2_tc": "fa2", "auto_tc": "auto"}.get(arm, arm)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", backend=backend, use_tensor_cores=arm.endswith("_tc")
        )

        def plan():
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                _NUM_KV_HEADS,
                head_dim,
                page_size,
                pos_encoding_mode="NONE",
                q_data_type=_Q_DTYPE,
                kv_data_type=_KV_DTYPE,
            )

        plan()
        resolved = wrapper.backend
        if _bench_args.step:

            def run_fn(q=q, kv=kv_data):
                plan()
                return wrapper.run(q, kv)

        else:

            def run_fn(q=q, kv=kv_data):
                return wrapper.run(q, kv)

    return KernelConfig(
        # The resolved backend is in the name, so an auto row is attributable.
        name=f"{arm}-{resolved}_{shape}",
        run_fn=torch.inference_mode()(run_fn),
        theoretical_flops=batch * _flops(kv_len, num_qo_heads, head_dim),
        theoretical_bytes=batch * _bytes(kv_len, num_qo_heads, _NUM_KV_HEADS, head_dim),
        num_tokens=batch,  # one decoded token per sequence
        label=f"{arm:<7s}({resolved:<6s}) b={batch:>3d} kv={kv_len:>5d} "
        f"h={num_qo_heads:>3d} p={page_size:>2d} d={head_dim:>3d}",
    )


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    arms = list(_bench_args.backend)
    if "aiter" in arms and not is_aiter_available(
        torch.device("cuda:0"), "batch_decode"
    ):
        print("AITER not supported on this device; dropping the aiter sweep.")
        arms = [a for a in arms if a != "aiter"]

    configs: list[KernelConfig] = []
    for arm in arms:
        for batch in _BATCHES:
            for kv_len in _KV_LENS:
                for num_qo_heads in _QO_HEADS:
                    for page_size in _PAGE_SIZES:
                        for head_dim in _HEAD_DIMS:
                            try:
                                configs.append(
                                    _make_wrapper_config(
                                        arm,
                                        batch,
                                        kv_len,
                                        num_qo_heads,
                                        page_size,
                                        head_dim,
                                    )
                                )
                            except Exception as e:  # report the cell, keep sweeping
                                print(
                                    f"skip {arm} b={batch} kv={kv_len} h={num_qo_heads} "
                                    f"p={page_size} d={head_dim}: {type(e).__name__}: {e}"
                                )
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
