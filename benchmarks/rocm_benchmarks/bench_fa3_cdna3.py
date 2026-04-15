"""
Copyright (c) 2025-2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FA3-CDNA3 vs FA2 comparison benchmark for AMD MI300X.

Benchmarks the FA3-CDNA3 8-wave ping-pong prefill kernel (head_dim=256)
against:
  - FlashInfer FA2 HIP path (existing baseline)
  - AITER flash_attn_varlen_func (if available)

Configurations: N=1K/2K/4K/8K, d=256, nhead=32 (GQA 32/8), causal+non-causal.

Run:
    # Full roofline pipeline:
    python benchmarks/rocm_benchmarks/bench_fa3_cdna3.py

    # Timing only (no rocprofv3):
    python benchmarks/rocm_benchmarks/bench_fa3_cdna3.py --timing-only

    # Compare against FA2 only (no AITER):
    python benchmarks/rocm_benchmarks/bench_fa3_cdna3.py --no-aiter

    # Select counter preset:
    python benchmarks/rocm_benchmarks/bench_fa3_cdna3.py --counters occupancy

Output files (all gitignored):
    benchmarks/rocm_benchmarks/fa3_cdna3_timing.csv
    benchmarks/rocm_benchmarks/fa3_cdna3_counters.yml
    benchmarks/rocm_benchmarks/fa3_cdna3_counter_collection.csv
    benchmarks/rocm_benchmarks/fa3_cdna3_roofline.png

Performance targets (single prefill, d=256, 32 Q-heads, 8 KV-heads, MI300X):
    N=1024  causal:  ~50-80  us   (FA2 baseline ~150 us)
    N=2048  causal:  ~100-160 us  (FA2 baseline ~350 us)
    N=4096  causal:  ~200-350 us  (FA2 baseline ~800 us)
    N=8192  causal:  ~400-700 us  (FA2 baseline ~1800 us)
    Hopper FA3 reference (H100, same configs): ~200-3000 us
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path

import torch
import flashinfer
from flashinfer.jit.core import logger as fi_logger

fi_logger.setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing
# ---------------------------------------------------------------------------
_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument(
    "--counters",
    default="roofline",
    metavar="PRESET_OR_FILE",
    help=(
        "Counter preset ('roofline', 'occupancy', 'stall', 'compute', "
        "'memory', 'basic') or path to a rocprofv3 YAML file. "
        "Default: roofline."
    ),
)
_bench_parser.add_argument(
    "--label",
    default=None,
    metavar="PREFIX",
    help="Output-file label prefix. Default: 'fa3_cdna3'.",
)
_bench_parser.add_argument(
    "--no-aiter",
    action="store_true",
    help="Skip AITER flash_attn_varlen_func comparison.",
)
_bench_args, _remaining = _bench_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("fa3_cdna3" if _counters == "roofline" else f"fa3_cdna3_{_counters}")
)
_include_aiter = not _bench_args.no_aiter

# ---------------------------------------------------------------------------
# Check AITER availability
# ---------------------------------------------------------------------------
_AITER_AVAILABLE = False
if _include_aiter:
    try:
        from aiter import flash_attn_varlen_func as aiter_fa_varlen

        _AITER_AVAILABLE = True
    except ImportError:
        warnings.warn(
            "AITER not available (import failed). "
            "Run with --no-aiter to suppress this warning.",
            stacklevel=1,
        )

# ---------------------------------------------------------------------------
# Benchmark configurations:
#   (seq_len, num_qo_heads, num_kv_heads, head_dim, causal)
#
# Focus: d=256 (the FA3-CDNA3 optimized path), GQA 32/8, seqlen 1K..8K.
# ---------------------------------------------------------------------------
_CONFIGS = [
    # d=256 sweep (FA3-CDNA3 target), causal
    (1024, 32, 8, 256, True),
    (2048, 32, 8, 256, True),
    (4096, 32, 8, 256, True),
    (8192, 32, 8, 256, True),
    # d=256 sweep, non-causal (2x FLOPs -> more compute-bound)
    (1024, 32, 8, 256, False),
    (2048, 32, 8, 256, False),
    (4096, 32, 8, 256, False),
    (8192, 32, 8, 256, False),
    # MHA (no GQA) at d=256
    (4096, 32, 32, 256, True),
    (8192, 32, 32, 256, True),
    # Use-case configs: GQA 16/4, non-causal (from single_prefill_example.py)
    (512, 16, 4, 256, False),
    (1024, 16, 4, 256, False),
    (2048, 16, 4, 256, False),
    (4096, 16, 4, 256, False),
    (8192, 16, 4, 256, False),
]

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(seq_len: int, nhead_q: int, head_dim: int, causal: bool) -> int:
    """Theoretical FLOPs for QK^T + softmax(S)V (ignoring softmax overhead)."""
    # Q*K^T: 2 * N * N * H * d (causal: N*(N+1)/2 * H * d * 2 ~ N^2 * H * d)
    # S*V:   same
    factor = 2 if causal else 4
    return seq_len * seq_len * nhead_q * head_dim * factor


def _bytes(seq_len: int, nhead_q: int, nhead_k: int, head_dim: int) -> int:
    """Theoretical bytes (cold-cache lower bound): Q, K, V, O in FP16."""
    return 2 * seq_len * head_dim * (nhead_q * 2 + nhead_k * 2)  # fp16


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    configs = []

    for seq_len, nhead_q, nhead_k, head_dim, causal in _CONFIGS:
        q = torch.randn(seq_len, nhead_q, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(seq_len, nhead_k, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(seq_len, nhead_k, head_dim, dtype=torch.half, device="cuda")

        flops = _flops(seq_len, nhead_q, head_dim, causal)
        theo_bytes = _bytes(seq_len, nhead_q, nhead_k, head_dim)
        causal_str = "causal" if causal else "nc"
        label_str = (
            f"FA3-CDNA3  seq={seq_len:>5d}  h={nhead_q}/{nhead_k}  d={head_dim}  "
            f"{causal_str}"
        )
        label_fa2 = (
            f"FA2        seq={seq_len:>5d}  h={nhead_q}/{nhead_k}  d={head_dim}  "
            f"{causal_str}"
        )

        # ---- FA3-CDNA3 (our kernel) ----
        def fa3_fn(q=q, k=k, v=v, c=causal):
            return flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=c, backend="fa3_cdna3"
            )

        configs.append(
            KernelConfig(
                name=f"fa3_cdna3_s{seq_len}_{causal_str}_d{head_dim}",
                run_fn=torch.inference_mode()(fa3_fn),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=label_str,
            )
        )

        # ---- FA2 baseline ----
        def fa2_fn(q=q, k=k, v=v, c=causal):
            return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=c)

        configs.append(
            KernelConfig(
                name=f"fa2_s{seq_len}_{causal_str}_d{head_dim}",
                run_fn=torch.inference_mode()(fa2_fn),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=label_fa2,
            )
        )

        # ---- AITER comparison ----
        if _AITER_AVAILABLE:
            # Build cu_seqlens for AITER varlen API.
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

            def aiter_fn(q=q, k=k, v=v, cs=cu_seqlens, N=seq_len, c=causal):
                q_flat = q.view(-1, q.shape[-2], q.shape[-1])
                k_flat = k.view(-1, k.shape[-2], k.shape[-1])
                v_flat = v.view(-1, v.shape[-2], v.shape[-1])
                return aiter_fa_varlen(
                    q_flat,
                    k_flat,
                    v_flat,
                    cs,
                    cs,
                    max_seqlen_q=N,
                    max_seqlen_k=N,
                    causal=c,
                )

            label_aiter = (
                f"AITER      seq={seq_len:>5d}  h={nhead_q}/{nhead_k}  d={head_dim}  "
                f"{causal_str}"
            )
            configs.append(
                KernelConfig(
                    name=f"aiter_s{seq_len}_{causal_str}_d{head_dim}",
                    run_fn=torch.inference_mode()(aiter_fn),
                    theoretical_flops=flops,
                    theoretical_bytes=theo_bytes,
                    label=label_aiter,
                )
            )

    return configs


@torch.inference_mode()
def _run_correctness_tests():
    """Compare FA3-CDNA3 output vs PyTorch SDPA ground truth."""
    import torch.nn.functional as F

    print("\n" + "=" * 72)
    print("  CORRECTNESS TEST: FA3-CDNA3 vs PyTorch SDPA ground truth")
    print("=" * 72)

    test_configs = [
        (64, 1, 1, 256, False),
        (64, 1, 1, 256, True),
        (128, 32, 8, 256, False),
        (128, 32, 8, 256, True),
        (512, 32, 8, 256, True),
        (1024, 32, 8, 256, True),
        (2048, 32, 8, 256, True),
        (4096, 32, 8, 256, True),
        (256, 32, 32, 256, True),
        (256, 32, 32, 256, False),
        # Use-case configs: GQA 16/4
        (512, 16, 4, 256, False),
        (1024, 16, 4, 256, False),
        (2048, 16, 4, 256, False),
        (4096, 16, 4, 256, False),
        (8192, 16, 4, 256, False),
    ]

    all_pass = True
    for seq_len, nhead_q, nhead_k, head_dim, causal in test_configs:
        torch.manual_seed(42)
        q = torch.randn(seq_len, nhead_q, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(seq_len, nhead_k, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(seq_len, nhead_k, head_dim, dtype=torch.half, device="cuda")

        sm_scale = 1.0 / (head_dim**0.5)
        gqa_ratio = nhead_q // nhead_k

        # Ground truth via PyTorch SDPA (FP32 for accuracy)
        q_sdpa = q.float().permute(1, 0, 2).unsqueeze(0)  # [1, nhead_q, N, D]
        k_sdpa = k.float().permute(1, 0, 2)  # [nhead_k, N, D]
        k_sdpa = k_sdpa.repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)
        v_sdpa = v.float().permute(1, 0, 2)
        v_sdpa = v_sdpa.repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)
        ref = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=causal, scale=sm_scale
        )
        ref = ref.squeeze(0).permute(1, 0, 2).half()  # [N, nhead_q, D]

        out = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=causal, backend="fa3_cdna3"
        )

        diff = (out.float() - ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()

        causal_str = "causal" if causal else "nc"
        status = "PASS" if max_err < 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  [{status}]  seq={seq_len:>5d}  h={nhead_q}/{nhead_k}  d={head_dim}  "
            f"{causal_str:>6s}  max_err={max_err:.6f}  mean_err={mean_err:.6f}"
        )

    print("=" * 72)
    if all_pass:
        print("  ALL CORRECTNESS TESTS PASSED")
    else:
        print("  SOME TESTS FAILED -- kernel produces incorrect output")
    print("=" * 72 + "\n")
    return all_pass


if __name__ == "__main__":
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv

    if not _skip_gpu:
        _run_correctness_tests()

    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(),
        num_warmup=5,
        dry_run_ms=200,
        repeat_ms=2000,
        counters=_counters,
        kernel_name_regex="fa3_cdna3_prefill_kernel|SinglePrefillWithKVCacheKernel",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
