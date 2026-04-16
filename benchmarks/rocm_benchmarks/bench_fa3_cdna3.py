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

FA3-CDNA3 chunked-prefill benchmark for AMD MI300X.

Benchmarks the FA3-CDNA3 kernel (head_dim=256, chunked prefill q_len != kv_len)
against:
  - FlashInfer FA2 HIP path (existing baseline)
  - AITER flash_attn_varlen_func (if available)

Alibaba use case: q_len=256, kv_len=512..8192, GQA 16/4, d=256.

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
#   (qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal)
#
# Alibaba chunked-prefill: q_len=256, kv_len varies, GQA 16/4, d=256.
# ---------------------------------------------------------------------------
_CONFIGS = [
    # Alibaba chunked-prefill configs (non-causal)
    (256, 512, 16, 4, 256, False),
    (256, 1024, 16, 4, 256, False),
    (256, 2048, 16, 4, 256, False),
    (256, 4096, 16, 4, 256, False),
    (256, 8192, 16, 4, 256, False),
    # Alibaba chunked-prefill configs (causal)
    (256, 512, 16, 4, 256, True),
    (256, 1024, 16, 4, 256, True),
    (256, 2048, 16, 4, 256, True),
    (256, 4096, 16, 4, 256, True),
    (256, 8192, 16, 4, 256, True),
]

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(qo_len: int, kv_len: int, nhead_q: int, head_dim: int, causal: bool) -> int:
    """Theoretical FLOPs for QK^T + PV (ignoring softmax overhead).

    Right-aligned causal: query i sees keys 0..(i + kv_len - qo_len).
    The masked triangle has qo_len*(qo_len-1)/2 entries.
    """
    if causal:
        effective_pairs = qo_len * kv_len - qo_len * (qo_len - 1) // 2
        return 4 * effective_pairs * nhead_q * head_dim
    return 4 * qo_len * kv_len * nhead_q * head_dim


def _bytes(qo_len: int, kv_len: int, nhead_q: int, nhead_k: int, head_dim: int) -> int:
    """Theoretical bytes (cold-cache lower bound): Q, K, V, O in FP16."""
    q_bytes = 2 * qo_len * nhead_q * head_dim
    kv_bytes = 2 * kv_len * nhead_k * head_dim * 2  # K + V
    o_bytes = 2 * qo_len * nhead_q * head_dim
    return q_bytes + kv_bytes + o_bytes


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    configs = []

    for qo_len, kv_len, nhead_q, nhead_k, head_dim, causal in _CONFIGS:
        q = torch.randn(qo_len, nhead_q, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")

        flops = _flops(qo_len, kv_len, nhead_q, head_dim, causal)
        theo_bytes = _bytes(qo_len, kv_len, nhead_q, nhead_k, head_dim)
        causal_str = "causal" if causal else "nc"
        label_str = (
            f"FA3-CDNA3  q={qo_len:>4d}  kv={kv_len:>5d}  h={nhead_q}/{nhead_k}  "
            f"d={head_dim}  {causal_str}"
        )
        label_fa2 = (
            f"FA2        q={qo_len:>4d}  kv={kv_len:>5d}  h={nhead_q}/{nhead_k}  "
            f"d={head_dim}  {causal_str}"
        )

        # ---- FA3-CDNA3 (our kernel) ----
        def fa3_fn(q=q, k=k, v=v, c=causal):
            return flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=c, backend="fa3_cdna3"
            )

        configs.append(
            KernelConfig(
                name=f"fa3_cdna3_q{qo_len}_kv{kv_len}_{causal_str}_d{head_dim}",
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
                name=f"fa2_q{qo_len}_kv{kv_len}_{causal_str}_d{head_dim}",
                run_fn=torch.inference_mode()(fa2_fn),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=label_fa2,
            )
        )

        # ---- AITER comparison ----
        if _AITER_AVAILABLE:
            cu_seqlens_q = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
            cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")

            def aiter_fn(
                q=q,
                k=k,
                v=v,
                csq=cu_seqlens_q,
                csk=cu_seqlens_k,
                Nq=qo_len,
                Nk=kv_len,
                c=causal,
            ):
                q_flat = q.view(-1, q.shape[-2], q.shape[-1])
                k_flat = k.view(-1, k.shape[-2], k.shape[-1])
                v_flat = v.view(-1, v.shape[-2], v.shape[-1])
                return aiter_fa_varlen(
                    q_flat,
                    k_flat,
                    v_flat,
                    csq,
                    csk,
                    max_seqlen_q=Nq,
                    max_seqlen_k=Nk,
                    causal=c,
                )

            label_aiter = (
                f"AITER      q={qo_len:>4d}  kv={kv_len:>5d}  h={nhead_q}/{nhead_k}  "
                f"d={head_dim}  {causal_str}"
            )
            configs.append(
                KernelConfig(
                    name=f"aiter_q{qo_len}_kv{kv_len}_{causal_str}_d{head_dim}",
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

    # (qo_len, kv_len, nhead_q, nhead_k, head_dim, causal)
    test_configs = [
        # Square configs (regression tests)
        (64, 64, 1, 1, 256, False),
        (64, 64, 1, 1, 256, True),
        (128, 128, 32, 8, 256, False),
        (128, 128, 32, 8, 256, True),
        # Chunked prefill: q_len < kv_len (non-causal)
        (256, 512, 16, 4, 256, False),
        (256, 1024, 16, 4, 256, False),
        (256, 2048, 16, 4, 256, False),
        (256, 4096, 16, 4, 256, False),
        (256, 8192, 16, 4, 256, False),
        # Chunked prefill: q_len < kv_len (causal)
        (256, 512, 16, 4, 256, True),
        (256, 1024, 16, 4, 256, True),
        (256, 2048, 16, 4, 256, True),
        (256, 4096, 16, 4, 256, True),
        (256, 8192, 16, 4, 256, True),
        # Edge cases
        (32, 256, 16, 4, 256, False),
        (32, 256, 16, 4, 256, True),
        (128, 512, 16, 4, 256, False),
        (128, 512, 16, 4, 256, True),
    ]

    all_pass = True
    for qo_len, kv_len, nhead_q, nhead_k, head_dim, causal in test_configs:
        torch.manual_seed(42)
        q = torch.randn(qo_len, nhead_q, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")

        sm_scale = 1.0 / (head_dim**0.5)
        gqa_ratio = nhead_q // nhead_k

        # Ground truth via PyTorch SDPA (FP32 for accuracy)
        q_sdpa = q.float().permute(1, 0, 2).unsqueeze(0)  # [1, nhead_q, qo_len, D]
        k_sdpa = k.float().permute(1, 0, 2)  # [nhead_k, kv_len, D]
        k_sdpa = k_sdpa.repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)
        v_sdpa = v.float().permute(1, 0, 2)
        v_sdpa = v_sdpa.repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)
        if causal and qo_len != kv_len:
            # FlashInfer uses right-aligned causal: row i sees cols 0..(i + kv_len - qo_len).
            # PyTorch SDPA is_causal uses left-aligned for the math backend, so build
            # the correct mask explicitly.
            causal_offset = kv_len - qo_len
            mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device="cuda")
            mask = mask.tril(diagonal=causal_offset)
            attn_mask = torch.zeros(qo_len, kv_len, dtype=torch.float32, device="cuda")
            attn_mask.masked_fill_(~mask, float("-inf"))
            ref = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                scale=sm_scale,
            )
        else:
            ref = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=causal, scale=sm_scale
            )
        ref = ref.squeeze(0).permute(1, 0, 2).half()  # [qo_len, nhead_q, D]

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
            f"  [{status}]  q={qo_len:>4d}  kv={kv_len:>5d}  h={nhead_q}/{nhead_k}  "
            f"d={head_dim}  {causal_str:>6s}  max_err={max_err:.6f}  mean_err={mean_err:.6f}"
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
        kernel_name_regex="fa3_cdna3_prefill_kernel_impl|SinglePrefillWithKVCacheKernel",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
