"""Correctness tests for the FA3-CDNA3 single-prefill kernel.

Compares ``flashinfer.single_prefill_with_kv_cache(..., backend="fa3_cdna3")``
against PyTorch SDPA computed in FP32. Migrated from the embedded sweep in
``benchmarks/rocm_benchmarks/bench_fa3_cdna3.py``.

Skipped when no HIP/ROCm GPU is available.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import flashinfer

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is not None),
    reason="FA3-CDNA3 requires an AMD ROCm GPU",
)

# (qo_len, kv_len, nhead_q, nhead_k, head_dim, causal)
_CONFIGS = [
    # Square configs (regression tests)
    (64, 64, 1, 1, 256, False),
    (64, 64, 1, 1, 256, True),
    (128, 128, 32, 8, 256, False),
    (128, 128, 32, 8, 256, True),
    # Chunked prefill: q_len < kv_len (non-causal)
    (256, 512, 16, 4, 256, False),
    (256, 1024, 16, 4, 256, False),
    (256, 2048, 16, 4, 256, False),
    (256, 3072, 16, 4, 256, False),
    (256, 4096, 16, 4, 256, False),
    (256, 8192, 16, 4, 256, False),
    # Chunked prefill: q_len < kv_len (causal)
    (256, 512, 16, 4, 256, True),
    (256, 1024, 16, 4, 256, True),
    (256, 2048, 16, 4, 256, True),
    (256, 3072, 16, 4, 256, True),
    (256, 4096, 16, 4, 256, True),
    (256, 8192, 16, 4, 256, True),
    # Edge cases
    (32, 256, 16, 4, 256, False),
    (32, 256, 16, 4, 256, True),
    (128, 512, 16, 4, 256, False),
    (128, 512, 16, 4, 256, True),
]


def _ids(cfg):
    qo, kv, hq, hk, d, c = cfg
    return f"q{qo}_kv{kv}_h{hq}-{hk}_d{d}_{'causal' if c else 'nc'}"


@pytest.mark.parametrize("cfg", _CONFIGS, ids=_ids)
@torch.inference_mode()
def test_fa3_cdna3_matches_sdpa(cfg):
    qo_len, kv_len, nhead_q, nhead_k, head_dim, causal = cfg

    torch.manual_seed(42)
    q = torch.randn(qo_len, nhead_q, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(kv_len, nhead_k, head_dim, dtype=torch.half, device="cuda")

    sm_scale = 1.0 / (head_dim**0.5)
    gqa_ratio = nhead_q // nhead_k

    # Ground truth via PyTorch SDPA in FP32
    q_sdpa = q.float().permute(1, 0, 2).unsqueeze(0)  # [1, nhead_q, qo_len, D]
    k_sdpa = k.float().permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)
    v_sdpa = v.float().permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0).unsqueeze(0)

    if causal and qo_len != kv_len:
        # FlashInfer uses right-aligned causal: row i sees cols 0..(i + kv_len - qo_len).
        # SDPA's is_causal flag is left-aligned for the math backend, so build the
        # right-aligned mask explicitly.
        causal_offset = kv_len - qo_len
        mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device="cuda").tril(
            diagonal=causal_offset
        )
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

    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 0.01, f"max_err={max_err:.6f} exceeds 0.01 tolerance"
