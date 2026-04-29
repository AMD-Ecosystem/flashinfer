# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import torch

import flashinfer


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: Optional[bool] = False,
    pos_encoding_mode: Optional[str] = "NONE",
    logits_soft_cap: Optional[float] = None,
    return_lse: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Naive PyTorch implementation of attention for reference.

    Args:
        q: query tensor, shape: [qo_len, num_qo_heads, head_dim]
        k: key tensor, shape: [kv_len, num_kv_heads, head_dim], NHD layout
        v: value tensor, shape: [kv_len, num_kv_heads, head_dim], NHD layout
    Optional Args:
        causal: whether to apply causal masking
        pos_encoding_mode: the position encoding mode to use
        logits_soft_cap: if not None, applies soft cap to logits: soft_cap * tanh(logits / soft_cap)
        return_lse: whether to return the log sum exp value of the attention logits
    Returns:
        A tuple of two tensors: (o, lse), where:
        - o: output tensor, shape: [qo_len, num_qo_heads, head_dim]
        - lse: log sum exp value of the attention logits, shape: [qo_len, num_qo_heads], or None if return_lse is False
    """

    # The current validation only supports simple cases without RoPE/ALiBi
    if pos_encoding_mode != "NONE":
        raise ValueError(
            f"Only pos_encoding_mode == NONE is supported for this validation, got {pos_encoding_mode}"
        )

    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape

    sm_scale = 1.0 / math.sqrt(head_dim)  # softmax scale

    # Handle grouped query attention (GQA)
    group_size = num_qo_heads // num_kv_heads

    # Expand k and v to match q's head dimension if using GQA
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]
        v = v.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]

    # Transpose for batch matrix multiply: [num_heads, seq_len, head_dim]
    q_t = q.transpose(0, 1)  # [num_qo_heads, qo_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]

    # Compute attention scores: [num_qo_heads, qo_len, kv_len]
    # When soft cap is used: compute raw scores WITHOUT sm_scale
    # When soft cap is NOT used: apply sm_scale directly
    if logits_soft_cap is not None:
        scores = torch.matmul(q_t, k_t.transpose(1, 2))
        scores = logits_soft_cap * torch.tanh(scores * sm_scale / logits_soft_cap)
    else:
        scores = torch.matmul(q_t, k_t.transpose(1, 2)) * sm_scale

    # Apply causal mask if needed (AFTER soft cap)
    if causal:
        mask = torch.tril(
            torch.ones((qo_len, kv_len), device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Compute LSE on the final scores (after soft cap is applied) and before softmax
    lse = None
    if return_lse:
        lse = torch.logsumexp(scores, dim=-1)  # [num_qo_heads, qo_len]
        lse = lse / math.log(2)  # to match FlashInfer implementation
        lse = lse.transpose(0, 1)  # [qo_len, num_qo_heads]

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values: [num_qo_heads, qo_len, head_dim]
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out, lse


def single_prefill_with_kv_cache_example(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    return_lse: bool,
    backend: str = "auto",
):
    """
    Run single_prefill_with_kv_cache and verify the output against a naive PyTorch reference implementation.

    This function creates random Q, K, V tensors and compares the output
    of flashinfer's single_prefill_with_kv_cache against a naive PyTorch reference implementation.

    Set ``backend`` (for example ``"fa3_cdna3"`` on ROCm) to exercise a non-default attention backend.
    """
    print("\nRunning configuration:")
    print(f"  qo_len={qo_len}")
    print(f"  kv_len={kv_len}")
    print(f"  num_qo_heads={num_qo_heads}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  causal={causal}")
    print(f"  kv_layout={kv_layout}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"  return_lse={return_lse}")
    print(f"  backend={backend}")

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    if kv_layout == "HND":
        k = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
        v_ref = v.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
    else:  # NHD layout
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        k_ref = k
        v_ref = v

    # Call flashinfer API
    logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend=backend,
        )
        print(f"  FlashInfer output shape: {o.shape}, LSE shape: {lse.shape}")
        # Compute reference in FP32 for better accuracy
        o_ref, lse_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            return_lse=True,
        )
        # Convert reference back to match FlashInfer dtype
        o_ref = o_ref.to(o.dtype)
        lse_ref = lse_ref.to(lse.dtype)
        print(f"  Reference output shape: {o_ref.shape}, LSE shape: {lse_ref.shape}")
        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output and LSE match reference implementation")
        except AssertionError:
            print("  ✗ FAIL: Output or LSE does not match reference implementation")
            max_diff_o = (o - o_ref).abs().max().item()
            max_diff_lse = (lse - lse_ref).abs().max().item()
            print(f"    Max absolute difference for output: {max_diff_o}")
            print(f"    Max absolute difference for LSE: {max_diff_lse}")
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend=backend,
        )
        print(f"  FlashInfer output shape: {o.shape}")

        # Compute reference in FP32 for better accuracy
        o_ref, _ = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            return_lse=False,
        )
        # Convert reference back to match FlashInfer dtype
        o_ref = o_ref.to(o.dtype)
        print(f"  Reference output shape: {o_ref.shape}")

        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output matches reference implementation")
        except AssertionError:
            print("  ✗ FAIL: Output does not match reference implementation")
            max_diff_o = (o - o_ref).abs().max().item()
            print(f"    Max absolute difference for output: {max_diff_o}")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Single Prefill Example")
    print("=" * 60)

    # Self-attention with logits soft cap
    single_prefill_with_kv_cache_example(
        128, 128, 1, 1, 64, False, "NHD", "NONE", 8.0, False
    )
    # Self-attention without logits soft cap
    single_prefill_with_kv_cache_example(
        128, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, False
    )
    # Multi-head attention (MHA)
    single_prefill_with_kv_cache_example(
        128, 128, 4, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # Grouped query attention (GQA)
    single_prefill_with_kv_cache_example(
        128, 128, 8, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # GQA with qo_len < kv_len (typical prefill)
    single_prefill_with_kv_cache_example(
        15, 127, 32, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # GQA with LSE enabled
    single_prefill_with_kv_cache_example(
        15, 127, 8, 4, 64, False, "NHD", "NONE", 0.0, True
    )
    # GQA with soft cap and LSE enabled
    single_prefill_with_kv_cache_example(
        15, 127, 8, 4, 64, False, "NHD", "NONE", 8.0, True
    )

    # Test case specifically for threadblock_sync_mdo_states validation
    # This config triggers CTA_TILE_Q=16, NUM_WARPS_KV=4, calling threadblock_sync_mdo_states
    print("\n" + "=" * 60)
    print("Testing threadblock_sync_mdo_states (CTA_TILE_Q=16, NUM_WARPS_KV=4)")
    print("=" * 60)
    single_prefill_with_kv_cache_example(
        16, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, False
    )
    single_prefill_with_kv_cache_example(
        16, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, True
    )

    # ROCm FA3-CDNA3 backend — configs align with benchmarks/rocm_benchmarks/bench_fa3_cdna3.py.
    print("\n" + "=" * 60)
    print(
        "FA3-CDNA3 backend (q_len=256, head_dim=256, GQA 16/4; "
        "kv_lens 512..8192; causal attention)"
    )
    print("=" * 60)

    # Causal Attention only
    single_prefill_with_kv_cache_example(
        256, 512, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
    single_prefill_with_kv_cache_example(
        256, 1024, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
    single_prefill_with_kv_cache_example(
        256, 2048, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
    single_prefill_with_kv_cache_example(
        256, 3072, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
    single_prefill_with_kv_cache_example(
        256, 4096, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
    single_prefill_with_kv_cache_example(
        256, 8192, 16, 4, 256, True, "NHD", "NONE", 0.0, False, backend="fa3_cdna3"
    )
