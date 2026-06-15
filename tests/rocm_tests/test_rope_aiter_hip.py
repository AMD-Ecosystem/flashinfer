# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the AITER rope backend exposed via
# flashinfer.apply_rope_with_cos_sin_cache(backend="aiter") and its inplace variant.
#
# Note on tolerances: AITER's rope_cached kernel consumes the cos/sin tables in
# the query dtype, whereas the native JIT kernel rotates in float32. For bfloat16
# this raises max abs error to ~5e-2 (native ~3e-2); fp16 stays at ~7e-3. The
# tolerances below reflect AITER's actual precision.

import pytest
import torch

import flashinfer
from flashinfer.aiter_utils import is_aiter_supported
from tests.test_helpers.rope_reference import RotaryEmbedding

pytestmark = pytest.mark.skipif(
    not is_aiter_supported(torch.device("cuda:0")),
    reason="AITER backend requires gfx942/gfx950",
)


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_size, rotary_dim, num_q_heads, num_kv_heads",
    [
        (64, 64, 8, 8),
        (128, 128, 8, 2),
        (128, 64, 8, 2),  # partial rotary (rotary_dim < head_size)
        (256, 128, 4, 2),
    ],
)
def test_rope_cos_sin_cache_aiter_vs_ref(
    is_neox_style, dtype, head_size, rotary_dim, num_q_heads, num_kv_heads
):
    torch.manual_seed(0x4011)
    device = torch.device("cuda:0")
    batch_size, seq_len = 4, 33

    rope = RotaryEmbedding(
        head_size, rotary_dim, 4096, 10000, is_neox_style, dtype, device
    )
    cos_sin_cache = rope.cos_sin_cache  # float32

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )

    query_ref, key_ref = rope.forward_native(pos_ids, query.clone(), key.clone())
    query_aiter, key_aiter = flashinfer.apply_rope_with_cos_sin_cache(
        pos_ids,
        query.clone(),
        key.clone(),
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
        backend="aiter",
    )

    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(
        query_aiter.float(), query_ref.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(key_aiter.float(), key_ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_cos_sin_cache_aiter_inplace(is_neox_style, dtype):
    """Inplace AITER backend matches its non-inplace counterpart and mutates inputs."""
    torch.manual_seed(0x4012)
    device = torch.device("cuda:0")
    head_size, rotary_dim = 128, 64
    batch_size, seq_len, num_q_heads, num_kv_heads = 4, 32, 8, 4

    rope = RotaryEmbedding(
        head_size, rotary_dim, 4096, 10000, is_neox_style, dtype, device
    )
    cos_sin_cache = rope.cos_sin_cache

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )

    query_out, key_out = flashinfer.apply_rope_with_cos_sin_cache(
        pos_ids,
        query.clone(),
        key.clone(),
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
        backend="aiter",
    )

    query_inplace = query.clone()
    key_inplace = key.clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        pos_ids,
        query_inplace,
        key_inplace,
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
        backend="aiter",
    )

    # inplace must mutate the inputs
    assert not torch.equal(query_inplace, query)
    # and must agree with the non-inplace result
    torch.testing.assert_close(query_inplace, query_out, rtol=0, atol=0)
    torch.testing.assert_close(key_inplace, key_out, rtol=0, atol=0)


def test_rope_auto_backend_selection():
    """auto picks AITER for fp16 + large-nnz (both inplace and out-of-place, since
    the _impl path needs no Q/K copy), where it is both faster (~1.3-2.8x) and
    precise enough (fp16 err ~7e-3); bf16 and small nnz stay native."""
    from flashinfer.rope import _AITER_ROPE_MIN_TOKENS, _auto_select_rope_backend

    device = torch.device("cuda:0")
    big = _AITER_ROPE_MIN_TOKENS
    small = _AITER_ROPE_MIN_TOKENS - 1

    # fp16 + nnz >= threshold routes to AITER (selection is shape/dtype-based and
    # backend dispatch is shared by both the inplace and out-of-place wrappers).
    q_fp16_big = torch.randn(big, 128, dtype=torch.float16, device=device)
    k_fp16_big = torch.randn(big, 128, dtype=torch.float16, device=device)
    assert _auto_select_rope_backend(q_fp16_big, k_fp16_big) == "aiter"

    # bf16 always native (precision: ~5e-2 vs native ~3e-2).
    q_bf16_big = torch.randn(big, 128, dtype=torch.bfloat16, device=device)
    k_bf16_big = torch.randn(big, 128, dtype=torch.bfloat16, device=device)
    assert _auto_select_rope_backend(q_bf16_big, k_bf16_big) == "native"

    # Below the token threshold, native's lower launch overhead wins.
    q_fp16_small = torch.randn(small, 128, dtype=torch.float16, device=device)
    k_fp16_small = torch.randn(small, 128, dtype=torch.float16, device=device)
    assert _auto_select_rope_backend(q_fp16_small, k_fp16_small) == "native"

    # Mixed q/k dtype falls back to native rather than raising: AITER can't rotate
    # both with one cos/sin table, but auto must never raise.
    assert _auto_select_rope_backend(q_fp16_big, k_bf16_big) == "native"


def test_rope_unknown_backend_raises():
    device = torch.device("cuda:0")
    cos_sin_cache = torch.randn(64, 64, dtype=torch.float32, device=device)
    pos_ids = torch.arange(8, device=device)
    query = torch.randn(8, 8 * 128, dtype=torch.float16, device=device)
    key = torch.randn(8, 8 * 128, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="Unknown backend"):
        flashinfer.apply_rope_with_cos_sin_cache(
            pos_ids, query, key, 128, cos_sin_cache, backend="nonsense"
        )


def test_rope_aiter_mixed_dtype_raises():
    """AITER rotates Q/K with one cos/sin table, so mismatched dtypes must error
    clearly rather than crash inside the kernel."""
    device = torch.device("cuda:0")
    cos_sin_cache = torch.randn(64, 64, dtype=torch.float32, device=device)
    pos_ids = torch.arange(8, device=device)
    query = torch.randn(8, 8 * 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(8, 8 * 128, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="share a dtype"):
        flashinfer.apply_rope_with_cos_sin_cache(
            pos_ids, query, key, 128, cos_sin_cache, backend="aiter"
        )


def test_rope_aiter_odd_rotary_dim_raises():
    """An odd cos_sin_cache last dim cannot split into equal cos||sin halves."""
    device = torch.device("cuda:0")
    cos_sin_cache = torch.randn(64, 63, dtype=torch.float32, device=device)
    pos_ids = torch.arange(8, device=device)
    query = torch.randn(8, 8 * 128, dtype=torch.float16, device=device)
    key = torch.randn(8, 8 * 128, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="even"):
        flashinfer.apply_rope_with_cos_sin_cache(
            pos_ids, query, key, 128, cos_sin_cache, backend="aiter"
        )


def test_rope_aiter_rotary_dim_exceeds_head_size_raises():
    """rotary_dim derived from cos_sin_cache must fit within head_size."""
    device = torch.device("cuda:0")
    head_size = 64
    cos_sin_cache = torch.randn(64, 128, dtype=torch.float32, device=device)
    pos_ids = torch.arange(8, device=device)
    query = torch.randn(8, 8 * head_size, dtype=torch.float16, device=device)
    key = torch.randn(8, 8 * head_size, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="exceeds head_size"):
        flashinfer.apply_rope_with_cos_sin_cache(
            pos_ids, query, key, head_size, cos_sin_cache, backend="aiter"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_aiter_noncontiguous_positions(dtype):
    """A strided positions tensor must be normalized before reaching the AITER
    kernel, whose C assert (stride(1) == 1) would otherwise abort the process.
    The result must match the contiguous-positions result."""
    device = torch.device("cuda:0")
    head_size, rotary_dim = 128, 64
    batch_size, seq_len, num_q_heads, num_kv_heads = 2, 16, 8, 2

    rope = RotaryEmbedding(head_size, rotary_dim, 4096, 10000, True, dtype, device)
    cos_sin_cache = rope.cos_sin_cache
    nnz = batch_size * seq_len

    # Build a non-contiguous (stride-2) positions tensor.
    pos_strided = torch.arange(2 * seq_len, device=device, dtype=torch.int64).repeat(
        batch_size
    )[::2]
    assert not pos_strided.is_contiguous()

    query = torch.randn(nnz, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(nnz, num_kv_heads * head_size, dtype=dtype, device=device)

    q_strided, k_strided = flashinfer.apply_rope_with_cos_sin_cache(
        pos_strided,
        query.clone(),
        key.clone(),
        head_size,
        cos_sin_cache,
        backend="aiter",
    )
    q_contig, k_contig = flashinfer.apply_rope_with_cos_sin_cache(
        pos_strided.contiguous(),
        query.clone(),
        key.clone(),
        head_size,
        cos_sin_cache,
        backend="aiter",
    )
    torch.testing.assert_close(q_strided, q_contig, rtol=0, atol=0)
    torch.testing.assert_close(k_strided, k_contig, rtol=0, atol=0)


def test_rope_auto_falls_back_when_aiter_unimportable(monkeypatch):
    """On a supported arch with a missing/broken aiter install, auto must fall
    back to native rather than raise — _auto_select_rope_backend probes the
    import and returns 'native' on failure."""
    from flashinfer import rope
    from flashinfer.rope import _AITER_ROPE_MIN_TOKENS, _auto_select_rope_backend

    device = torch.device("cuda:0")
    n = _AITER_ROPE_MIN_TOKENS
    q = torch.randn(n, 128, dtype=torch.float16, device=device)
    k = torch.randn(n, 128, dtype=torch.float16, device=device)

    # Sanity: with aiter importable this shape selects aiter.
    assert _auto_select_rope_backend(q, k) == "aiter"

    def _boom():
        raise ImportError("simulated missing aiter")

    monkeypatch.setattr(rope, "_aiter_rope_ops", _boom)
    assert _auto_select_rope_backend(q, k) == "native"
