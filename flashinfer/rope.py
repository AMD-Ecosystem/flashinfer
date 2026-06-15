"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import weakref
from typing import Optional, Tuple

import torch

from .device_utils import IS_HIP
from .jit.rope import gen_rope_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_rope_module():
    return gen_rope_module().build_and_load()


if IS_HIP:

    @functools.cache
    def _aiter_rope_ops():
        import aiter as _aiter

        return _aiter

    # Token count above which AITER's cos/sin-cache rope beats the native JIT
    # kernel. Measured on gfx942 (fp16, q32/k8, hd128): AITER crosses over around
    # nnz~1024-1536 and reaches ~2.6-2.8x at large nnz, while native's lower
    # launch overhead wins below it. 2048 leaves headroom over launch jitter.
    _AITER_ROPE_MIN_TOKENS = 2048

    def _auto_select_rope_backend(query: torch.Tensor, key: torch.Tensor) -> str:
        # AITER's cos/sin-cache rope consumes the cos/sin tables in the query
        # dtype, whereas the native JIT kernel rotates in float32. For bf16 this
        # pushes max abs error to ~5e-2 (vs native ~3e-2), at the edge of the
        # rope test tolerance, so auto never picks AITER for bf16 — only fp16,
        # whose AITER error (~7e-3) stays comfortably inside tolerance.
        #
        # AITER wins on both the inplace and out-of-place paths once nnz is large
        # enough: the helper uses the ..._impl entry point with distinct in/out
        # tensors, so out-of-place needs no Q/K copy. Below the token threshold,
        # native's lower launch overhead wins.
        if query.dtype != torch.float16:
            return "native"
        if key.dtype != query.dtype:
            # AITER rotates Q and K with one cos/sin table in the query dtype and
            # rejects mismatched dtypes; native handles them. auto must not raise,
            # so fall back rather than route into the helper's dtype guard.
            return "native"
        if query.shape[0] < _AITER_ROPE_MIN_TOKENS:
            return "native"
        from .aiter_utils import is_aiter_supported

        if not is_aiter_supported(query.device):
            return "native"
        try:
            # Best-effort probe: a supported arch can still lack a usable aiter
            # (not installed, or its compiled extension fails to load). auto must
            # always fall back to native, never raise — matching norm/activation.
            _aiter_rope_ops()
        except Exception:
            return "native"
        return "aiter"

    # Memoize the AITER-format cos/sin tables. ``cos_sin_cache`` is a fixed
    # precomputed float32 table (typically a persistent module buffer) reused
    # across every forward pass, so converting the whole table to the query
    # dtype on each call is pure overhead — significant during decode, where
    # nnz is tiny but max_seq_len is large. Keyed by id() (tensors aren't
    # value-hashable) with a finalizer that evicts the entry when the cache is
    # GC'd, so the cached tables never outlive their source. A weakref to the
    # source is stored alongside the entry so a hit is only honored when it
    # still resolves to the *same* tensor — guarding against id() reuse if a
    # transient cache is freed and a new tensor lands on the same address.
    _aiter_cos_sin_tables: dict = {}

    def _aiter_rope_cos_sin(
        cos_sin_cache: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(cos_sin_cache)
        cached = _aiter_cos_sin_tables.get(key)
        if cached is not None and cached[0] == dtype and cached[1]() is cos_sin_cache:
            return cached[2], cached[3]
        half = cos_sin_cache.shape[-1] // 2
        cos = cos_sin_cache[:, :half].unsqueeze(1).unsqueeze(1).to(dtype).contiguous()
        sin = cos_sin_cache[:, half:].unsqueeze(1).unsqueeze(1).to(dtype).contiguous()
        if cached is None:
            weakref.finalize(cos_sin_cache, _aiter_cos_sin_tables.pop, key, None)
        _aiter_cos_sin_tables[key] = (dtype, weakref.ref(cos_sin_cache), cos, sin)
        return cos, sin

    def _apply_rope_cos_sin_cache_aiter(
        query: torch.Tensor,
        key: torch.Tensor,
        query_out: torch.Tensor,
        key_out: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        head_size: int,
        is_neox: bool,
    ) -> None:
        r"""Dispatch the cos/sin-cache rope to AITER's rope_cached_positions_2c kernel.

        FlashInfer stores ``cos_sin_cache`` as ``(max_seq_len, rotary_dim)`` float32
        with cosine in the first half and sine in the second half. AITER wants two
        separate ``(max_seq_len, 1, 1, rotary_dim // 2)`` tables in the query dtype
        with ``reuse_freqs_front_part=True``. Q/K are reshaped to AITER's
        ``(1, nnz, num_heads, head_dim)`` layout and only the leading ``rotary_dim``
        slice is rotated (matching ``nope_first=False``).

        Uses the ``..._impl`` entry point with distinct input/output tensors so the
        out-of-place case writes straight into ``query_out``/``key_out`` with no Q/K
        copy (measured ~1.3-2.8x faster than copy-then-rotate). The kernel only
        writes the rotated ``[:rotary_dim]`` slice, so for partial rotary
        (``rotary_dim < head_size``) the untouched nope tail is copied across when
        the output is a fresh tensor. When ``query_out``/``key_out`` alias the
        inputs (the inplace variant) this is ``_impl(x, y, x, y, ...)``.
        """
        from .aiter_utils import is_aiter_supported

        if not is_aiter_supported(query.device):
            raise ValueError(
                "AITER rope backend requires an AMD gfx942/gfx950 device; "
                "use backend='native' instead."
            )
        try:
            aiter_ops = _aiter_rope_ops()
        except Exception as e:
            raise ValueError(
                "backend='aiter' requires the aiter package, which failed to "
                f"import: {e}"
            ) from e
        if key.dtype != query.dtype:
            # AITER rotates Q and K with a single cos/sin table built in the
            # query dtype; the native path tolerates mixed dtypes by rotating
            # in float32, but AITER cannot.
            raise ValueError(
                "AITER rope backend requires query and key to share a dtype; "
                f"got query={query.dtype}, key={key.dtype}. Use backend='native'."
            )

        rotary_dim = cos_sin_cache.shape[-1]
        # cos_sin_cache stacks cos||sin on its last dim, so rotary_dim must be
        # even, and the rotated slice q[..., :rotary_dim] must fit in head_size.
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"cos_sin_cache last dim must be even (cos||sin); got {rotary_dim}."
            )
        if rotary_dim > head_size:
            raise ValueError(
                f"rotary_dim ({rotary_dim}) from cos_sin_cache exceeds head_size "
                f"({head_size})."
            )

        nnz = query.shape[0]
        cos, sin = _aiter_rope_cos_sin(cos_sin_cache, query.dtype)

        q_in = query.view(1, nnz, -1, head_size)
        k_in = key.view(1, nnz, -1, head_size)
        q_out = query_out.view(1, nnz, -1, head_size)
        k_out = key_out.view(1, nnz, -1, head_size)

        # The kernel only writes the rotated [:rotary_dim] slice. For partial
        # rotary into a fresh output, copy the untouched nope tail [rotary_dim:]
        # across first (skipped when output aliases input, and when rotary_dim
        # covers the full head_size).
        if rotary_dim < head_size:
            if query_out.data_ptr() != query.data_ptr():
                q_out[..., rotary_dim:].copy_(q_in[..., rotary_dim:])
            if key_out.data_ptr() != key.data_ptr():
                k_out[..., rotary_dim:].copy_(k_in[..., rotary_dim:])

        # AITER's HIP kernel asserts int64, contiguous positions of shape
        # (1, nnz) (stride(1) == 1) — a strided/non-int64 positions tensor
        # otherwise trips a C assert that aborts the process.
        pos = positions.to(torch.int64).contiguous().view(1, nnz)

        aiter_ops.rope_cached_positions_2c_fwd_impl(
            q_out[..., :rotary_dim],
            k_out[..., :rotary_dim],
            q_in[..., :rotary_dim],
            k_in[..., :rotary_dim],
            cos,
            sin,
            pos,
            0 if is_neox else 1,  # rotate_style: 0=NEOX, 1=GPT-J
            True,  # reuse_freqs_front_part (cos/sin are rotary_dim//2 sized)
            False,  # nope_first
        )


@register_custom_op("flashinfer::apply_rope", mutates_args=("q_rope", "k_rope"))
def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    get_rope_module().apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope")
def _fake_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass  # pragma: no cover


@register_custom_op("flashinfer::apply_llama31_rope", mutates_args=("q_rope", "k_rope"))
def _apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    get_rope_module().apply_llama31_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope")
def _fake_apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass  # pragma: no cover


@register_custom_op("flashinfer::apply_rope_pos_ids", mutates_args=("q_rope", "k_rope"))
def _apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    get_rope_module().apply_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids")
def _fake_apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass  # pragma: no cover


@register_custom_op(
    "flashinfer::rope_quantize",
    mutates_args=("q_rope_out", "k_rope_out", "q_nope_out", "k_nope_out"),
)
def _rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    r"""Custom operator that routes to the CUDA kernel implementation.

    Converts is_neox parameter to interleave format and dispatches to the underlying
    CUDA kernel via the JIT-compiled module.
    """
    get_rope_module().rope_quantize(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        cos_sin_cache,
        pos_ids,
        quant_scale_q,
        quant_scale_kv,
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize")
def _fake_rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass  # pragma: no cover


@register_custom_op(
    "flashinfer::rope_quantize_append_paged_kv_cache",
    mutates_args=(
        "q_rope_out",
        "q_nope_out",
        "k_cache",
        "v_cache",
        "ckv_cache",
        "kpe_cache",
    ),
)
def _rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    r"""Custom operator that routes to the CUDA kernel implementation.

    Fuses RoPE application, FP8 quantization, and paged KV cache append into a single kernel.

    Converts is_neox parameter to interleave format and dispatches to the underlying
    CUDA kernel via the JIT-compiled module.
    """
    get_rope_module().rope_quantize_append_paged_kv_cache(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        v_in,
        q_rope_out,
        q_nope_out,
        cos_sin_cache,
        pos_ids,
        k_cache,
        v_cache,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        kv_layout_code,
        page_size,
        quant_scale_q,
        quant_scale_kv,
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize_append_paged_kv_cache")
def _fake_rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass  # pragma: no cover


@register_custom_op(
    "flashinfer::apply_rope_pos_ids_cos_sin_cache", mutates_args=("q_rope", "k_rope")
)
def _apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    get_rope_module().apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids_cos_sin_cache")
def _fake_apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    pass  # pragma: no cover


@register_custom_op(
    "flashinfer::apply_llama31_rope_pos_ids", mutates_args=("q_rope", "k_rope")
)
def _apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    get_rope_module().apply_llama31_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope_pos_ids")
def _fake_apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass  # pragma: no cover


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope(
        q, k, q, k, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta
    )


def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    See Also
    --------
    apply_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope_pos_ids(
        q, k, q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )


def apply_llama31_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor) inplace. cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_llama31_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_llama31_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope(
        q,
        k,
        q,
        k,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor) inplace. cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    See Also
    --------
    apply_llama31_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q,
        k,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )
    return q_rope, k_rope


def apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(batch_size + 1)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope_pos_ids(
        q, k, q_rope, k_rope, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )
    return q_rope, k_rope


def apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor). cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_llama31_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_llama31_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


def apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor). cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``
    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_llama31_rope_pos_ids_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


def apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

    backend : str
        Kernel backend to use. ``"auto"`` (default) selects the best backend for
        the call: on ROCm (gfx942/gfx950) it picks AITER for fp16 inputs with at
        least ~2048 tokens (where the AITER kernel is measurably faster), and stays
        on ``"native"`` otherwise — for bf16 (precision), small token counts
        (launch overhead), and non-ROCm platforms.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's rope_cached kernel — ROCm (gfx942/gfx950) only;
        requires the ``aiter`` package. Precision is slightly lower than ``"native"``
        for bfloat16 (max abs error ~5e-2 vs ~3e-2) because AITER consumes the cos/sin
        tables in the query dtype rather than float32.

    Returns
    -------
    query_out : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key_out : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads * head_size)``.

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    if IS_HIP:
        _backend = (
            backend if backend != "auto" else _auto_select_rope_backend(query, key)
        )
        if _backend == "aiter":
            _apply_rope_cos_sin_cache_aiter(
                query=query,
                key=key,
                query_out=query_out,
                key_out=key_out,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                head_size=head_size,
                is_neox=is_neox,
            )
            return query_out, key_out
        if _backend != "native":
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
            )

    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query_out.view(query_out.shape[0], -1, head_size),
        k_rope=key_out.view(key_out.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )

    return query_out, key_out


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    backend: str = "auto",
) -> None:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.
    The result is inplace applied to the input tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

    backend : str
        Kernel backend to use. ``"auto"`` (default) selects the best backend for
        the call: on ROCm (gfx942/gfx950) it picks AITER for fp16 inputs with at
        least ~2048 tokens (where AITER's kernel is measurably faster), and stays
        on ``"native"`` otherwise — for bf16 (precision), small token counts
        (launch overhead), and non-ROCm platforms.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's rope_cached kernel — ROCm (gfx942/gfx950) only;
        requires the ``aiter`` package. Precision is slightly lower than ``"native"``
        for bfloat16 (max abs error ~5e-2 vs ~3e-2) because AITER consumes the cos/sin
        tables in the query dtype rather than float32.

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if IS_HIP:
        _backend = (
            backend if backend != "auto" else _auto_select_rope_backend(query, key)
        )
        if _backend == "aiter":
            _apply_rope_cos_sin_cache_aiter(
                query=query,
                key=key,
                query_out=query,
                key_out=key,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                head_size=head_size,
                is_neox=is_neox,
            )
            return
        if _backend != "native":
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
            )

    # pass q_rope and k_rope as q and k to perform inplace operation
    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query.view(query.shape[0], -1, head_size),
        k_rope=key.view(key.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )


def mla_rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return rope_quantize_fp8(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        is_neox,
        quantize_dtype,
        quant_scale_q,
        quant_scale_kv,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        enable_pdl,
    )


def rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply RoPE (Rotary Positional Embeddings) and quantize to FP8 format.

    This function takes pre-split query/key tensors (rotary and non-rotary dimensions separated),
    applies RoPE to the rotary dimension tensors, and quantizes both rotary and non-rotary
    tensors to FP8 format. Supports MLA, GQA, and MHA architectures.

    Parameters
    ----------
    q_rope : torch.Tensor
        Query tensor (rotary dimensions), shape: ``(nnz, num_qo_heads, rope_dim)``.
        Must be float16 or bfloat16.
    k_rope : torch.Tensor
        Key tensor (rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, rope_dim)``.
        For MLA: ``(nnz, rope_dim)``. Must be float16 or bfloat16.
    q_nope : Optional[torch.Tensor]
        Query tensor (non-rotary dimensions), shape: ``(nnz, num_qo_heads, no_rope_dim)``.
        If ``None``, treated as zero-dim: a size-0 tensor will be created internally.
    k_nope : Optional[torch.Tensor]
        Key tensor (non-rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, no_rope_dim)``.
        For MLA: ``(nnz, no_rope_dim)``. If ``None``, treated as zero-dim and created internally.
    cos_sin_cache : torch.Tensor
        Precomputed cosine and sine values, shape: ``(max_seq_len, rope_dim)``.
        First half contains cosine values, second half contains sine values. Must be float32.
    pos_ids : torch.Tensor
        Position indices for each token, shape: ``(nnz,)``.
    is_neox : bool
        RoPE layout style. If ``True`` (default), use non-interleaved layout (first/second half).
        If ``False``, use interleaved layout (even/odd dimensions).
    quantize_dtype : Optional[torch.dtype]
        Target quantization dtype. If ``None``, inferred from output tensors or defaults to
        ``torch.float8_e4m3fn``. Must be ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
    quant_scale_q : float
        Quantization scaling factor for query tensors, default: ``1.0``.
    quant_scale_kv : float
        Quantization scaling factor for key tensors, default: ``1.0``.
    q_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (rotary). If ``None``, allocated automatically.
    k_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized key (rotary). If ``None``, allocated automatically.
    q_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (non-rotary). If ``None``, allocated automatically.
    k_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized key (non-rotary). If ``None``, allocated automatically.
    enable_pdl : bool
        Whether to enable PDL (Programmatic Dependent Launch). Default: ``False``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Quantized tensors: (q_rope_out, k_rope_out, q_nope_out, k_nope_out).
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # Allow None for nope tensors and normalize to size-0 tensors with correct shapes
    nnz = q_rope.shape[0]
    num_qo_heads = q_rope.shape[1]
    is_mla = k_rope.ndim == 2
    num_kv_heads = 1 if is_mla else k_rope.shape[1]
    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_qo_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        if is_mla:
            k_nope = torch.empty(nnz, 0, dtype=k_rope.dtype, device=k_rope.device)
        else:
            k_nope = torch.empty(
                nnz, num_kv_heads, 0, dtype=k_rope.dtype, device=k_rope.device
            )

    # Infer quantize_dtype from output tensors or default to float8_e4m3fn
    if quantize_dtype is None:
        for out in (q_rope_out, k_rope_out, q_nope_out, k_nope_out):
            if out is not None:
                quantize_dtype = out.dtype
                break
        else:
            quantize_dtype = torch.float8_e4m3fn

    # Allocate output tensors if not provided
    q_rope_out = (
        q_rope_out
        if q_rope_out is not None
        else torch.empty_like(q_rope, dtype=quantize_dtype)
    )
    k_rope_out = (
        k_rope_out
        if k_rope_out is not None
        else torch.empty_like(k_rope, dtype=quantize_dtype)
    )
    q_nope_out = (
        q_nope_out
        if q_nope_out is not None
        else torch.empty_like(q_nope, dtype=quantize_dtype)
    )
    k_nope_out = (
        k_nope_out
        if k_nope_out is not None
        else torch.empty_like(k_nope, dtype=quantize_dtype)
    )

    # The ROCm C++ binding uses int32_t for IdType; coerce here so callers
    # that pass the default int64 torch.arange() tensor still work correctly.
    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)

    _rope_quantize(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        quant_scale_q,
        quant_scale_kv,
        not is_neox,  # interleave
        enable_pdl,
    )

    return q_rope_out, k_rope_out, q_nope_out, k_nope_out


def rope_quantize_fp8_append_paged_kv_cache(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
    kv_layout: str = "NHD",
    q_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply RoPE (Rotary Positional Embeddings), quantize to FP8, and append K/V to paged cache.

    This fused function applies RoPE to query/key (Q/K) rotary dimension tensors, quantizes all Q/K tensors
    (and V for GQA/MHA) to FP8 format, and directly appends the quantized K/V to a paged KV cache.
    It returns quantized Q tensors for use in attention computation. Supports MLA, GQA, and MHA
    architectures with automatic detection based on input tensor shapes.

    Parameters
    ----------
    q_rope : torch.Tensor
        Query tensor (rotary dimensions), shape: ``(nnz, num_qo_heads, rope_dim)``.
        Must be float16 or bfloat16.
    k_rope : torch.Tensor
        Key tensor (rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, rope_dim)``.
        For MLA: ``(nnz, rope_dim)``. Must be float16 or bfloat16.
    q_nope : torch.Tensor
        Query tensor (non-rotary dimensions), shape: ``(nnz, num_qo_heads, no_rope_dim)``.
        Must be float16 or bfloat16.
    k_nope : torch.Tensor
        Key tensor (non-rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, no_rope_dim)``.
        For MLA: ``(nnz, no_rope_dim)``. Must be float16 or bfloat16.
    v : Optional[torch.Tensor]
        Value tensor for GQA/MHA: ``(nnz, num_kv_heads, head_dim)``. Must be float16 or bfloat16.
        For MLA: pass ``None`` (MLA does not use separate V; K non-RoPE acts as compressed KV).
    cos_sin_cache : torch.Tensor
        Precomputed cosine and sine values, shape: ``(max_seq_len, rope_dim)``.
        First half contains cosine values, second half contains sine values. Must be float32.
    pos_ids : torch.Tensor
        Position indices for each token, shape: ``(nnz,)``.
    paged_kv_cache : Tuple[torch.Tensor, torch.Tensor]
        For MLA: ``(ckv_cache, kpe_cache)`` where:
            - ckv_cache: ``(max_pages, page_size, no_rope_dim)`` in FP8
            - kpe_cache: ``(max_pages, page_size, rope_dim)`` in FP8
        For GQA/MHA: ``(k_cache, v_cache)`` where:
            - k_cache: ``(max_pages, page_size, num_kv_heads, head_dim)`` or
              ``(max_pages, num_kv_heads, page_size, head_dim)`` depending on layout, in FP8
            - v_cache: same shape as k_cache, in FP8
    kv_indices : torch.Tensor
        Page indices mapping, shape: ``(total_pages,)``. Typically ``torch.arange(total_pages)``.
    kv_indptr : torch.Tensor
        Page indptr array for each request, shape: ``(batch_size + 1,)``.
        ``kv_indptr[i]`` is the starting page index for request ``i``.
    batch_indices : torch.Tensor
        Batch index for each token, shape: ``(nnz,)``. Maps each token to its request.
    positions : torch.Tensor
        Position within each request's sequence for each token, shape: ``(nnz,)``.
    is_neox : bool
        RoPE layout style. If ``True`` (default), use non-interleaved layout (first/second half).
        If ``False``, use interleaved layout (even/odd dimensions).
    quantize_dtype : Optional[torch.dtype]
        Target quantization dtype. If ``None``, inferred from output tensors or defaults to
        ``torch.float8_e4m3fn``. Must be ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
    quant_scale_q : float
        Quantization scaling factor for query tensors, default: ``1.0``.
    quant_scale_kv : float
        Quantization scaling factor for key/value tensors, default: ``1.0``.
    page_size : int
        Number of entries per page in the paged cache, default: ``16``.
    kv_layout : str
        Cache memory layout for GQA/MHA. Options: ``"NHD"`` (page, seq, head, dim) or
        ``"HND"`` (page, head, seq, dim). Default: ``"NHD"``. Ignored for MLA.
    q_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (rotary). If ``None``, allocated automatically.
    q_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (non-rotary). If ``None``, allocated automatically.
    enable_pdl : bool
        Whether to enable PDL (Programmatic Dependent Launch). Default: ``False``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Quantized query tensors: (q_rope_out, q_nope_out).
        K/V are written directly to the paged cache and not returned.

    Notes
    -----
    - Architecture detection: Automatically distinguishes MLA (2D K tensors) from GQA/MHA (3D K tensors).
    - MLA writes K-RoPE to ``kpe_cache`` and K-noRoPE to ``ckv_cache``; V is not used.
    - GQA/MHA writes full K (RoPE+noRoPE) to ``k_cache`` and V to ``v_cache``.
    - The ``batch_indices`` and ``positions`` tensors are typically obtained from
      ``flashinfer.get_batch_indices_positions()``.
    - Cache tensors must already be allocated in the target FP8 dtype.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # Detect architecture
    is_mla = k_rope.ndim == 2

    # Allow None for nope tensors and normalize to size-0 tensors with correct shapes
    nnz = q_rope.shape[0]
    num_qo_heads = q_rope.shape[1]
    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_qo_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        if is_mla:
            k_nope = torch.empty(nnz, 0, dtype=k_rope.dtype, device=k_rope.device)
        else:
            num_kv_heads = k_rope.shape[1]
            k_nope = torch.empty(
                nnz, num_kv_heads, 0, dtype=k_rope.dtype, device=k_rope.device
            )

    # Infer quantize_dtype from output tensors or default
    if quantize_dtype is None:
        if q_rope_out is not None:
            quantize_dtype = q_rope_out.dtype
        elif q_nope_out is not None:
            quantize_dtype = q_nope_out.dtype
        else:
            quantize_dtype = torch.float8_e4m3fn

    # Allocate Q output tensors if not provided
    if q_rope_out is None:
        q_rope_out = torch.empty_like(q_rope, dtype=quantize_dtype)
    if q_nope_out is None:
        q_nope_out = torch.empty_like(q_nope, dtype=quantize_dtype)

    # Handle MLA normalization and V (create empty dummy tensor, not used)
    if is_mla:
        # Normalize MLA K tensors to 3D (nnz, 1, dim) so C++ binding can always assume 3D
        if k_rope.ndim == 2:
            k_rope = k_rope.unsqueeze(1)
        if k_nope.ndim == 2:
            k_nope = k_nope.unsqueeze(1)
        if v is None:
            v = torch.empty(0, dtype=q_rope.dtype, device=q_rope.device)
        else:
            raise ValueError("MLA should not have V input (pass None)")

    # Unpack and validate cache tensors
    if len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be a tuple of 2 tensors")

    cache_0, cache_1 = paged_kv_cache

    if is_mla:
        # MLA: Expect (ckv_cache, kpe_cache)
        ckv_cache = cache_0
        kpe_cache = cache_1
        if ckv_cache.dtype != quantize_dtype or kpe_cache.dtype != quantize_dtype:
            raise ValueError(
                f"MLA cache dtype mismatch: expected {quantize_dtype}, "
                f"got ckv={ckv_cache.dtype}, kpe={kpe_cache.dtype}"
            )
        if ckv_cache.ndim != 3 or kpe_cache.ndim != 3:
            raise ValueError(
                f"MLA cache must be 3D: (max_pages, page_size, dim), "
                f"got ckv={ckv_cache.ndim}D, kpe={kpe_cache.ndim}D"
            )
        # Create dummy tensors for GQA/MHA cache (not used)
        k_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
        v_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
    else:
        # GQA/MHA: Expect (k_cache, v_cache)
        k_cache = cache_0
        v_cache = cache_1
        # Validate V input is provided for GQA/MHA
        if v is None:
            raise ValueError(
                "GQA/MHA expects a V tensor, but got None. "
                "Only MLA uses None for V (compressed KV representation)."
            )
        if v.dtype != q_rope.dtype:
            raise ValueError(
                f"v dtype must match q_rope/k_rope ({q_rope.dtype}); got {v.dtype}"
            )
        if k_cache.dtype != quantize_dtype or v_cache.dtype != quantize_dtype:
            raise ValueError(
                f"GQA/MHA cache dtype mismatch: expected {quantize_dtype}, "
                f"got k={k_cache.dtype}, v={v_cache.dtype}"
            )
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                f"GQA/MHA cache must be 4D, got k={k_cache.ndim}D, v={v_cache.ndim}D"
            )
        # Create dummy tensors for MLA cache (not used)
        ckv_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
        kpe_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)

    # Import TensorLayout enum
    from .utils import TensorLayout

    kv_layout_code = TensorLayout[kv_layout].value

    # The ROCm C++ binding uses int32_t for IdType; coerce here so callers
    # that pass the default int64 torch.arange() tensor still work correctly.
    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)

    # Call custom op
    _rope_quantize_fp8_append_paged_kv_cache(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        v,
        q_rope_out,
        q_nope_out,
        cos_sin_cache,
        pos_ids,
        k_cache,
        v_cache,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        kv_layout_code,
        page_size,
        quant_scale_q,
        quant_scale_kv,
        not is_neox,  # interleave
        enable_pdl,
    )

    return q_rope_out, q_nope_out
