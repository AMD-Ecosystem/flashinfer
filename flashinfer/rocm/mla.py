# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ROCm MLA (Multi-head Latent Attention) wrapper backed by AITER.
# Mirrors the public API of flashinfer.mla.BatchMLAPagedAttentionWrapper
# and routes through aiter.mla.mla_decode_fwd / mla_prefill_fwd.

import functools
import warnings
from typing import Optional, Tuple, Union

import torch

from .api_compat import reject_cuda_only
from .arch_caps import require_capability


@functools.cache
def _aiter_mla():
    import aiter.mla as _m

    return _m


def _combined_kv_view(
    ckv_cache: torch.Tensor, kpe_cache: torch.Tensor
) -> Optional[torch.Tensor]:
    """AITER's ``[num_pages, page_size, 1, ckv+kpe]`` buffer as a zero-copy view.

    Returns the view when ``ckv_cache`` and ``kpe_cache`` are exactly the two
    halves of one contiguous ``[num_pages, page_size, ckv+kpe]`` buffer, and
    ``None`` otherwise, in which case :meth:`run` must concatenate them.
    """
    if ckv_cache.dtype != kpe_cache.dtype:
        return None
    if ckv_cache.dim() != 3 or kpe_cache.dim() != 3:
        return None
    if ckv_cache.shape[:2] != kpe_cache.shape[:2]:
        return None
    if ckv_cache.device != kpe_cache.device:
        return None
    if ckv_cache.untyped_storage().data_ptr() != kpe_cache.untyped_storage().data_ptr():
        return None

    num_pages, page_size, head_dim_ckv = ckv_cache.shape
    width = head_dim_ckv + kpe_cache.shape[2]
    # kpe must begin exactly where ckv ends, and both must be column slices of one
    # contiguous [num_pages, page_size, width] buffer. Anything else (padding
    # between pages, a transposed layout, separate allocations) fails these and
    # falls back to the copy.
    if kpe_cache.storage_offset() != ckv_cache.storage_offset() + head_dim_ckv:
        return None
    expected_stride = (page_size * width, width, 1)
    if ckv_cache.stride() != expected_stride or kpe_cache.stride() != expected_stride:
        return None

    return ckv_cache.as_strided(
        (num_pages, page_size, 1, width),
        (page_size * width, width, width, 1),
        ckv_cache.storage_offset(),
    )


def _gather_plan_inputs_on_host(
    qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_len_arr: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch the three index tensors :meth:`plan` needs host-side in one transfer.

    Cost here is round-trip count, not payload. Host-resident and mixed-device
    inputs fall back to per-tensor copies: there is no sync to save, and
    ``torch.cat`` rejects the mixed case.
    """
    tensors = (qo_indptr, kv_indptr, kv_len_arr)
    if any(t.device.type == "cpu" for t in tensors) or not (
        qo_indptr.device == kv_indptr.device == kv_len_arr.device
    ):
        return tuple(t.to("cpu") for t in tensors)  # type: ignore[return-value]
    host = torch.cat(list(tensors)).to("cpu")
    n_qo, n_kv = qo_indptr.numel(), kv_indptr.numel()
    return host[:n_qo], host[n_qo : n_qo + n_kv], host[n_qo + n_kv :]


def _kv_lens_to_last_page_len_cpu(
    kv_indptr_cpu: torch.Tensor, kv_lens_cpu: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Convert FlashInfer MLA total KV lengths → AITER last-page fill counts (int32).

    CUDA ``flashinfer.mla`` planners take per-batch **total** KV lengths
    (:attr:`kv_len_arr`). AITER's ``mla_decode_fwd`` / ``mla_prefill_fwd`` expect per-batch
    **filled token count on the final page**, in ``[1, page_size]`` (same notion as
    :attr:`paged_kv_last_page_len` elsewhere in FlashInfer).
    """

    kv_indptr_cpu = kv_indptr_cpu.to(torch.device("cpu")).to(torch.int64)
    kv_lens_cpu = kv_lens_cpu.to(torch.device("cpu")).to(torch.int64)

    npages_batch = kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]
    # Reject ambiguous / degenerate bookkeeping (consistent with paging utilities).
    if bool((npages_batch < 1).any().item()):
        idx = torch.nonzero(npages_batch < 1, as_tuple=False)[0].item()
        raise ValueError(
            f"kv_indptr assigns no pages at batch idx {idx} "
            f"(kv_indptr[{idx}:{idx + 2}] = "
            f"{tuple(int(x) for x in kv_indptr_cpu[idx : idx + 2].tolist())})."
        )

    lp = kv_lens_cpu - (npages_batch - 1) * int(page_size)
    invalid = (lp < 1) | (lp > int(page_size))
    if bool(invalid.any().item()):
        b = torch.nonzero(invalid, as_tuple=False)[0].item()
        n = int(npages_batch[b].item())
        L = int(kv_lens_cpu[b].item())
        lpl = int(lp[b].item())
        raise ValueError(
            f"kv_len_arr[{b}]={L} is inconsistent with paging: num_pages={n}, "
            f"page_size={page_size} ⇒ last-page length must be "
            f"kv_len − (num_pages−1)·page_size ∈ [1, {page_size}], got {lpl}."
        )

    return lp.to(dtype=torch.int32)


def _require_aiter_mla(device: torch.device) -> None:
    """MLA is AITER-only -- there is no HIP fallback -- so the arch gate here is
    the only thing between a user and a kernel that cannot run.

    ``ArchCapabilityError`` subclasses ``RuntimeError``, preserving the previous
    contract.
    """
    require_capability(device, "mla", "aiter")
    try:
        _aiter_mla()
    except ImportError as exc:
        from .aiter_utils import AITER_MIN_VERSION

        raise ImportError(
            "The 'aiter' package is required for MLA on ROCm. Install a wheel >= "
            f"{AITER_MIN_VERSION}; see docs/rocm/backends.md for the index and the "
            "pinned version. A source build tracks master, whose C ABI does not "
            "match the structs vendored here."
        ) from exc


class BatchMLAPagedAttentionWrapper:
    r"""ROCm MLA paged attention wrapper backed by AITER.

    Mirrors the public API of :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper`
    for use on AMD gfx942/gfx950 GPUs, with one divergence: the CUDA wrapper
    honours ``causal=False`` for multi-token queries, and AITER's prefill kernel
    cannot, so :meth:`plan` rejects it rather than silently masking.
    Implements the Matrix Absorption variant
    (absorbed W_UQ·W_UK and W_UV·W_O) where the KV-cache stores compressed-KV
    (``ckv``) and rope-key (``kpe``) tensors concatenated into a single buffer.

    KV-cache layout expected by AITER:
        ``kv_buffer[num_pages, page_size, 1, head_dim_ckv + head_dim_kpe]``

    This wrapper accepts the FlashInfer-style separate ``(ckv_cache, kpe_cache)``
    pair.  **Allocate them as one buffer** — AITER reads a single combined tensor,
    so two separate allocations force a concatenation on every :meth:`run`, and
    that copy spans the whole allocated page pool rather than the live pages::

        cache = torch.empty(
            num_pages, page_size, head_dim_ckv + head_dim_kpe, dtype=..., device=...
        )
        ckv_cache, kpe_cache = cache.split([head_dim_ckv, head_dim_kpe], dim=-1)

    :meth:`run` detects that layout and passes a view, copying nothing.  This is
    how vLLM (``kv_c_and_k_pe_cache``) and SGLang (``K_Buffer``) already store MLA
    caches, so those callers get the fast path unchanged.  Separate allocations
    still work but warn once.

    Parameters
    ----------
    float_workspace_buffer : torch.Tensor
        Reserved workspace.  Size is ignored; only the device is used.
    use_cuda_graph, qo_indptr, kv_indptr, kv_indices, kv_len_arr
        Upstream's CUDA-graph capture buffers, declared in upstream's positional
        order so a CUDA caller binds correctly. Each raises when set: the AITER
        MLA path cannot pre-bind capture-time pointers.
    backend : str
        Either ``"auto"`` (the default, resolves to ``"aiter"`` on ROCm)
        or ``"aiter"``. Any other value raises ``ValueError``.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "aiter"):
            raise ValueError(
                f"Only backend='aiter' (or 'auto', which resolves to "
                f"'aiter') is supported on ROCm; got {backend!r}."
            )
        backend = "aiter"
        # Declared so a CUDA-graph caller's arguments bind where it intends;
        # the AITER MLA path has no capture support to hand them to.
        if use_cuda_graph:
            raise NotImplementedError(
                "use_cuda_graph=True is not supported by the AITER MLA backend "
                "on ROCm; the wrapper cannot pre-bind capture-time buffers."
            )
        reject_cuda_only("qo_indptr", qo_indptr, None)
        reject_cuda_only("kv_indptr", kv_indptr, None)
        reject_cuda_only("kv_indices", kv_indices, None)
        reject_cuda_only("kv_len_arr", kv_len_arr, None)

        self.device = float_workspace_buffer.device
        _require_aiter_mla(self.device)

        self._qo_indptr: Optional[torch.Tensor] = None
        self._kv_indptr: Optional[torch.Tensor] = None
        self._kv_indices: Optional[torch.Tensor] = None
        self._kv_last_page_len: Optional[torch.Tensor] = None
        self._sm_scale: float = 1.0
        self._max_seqlen_q: int = 1
        # run() warns at most once per wrapper about a non-adjacent KV cache.
        self._warned_separate_kv: bool = False

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan MLA attention.

        Parameters
        ----------
        qo_indptr : torch.IntTensor
            Query/output indptr, shape ``[batch_size + 1]``.
            For decode, content is ``[0, 1, …, batch_size]``.
        kv_indptr : torch.IntTensor
            Paged KV indptr, shape ``[batch_size + 1]``.
        kv_indices : torch.IntTensor
            Page indices, shape ``[kv_indptr[-1]]``.
        kv_len_arr : torch.IntTensor
            Per-batch **total** KV sequence lengths (logical token count past ``kv_indptr``
            pages), shape ``[batch_size]``. This matches CUDA
            ``flashinfer.mla.BatchMLAPagedAttentionWrapper.plan`` (**not** ``1..page_size``
            tail counts). Values must satisfy::

                kv_len_arr[i]
                    == (kv_indptr[i+1]-kv_indptr[i]-1) * page_size + kv_last_page_len[i]

            with ``1 <= kv_last_page_len[i] <= page_size``. Converted internally for AITER.
        num_heads : int
            Number of query/output heads.
        head_dim_ckv : int
            Compressed-KV head dimension (512 for DeepSeek V2/V3).
        head_dim_kpe : int
            Rope-key head dimension (64 for DeepSeek V2/V3).
        page_size : int
            Page size of the paged KV-cache.
        causal : bool
            Whether to apply causal masking. No-op for single-token decode.
            Multi-token queries are always masked causally by AITER, so
            ``causal=False`` raises rather than silently masking.
        sm_scale : float
            Softmax scale (typically ``1 / sqrt(head_dim_ckv + head_dim_kpe)``).
        q_data_type : torch.dtype
            Query dtype; must be ``torch.float16`` or ``torch.bfloat16``.
        kv_data_type : torch.dtype
            KV dtype; must match ``q_data_type``.
        use_profiler : bool
            Ignored (AITER does not expose a per-kernel profiler through this API).
        """
        if use_profiler:
            raise ValueError(
                "use_profiler=True is not supported with the AITER MLA backend."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"AITER MLA requires q_data_type in {{fp16, bf16}}; got {q_data_type}."
            )
        if q_data_type != kv_data_type:
            raise ValueError(
                f"AITER MLA requires q_data_type == kv_data_type; "
                f"got {q_data_type} vs {kv_data_type}."
            )
        for t, name in [
            (qo_indptr, "qo_indptr"),
            (kv_indptr, "kv_indptr"),
            (kv_indices, "kv_indices"),
            (kv_len_arr, "kv_len_arr"),
        ]:
            if t.dtype != torch.int32:
                raise ValueError(
                    f"Expected {name}.dtype == torch.int32, got {t.dtype}."
                )

        batch = int(kv_indptr.numel()) - 1
        if int(kv_len_arr.numel()) != batch:
            raise ValueError(
                f"Expected kv_len_arr.shape[0]==batch_size ({batch}); "
                f"got {tuple(kv_len_arr.shape)} for kv_indptr with length {kv_indptr.numel()}."
            )

        qo_host, kv_indptr_host, kv_lens_host = _gather_plan_inputs_on_host(
            qo_indptr, kv_indptr, kv_len_arr
        )
        qo_lens = qo_host[1:] - qo_host[:-1]
        max_seqlen_q = int(qo_lens.max()) if qo_lens.numel() > 0 else 1

        # AITER's mla_prefill_fwd takes no causal flag, so causal=False cannot be
        # honoured for multi-token queries. Single-token masks nothing either way.
        if max_seqlen_q > 1 and not causal:
            raise ValueError(
                "AITER MLA applies causal masking unconditionally for "
                f"multi-token queries (max_seqlen_q={max_seqlen_q}); "
                "causal=False cannot be honoured. Pass causal=True, or use "
                "single-token queries where the distinction does not arise."
            )
        last_cpu = _kv_lens_to_last_page_len_cpu(
            kv_indptr_host, kv_lens_host, page_size
        )

        # Every validation above this line runs before any state is written, so a
        # rejected plan() leaves a previously-planned wrapper untouched.
        self._qo_indptr = qo_indptr.to(self.device, non_blocking=True)
        self._kv_indptr = kv_indptr.to(self.device, non_blocking=True)
        self._kv_indices = kv_indices.to(self.device, non_blocking=True)
        self._kv_last_page_len = last_cpu.to(self.device, non_blocking=True)
        self._sm_scale = sm_scale
        self._max_seqlen_q = max_seqlen_q

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run MLA attention.

        Parameters
        ----------
        q_nope : torch.Tensor
            Query without rope, shape ``[total_q, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            Rope part of query, shape ``[total_q, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            Compressed-KV cache (without rope), shape
            ``[num_pages, page_size, head_dim_ckv]``.
        kpe_cache : torch.Tensor
            Rope-key cache, shape ``[num_pages, page_size, head_dim_kpe]``.
            When ``ckv_cache`` and ``kpe_cache`` are adjacent halves of one
            ``[num_pages, page_size, head_dim_ckv + head_dim_kpe]`` allocation
            (see the class docstring) they are passed to AITER as a view;
            otherwise they are concatenated on every call, which warns once.
        out : Optional[torch.Tensor]
            Pre-allocated output, shape ``[total_q, num_heads, head_dim_ckv]``.
        return_lse : bool
            Not supported; raises ``NotImplementedError`` if ``True``.
        lse, profiler_buffer, kv_len, page_table, return_lse_base_on_e, o_scale, ckv_scale, ckv_scale_arr, kpe_scale
            Accepted in upstream's positional order so a CUDA caller binds
            correctly; each raises when set. ROCm produces no LSE, and plan()
            owns the page table and sequence lengths.

        Returns
        -------
        out : torch.Tensor
            Attention output, shape ``[total_q, num_heads, head_dim_ckv]``.
        """
        if return_lse:
            raise NotImplementedError(
                "return_lse is not currently supported by the AITER MLA backend."
            )
        # No LSE is produced, so a caller-supplied buffer cannot be filled.
        reject_cuda_only("lse", lse, None)
        reject_cuda_only("return_lse_base_on_e", return_lse_base_on_e, False)
        reject_cuda_only("profiler_buffer", profiler_buffer, None)
        # plan() owns the page table and lengths on ROCm.
        reject_cuda_only("kv_len", kv_len, None)
        reject_cuda_only("page_table", page_table, None)
        reject_cuda_only("o_scale", o_scale, None, neutral=1.0)
        reject_cuda_only("ckv_scale", ckv_scale, None, neutral=1.0)
        reject_cuda_only("ckv_scale_arr", ckv_scale_arr, None)
        reject_cuda_only("kpe_scale", kpe_scale, None, neutral=1.0)
        if self._qo_indptr is None:
            raise RuntimeError("plan() must be called before run().")

        total_q, num_heads, head_dim_ckv = q_nope.shape
        if out is None:
            out = torch.empty(
                (total_q, num_heads, head_dim_ckv),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )

        q = torch.cat([q_nope, q_pe], dim=-1)
        # 4-D input cannot work: the fallback below would unsqueeze it to 5-D,
        # which AITER rejects. Raise rather than warn about false non-adjacency.
        if ckv_cache.dim() != 3 or kpe_cache.dim() != 3:
            raise ValueError(
                f"ckv_cache and kpe_cache must be 3-D "
                f"[num_pages, page_size, head_dim]; got {ckv_cache.dim()}-D and "
                f"{kpe_cache.dim()}-D. If you allocated a combined "
                f"[num_pages, page_size, 1, head_dim_ckv + head_dim_kpe] buffer, "
                f"drop the size-1 axis first: "
                f"cache.squeeze(2).split([head_dim_ckv, head_dim_kpe], dim=-1)."
            )
        kv_buffer = _combined_kv_view(ckv_cache, kpe_cache)
        if kv_buffer is None:
            # Explicit guard: warnings' own dedup is defeated by an "always"
            # filter or logging capture, and this sits in a per-decode-step path.
            # Per wrapper, so a second mis-allocated wrapper is still told.
            if not self._warned_separate_kv:
                self._warned_separate_kv = True
                warnings.warn(
                    "MLA: ckv_cache and kpe_cache are not adjacent halves of a single "
                    "allocation, so run() must concatenate them on every call. That "
                    "copy covers the whole allocated page pool, not just the live "
                    "pages, so it scales with cache capacity. Allocate one "
                    "[num_pages, page_size, head_dim_ckv + head_dim_kpe] buffer and "
                    "pass torch.split(buf, [head_dim_ckv, head_dim_kpe], dim=-1) to "
                    "run() for the zero-copy path.",
                    UserWarning,
                    stacklevel=2,
                )
            kv_buffer = torch.cat(
                [ckv_cache.unsqueeze(2), kpe_cache.unsqueeze(2)], dim=-1
            )

        if self._max_seqlen_q == 1:
            _aiter_mla().mla_decode_fwd(
                q,
                kv_buffer,
                out,
                self._qo_indptr,
                self._kv_indptr,
                self._kv_indices,
                self._kv_last_page_len,
                self._max_seqlen_q,
                sm_scale=self._sm_scale,
            )
        else:
            _aiter_mla().mla_prefill_fwd(
                q,
                kv_buffer,
                out,
                self._qo_indptr,
                self._kv_indptr,
                self._kv_indices,
                self._kv_last_page_len,
                self._max_seqlen_q,
                sm_scale=self._sm_scale,
            )

        return out
