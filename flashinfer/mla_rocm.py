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

from .arch_caps import require_capability


@functools.cache
def _aiter_mla():
    import aiter.mla as _m

    return _m


def _combined_kv_view(
    ckv_cache: torch.Tensor, kpe_cache: torch.Tensor
) -> Optional[torch.Tensor]:
    """AITER's ``[num_pages, page_size, 1, ckv+kpe]`` buffer as a zero-copy view.

    AITER reads a single combined buffer, so :meth:`run` otherwise has to
    concatenate ``ckv_cache`` and ``kpe_cache`` on every call. That copy covers the
    *entire allocated page pool*, not just the live pages, so its cost tracks cache
    capacity rather than the working set.

    Callers that keep the two halves adjacent in one allocation can skip it
    entirely — which is what production already does (vLLM's
    ``kv_c_and_k_pe_cache``, SGLang's ``K_Buffer``, both sliced with
    ``torch.split``). Returns the view when ``ckv_cache`` and ``kpe_cache`` are
    exactly the two halves of one contiguous ``[num_pages, page_size, ckv+kpe]``
    buffer, and ``None`` otherwise.
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

    ``plan`` reads ``kv_indptr``/``kv_len_arr`` to derive last-page lengths and
    ``qo_indptr`` to derive ``max_seqlen_q``.  Doing those as separate copies costs
    a device→host sync each, and a sync is ~70 us on gfx942 regardless of payload —
    these tensors are only ``batch+1`` int32 elements, so the round-trip count is
    the whole cost.  Concatenating first makes it one.

    Batching only helps when there is a sync to save, so inputs that are already
    on the host are passed through untouched — concatenating them would add a real
    CPU allocation and copy for no benefit. Mixed-device inputs also fall back to
    per-tensor copies, since ``torch.cat`` rejects them.
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
        raise ImportError(
            "The 'aiter' package is required for MLA on ROCm. "
            "Install it via:\n"
            "  git clone --recursive https://github.com/ROCm/aiter.git\n"
            "  cd aiter && python3 setup.py develop"
        ) from exc


class BatchMLAPagedAttentionWrapper:
    r"""ROCm MLA paged attention wrapper backed by AITER.

    Mirrors the public API of :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper`
    for use on AMD gfx942/gfx950 GPUs.  Implements the Matrix Absorption variant
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
    backend : str
        Either ``"auto"`` (the default, resolves to ``"aiter"`` on ROCm)
        or ``"aiter"``. Any other value raises ``ValueError``.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "aiter"):
            raise ValueError(
                f"Only backend='aiter' (or 'auto', which resolves to "
                f"'aiter') is supported on ROCm; got {backend!r}."
            )
        backend = "aiter"
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
            Whether to apply causal masking (no-op for single-token decode).
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

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=True)
        self._kv_indptr = kv_indptr.to(self.device, non_blocking=True)
        self._kv_indices = kv_indices.to(self.device, non_blocking=True)
        qo_host, kv_indptr_host, kv_lens_host = _gather_plan_inputs_on_host(
            qo_indptr, kv_indptr, kv_len_arr
        )
        last_cpu = _kv_lens_to_last_page_len_cpu(
            kv_indptr_host, kv_lens_host, page_size
        )
        self._kv_last_page_len = last_cpu.to(self.device, non_blocking=True)
        self._sm_scale = sm_scale
        qo_lens = qo_host[1:] - qo_host[:-1]
        self._max_seqlen_q = int(qo_lens.max()) if qo_lens.numel() > 0 else 1

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        return_lse: bool = False,
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

        Returns
        -------
        out : torch.Tensor
            Attention output, shape ``[total_q, num_heads, head_dim_ckv]``.
        """
        if return_lse:
            raise NotImplementedError(
                "return_lse is not currently supported by the AITER MLA backend."
            )
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
        # An earlier version of this docstring told callers to allocate
        # [num_pages, page_size, 1, ckv+kpe] and pass 4-D slices. That shape never
        # worked -- the fallback below would unsqueeze it to 5-D, which AITER
        # rejects -- so say so plainly instead of warning that a genuinely adjacent
        # buffer is not adjacent.
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
            # Guarded explicitly rather than left to the "default" warning filter,
            # which dedups by (message, category, module, lineno) and so is
            # defeated by an "always" filter or by logging capture -- this sits in
            # a per-decode-step path, once per layer. Per wrapper rather than per
            # process so a second mis-allocated wrapper is still told.
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
