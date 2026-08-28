# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.core import logger
from flashinfer.aiter_utils import is_aiter_supported
from flashinfer.prefill_rocm import (
    _aiter_native_page_sizes,
    _aiter_native_paging_available,
    _aiter_ops_importable,
)
import logging

logger.setLevel(logging.ERROR)


def _skip_if_prefill_gated(device):
    """Skip when the capability table gates AITER batch prefill on this toolchain.

    A test that asks for ``backend="aiter"`` explicitly gets an
    ``ArchCapabilityError`` out of the wrapper constructor once the gate applies
    -- currently gfx950 on ROCm 7.2.x, where the causal kernel is miscompiled.
    That is the gate doing its job, so a test comparing AITER output against a
    reference has nothing left to prove and should stand down.

    Use this for tests that assert *numbers*. Tests that only assert plumbing can
    set ``FLASHINFER_ARCH_ALLOW_KNOWN_BAD=1`` instead and keep their coverage.
    """
    from flashinfer.arch_caps import capability_available, capability_reason

    if not capability_available(device, "batch_prefill", "aiter"):
        pytest.skip(capability_reason(device, "batch_prefill", "aiter"))


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
            ],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
@pytest.mark.parametrize("backend", ["fa2", "aiter"])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    if backend == "aiter" and (
        not is_aiter_supported(torch.device("cuda:0")) or not _aiter_ops_importable()
    ):
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    if backend == "aiter":
        _skip_if_prefill_gated(torch.device("cuda:0"))

    if backend == "aiter" and (causal or kv_layout != "NHD"):
        pytest.skip("Not testing for aiter backend with causal or kv_layout != NHD")

    max_num_batched_tokens = 4096

    if batch_size * qo_len > max_num_batched_tokens:
        pytest.skip(
            f"batch_size * qo_len ({batch_size * qo_len}) exceeds max_num_batched_tokens ({max_num_batched_tokens}). You may see OOM errors."
        )

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend=backend
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        wrapper.run(q, kv_data, out=o_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
            backend=backend,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )

        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="fa2",
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
@pytest.mark.parametrize("backend", ["fa2", "aiter"])
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    if backend == "aiter" and (
        not is_aiter_supported(torch.device("cuda:0")) or not _aiter_ops_importable()
    ):
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    if backend == "aiter":
        _skip_if_prefill_gated(torch.device("cuda:0"))

    if backend == "aiter" and (causal or kv_layout != "NHD"):
        pytest.skip("Not testing for aiter backend with causal")

    max_num_batched_tokens = 4096

    if batch_size * qo_len > max_num_batched_tokens:
        pytest.skip(
            f"batch_size * qo_len ({batch_size * qo_len}) exceeds max_num_batched_tokens ({max_num_batched_tokens}). You may see OOM errors."
        )

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend=backend
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
            backend=backend,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="fa2",
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("backend", ["fa2", "aiter"])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    backend,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    if backend == "aiter":
        if (
            not is_aiter_supported(torch.device("cuda:0"))
            or not _aiter_ops_importable()
        ):
            pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
        _skip_if_prefill_gated(torch.device("cuda:0"))
    kv_layout = "NHD"
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o = wrapper.run(q, k, v)

    for i in range(batch_size):
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            q[q_indptr[i] : q_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="fa2",
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("page_size", [1, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_batch_prefill_auto_selects_aiter(page_size, causal, return_lse):
    """backend='auto' should resolve to 'aiter' on gfx942/gfx950 for NHD fp16."""
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    # Use qo_len < kv_len (prefill-with-history) to exercise the meaningful causal case.
    # Both flat-gather and native-paged paths use mask_bottom_right matching FA2.
    batch_size, qo_len, kv_len = 4, 16, 128
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16
    )
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    kv_data = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_auto = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper_auto.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )

    # `auto` resolves to aiter only where the capability table allows it. On a
    # gated toolchain -- ROCm 7.2.x on gfx950 miscompiles this kernel -- the
    # correct behaviour is to steer away from AITER, so assert that instead and
    # skip the numeric comparison, which would otherwise be fa2 against fa2.
    from flashinfer.arch_caps import capability_available, capability_reason

    if not capability_available(device, "batch_prefill", "aiter"):
        assert wrapper_auto._backend != "aiter", (
            "capability table gates AITER batch prefill here, but auto still "
            f"chose it: {capability_reason(device, 'batch_prefill', 'aiter')}"
        )
        pytest.skip(capability_reason(device, "batch_prefill", "aiter"))

    assert wrapper_auto._backend == "aiter", (
        f"Expected backend='aiter' on gfx942/gfx950, got '{wrapper_auto._backend}'"
    )

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )

    if return_lse:
        o_auto, lse_auto = wrapper_auto.run(q, kv_data, return_lse=True)
        o_ref, lse_ref = wrapper_ref.run(q, kv_data, return_lse=True)
        torch.testing.assert_close(lse_auto, lse_ref, rtol=1e-2, atol=1e-2)
    else:
        o_auto = wrapper_auto.run(q, kv_data)
        o_ref = wrapper_ref.run(q, kv_data)

    torch.testing.assert_close(o_auto, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_batch_prefill_aiter_flat_gather_bf16(page_size, causal, return_lse):
    """Non-native page sizes take AITER's flat-gather route, which must bootstrap its .so.

    The mha_varlen_fwd_*.so variant is keyed on (dtype, causal, has_lse, has_logits_cap)
    — whether a logits soft cap is enabled, not its value — and
    AITER pre-ships only a subset, so an un-bootstrapped combo fails at run() with
    "AITER .so not found". bf16 is covered here because it is SGLang's serving dtype and
    is a distinct .so family from the fp16 cases above.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    _skip_if_prefill_gated(device)

    batch_size, qo_len, kv_len = 4, 16, 128
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128
    dtype = torch.bfloat16

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=dtype
    )
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Guard the branch under test: a native page size would take the paged kernel instead.
    assert wrapper._aiter_flat_gather_idx is not None, (
        f"page_size={page_size} did not take the AITER flat-gather path"
    )

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    if return_lse:
        o, lse = wrapper.run(q, kv_data, return_lse=True)
        o_ref, lse_ref = wrapper_ref.run(q, kv_data, return_lse=True)
        torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)
    else:
        o = wrapper.run(q, kv_data)
        o_ref = wrapper_ref.run(q, kv_data)

    # bf16 tolerance: AITER and FA2 accumulate in a different order, which shows up as
    # occasional 1-ULP (0.0078) differences at this magnitude.
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("page_size", [128, 256])
def test_batch_prefill_aiter_falls_back_when_native_paging_missing(
    page_size, monkeypatch
):
    """A native page size AITER cannot actually serve must degrade to flat-gather.

    _aiter_native_page_sizes() only reflects the validated amd-aiter release;
    serving stacks build AITER from source and such a build can reject a page size
    the predicate calls native ("no matching kernel found"). Simulate that by making
    the bootstrap raise, and require plan() to fall back rather than propagate.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    if page_size not in _aiter_native_page_sizes():
        pytest.skip(f"page_size={page_size} is not native on this amd-aiter build")
    # Ends in an assert_close against fa2 with causal=True, so it is a numerics
    # test despite being named for the fallback.
    _skip_if_prefill_gated(device)

    def _reject(*args, **kwargs):
        raise RuntimeError(
            f"invalid argument for batch_prefill: no matching kernel found. "
            f"page_size={page_size}, num_pages=1, dtype=bf16"
        )

    # The probe is cached, so clear it to keep this test independent of ordering.
    _aiter_native_paging_available.cache_clear()
    monkeypatch.setattr(
        flashinfer.prefill_rocm, "_aiter_bootstrap_batch_prefill", _reject
    )

    batch_size, qo_len, kv_len = 2, 16, 256
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128
    dtype = torch.bfloat16

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=dtype
    )
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    try:
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="aiter"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        # Degraded to flat-gather instead of raising out of plan().
        assert wrapper._aiter_flat_gather_idx is not None, (
            "plan() kept the native paged path despite the kernel being unavailable"
        )

        o = wrapper.run(q, kv_data)
    finally:
        _aiter_native_paging_available.cache_clear()

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # The fallback must be correct, not merely non-crashing.
    torch.testing.assert_close(o, wrapper_ref.run(q, kv_data), rtol=2e-2, atol=2e-2)


def _plan_softcap_flat_gather(wrapper, device, page_size, kv_len, dtype):
    """plan() a defect-shape call (causal, cap>0, head_dim=128, kv_len>=512)."""
    batch_size, qo_len = 1, 16
    num_qo_heads = num_kv_heads = 8
    num_pages = (kv_len + page_size - 1) // page_size
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(
        0, num_pages * batch_size, dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        128,
        page_size,
        causal=True,
        logits_soft_cap=30.0,
        q_data_type=dtype,
        kv_data_type=dtype,
    )


@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_softcap_guard_survives_a_native_page_size_degrading(backend, monkeypatch):
    """The soft-cap guard disarms on a native page size; the probe can undo that.

    softcap_kv_len is None whenever page_size looks native, because native paging
    uses mha_batch_prefill and is exact. When the runtime probe then degrades the
    call to flat-gather, it lands on the defective mha_varlen_fwd and has to be
    re-guarded against the real kv_len.
    """
    # Plumbing only -- no numbers compared, so the gfx950 causal gate is irrelevant.
    monkeypatch.setenv("FLASHINFER_ARCH_ALLOW_KNOWN_BAD", "1")
    # Strict turns the probe into a raise, so there would be nothing to demote.
    monkeypatch.delenv("FLASHINFER_AITER_STRICT", raising=False)
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    page_size = 128
    if page_size not in _aiter_native_page_sizes():
        pytest.skip(f"page_size={page_size} is not native on this amd-aiter build")

    def _reject(*args, **kwargs):
        raise RuntimeError(
            "invalid argument for batch_prefill: no matching kernel found. "
            f"page_size={page_size}, num_pages=1, dtype=bf16"
        )

    _aiter_native_paging_available.cache_clear()
    monkeypatch.setattr(
        flashinfer.prefill_rocm, "_aiter_bootstrap_batch_prefill", _reject
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    try:
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend=backend
        )
        if backend == "aiter":
            with pytest.raises(ValueError, match="logits_soft_cap"):
                _plan_softcap_flat_gather(
                    wrapper, device, page_size, 1024, torch.bfloat16
                )
        else:
            _plan_softcap_flat_gather(wrapper, device, page_size, 1024, torch.bfloat16)
            assert wrapper._backend == "fa2", (
                "auto stayed on the defective flat-gather kernel"
            )
            assert "logits_soft_cap" in (wrapper.backend_fallback_reason or "")
    finally:
        _aiter_native_paging_available.cache_clear()


def test_batch_prefill_aiter_strict_mode_raises(monkeypatch):
    """FLASHINFER_AITER_STRICT=1 must surface the AITER failure instead of degrading."""
    # Asserts plumbing and compares no numbers, so the ROCm 7.2 gfx950 causal
    # miscompile cannot affect the outcome. Opt past the capability gate rather
    # than skip and lose the coverage on the one architecture we can run.
    monkeypatch.setenv("FLASHINFER_ARCH_ALLOW_KNOWN_BAD", "1")
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    page_size = 128
    if page_size not in _aiter_native_page_sizes():
        pytest.skip(f"page_size={page_size} is not native on this amd-aiter build")

    def _reject(*args, **kwargs):
        raise RuntimeError("no matching kernel found. page_size=128")

    _aiter_native_paging_available.cache_clear()
    monkeypatch.setattr(
        flashinfer.prefill_rocm, "_aiter_bootstrap_batch_prefill", _reject
    )
    monkeypatch.setenv("FLASHINFER_AITER_STRICT", "1")

    batch_size, qo_len, kv_len = 1, 16, 128
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    try:
        with pytest.raises(RuntimeError, match="no matching kernel found"):
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )
    finally:
        _aiter_native_paging_available.cache_clear()


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True, "fa2"
    )
    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True, "fa2"
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "HND", "NONE", False, 0.0, False, True, "fa2"
    )

    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "NHD", "NONE", True, 0.0, False, True, "aiter"
    )
    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "NHD", "NONE", True, 0.0, False, True, "aiter"
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "NHD", "NONE", False, 0.0, False, True, "aiter"
    )

    test_batch_prefill_with_ragged_kv_cache(
        12, 54, 37, 8, 8, 128, True, "NONE", 0.0, False
    )


@pytest.mark.parametrize("kv_len", [512, 2048])
@pytest.mark.parametrize("qo_len", [37, 127])
def test_ragged_softcap_avoids_broken_aiter_kernel(kv_len, qo_len):
    """backend='auto' must stay numerically correct for causal soft-cap prefill.

    The ragged wrapper always dispatches through mha_varlen_fwd, which AITER
    miscomputes when logits_soft_cap > 0; without the fallback this returns
    plausible-looking values roughly 0.17 off an fp32 reference.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    head_dim, num_heads, soft_cap = 128, 4, 8.0
    torch.manual_seed(0)
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=device)

    qs, ks, vs = (t.transpose(0, 1).float() for t in (q, k, v))
    logits = soft_cap * torch.tanh(
        (qs @ ks.transpose(-1, -2)) * head_dim**-0.5 / soft_cap
    )
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=device).tril(
        diagonal=kv_len - qo_len
    )
    ref = (
        torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1) @ vs
    ).transpose(0, 1)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    indptr_q = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    indptr_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    wrapper.plan(
        indptr_q,
        indptr_kv,
        num_heads,
        num_heads,
        head_dim,
        causal=True,
        logits_soft_cap=soft_cap,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    torch.testing.assert_close(wrapper.run(q, k, v).float(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kv_len", [512, 2048])
@pytest.mark.parametrize("page_size", [128, 16])
def test_paged_softcap_is_numerically_correct(kv_len, page_size):
    """Causal soft-cap paged prefill must be correct whichever route plan() picks.

    page_size=128 is advertised as native but the probe can degrade it to
    flat-gather, and 16 never is; both land on the defective mha_varlen_fwd
    unless the guard redirects. Asserts the numbers, not the route.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")

    head_dim, num_heads, soft_cap, qo_len = 128, 4, 8.0, 37
    torch.manual_seed(0)
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=device)
    num_pages = (kv_len + page_size - 1) // page_size
    kv_data = torch.randn(
        num_pages, 2, page_size, num_heads, head_dim, dtype=torch.float16, device=device
    )
    # Flatten the pages back into the [kv_len, heads, dim] view the reference wants.
    k = kv_data[:, 0].reshape(-1, num_heads, head_dim)[:kv_len]
    v = kv_data[:, 1].reshape(-1, num_heads, head_dim)[:kv_len]

    qs, ks, vs = (t.transpose(0, 1).float() for t in (q, k, v))
    logits = soft_cap * torch.tanh(
        (qs @ ks.transpose(-1, -2)) * head_dim**-0.5 / soft_cap
    )
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=device).tril(
        diagonal=kv_len - qo_len
    )
    ref = (
        torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1) @ vs
    ).transpose(0, 1)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper.plan(
        torch.tensor([0, qo_len], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([(kv_len - 1) % page_size + 1], dtype=torch.int32, device=device),
        num_heads,
        num_heads,
        head_dim,
        page_size,
        causal=True,
        logits_soft_cap=soft_cap,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    torch.testing.assert_close(
        wrapper.run(q, kv_data).float(), ref, rtol=1e-3, atol=1e-3
    )


def test_paged_softcap_guard_tracks_the_paging_route():
    """The explicit-aiter soft-cap guard must key on the route, not the shape.

    Native page sizes dispatch to mha_batch_prefill, which is exact; only
    flat-gather carries the defect. A guard that ignores page_size rejects
    calls that backend='auto' happily serves.
    """
    device = torch.device("cuda:0")
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    kv_len, qo_len, num_heads, head_dim, soft_cap = 512, 37, 4, 128, 8.0
    # Only page sizes that divide kv_len: a partial trailing page would need a
    # kv_last_page_len this test does not model, and one larger than kv_len
    # floor-divides to zero pages.
    native = sorted(
        p for p in _aiter_native_page_sizes() if p <= kv_len and kv_len % p == 0
    )
    if not native:
        pytest.skip(f"no native AITER page size divides kv_len={kv_len}")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def plan(page_size):
        num_pages = kv_len // page_size
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="aiter"
        )
        wrapper.plan(
            torch.tensor([0, qo_len], dtype=torch.int32, device=device),
            torch.tensor([0, num_pages], dtype=torch.int32, device=device),
            torch.arange(num_pages, dtype=torch.int32, device=device),
            torch.tensor([page_size], dtype=torch.int32, device=device),
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=True,
            logits_soft_cap=soft_cap,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # "Native page size" is a hint; only the probe settles the route. Where it
    # degrades to flat-gather the guard is meant to fire, so the premise is gone.
    if not _aiter_native_paging_available(
        torch.float16, True, True, native[0], head_dim, device.index or 0
    ):
        pytest.skip(f"aiter cannot serve page_size={native[0]} natively on this build")

    plan(native[0])

    non_native = next(
        (p for p in (16, 32, 64, 8) if p not in _aiter_native_page_sizes()), None
    )
    if non_native is not None:
        with pytest.raises(ValueError, match="logits_soft_cap"):
            plan(non_native)
