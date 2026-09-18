# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Negative cases for the batch-prefill wrappers' argument guards.

Every check here rejects a call before any kernel runs, so the whole file costs
one workspace allocation and no JIT. The positive paths live in
``test_batch_prefill_kernels.py``; without these, a guard that stopped guarding
would be invisible -- the wrapper would accept the bad argument and the failure
would surface as wrong output or a CUDA-graph replay crash much later.
"""

import pytest
import torch

import flashinfer

_WORKSPACE_BYTES = 16 * 1024 * 1024


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("the wrapper allocates its scratch on the workspace's device")
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda:0")


def _indptr(values, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


class TestPagedCudaGraphBuffers:
    """In CUDA-graph mode every buffer is captured, so each must be a tensor.

    A None here is a replay-time crash in the caller's graph, far from the call
    that caused it.
    """

    @pytest.mark.parametrize(
        "missing",
        [
            "qo_indptr_buf",
            "paged_kv_indptr_buf",
            "paged_kv_indices_buf",
            "paged_kv_last_page_len_buf",
        ],
    )
    def test_a_non_tensor_buffer_is_rejected(self, workspace, missing):
        device = workspace.device
        bufs = dict(
            qo_indptr_buf=_indptr([0, 4], device),
            paged_kv_indptr_buf=_indptr([0, 2], device),
            paged_kv_indices_buf=_indptr([0, 1], device),
            paged_kv_last_page_len_buf=_indptr([2], device),
        )
        bufs[missing] = None

        with pytest.raises(ValueError, match=f"{missing} should be a torch.Tensor"):
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, use_cuda_graph=True, **bufs
            )

    def test_an_indptr_of_the_wrong_length_is_rejected(self, workspace):
        device = workspace.device
        with pytest.raises(
            ValueError, match=r"paged_kv_indptr_buf should be batch_size \+ 1"
        ):
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                use_cuda_graph=True,
                qo_indptr_buf=_indptr([0, 4, 8], device),  # batch_size 2
                paged_kv_indptr_buf=_indptr([0, 2], device),  # needs 3 entries
                paged_kv_indices_buf=_indptr([0, 1], device),
                paged_kv_last_page_len_buf=_indptr([2, 2], device),
            )

    def test_a_last_page_len_of_the_wrong_length_is_rejected(self, workspace):
        device = workspace.device
        with pytest.raises(
            ValueError, match="paged_kv_last_page_len_buf should be batch_size"
        ):
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                use_cuda_graph=True,
                qo_indptr_buf=_indptr([0, 4, 8], device),
                paged_kv_indptr_buf=_indptr([0, 2, 4], device),
                paged_kv_indices_buf=_indptr([0, 1, 2, 3], device),
                paged_kv_last_page_len_buf=_indptr([2], device),  # needs 2
            )

    def test_without_cuda_graph_the_batch_size_is_not_fixed(self, workspace):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace)
        assert wrapper._fixed_batch_size == 0


class TestRaggedCudaGraphBuffers:
    @pytest.mark.parametrize("missing", ["qo_indptr_buf", "kv_indptr_buf"])
    def test_a_non_tensor_buffer_is_rejected(self, workspace, missing):
        device = workspace.device
        bufs = dict(
            qo_indptr_buf=_indptr([0, 4], device),
            kv_indptr_buf=_indptr([0, 8], device),
        )
        bufs[missing] = None

        with pytest.raises(ValueError, match=f"{missing} should be a torch.Tensor"):
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, use_cuda_graph=True, **bufs
            )

    def test_mismatched_indptr_lengths_are_rejected(self, workspace):
        device = workspace.device
        with pytest.raises(ValueError, match="length of kv_indptr_buf"):
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                use_cuda_graph=True,
                qo_indptr_buf=_indptr([0, 4, 8], device),
                kv_indptr_buf=_indptr([0, 8], device),
            )


class TestBackendSelection:
    def test_an_unknown_backend_is_rejected_by_the_ragged_wrapper(self, workspace):
        with pytest.raises(ValueError, match="backend must be one of"):
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, backend="trtllm-gen"
            )

    def test_an_unknown_backend_falls_back_on_the_paged_wrapper(self, workspace):
        """The paged wrapper warns and picks fa2 rather than raising."""
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, backend="trtllm-gen"
        )
        assert wrapper.backend == "fa2"
