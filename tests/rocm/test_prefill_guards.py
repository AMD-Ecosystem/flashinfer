# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Negative cases for the batch-prefill wrappers' argument guards.

Most checks here reject a call before any kernel runs -- the cuda-graph plan
guards included, which raise well before ``get_batch_prefill_module``. Two do
not: ``test_a_plan_within_the_captured_shape_copies_into_the_buffers`` runs a
plan to completion, and the custom-mask case runs ``segment_packbits`` before
the guard it asserts on. Both pin ``backend="fa2"`` so neither can trigger an
AITER variant build. The positive paths live in
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


@pytest.fixture(scope="module")
def aiter_workspace(workspace):
    """For the explicit `backend="aiter"` cases, whose wrapper constructor calls
    `_require_aiter_runtime` before the guard under test can be reached."""
    from flashinfer.rocm.aiter_utils import _aiter_importable, is_aiter_supported

    if not (is_aiter_supported(workspace.device) and _aiter_importable()):
        pytest.skip("the aiter runtime check refuses before the guard under test")
    return workspace


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


def _paged_plan_args(device, **over):
    args = dict(
        qo_indptr=_indptr([0, 4], device),
        paged_kv_indptr=_indptr([0, 2], device),
        paged_kv_indices=_indptr([0, 1], device),
        paged_kv_last_page_len=_indptr([8], device),
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim_qk=128,
        page_size=16,
    )
    args.update(over)
    return args


def _ragged_plan_args(device, **over):
    args = dict(
        qo_indptr=_indptr([0, 4], device),
        kv_indptr=_indptr([0, 8], device),
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim_qk=128,
    )
    args.update(over)
    return args


class TestAiterConstraints:
    """`backend="aiter"` is a promise the wrapper must refuse when it cannot keep it.

    Each check sits immediately before ``get_batch_prefill_module``, so these
    cost no JIT. Under ``auto`` the same conditions fall back to fa2 instead;
    that path is covered by ``test_aiter_auto_fallback.py``.
    """

    def test_paged_rejects_a_pos_encoding_mode_it_cannot_do(self, aiter_workspace):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            aiter_workspace, backend="aiter"
        )
        with pytest.raises(ValueError, match="does not support pos_encoding_mode"):
            wrapper.plan(
                **_paged_plan_args(
                    aiter_workspace.device, pos_encoding_mode="ROPE_LLAMA"
                )
            )

    def test_paged_rejects_a_kv_layout_it_cannot_do(self, aiter_workspace):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            aiter_workspace, kv_layout="HND", backend="aiter"
        )
        with pytest.raises(ValueError, match="only supports kv_layout='NHD'"):
            wrapper.plan(**_paged_plan_args(aiter_workspace.device))

    def test_ragged_rejects_a_pos_encoding_mode_it_cannot_do(self, aiter_workspace):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            aiter_workspace, backend="aiter"
        )
        with pytest.raises(ValueError, match="does not support pos_encoding_mode"):
            wrapper.plan(
                **_ragged_plan_args(
                    aiter_workspace.device, pos_encoding_mode="ROPE_LLAMA"
                )
            )

    def test_ragged_rejects_a_kv_layout_it_cannot_do(self, aiter_workspace):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            aiter_workspace, kv_layout="HND", backend="aiter"
        )
        with pytest.raises(ValueError, match="only supports kv_layout='NHD'"):
            wrapper.plan(**_ragged_plan_args(aiter_workspace.device))


class TestPagedCudaGraphPlan:
    """Shapes are fixed at capture; a later plan must not silently exceed them."""

    def _graph_wrapper(self, workspace, rows=8, batch=1, indices=4):
        device = workspace.device
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            use_cuda_graph=True,
            qo_indptr_buf=_indptr([0] + [rows] * batch, device),
            paged_kv_indptr_buf=_indptr([0] + [indices] * batch, device),
            paged_kv_indices_buf=_indptr(list(range(indices)), device),
            paged_kv_last_page_len_buf=_indptr([8] * batch, device),
        )

    def test_more_rows_than_the_first_plan_saw_is_rejected(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace)
        wrapper._max_total_num_rows = 4

        with pytest.raises(ValueError, match="cannot exceed the number of rows"):
            wrapper.plan(**_paged_plan_args(device, qo_indptr=_indptr([0, 8], device)))

    def test_a_different_batch_size_is_rejected(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace, batch=1)

        with pytest.raises(ValueError, match="batch size should be fixed"):
            wrapper.plan(
                **_paged_plan_args(
                    device,
                    qo_indptr=_indptr([0, 4, 8], device),
                    paged_kv_indptr=_indptr([0, 2, 4], device),
                    paged_kv_last_page_len=_indptr([8, 8], device),
                )
            )

    def test_more_indices_than_the_buffer_holds_is_rejected(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace, indices=2)

        with pytest.raises(ValueError, match="exceeds the allocated buffer size"):
            wrapper.plan(
                **_paged_plan_args(
                    device,
                    paged_kv_indptr=_indptr([0, 6], device),
                    paged_kv_indices=_indptr(list(range(6)), device),
                )
            )


class TestRaggedCudaGraphPlan:
    def _graph_wrapper(self, workspace, rows=8, batch=1):
        device = workspace.device
        return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            use_cuda_graph=True,
            # fa2, not auto: the copy under test happens before backend
            # resolution, and auto can spend minutes building an AITER variant.
            backend="fa2",
            qo_indptr_buf=_indptr([0] + [rows] * batch, device),
            kv_indptr_buf=_indptr([0] + [rows * 2] * batch, device),
        )

    def test_a_plan_within_the_captured_shape_copies_into_the_buffers(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace)

        # Plan a shorter shape than the buffer was captured with: planning [0, 8]
        # into a buffer built as [0, 8] passes whether or not the copy happens.
        wrapper.plan(**_ragged_plan_args(device, qo_indptr=_indptr([0, 6], device)))

        assert wrapper._max_total_num_rows == 6
        assert wrapper._qo_indptr_buf.tolist() == [0, 6]

    def test_more_rows_than_the_first_plan_saw_is_rejected(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace)
        wrapper._max_total_num_rows = 4

        with pytest.raises(ValueError, match="cannot exceed the number of rows"):
            wrapper.plan(**_ragged_plan_args(device, qo_indptr=_indptr([0, 8], device)))

    def test_a_different_batch_size_is_rejected(self, workspace):
        device = workspace.device
        wrapper = self._graph_wrapper(workspace, batch=1)

        with pytest.raises(ValueError, match="batch size should be fixed"):
            wrapper.plan(
                **_ragged_plan_args(
                    device,
                    qo_indptr=_indptr([0, 4, 8], device),
                    kv_indptr=_indptr([0, 8, 16], device),
                )
            )

    @pytest.mark.parametrize("absent", ["custom_mask_buf", "mask_indptr_buf"])
    def test_a_custom_mask_without_its_capture_buffer_is_rejected(
        self, workspace, absent
    ):
        """The mask buffers are optional at construction, so plan() is the only
        place a custom mask can discover they were never allocated."""
        device = workspace.device
        bufs = dict(
            custom_mask_buf=torch.zeros(4096, dtype=torch.uint8, device=device),
            mask_indptr_buf=_indptr([0, 64], device),
        )
        bufs[absent] = None
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            use_cuda_graph=True,
            qo_indptr_buf=_indptr([0, 8], device),
            kv_indptr_buf=_indptr([0, 16], device),
            **bufs,
        )

        with pytest.raises(ValueError, match=f"{absent} must be initialized"):
            wrapper.plan(
                **_ragged_plan_args(
                    device,
                    qo_indptr=_indptr([0, 8], device),
                    kv_indptr=_indptr([0, 16], device),
                    custom_mask=torch.ones(8 * 16, dtype=torch.bool, device=device),
                )
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


class TestSinglePrefillAiterConstraints:
    def test_a_pos_encoding_mode_aiter_cannot_do_is_refused(self, workspace):
        q, k, v = (
            torch.randn(8, 4, 128, dtype=torch.float16, device=workspace.device)
            for _ in range(3)
        )
        with pytest.raises(ValueError, match="does not support pos_encoding_mode"):
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, backend="aiter", pos_encoding_mode="ROPE_LLAMA"
            )

    def test_a_kv_layout_aiter_cannot_do_is_refused(self, workspace):
        q, k, v = (
            torch.randn(8, 4, 128, dtype=torch.float16, device=workspace.device)
            for _ in range(3)
        )
        with pytest.raises(ValueError, match="only supports kv_layout='NHD'"):
            flashinfer.single_prefill_with_kv_cache(
                q, k, v, backend="aiter", kv_layout="HND"
            )


class TestMaskIndptr:
    def test_mismatched_indptr_lengths_are_refused(self, workspace):
        """A custom mask is laid out per (qo, kv) pair, so the two indptrs must
        describe the same number of requests."""
        from flashinfer.rocm.prefill import _compute_mask_indptr

        device = workspace.device
        with pytest.raises(ValueError, match="qo_indptr and kv_indptr"):
            _compute_mask_indptr(_indptr([0, 4, 8], device), _indptr([0, 8], device))

    def test_it_accumulates_the_per_request_mask_sizes(self, workspace):
        from flashinfer.rocm.prefill import _compute_mask_indptr

        device = workspace.device
        got = _compute_mask_indptr(
            _indptr([0, 2, 5], device), _indptr([0, 3, 7], device)
        )
        # 2*3 = 6, then 3*4 = 12
        assert got.tolist() == [0, 6, 18]

    def test_a_ragged_plan_with_a_short_kv_indptr_is_refused(self, workspace):
        device = workspace.device
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="fa2"
        )
        with pytest.raises(ValueError, match="kv_indptr length"):
            wrapper.plan(
                **_ragged_plan_args(
                    device,
                    qo_indptr=_indptr([0, 4, 8], device),
                    kv_indptr=_indptr([0, 8], device),
                )
            )
