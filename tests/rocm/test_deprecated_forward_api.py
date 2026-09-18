# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The deprecated ``begin_forward``/``forward``/``end_forward`` aliases.

Upstream kept them and callers still reach them, but nothing here exercised
them: each ``forward`` copies eight or nine keyword arguments onto ``self``
before delegating, so a dropped assignment is a silently ignored argument
rather than an error. Each case asserts the alias agrees with the modern call,
which is what makes the copy meaningful rather than merely executed.
"""

import pytest
import torch

import flashinfer

_HEAD_DIM = 128
_NUM_HEADS = 4
_PAGE_SIZE = 16
_WORKSPACE_BYTES = 128 * 1024 * 1024


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def workspace(device):
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)


def _paged_case(device, qo_len=8, kv_len=32):
    pages = kv_len // _PAGE_SIZE
    q = torch.randn(qo_len, _NUM_HEADS, _HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(
        pages, 2, _PAGE_SIZE, _NUM_HEADS, _HEAD_DIM, dtype=torch.float16, device=device
    )
    ints = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    return (
        q,
        kv,
        dict(
            qo_indptr=ints([0, qo_len]),
            paged_kv_indptr=ints([0, pages]),
            paged_kv_indices=ints(list(range(pages))),
            paged_kv_last_page_len=ints([_PAGE_SIZE]),
            num_qo_heads=_NUM_HEADS,
            num_kv_heads=_NUM_HEADS,
            head_dim_qk=_HEAD_DIM,
            page_size=_PAGE_SIZE,
        ),
    )


def _ragged_case(device, qo_len=8, kv_len=32):
    shape = lambda n: (n, _NUM_HEADS, _HEAD_DIM)  # noqa: E731
    q = torch.randn(*shape(qo_len), dtype=torch.float16, device=device)
    k = torch.randn(*shape(kv_len), dtype=torch.float16, device=device)
    v = torch.randn(*shape(kv_len), dtype=torch.float16, device=device)
    ints = lambda val: torch.tensor(val, dtype=torch.int32, device=device)  # noqa: E731
    return (
        q,
        k,
        v,
        dict(
            qo_indptr=ints([0, qo_len]),
            kv_indptr=ints([0, kv_len]),
            num_qo_heads=_NUM_HEADS,
            num_kv_heads=_NUM_HEADS,
            head_dim_qk=_HEAD_DIM,
        ),
    )


class TestPagedPrefillAliases:
    def test_forward_agrees_with_run(self, workspace, device):
        q, kv, plan_args = _paged_case(device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, backend="fa2"
        )
        wrapper.begin_forward(**plan_args, causal=True)

        expected = wrapper.run(q, kv)
        got = wrapper.forward(q, kv, causal=True)

        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
        wrapper.end_forward()  # deprecated and inert, but must not raise

    def test_forward_overrides_what_plan_set(self, workspace, device):
        """Each argument must be *copied*; plan() has already set these, so a
        dropped assignment is invisible unless the two states differ."""
        q, kv, plan_args = _paged_case(device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, backend="fa2"
        )
        wrapper.begin_forward(**plan_args, causal=False, sm_scale=0.5)
        assert (wrapper._causal, wrapper._sm_scale) == (False, 0.5)

        wrapper.forward(q, kv, causal=True, sm_scale=0.125, rope_theta=2e4)

        assert (wrapper._causal, wrapper._sm_scale) == (True, 0.125)
        assert wrapper._rope_theta == 2e4


class TestRaggedPrefillAliases:
    def test_forward_agrees_with_run(self, workspace, device):
        q, k, v, plan_args = _ragged_case(device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="fa2"
        )
        wrapper.begin_forward(**plan_args, causal=True)

        expected = wrapper.run(q, k, v)
        got = wrapper.forward(q, k, v, causal=True)

        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
        wrapper.end_forward()

    def test_forward_overrides_what_plan_set(self, workspace, device):
        q, k, v, plan_args = _ragged_case(device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="fa2"
        )
        wrapper.begin_forward(**plan_args, causal=False, sm_scale=0.5)

        wrapper.forward(q, k, v, causal=True, sm_scale=0.125, rope_theta=2e4)

        assert (wrapper._causal, wrapper._sm_scale) == (True, 0.125)
        assert wrapper._rope_theta == 2e4

    def test_forward_return_lse_matches_run_return_lse(self, workspace, device):
        q, k, v, plan_args = _ragged_case(device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="fa2"
        )
        wrapper.begin_forward(**plan_args, causal=False)

        want_o, want_lse = wrapper.run_return_lse(q, k, v)
        got_o, got_lse = wrapper.forward_return_lse(q, k, v)

        torch.testing.assert_close(got_o, want_o, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(got_lse, want_lse, rtol=1e-3, atol=1e-3)

        wrapper.forward_return_lse(q, k, v, causal=True, sm_scale=0.125)
        assert (wrapper._causal, wrapper._sm_scale) == (True, 0.125)


class TestDecodeAliases:
    def test_forward_matches_run_and_records_its_arguments(self, workspace, device):
        pages = 32 // _PAGE_SIZE
        q = torch.randn(1, _NUM_HEADS, _HEAD_DIM, dtype=torch.float16, device=device)
        kv = torch.randn(
            pages,
            2,
            _PAGE_SIZE,
            _NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.float16,
            device=device,
        )
        ints = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731

        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
        wrapper.begin_forward(
            ints([0, pages]),
            ints(list(range(pages))),
            ints([_PAGE_SIZE]),
            _NUM_HEADS,
            _NUM_HEADS,
            _HEAD_DIM,
            _PAGE_SIZE,
            q_data_type=torch.float16,
        )

        expected = wrapper.run(q, kv)
        got = wrapper.forward(q, kv)

        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)

        # plan() left sm_scale unset, so the copy is the only way it can arrive.
        wrapper.forward(q, kv, sm_scale=0.125, rope_theta=2e4)
        assert (wrapper._sm_scale, wrapper._rope_theta) == (0.125, 2e4)
        wrapper.end_forward()


class TestBatchDecodeScaling:
    """`q_scale`/`k_scale` fold into `sm_scale`; `v_scale` rescales the output.

    Three separate copies of the `v_scale` block exist in `run()` -- one per
    routing arm -- and none was exercised. A dropped multiply there changes the
    numbers and raises nothing.
    """

    def _planned(self, workspace, device, pages=2, sm_scale=None):
        ints = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
        kv = torch.randn(
            pages,
            2,
            _PAGE_SIZE,
            _NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.float16,
            device=device,
        )
        q = torch.randn(1, _NUM_HEADS, _HEAD_DIM, dtype=torch.float16, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
        wrapper.plan(
            ints([0, pages]),
            ints(list(range(pages))),
            ints([_PAGE_SIZE]),
            _NUM_HEADS,
            _NUM_HEADS,
            _HEAD_DIM,
            _PAGE_SIZE,
            q_data_type=torch.float16,
            sm_scale=sm_scale,
        )
        return wrapper, q, kv

    def test_q_and_k_scales_fold_into_sm_scale(self, workspace, device):
        """sm_scale is a plan() argument; the two scales multiply it at run()."""
        small, q, kv = self._planned(workspace, device, sm_scale=0.05)
        folded = small.run(q, kv, q_scale=2.0, k_scale=5.0)

        big, _, _ = self._planned(workspace, device, sm_scale=0.5)
        expected = big.run(q, kv)

        torch.testing.assert_close(folded, expected, rtol=1e-2, atol=1e-2)

    def test_v_scale_rescales_the_output(self, workspace, device):
        wrapper, q, kv = self._planned(workspace, device)

        base = wrapper.run(q, kv)
        scaled = wrapper.run(q, kv, v_scale=2.0)

        torch.testing.assert_close(scaled, base * 2.0, rtol=1e-2, atol=1e-2)

    def test_a_caller_supplied_lse_buffer_is_shape_checked(self, workspace, device):
        wrapper, q, kv = self._planned(workspace, device)
        wrong = torch.empty(
            (q.size(0), q.size(1) + 1), dtype=torch.float32, device=device
        )

        with pytest.raises(Exception, match="lse"):
            wrapper.run(q, kv, return_lse=True, lse=wrong)
