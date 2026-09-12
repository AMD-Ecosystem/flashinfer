# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`auto` sends short-query paged prefill to fa2, and re-decides every plan.

AITER's mha_batch_prefill costs the same whatever the query length, so a
spec-decode verify or a chunked-prefill tail pays a full KV scan for a handful
of rows. The wrapper is long-lived and re-planned per step, so the choice has
to follow the shape rather than stick to whatever the first plan saw.
"""

import pytest
import torch

import flashinfer
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.prefill import _AITER_SHORT_QO_LEN, _aiter_ops_importable

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm device"
)

HEAD_DIM = 128
NHQ, NHKV = 32, 8
PAGE = 16
LONG_QO = 128  # comfortably past the threshold, and a native tile multiple


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def _aiter_declined():
    """Positive control, run once: plan a query that should keep AITER.

    Without AITER every plan resolves to fa2 and the file passes for the wrong
    reason. Capability is not enough to check -- plan() also probes, and a
    build that cannot compile the variant demotes with a bootstrap reason.
    """
    dev = torch.device("cuda:0")
    if not is_aiter_supported(dev) or not _aiter_ops_importable():
        return "AITER requires a gfx942/gfx950 GPU and the aiter package"
    wrapper = _plan_paged(_paged_wrapper(dev), dev, LONG_QO)
    if wrapper._backend == "aiter":
        return None
    return f"AITER declined a long query here: {wrapper._backend_fallback_reason}"


@pytest.fixture(autouse=True)
def _require_aiter(_aiter_declined):
    if _aiter_declined is not None:
        pytest.skip(_aiter_declined)


def _paged_wrapper(device, backend="auto"):
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=backend)


def _plan_paged(wrapper, device, s_qo, s_kv=1024, dtype=torch.bfloat16, **kwargs):
    npages = s_kv // PAGE
    wrapper.plan(
        torch.tensor([0, s_qo], dtype=torch.int32, device=device),
        torch.tensor([0, npages], dtype=torch.int32, device=device),
        torch.arange(npages, dtype=torch.int32, device=device),
        torch.tensor([PAGE], dtype=torch.int32, device=device),
        NHQ,
        NHKV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        **kwargs,
    )
    return wrapper


def _plan_ragged(wrapper, device, s_qo, s_kv=1024):
    wrapper.plan(
        torch.tensor([0, s_qo], dtype=torch.int32, device=device),
        torch.tensor([0, s_kv], dtype=torch.int32, device=device),
        NHQ,
        NHKV,
        HEAD_DIM,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return wrapper


class TestPagedRouting:
    @pytest.mark.parametrize("s_qo", [1, 2, 8, _AITER_SHORT_QO_LEN])
    def test_a_short_query_plans_onto_fa2(self, device, s_qo):
        w = _plan_paged(_paged_wrapper(device), device, s_qo)

        assert w._backend == "fa2"
        assert f"qo_len={s_qo}" in w._backend_fallback_reason

    def test_a_long_query_stays_on_aiter(self, device):
        w = _plan_paged(_paged_wrapper(device), device, LONG_QO)

        assert w._backend == "aiter"

    def test_an_explicit_aiter_backend_is_not_redirected(self, device):
        """The gate is a preference, not a capability: asking for AITER at a
        short query must still get AITER."""
        w = _plan_paged(_paged_wrapper(device, backend="aiter"), device, 1)

        assert w._backend == "aiter"

    def test_a_short_fp8_query_still_reaches_aiter(self, device):
        """fa2 has no fp8 kernel, so preferring it on speed would turn a call
        that worked into a NotImplementedError with no route left."""
        import aiter

        w = _plan_paged(_paged_wrapper(device), device, 8, dtype=aiter.dtypes.fp8)

        assert w._backend == "aiter"


class TestReplanning:
    """The wrapper resolved `auto` once and kept the answer, so before this the
    first plan's shape decided every later step's backend."""

    def test_planning_longer_returns_to_aiter(self, device):
        w = _paged_wrapper(device)

        _plan_paged(w, device, 1)
        assert w._backend == "fa2"

        _plan_paged(w, device, LONG_QO)
        assert w._backend == "aiter"

    def test_planning_shorter_moves_to_fa2(self, device):
        w = _paged_wrapper(device)

        _plan_paged(w, device, LONG_QO)
        assert w._backend == "aiter"

        _plan_paged(w, device, 1)
        assert w._backend == "fa2"

    def test_a_graph_enabled_wrapper_keeps_its_first_choice(self, device):
        """A captured graph holds buffers belonging to the backend it captured,
        so re-resolution is off there even though the shape changed."""
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws,
            "NHD",
            backend="auto",
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(2, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=device),
            paged_kv_indices_buf=torch.zeros(64, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buf=torch.zeros(1, dtype=torch.int32, device=device),
        )

        # Long first: graph mode caps later plans at the first plan's row count.
        _plan_paged(w, device, LONG_QO, s_kv=1024)
        assert w._backend == "aiter"

        _plan_paged(w, device, 1, s_kv=1024)

        assert w._backend == "aiter"
        # Not merely "still aiter": the selector must not have run at all, which
        # a re-resolution that happened to re-pick aiter would not satisfy.
        assert "qo_len" not in (w._backend_fallback_reason or "")


class TestRaggedIsNotGated:
    """mha_varlen_fwd's short-query cost splits by architecture -- gfx950 runs
    AITER 1.7x faster than fa2 at bs32/kv2048/q16 where gfx942 runs it slower --
    so the ragged wrapper deliberately passes no qo_len."""

    def test_a_short_query_keeps_aiter(self, device):
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="auto")

        _plan_ragged(w, device, 1)

        assert w._backend == "aiter"


class TestNumerics:
    """Routing away from AITER must not change the answer."""

    @pytest.mark.parametrize("s_qo", [1, 8, _AITER_SHORT_QO_LEN])
    def test_the_short_query_result_matches_an_aiter_reference(self, device, s_qo):
        s_kv, npages = 1024, 1024 // PAGE
        torch.manual_seed(0)
        q = torch.randn(s_qo, NHQ, HEAD_DIM, dtype=torch.bfloat16, device=device)
        kv = torch.randn(
            npages, 2, PAGE, NHKV, HEAD_DIM, dtype=torch.bfloat16, device=device
        )

        auto_w = _plan_paged(_paged_wrapper(device), device, s_qo, s_kv=s_kv)
        ref_w = _plan_paged(
            _paged_wrapper(device, backend="aiter"), device, s_qo, s_kv=s_kv
        )
        # Or the comparison is fa2 against fa2 and proves nothing about routing.
        assert (auto_w._backend, ref_w._backend) == ("fa2", "aiter")

        torch.testing.assert_close(
            auto_w.run(q, kv), ref_w.run(q, kv), rtol=2e-2, atol=2e-2
        )
