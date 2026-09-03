# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime half of the ROCm/upstream API parity contract.

Asserts that each CUDA-only argument raises rather than being ignored, and that
the three mis-bound positions now bind what upstream binds. Needs a GPU; the
static signature diff is in ``test_api_parity.py``, which does not.
"""

import importlib.util
import inspect
import sys
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer.rocm.api_compat import reject_cuda_only

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    name = "_fi_rocm_api_parity"
    target = _REPO_ROOT / "scripts" / "rocm_api_parity.py"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises on a module that is not there yet.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = _load_tool()


class TestRejectHelper:
    def test_default_is_a_no_op(self):
        reject_cuda_only("x", None, None)
        reject_cuda_only("x", False, False)
        reject_cuda_only("x", 0, 0)

    def test_non_default_raises_naming_the_argument(self):
        with pytest.raises(NotImplementedError, match="kv_cache_sf"):
            reject_cuda_only("kv_cache_sf", object(), None)

    def test_a_tensor_never_reaches_an_ambiguous_comparison(self):
        """``tensor == None`` has no truth value; the helper must not evaluate it."""
        with pytest.raises(NotImplementedError, match="kv_cache_sf"):
            reject_cuda_only("kv_cache_sf", torch.zeros(4), None)

    def test_a_neutral_value_is_accepted(self):
        """Asking for a feature to be *off* is not asking for the feature."""
        reject_cuda_only("use_fp16_softmax", False, None, neutral=False)
        reject_cuda_only("o_scale", 1.0, None, neutral=1.0)
        with pytest.raises(NotImplementedError):
            reject_cuda_only("o_scale", 2.0, None, neutral=1.0)


class TestScalesAreHonouredNotRefused:
    """q/k/v_scale are implemented on every path, so they must not raise.

    They were refused on the ragged and single paths in an earlier revision;
    the fold is three lines the paged wrapper already had.
    """

    def test_single_prefill_folds_k_scale_and_v_scale(self):
        torch.manual_seed(0)
        q = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")
        k = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")
        v = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")
        base = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)
        scaled = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, v_scale=2.0
        )
        torch.testing.assert_close(
            scaled.float(), base.float() * 2.0, rtol=2e-2, atol=2e-2
        )

        # k_scale folds into sm_scale, so it reshapes the softmax rather than
        # scaling the result; assert only that it is not dropped.
        k_scaled = flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, k_scale=4.0
        )
        assert not torch.allclose(k_scaled.float(), base.float(), rtol=1e-2, atol=1e-2)

    def test_ragged_run_folds_v_scale(self):
        torch.manual_seed(0)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        q = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")
        k = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")
        v = torch.randn(4, 4, 128, dtype=torch.float16, device="cuda")

        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
        wrapper.plan(qo_indptr, kv_indptr, 4, 4, 128, causal=False)
        base = wrapper.run(q, k, v)
        scaled = wrapper.run(q, k, v, v_scale=2.0)
        torch.testing.assert_close(
            scaled.float(), base.float() * 2.0, rtol=2e-2, atol=2e-2
        )


def _plan_kwargs(name, method):
    """Every CUDA-only parameter the given callable declares, with a live value."""
    values = {
        bool: True,
        int: 7,
        float: 2.0,
    }
    out = {}
    for pname, param in inspect.signature(method).parameters.items():
        if pname not in parity.CUDA_ONLY_PARAMS:
            continue
        if isinstance(param.default, bool):
            out[pname] = not param.default
        elif isinstance(param.default, (int, float)) and param.default is not None:
            out[pname] = values[type(param.default)]
        else:
            out[pname] = torch.zeros(1) if "indptr" in pname or "sf" in pname else 2.0
    return out


class TestCudaOnlyArgumentsRaise:
    """Each declared CUDA-only argument must raise, not be quietly dropped."""

    @pytest.mark.parametrize(
        "factory, method_name",
        [
            (
                lambda ws: flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD"),
                "plan",
            ),
            (
                lambda ws: flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD"),
                "plan",
            ),
            (
                lambda ws: flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD"),
                "plan",
            ),
        ],
    )
    def test_each_cuda_only_plan_argument_raises(self, factory, method_name):
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = factory(ws)
        method = getattr(wrapper, method_name)
        target = getattr(wrapper, "_plan_impl", method)
        candidates = _plan_kwargs(method_name, target)
        assert candidates, "no CUDA-only parameters found -- the table went stale"
        for pname, value in candidates.items():
            with pytest.raises(NotImplementedError, match=pname):
                method(*_MINIMAL_PLAN_ARGS[type(wrapper).__name__], **{pname: value})

    def test_decode_rejects_multi_token_query(self):
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        with pytest.raises(NotImplementedError, match="q_len_per_req"):
            wrapper.plan(
                *_MINIMAL_PLAN_ARGS["BatchDecodeWithPagedKVCacheWrapper"],
                q_len_per_req=2,
            )

    def test_workspace_size_is_unavailable_not_missing(self):
        """AttributeError would read as "old FlashInfer"; this says what is wrong."""
        ws = torch.empty(1024, dtype=torch.int8, device="cuda")
        decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        for wrapper in (decode, prefill):
            assert hasattr(wrapper, "workspace_size")
            with pytest.raises(NotImplementedError, match="workspace_size"):
                wrapper.workspace_size(*_MINIMAL_PLAN_ARGS[type(wrapper).__name__])


_MINIMAL_PLAN_ARGS = {
    # indptr/indices/last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size
    "BatchDecodeWithPagedKVCacheWrapper": (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        8,
        8,
        128,
        16,
    ),
    # qo_indptr, kv_indptr, kv_indices, last_page_len, heads, head_dim_qk, page_size
    "BatchPrefillWithPagedKVCacheWrapper": (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        8,
        8,
        128,
        16,
    ),
    # qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk
    "BatchPrefillWithRaggedKVCacheWrapper": (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        8,
        8,
        128,
    ),
}


class TestPositionalBindings:
    """The three signatures where upstream inserted a parameter mid-list.

    A wrong bind is silent, so each case asserts the error names the argument
    upstream puts at that position -- not merely that something raised.
    """

    def test_mla_second_positional_is_use_cuda_graph(self):
        ws = torch.empty(1024, dtype=torch.int8)
        with pytest.raises(NotImplementedError, match="use_cuda_graph"):
            flashinfer.BatchMLAPagedAttentionWrapper(ws, True)

    def test_mla_backend_is_still_reachable_by_keyword(self):
        ws = torch.empty(1024, dtype=torch.int8)
        with pytest.raises(ValueError, match="backend"):
            flashinfer.BatchMLAPagedAttentionWrapper(ws, backend="fa2")

    def test_mla_run_sixth_positional_is_lse(self):
        ws = torch.empty(1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchMLAPagedAttentionWrapper(ws)
        q = torch.zeros(1, 1, 512, device="cuda")
        with pytest.raises(NotImplementedError, match="lse"):
            wrapper.run(q, q, q, q, None, torch.zeros(1))

    def test_decode_plan_sixth_optional_positional_is_o_data_type(self):
        """Upstream inserted o_data_type here; ROCm used to bind data_type."""
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        args = _MINIMAL_PLAN_ARGS["BatchDecodeWithPagedKVCacheWrapper"]
        legacy = ("NONE", -1, None, torch.float16, torch.float16, torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(NotImplementedError, match="o_data_type"):
                wrapper.plan(*args, *legacy)


class TestDeprecatedPositionalPlan:
    """pyproject silences DeprecationWarning suite-wide, so assert it explicitly."""

    def test_positional_optionals_warn(self):
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        args = _MINIMAL_PLAN_ARGS["BatchDecodeWithPagedKVCacheWrapper"]
        with pytest.warns(DeprecationWarning, match="positionally is"):
            wrapper.plan(*args, "NONE", q_data_type=torch.float16)

    def test_keyword_only_call_does_not_warn(self):
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        args = _MINIMAL_PLAN_ARGS["BatchDecodeWithPagedKVCacheWrapper"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            wrapper.plan(*args, pos_encoding_mode="NONE", q_data_type=torch.float16)

    def test_duplicate_between_positional_and_keyword_is_a_type_error(self):
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        args = _MINIMAL_PLAN_ARGS["BatchDecodeWithPagedKVCacheWrapper"]
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            wrapper.plan(*args, "NONE", pos_encoding_mode="NONE")


class TestUpstreamPlanShim:
    """`sparse.py` reaches the shared prefill binding directly with 20 arguments.

    The ROCm binding takes 15, so the shim has to bind the rest or the call
    fails with "expected at most 15 argument(s) but received 20".
    """

    @staticmethod
    def _shim():
        from flashinfer.rocm.prefill import _plan_with_upstream_signature

        seen = []
        plan = _plan_with_upstream_signature(lambda *a: seen.append(a) or "planned")
        return plan, seen

    def test_upstream_shape_forwards_only_the_first_fifteen(self):
        plan, seen = self._shim()
        args = list(range(15))
        assert plan(*args, -1, -1, False, 0, 0) == "planned"
        assert seen == [tuple(args)]

    def test_the_rocm_shape_still_works(self):
        plan, seen = self._shim()
        args = list(range(15))
        assert plan(*args) == "planned"
        assert seen == [tuple(args)]

    @pytest.mark.parametrize(
        "name,value",
        [
            ("fixed_split_size", 128),
            ("disable_split_kv", True),
            ("num_colocated_ctas", 4),
            ("uniform_q_len", 8),
        ],
    )
    def test_cuda_only_scheduler_knobs_raise_by_keyword(self, name, value):
        plan, _ = self._shim()
        with pytest.raises(NotImplementedError, match=name):
            plan(*range(15), **{name: value})

    def test_window_left_is_accepted_and_dropped(self):
        """ROCm has no plan-time slot for it; run() applies the mask instead."""
        plan, seen = self._shim()
        assert plan(*range(15), 64) == "planned"
        assert seen == [tuple(range(15))]


class TestTopLevelExports:
    def test_the_sparse_wrappers_are_exported(self):
        for name in (
            "BlockSparseAttentionWrapper",
            "VariableBlockSparseAttentionWrapper",
            "convert_bsr_mask_layout",
        ):
            assert hasattr(flashinfer, name), f"flashinfer.{name} missing on ROCm"

    def test_the_git_dunders_carry_the_real_hash(self):
        """`isinstance(str)` would also pass on the permanent "unknown" fallback."""
        from flashinfer import _version

        expected = (_version.__commit_id__ or "unknown").removeprefix("g")
        assert flashinfer.__git_commit__ == expected
        assert flashinfer.__git_version__ == expected
        if _version.__commit_id__:
            assert flashinfer.__git_commit__ != "unknown"

    @pytest.mark.parametrize(
        "name", ["check_torch_rocm_compatibility", "gate_cuda_only_modules"]
    )
    def test_internals_do_not_leak_into_the_namespace(self, name):
        assert not hasattr(flashinfer, name), f"flashinfer.{name} leaked via import *"
