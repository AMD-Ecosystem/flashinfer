# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Negative cases for the batch-decode wrapper's argument guards.

Each check rejects a call before any kernel runs. The AITER PA v1 cases stub the
``csrc.cpp_itfs`` import the resolver performs first: in a source checkout that
name resolves to the repository's own ``csrc/``, so the guards below it are
otherwise unreachable in-tree.
"""

import sys
import types

import pytest
import torch

import flashinfer
from flashinfer.rocm.decode import (
    _aiter_pa_v1_resolve,
    _merge_deprecated_plan_kwargs,
)

_WORKSPACE_BYTES = 16 * 1024 * 1024


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("the wrapper allocates its scratch on the workspace's device")
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda:0")


class TestDeprecatedPositionalMerge:
    """`plan()` still accepts upstream's legacy positional tail."""

    def test_too_many_positionals_are_rejected(self):
        with pytest.raises(TypeError, match="accepts at most 2"):
            _merge_deprecated_plan_kwargs(
                "BatchDecode", ("a", "b", "c"), ("first", "second"), {}
            )

    def test_a_positional_that_repeats_a_keyword_is_rejected(self):
        with pytest.raises(TypeError, match="multiple values for argument 'first'"):
            _merge_deprecated_plan_kwargs(
                "BatchDecode", ("a",), ("first", "second"), {"first": "x"}
            )

    def test_positionals_become_keywords_in_order(self):
        merged = _merge_deprecated_plan_kwargs(
            "BatchDecode", ("a", "b"), ("first", "second"), {"other": 1}
        )
        assert merged == {"first": "a", "second": "b", "other": 1}


@pytest.fixture
def stub_aiter_pa(monkeypatch):
    """Stand in for aiter's `csrc.cpp_itfs`, which the resolver imports first."""
    pa_v1 = types.ModuleType("csrc.cpp_itfs.pa.pa_v1")
    pa_v1.compile = lambda **_kwargs: pytest.fail(
        "the guard under test let an unsupported problem reach aiter's compile()"
    )
    utils = types.ModuleType("csrc.cpp_itfs.utils")
    utils.BUILD_DIR = "/nonexistent"

    for name, module in (
        ("csrc.cpp_itfs", types.ModuleType("csrc.cpp_itfs")),
        ("csrc.cpp_itfs.pa", types.ModuleType("csrc.cpp_itfs.pa")),
        ("csrc.cpp_itfs.pa.pa_v1", pa_v1),
        ("csrc.cpp_itfs.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _pa_args(**over):
    args = dict(
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        head_dim=128,
        num_qo_heads=8,
        num_kv_heads=8,
        page_size=16,
        max_context_len=1024,
        logits_soft_cap=0.0,
        sliding_window=-1,
    )
    args.update(over)
    return args


class TestAiterPaV1Constraints:
    def test_an_unsupported_query_dtype_is_rejected(self, stub_aiter_pa):
        with pytest.raises(ValueError, match="dtype_q="):
            _aiter_pa_v1_resolve(**_pa_args(dtype_q=torch.float32))

    def test_an_unsupported_kv_dtype_is_rejected(self, stub_aiter_pa):
        with pytest.raises(ValueError, match="dtype_kv="):
            _aiter_pa_v1_resolve(**_pa_args(dtype_kv=torch.float8_e4m3fnuz))

    def test_a_head_count_that_does_not_divide_is_rejected(self, stub_aiter_pa):
        with pytest.raises(ValueError, match="must be divisible by num_kv_heads"):
            _aiter_pa_v1_resolve(**_pa_args(num_qo_heads=7, num_kv_heads=2))


class TestWorkspaceSizeQuery:
    def test_it_says_ROCm_has_no_such_query(self, workspace):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
        with pytest.raises(NotImplementedError, match="not available on ROCm"):
            wrapper.workspace_size(
                indptr=torch.tensor([0, 1], dtype=torch.int32),
                indices=torch.tensor([0], dtype=torch.int32),
                last_page_len=torch.tensor([1], dtype=torch.int32),
                num_qo_heads=8,
                num_kv_heads=8,
                head_dim=128,
                page_size=16,
            )
