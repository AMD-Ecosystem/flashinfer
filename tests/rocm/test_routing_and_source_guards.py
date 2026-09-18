# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The last uncovered refusals: RoPE, MLA, the AOT driver and the AITER source
locator.

Every case rejects before a kernel or a build, so the file costs no JIT.
"""

import os

import pytest
import torch

import flashinfer
from flashinfer.jit.rocm import aiter_source
from flashinfer.rocm import aot


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("the entry points take device tensors")
    return torch.device("cuda:0")


class TestRopeRouting:
    def test_an_unknown_backend_is_named(self, device):
        n, head_dim = 4, 128
        q = torch.randn(n, 2, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(n, 2, head_dim, dtype=torch.float16, device=device)
        pos_ids = torch.arange(n, dtype=torch.int32, device=device)
        cache = torch.randn(64, head_dim, dtype=torch.float32, device=device)

        with pytest.raises(ValueError, match="Unknown backend 'cudnn'"):
            flashinfer.apply_rope_with_cos_sin_cache(
                pos_ids, q, k, head_dim, cache, backend="cudnn"
            )

    def test_a_non_float32_cache_is_refused(self, device):
        n, head_dim = 4, 128
        q = torch.randn(n, 2, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(n, 2, head_dim, dtype=torch.float16, device=device)
        pos_ids = torch.arange(n, dtype=torch.int32, device=device)
        cache = torch.randn(64, head_dim, dtype=torch.float16, device=device)

        with pytest.raises(ValueError, match="cos_sin_cache should be float32"):
            flashinfer.apply_rope_with_cos_sin_cache(pos_ids, q, k, head_dim, cache)


class TestMlaPlanGuards:
    def _wrapper(self, device):
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        return flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="aiter")

    def _plan_args(self, device, **over):
        ints = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
        args = dict(
            qo_indptr=ints([0, 4]),
            kv_indptr=ints([0, 2]),
            kv_indices=ints([0, 1]),
            kv_len_arr=ints([32]),
            num_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        args.update(over)
        return args

    def test_an_index_tensor_that_is_not_int32_is_refused(self, device):
        """The shim reads these as int32; a silent reinterpretation of int64
        indices would index the wrong pages."""
        wrapper = self._wrapper(device)
        args = self._plan_args(
            device, kv_indices=torch.tensor([0, 1], dtype=torch.int64, device=device)
        )
        with pytest.raises(ValueError, match="kv_indices.dtype == torch.int32"):
            wrapper.plan(**args)

    def test_a_kv_len_arr_of_the_wrong_length_is_refused(self, device):
        wrapper = self._wrapper(device)
        args = self._plan_args(
            device, kv_len_arr=torch.tensor([32, 32], dtype=torch.int32, device=device)
        )
        with pytest.raises(ValueError, match="kv_len_arr.shape\\[0\\]==batch_size"):
            wrapper.plan(**args)


class TestAiterSourceLocator:
    def test_a_missing_csrc_include_is_a_named_error(self, request, monkeypatch):
        """Both discovery routes failing must say which two were tried."""
        import sys

        aiter_source._aiter_csrc_include_dir.cache_clear()
        request.addfinalizer(aiter_source._aiter_csrc_include_dir.cache_clear)

        # The locator is lru_cached, so anything earlier in the session that
        # resolved it would make the blocked imports below unreachable. Clear it
        # on the way in and out; the next caller simply resolves it again.

        # Both routes go through an `import`, so blocking the packages is what
        # drives the function into its `except Exception: pass` and the raise.
        for name in ("aiter_meta", "aiter.jit.core"):
            monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(RuntimeError, match="Could not locate AITER's csrc/include"):
            aiter_source._aiter_csrc_include_dir()


class TestAotEnvScope:
    def test_a_previously_unset_workspace_base_is_removed_again(
        self, tmp_path, monkeypatch
    ):
        """The restore has two arms and only one of them runs per process; the
        'was unset' arm is the one a normal run never takes."""
        monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)

        with aot._redirected_jit_env(tmp_path):
            assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path)

        assert "FLASHINFER_WORKSPACE_BASE" not in os.environ

    def test_a_previously_set_workspace_base_is_put_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/previous")

        with aot._redirected_jit_env(tmp_path):
            assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path)

        assert os.environ["FLASHINFER_WORKSPACE_BASE"] == "/previous"


class TestSingleDecodeScaling:
    """`single_decode_with_kv_cache`'s scale and LSE arms.

    `q_scale`/`k_scale` fold into `sm_scale` before the launch and `v_scale`
    rescales the output after it, so each is a plain multiply that no test
    pinned -- a dropped one changes the numbers and nothing else.
    """

    def _qkv(self, device, kv_len=64, heads=4, head_dim=128):
        q = torch.randn(heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(kv_len, heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(kv_len, heads, head_dim, dtype=torch.float16, device=device)
        return q, k, v

    def test_q_and_k_scales_fold_into_sm_scale(self, device):
        q, k, v = self._qkv(device)
        base = flashinfer.single_decode_with_kv_cache(q, k, v, sm_scale=0.05)
        folded = flashinfer.single_decode_with_kv_cache(
            q, k, v, sm_scale=0.05, q_scale=2.0, k_scale=5.0
        )
        expected = flashinfer.single_decode_with_kv_cache(q, k, v, sm_scale=0.5)

        torch.testing.assert_close(folded, expected, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(folded, base, rtol=1e-2, atol=1e-2)

    def test_v_scale_rescales_the_output(self, device):
        q, k, v = self._qkv(device)
        base = flashinfer.single_decode_with_kv_cache(q, k, v)
        scaled = flashinfer.single_decode_with_kv_cache(q, k, v, v_scale=2.0)

        torch.testing.assert_close(scaled, base * 2.0, rtol=1e-2, atol=1e-2)

    def test_return_lse_gives_a_float32_row_per_head(self, device):
        q, k, v = self._qkv(device)
        out, lse = flashinfer.single_decode_with_kv_cache(q, k, v, return_lse=True)

        assert out.shape == q.shape
        assert lse.shape == (q.shape[0],)
        assert lse.dtype == torch.float32
        torch.testing.assert_close(
            out, flashinfer.single_decode_with_kv_cache(q, k, v), rtol=1e-3, atol=1e-3
        )
