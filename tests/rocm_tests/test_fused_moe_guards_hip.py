# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shape and dtype guards around the AITER fused-MoE path.

CK reports a tile mismatch without naming a dimension, so these functions have
to say which one is wrong before the kernel is reached. The fp8 tile rules are
arch-dependent and pure arithmetic, so they are checked directly rather than
through a build.
"""

import pytest
import torch

from flashinfer import fused_moe_rocm
from flashinfer.jit import fused_moe_rocm as jit_moe


class TestFp8ShapeProblem:
    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    @pytest.mark.parametrize("block_m", fused_moe_rocm._SUPPORTED_BLOCK_M)
    def test_a_conforming_shape_has_no_problem(self, arch, block_m):
        k_tile = fused_moe_rocm._fp8_stage2_k_tile(arch, block_m, 8192)
        inter = k_tile * 4
        model = fused_moe_rocm._FP8_STAGE1_K_TILE * 4

        assert fused_moe_rocm._fp8_shape_problem(arch, block_m, model, inter) is None

    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_an_indivisible_model_dim_names_the_dimension(self, arch):
        problem = fused_moe_rocm._fp8_shape_problem(
            arch, 32, fused_moe_rocm._FP8_STAGE1_K_TILE + 1, 8192
        )

        assert problem is not None
        assert "model_dim" in problem

    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_an_indivisible_inter_dim_names_the_dimension_and_the_tile(self, arch):
        k_tile = fused_moe_rocm._fp8_stage2_k_tile(arch, 32, 8192)
        model = fused_moe_rocm._FP8_STAGE1_K_TILE * 4

        problem = fused_moe_rocm._fp8_shape_problem(arch, 32, model, k_tile + 1)

        assert problem is not None
        assert "inter_dim" in problem and arch in problem


class TestMoeFp8Dtype:
    def test_a_supported_arch_resolves_to_a_dtype(self, monkeypatch):
        monkeypatch.setattr(
            fused_moe_rocm, "resolve_aiter_build_arch", lambda: "gfx942"
        )
        assert fused_moe_rocm.moe_fp8_dtype() in fused_moe_rocm.FP8_DTYPES

    def test_an_unsupported_arch_lists_the_ones_that_work(self, monkeypatch):
        monkeypatch.setattr(
            fused_moe_rocm, "resolve_aiter_build_arch", lambda: "gfx1100"
        )

        with pytest.raises(ValueError, match="not supported on gfx1100"):
            fused_moe_rocm.moe_fp8_dtype()


class TestQuantizeMoeWeight:
    def test_a_non_3d_weight_is_refused_by_rank(self):
        """[num_experts, n, k] is the only layout the shim can quantize; a 2-D
        weight would otherwise be reshaped into nonsense."""
        with pytest.raises(ValueError, match="3-D"):
            fused_moe_rocm.quantize_moe_weight(torch.zeros(8, 16))


class TestJitModuleSelection:
    def test_an_unsupported_activation_dtype_is_refused(self):
        with pytest.raises(ValueError, match="fused MoE"):
            jit_moe.gen_fused_moe_aiter_module(torch.float32, "silu")

    def test_a_weight_dtype_that_is_neither_fp8_nor_the_activation_dtype(self):
        with pytest.raises(ValueError, match="expert weights"):
            jit_moe.gen_fused_moe_aiter_module(
                torch.bfloat16, "silu", weight_dtype=torch.float16
            )

    def test_an_unsupported_activation_is_refused(self):
        with pytest.raises(ValueError, match="activations"):
            jit_moe.gen_fused_moe_aiter_module(torch.bfloat16, "relu")
