# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""URI and JitSpec generation in jit/attention/modules_hip.py.

Only the generator is exercised -- no source is compiled and no kernel runs.
The branches covered here are the ones a gfx942 suite never takes: the ``fa3``
backend upstream still offers and this port does not implement, and the
``auto``/unknown backends that must be refused rather than guessed at.
"""

import pytest
import torch

from flashinfer.jit.rocm import modules as modules_hip

_DT = dict(dtype_q=torch.float16, dtype_kv=torch.float16, dtype_o=torch.float16)
_DIMS = dict(
    head_dim_qk=128,
    head_dim_vo=128,
    pos_encoding_mode=0,
    use_sliding_window=False,
    use_logits_soft_cap=False,
    use_fp16_qk_reduction=False,
)


class TestUriGeneration:
    def test_fa2_and_aiter_do_not_share_a_cache_key(self):
        fa2 = modules_hip.get_single_prefill_uri("fa2", **_DT, **_DIMS)
        aiter = modules_hip.get_single_prefill_uri("aiter", **_DT, **_DIMS)

        assert fa2 != aiter
        assert "aiter" in aiter

    def test_fa3_produces_the_fa2_uri_so_the_error_comes_from_the_generator(self):
        """Upstream callers still pass fa3. Sharing fa2's cache key is what keeps
        the refusal in gen_*, with a reason, rather than in a lookup miss."""
        assert modules_hip.get_single_prefill_uri(
            "fa3", **_DT, **_DIMS
        ) == modules_hip.get_single_prefill_uri("fa2", **_DT, **_DIMS)
        assert modules_hip.get_batch_prefill_uri(
            "fa3", **_DT, dtype_idx=torch.int32, **_DIMS
        ) == modules_hip.get_batch_prefill_uri(
            "fa2", **_DT, dtype_idx=torch.int32, **_DIMS
        )

    def test_batch_and_single_uris_differ(self):
        single = modules_hip.get_single_prefill_uri("fa2", **_DT, **_DIMS)
        batch = modules_hip.get_batch_prefill_uri(
            "fa2", **_DT, dtype_idx=torch.int32, **_DIMS
        )

        assert single != batch


class TestUnsupportedBackends:
    """fa3 is a real upstream backend with no ROCm implementation. Refusing it
    by name is what stops a silent fallback to the wrong kernel."""

    def test_single_prefill_fa3_is_refused(self):
        with pytest.raises(ValueError, match="FA3 backend not currently supported"):
            modules_hip.gen_single_prefill_module("fa3", **_DT, **_DIMS)

    def test_batch_prefill_fa3_is_refused(self):
        with pytest.raises(ValueError, match="FA3 backend not currently supported"):
            modules_hip.gen_batch_prefill_module(
                "fa3", **_DT, dtype_idx=torch.int32, **_DIMS
            )

    @pytest.mark.parametrize(
        "gen, extra",
        [
            ("gen_customize_single_prefill_module", {}),
            ("gen_customize_batch_prefill_module", {"idtype": torch.int32}),
        ],
    )
    @pytest.mark.parametrize(
        "backend, match",
        [
            ("auto", "should not be auto"),
            ("fa3", "FA3 backend not currently supported"),
            ("nonsense", "Invalid backend"),
        ],
    )
    def test_customize_generators_refuse_every_backend_they_cannot_build(
        self, gen, extra, backend, match
    ):
        """`auto` is a caller-facing word; by the time a module is generated the
        backend must already be resolved. fa3 and anything unrecognised have to
        be named rather than fall through to the fa2 path."""
        with pytest.raises(ValueError, match=match):
            getattr(modules_hip, gen)(
                backend,
                "uri",
                **_DT,
                **extra,
                head_dim_qk=128,
                head_dim_vo=128,
                additional_tensor_names=[],
                additional_tensor_dtypes=[],
                additional_scalar_names=["sm_scale"],
                additional_scalar_dtypes=["double"],
                variant_name="DefaultAttention<false>",
                variant_decl="#include<x.cuh>",
            )


class TestAdditionalParams:
    def test_sm90_template_emits_optional_pointer_setters(self):
        """The sm90 arm is dead on ROCm but reachable through the shared
        generator, and it formats `maybe_` names differently."""
        decl, func_params, setter = modules_hip._generate_additional_params_hip(
            ["maybe_custom_mask"],
            ["uint8_t"],
            ["sm_scale"],
            ["double"],
            is_sm90_template=True,
        )

        assert "maybe_custom_mask" in decl
        assert "sm_scale" in func_params
        assert "static_cast<uint8_t*>" in setter

    def test_default_template_still_names_every_parameter(self):
        decl, func_params, setter = modules_hip._generate_additional_params_hip(
            ["maybe_alibi_slopes"], ["float"], ["logits_soft_cap"], ["double"]
        )

        assert "maybe_alibi_slopes" in decl
        assert "logits_soft_cap" in func_params
        assert setter
