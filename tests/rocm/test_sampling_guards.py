# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Negative cases for the sampling entry points' argument guards.

Every check here runs before ``get_sampling_module()``, so the file costs no
JIT and no kernel. ``check_nan`` in particular is the only thing standing
between a NaN logit and a silently degenerate token, and nothing exercised it.
"""

import pytest
import torch

from flashinfer.sampling import (
    _validate_and_convert_seed_offset,
    min_p_sampling_from_probs,
    sampling_from_logits,
    sampling_from_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_logits,
    top_k_top_p_sampling_from_probs,
    top_p_sampling_from_probs,
)

_CPU = torch.device("cpu")


def _seeds(**over):
    args = dict(seed=1, offset=0, device=_CPU, batch_size=4)
    args.update(over)
    return args


class TestSeedOffsetValidation:
    def test_one_tensor_and_one_scalar_is_rejected(self):
        with pytest.raises(ValueError, match="both be tensors or both be scalars"):
            _validate_and_convert_seed_offset(
                **_seeds(seed=torch.zeros(1, dtype=torch.int64), offset=0)
            )

    def test_scalars_pass_through_unchanged(self):
        assert _validate_and_convert_seed_offset(**_seeds(seed=7, offset=3)) == (
            None,
            7,
            None,
            3,
        )

    @pytest.mark.parametrize("which", ["seed", "offset"])
    def test_a_tensor_on_the_wrong_device_is_rejected(self, which):
        if not torch.cuda.is_available():
            pytest.skip("needs a second device to be on the wrong one")
        good = torch.zeros(1, dtype=torch.int64, device="cuda:0")
        bad = torch.zeros(1, dtype=torch.int64, device=_CPU)
        pair = {"seed": good, "offset": good, which: bad}

        with pytest.raises(ValueError, match=f"{which} tensor must be on"):
            _validate_and_convert_seed_offset(
                **_seeds(device=torch.device("cuda:0"), **pair)
            )

    @pytest.mark.parametrize("which", ["seed", "offset"])
    def test_a_tensor_of_the_wrong_dtype_is_rejected(self, which):
        good = torch.zeros(1, dtype=torch.int64)
        bad = torch.zeros(1, dtype=torch.int32)
        pair = {"seed": good, "offset": good, which: bad}

        with pytest.raises(ValueError, match=f"{which} tensor must be int64/uint64"):
            _validate_and_convert_seed_offset(**_seeds(**pair))

    @pytest.mark.parametrize("which", ["seed", "offset"])
    def test_a_tensor_that_is_not_1d_is_rejected(self, which):
        good = torch.zeros(1, dtype=torch.int64)
        bad = torch.zeros((1, 1), dtype=torch.int64)
        pair = {"seed": good, "offset": good, which: bad}

        with pytest.raises(ValueError, match=f"{which} tensor must be 1D"):
            _validate_and_convert_seed_offset(**_seeds(**pair))

    @pytest.mark.parametrize("which", ["seed", "offset"])
    def test_a_tensor_of_the_wrong_length_is_rejected(self, which):
        good = torch.zeros(1, dtype=torch.int64)
        bad = torch.zeros(3, dtype=torch.int64)  # batch_size is 4
        pair = {"seed": good, "offset": good, which: bad}

        with pytest.raises(ValueError, match=f"{which} tensor length must be 1 or 4"):
            _validate_and_convert_seed_offset(**_seeds(**pair))

    def test_a_per_request_tensor_of_the_batch_length_is_accepted(self):
        seed = torch.arange(4, dtype=torch.int64)
        offset = torch.zeros(4, dtype=torch.int64)
        arr_seed, seed_val, arr_offset, offset_val = _validate_and_convert_seed_offset(
            **_seeds(seed=seed, offset=offset)
        )
        assert arr_seed is seed and arr_offset is offset
        assert (seed_val, offset_val) == (0, 0)


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("the sampling entry points take device tensors")
    return torch.device("cuda:0")


def _nan_probs(device):
    probs = torch.full((2, 8), 0.125, device=device)
    probs[1, 3] = float("nan")
    return probs


class TestCheckNan:
    """The guard runs before the module loads, so no kernel is built here."""

    def test_sampling_from_logits_rejects_nan(self, device):
        with pytest.raises(ValueError, match="logits contains NaN"):
            sampling_from_logits(_nan_probs(device), check_nan=True)

    def test_sampling_from_probs_rejects_nan(self, device):
        with pytest.raises(ValueError, match="probs contains NaN"):
            sampling_from_probs(_nan_probs(device), check_nan=True)

    def test_top_p_rejects_nan(self, device):
        with pytest.raises(ValueError, match="probs contains NaN"):
            top_p_sampling_from_probs(_nan_probs(device), 0.9, check_nan=True)

    def test_top_k_rejects_nan(self, device):
        with pytest.raises(ValueError, match="probs contains NaN"):
            top_k_sampling_from_probs(_nan_probs(device), 4, check_nan=True)

    def test_min_p_rejects_nan(self, device):
        with pytest.raises(ValueError, match="probs contains NaN"):
            min_p_sampling_from_probs(_nan_probs(device), 0.1, check_nan=True)


_JOINT_ENTRY_POINTS = pytest.mark.parametrize(
    "entry_point",
    [top_k_top_p_sampling_from_logits, top_k_top_p_sampling_from_probs],
    ids=["from_logits", "from_probs"],
)


class TestFilterApplyOrder:
    """Only ``top_k_first`` and ``joint`` exist; anything else must not silently
    fall through to one of them."""

    @_JOINT_ENTRY_POINTS
    def test_an_unknown_order_is_rejected(self, device, entry_point):
        probs = torch.full((2, 8), 0.125, device=device)
        with pytest.raises(ValueError, match="Invalid filter_apply_order"):
            entry_point(probs, 4, 0.9, filter_apply_order="sideways")

    @_JOINT_ENTRY_POINTS
    def test_the_joint_arm_checks_nan_too(self, device, entry_point):
        with pytest.raises(ValueError, match="contains NaN"):
            entry_point(
                _nan_probs(device), 4, 0.9, filter_apply_order="joint", check_nan=True
            )
