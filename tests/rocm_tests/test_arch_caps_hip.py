# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for flashinfer.arch_caps.

Deliberately GPU-free: every assertion here holds on any machine, so this file
protects the architectures we cannot physically test against as well as the one
we can.
"""

import pathlib
import subprocess
import sys

import pytest

from flashinfer import arch_caps
from flashinfer.arch_caps import normalize_arch


class TestNormalizeArch:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # The forms torch's gcnArchName actually produces.
            ("gfx942:sramecc+:xnack-", "gfx942"),
            ("gfx950:sramecc+:xnack-", "gfx950"),
            # Already-normalized input must round-trip, so the helper is safe to
            # apply twice.
            ("gfx942", "gfx942"),
            ("gfx950", "gfx950"),
            # A single qualifier, and one with no value.
            ("gfx942:xnack-", "gfx942"),
            ("gfx90a:sramecc+", "gfx90a"),
        ],
    )
    def test_strips_qualifiers(self, raw, expected):
        assert normalize_arch(raw) == expected

    @pytest.mark.parametrize("arch", ["gfx90a", "gfx1201", "gfx1100"])
    def test_preserves_letter_suffixes_and_four_digit_archs(self, arch):
        """Regression: the previous ``re.match(r"(gfx\\d+)")`` form truncated
        ``gfx90a`` to ``gfx90``, naming an architecture that does not exist."""
        assert normalize_arch(arch) == arch

    def test_strips_surrounding_whitespace(self):
        assert normalize_arch("  gfx942  ") == "gfx942"

    @pytest.mark.parametrize("raw", ["", "   ", ":"])
    def test_degenerate_input_does_not_raise(self, raw):
        """Callers gate on the result rather than on an exception, so empty
        input must come back empty instead of blowing up."""
        assert normalize_arch(raw) == ""


def test_module_does_not_import_torch():
    """arch_caps.py must not import torch.

    hip_utils imports this module at module scope, and hip_utils is imported
    early -- tests/conftest.py uses it to choose a GPU *before* pinning
    HIP_VISIBLE_DEVICES. Keeping this module torch-free is what lets it sit on
    that path without dragging torch into it.

    The module is loaded directly from its file rather than as
    ``flashinfer.arch_caps``, because importing anything from the package runs
    flashinfer/__init__.py, which pulls in torch via device_utils. Going through
    the package would therefore prove nothing about this module. Run in a
    subprocess so the result does not depend on what this session imported.
    """
    module_path = pathlib.Path(arch_caps.__file__).resolve()
    code = f"""
import importlib.util, sys
assert "torch" not in sys.modules, "torch preloaded; test would be meaningless"
spec = importlib.util.spec_from_file_location("_arch_caps_isolated", r"{module_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.normalize_arch("gfx950:sramecc+:xnack-") == "gfx950"
assert "torch" not in sys.modules, "arch_caps.py imported torch"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Capability table
#
# Everything below is GPU-free by construction: the arch is monkeypatched.
# That is the point -- it lets the architecture we cannot physically reach be
# tested on whichever one we happen to have. Historically that meant simulating
# gfx950 on a CDNA3 box; today it is the reverse.
# --------------------------------------------------------------------------

SUPPORTED_ARCHS = ("gfx942", "gfx950")


@pytest.fixture
def as_arch(monkeypatch):
    """Pretend the running device is a given architecture."""

    def _set(arch):
        monkeypatch.setattr(arch_caps, "_device_arch", lambda _device=None: arch)

    return _set


@pytest.fixture
def as_toolchain(monkeypatch):
    """Pretend a given (rocm, aiter) pair is installed."""

    def _set(rocm, aiter="0.1.10"):
        monkeypatch.setattr(arch_caps, "_live_versions", lambda: (rocm, aiter))

    return _set


class TestTableWellFormed:
    def test_keys_are_unique(self):
        keys = [(c.op, c.backend) for c in arch_caps.CAPABILITIES]
        assert len(keys) == len(set(keys)), "duplicate (op, backend) row"

    def test_backends_are_known(self):
        assert {c.backend for c in arch_caps.CAPABILITIES} == {"aiter", "hip"}

    def test_every_supported_arch_declared_in_every_row(self):
        """The guard rail: adding an arch to FLASHINFER_SUPPORTED_ROCM_ARCHS
        must fail here until each op declares it, rather than silently
        inheriting support."""
        from flashinfer.hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

        for cap in arch_caps.CAPABILITIES:
            missing = set(FLASHINFER_SUPPORTED_ROCM_ARCHS) - set(cap.archs)
            assert not missing, f"{cap.op}/{cap.backend} does not declare {missing}"

    def test_no_arch_keys_outside_the_supported_list(self):
        from flashinfer.hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

        for cap in arch_caps.CAPABILITIES:
            extra = set(cap.archs) - set(FLASHINFER_SUPPORTED_ROCM_ARCHS)
            assert not extra, f"{cap.op}/{cap.backend} declares unknown {extra}"

    def test_known_bad_rows_explain_themselves(self):
        """A gate with no detail is unactionable for whoever hits it."""
        for cap in arch_caps.CAPABILITIES:
            for arch, entry in cap.archs.items():
                for bad in entry.known_bad:
                    assert bad.detail, f"{cap.op}/{cap.backend}/{arch}: empty detail"


class TestVersionWindow:
    @pytest.mark.parametrize(
        "rocm,expected",
        [
            ("7.1", False),
            ("7.1.1", False),
            ("7.2", True),
            ("7.2.0", True),
            ("7.2.4", True),  # measured: bit-identical failure to 7.2.0
            ("7.3", False),
            ("7.14", False),  # (7,14) > (7,3): a later release, not 7.1.4
        ],
    )
    def test_rocm_window_is_half_open(self, rocm, expected):
        bad = arch_caps.KnownBad(rocm_min="7.2", rocm_max="7.3")
        assert bad.matches(rocm, None) is expected

    def test_unknown_version_does_not_match(self):
        """Refusing to route because a version could not be read would break
        machines that are probably fine."""
        bad = arch_caps.KnownBad(rocm_min="7.2", rocm_max="7.3")
        assert bad.matches(None, None) is False


class TestGating:
    @pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
    @pytest.mark.parametrize("backend", ["aiter", "hip"])
    def test_declared_rows_are_routable_on_a_clean_toolchain(
        self, as_arch, as_toolchain, arch, backend
    ):
        as_arch(arch)
        as_toolchain("7.1")  # outside every known_bad window
        for cap in arch_caps.CAPABILITIES:
            if cap.backend != backend:
                continue
            assert arch_caps.capability_available(cap.op, cap.backend, None), (
                f"{cap.op}/{cap.backend} unexpectedly gated on {arch}"
            )

    def test_undeclared_arch_is_refused(self, as_arch):
        """An arch nobody declared grants nothing, even for a real op."""
        as_arch("gfx90a")
        assert not arch_caps.capability_available("rmsnorm", "aiter", None)
        with pytest.raises(arch_caps.ArchCapabilityError, match="gfx90a"):
            arch_caps.require_capability("rmsnorm", "aiter", None)

    def test_unknown_op_is_refused(self, as_arch):
        as_arch("gfx950")
        with pytest.raises(arch_caps.ArchCapabilityError, match="not declared"):
            arch_caps.require_capability("no_such_op", "aiter", None)


class TestRocm72CausalPrefill:
    """The one measured defect: gfx950 + ROCm 7.2.x miscompiles AITER causal
    batch prefill. gfx942 is fine on the same toolchain."""

    def test_gated_on_gfx950_under_rocm_72(self, as_arch, as_toolchain):
        as_arch("gfx950")
        as_toolchain("7.2.0")
        assert not arch_caps.capability_available("batch_prefill", "aiter", None)
        with pytest.raises(arch_caps.ArchCapabilityError, match="known-broken"):
            arch_caps.require_capability("batch_prefill", "aiter", None)

    def test_still_gated_on_the_latest_affected_patch(self, as_arch, as_toolchain):
        as_arch("gfx950")
        as_toolchain("7.2.4")
        assert not arch_caps.capability_available("batch_prefill", "aiter", None)

    def test_open_on_gfx950_under_rocm_71(self, as_arch, as_toolchain):
        """Measured clean: max_abs_err 0.000250, 12/12 parametrizations pass."""
        as_arch("gfx950")
        as_toolchain("7.1")
        assert arch_caps.capability_available("batch_prefill", "aiter", None)

    def test_gfx942_unaffected_on_the_same_toolchain(self, as_arch, as_toolchain):
        """This is the whole point of keying on arch as well as version."""
        as_arch("gfx942")
        as_toolchain("7.2.0")
        assert arch_caps.capability_available("batch_prefill", "aiter", None)

    def test_hip_fallback_stays_open_where_aiter_is_gated(self, as_arch, as_toolchain):
        """The gate is only useful if `auto` has somewhere correct to fall back
        to -- fa2 was measured correct on the same hardware (2.6e-4)."""
        as_arch("gfx950")
        as_toolchain("7.2.0")
        assert arch_caps.capability_available("batch_prefill", "hip", None)

    def test_escape_hatch_opts_in_to_danger(self, as_arch, as_toolchain, monkeypatch):
        """Opt in to the broken path, never opt in to safety."""
        as_arch("gfx950")
        as_toolchain("7.2.0")
        monkeypatch.setenv("FLASHINFER_ARCH_ALLOW_KNOWN_BAD", "1")
        assert arch_caps.capability_available("batch_prefill", "aiter", None)


class TestArchCapabilityError:
    def test_satisfies_every_existing_catcher(self):
        """Routing three divergent exception types through one class only works
        if the old ones still catch it.

        ValueError: test_activation_aiter_hip.py:67 asserts on it.
        RuntimeError: test_batch_prefill_bf16_custom_mask_hip.py:157 catches it.
        """
        err = arch_caps.ArchCapabilityError("boom")
        assert isinstance(err, ValueError)
        assert isinstance(err, RuntimeError)

    def test_is_not_an_import_error(self):
        """A missing aiter package is a different condition and keeps its own
        exception type."""
        assert not isinstance(arch_caps.ArchCapabilityError("x"), ImportError)
