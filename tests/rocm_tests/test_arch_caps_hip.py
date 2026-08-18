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
