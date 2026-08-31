# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Branches in the small ROCm helper modules that a gfx942 run never reaches.

Backend reporting, version parsing and the AITER availability gate each carry
arms for a machine this one is not -- a CUDA build, a CPU-only build, an
unreadable toolchain version, a missing package. They are reached here by
rebinding the module-level flags rather than by pretending to be that machine.

No GPU is required except where a test says otherwise.
"""

import pytest
import torch

from flashinfer.rocm import aiter_utils, arch_caps


class TestVersionTuple:
    """Version strings arrive with packaging suffixes; comparison has to stop at
    the first component it cannot read rather than guess or raise."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("7.2.0", (7, 2, 0)),
            ("7.2.0-76", (7, 2, 0)),
            ("0.1.10", (0, 1, 10)),
            ("7.2.0rc1", (7, 2, 0)),
            ("7.x.0", (7,)),
            ("", ()),
            ("rc1", ()),
        ],
    )
    def test_parsing_stops_at_the_first_unreadable_component(self, text, expected):
        assert arch_caps._version_tuple(text) == expected


class TestLiveVersions:
    def test_an_unreadable_toolchain_reports_none_rather_than_raising(
        self, monkeypatch
    ):
        """Both probes are best-effort: a missing rocm-core or amd-aiter must
        not stop capability lookup, it just means no known-bad window matches."""
        import importlib.metadata

        monkeypatch.setattr(
            arch_caps, "_live_versions", arch_caps._live_versions.__wrapped__
        )
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: (_ for _ in ()).throw(RuntimeError("no metadata")),
        )
        import flashinfer.rocm.hip_utils as hip_utils

        monkeypatch.setattr(
            hip_utils,
            "get_system_rocm_version",
            lambda: (_ for _ in ()).throw(RuntimeError("no rocm")),
        )

        assert arch_caps._live_versions() == (None, None)


class TestBlockingReason:
    def test_an_arch_absent_from_the_table_is_blocked_and_named(self):
        cap = arch_caps.CAPABILITIES[0]
        reason = arch_caps._blocking_reason(cap.op, cap.backend, "gfx900")

        assert reason is not None
        assert "gfx900" in reason

    def test_the_suggested_fallback_is_included_when_the_row_declares_one(self):
        cap = next(c for c in arch_caps.CAPABILITIES if c.fallback)
        reason = arch_caps._blocking_reason(cap.op, cap.backend, "gfx900")

        assert f"backend={cap.fallback!r}" in reason

    def test_an_unknown_op_is_not_blocked_by_the_table(self):
        assert arch_caps._lookup("no_such_op", "aiter", "gfx942") is None


class TestAiterAvailabilityGate:
    @pytest.fixture(autouse=True)
    def _uncached(self):
        """Teardown, not a trailing statement: an assertion failure would
        otherwise strand a probe answer computed under a fake torch or a
        blocked import, and every later AITER test in the worker would skip."""
        for probe in (aiter_utils.is_aiter_supported, aiter_utils._aiter_importable):
            probe.cache_clear()
        yield
        for probe in (aiter_utils.is_aiter_supported, aiter_utils._aiter_importable):
            probe.cache_clear()

    def test_a_non_hip_torch_is_never_an_aiter_target(self, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)

        assert aiter_utils.is_aiter_supported(torch.device("cuda:0")) is False

    def test_an_unreadable_device_is_not_an_aiter_target(self, monkeypatch):
        """get_device_properties raises for an index that does not exist; that is
        an answer, not a crash."""
        assert aiter_utils.is_aiter_supported(torch.device("cuda:999")) is False

    def test_a_missing_package_makes_the_probe_report_unavailable(self, monkeypatch):
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _no_aiter(name, *args, **kwargs):
            if name.split(".")[0] in ("aiter", "aiter_meta"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_aiter)

        assert aiter_utils._aiter_importable() is False

    def test_explicit_aiter_without_the_package_names_the_alternative(
        self, monkeypatch
    ):
        """An explicit backend='aiter' must not silently demote; it explains."""
        monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: False)
        monkeypatch.setattr(aiter_utils, "require_capability", lambda *a: None)

        with pytest.raises(ValueError, match="backend='native'"):
            aiter_utils.require_aiter(torch.device("cuda:0"), "rmsnorm")
