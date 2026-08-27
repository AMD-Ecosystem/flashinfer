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

from flashinfer import aiter_utils, arch_caps, device_utils


class TestBackendReporting:
    """Three functions branch on the same pair of flags; all six arms matter,
    because the answer ends up in bug reports and the run header."""

    @pytest.fixture
    def as_backend(self, monkeypatch):
        def _set(is_hip, is_cuda):
            monkeypatch.setattr(device_utils, "IS_HIP", is_hip)
            monkeypatch.setattr(device_utils, "IS_CUDA", is_cuda)

        return _set

    @pytest.mark.parametrize(
        "is_hip, is_cuda, expected",
        [(True, False, "hip"), (False, True, "cuda"), (False, False, "cpu")],
    )
    def test_backend_id(self, as_backend, is_hip, is_cuda, expected):
        as_backend(is_hip, is_cuda)
        assert device_utils.get_device_backend() == expected

    @pytest.mark.parametrize(
        "is_hip, is_cuda, expected",
        [(True, False, "ROCm/HIP"), (False, True, "CUDA"), (False, False, "CPU")],
    )
    def test_human_readable_name(self, as_backend, is_hip, is_cuda, expected):
        as_backend(is_hip, is_cuda)
        assert device_utils.get_backend_name() == expected

    def test_version_comes_from_the_matching_torch_attribute(self, as_backend):
        as_backend(True, False)
        assert device_utils.get_backend_version() == torch.version.hip

        as_backend(False, True)
        assert device_utils.get_backend_version() == torch.version.cuda

    def test_no_backend_has_no_version(self, as_backend):
        as_backend(False, False)
        assert device_utils.get_backend_version() is None


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
        import flashinfer.hip_utils as hip_utils

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
    def test_a_non_hip_torch_is_never_an_aiter_target(self, monkeypatch):
        aiter_utils.is_aiter_supported.cache_clear()
        monkeypatch.setattr(torch.version, "hip", None)

        assert aiter_utils.is_aiter_supported(torch.device("cuda:0")) is False

        aiter_utils.is_aiter_supported.cache_clear()

    def test_an_unreadable_device_is_not_an_aiter_target(self, monkeypatch):
        """get_device_properties raises for an index that does not exist; that is
        an answer, not a crash."""
        aiter_utils.is_aiter_supported.cache_clear()

        assert aiter_utils.is_aiter_supported(torch.device("cuda:999")) is False

        aiter_utils.is_aiter_supported.cache_clear()

    def test_a_missing_package_makes_the_probe_report_unavailable(self, monkeypatch):
        aiter_utils._aiter_importable.cache_clear()
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

        aiter_utils._aiter_importable.cache_clear()

    def test_explicit_aiter_without_the_package_names_the_alternative(
        self, monkeypatch
    ):
        """An explicit backend='aiter' must not silently demote; it explains."""
        monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: False)
        monkeypatch.setattr(aiter_utils, "require_capability", lambda *a: None)

        with pytest.raises(ValueError, match="backend='native'"):
            aiter_utils.require_aiter(torch.device("cuda:0"), "rmsnorm")


def test_the_aiter_mha_module_is_resolved_once_and_cached():
    """The shim imports aiter.ops.mha lazily; a typo there surfaces only here."""
    pytest.importorskip("aiter.ops.mha")

    first = aiter_utils.get_aiter_mha_module()

    assert first is aiter_utils.get_aiter_mha_module()
    assert hasattr(first, "__name__")
