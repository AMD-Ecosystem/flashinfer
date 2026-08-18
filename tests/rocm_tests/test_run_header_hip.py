# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ROCm run-descriptor header in tests/rocm_tests/conftest.py.

GPU-free: the branch worth protecting here is the one no GPU run ever takes --
a host with no supported card must say so, rather than reporting the same
"unknown" a broken probe would.

The conftest is loaded by path rather than imported as a module. pytest owns
conftest import, and there is no package to import it from; loading it here
gives a second, independent module object whose globals can be patched without
touching the hooks pytest is actually running.
"""

import importlib.util
import pathlib

import pytest

_CONFTEST = pathlib.Path(__file__).with_name("conftest.py")


@pytest.fixture
def rocm_conftest(monkeypatch):
    """A private copy of the conftest module, with HIP_VISIBLE_DEVICES cleared."""
    spec = importlib.util.spec_from_file_location("_rocm_conftest_isolated", _CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    return module


class TestDevices:
    def test_visible_devices_env_wins(self, rocm_conftest, monkeypatch):
        """tests/conftest.py pins each xdist worker to one card through this."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
        monkeypatch.setattr(
            rocm_conftest, "get_physical_card_device_indices", lambda: (0, 1)
        )
        assert rocm_conftest._devices() == "3"

    def test_falls_back_to_detected_cards(self, rocm_conftest, monkeypatch):
        monkeypatch.setattr(
            rocm_conftest, "get_physical_card_device_indices", lambda: (0, 1)
        )
        assert rocm_conftest._devices() == "0,1"

    def test_no_supported_card_is_none_not_unknown(self, rocm_conftest, monkeypatch):
        """An empty device list is an answer; "unknown" would read as a failure."""
        monkeypatch.setattr(rocm_conftest, "get_physical_card_device_indices", tuple)
        assert rocm_conftest._devices() == "none"
        assert rocm_conftest._probe(rocm_conftest._devices) == "none"

    def test_broken_probe_is_unknown(self, rocm_conftest, monkeypatch):
        def boom():
            raise RuntimeError("rocminfo not on PATH")

        monkeypatch.setattr(rocm_conftest, "get_physical_card_device_indices", boom)
        assert rocm_conftest._probe(rocm_conftest._devices) == "unknown"


def test_header_line_survives_every_probe_failing(rocm_conftest, monkeypatch):
    """A descriptive header must never be the reason a test session dies."""
    for name in (
        "get_physical_card_device_indices",
        "get_system_rocm_version",
        "rocminfo_gpu_agents",
    ):
        monkeypatch.setattr(
            rocm_conftest, name, lambda: (_ for _ in ()).throw(RuntimeError("no ROCm"))
        )

    line = rocm_conftest._header_line()
    assert line.startswith("rocm: ")
    assert "arch=" in line and "sku=" in line and "devices=" in line
