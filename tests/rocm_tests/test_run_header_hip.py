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
import sys
import types

import pytest

_CONFTEST = pathlib.Path(__file__).with_name("conftest.py")


def _stub_torch(gcn_arch_name: str):
    """A torch stand-in exposing only what _arch_and_sku reads from it.

    The real torch reports whichever card the runner has, which would decide
    these cases for them.
    """
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda _: types.SimpleNamespace(
                gcnArchName=gcn_arch_name
            ),
        )
    )


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


class TestArchAndSku:
    """SKU resolution from rocminfo agents.

    torch is not available to these tests as a source of truth for the arch, and
    forcing it would tie them to whatever card the runner has. Each case pins the
    agent list instead and lets the arch fall out of it, which is the same path a
    host with no torch takes.
    """

    @pytest.fixture
    def resolve(self, rocm_conftest, monkeypatch):
        def _resolve(agents, torch_arch=None):
            monkeypatch.setattr(rocm_conftest, "rocminfo_gpu_agents", lambda: agents)
            if torch_arch is None:
                # sys.modules[name] = None is what makes "import torch" raise,
                # which is the path a host without torch takes.
                monkeypatch.setitem(sys.modules, "torch", None)
            else:
                monkeypatch.setitem(sys.modules, "torch", _stub_torch(torch_arch))
            return rocm_conftest._arch_and_sku()

        return _resolve

    def test_single_card(self, resolve):
        assert resolve((("gfx942", "AMD Instinct MI300X"),)) == ("gfx942", "MI300X")

    def test_homogeneous_multi_card(self, resolve):
        """The common case: eight identical boards must still name the SKU."""
        agents = (("gfx950", "AMD Instinct MI355X"),) * 8
        assert resolve(agents) == ("gfx950", "MI355X")

    def test_mixed_arch_picks_the_matching_board(self, resolve):
        """A card of another arch must not lend its name to this one."""
        agents = (("gfx942", "AMD Instinct MI300X"), ("gfx90a", "AMD Instinct MI250X"))
        assert resolve(agents) == ("gfx942", "MI300X")

    def test_torch_arch_selects_the_agent_not_agent_order(self, resolve):
        """The production path: torch names the arch this session is pinned to,
        and the SKU follows that -- not whichever agent rocminfo listed first."""
        agents = (("gfx942", "AMD Instinct MI300X"), ("gfx90a", "AMD Instinct MI250X"))
        assert resolve(agents, torch_arch="gfx90a:sramecc+:xnack-") == (
            "gfx90a",
            "MI250X",
        )

    def test_same_arch_different_boards_is_unknown(self, resolve):
        """MI300X and MI325X are both gfx942, and rocminfo ignores
        HIP_VISIBLE_DEVICES -- so which board this session got is unknowable,
        and guessing one would read as an answer."""
        agents = (("gfx942", "AMD Instinct MI300X"), ("gfx942", "AMD Instinct MI325X"))
        assert resolve(agents) == ("gfx942", "unknown")

    def test_generic_marketing_name_is_unknown(self, resolve):
        """Seen in this repo's own ROCm container for an Instinct part."""
        assert resolve((("gfx950", "AMD Radeon Graphics"),)) == ("gfx950", "unknown")

    def test_no_agents_at_all(self, resolve):
        assert resolve(()) == ("unknown", "unknown")

    @pytest.mark.parametrize("degenerate", ["", "   ", ":"])
    def test_unusable_torch_arch_falls_back_to_rocminfo(self, resolve, degenerate):
        """normalize_arch returns "" for degenerate input rather than raising,
        and "" is not _UNKNOWN -- so an unfolded empty arch would skip the
        rocminfo fallback and print a blank arch in the header."""
        agents = (("gfx942", "AMD Instinct MI300X"),)
        assert resolve(agents, torch_arch=degenerate) == ("gfx942", "MI300X")


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
