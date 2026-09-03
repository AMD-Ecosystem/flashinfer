# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the xdist worker/GPU mapping in the repo-root conftest.py.

GPU-free. The branch worth protecting is the one a well-provisioned runner
never takes: more workers than physical cards, where handing a worker its own
index names a device that does not exist. That value is harmless in the worker
itself -- HIP is already initialized by then -- and fatal in every subprocess
the worker spawns, which is what makes it hard to attribute.

The conftest is loaded by path, as in test_run_header.py: pytest owns
conftest import, and a second module object can be exercised without touching
the hooks pytest is actually running.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_CONFTEST = pathlib.Path(__file__).parents[2] / "conftest.py"


@pytest.fixture
def root_conftest(monkeypatch):
    """A private copy of the repo-root conftest.py, loaded with pinning inert.

    Clearing PYTEST_XDIST_WORKER first skips the module-level pinning block, so
    loading the module cannot rewrite this session's own HIP_VISIBLE_DEVICES.
    """
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    spec = importlib.util.spec_from_file_location("_root_conftest_isolated", _CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maps_through_the_supported_list(root_conftest):
    """Worker index is an offset into the supported cards, not a device id."""
    assert root_conftest._worker_gpu_index(1, (2, 5)) == 5


def test_more_workers_than_cards_wraps(root_conftest):
    """pytest -n 4 on a one-card host: share card 0, never invent cards 1-3."""
    assert [root_conftest._worker_gpu_index(i, (0,)) for i in range(4)] == [0, 0, 0, 0]


def test_no_supported_card_pins_nothing(root_conftest):
    """Nothing to pin to is not the same as pinning to device 0."""
    assert root_conftest._worker_gpu_index(0, ()) is None


@pytest.mark.parametrize("supported", [(0,), (0, 1), (0, 2, 4, 6), (1, 3)])
def test_never_names_an_unsupported_card(root_conftest, supported):
    for worker_idx in range(16):
        assert root_conftest._worker_gpu_index(worker_idx, supported) in supported


def test_hook_lives_in_the_root_conftest(root_conftest):
    """Where the hook lives is the behaviour: xdist resolves '-n auto' from the
    initial conftests, so a subdirectory one yields the CPU count instead."""
    assert hasattr(root_conftest, "pytest_xdist_auto_num_workers")


@pytest.mark.parametrize("cards, expected", [(8, 4), (4, 2), (2, 1), (1, 1)])
def test_worker_count_halves_the_cards(root_conftest, monkeypatch, cards, expected):
    """Half the physical cards, and never zero -- one card still runs."""
    monkeypatch.setattr(
        "flashinfer.rocm.hip_utils.get_physical_card_device_indices",
        lambda: tuple(range(cards)),
    )
    assert root_conftest.pytest_xdist_auto_num_workers(config=None) == expected


def test_worker_count_falls_back_to_torch(root_conftest, monkeypatch):
    """rocminfo reporting nothing degrades to torch, as the pinning does --
    it must not raise, or a GPU-free checkout cannot run GPU-free tests."""
    import torch

    monkeypatch.setattr(
        "flashinfer.rocm.hip_utils.get_physical_card_device_indices", tuple
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 6)
    assert root_conftest.pytest_xdist_auto_num_workers(config=None) == 3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert root_conftest.pytest_xdist_auto_num_workers(config=None) == 1
