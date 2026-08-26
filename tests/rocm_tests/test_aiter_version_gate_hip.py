# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The amd-aiter ABI floor.

The vendored structs under include/flashinfer/attention/aiter/ follow the 0.1.16
layout and travel by value through dlsym'd pointers, so an older AITER shifts
field offsets instead of failing to load. Nothing downstream can detect that,
which is why the floor is enforced before routing rather than at the call.
"""

import pathlib

import pytest

from flashinfer import aiter_utils


@pytest.mark.parametrize(
    "version,supported",
    [
        ("0.1.9", False),
        ("0.1.10", False),  # lexically ">= 0.1.16"; the pin this repo shipped before
        ("0.1.15.post9", False),
        ("0.1.16", True),
        # The nightly wheel: PEP 440 sorts a .dev0 segment *below* its release,
        # so a naive Version(...) >= Version(floor) rejects the only cp314 wheel.
        ("0.1.16.post3.dev0+g620287969.d20260725", True),
        ("0.1.21", True),
        ("0.2.0", True),
    ],
)
def test_version_floor(monkeypatch, version, supported):
    monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: version)
    assert aiter_utils._aiter_version_supported() is supported


def test_absent_aiter_is_not_a_version_failure(monkeypatch):
    """A missing package must not be reported as an out-of-date one."""
    monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: None)
    assert aiter_utils._aiter_version_supported() is False


def test_header_records_the_same_floor():
    """The vendored header's stated minimum must track AITER_MIN_VERSION."""
    header = (
        pathlib.Path(__file__).resolve().parents[2]
        / "include/flashinfer/attention/aiter/mha_fwd_args.h"
    )
    text = header.read_text()
    assert f"amd-aiter>={aiter_utils.AITER_MIN_VERSION}" in text
