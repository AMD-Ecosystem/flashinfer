# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The amd-aiter ABI floor.

The vendored structs under include/flashinfer/rocm/attention/aiter/ follow the 0.1.20
layout and travel by value through dlsym'd pointers, so an older AITER shifts
field offsets instead of failing to load. Nothing downstream can detect that,
which is why the floor is enforced before routing rather than at the call.
"""

import pathlib
import re

import pytest

from flashinfer import aiter_utils

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOC_REF = re.compile(r"docs/[\w./-]+\.md")


def _doc_references():
    for src in sorted(
        (*_REPO_ROOT.glob("flashinfer/**/*.py"), *_REPO_ROOT.glob("scripts/*.py"))
    ):
        for ref in dict.fromkeys(_DOC_REF.findall(src.read_text(encoding="utf-8"))):
            yield src.relative_to(_REPO_ROOT), ref


@pytest.mark.parametrize(
    "version,supported",
    [
        # "0.1.9" > "0.1.16" as strings, so this row is the one that catches a
        # comparison done on base_version without re-wrapping it in Version.
        ("0.1.9", False),
        ("0.1.10", False),  # the pin this repo shipped before the 0.1.16 floor
        ("0.1.15.post9", False),
        # A pre-release of the floor is below it. Comparing on base_version
        # strips the segment and wrongly admits this one.
        ("0.1.16.dev0", False),
        # 0.1.16 declared rmsnorm without gemma_norm, so its mangled name differs.
        ("0.1.16", False),
        ("0.1.20.dev0", False),
        ("0.1.20", True),
        # A .dev0 *of post3* sorts above 0.1.16, which is what makes the
        # nightly-vs-prerelease distinction non-obvious -- still below 0.1.20.
        ("0.1.16.post3.dev0+g620287969.d20260725", False),
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


def test_every_doc_path_named_in_the_source_resolves():
    """Error messages send users to these paths, so a rename that misses one
    ships a dead pointer to whoever is already stuck. Needs no GPU."""
    refs = list(_doc_references())

    assert refs, "no doc references found -- the scan is not looking where it should"
    missing = [f"{src}: {ref}" for src, ref in refs if not (_REPO_ROOT / ref).is_file()]
    assert not missing, "doc paths named in source but absent: " + ", ".join(missing)


def test_header_records_the_same_floor():
    """The vendored header's stated minimum must track AITER_MIN_VERSION."""
    header = (
        pathlib.Path(__file__).resolve().parents[2]
        / "include/flashinfer/rocm/attention/aiter/mha_fwd_args.h"
    )
    text = header.read_text()
    assert f"amd-aiter>={aiter_utils.AITER_MIN_VERSION}" in text
