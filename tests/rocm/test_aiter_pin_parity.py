# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every place that names the validated amd-aiter tag names the same one.

A bump that misses one leaves the image, the loader's error text and the install
docs disagreeing about which AITER was validated. Text only -- no GPU and no
torch, so CI runs this file with ``--noconftest``.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _validated() -> str:
    text = (_ROOT / "flashinfer" / "rocm" / "prefill.py").read_text()
    found = re.findall(r'^_AITER_LAST_VALIDATED = "([^"]+)"$', text, re.M)
    assert len(found) == 1, "prefill.py no longer spells _AITER_LAST_VALIDATED once"
    return found[0]


def test_the_dockerfile_builds_the_validated_tag():
    text = (_ROOT / "docker" / "Dockerfile.rocm").read_text()
    assert re.findall(r"^ARG AITER_REF=(\S+)$", text, re.M) == [f"v{_validated()}"]


@pytest.mark.parametrize(
    "path, pattern",
    [
        ("csrc/rocm/aiter_loader.cc", r"amd-aiter (\d+(?:\.\w+)*)"),
        ("csrc/rocm/aiter_loader.cc", r"at tag v(\d+(?:\.\w+)*)"),
        (
            "docs/rocm/backends.md",
            r"--branch v(\d+(?:\.\w+)*) https://github\.com/ROCm/aiter",
        ),
        ("README.md", r"\| `amd-aiter` \| (\d+(?:\.\w+)*) \(source build\)"),
        ("CLAUDE.md", r"^`v(\d+(?:\.\w+)*)` and installs it non-editable"),
        (
            ".claude/skills/debug-rocm-crash/SKILL.md",
            r"source build at tag `v(\d+(?:\.\w+)*)`",
        ),
    ],
)
def test_the_docs_and_loader_name_the_validated_tag(path, pattern):
    found = re.findall(pattern, (_ROOT / path).read_text(), re.M)
    assert found, f"{path} no longer matches {pattern!r}; is this test stale?"
    assert set(found) == {_validated()}, f"{path} names {sorted(set(found))}"
