# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fork's GitHub workflows are exactly the ones it owns.

An upstream sync merges a newly added upstream workflow without a conflict, so
nothing else notices it landing and running here. No GPU and no torch, so CI
runs this file with ``--noconftest``.
"""

from pathlib import Path

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

_OWNED = {
    "arch-caps-conformance.yml",
    "pr-security-scan.yml",
    "pre-commit.yml",
    "weekly-security-scan.yml",
}


def test_only_fork_owned_workflows_exist():
    found = {p.name for p in _WORKFLOWS.glob("*.y*ml")}
    assert found == _OWNED, (
        f"unexpected: {sorted(found - _OWNED)}, missing: {sorted(_OWNED - found)}"
        " -- git rm upstream workflows on sync (CONTRIBUTING.md), or add a new"
        " fork-owned one here"
    )
