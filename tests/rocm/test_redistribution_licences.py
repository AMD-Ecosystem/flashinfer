# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Keeps non-redistributable upstream sources out of the release artifacts.

Parts of upstream carry the NVIDIA TensorRT Source Code License or
LicenseRef-NvidiaProprietary, both of which forbid redistribution. They are
excluded in pyproject.toml (wheel) and MANIFEST.in (sdist). Building an artifact
to check that is far too slow for the suite, so this reads the exclusions
instead -- the point is that a *new* such file cannot arrive unnoticed, which is
exactly what v0.6.18 did.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MARKERS = ("TensorRT Source Code License", "LicenseRef-NvidiaProprietary")
_SUFFIXES = {".h", ".hpp", ".cuh", ".cu", ".cpp", ".py"}

# Every path below is excluded from both artifacts. Adding an entry means adding
# the matching prune/exclude too -- see test_every_restricted_file_is_excluded.
_EXCLUDED_PREFIXES = (
    "csrc/fmha_v2/",
    "flashinfer/jit/attention/fmha_v2/",
    "csrc/cudnn_sdpa_utils.h",
)


def _tracked_restricted_files() -> list[str]:
    """git-tracked sources carrying a marker. Tracked, because the sdist's file
    finder is setuptools-scm's, which only ever sees tracked files."""
    listing = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    hits = []
    for name in listing:
        if not name or pathlib.Path(name).suffix not in _SUFFIXES:
            continue
        try:
            text = (_ROOT / name).read_text(errors="ignore")
        except OSError:
            continue
        if any(m in text for m in _MARKERS):
            hits.append(name)
    return sorted(hits)


def test_every_restricted_file_is_excluded():
    """A new one appearing on an upstream sync fails here, not in a release."""
    stray = [
        f for f in _tracked_restricted_files() if not f.startswith(_EXCLUDED_PREFIXES)
    ]
    assert not stray, (
        "these carry a no-redistribution licence and are in neither exclusion "
        f"list: {stray}. Add a prune to MANIFEST.in (sdist) and, if under "
        "flashinfer/, an exclude to [tool.setuptools.packages.find] (wheel)."
    )


def test_the_exclusions_are_still_configured():
    """Guards the config itself: the artifact test cannot see these.

    test_build_backend.py builds in a non-git tree, so setuptools-scm's file
    finder returns nothing there and the wheel never contains these paths
    regardless -- it would pass with the exclusions deleted.
    """
    manifest = (_ROOT / "MANIFEST.in").read_text()
    assert re.search(r"^prune csrc/fmha_v2$", manifest, re.M)
    assert re.search(r"^prune flashinfer/jit/attention/fmha_v2$", manifest, re.M)
    assert re.search(r"^exclude csrc/cudnn_sdpa_utils\.h$", manifest, re.M)

    pyproject = (_ROOT / "pyproject.toml").read_text()
    assert '"flashinfer.jit.attention.fmha_v2*"' in pyproject, (
        "the wheel exclude is gone; packages.find defaults to namespaces=true, "
        "so the directory ships as a namespace package without it"
    )


@pytest.mark.parametrize("workflow", ["pre-commit.yml", "release.yml"])
def test_the_cuda_validator_is_not_run_on_this_fork(workflow):
    """It is unsatisfiable here, and both workflows fire on PRs to this branch."""
    text = (_ROOT / ".github" / "workflows" / workflow).read_text()
    assert "run: python3 ci/validate_cuda_versions.py" not in text
