# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Keeps non-redistributable upstream sources out of the release artifacts.

Parts of upstream carry the NVIDIA TensorRT Source Code License or
LicenseRef-NvidiaProprietary, both of which forbid redistribution. They are
excluded in MANIFEST.in (sdist) and pyproject.toml (wheel). Building an artifact
to check is far too slow for the suite, so coverage is read out of MANIFEST.in
itself -- a hand-kept list here would go green the moment someone appended to
it, without anything actually being excluded.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MARKERS = ("TensorRT Source Code License", "LicenseRef-NvidiaProprietary")

# These name the markers in prose rather than carrying the licence.
_SPELLS_THE_MARKERS = frozenset(
    {
        str(pathlib.Path(__file__).resolve().relative_to(_ROOT)),
        "MANIFEST.in",
        "pyproject.toml",
    }
)


def _manifest_exclusions() -> tuple[tuple[str, ...], frozenset[str]]:
    """(directory prefixes from `prune`, exact paths from `exclude`).

    Kept apart deliberately: matching an `exclude`d *file* with startswith would
    also cover a sibling differing only in suffix, which MANIFEST.in would not.
    """
    text = (_ROOT / "MANIFEST.in").read_text()
    prunes = tuple(f"{d}/" for d in re.findall(r"^prune\s+(\S+)\s*$", text, re.M))
    excludes = frozenset(re.findall(r"^exclude\s+(\S+)\s*$", text, re.M))
    return prunes, excludes


def _tracked_restricted_files() -> list[str]:
    """Every tracked file carrying a marker, whatever its extension.

    No suffix allowlist: .jinja, .inl and .cc all ship, and an allowlist is the
    kind of thing that quietly stops covering a newly added file type.
    """
    listing = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    hits = []
    for name in listing:
        if not name or name in _SPELLS_THE_MARKERS:
            continue
        try:
            text = (_ROOT / name).read_text(errors="ignore")
        except OSError:
            continue
        if any(m in text for m in _MARKERS):
            hits.append(name)
    return sorted(hits)


def test_every_restricted_file_is_excluded_from_the_sdist():
    """A new one arriving on an upstream sync fails here, not in a release."""
    prunes, excludes = _manifest_exclusions()
    stray = [
        f
        for f in _tracked_restricted_files()
        if not f.startswith(prunes) and f not in excludes
    ]
    assert not stray, (
        f"these carry a no-redistribution licence and no MANIFEST.in rule covers "
        f"them: {stray}. Add a `prune` (directory) or `exclude` (single file)."
    )


def test_restricted_files_under_the_package_are_excluded_from_the_wheel():
    """MANIFEST.in governs the sdist only; anything under flashinfer/ needs both."""
    pyproject = (_ROOT / "pyproject.toml").read_text()
    find = re.search(
        r"\[tool\.setuptools\.packages\.find\](.*?)^\[", pyproject, re.M | re.S
    )
    assert find, "[tool.setuptools.packages.find] is gone"
    patterns = re.findall(r'"([^"]+)"', find.group(1).split("exclude")[-1])

    for f in _tracked_restricted_files():
        if not f.startswith("flashinfer/"):
            continue
        pkg = str(pathlib.Path(f).parent).replace("/", ".")
        assert any(pkg.startswith(p.rstrip("*")) for p in patterns), (
            f"{f} ships in the wheel: packages.find defaults to namespaces=true, "
            f"so {pkg} is discovered even without an __init__.py, and no exclude "
            "pattern covers it"
        )


def _fires_on_this_fork(triggers: dict) -> bool:
    """A push/pull_request trigger with no `branches:` filter reaches our branch.

    The ones upstream gates on `main`, or on a schedule against the default
    branch, are left untouched on purpose -- editing a dormant upstream file
    buys a permanent conflict on every sync.
    """
    for event in ("push", "pull_request", "pull_request_target"):
        spec = triggers.get(event)
        if event in triggers and not (isinstance(spec, dict) and "branches" in spec):
            return True
    return False


def test_no_workflow_that_fires_here_runs_the_cuda_validator():
    """It is unsatisfiable on this fork -- its TOML reader chokes on our `dev = [`."""
    import yaml

    offenders = []
    for p in sorted((_ROOT / ".github" / "workflows").glob("*.yml")):
        text = p.read_text()
        if not re.search(r"^\s*run:.*validate_cuda_versions\.py", text, re.M):
            continue
        # YAML 1.1 reads a bare `on:` key as the boolean True.
        parsed = yaml.safe_load(text)
        if _fires_on_this_fork(parsed.get("on", parsed.get(True)) or {}):
            offenders.append(p.name)
    assert not offenders, f"workflows still running the CUDA validator: {offenders}"
