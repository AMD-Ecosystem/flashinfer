# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ownership-classification behaviour for ``scripts/amd_coverage.py``.

The coverage number is only as honest as the tier assignment behind it, and a
misparsed hunk or a missed ``if IS_CUDA:`` guard shifts it silently in either
direction. These build throwaway git repositories and assert exact line sets.

Git and filesystem only -- no GPU, no torch, no ``coverage`` package -- but
collection still pulls in the suite's torch-importing conftest, so CI runs this
file with ``--noconftest`` (see ``.github/workflows/arch-caps-conformance.yml``).
"""

import importlib.util
import re
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "_fi_amd_coverage", _REPO_ROOT / "scripts" / "amd_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ac = _load_tool()


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )


@pytest.fixture
def repo(tmp_path):
    """A repo whose base commit stands in for the upstream merge base."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "flashinfer").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / ac._MANIFEST).write_text("", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _base(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestTierAssignment:
    def test_added_file_is_tier_a_and_scored_whole(self, repo):
        base = _base(repo)
        _write(repo, "flashinfer/new_rocm.py", "a = 1\nb = 2\n")
        _git(repo, "add", "-A")

        owned, _, unowned = ac.classify(str(repo), base, {})

        assert owned["flashinfer/new_rocm.py"].tier == "A"
        # None means "whole file"; scoring narrows it to statements.
        assert owned["flashinfer/new_rocm.py"].changed is None
        assert "flashinfer/new_rocm.py" not in unowned

    def test_modified_file_is_tier_b_with_only_our_lines(self, repo):
        _write(
            repo, "flashinfer/up.py", "\n".join(f"x{i} = {i}" for i in range(10)) + "\n"
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)

        lines = (repo / "flashinfer/up.py").read_text().splitlines()
        lines[4] = "x4 = 999"  # edit line 5
        lines.insert(7, "inserted = 1")  # new line 8
        _write(repo, "flashinfer/up.py", "\n".join(lines) + "\n")

        owned, _, _ = ac.classify(str(repo), base, {})
        entry = owned["flashinfer/up.py"]

        assert entry.tier == "B"
        assert entry.changed == {5, 8}

    def test_rename_is_tracked_not_dropped(self, repo):
        body = "\n".join(f"y{i} = {i}" for i in range(30)) + "\n"
        _write(repo, "flashinfer/old.py", body)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)

        _git(repo, "mv", "flashinfer/old.py", "flashinfer/new.py")
        _write(repo, "flashinfer/new.py", body.replace("y3 = 3", "y3 = 300"))
        _git(repo, "add", "-A")

        owned, _, unowned = ac.classify(str(repo), base, {})

        # Without an R row this file would land in unowned and stop being measured.
        assert "flashinfer/new.py" in owned
        assert owned["flashinfer/new.py"].tier == "B"
        assert owned["flashinfer/new.py"].changed == {4}
        assert "flashinfer/new.py" not in unowned

    def test_uncommitted_new_file_is_still_ours(self, repo):
        """`git ls-files` alone hides it, and it is exactly what someone is writing."""
        base = _base(repo)
        _write(repo, "flashinfer/wip_rocm.py", "a = 1\n")  # never `git add`ed

        owned, _, unowned = ac.classify(str(repo), base, {})

        assert owned["flashinfer/wip_rocm.py"].tier == "A"
        assert owned["flashinfer/wip_rocm.py"].reason == "untracked"

    def test_ignored_file_is_not_picked_up(self, repo):
        base = _base(repo)
        _write(repo, ".gitignore", "flashinfer/generated.py\n")
        _write(repo, "flashinfer/generated.py", "a = 1\n")

        owned, _, unowned = ac.classify(str(repo), base, {})

        assert "flashinfer/generated.py" not in owned
        assert "flashinfer/generated.py" not in unowned

    def test_untouched_upstream_file_is_unowned(self, repo):
        _write(repo, "flashinfer/untouched.py", "z = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)

        owned, _, unowned = ac.classify(str(repo), base, {})

        assert "flashinfer/untouched.py" not in owned
        assert "flashinfer/untouched.py" in unowned

    def test_every_file_lands_in_exactly_one_bucket(self, repo):
        base = _base(repo)
        _write(repo, "flashinfer/added.py", "a = 1\n")
        _write(repo, "flashinfer/redirected.py", "b = 1\n")
        _write(repo, "flashinfer/gen.py", "c = 1\n")
        _git(repo, "add", "-A")
        manifest = {
            "redirect_owned": [{"path": "flashinfer/redirected.py", "reason": "r"}],
            "excluded": [{"path": "flashinfer/gen.py", "reason": "e"}],
        }

        owned, excluded, unowned = ac.classify(str(repo), base, manifest)
        tracked = ac._surface_python(str(repo))[0]

        assert set(owned) | set(excluded) | set(unowned) == tracked
        assert not set(owned) & set(unowned)
        assert not set(owned) & set(excluded)
        assert owned["flashinfer/redirected.py"].tier == "C"


class TestManifest:
    def test_stale_entry_fails_rather_than_being_skipped(self, repo):
        base = _base(repo)
        manifest = {"redirect_owned": [{"path": "flashinfer/gone.py", "reason": "r"}]}

        with pytest.raises(ac.ToolError, match="does not exist"):
            ac.classify(str(repo), base, manifest)

    def test_contradictory_entry_is_rejected(self, repo):
        base = _base(repo)
        _write(repo, "flashinfer/both.py", "a = 1\n")
        _git(repo, "add", "-A")
        manifest = {
            "redirect_owned": [{"path": "flashinfer/both.py", "reason": "r"}],
            "unowned": [{"path": "flashinfer/both.py", "reason": "u"}],
        }

        with pytest.raises(ac.ToolError, match="both redirect_owned and unowned"):
            ac.classify(str(repo), base, manifest)

    def test_shipped_manifest_matches_the_tree(self):
        """Catches a manifest entry left behind by a rename or deletion."""
        repo = ac._git(None, "rev-parse", "--show-toplevel")
        manifest = ac._load_toml(Path(repo) / ac._MANIFEST)
        for key in ("redirect_owned", "unowned", "excluded"):
            for item in manifest.get(key, []):
                assert (Path(repo) / item["path"]).exists(), item["path"]
                assert item.get("reason"), f"{item['path']} has no reason"


class TestCudaGuardExclusion:
    """`exclude_also` in pyproject.toml does the excluding; these guard its regexes.

    coverage applies them, so what can break is drift: a guard written in a shape
    no pattern matches silently re-enters the denominator as permanently-missing
    lines, and the number drops for a reason nobody can see.
    """

    @staticmethod
    def _patterns():
        report = (
            ac._load_toml(_REPO_ROOT / "pyproject.toml")
            .get("tool", {})
            .get("coverage", {})
            .get("report", {})
        )
        return [re.compile(p) for p in report.get("exclude_also", []) if "IS_CUDA" in p]

    def test_every_guard_in_the_tree_matches_a_pattern(self):
        patterns = self._patterns()
        assert patterns, "no IS_CUDA exclusion patterns configured"

        unmatched = []
        for path in sorted(_REPO_ROOT.glob("flashinfer/**/*.py")):
            for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if not re.fullmatch(r"(el)?if IS_CUDA:", stripped):
                    continue
                if not any(p.match(line) for p in patterns):
                    unmatched.append(f"{path.relative_to(_REPO_ROOT)}:{n}: {stripped}")

        assert not unmatched, "guards no exclusion pattern matches:\n" + "\n".join(
            unmatched
        )

    def test_patterns_do_not_match_a_compound_guard(self):
        """A compound condition can still be reachable; excluding it would hide code."""
        patterns = self._patterns()

        for line in ("if IS_CUDA and enabled:", "    if IS_CUDA or x:"):
            assert not any(p.match(line) for p in patterns), line

    def test_exclusion_stops_at_the_elif_is_hip_arm(self, tmp_path):
        """The shape of flashinfer/jit/env.py: if IS_CUDA / elif IS_HIP / else.

        If coverage excluded the whole chain rather than the one arm, it would
        hide the port's own HIP code -- the opposite of the point.
        """
        coverage = pytest.importorskip("coverage")
        cfg = _REPO_ROOT / "pyproject.toml"
        src = tmp_path / "m.py"
        src.write_text(
            "IS_CUDA = False\n"  # 1
            "IS_HIP = True\n"  # 2
            "if IS_CUDA:\n"  # 3
            "    CUDA_ONLY = 1\n"  # 4
            "elif IS_HIP:\n"  # 5
            "    HIP_ONE = 1\n"  # 6
            "else:\n"  # 7
            "    raise RuntimeError\n",  # 8
            encoding="utf-8",
        )

        cov = coverage.Coverage(config_file=str(cfg))
        _, statements, excluded, _, _ = cov.analysis2(str(src))

        assert set(excluded) == {3, 4}
        assert {5, 6}.issubset(set(statements)), "the IS_HIP arm must stay measurable"

    def test_the_tree_still_has_guards_to_exclude(self):
        """If this fails the exclusion is dead config, not a passing safety net."""
        guards = [
            path
            for path in _REPO_ROOT.glob("flashinfer/**/*.py")
            if re.search(
                r"^\s*(el)?if IS_CUDA:\s*$", path.read_text(encoding="utf-8"), re.M
            )
        ]

        assert guards, "no `if IS_CUDA:` guards found; is exclude_also still needed?"


class TestHunkParsing:
    def test_pure_deletion_contributes_no_lines(self, repo):
        _write(repo, "flashinfer/d.py", "a = 1\nb = 2\nc = 3\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)
        _write(repo, "flashinfer/d.py", "a = 1\nc = 3\n")

        assert ac._changed_lines(str(repo), base, "flashinfer/d.py", None) == set()

    def test_single_line_hunk_without_a_count(self, repo):
        _write(repo, "flashinfer/s.py", "a = 1\nb = 2\nc = 3\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)
        _write(repo, "flashinfer/s.py", "a = 1\nb = 22\nc = 3\n")

        assert ac._changed_lines(str(repo), base, "flashinfer/s.py", None) == {2}


class TestReportInputs:
    def test_junit_counts_separate_skips_from_failures(self, tmp_path):
        (tmp_path / "j.xml").write_text(
            '<testsuites><testsuite tests="10" skipped="3" failures="1" errors="1"/>'
            "</testsuites>",
            encoding="utf-8",
        )

        assert ac._junit_counts(tmp_path / "j.xml") == {
            "total": 10,
            "skipped": 3,
            "failed": 2,
            "passed": 5,
        }

    def test_junit_missing_or_corrupt_is_not_fatal(self, tmp_path):
        (tmp_path / "bad.xml").write_text("not xml", encoding="utf-8")

        assert ac._junit_counts(tmp_path / "absent.xml") is None
        assert ac._junit_counts(tmp_path / "bad.xml") is None

    def test_site_packages_path_maps_back_to_the_owned_file(self, tmp_path):
        """A container or wheel install reports absolute paths outside the tree."""
        measured = "/usr/lib/python3.12/site-packages/flashinfer/jit/core.py"

        assert ac._canonical(measured, tmp_path, ["flashinfer/jit/core.py"]) == (
            "flashinfer/jit/core.py"
        )

    def test_unrelated_path_maps_to_nothing(self, tmp_path):
        assert (
            ac._canonical("/usr/lib/python3.12/json/decoder.py", tmp_path, []) is None
        )
