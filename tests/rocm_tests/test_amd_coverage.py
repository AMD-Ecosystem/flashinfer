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
import json
import re
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    target = _REPO_ROOT / "scripts" / "amd_coverage.py"
    spec = importlib.util.spec_from_file_location("_fi_amd_coverage", target)
    # A None spec or loader would surface as an AttributeError on the next line,
    # hiding which file failed to load.
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
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


@pytest.fixture(autouse=True, scope="module")
def _repo_coverage_data_is_untouched():
    """No test here may write the repo's own .coverage.

    Constructing `coverage.Coverage(config_file=<repo pyproject>)` without an
    explicit data_file initializes ./.coverage and erases it -- which once
    destroyed a full suite run. Any data file a test needs goes in tmp_path.
    """
    data = _REPO_ROOT / ".coverage"
    before = data.read_bytes() if data.exists() else None
    yield
    after = data.read_bytes() if data.exists() else None
    assert after == before, "a test wrote the repo's .coverage"


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

        owned, _, unowned, _ = ac.classify(str(repo), base, {})

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

        owned, _, _, _ = ac.classify(str(repo), base, {})
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

        owned, _, unowned, _ = ac.classify(str(repo), base, {})

        # Without an R row this file would land in unowned and stop being measured.
        assert "flashinfer/new.py" in owned
        assert owned["flashinfer/new.py"].tier == "B"
        assert owned["flashinfer/new.py"].changed == {4}
        assert "flashinfer/new.py" not in unowned

    def test_file_that_becomes_python_by_rename_is_scored_whole(self, repo):
        """A .sh promoted to .py has no upstream Python lines, so all of it is ours."""
        body = "\n".join(f"# line {i}" for i in range(20)) + "\n"
        _write(repo, "scripts/helper.txt", body)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)

        _git(repo, "mv", "scripts/helper.txt", "scripts/helper.py")
        _write(repo, "scripts/helper.py", body + "VALUE = 1\n")
        _git(repo, "add", "-A")

        owned, _, _, _ = ac.classify(str(repo), base, {})

        # Tier B would score only the one appended line and drop the other 20.
        assert owned["scripts/helper.py"].tier == "A"
        assert owned["scripts/helper.py"].changed is None

    def test_uncommitted_new_file_is_still_ours(self, repo):
        """`git ls-files` alone hides it, and it is exactly what someone is writing."""
        base = _base(repo)
        _write(repo, "flashinfer/wip_rocm.py", "a = 1\n")  # never `git add`ed

        owned, _, unowned, _ = ac.classify(str(repo), base, {})

        assert owned["flashinfer/wip_rocm.py"].tier == "A"
        assert owned["flashinfer/wip_rocm.py"].reason == "untracked"

    def test_ignored_file_is_not_picked_up(self, repo):
        base = _base(repo)
        _write(repo, ".gitignore", "flashinfer/generated.py\n")
        _write(repo, "flashinfer/generated.py", "a = 1\n")

        owned, _, unowned, _ = ac.classify(str(repo), base, {})

        assert "flashinfer/generated.py" not in owned
        assert "flashinfer/generated.py" not in unowned

    def test_untouched_upstream_file_is_unowned(self, repo):
        _write(repo, "flashinfer/untouched.py", "z = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)

        owned, _, unowned, _ = ac.classify(str(repo), base, {})

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

        owned, excluded, unowned, ruled = ac.classify(str(repo), base, manifest)
        tracked = ac._surface_python(str(repo))[0]

        assert set(owned) | set(excluded) | set(unowned) == tracked
        assert not set(owned) & set(unowned)
        assert not set(owned) & set(excluded)
        assert owned["flashinfer/redirected.py"].tier == "C"


class TestUpstreamBase:
    """The base is the upstream *release* the port is built on, not a branch.

    `upstream/main` moves on its own, which reclassifies files between runs and
    makes two numbers weeks apart incomparable. A release tag never moves.
    """

    def _tagged(self, repo, amd_tag, upstream_tag="v0.5.3"):
        """Both tags, as in the real fork: upstream's release, then ours on top."""
        _write(repo, "flashinfer/a.py", "a = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream release")
        _git(repo, "tag", upstream_tag)
        _write(repo, "flashinfer/port.py", "b = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "port work")
        _git(repo, "tag", amd_tag)

    def test_release_tag_yields_the_upstream_version(self, repo):
        self._tagged(repo, "v0.5.3+amd.2")

        assert ac._upstream_release(str(repo)) == "v0.5.3"

    def test_release_candidate_suffix_is_handled(self, repo):
        self._tagged(repo, "v0.5.3+amd.1rc1")

        assert ac._upstream_release(str(repo)) == "v0.5.3"

    def test_base_is_the_upstream_release_not_our_tip(self, repo):
        """Everything after the upstream tag is the port, and all of it counts."""
        self._tagged(repo, "v0.5.3+amd.1")
        upstream_commit = ac._git(str(repo), "rev-parse", "v0.5.3^{commit}")
        _write(repo, "flashinfer/c.py", "c = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "after the release")

        assert ac._resolve_base(str(repo), None) == upstream_commit

    def test_explicit_ref_overrides_the_derived_one(self, repo):
        self._tagged(repo, "v0.5.3+amd.1")
        _git(repo, "tag", "v9.9.9")

        assert ac._resolve_base(str(repo), "v9.9.9") == _base(repo)

    def test_missing_release_tag_says_how_to_fix_it(self, repo):
        with pytest.raises(ac.ToolError, match="fetch --tags"):
            ac._resolve_base(str(repo), None)

    def test_disconnected_history_is_not_reported_as_a_missing_ref(self, repo):
        """`merge-base` can fail on a ref that resolved perfectly well.

        Without its own check that reads as a bare "merge-base failed" with an
        empty stderr, which names neither the ref nor anything to do about it.
        """
        self._tagged(repo, "v0.5.3+amd.1")
        _git(repo, "checkout", "-q", "--orphan", "unrelated")
        _write(repo, "flashinfer/d.py", "d = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "unrelated root")

        with pytest.raises(ac.ToolError, match="shares no history with HEAD"):
            ac._resolve_base(str(repo), "v0.5.3")

    def test_shallow_clone_is_told_to_unshallow_not_to_fetch_tags(self, repo, tmp_path):
        """The default path dies in `describe`, before `merge-base` is reached.

        Fetching tags into a grafted clone leaves them unreachable, so advice to
        do that sends people round a loop that cannot terminate.
        """
        self._tagged(repo, "v0.5.3+amd.1")
        _write(repo, "flashinfer/later.py", "c = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "after the release")

        # A sibling, not a child: `repo` *is* tmp_path, and cloning into it
        # would leave the clone sitting inside its own origin.
        shallow = tmp_path.parent / f"{tmp_path.name}-shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth=1", f"file://{repo}", str(shallow)],
            check=True,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        _git(shallow, "fetch", "-q", "--tags", "origin")
        assert ac._is_shallow(str(shallow)), "fixture did not produce a shallow clone"
        tags = ac._git(str(shallow), "tag", "-l")
        assert "v0.5.3+amd.1" in tags, "the tag is present, just not reachable"

        with pytest.raises(ac.ToolError, match="unshallow"):
            ac._resolve_base(str(shallow), None)


class TestManifest:
    def test_stale_entry_fails_rather_than_being_skipped(self, repo):
        base = _base(repo)
        manifest = {"redirect_owned": [{"path": "flashinfer/gone.py", "reason": "r"}]}

        with pytest.raises(ac.ToolError, match="does not exist"):
            ac.classify(str(repo), base, manifest)

    @pytest.mark.parametrize("key", ["redirect_owned", "unowned"])
    def test_tier_ruling_outside_the_measured_surface_is_rejected(self, repo, key):
        """It exists, so the old check passed -- and then classify never saw it."""
        base = _base(repo)
        _write(repo, "docs/notes.py", "a = 1\n")
        _git(repo, "add", "-A")
        manifest = {key: [{"path": "docs/notes.py", "reason": "r"}]}

        with pytest.raises(ac.ToolError, match="not on the measured surface"):
            ac.classify(str(repo), base, manifest)

    def test_excluded_may_name_a_gitignored_file(self, repo):
        """The case the section exists for: coverage measures generated modules
        that git never tracks, and the entry records why they are not counted.
        Requiring them on the surface rejected the shipped manifest outright."""
        base = _base(repo)
        _write(repo, ".gitignore", "flashinfer/_version.py\n")
        _write(repo, "flashinfer/_version.py", "__version__ = '1'\n")
        _git(repo, "add", "-A")
        manifest = {
            "excluded": [{"path": "flashinfer/_version.py", "reason": "generated"}]
        }

        _, excluded, _, _ = ac.classify(str(repo), base, manifest)

        assert excluded == {"flashinfer/_version.py": "generated"}

    @pytest.mark.parametrize("key", ["redirect_owned", "unowned"])
    def test_tier_ruling_naming_a_missing_file_is_rejected(self, repo, key):
        base = _base(repo)
        manifest = {key: [{"path": "flashinfer/gone.py", "reason": "r"}]}

        with pytest.raises(ac.ToolError, match="does not exist"):
            ac.classify(str(repo), base, manifest)

    def test_excluded_may_name_a_file_that_is_not_built_yet(self, repo):
        """`_version.py` is generated, so a fresh checkout does not have it --
        which is exactly how this failed in CI."""
        base = _base(repo)
        manifest = {
            "excluded": [{"path": "flashinfer/_version.py", "reason": "generated"}]
        }

        _, excluded, _, _ = ac.classify(str(repo), base, manifest)

        assert excluded == {"flashinfer/_version.py": "generated"}

    def test_entry_without_a_reason_is_rejected(self, repo):
        """Every ruling is hand-made, so it has to say why."""
        base = _base(repo)
        _write(repo, "flashinfer/x.py", "a = 1\n")
        _git(repo, "add", "-A")

        for reason in ("", "   "):
            manifest = {"unowned": [{"path": "flashinfer/x.py", "reason": reason}]}
            with pytest.raises(ac.ToolError, match="no reason"):
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

    def test_ruled_unowned_reasons_survive_classification(self, repo):
        """The manifest demands a reason; dropping it makes the ruling unauditable."""
        base = _base(repo)
        _write(repo, "flashinfer/theirs.py", "a = 1\n")
        _git(repo, "add", "-A")
        manifest = {
            "unowned": [
                {"path": "flashinfer/theirs.py", "reason": "upstream, untouched"}
            ]
        }

        _, _, unowned, ruled = ac.classify(str(repo), base, manifest)

        assert "flashinfer/theirs.py" in unowned
        assert ruled == {"flashinfer/theirs.py": "upstream, untouched"}

    def test_shipped_manifest_actually_classifies(self):
        """Run classify() against the real manifest, not just check the paths.

        A validation change once made this raise on every invocation while the
        path-existence test above still passed, so the tool has to be exercised.
        """
        repo = ac._git(None, "rev-parse", "--show-toplevel")
        try:
            base = ac._resolve_base(repo, None)
        except ac.ToolError as exc:
            # A shallow clone reaches neither the release tag nor the fork point.
            pytest.skip(f"no upstream base in this clone: {exc}")
        manifest = ac._load_toml(Path(repo) / ac._MANIFEST)

        owned, excluded, unowned, ruled = ac.classify(repo, base, manifest)

        assert owned, "the shipped manifest must not empty the owned set"
        assert set(excluded) | set(ruled), "its rulings must survive validation"

    def test_shipped_manifest_matches_the_tree(self):
        """Catches a manifest entry left behind by a rename or deletion.

        Existence is asserted only for the tier rulings. `excluded` names build
        artifacts, which a checkout that has never been built does not have --
        asserting on those failed in CI while passing on every dev machine.
        """
        repo = ac._git(None, "rev-parse", "--show-toplevel")
        manifest = ac._load_toml(Path(repo) / ac._MANIFEST)
        for key in ("redirect_owned", "unowned", "excluded"):
            for item in manifest.get(key, []):
                if key != "excluded":
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

        # data_file must point into tmp_path: constructing Coverage against the
        # repo config otherwise initializes ./.coverage and erases a real run.
        cov = coverage.Coverage(
            data_file=str(tmp_path / ".coverage"), config_file=str(cfg)
        )
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


class TestDirtyTree:
    def test_clean_tree_is_clean(self, repo):
        assert ac._is_dirty(str(repo)) is False

    def test_untracked_owned_file_counts_as_dirty(self, repo):
        """It is scored as tier A, so the header cannot call the tree clean."""
        _write(repo, "flashinfer/wip.py", "a = 1\n")

        assert ac._is_dirty(str(repo)) is True

    def test_the_tools_own_artifacts_do_not_count(self, repo):
        """run() writes these into the repo by default and only .coverage is
        gitignored, so a blanket untracked check calls every run dirty."""
        for name in ("junit.xml", "import-baseline.coverage", "jit-reach.gw0.json"):
            _write(repo, name, "x")

        assert ac._is_dirty(str(repo)) is False

    def test_modified_tracked_file_counts_as_dirty(self, repo):
        _write(repo, "flashinfer/m.py", "a = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "add")
        _write(repo, "flashinfer/m.py", "a = 2\n")

        assert ac._is_dirty(str(repo)) is True


class TestStaleArtifacts:
    """Guards against reporting a confident number for code that never ran."""

    def test_sources_newer_than_the_data_are_named(self, tmp_path):
        (tmp_path / "flashinfer").mkdir()
        data = tmp_path / ".coverage"
        old = tmp_path / "flashinfer" / "old.py"
        new = tmp_path / "flashinfer" / "new.py"
        for path in (data, old, new):
            path.write_text("a = 1\n", encoding="utf-8")
        # Explicit stamps: written in the same instant, mtimes compare equal.
        os.utime(old, (1000, 1000))
        os.utime(data, (2000, 2000))
        os.utime(new, (3000, 3000))

        stale = ac._stale_sources(
            tmp_path, data, ["flashinfer/old.py", "flashinfer/new.py"]
        )

        assert stale == ["flashinfer/new.py"]

    def test_a_run_that_wrote_nothing_is_refused(self, tmp_path, monkeypatch):
        """A plugin ImportError exits 1, exactly like ordinary test failures."""
        (tmp_path / "junit.xml").write_text("<testsuites/>", encoding="utf-8")
        (tmp_path / ".coverage").write_text("stale", encoding="utf-8")
        os.utime(tmp_path / "junit.xml", (1, 1))
        os.utime(tmp_path / ".coverage", (1, 1))

        class _Dead:
            returncode = 1  # pytest's "tests failed" *and* "plugin failed to import"

        monkeypatch.setattr(ac.subprocess, "run", lambda *a, **k: _Dead())

        with pytest.raises(ac.ToolError, match="without writing"):
            ac._run_pytest(tmp_path, tmp_path, tmp_path / ".coverage", [])

    @pytest.mark.parametrize("returncode", [1, 4])
    def test_a_failed_run_leaves_the_previous_data_intact(
        self, tmp_path, monkeypatch, returncode
    ):
        """Clearing the data file up front must not cost a completed suite's only copy.

        Exit 4 aborts before the artifact check, so both paths have to restore.
        """
        data = tmp_path / ".coverage"
        data.write_text("a completed 40-minute run", encoding="utf-8")

        class _Dead:
            pass

        _Dead.returncode = returncode
        monkeypatch.setattr(ac.subprocess, "run", lambda *a, **k: _Dead())

        with pytest.raises(ac.ToolError):
            ac._run_pytest(tmp_path, tmp_path, data, [])

        assert data.read_text(encoding="utf-8") == "a completed 40-minute run"
        assert not (tmp_path / ".coverage.prev").exists(), "sidecar left behind"


class TestImportBaseline:
    def test_failure_is_an_error_not_a_silent_downgrade(self, tmp_path, monkeypatch):
        """Losing the baseline silently folds import-time lines into the headline."""

        class _Failed:
            returncode = 1
            stderr = "ModuleNotFoundError: No module named 'flashinfer'"

        monkeypatch.setattr(ac.subprocess, "run", lambda *a, **k: _Failed())

        with pytest.raises(ac.ToolError, match="--no-baseline"):
            ac._capture_baseline(tmp_path, tmp_path / ".coverage.baseline")

        assert not (tmp_path / "_import_probe.py").exists(), "probe left behind"

    def test_repo_is_importable_by_the_probe(self, tmp_path, monkeypatch):
        """`coverage run <script>` puts the script's dir on sys.path, not the cwd."""
        seen = {}

        class _Ok:
            returncode = 0
            stderr = ""

        def _capture(*args, **kwargs):
            seen.update(kwargs.get("env") or {})
            return _Ok()

        monkeypatch.setattr(ac.subprocess, "run", _capture)
        out = tmp_path / "cov" / ".coverage.baseline"
        out.parent.mkdir()

        ac._capture_baseline(tmp_path, out)

        assert str(tmp_path) in seen.get("PYTHONPATH", "").split(os.pathsep)


class TestDataFileHygiene:
    def test_baseline_is_not_named_like_parallel_coverage_data(self):
        """`.coverage.*` is coverage's parallel glob; pytest-cov combine() eats it."""
        src = (_REPO_ROOT / "scripts" / "amd_coverage.py").read_text(encoding="utf-8")
        match = re.search(r'baseline = out_dir / "([^"]+)"', src)
        assert match, "baseline filename moved; this guard no longer reads it"
        name = match.group(1)

        assert not name.startswith(".coverage."), (
            f"{name} would be absorbed and unlinked by combine(), "
            "silently removing the import-time split"
        )

    def test_run_writes_the_file_it_scores(self, tmp_path, monkeypatch):
        """--run --data-file X wrote repo/.coverage and then scored X."""
        seen = {}

        class _Ok:
            returncode = 0

        def _capture(cmd, **kwargs):
            seen.update(kwargs.get("env") or {})
            for name in ("junit.xml", "chosen.cov"):
                (tmp_path / name).write_text("x", encoding="utf-8")
                os.utime(tmp_path / name, (9000, 9000))  # distinct from "absent"
            return _Ok()

        monkeypatch.setattr(ac.subprocess, "run", _capture)

        ac._run_pytest(tmp_path, tmp_path, tmp_path / "chosen.cov", [])

        assert seen.get("COVERAGE_FILE") == str(tmp_path / "chosen.cov")
        # Without the stamp, score-only mode cannot tell this run's reach shards
        # from a previous run's and reports nothing.
        assert ac._stamp_matches(tmp_path, tmp_path / "chosen.cov")

    def test_a_rewrite_inside_one_clock_tick_is_still_accepted(self, tmp_path):
        """Coarse mtime granularity must not read as "pytest wrote nothing"."""
        junit, data = tmp_path / "junit.xml", tmp_path / ".coverage"
        for f in (junit, data):
            f.write_text("old", encoding="utf-8")
            os.utime(f, (5000, 5000))

        class _Ok:
            returncode = 0

        def _rewrite(cmd, **kwargs):
            for f in (junit, data):
                f.write_text("new content, different length", encoding="utf-8")
                os.utime(f, (5000, 5000))  # same timestamp, as on a coarse clock
            return _Ok()

        import unittest.mock

        with unittest.mock.patch.object(ac.subprocess, "run", _rewrite):
            ac._run_pytest(tmp_path, tmp_path, data, [])  # must not raise

    def test_reach_needs_a_stamp_tying_shards_to_this_data_file(self, tmp_path):
        """Shards are always fractionally older than .coverage, so mtime cannot decide.

        pytest-cov combines the data file after the workers write their shards.
        A cutoff on mtime therefore rejects every genuine run; the stamp is what
        distinguishes "this run" from a previous one left in the same directory.
        """
        (tmp_path / "flashinfer" / "csrc_rocm").mkdir(parents=True)
        (tmp_path / "flashinfer" / "csrc_rocm" / "a.cu").write_text("x")
        data = tmp_path / ".coverage"
        shard = tmp_path / "jit-reach.gw0.json"
        shard.write_text('["a.cu"]', encoding="utf-8")
        data.write_text("x", encoding="utf-8")  # written last, as in a real run

        assert ac._jit_reach(tmp_path, tmp_path, data) is None, "no stamp, no reach"

        ac._write_stamp(tmp_path, data)
        assert ac._jit_reach(tmp_path, tmp_path, data) == (1, [])

        # Same mtime, different content: NFS granularity makes this reachable,
        # so the stamp carries size as well as the timestamp.
        stamped = data.stat()
        data.write_text("a different length entirely", encoding="utf-8")
        os.utime(data, ns=(stamped.st_mtime_ns, stamped.st_mtime_ns))
        assert ac._jit_reach(tmp_path, tmp_path, data) is None

    def test_zero_reached_is_reported_not_treated_as_absent(self, tmp_path):
        """ "0 of N loaded" and "the plugin never ran" are different answers.

        The plugin writes a shard even with nothing loaded, so an empty shard
        must survive as a real result rather than collapsing to None.
        """
        (tmp_path / "flashinfer" / "csrc_rocm").mkdir(parents=True)
        for name in ("a.cu", "b.cu"):
            (tmp_path / "flashinfer" / "csrc_rocm" / name).write_text("x")
        data = tmp_path / ".coverage"
        (tmp_path / "jit-reach.gw0.json").write_text("[]", encoding="utf-8")
        data.write_text("x", encoding="utf-8")
        ac._write_stamp(tmp_path, data)

        assert ac._jit_reach(tmp_path, tmp_path, data) == (0, ["a.cu", "b.cu"])


class TestExclusionAccounting:
    def test_tier_b_counts_only_exclusions_inside_our_diff(self, repo, tmp_path):
        """A guard outside our diff was never in our denominator to remove."""
        coverage = pytest.importorskip("coverage")
        body = [
            "IS_CUDA = False",  # 1
            "if IS_CUDA:",  # 2  guard upstream owns
            "    THEIRS = 1",  # 3
            "VALUE = 2",  # 4
            "if IS_CUDA:",  # 5  guard inside our diff
            "    OURS = 3",  # 6
        ]
        _write(repo, "flashinfer/up.py", "\n".join(body) + "\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "upstream")
        base = _base(repo)
        # Our diff touches an ordinary statement (4) and a guarded one (6).
        edited = body[:3] + ["VALUE = 22", body[4], "    OURS = 99"]
        _write(repo, "flashinfer/up.py", "\n".join(edited) + "\n")
        (repo / "pyproject.toml").write_text(
            "[tool.coverage.run]\nsource = ['.']\n\n"
            # TOML literal strings do not escape, so this is the regex verbatim.
            "[tool.coverage.report]\nexclude_also = ['^\\s*if IS_CUDA:\\s*$']\n",
            encoding="utf-8",
        )
        data = tmp_path / ".coverage"
        seeded = coverage.CoverageData(basename=str(data))
        # One executed owned line, or score()'s empty-run guard fires first.
        seeded.add_lines({str(repo / "flashinfer" / "up.py"): [4]})
        seeded.write()

        owned, _, _, _ = ac.classify(str(repo), base, {})
        scores, _ = ac.score(repo, owned, data, None)
        entry = next(s for s in scores if s.path == "flashinfer/up.py")

        assert entry.owned == {4}, "the guarded line is excluded from the denominator"
        # Whole-file counting would report 4 (both guards and both bodies);
        # only line 6 is inside our diff.
        assert entry.excluded == 1

    def test_foreign_baseline_root_is_reported_too(self, repo, tmp_path):
        """Import-time lines are subtracted, so a baseline from a different
        checkout shrinks the numerator -- silently, unless its root surfaces."""
        coverage = pytest.importorskip("coverage")
        base = _base(repo)
        _write(repo, "flashinfer/m.py", "a = 1\nb = 2\n")
        # score() passes this to coverage as an explicit config; it must exist.
        _write(repo, "pyproject.toml", "[tool.coverage.run]\nsource = ['.']\n")
        _git(repo, "add", "-A")
        owned, _, _, _ = ac.classify(str(repo), base, {})

        data = tmp_path / ".coverage"
        run = coverage.CoverageData(basename=str(data))
        run.add_lines({str(repo / "flashinfer" / "m.py"): [1, 2]})
        run.write()

        stale = tmp_path / "baseline.coverage"
        old = coverage.CoverageData(basename=str(stale))
        old.add_lines({"/elsewhere/checkout/flashinfer/m.py": [1]})
        old.write()

        _, foreign = ac.score(repo, owned, data, stale)

        assert "/elsewhere/checkout" in foreign

    def test_lines_from_another_checkout_are_reported(self, tmp_path):
        """The main checkout's editable install shadows a worktree without PYTHONPATH."""
        coverage = pytest.importorskip("coverage")
        data = tmp_path / ".coverage"
        d = coverage.CoverageData(basename=str(data))
        d.add_lines({"/elsewhere/main-checkout/flashinfer/page.py": [1, 2]})
        d.write()

        executed, foreign = ac._executed(data, tmp_path, ["flashinfer/page.py"])

        assert executed["flashinfer/page.py"] == {1, 2}
        assert foreign == {"/elsewhere/main-checkout"}

    def test_in_tree_paths_are_not_flagged(self, tmp_path):
        coverage = pytest.importorskip("coverage")
        data = tmp_path / ".coverage"
        d = coverage.CoverageData(basename=str(data))
        d.add_lines({str(tmp_path / "flashinfer" / "page.py"): [1]})
        d.write()

        _, foreign = ac._executed(data, tmp_path, ["flashinfer/page.py"])

        assert foreign == set()


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


class TestBaseRefHint:
    def test_derived_ref_hint_names_origin_tags(self, repo):
        """The amd tag is reachable but upstream's own tag was never fetched."""
        _write(repo, "flashinfer/a.py", "a = 1\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "port work")
        _git(repo, "tag", "v0.5.3+amd.2")  # no bare v0.5.3 in this clone

        with pytest.raises(ac.ToolError, match=r"git fetch origin tag v0\.5\.3"):
            ac._resolve_base(str(repo), None)

    def test_explicit_branch_ref_is_not_told_to_fetch_a_tag(self, repo):
        """`--upstream-ref` takes any ref; only the derived one is a tag on origin."""
        with pytest.raises(ac.ToolError) as exc:
            ac._resolve_base(str(repo), "upstream/main")

        assert "fetch origin tag" not in str(exc.value)
        assert "--upstream-ref" in str(exc.value)


class TestNoExecutableStatements:
    """Every owned statement running at import leaves no percentage to report."""

    def _scores(self):
        return [
            ac.Score(
                path="flashinfer/a.py",
                tier="A",
                reason="",
                owned={1, 2},
                covered={1, 2},
                import_time={1, 2},  # the baseline subtracts the whole denominator
                excluded=0,
            )
        ]

    def _report(self, tmp_path):
        return ac._report(
            tmp_path,
            "base",
            False,
            "gfx942",
            self._scores(),
            {},
            [],
            {},
            None,
            None,
            [],
            set(),
            False,
        )

    def test_report_returns_none_rather_than_a_fabricated_zero(self, tmp_path, capsys):
        pct = self._report(tmp_path)
        capsys.readouterr()

        assert pct is None, "0.0 here would fail a --fail-under on nothing measured"

    def test_printed_report_and_return_value_agree(self, tmp_path, capsys):
        """The bug was the two disagreeing: printed "n/a", returned 0.0."""
        pct = self._report(tmp_path)
        out = capsys.readouterr().out

        assert "n/a" in out
        assert pct is None, "printed n/a but returned a number --fail-under would use"


class TestReachPlugin:
    """The plugin is what decides whether "0 loaded" is reportable at all."""

    def _plugin(self):
        spec = importlib.util.spec_from_file_location(
            "_fi_jit_reach", _REPO_ROOT / "tests" / "jit_reach_plugin.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_run_that_loaded_nothing_still_writes_a_shard(
        self, tmp_path, monkeypatch
    ):
        """No shard reads as "the plugin never ran", which is a different claim."""
        plugin = self._plugin()
        plugin._installed = True  # the JitSpec.load wrap went in
        monkeypatch.setenv(plugin._ENV_DIR, str(tmp_path))
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

        plugin.pytest_sessionfinish(session=None, exitstatus=0)

        shard = tmp_path / "jit-reach.master.json"
        assert shard.exists(), "an empty run must still record that it ran"
        assert json.loads(shard.read_text(encoding="utf-8")) == []

    def test_no_shard_when_the_wrap_never_installed(self, tmp_path, monkeypatch):
        """`0 of N` would be a confident answer from a run that never looked.

        pytest_configure bails out when flashinfer is unimportable, so the env
        var alone does not mean anything was instrumented.
        """
        plugin = self._plugin()
        assert plugin._installed is False
        monkeypatch.setenv(plugin._ENV_DIR, str(tmp_path))

        plugin.pytest_sessionfinish(session=None, exitstatus=0)

        assert not list(tmp_path.glob("jit-reach.*.json"))

    def test_without_the_env_var_it_writes_nothing(self, tmp_path, monkeypatch):
        plugin = self._plugin()
        plugin._installed = True
        monkeypatch.delenv(plugin._ENV_DIR, raising=False)

        plugin.pytest_sessionfinish(session=None, exitstatus=0)

        assert not list(tmp_path.glob("jit-reach.*.json"))


class TestFailUnderAgainstNoPercentage:
    """The caller has to survive `_report` returning None, not just produce it."""

    def _args(self, repo, **over):
        defaults = dict(
            upstream_ref=None,
            run=False,
            out_dir=str(repo),
            data_file=str(repo / ".coverage"),
            json_out=str(repo / "cov.json"),
            fail_under=60.0,
            no_baseline=True,
            show_files=False,
            pytest_args=[],
        )
        defaults.update(over)
        return __import__("argparse").Namespace(**defaults)

    def _repo_with_no_executable_lines(self, repo, monkeypatch):
        """Every owned statement is import-time, so the denominator is zero."""
        _git(repo, "tag", "v0.5.3")  # the upstream release the fork sits on
        _write(repo, "flashinfer/a.py", "import os\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "port work")
        _git(repo, "tag", "v0.5.3+amd.1")
        (repo / ".coverage").write_text("x", encoding="utf-8")

        score = ac.Score(
            path="flashinfer/a.py",
            tier="A",
            reason="",
            owned={1},
            covered={1},
            import_time={1},
            excluded=0,
        )
        monkeypatch.setattr(ac, "score", lambda *a, **k: ([score], set()))
        monkeypatch.chdir(repo)

    def test_it_exits_without_comparing_and_writes_null(
        self, repo, monkeypatch, capsys
    ):
        self._repo_with_no_executable_lines(repo, monkeypatch)

        # Would raise TypeError on `pct < fail_under` if None leaked through.
        code = ac.run(self._args(repo))
        out = capsys.readouterr().out

        assert code == ac.EXIT_ERROR, "no threshold was missed; none exists"
        assert "no execution percentage to compare" in out
        assert json.loads((repo / "cov.json").read_text())["execution_percent"] is None

    def test_without_fail_under_it_is_not_an_error(self, repo, monkeypatch, capsys):
        self._repo_with_no_executable_lines(repo, monkeypatch)

        code = ac.run(self._args(repo, fail_under=None))
        capsys.readouterr()

        assert code == ac.EXIT_OK


def _score(
    path="flashinfer/a.py",
    tier="A",
    owned=(1, 2, 3),
    covered=(1, 2),
    import_time=(),
    excluded=0,
    reason="",
):
    return ac.Score(
        path=path,
        tier=tier,
        reason=reason,
        owned=set(owned),
        covered=set(covered),
        import_time=set(import_time),
        excluded=excluded,
    )


def _report(**overrides):
    kwargs = dict(
        repo=Path("/wt"),
        base_desc="abc123 2026-01-01 base",
        dirty=False,
        arch="gfx942",
        scores=[_score()],
        excluded={},
        unowned=[],
        ruled={},
        tests=None,
        reach=None,
        stale=[],
        foreign=set(),
        show_files=False,
    )
    kwargs.update(overrides)
    return ac._report(**kwargs)


class TestScoreArithmetic:
    def test_import_time_lines_leave_both_sides_of_the_ratio(self):
        s = _score(owned=(1, 2, 3, 4), covered=(1, 2, 3), import_time=(1, 2))
        assert (s.exec_total, s.exec_covered) == (2, 1)

    def test_a_file_that_is_all_import_has_no_execution_ratio(self):
        s = _score(owned=(1, 2), covered=(1, 2), import_time=(1, 2))
        assert (s.exec_total, s.exec_covered) == (0, 0)

    def test_covered_lines_outside_owned_do_not_count(self):
        """Tier B scores only our hunks; upstream's covered lines are not ours."""
        assert _score(owned=(1, 2), covered=(1, 2, 99)).exec_covered == 2


class TestReportHeadline:
    def test_percentage_is_execution_only(self, capsys):
        pct = _report(
            scores=[_score(owned=(1, 2, 3, 4), covered=(1, 2, 3), import_time=(1,))]
        )
        out = capsys.readouterr().out

        assert pct == pytest.approx(100 * 2 / 3)
        assert "import-time lines (always covered)" in out
        assert "total, conventional" in out

    def test_no_executable_statements_yields_none_not_zero(self, capsys):
        """0.0 would print n/a and still fail a --fail-under, on nothing measured."""
        assert (
            _report(scores=[_score(owned=(1,), covered=(1,), import_time=(1,))]) is None
        )
        assert "n/a" in capsys.readouterr().out

    def test_without_a_baseline_the_headline_says_so(self, capsys):
        _report(scores=[_score(import_time=())])
        assert "no import-time baseline" in capsys.readouterr().out

    def test_dirty_tree_is_flagged_in_the_header(self, capsys):
        _report(dirty=True)
        assert "(uncommitted changes)" in capsys.readouterr().out

    def test_stale_sources_are_named_and_capped(self, capsys):
        _report(stale=[f"flashinfer/f{i}.py" for i in range(5)])
        out = capsys.readouterr().out

        assert "STALE : 5 owned files changed after this run" in out
        assert "+2 more" in out

    def test_foreign_root_is_named_rather_than_folded_in(self, capsys):
        _report(foreign={"/other/checkout"})
        out = capsys.readouterr().out

        assert "NOTE  : executed lines came from /other/checkout" in out
        assert "set PYTHONPATH" in out

    def test_skips_are_called_out_because_they_lower_the_number(self, capsys):
        _report(tests={"passed": 10, "skipped": 3, "failed": 1, "total": 14})
        out = capsys.readouterr().out

        assert "tests : 10 passed, 3 skipped, 1 failed" in out
        assert "skips lower the number" in out

    def test_zero_skips_omits_the_caveat(self, capsys):
        _report(tests={"passed": 10, "skipped": 0, "failed": 0, "total": 10})
        assert "skips lower the number" not in capsys.readouterr().out

    def test_per_file_table_is_opt_in(self, capsys):
        _report(show_files=False)
        assert "== per file ==" not in capsys.readouterr().out

    def test_per_file_table_shows_the_excluded_count(self, capsys):
        _report(scores=[_score(excluded=7)], show_files=True)
        out = capsys.readouterr().out

        assert "== per file ==" in out
        assert "(7 excluded)" in out

    def test_tier_subtotals_cover_every_tier_present(self, capsys):
        _report(
            scores=[
                _score(path="flashinfer/a.py", tier="A"),
                _score(path="flashinfer/b.py", tier="B"),
                _score(path="flashinfer/c.py", tier="C"),
            ]
        )
        out = capsys.readouterr().out

        for tier in "ABC":
            assert f"tier {tier}" in out

    def test_not_counted_section_lists_rulings_and_exclusions(self, capsys):
        _report(
            scores=[_score(excluded=4)],
            excluded={"flashinfer/_version.py": "generated"},
            ruled={"flashinfer/fp4_quantization.py": "basename collision"},
            unowned=["flashinfer/up.py"],
        )
        out = capsys.readouterr().out

        assert "4 lines in 1 files excluded" in out
        assert "flashinfer/_version.py -- generated" in out
        assert "flashinfer/fp4_quantization.py -- basename collision" in out
        assert "1 upstream files on the measured surface" in out

    def test_reach_is_labelled_as_not_coverage(self, capsys):
        _report(reach=(44, ["activation.cu", "flashinfer_ops.cu"]))
        out = capsys.readouterr().out

        assert "44 of 46 translation units" in out
        assert "not a coverage figure" in out
        assert "never loaded: activation.cu, flashinfer_ops.cu" in out

    def test_a_long_unreached_list_is_truncated(self, capsys):
        _report(reach=(0, [f"u{i}.cu" for i in range(10)]))
        assert "+4 more" in capsys.readouterr().out


class TestJitReach:
    @pytest.fixture
    def reach(self, tmp_path):
        """An out-dir whose stamp matches its data file, plus two csrc units."""
        repo, out_dir = tmp_path / "repo", tmp_path / "out"
        (repo / ac._CSRC_DIR).mkdir(parents=True)
        for name in ("a.cu", "b.cu", "notes.txt"):
            (repo / ac._CSRC_DIR / name).write_text("// x\n")
        out_dir.mkdir()
        data = tmp_path / "data.coverage"
        data.write_text("x")
        ac._write_stamp(out_dir, data)
        return repo, out_dir, data

    def test_counts_units_loaded_across_shards(self, reach):
        repo, out_dir, data = reach
        (out_dir / "jit-reach.gw0.json").write_text(json.dumps(["a.cu"]))
        (out_dir / "jit-reach.gw1.json").write_text(json.dumps(["a.cu", "b.cu"]))

        assert ac._jit_reach(repo, out_dir, data) == (2, [])

    def test_non_source_files_are_not_translation_units(self, reach):
        repo, out_dir, data = reach
        (out_dir / "jit-reach.gw0.json").write_text(json.dumps(["a.cu"]))

        assert ac._jit_reach(repo, out_dir, data) == (1, ["b.cu"])

    def test_a_corrupt_shard_is_skipped_not_fatal(self, reach):
        repo, out_dir, data = reach
        (out_dir / "jit-reach.gw0.json").write_text("{not json")
        (out_dir / "jit-reach.gw1.json").write_text(json.dumps(["a.cu"]))

        assert ac._jit_reach(repo, out_dir, data) == (1, ["b.cu"])

    def test_a_stale_stamp_suppresses_the_figure(self, reach):
        """Shards from an earlier run would otherwise be reported as this one's."""
        repo, out_dir, data = reach
        (out_dir / "jit-reach.gw0.json").write_text(json.dumps(["a.cu"]))
        data.write_text("rewritten, so the stamp no longer matches")

        assert ac._jit_reach(repo, out_dir, data) is None

    def test_no_shards_means_no_figure(self, reach):
        assert ac._jit_reach(*reach) is None

    def test_no_translation_units_means_no_figure(self, reach, tmp_path):
        repo, out_dir, data = reach
        (out_dir / "jit-reach.gw0.json").write_text(json.dumps([]))
        for unit in (repo / ac._CSRC_DIR).glob("*.cu"):
            unit.unlink()

        assert ac._jit_reach(repo, out_dir, data) is None


class TestPathAnchoring:
    """`--out-dir`, `--data-file` and `--json-out` all resolve against the repo
    root. pytest runs with cwd=repo, so a cwd-relative path would be written in
    one place and read from another."""

    def test_a_relative_path_lands_under_the_repo_root(self, tmp_path):
        assert ac._anchor(tmp_path, "docs/rocm/cov.json", tmp_path) == (
            tmp_path / "docs/rocm/cov.json"
        )

    def test_an_absolute_path_is_honoured_as_given(self, tmp_path):
        outside = Path("/var/tmp/cov.json")
        assert ac._anchor(tmp_path, str(outside), tmp_path) == outside

    def test_an_unset_option_falls_back_to_the_default(self, tmp_path):
        default = tmp_path / ".coverage"
        assert ac._anchor(tmp_path, None, default) == default
        assert ac._anchor(tmp_path, "", default) == default

    def test_every_path_option_goes_through_it(self):
        """json_out bypassed the anchoring until this branch; the print that
        follows also has to tolerate the out-of-repo path _anchor allows."""
        source = (_REPO_ROOT / "scripts" / "amd_coverage.py").read_text()
        assert source.count("_anchor(repo,") == 3
        guard = source[source.index("json_out = _anchor(repo,") :]
        assert (
            "contextlib.suppress(ValueError)" in guard[: guard.index('print(f"wrote')]
        )


class TestMain:
    def _argv(self, monkeypatch, *args):
        monkeypatch.setattr(sys, "argv", ["amd_coverage.py", *args])

    def test_tool_error_exits_two_with_the_message_on_stderr(self, monkeypatch, capsys):
        self._argv(monkeypatch)
        monkeypatch.setattr(
            ac, "run", lambda a: (_ for _ in ()).throw(ac.ToolError("boom"))
        )

        assert ac.main() == ac.EXIT_ERROR
        assert "error: boom" in capsys.readouterr().err

    def test_defaults_are_what_the_documented_recipe_assumes(self, monkeypatch):
        self._argv(monkeypatch)
        seen = {}
        monkeypatch.setattr(ac, "run", lambda a: seen.setdefault("a", a) and ac.EXIT_OK)

        ac.main()
        args = seen["a"]

        assert (args.run, args.show_files, args.no_baseline) == (False, False, False)
        assert args.upstream_ref is None
        assert (args.data_file, args.out_dir, args.json_out) == (None, None, None)
        assert args.fail_under is None
        assert args.pytest_args == []

    def test_pytest_args_after_the_separator_are_collected(self, monkeypatch):
        self._argv(monkeypatch, "--run", "--", "-n", "4", "-m", "not slow")
        seen = {}
        monkeypatch.setattr(ac, "run", lambda a: seen.setdefault("a", a) and ac.EXIT_OK)

        ac.main()

        assert seen["a"].run is True
        assert seen["a"].pytest_args == ["-n", "4", "-m", "not slow"]

    def test_fail_under_is_a_float(self, monkeypatch):
        self._argv(monkeypatch, "--fail-under", "90.5")
        seen = {}
        monkeypatch.setattr(ac, "run", lambda a: seen.setdefault("a", a) and ac.EXIT_OK)

        ac.main()

        assert seen["a"].fail_under == pytest.approx(90.5)
