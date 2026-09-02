# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The recorded upstream base for ``scripts/upstream_base.py``.

A squash-merged sync leaves no merge parent, so anchoring on our own tip walks
back to the *previous* fork point and misreports the whole delta as ours. These
build throwaway repos in that exact topology -- ``theirs`` a sibling of the
recorded base, ``ours`` with no ancestry to either.

Git and filesystem only -- no GPU, no torch -- so CI runs this with ``--noconftest``.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    target = _REPO_ROOT / "scripts" / "upstream_base.py"
    spec = importlib.util.spec_from_file_location("_fi_upstream_base", target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ub = _load_tool()


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def _rev(repo, ref="HEAD"):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(repo, *args, check=True):
    """The Runner the tool expects, matching each script's own helper."""
    proc = subprocess.run(
        ["git"] + (["-C", repo] if repo else []) + list(args),
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if check and proc.returncode:
        raise AssertionError(proc.stderr)
    return proc


def _write(repo, rel, text):
    path = Path(repo) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)


@pytest.fixture
def forked(tmp_path):
    """The real post-squash shape.

    ``B -> R0 -> R1`` is upstream's release line (``R1`` is the recorded base,
    tagged ``v1.0``); ``R0 -> M1`` is ``main``, a sibling that never merged back;
    ``B -> O1`` is our squashed tip, which reaches neither.
    """
    _git(tmp_path, "init", "-q", "-b", "ours")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _write(tmp_path, "shared.cu", "base\n")
    _commit(tmp_path, "B")
    base = _rev(tmp_path)

    _git(tmp_path, "checkout", "-q", "-b", "release")
    _write(tmp_path, "shared.cu", "r0\n")
    _commit(tmp_path, "R0")
    r0 = _rev(tmp_path)
    _write(tmp_path, "shared.cu", "r1\n")
    _commit(tmp_path, "R1")
    r1 = _rev(tmp_path)
    _git(tmp_path, "tag", "v1.0")

    _git(tmp_path, "checkout", "-q", "-b", "main", r0)
    _write(tmp_path, "other.cu", "m1\n")
    _commit(tmp_path, "M1")

    _git(tmp_path, "checkout", "-q", "ours", "--")
    _git(tmp_path, "checkout", "-q", "ours")
    _write(tmp_path, ub.FILENAME, f"v1.0 {r1}\n")
    _commit(tmp_path, "O1: squashed sync")

    return type("Forked", (), {"path": tmp_path, "base": base, "r0": r0, "r1": r1})()


class TestParse:
    def test_reads_ref_and_sha_past_comments_and_blanks(self):
        sha = "a" * 40
        got = ub.parse(f"# note\n\nv0.6.18 {sha}\n", "src")
        assert got == ub.UpstreamBase("v0.6.18", sha)

    @pytest.mark.parametrize("text", ["v0.6.18\n", "v0.6.18 %s extra\n" % ("a" * 40)])
    def test_wrong_field_count_is_an_error(self, text):
        with pytest.raises(ub.UpstreamBaseError, match="expected '<ref> <sha>'"):
            ub.parse(text, "src")

    @pytest.mark.parametrize("sha", ["deadbeef", "A" * 40, "g" * 40, "-" + "a" * 39])
    def test_a_sha_that_is_not_40_lowercase_hex_is_an_error(self, sha):
        # The '-' case also keeps the value from reaching git as an option.
        with pytest.raises(ub.UpstreamBaseError, match="40-character"):
            ub.parse(f"v0.6.18 {sha}\n", "src")

    def test_a_file_of_only_comments_is_an_error_not_a_silent_none(self):
        with pytest.raises(ub.UpstreamBaseError, match="no '<ref> <sha>' line"):
            ub.parse("# nothing here\n\n", "src")


class TestRead:
    def test_worktree_returns_none_when_absent(self, tmp_path):
        assert ub.read_worktree(str(tmp_path)) is None

    def test_worktree_reads_the_file(self, forked):
        assert ub.read_worktree(str(forked.path)).sha == forked.r1

    def test_worktree_propagates_a_malformed_file(self, tmp_path):
        (tmp_path / ub.FILENAME).write_text("garbage\n", encoding="utf-8")
        with pytest.raises(ub.UpstreamBaseError):
            ub.read_worktree(str(tmp_path))

    def test_ref_reads_the_committed_copy_not_the_worktree(self, forked):
        """``--ours`` must mean the commit, or an uncommitted edit changes it."""
        _write(forked.path, ub.FILENAME, f"v9.9 {'b' * 40}\n")
        assert ub.read_ref(_run, str(forked.path), "HEAD").sha == forked.r1

    def test_ref_returns_none_when_that_commit_predates_the_file(self, forked):
        assert ub.read_ref(_run, str(forked.path), forked.base) is None


class TestSelect:
    def test_absorbed_release_resolves_to_the_recorded_commit(self, forked):
        """The case ancestry gets wrong: merge-base against our tip gives B."""
        recorded = ub.read_worktree(str(forked.path))
        stale = _run(str(forked.path), "merge-base", "HEAD", "v1.0").stdout.strip()
        assert stale == forked.base, "fixture no longer reproduces the stale base"

        base, source = ub.select(_run, str(forked.path), "HEAD", "v1.0", recorded)
        assert (base, source) == (forked.r1, "recorded")

    def test_a_sibling_branch_resolves_to_the_real_divergence_point(self, forked):
        """`main` never merged back, so the answer is R0 -- not R1, and not B."""
        recorded = ub.read_worktree(str(forked.path))
        base, source = ub.select(_run, str(forked.path), "HEAD", "main", recorded)
        assert (base, source) == (forked.r0, "recorded")

    def test_without_a_record_it_falls_back_to_our_own_ancestry(self, forked):
        base, source = ub.select(_run, str(forked.path), "HEAD", "v1.0", None)
        assert (base, source) == (forked.base, "ancestry")

    def test_unrelated_history_names_the_ref_rather_than_failing_bare(self, forked):
        _git(forked.path, "checkout", "-q", "--orphan", "alien")
        _git(forked.path, "rm", "-rq", "--cached", ".")
        _write(forked.path, "z.cu", "z\n")
        _commit(forked.path, "unrelated root")
        with pytest.raises(ub.UpstreamBaseError, match="shares no history"):
            ub.select(_run, str(forked.path), "HEAD", "v1.0", None)


class TestRecordedCommitIsChecked:
    def test_a_sha_not_in_the_object_store_says_which_tag_to_fetch(self, forked):
        recorded = ub.UpstreamBase("v1.0", "b" * 40)
        with pytest.raises(ub.UpstreamBaseError, match="upstream-base/v1.0"):
            ub.select(_run, str(forked.path), "HEAD", "v1.0", recorded)

    def test_a_ref_that_moved_away_from_the_recorded_sha_is_loud(self, forked):
        """Catches a hand-edit, or upstream re-pointing a tag under us."""
        recorded = ub.UpstreamBase("v1.0", forked.r0)  # v1.0 is really r1
        with pytest.raises(ub.UpstreamBaseError, match="resolves to"):
            ub.select(_run, str(forked.path), "HEAD", "v1.0", recorded)


class TestMultipleBases:
    def test_single_base_is_not_announced(self, forked):
        assert not ub.multiple_bases(_run, str(forked.path), forked.r1, "main")


class TestShippedFile:
    """The repo's own record. A sync PR that forgets it reports a wrong number."""

    def test_it_parses_and_names_a_commit_we_have(self):
        recorded = ub.read_worktree(str(_REPO_ROOT))
        assert recorded is not None, f"{ub.FILENAME} is missing from the checkout"
        probe = _run(
            str(_REPO_ROOT),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{recorded.sha}^{{commit}}",
            check=False,
        )
        if probe.returncode:
            pytest.skip(f"{recorded.sha[:12]} not in this clone (shallow or unfetched)")
        assert probe.stdout.strip() == recorded.sha

    def test_the_ref_agrees_with_the_forks_own_release_tag(self):
        """`v0.6.18+amd.N` and `upstream-base` must name the same upstream release."""
        recorded = ub.read_worktree(str(_REPO_ROOT))
        described = _run(
            str(_REPO_ROOT),
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            "*+amd.*",
            check=False,
        )
        if described.returncode:
            pytest.skip("no reachable *+amd.* tag in this clone")
        assert described.stdout.strip().split("+amd.")[0] == recorded.ref


class TestShallowCloneAdviceDoesNotContradictItself:
    def test_a_missing_object_is_its_own_error_type(self, forked):
        """An explicit base needs presence, not reachability.

        amd_coverage appends `_unshallow_hint` ("fetching tags will not help") to
        every other base failure, which would contradict the remedy here.
        """
        recorded = ub.UpstreamBase("v1.0", "b" * 40)
        with pytest.raises(ub.MissingBaseObject):
            ub.select(_run, str(forked.path), "HEAD", "v1.0", recorded)

    def test_other_base_failures_stay_the_general_type(self, forked):
        recorded = ub.UpstreamBase("v1.0", forked.r0)
        with pytest.raises(ub.UpstreamBaseError) as excinfo:
            ub.select(_run, str(forked.path), "HEAD", "v1.0", recorded)
        assert not isinstance(excinfo.value, ub.MissingBaseObject)
