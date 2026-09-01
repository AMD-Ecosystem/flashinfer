# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Version-string derivation for ``scripts/git_describe_rocm.py``.

setuptools_scm calls this to turn a ``v0.5.3+amd.2`` style tag into a PEP 440
version. Getting it wrong renames the wheel, so the cases below pin the exact
string for each tag shape rather than just checking it produced something.

Git and filesystem only -- no GPU, no torch -- so CI runs it with ``--noconftest``.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    target = _REPO_ROOT / "scripts" / "git_describe_rocm.py"
    spec = importlib.util.spec_from_file_location("_fi_git_describe_rocm", target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gd = _load_tool()


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def _commit(repo, msg):
    (Path(repo) / msg).write_text(msg, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)


def _short_head(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty repo with one commit, already the process working directory.

    The script shells out to git without ``-C``, so cwd is the only handle it has.
    """
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _commit(tmp_path, "base")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestNoUsableTag:
    def test_no_tags_at_all_exits_one_with_the_placeholder(self, repo, capsys):
        """This path calls sys.exit rather than returning, unlike every other."""
        with pytest.raises(SystemExit) as excinfo:
            gd.main()

        assert excinfo.value.code == 1
        assert capsys.readouterr().out.strip() == "v0.0.0-0-g0000000"

    def test_tag_that_is_not_an_ancestor_is_skipped(self, repo, capsys):
        _git(repo, "checkout", "-q", "-b", "side")
        _commit(repo, "side-commit")
        _git(repo, "tag", "v9.9.9")
        _git(repo, "checkout", "-q", "main")

        assert gd.main() == 1
        assert capsys.readouterr().out.strip() == "v0.0.0-0-g0000000"

    def test_non_v_tags_are_not_considered(self, repo, capsys):
        _git(repo, "tag", "release-1")

        with pytest.raises(SystemExit):
            gd.main()

        assert capsys.readouterr().out.strip() == "v0.0.0-0-g0000000"


class TestPlainTag:
    def test_exact_tag_reports_zero_distance(self, repo, capsys):
        _git(repo, "tag", "v1.2.3")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip() == f"v1.2.3-0-g{_short_head(repo)}"

    def test_distance_is_the_commit_count_since_the_tag(self, repo, capsys):
        _git(repo, "tag", "v1.2.3")
        _commit(repo, "one")
        _commit(repo, "two")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip() == f"v1.2.3-2-g{_short_head(repo)}"


class TestLocalVersionTag:
    def test_exact_local_tag_keeps_the_standard_format(self, repo, capsys):
        """At distance 0 there is no .devN to embed, so the '+' path is not taken."""
        _git(repo, "tag", "v0.5.3+amd.2")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip() == f"v0.5.3+amd.2-0-g{_short_head(repo)}"

    def test_distance_moves_into_the_local_part_and_resets_to_zero(self, repo, capsys):
        """setuptools_scm would otherwise append its own suffix to the local part."""
        _git(repo, "tag", "v0.5.3+amd.2")
        _commit(repo, "one")
        _commit(repo, "two")
        _commit(repo, "three")

        assert gd.main() == 0
        assert (
            capsys.readouterr().out.strip()
            == f"v0.5.3+amd.2.dev3-0-g{_short_head(repo)}"
        )

    def test_only_the_first_plus_splits_the_tag(self, repo, capsys):
        _git(repo, "tag", "v1.0.0+a+b")
        _commit(repo, "one")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip().startswith("v1.0.0+a+b.dev1-0-g")


class TestClosestTagSelection:
    def test_the_nearest_ancestor_tag_wins(self, repo, capsys):
        _git(repo, "tag", "v1.0.0")
        _commit(repo, "one")
        _git(repo, "tag", "v2.0.0")
        _commit(repo, "two")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip() == f"v2.0.0-1-g{_short_head(repo)}"

    def test_tags_on_one_commit_resolve_to_the_lowest_version(self, repo, capsys):
        """Ties break toward the last tag seen, and `--sort=-version:refname`
        lists versions descending, so the oldest of a co-located pair wins."""
        _git(repo, "tag", "v1.0.0")
        _git(repo, "tag", "v2.0.0")

        assert gd.main() == 0
        assert capsys.readouterr().out.strip() == f"v1.0.0-0-g{_short_head(repo)}"

    def test_a_nearer_upstream_tag_does_not_displace_the_fork_tag(self, repo, capsys):
        """Merging an upstream release makes its bare tag a nearer ancestor than
        the fork's own release tag. Picking it would emit a version with no
        `+amd` local segment -- indistinguishable from upstream's wheel, and
        with nothing to signal it. The fork tag wins at any distance."""
        _git(repo, "tag", "v0.5.3+amd.2")
        _commit(repo, "one")
        _git(repo, "tag", "v0.6.18")
        _commit(repo, "two")

        assert gd.main() == 0
        assert (
            capsys.readouterr().out.strip()
            == f"v0.5.3+amd.2.dev2-0-g{_short_head(repo)}"
        )


class TestFailure:
    def test_git_failure_is_reported_on_stderr_and_returns_one(
        self, tmp_path, monkeypatch, capsys
    ):
        """Outside a repository `git tag` fails; the wrapper must not traceback."""
        monkeypatch.chdir(tmp_path)

        assert gd.main() == 1
        assert capsys.readouterr().err.startswith("Error:")

    def test_an_unreadable_distance_fails_loudly_rather_than_skipping_the_tag(
        self, repo, monkeypatch, capsys
    ):
        """Only CalledProcessError means "not an ancestor". Anything else here is
        a broken git, and treating it as a miss would invent a version."""
        _git(repo, "tag", "v1.2.3")
        real = gd.subprocess.run

        def _garbled(cmd, **kwargs):
            if "rev-list" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="not-a-number\n", stderr=""
                )
            return real(cmd, **kwargs)

        monkeypatch.setattr(gd.subprocess, "run", _garbled)

        assert gd.main() == 1
        assert capsys.readouterr().err.startswith("Error:")
