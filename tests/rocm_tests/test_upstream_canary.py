# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Merge-cost reporting for ``scripts/upstream_canary.py``.

The canary decides whether an upstream sync is affordable, so a conflict it
fails to report is the expensive direction of wrong. These build throwaway git
repositories and assert on the reported paths, kinds and exit codes.

Git and filesystem only -- no GPU, no torch -- but collection still pulls in the
suite's torch-importing conftest, so CI runs this file with ``--noconftest``.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    target = _REPO_ROOT / "scripts" / "upstream_canary.py"
    spec = importlib.util.spec_from_file_location("_fi_upstream_canary", target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


uc = _load_tool()


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def _write(repo, rel, text):
    path = Path(repo) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit(repo, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)


def _edit_line(repo, rel, index, text):
    path = Path(repo) / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[index] = text
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rev(repo, ref="HEAD"):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _Proc:
    """Stand-in for CompletedProcess, for the stream-corruption paths."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def repo(tmp_path):
    """A repo with a base commit and `ours`/`theirs` branches off it."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _write(tmp_path, "shared.cu", "\n".join(f"line {i}" for i in range(20)) + "\n")
    _write(tmp_path, "README.md", "base readme\n")
    _commit(tmp_path, "base")
    _git(tmp_path, "branch", "theirs")
    return tmp_path


class TestChurn:
    def test_str_and_total(self):
        c = uc.Churn(added=3, deleted=4)
        assert str(c) == "+3/-4"
        assert c.total == 7

    def test_defaults_to_zero(self):
        assert uc.Churn().total == 0


class TestConflictClassification:
    def test_weight_multiplies_both_sides_so_two_sided_ranks_first(self):
        two_sided = uc.Conflict("a.cu", uc.Churn(2, 2), uc.Churn(2, 2), "contents")
        one_sided = uc.Conflict("b.cu", uc.Churn(50, 50), uc.Churn(), "contents")
        assert two_sided.weight > one_sided.weight

    @pytest.mark.parametrize(
        "path",
        ["README.md", "docs/deep/notes.md", "pyproject.toml", "sub/x.toml"],
    )
    def test_basename_globs_are_expected(self, path):
        assert uc.Conflict(path, uc.Churn(), uc.Churn(), "contents").expected

    @pytest.mark.parametrize(
        "path", [".gitignore", ".pre-commit-config.yaml", "version.txt"]
    )
    def test_exact_paths_are_expected(self, path):
        assert uc.Conflict(path, uc.Churn(), uc.Churn(), "contents").expected

    def test_prefix_paths_are_expected(self):
        c = uc.Conflict(
            "benchmarks/samples/out.txt", uc.Churn(), uc.Churn(), "contents"
        )
        assert c.expected

    @pytest.mark.parametrize(
        "path",
        [
            "flashinfer/prefill.py",
            ".github/workflows/ci.yml",
            "docker/Dockerfile",
            "nested/.gitignore",
        ],
    )
    def test_code_and_infra_paths_are_not_expected(self, path):
        """An exact-match entry must not match the same basename in a subdirectory."""
        assert not uc.Conflict(path, uc.Churn(), uc.Churn(), "contents").expected


class TestGitHelpers:
    def test_run_raises_tool_error_carrying_stderr(self, repo):
        with pytest.raises(uc.ToolError) as excinfo:
            uc._run(str(repo), "rev-parse", "--verify", "no/such/ref")
        assert "no/such/ref" in str(excinfo.value)

    def test_run_without_check_returns_the_failure(self, repo):
        proc = uc._run(str(repo), "rev-parse", "--verify", "nope", check=False)
        assert proc.returncode != 0

    def test_git_strips_trailing_newline(self, repo):
        assert uc._git(str(repo), "rev-parse", "HEAD") == _rev(repo)

    def test_ls_tree_scopes_to_a_path_and_drops_empty_fields(self, repo):
        _write(repo, "include/flashinfer/rocm/a.cuh", "// a\n")
        _write(repo, "include/flashinfer/rocm/b.cuh", "// b\n")
        _commit(repo, "headers")

        scoped = uc._ls_tree(str(repo), "HEAD", "include/flashinfer/rocm")
        assert scoped == {
            "include/flashinfer/rocm/a.cuh",
            "include/flashinfer/rocm/b.cuh",
        }
        assert "shared.cu" in uc._ls_tree(str(repo), "HEAD", "")


class TestChurnMap:
    def test_counts_additions_and_deletions(self, repo):
        base = _rev(repo)
        _write(
            repo,
            "shared.cu",
            "changed\n" + "\n".join(f"line {i}" for i in range(19)) + "\n",
        )
        _commit(repo, "edit")

        churn, renames = uc._churn_map(str(repo), base, "HEAD")

        assert churn["shared.cu"].total > 0
        assert renames == {}

    def test_binary_file_dashes_count_as_zero(self, repo):
        base = _rev(repo)
        (Path(repo) / "blob.bin").write_bytes(bytes(range(256)))
        _commit(repo, "binary")

        churn, _ = uc._churn_map(str(repo), base, "HEAD")

        assert churn["blob.bin"] == uc.Churn(0, 0)

    def test_rename_maps_new_path_back_to_old(self, repo):
        base = _rev(repo)
        _git(repo, "mv", "shared.cu", "moved.cu")
        _commit(repo, "rename")

        churn, renames = uc._churn_map(str(repo), base, "HEAD")

        assert renames == {"moved.cu": "shared.cu"}
        assert "moved.cu" in churn

    def test_malformed_record_raises_rather_than_undercounting(self, monkeypatch):
        monkeypatch.setattr(uc, "_run", lambda *a, **k: _Proc(stdout="oops\0"))
        with pytest.raises(uc.ToolError, match="malformed numstat"):
            uc._churn_map("/nowhere", "a", "b")

    def test_truncated_rename_record_raises(self, monkeypatch):
        # Rename record with the old path present but the new path missing.
        monkeypatch.setattr(uc, "_run", lambda *a, **k: _Proc(stdout="1\t1\t\0old.cu"))
        with pytest.raises(uc.ToolError, match="truncated numstat"):
            uc._churn_map("/nowhere", "a", "b")


def _merge_tree_stream(*records):
    """A merge-tree -z payload: tree OID, empty terminator, then records."""
    return "\0".join(("treeoid", "") + records) + "\0"


class TestMergeTree:
    def test_clean_merge_reports_nothing(self, repo):
        _git(repo, "checkout", "-q", "theirs")
        _write(repo, "theirs_only.cu", "new\n")
        _commit(repo, "upstream")
        _git(repo, "checkout", "-q", "main")

        assert uc._merge_tree(str(repo), "main", "theirs") == []

    def test_content_conflict_is_reported_with_its_kind(self, repo):
        _git(repo, "checkout", "-q", "theirs")
        _write(repo, "shared.cu", "upstream version\n")
        _commit(repo, "upstream edit")
        _git(repo, "checkout", "-q", "main")
        _write(repo, "shared.cu", "our version\n")
        _commit(repo, "our edit")

        result = uc._merge_tree(str(repo), "main", "theirs")

        assert [p for p, _ in result] == ["shared.cu"]
        assert result[0][1] == "contents"

    def test_git_failure_other_than_conflict_raises(self, monkeypatch):
        monkeypatch.setattr(
            uc, "_run", lambda *a, **k: _Proc(stderr="boom", returncode=128)
        )
        with pytest.raises(uc.ToolError, match="merge-tree failed"):
            uc._merge_tree("/nowhere", "a", "b")

    def test_non_conflict_records_are_skipped(self, monkeypatch):
        stream = _merge_tree_stream("1", "shared.cu", "Auto-merging", "msg")
        monkeypatch.setattr(
            uc, "_run", lambda *a, **k: _Proc(stdout=stream, returncode=1)
        )
        assert uc._merge_tree("/nowhere", "a", "b") == []

    def test_multi_path_record_reports_every_path_once(self, monkeypatch):
        stream = _merge_tree_stream(
            "2",
            "old.cu",
            "new.cu",
            "CONFLICT (rename/delete)",
            "msg",
            "1",
            "new.cu",
            "CONFLICT (contents)",
            "msg",
        )
        monkeypatch.setattr(
            uc, "_run", lambda *a, **k: _Proc(stdout=stream, returncode=1)
        )

        result = uc._merge_tree("/nowhere", "a", "b")

        # First-seen order is kept, but the later kind wins for a repeated path.
        assert [p for p, _ in result] == ["old.cu", "new.cu"]
        assert dict(result)["new.cu"] == "contents"

    def test_unparsable_count_raises_rather_than_reporting_fewer(self, monkeypatch):
        stream = _merge_tree_stream("notanumber", "x")
        monkeypatch.setattr(
            uc,
            "_run",
            lambda repo, *a, **k: _Proc(stdout=stream, returncode=1)
            if a[0] == "merge-tree"
            else _Proc(stdout="git version 2.43.0\n"),
        )
        with pytest.raises(uc.ToolError, match="unparsed merge-tree"):
            uc._merge_tree("/nowhere", "a", "b")

    def test_truncated_record_raises(self, monkeypatch):
        # Claims two paths but supplies one, and no type/message follow.
        stream = _merge_tree_stream("2", "only.cu")
        monkeypatch.setattr(
            uc, "_run", lambda *a, **k: _Proc(stdout=stream, returncode=1)
        )
        with pytest.raises(uc.ToolError, match="truncated merge-tree"):
            uc._merge_tree("/nowhere", "a", "b")

    def test_kind_without_parentheses_is_kept_verbatim(self, monkeypatch):
        stream = _merge_tree_stream("1", "x.cu", "CONFLICT", "msg")
        monkeypatch.setattr(
            uc, "_run", lambda *a, **k: _Proc(stdout=stream, returncode=1)
        )
        assert uc._merge_tree("/nowhere", "a", "b") == [("x.cu", "CONFLICT")]


class TestReporting:
    def test_position_prints_both_sides(self, capsys):
        uc._report_position("abc123 2026-01-01 base", (4, 9), (7, 11))
        out = capsys.readouterr().out
        assert "abc123 2026-01-01 base" in out
        assert "4" in out and "9" in out and "7" in out and "11" in out

    def test_clean_merge_reports_zero_code_conflicts(self, capsys):
        assert uc._report_conflicts([]) == 0
        assert "clean merge" in capsys.readouterr().out

    def test_expected_paths_are_tagged_and_excluded_from_the_count(self, capsys):
        conflicts = [
            uc.Conflict("README.md", uc.Churn(1, 1), uc.Churn(1, 1), "contents"),
            uc.Conflict("flashinfer/a.py", uc.Churn(2, 2), uc.Churn(2, 2), "contents"),
        ]

        code = uc._report_conflicts(conflicts)
        out = capsys.readouterr().out

        assert code == 1
        assert "[expected]" in out
        assert "2 conflicted, 1 of them code" in out
        # Code sorts above expected regardless of input order.
        assert out.index("flashinfer/a.py") < out.index("README.md")

    def test_non_contents_kind_is_tagged(self, capsys):
        uc._report_conflicts(
            [uc.Conflict("a.cu", uc.Churn(), uc.Churn(), "rename/delete")]
        )
        assert "[rename/delete]" in capsys.readouterr().out

    def test_higher_weight_sorts_first_within_code(self, capsys):
        uc._report_conflicts(
            [
                uc.Conflict("light.py", uc.Churn(1, 0), uc.Churn(1, 0), "contents"),
                uc.Conflict("heavy.py", uc.Churn(9, 9), uc.Churn(9, 9), "contents"),
            ]
        )
        out = capsys.readouterr().out
        assert out.index("heavy.py") < out.index("light.py")


class TestDrift:
    def _forked(self, repo, *names):
        for name in names:
            _write(repo, f"{uc._FORKED_DIR}/{name}", "// forked\n")

    def test_no_forked_headers_says_so(self, repo, capsys):
        uc._report_drift(str(repo), "HEAD", "HEAD", {})
        assert "no forked headers" in capsys.readouterr().out

    def test_changed_counterparts_are_listed_with_upstream_churn(self, repo, capsys):
        self._forked(repo, "page.cuh")
        _write(repo, "include/flashinfer/attention/page.cuh", "// upstream\n")
        _commit(repo, "headers")

        uc._report_drift(
            str(repo),
            "HEAD",
            "HEAD",
            {"include/flashinfer/attention/page.cuh": uc.Churn(5, 2)},
        )
        out = capsys.readouterr().out

        assert "include/flashinfer/attention/page.cuh" in out
        assert "+5/-2" in out
        assert "1 of 1 forked headers have upstream changes" in out

    def test_unchanged_counterparts_report_no_changes(self, repo, capsys):
        self._forked(repo, "page.cuh")
        _write(repo, "include/flashinfer/attention/page.cuh", "// upstream\n")
        _commit(repo, "headers")

        uc._report_drift(str(repo), "HEAD", "HEAD", {})

        assert (
            "no upstream changes to any of the 1 forked headers"
            in capsys.readouterr().out
        )

    def test_zero_churn_entry_is_not_listed_as_changed(self, repo, capsys):
        self._forked(repo, "page.cuh")
        _write(repo, "include/flashinfer/attention/page.cuh", "// upstream\n")
        _commit(repo, "headers")

        uc._report_drift(
            str(repo),
            "HEAD",
            "HEAD",
            {"include/flashinfer/attention/page.cuh": uc.Churn(0, 0)},
        )

        assert "no upstream changes" in capsys.readouterr().out

    def test_rocm_only_header_is_reported_as_an_orphan(self, repo, capsys):
        self._forked(repo, "rocm_only.cuh")
        _commit(repo, "headers")

        uc._report_drift(str(repo), "HEAD", "HEAD", {})
        out = capsys.readouterr().out

        assert "1 ROCm-only" in out
        assert "rocm_only.cuh" in out

    def test_ambiguous_counterpart_notes_the_choice(self, repo, capsys):
        """The same basename in both upstream dirs must not be resolved silently."""
        self._forked(repo, "page.cuh")
        _write(repo, "include/flashinfer/attention/page.cuh", "// a\n")
        _write(repo, "include/flashinfer/page.cuh", "// b\n")
        _commit(repo, "headers")

        uc._report_drift(str(repo), "HEAD", "HEAD", {})
        out = capsys.readouterr().out

        assert "matches 2 upstream paths" in out
        assert "using include/flashinfer/attention/page.cuh" in out

    def test_non_header_suffixes_are_ignored(self, repo, capsys):
        _write(repo, f"{uc._FORKED_DIR}/notes.txt", "not a header\n")
        _commit(repo, "headers")

        uc._report_drift(str(repo), "HEAD", "HEAD", {})

        assert "no forked headers" in capsys.readouterr().out

    def test_excluded_header_is_not_paired_against_upstream(self, repo, capsys):
        """utils.cuh shares upstream's basename but forks nothing.

        Without the exclusion the pairing reports upstream churn against a HIP
        rewrite that never tracked it.
        """
        for path in uc._FORKED_EXCLUDE:
            _write(repo, path, "// hip rewrite\n")
            _write(
                repo, f"include/flashinfer/{path.rsplit('/', 1)[1]}", "// upstream\n"
            )
        _commit(repo, "headers")

        uc._report_drift(
            str(repo),
            "HEAD",
            "HEAD",
            {
                f"include/flashinfer/{p.rsplit('/', 1)[1]}": uc.Churn(5, 5)
                for p in uc._FORKED_EXCLUDE
            },
        )
        out = capsys.readouterr().out

        assert "no forked headers" in out
        assert "utils.cuh" not in out

    def test_a_sibling_of_an_excluded_header_is_still_paired(self, repo, capsys):
        """The exclusion is exact paths, not a prefix or a basename match."""
        self._forked(repo, "layout.cuh")
        _write(repo, "include/flashinfer/layout.cuh", "// upstream\n")
        _commit(repo, "headers")

        uc._report_drift(
            str(repo),
            "HEAD",
            "HEAD",
            {"include/flashinfer/layout.cuh": uc.Churn(4, 4)},
        )

        assert (
            "include/flashinfer/layout.cuh  upstream +4/-4" in capsys.readouterr().out
        )


class TestResolve:
    def test_returns_the_full_oid(self, repo):
        assert uc._resolve(str(repo), "HEAD", "--ours") == _rev(repo)

    def test_unknown_upstream_ref_suggests_adding_the_remote(self, repo):
        with pytest.raises(uc.ToolError) as excinfo:
            uc._resolve(str(repo), "upstream/main", "--upstream-ref")
        message = str(excinfo.value)
        assert "unknown ref 'upstream/main'" in message
        assert "git remote add upstream" in message

    def test_unknown_ours_ref_omits_the_remote_hint(self, repo):
        with pytest.raises(uc.ToolError) as excinfo:
            uc._resolve(str(repo), "no-such-branch", "--ours")
        assert "git remote add upstream" not in str(excinfo.value)


def _diverge(repo):
    """Give `theirs` and `main` one conflicting edit and one expected-file edit."""
    _git(repo, "checkout", "-q", "theirs")
    _write(repo, "shared.cu", "upstream version\n")
    _write(repo, "README.md", "upstream readme\n")
    _commit(repo, "upstream")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "shared.cu", "our version\n")
    _write(repo, "README.md", "our readme\n")
    _commit(repo, "ours")


def _args(**overrides):
    defaults = {"upstream_ref": "theirs", "ours": "HEAD", "fail_over": None}
    defaults.update(overrides)
    return type("Args", (), defaults)()


class TestRun:
    def test_reports_conflicts_and_exits_clean_without_a_ratchet(
        self, repo, monkeypatch, capsys
    ):
        _diverge(repo)
        monkeypatch.chdir(repo)

        assert uc.run(_args()) == uc.EXIT_OK
        out = capsys.readouterr().out

        assert "== position ==" in out
        assert "shared.cu" in out
        assert "2 conflicted, 1 of them code" in out

    def test_fail_over_trips_on_code_conflicts_only(self, repo, monkeypatch, capsys):
        _diverge(repo)
        monkeypatch.chdir(repo)

        assert uc.run(_args(fail_over=0)) == uc.EXIT_RATCHET
        assert "FAIL: 1 code conflicts exceeds --fail-over 0" in capsys.readouterr().out

    def test_fail_over_at_the_limit_passes(self, repo, monkeypatch):
        _diverge(repo)
        monkeypatch.chdir(repo)

        assert uc.run(_args(fail_over=1)) == uc.EXIT_OK

    def test_clean_merge_runs_to_completion(self, repo, monkeypatch, capsys):
        _git(repo, "checkout", "-q", "theirs")
        _write(repo, "theirs_only.cu", "new\n")
        _commit(repo, "upstream")
        _git(repo, "checkout", "-q", "main")
        monkeypatch.chdir(repo)

        assert uc.run(_args()) == uc.EXIT_OK
        assert "clean merge" in capsys.readouterr().out

    def test_upstream_rename_is_attributed_to_the_name_we_edited(
        self, repo, monkeypatch, capsys
    ):
        """Without following the rename the costliest conflict reports +0/-0.

        The edits stay small on purpose: rewrite the file and git stops detecting
        the rename, reporting modify/delete against the old name instead.
        """
        _git(repo, "checkout", "-q", "theirs")
        _git(repo, "mv", "shared.cu", "renamed.cu")
        _edit_line(repo, "renamed.cu", 2, "upstream line")
        _commit(repo, "upstream rename")
        _git(repo, "checkout", "-q", "main")
        _edit_line(repo, "shared.cu", 2, "our line")
        _commit(repo, "ours")
        monkeypatch.chdir(repo)

        uc.run(_args())
        out = capsys.readouterr().out

        assert "renamed.cu" in out
        assert "[rename]" in out
        # Read the churn back rather than matching a padded column: the width is
        # computed from the longest path, so a literal is both brittle and, at
        # the wrong space count, unable to fail.
        ours = re.search(r"renamed\.cu\s+ours\s+(\S+)", out)
        assert ours is not None, out
        assert ours.group(1) != "+0/-0"

    def test_add_add_conflict_is_labelled(self, repo, monkeypatch, capsys):
        _git(repo, "checkout", "-q", "theirs")
        _write(repo, "both.cu", "upstream\n")
        _commit(repo, "upstream")
        _git(repo, "checkout", "-q", "main")
        _write(repo, "both.cu", "ours\n")
        _commit(repo, "ours")
        monkeypatch.chdir(repo)

        uc.run(_args())

        assert "[add/add]" in capsys.readouterr().out

    def test_multiple_merge_bases_are_announced(self, repo, monkeypatch, capsys):
        """Criss-cross history: churn and the merge itself use different bases.

        Each side merges the other's *pre-merge* tip. Merging the merge instead
        makes one branch an ancestor of the other and leaves a single base.
        """
        _git(repo, "checkout", "-q", "theirs")
        _write(repo, "t.cu", "t\n")
        _commit(repo, "t")
        theirs_tip = _rev(repo)
        _git(repo, "checkout", "-q", "main")
        _write(repo, "m.cu", "m\n")
        _commit(repo, "m")
        main_tip = _rev(repo)
        _git(repo, "merge", "-q", "--no-edit", theirs_tip)
        _git(repo, "checkout", "-q", "theirs")
        _git(repo, "merge", "-q", "--no-edit", main_tip)
        _git(repo, "checkout", "-q", "main")
        monkeypatch.chdir(repo)

        uc.run(_args())

        assert "multiple merge bases" in capsys.readouterr().out


class TestMain:
    def test_tool_error_becomes_exit_error(self, repo, monkeypatch, capsys):
        monkeypatch.chdir(repo)
        monkeypatch.setattr(
            sys, "argv", ["upstream_canary.py", "--upstream-ref", "nope"]
        )

        assert uc.main() == uc.EXIT_ERROR
        assert "error: unknown ref 'nope'" in capsys.readouterr().err

    def test_defaults_are_wired_through_to_run(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        monkeypatch.setattr(sys, "argv", ["upstream_canary.py"])
        seen = {}

        def _capture(args):
            seen["args"] = args
            return uc.EXIT_OK

        monkeypatch.setattr(uc, "run", _capture)

        assert uc.main() == uc.EXIT_OK
        assert seen["args"].upstream_ref == "upstream/main"
        assert seen["args"].ours == "HEAD"
        assert seen["args"].fail_over is None

    def test_fail_over_is_parsed_as_an_int(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        monkeypatch.setattr(
            sys, "argv", ["upstream_canary.py", "--fail-over", "14", "--ours", "main"]
        )
        seen = {}
        monkeypatch.setattr(uc, "run", lambda a: seen.setdefault("a", a) and uc.EXIT_OK)

        uc.main()

        assert seen["a"].fail_over == 14
        assert seen["a"].ours == "main"
