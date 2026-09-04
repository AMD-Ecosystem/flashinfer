# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""README matrix generation for ``scripts/gen_arch_support_matrix.py``.

The generated block is the user-facing answer to "does this op work on my card",
and ``--check`` runs as a pre-commit hook, so a renderer that drifts from
``flashinfer/arch_caps.py`` is wrong in both directions at once. These assert on
the emitted markdown rather than on the fact that something was emitted.

No GPU and no torch -- ``arch_caps`` is deliberately torch-free at module scope
-- so CI runs this file with ``--noconftest``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    target = _REPO_ROOT / "scripts" / "gen_arch_support_matrix.py"
    spec = importlib.util.spec_from_file_location("_fi_gen_arch_matrix", target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_tool()
caps = gen._load_arch_caps()


def _support(evidence="", known_bad=()):
    return caps.ArchSupport(
        support=caps.Support.SUPPORTED, evidence=evidence, known_bad=tuple(known_bad)
    )


def _unsupported():
    return caps.ArchSupport(support=caps.Support.UNSUPPORTED)


def _table(*capabilities):
    """A stand-in capability module holding just the rows a test cares about."""
    module = types.SimpleNamespace()
    module.CAPABILITIES = tuple(capabilities)
    return module


class TestLoadArchCaps:
    def test_loads_the_real_table_without_importing_the_package(self):
        """flashinfer/__init__.py raises on a CPU-only torch build."""
        assert caps.CAPABILITIES
        assert hasattr(caps, "Support")

    def test_missing_source_names_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gen, "PKG_DIR", tmp_path)
        with pytest.raises(SystemExit, match="cannot read the capability table"):
            gen._load_arch_caps()

    def test_absent_source_loader_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            gen.importlib.util, "spec_from_file_location", lambda *a, **k: None
        )
        with pytest.raises(SystemExit, match="no Python source loader"):
            gen._load_arch_caps()


class TestArchOrder:
    def test_columns_follow_first_appearance_not_sorting(self):
        table = _table(
            caps.Capability(op="a", backend="hip", archs={"gfx950": _support()}),
            caps.Capability(
                op="b",
                backend="hip",
                archs={"gfx942": _support(), "gfx950": _support()},
            ),
        )
        assert gen._archs_in_order(table) == ["gfx950", "gfx942"]


class TestWindow:
    def test_both_bounds_render_as_a_half_open_interval(self):
        bad = caps.KnownBad(rocm_min="7.2", rocm_max="7.3")
        assert gen._window(bad) == "ROCm [7.2, 7.3)"

    def test_open_lower_bound(self):
        assert gen._window(caps.KnownBad(rocm_max="7.3")) == "ROCm (-∞, 7.3)"

    def test_open_upper_bound(self):
        assert gen._window(caps.KnownBad(rocm_min="7.2")) == "ROCm [7.2, +∞)"

    def test_both_dimensions_are_joined(self):
        bad = caps.KnownBad(rocm_min="7.2", aiter_max="0.1.11")
        assert gen._window(bad) == "ROCm [7.2, +∞) and amd-aiter (-∞, 0.1.11)"

    def test_unbounded_window_says_all_versions(self):
        assert gen._window(caps.KnownBad()) == "all versions"


class TestCell:
    def test_absent_arch_is_unsupported(self):
        assert gen._cell(None, {}) == gen.UNSUPPORTED

    def test_declared_arch_marked_unsupported_is_unsupported(self):
        assert gen._cell(_unsupported(), {}) == gen.UNSUPPORTED

    def test_evidence_does_not_change_the_rendered_cell(self):
        """`evidence` is provenance, not a support tier -- a reader cannot act
        on whether someone pasted a measurement string."""
        assert gen._cell(_support(evidence="MI300X, ROCm 7.2"), {}) == gen.SUPPORTED
        assert gen._cell(_support(), {}) == gen.SUPPORTED

    def test_known_bad_wins_over_evidence_and_carries_its_footnotes(self):
        bad = caps.KnownBad(rocm_min="7.2")
        cell = gen._cell(_support(evidence="measured", known_bad=[bad]), {id(bad): 3})
        assert cell == f"{gen.KNOWN_BAD}[^kb3]"


class TestNoteValidation:
    def test_plain_note_passes_through_stripped(self):
        assert gen._note("op/backend", "  fine  ") == "fine"

    def test_pipe_is_rejected_even_inside_backticks(self):
        with pytest.raises(SystemExit, match="MD056"):
            gen._note("op/backend", "use `a | b`")

    def test_newline_is_rejected(self):
        with pytest.raises(SystemExit, match="one line"):
            gen._note("op/backend", "two\nlines")

    def test_bare_url_is_rejected(self):
        with pytest.raises(SystemExit, match="MD034"):
            gen._note("op/backend", "see https://example.invalid/x")

    def test_url_inside_a_code_span_is_allowed(self):
        assert gen._note("op/backend", "see `https://example.invalid/x`")

    def test_markdown_link_is_allowed(self):
        assert gen._note("op/backend", "see [docs](https://example.invalid/x)")

    def test_inline_html_is_rejected(self):
        with pytest.raises(SystemExit, match="MD033"):
            gen._note("op/backend", "line<br>break")

    def test_inline_html_inside_a_code_span_is_allowed(self):
        assert gen._note("op/backend", "write `<br>` for a break")


class TestRender:
    def test_header_uses_the_friendly_arch_labels(self):
        table = _table(
            caps.Capability(op="rope", backend="hip", archs={"gfx942": _support()})
        )
        assert "| gfx942 (CDNA3) |" in gen.render(table)

    def test_unlabelled_arch_renders_under_its_bare_name(self):
        table = _table(
            caps.Capability(op="rope", backend="hip", archs={"gfx1100": _support()})
        )
        assert "| gfx1100 |" in gen.render(table)

    def test_legend_lists_only_the_symbols_a_row_actually_uses(self):
        table = _table(
            caps.Capability(op="rope", backend="hip", archs={"gfx942": _support()})
        )
        out = gen.render(table)
        assert f"{gen.BULLET} {gen.SUPPORTED}" in out
        assert f"{gen.BULLET} {gen.UNSUPPORTED}" not in out

    def test_a_note_mentioning_a_symbol_does_not_add_a_legend_entry(self):
        """The legend is collected from status cells, not the rendered row."""
        table = _table(
            caps.Capability(
                op="rope",
                backend="hip",
                archs={"gfx942": _support()},
                note=f"unlike {gen.UNSUPPORTED} rows",
            )
        )
        assert f"{gen.BULLET} {gen.UNSUPPORTED}" not in gen.render(table)

    def test_no_footnotes_leaves_no_double_blank_line(self):
        """The whitespace pre-commit hook strips a doubled blank line, so an
        unconditional separator here makes --check unsatisfiable."""
        table = _table(
            caps.Capability(op="rope", backend="hip", archs={"gfx942": _support()})
        )
        out = gen.render(table)
        assert "\n\n\n" not in out
        assert out.endswith(gen.END)

    def test_footnotes_are_numbered_in_table_order(self):
        first = caps.KnownBad(rocm_min="7.2", detail="broken on CDNA4")
        second = caps.KnownBad(aiter_max="0.1.9", detail="old aiter")
        table = _table(
            caps.Capability(
                op="a", backend="hip", archs={"gfx950": _support(known_bad=[first])}
            ),
            caps.Capability(
                op="b", backend="hip", archs={"gfx950": _support(known_bad=[second])}
            ),
        )

        out = gen.render(table)

        assert "[^kb1]: `a/hip` on gfx950, ROCm [7.2, +∞): broken on CDNA4." in out
        assert "[^kb2]: `b/hip` on gfx950" in out
        assert out.index("[^kb1]:") < out.index("[^kb2]:")

    def test_one_window_shared_by_two_rows_gets_a_single_footnote(self):
        shared = caps.KnownBad(rocm_min="7.2", detail="one defect")
        table = _table(
            caps.Capability(
                op="a", backend="hip", archs={"gfx950": _support(known_bad=[shared])}
            ),
            caps.Capability(
                op="b", backend="hip", archs={"gfx950": _support(known_bad=[shared])}
            ),
        )

        out = gen.render(table)

        assert out.count("[^kb1]:") == 1
        assert "[^kb2]" not in out

    def test_footnote_appends_the_url_when_one_is_recorded(self):
        bad = caps.KnownBad(
            rocm_min="7.2", detail="see the tracker", url="https://example.invalid/1"
        )
        table = _table(
            caps.Capability(
                op="a", backend="hip", archs={"gfx950": _support(known_bad=[bad])}
            )
        )
        assert "<https://example.invalid/1>" in gen.render(table)

    def test_the_real_table_renders_and_is_delimited(self):
        out = gen.render(caps)
        assert out.startswith(gen.BEGIN)
        assert out.endswith(gen.END)


class TestSplice:
    def test_replaces_only_the_marked_block(self):
        text = f"before\n{gen.BEGIN}\nold\n{gen.END}\nafter\n"
        assert gen.splice(text, f"{gen.BEGIN}\nnew\n{gen.END}") == (
            f"before\n{gen.BEGIN}\nnew\n{gen.END}\nafter\n"
        )

    @pytest.mark.parametrize("text", ["no markers", "only the BEGIN marker"])
    def test_missing_markers_are_fatal(self, text):
        with pytest.raises(SystemExit, match="missing generated-block markers"):
            gen.splice(text, "block")


@pytest.fixture
def readme(tmp_path, monkeypatch):
    """A throwaway README, so no test can rewrite the repository's own."""
    path = tmp_path / "README.md"
    monkeypatch.setattr(gen, "README", path)
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    path.write_text(
        f"intro\n\n{gen.BEGIN}\nstale\n{gen.END}\n\noutro\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def argv(monkeypatch):
    """main() reads sys.argv directly; hand it one."""

    def _set(*args):
        monkeypatch.setattr(sys, "argv", ["gen_arch_support_matrix.py", *args])

    return _set


class TestMain:
    def test_write_mode_updates_a_stale_readme(self, readme, argv, capsys):
        argv()

        assert gen.main() == 0
        out = capsys.readouterr().out

        assert "updated README.md" in out
        assert "stale" not in readme.read_text(encoding="utf-8")
        assert readme.read_text(encoding="utf-8").startswith("intro")

    def test_write_mode_is_idempotent(self, readme, argv, capsys):
        argv()
        gen.main()
        capsys.readouterr()

        assert gen.main() == 0
        assert "already up to date" in capsys.readouterr().out

    def test_check_passes_on_a_current_readme(self, readme, argv, capsys):
        argv()
        gen.main()
        capsys.readouterr()
        argv("--check")

        assert gen.main() == 0
        assert capsys.readouterr().err == ""

    def test_check_fails_with_a_diff_and_leaves_the_file_alone(
        self, readme, argv, capsys
    ):
        before = readme.read_text(encoding="utf-8")
        argv("--check")

        assert gen.main() == 1
        err = capsys.readouterr().err

        assert "out of date" in err
        assert "--- README.md (committed)" in err
        assert readme.read_text(encoding="utf-8") == before

    def test_filenames_from_pre_commit_are_accepted_and_ignored(
        self, readme, argv, capsys
    ):
        argv("README.md", "flashinfer/arch_caps.py")

        assert gen.main() == 0
        assert "updated README.md" in capsys.readouterr().out

    def test_repository_readme_is_in_sync_with_the_capability_table(self):
        """The pre-commit hook's own assertion, run here so a stale README is
        caught even by someone who committed with hooks disabled."""
        current = gen.README.read_text(encoding="utf-8")
        assert gen.splice(current, gen.render(caps)) == current
