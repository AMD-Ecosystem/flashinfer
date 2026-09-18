# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ROCm twins must present upstream's decode/prefill/MLA signatures.

A vLLM or SGLang caller reaches ``flashinfer/rocm/{decode,prefill,mla}.py``
through the shadow install, so an upstream parameter the twin does not declare
is a ``TypeError`` at their call site, and one declared in the wrong position
binds the wrong value silently.

Deliberately free of torch and of ``import flashinfer``, so the guard runs in
the hardware-less lane on every pull request -- gating a sync is the whole point
of it, and the GPU suite runs on no PR. The runtime half that asserts each
CUDA-only argument raises lives in ``test_api_parity_runtime.py``.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    name = "_fi_rocm_api_parity"
    target = _REPO_ROOT / "scripts" / "rocm_api_parity.py"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None, f"cannot load {target}"
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises on a module that is not there yet.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = _load_tool()


class TestStaticParity:
    def test_no_divergence_from_upstream(self):
        findings, _ = parity.audit(_REPO_ROOT)
        assert findings == [], "\n".join(
            f"{f.kind}: {f.where} -- {f.detail}" for f in findings
        )

    def test_no_stale_allowlist_entries(self):
        assert parity.stale_allowlist_entries(_REPO_ROOT) == []

    def test_a_dropped_parameter_is_reported(self, tmp_path):
        """The guard is only worth having if removing a kwarg fails it."""
        root = _make_shadow_tree(tmp_path)
        target = root / "flashinfer/rocm/decode.py"
        text = target.read_text()
        dropped = "        kv_cache_sf: Optional[torch.Tensor] = None,\n"
        assert dropped in text
        # Every occurrence: the audit reads the implementation, and leaving it
        # while stripping the @overload stubs would not change what binds.
        target.write_text(text.replace(dropped, ""))

        findings, _ = parity.audit(root)
        assert any(
            f.kind == "missing-param" and "kv_cache_sf" in f.detail for f in findings
        ), findings

    def test_a_reordered_parameter_is_reported(self, tmp_path):
        """Mis-binds are the failure the ordering check exists for."""
        root = _make_shadow_tree(tmp_path)
        target = root / "flashinfer/rocm/mla.py"
        text = target.read_text()
        pair = (
            "        use_cuda_graph: bool = False,\n"
            "        qo_indptr: Optional[torch.Tensor] = None,\n"
        )
        assert pair in text
        swapped = (
            "        qo_indptr: Optional[torch.Tensor] = None,\n"
            "        use_cuda_graph: bool = False,\n"
        )
        target.write_text(text.replace(pair, swapped, 1))

        findings, _ = parity.audit(root)
        assert any(f.kind == "misbind" for f in findings), findings

    def test_a_stale_legacy_positional_copy_is_reported(self, tmp_path):
        root = _make_shadow_tree(tmp_path)
        target = root / "flashinfer/rocm/decode.py"
        text = target.read_text()
        assert '    "o_data_type",\n' in text
        target.write_text(text.replace('    "o_data_type",\n', "", 1))

        findings, _ = parity.audit(root)
        assert any(f.kind == "stale-copy" for f in findings), findings


def _make_shadow_tree(tmp_path):
    """Copy only the files the audit reads, so a mutation cannot touch the repo."""
    root = tmp_path / "tree"
    for rel in (
        "flashinfer/decode.py",
        "flashinfer/prefill.py",
        "flashinfer/mla/_core.py",
        "flashinfer/rocm/decode.py",
        "flashinfer/rocm/prefill.py",
        "flashinfer/rocm/mla.py",
    ):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((_REPO_ROOT / rel).read_text())
    return root


def test_the_guard_stays_importable_without_torch():
    """This file runs in the hardware-less lane, which installs no torch.

    Checked in a subprocess with torch blocked: ``sys.modules`` is process-wide,
    so an in-process check would trip merely because the GPU half ran first.
    """
    probe = textwrap.dedent(
        f"""
        import sys, importlib.util
        sys.modules["torch"] = None          # any `import torch` now raises
        for name, path in (
            ("_probe_tool", {str(_REPO_ROOT / "scripts" / "rocm_api_parity.py")!r}),
            ("_probe_test", {str(Path(__file__).resolve())!r}),
        ):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        """
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_rocm_api_all_matches_the_re_exported_imports():
    """`__all__` is what `from .rocm.api import *` publishes as `flashinfer.*`.

    Parsed rather than imported so this stays in the torch-free lane. A
    re-export is either the ``X as X`` idiom or an annotated module-level
    binding; a plain ``import X`` is a setup helper and must not be published.
    """
    import ast

    source = (_REPO_ROOT / "flashinfer" / "rocm" / "api.py").read_text()
    tree = ast.parse(source)

    re_exported = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            re_exported += [a.name for a in node.names if a.asname == a.name]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            re_exported.append(node.target.id)
    declared = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )

    assert declared == re_exported, (
        "flashinfer/rocm/api.py __all__ is out of step with its exports.\n"
        f"  declared: {declared}\n"
        f"  exported: {re_exported}\n"
        f"  missing={sorted(set(re_exported) - set(declared))} "
        f"stale={sorted(set(declared) - set(re_exported))}"
    )


_LEGACY_CONST = '_BATCH_DECODE_PLAN_LEGACY_POS_ARGS = ("alpha", "beta")\n'

# A pair the audit reads as clean. Synthetic rather than a copy of the real
# modules so a signature change in the port cannot silently retire these cases.
_CLEAN_UPSTREAM = """\
def public_fn(first, second=1, *rest, flag=False, **extra):
    pass


class Thing:
    def __init__(self, value):
        pass
"""


def _make_synthetic_tree(tmp_path, **overrides):
    """Write the six files ``audit`` reads; each override replaces one body."""
    bodies = {
        "flashinfer/decode.py": _LEGACY_CONST + _CLEAN_UPSTREAM,
        "flashinfer/rocm/decode.py": _LEGACY_CONST + _CLEAN_UPSTREAM,
        "flashinfer/prefill.py": _CLEAN_UPSTREAM,
        "flashinfer/rocm/prefill.py": _CLEAN_UPSTREAM,
        "flashinfer/mla/_core.py": _CLEAN_UPSTREAM,
        "flashinfer/rocm/mla.py": _CLEAN_UPSTREAM,
    }
    bodies.update(overrides)
    root = tmp_path / "synthetic"
    for rel, text in bodies.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    return root


def _kinds(findings):
    return sorted(f.kind for f in findings)


class TestSyntheticDivergence:
    """One case per ``Finding`` kind the audit can emit.

    The real tree is clean by construction, so without these the branches that
    build each finding never run and a broken one would ship unnoticed.
    """

    def test_the_synthetic_pair_is_clean(self, tmp_path):
        # Guards every case below: a finding here would make them pass for the
        # wrong reason.
        findings, _ = parity.audit(_make_synthetic_tree(tmp_path))
        assert findings == [], _kinds(findings)

    def test_a_missing_class_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": "def public_fn(first, second=1, *rest, flag=False, **extra):\n    pass\n"
            },
        )
        findings, _ = parity.audit(root)
        assert "missing-symbol" in _kinds(findings), findings
        assert any("Thing" in f.where for f in findings), findings

    def test_a_missing_function_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": "class Thing:\n    def __init__(self, value):\n        pass\n"
            },
        )
        findings, _ = parity.audit(root)
        assert any(
            f.kind == "missing-symbol" and "public_fn" in f.where for f in findings
        ), findings

    def test_a_drifted_default_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": _CLEAN_UPSTREAM.replace(
                    "second=1", "second=2"
                )
            },
        )
        findings, _ = parity.audit(root)
        assert any(
            f.kind == "default-drift" and "second" in f.detail for f in findings
        ), findings

    def test_a_keyword_only_parameter_gone_positional_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": (
                    "def public_fn(first, second=1, flag=False, *rest, **extra):\n"
                    "    pass\n\n\n"
                    "class Thing:\n    def __init__(self, value):\n        pass\n"
                )
            },
        )
        findings, _ = parity.audit(root)
        assert any(f.kind == "kind-drift" and "flag" in f.detail for f in findings), (
            findings
        )

    def test_a_rocm_only_positional_inside_upstreams_range_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": _CLEAN_UPSTREAM.replace(
                    "def public_fn(first, second=1,",
                    "def public_fn(first, extra_arg=None, second=1,",
                )
            },
        )
        findings, _ = parity.audit(root)
        assert any(
            f.kind == "inserted-extra" and "extra_arg" in f.detail for f in findings
        ), findings

    def test_a_dropped_vararg_and_kwarg_are_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/prefill.py": (
                    "def public_fn(first, second=1, *, flag=False):\n"
                    "    pass\n\n\n"
                    "class Thing:\n    def __init__(self, value):\n        pass\n"
                )
            },
        )
        findings, _ = parity.audit(root)
        details = " ".join(f.detail for f in findings if f.kind == "missing-param")
        assert "*rest" in details and "**extra" in details, findings

    def test_a_missing_legacy_tuple_copy_is_reported(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path, **{"flashinfer/rocm/decode.py": _CLEAN_UPSTREAM}
        )
        findings, _ = parity.audit(root)
        # The kind matters: without it, "stale-copy" from the drift arm below
        # satisfies the assertion and the missing-copy branch goes untested.
        assert any(
            f.kind == "missing-symbol"
            and "_BATCH_DECODE_PLAN_LEGACY_POS_ARGS" in f.where
            for f in findings
        ), findings

    def test_a_non_literal_legacy_tuple_reads_as_absent(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path,
            **{
                "flashinfer/rocm/decode.py": "_BATCH_DECODE_PLAN_LEGACY_POS_ARGS = (name,)\n"
                + _CLEAN_UPSTREAM
            },
        )
        findings, _ = parity.audit(root)
        assert any(
            f.kind == "missing-symbol"
            and "_BATCH_DECODE_PLAN_LEGACY_POS_ARGS" in f.where
            for f in findings
        ), findings

    def test_an_absent_upstream_legacy_tuple_is_a_tool_error(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path, **{"flashinfer/decode.py": _CLEAN_UPSTREAM}
        )
        with pytest.raises(parity.ToolError, match="not found in flashinfer/decode.py"):
            parity.audit(root)

    def test_an_unparsable_module_is_a_tool_error(self, tmp_path):
        root = _make_synthetic_tree(
            tmp_path, **{"flashinfer/rocm/mla.py": "def broken(\n"}
        )
        with pytest.raises(parity.ToolError, match="cannot parse"):
            parity.audit(root)


class TestCommandLine:
    def test_clean_tree_exits_zero(self, capsys):
        assert parity.main(["--root", str(_REPO_ROOT)]) == parity.EXIT_OK
        assert "clean" in capsys.readouterr().out

    def test_divergence_exits_one_and_prints_each_finding(self, tmp_path, capsys):
        root = _make_shadow_tree(tmp_path)
        target = root / "flashinfer/rocm/decode.py"
        target.write_text(target.read_text().replace('    "o_data_type",\n', "", 1))

        assert parity.main(["--root", str(root)]) == parity.EXIT_DIVERGED
        out = capsys.readouterr().out
        assert "stale-copy" in out and "divergence(s)" in out

    def test_json_output_is_machine_readable(self, tmp_path, capsys):
        root = _make_shadow_tree(tmp_path)
        target = root / "flashinfer/rocm/decode.py"
        target.write_text(target.read_text().replace('    "o_data_type",\n', "", 1))

        assert parity.main(["--json", "--root", str(root)]) == parity.EXIT_DIVERGED
        payload = json.loads(capsys.readouterr().out)
        assert {f["kind"] for f in payload} >= {"stale-copy"}

    def test_a_tool_error_exits_two(self, tmp_path, capsys):
        root = _make_shadow_tree(tmp_path)
        (root / "flashinfer/rocm/mla.py").write_text("def broken(\n")

        assert parity.main(["--root", str(root)]) == parity.EXIT_ERROR
        assert "cannot parse" in capsys.readouterr().err

    def test_a_stale_allowlist_entry_is_reported(self, tmp_path, capsys):
        """An allowlisted symbol upstream has dropped must surface, not go quiet."""
        root = _make_synthetic_tree(tmp_path)
        stale = parity.stale_allowlist_entries(root)
        assert stale, "the synthetic tree defines none of the allowlisted symbols"

        assert parity.main(["--root", str(root)]) == parity.EXIT_DIVERGED
        assert "stale-allowlist" in capsys.readouterr().out


class TestAllowlistMechanics:
    """``CUDA_ONLY_METHODS`` is empty today, so its arms need an entry to run."""

    def test_a_method_entry_supplies_the_reason(self, monkeypatch):
        monkeypatch.setitem(
            parity.CUDA_ONLY_METHODS, "flashinfer/decode.py::Thing.only_cuda", "why"
        )
        assert parity._allowlisted("flashinfer/decode.py", "Thing.only_cuda") == "why"

    def test_a_method_entry_upstream_dropped_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            parity.CUDA_ONLY_METHODS, "flashinfer/decode.py::Thing.gone", "why"
        )
        root = _make_synthetic_tree(tmp_path)
        assert "flashinfer/decode.py::Thing.gone" in parity.stale_allowlist_entries(
            root
        )

    def test_a_pattern_match_is_recorded_rather_than_reported(self, tmp_path):
        """An allowlisted name present on both sides lands in used_allowlist."""
        body = _CLEAN_UPSTREAM + "\n\ndef trtllm_helper(only_upstream):\n    pass\n"
        root = _make_synthetic_tree(
            tmp_path,
            **{"flashinfer/prefill.py": body, "flashinfer/rocm/prefill.py": body},
        )
        findings, used = parity.audit(root)
        assert findings == [], _kinds(findings)
        assert "flashinfer/prefill.py::trtllm_helper" in used

    def test_an_overload_stub_is_skipped_by_attribute_decorator(self, tmp_path):
        """``@typing.overload`` is the Attribute form of the decorator check."""
        body = (
            "import typing\n\n\n"
            "@typing.overload\n"
            "def public_fn(first):\n    ...\n\n\n"
            "def public_fn(first, second=1, *rest, flag=False, **extra):\n"
            "    pass\n\n\n"
            "class Thing:\n    def __init__(self, value):\n        pass\n"
        )
        root = _make_synthetic_tree(tmp_path, **{"flashinfer/prefill.py": body})
        findings, _ = parity.audit(root)
        assert findings == [], _kinds(findings)


class TestLegacyPositionalTuple:
    def test_an_unparsable_file_is_a_tool_error(self, tmp_path):
        broken = tmp_path / "broken.py"
        broken.write_text("def f(\n")
        with pytest.raises(parity.ToolError, match="cannot parse"):
            parity._legacy_positional_tuple(broken, "_ANY")

    def test_other_module_level_assignments_are_skipped(self, tmp_path):
        source = tmp_path / "mod.py"
        source.write_text('OTHER = ("x",)\n' + _LEGACY_CONST)
        assert parity._legacy_positional_tuple(
            source, "_BATCH_DECODE_PLAN_LEGACY_POS_ARGS"
        ) == ("alpha", "beta")
