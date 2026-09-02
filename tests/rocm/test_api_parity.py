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
import subprocess
import sys
import textwrap
from pathlib import Path

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
