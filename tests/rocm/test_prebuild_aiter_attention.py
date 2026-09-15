# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parity for docker/prebuild_aiter_attention.py.

That driver runs inside `docker build` with no GPU and no torch, so it cannot
import flashinfer's variant table and carries its own copy. Two things can go
wrong silently, and one test guards each:

1. Its filenames drift from what `aiter_loader.cc` asks for. The artifact is
   built and then never found -- a silent latency loss, not an error.
2. Its CK receipts or filter tokens drift from AITER's. The right filename gets
   a *different kernel* inside, which nothing downstream detects.

Both read the installed AITER source rather than a checkout: the two disagree,
and only the installed one is what runs. No build, no GPU.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from flashinfer.jit.rocm import aiter_variants as fi_variants

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "docker" / "prebuild_aiter_attention.py"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "_prebuild_aiter_attention", _DRIVER_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if not _DRIVER_PATH.is_file():
    # An installed wheel has no docker/ tree; skip rather than fail collection.
    pytest.skip(f"{_DRIVER_PATH} not present", allow_module_level=True)

driver = _load_driver()


def _aiter_ops_mha_source() -> str:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("amd-aiter is not installed")
    src = Path(list(spec.submodule_search_locations)[0]) / "ops" / "mha.py"
    if not src.is_file():
        pytest.skip(f"{src} not found")
    return src.read_text()


def _composer_body(family: str) -> str:
    """The body of AITER's `cmdGenFunc_<family>`, where md_name and filter are built."""
    text = _aiter_ops_mha_source()
    start = text.find(f"def cmdGenFunc_{family}(")
    assert start != -1, f"aiter/ops/mha.py has no cmdGenFunc_{family}"
    nxt = text.find("\ndef ", start + 1)
    return text[start : nxt if nxt != -1 else len(text)]


# ---------------------------------------------------------------------------
# 1. filenames match what the loader asks for
# ---------------------------------------------------------------------------

_FAMILY_BY_NAME = {
    "mha_fwd": fi_variants.Family.MHA_FWD,
    "mha_varlen_fwd": fi_variants.Family.MHA_VARLEN_FWD,
    "mha_batch_prefill": fi_variants.Family.MHA_BATCH_PREFILL,
}


def _fi_so_names() -> set:
    return {fi_variants.so_name(k) for k in fi_variants.reachable_variants()}


def test_the_driver_covers_flashinfers_table_and_the_fp8_arm_it_omits():
    """flashinfer's table is the store's domain, which has no fp8 arm; the
    driver writes where the loader looks, so it must carry fp8 as well."""
    assert len(driver.reachable_variants()) == 44
    assert _fi_so_names() < {v.so_name for v in driver.reachable_variants()}


def test_fp8_carries_no_lse_arm():
    """AITER ships no LSE instance of the fp8 kernel, and prefill.py raises on
    the combination -- building it yields a dispatcher with no kernel behind it."""
    fp8 = [v for v in driver.reachable_variants() if v.dtype == "fp8bf16"]
    assert len(fp8) == 4
    assert not [v for v in fp8 if v.has_lse]
    # The guard this mirrors, so a change there shows up here.
    prefill = (
        Path(__file__).resolve().parents[2] / "flashinfer" / "rocm" / "prefill.py"
    ).read_text()
    assert "fp8 prefill cannot produce LSE" in prefill


def test_nothing_is_trimmed_from_the_built_set():
    """A trim buys a few minutes of image build for a multi-minute stall on the
    arm that turns out to be missing."""
    assert set(driver.selected_variants()) == set(driver.reachable_variants())
    assert len(driver.selected_variants()) == 44


@pytest.mark.parametrize("family", sorted(_FAMILY_BY_NAME))
def test_only_restricts_without_renaming(family):
    picked = driver.selected_variants(only=[family])
    assert picked
    assert {v.so_name for v in picked} <= {
        v.so_name for v in driver.reachable_variants()
    }
    assert all(v.family == family for v in picked)


def test_check_does_not_demand_what_only_never_builds(tmp_path, monkeypatch):
    """`--only` skips the whole-module set, so `_check` must skip it too or a
    healthy restricted build exits 1."""
    monkeypatch.setattr(driver, "_aiter_jit_core", lambda: object())
    monkeypatch.setattr(driver, "jit_dir", lambda _core: tmp_path)
    picked = driver.selected_variants(only=["mha_fwd"])
    for v in picked:
        # Enough markers to clear the instance check; that is a separate test.
        (tmp_path / v.so_name).write_bytes(b"ck_tile" * driver._MIN_CK_MARKERS)

    assert driver.main(["--check", "--only", "mha_fwd"]) == 0
    # Unrestricted, the same directory is incomplete: the module is absent.
    assert driver.main(["--check"]) == 1


def test_check_rejects_an_artifact_with_no_ck_instances(tmp_path, monkeypatch):
    """A --filter that selected nothing still compiles and links. Measured on a
    real one: 20 `ck_tile` markers with no instances, 1701 with them."""
    monkeypatch.setattr(driver, "_aiter_jit_core", lambda: object())
    monkeypatch.setattr(driver, "jit_dir", lambda _core: tmp_path)
    picked = driver.selected_variants(only=["mha_fwd"])
    for v in picked:
        (tmp_path / v.so_name).write_bytes(b"ck_tile" * driver._MIN_CK_MARKERS)

    assert driver.main(["--check", "--only", "mha_fwd"]) == 0

    hollow = tmp_path / picked[0].so_name
    hollow.write_bytes(b"ck_tile" * (driver._MIN_CK_MARKERS - 1))
    assert driver.main(["--check", "--only", "mha_fwd"]) == 1


# ---------------------------------------------------------------------------
# 1b. every token the loader can compose is one the driver emits
# ---------------------------------------------------------------------------

# alibi is the one arm the loader can spell but no call site reaches: all three
# construction sites hard-code has_alibi false.
_UNREACHABLE_LOADER_TOKENS = {"_alibi"}


def _loader_name_tokens() -> set:
    """Every string literal the loader can put in a filename.

    Two sources, and missing the second is how `_nsink` could have slipped in:
    the axis tokens inside `build_so_name`/`dtype_token`, and the
    prefix/infix/suffix literals its three call sites pass in.
    """
    text = _LOADER_CC.read_text()
    start = text.find("const char* dtype_token(")
    end = text.find("\n}", text.find("std::string build_so_name("))
    assert start != -1 and end != -1, "aiter_loader.cc: name composition not found"
    tokens = set(re.findall(r'"([a-z0-9_]+)"', text[start:end]))

    call_sites = re.findall(r"build_so_name\(key,([^;]*?)\)", text, re.S)
    assert call_sites, "aiter_loader.cc: no build_so_name call sites found"
    for args in call_sites:
        for lit in re.findall(r'"([a-z0-9_.]*)"', args):
            tokens.update(s for s in lit.replace(".so", "").split("_") if s)
    return tokens


def test_every_name_token_the_loader_can_emit_is_one_the_driver_builds():
    """Table-vs-table parity cannot see a token both tables lack -- which is how
    the fp8/pertensor batch-prefill arm went unbuilt."""
    segments = set()
    for v in driver.reachable_variants():
        segments.update(v.md_name.split("_"))

    tokens = _loader_name_tokens()
    assert "fp8bf16" in tokens, "loader no longer spells fp8; is this test stale?"
    missing = {
        t
        for t in tokens
        if t not in _UNREACHABLE_LOADER_TOKENS and t.lstrip("_") not in segments
    }
    assert not missing, f"aiter_loader.cc can emit {sorted(missing)}, never built"


class _FakeCore:
    """build_recipe reads only CK_DIR off core for the two composed families."""

    CK_DIR = "/nonexistent"


def test_the_dockerfile_floor_matches_aiter_min_version():
    """Two literals, one meaning. If they drift the image builds green against an
    AITER the library then refuses, and every path degrades to fa2 at run time."""
    from flashinfer.rocm import aiter_utils

    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.rocm"
    ).read_text()
    found = re.findall(r'^FLOOR = "([0-9][^"]*)"', dockerfile, re.M)
    assert found, "docker/Dockerfile.rocm no longer spells FLOOR; is this test stale?"
    assert found == [aiter_utils.AITER_MIN_VERSION], (
        f"Dockerfile FLOOR {found} != AITER_MIN_VERSION "
        f"{aiter_utils.AITER_MIN_VERSION!r}"
    )


# ---------------------------------------------------------------------------
# 2. receipts and token spellings match the installed aiter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "direction", "receipt"),
    [
        ("mha_fwd", "fwd", 100),
        ("mha_varlen_fwd", "fwd", 200),
        ("mha_batch_prefill", "batch_prefill", 200),
    ],
)
def test_the_receipt_and_direction_match_aiter(family, direction, receipt):
    """A wrong receipt ships a different instance set under the right filename."""
    body = _composer_body(family)
    assert f"-d {direction} " in body
    assert f"--receipt {receipt} " in body

    from_driver = {
        "mha_fwd": driver.RECEIPT_MHA_FWD,
        "mha_varlen_fwd": driver.RECEIPT_VARLEN,
        "mha_batch_prefill": driver.RECEIPT_BATCH_PREFILL,
    }[family]
    assert from_driver == receipt

    # The receipt was pinned but the -d literal beside it was not: a typo there
    # emits an empty instance set that still links and still passes --check.
    # varlen is excluded because it takes its command from AITER's own helper.
    if family != "mha_varlen_fwd":
        v = next(x for x in driver.reachable_variants() if x.family == family)
        cmd = driver.build_recipe(v, _FakeCore())["blob_gen_cmd"][0]
        assert f"-d {direction} " in cmd, cmd


def test_mha_fwd_masks_on_underscore_m_not_underscore_mask():
    """The one filter token that differs between families. Spelling `_mask*` here
    selects nothing, so the module builds and dispatch then finds no kernel."""
    body = _composer_body("mha_fwd")
    assert 'filter += "_m*"' in body

    v = driver.Variant("mha_fwd", "bf16", False, needs_mask=True, has_lse=False)
    assert "_m*" in driver._filter_mha_fwd(v)
    assert "_mask*" not in driver._filter_mha_fwd(v)


def test_batch_prefill_dtype_filter_has_no_leading_underscore():
    """Unlike mha_fwd's `_bf16*`, batch_prefill appends a bare `bf16*`."""
    body = _composer_body("mha_batch_prefill")
    assert 'filter_fwd += "bf16*"' in body

    v = driver.Variant("mha_batch_prefill", "bf16", False, True, False)
    assert "*bf16*" in driver._filter_batch_prefill(v)
    assert "_bf16*" not in driver._filter_batch_prefill(v)


def _mha_recipes_source() -> str:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("amd-aiter is not installed")
    src = (
        Path(list(spec.submodule_search_locations)[0])
        / "jit"
        / "utils"
        / "mha_recipes.py"
    )
    if not src.is_file():
        pytest.skip(f"{src} not found")
    return src.read_text()


def _md_token_order(family: str) -> dict:
    """Name token -> position of its first appearance in AITER's own source.

    findall preserves source order, so the index doubles as the token's slot in
    the composed name. varlen delegates its suffix to
    `compose_mha_fwd_variant_suffix_and_filter`, so its literals live in
    mha_recipes.py rather than in the composer body.
    """
    source = (
        _mha_recipes_source() if family == "mha_varlen_fwd" else _composer_body(family)
    )
    order = {}
    for i, tok in enumerate(re.findall(r'"(_[a-z0-9]+)"', source)):
        order.setdefault(tok, i)
    return order


@pytest.mark.parametrize("family", sorted(_FAMILY_BY_NAME))
def test_every_md_name_decomposes_into_aiter_token_literals(family):
    """Each composed name must be the family base plus AITER's own literals, in
    AITER's own order. Membership alone is not enough: a reordered token leaves
    every filename for that arm undiscoverable while still decomposing."""
    order = _md_token_order(family)
    assert order, f"no md_name += literals found in cmdGenFunc_{family}"

    for v in driver.reachable_variants():
        if v.family != family:
            continue
        assert v.md_name.startswith(f"{family}_{v.dtype}")
        rest = v.md_name[len(f"{family}_{v.dtype}") :]
        seen = -1
        while rest:
            match = max((t for t in order if rest.startswith(t)), key=len, default=None)
            assert match, f"{v.md_name!r}: no aiter token matches at {rest!r}"
            assert order[match] >= seen, (
                f"{v.md_name!r}: {match!r} is out of AITER's order -- it appears "
                f"at index {order[match]} after a token at {seen}"
            )
            seen = order[match]
            rest = rest[len(match) :]


def test_varlen_goes_through_aiters_own_helper():
    """varlen is the control: its builds must stay byte-identical to the lazy
    ones, which only holds while we call AITER's composer rather than mirror it."""
    src = _DRIVER_PATH.read_text()
    assert "get_mha_varlen_prebuild_variants_by_names" in src
    assert "_md_varlen" in src  # the mirror exists only to name the file we ask for


# ---------------------------------------------------------------------------
# 3. whole modules the loader dlopens by name
# ---------------------------------------------------------------------------

_LOADER_CC = Path(__file__).resolve().parents[2] / "csrc" / "rocm" / "aiter_loader.cc"


def test_the_driver_builds_every_module_the_loader_dlopens():
    """A PREBUILD_KERNELS=0 source install ships no module_*.so at all, so any
    name the loader opens has to be in the driver's set or the load throws --
    which is what happened to module_fmha_v3_fwd when the asm arm landed."""
    opened = set(re.findall(r'"(module_[a-z0-9_]+)\.so"', _LOADER_CC.read_text()))

    assert opened, "no module_*.so names found in aiter_loader.cc; did the form change?"
    assert opened <= set(driver.LOADER_MODULES), (
        f"aiter_loader.cc dlopens {sorted(opened - set(driver.LOADER_MODULES))}, which "
        "docker/prebuild_aiter_attention.py does not build"
    )
