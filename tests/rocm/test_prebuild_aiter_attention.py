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


def test_the_driver_enumerates_the_same_40_variants_as_flashinfer():
    assert (
        len(driver.reachable_variants()) == len(fi_variants.reachable_variants()) == 40
    )


def test_every_driver_filename_is_one_the_loader_asks_for():
    """A name the loader never asks for is built and then never opened."""
    assert {v.so_name for v in driver.reachable_variants()} == _fi_so_names()


def test_the_default_set_is_the_full_set_minus_mha_fwd_lse():
    selected = driver.selected_variants()
    skipped = set(driver.reachable_variants()) - set(selected)

    assert len(selected) == 36
    assert all(v.family == "mha_fwd" and v.has_lse for v in skipped)
    assert len(skipped) == 4


def test_the_blaze_lum3_variant_is_in_the_default_set():
    """Blaze-O1's LUM3 pins fp16 / batch mode / causal / no soft-cap / no LSE;
    batch mode is mha_fwd, so this exact file is the one it dlopens."""
    assert "mha_fwd_fp16_nbias_mask_nlse_ndropout_nqscale.so" in {
        v.so_name for v in driver.selected_variants()
    }


@pytest.mark.parametrize("family", sorted(_FAMILY_BY_NAME))
def test_only_restricts_without_renaming(family):
    picked = driver.selected_variants(only=[family], everything=True)
    assert picked
    assert {v.so_name for v in picked} <= _fi_so_names()
    assert all(v.family == family for v in picked)


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


def _md_tokens(family: str) -> set:
    """Every name token AITER can emit for this family, dtype excluded.

    mha_fwd and batch_prefill append string literals to md_name directly. varlen
    delegates its whole suffix to `compose_mha_fwd_variant_suffix_and_filter`, so
    its literals live in mha_recipes.py instead. Both spell the dtype with an
    f-string, which leaves no literal to find -- hence dtype is stripped by the
    caller rather than matched here.
    """
    source = (
        _mha_recipes_source() if family == "mha_varlen_fwd" else _composer_body(family)
    )
    return set(re.findall(r'"(_[a-z0-9]+)"', source))


@pytest.mark.parametrize("family", sorted(_FAMILY_BY_NAME))
def test_every_md_name_decomposes_into_aiter_token_literals(family):
    """Catches a renamed or reordered token: each name we compose must be the
    family base plus a concatenation of literals AITER itself appends."""
    tokens = _md_tokens(family)
    assert tokens, f"no md_name += literals found in cmdGenFunc_{family}"

    for v in driver.reachable_variants():
        if v.family != family:
            continue
        assert v.md_name.startswith(f"{family}_{v.dtype}")
        rest = v.md_name[len(f"{family}_{v.dtype}") :]
        while rest:
            match = max(
                (t for t in tokens if rest.startswith(t)), key=len, default=None
            )
            assert match, f"{v.md_name!r}: no aiter token matches at {rest!r}"
            rest = rest[len(match) :]


def test_varlen_goes_through_aiters_own_helper():
    """varlen is the control: its builds must stay byte-identical to the lazy
    ones, which only holds while we call AITER's composer rather than mirror it."""
    src = _DRIVER_PATH.read_text()
    assert "get_mha_varlen_prebuild_variants_by_names" in src
    assert "_md_varlen" in src  # the mirror exists only to name the file we ask for
