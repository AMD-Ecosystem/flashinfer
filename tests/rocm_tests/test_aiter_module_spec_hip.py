# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`AiterModule` and the fused-MoE JIT spec's use of it.

No GPU and no build: these cover the naming and flag construction that decide
*which* lib a shim links. The ``_ck2stages_module`` tests still need ``aiter``
importable, because the codegen command embeds AITER's own csrc path.

What is being protected. Each specialization of ``module_moe_ck2stages`` (bf16
vs fp16, silu vs gelu, unquantized vs fp8) is a separate build of the same AITER
source tree. They are distinguished only by ``md_name``, which names the cached
``lib*.so``. If they collided on one filename the second build would be skipped
and the first lib silently reused -- a wrong-dtype, wrong-activation or
unquantized kernel, with correct shapes and no error.
"""

import itertools

import pytest
import torch

from flashinfer.aiter_utils import _aiter_importable
from flashinfer.jit.aiter_source import AiterModule, aiter_jitspec_flags
from flashinfer.jit.fused_moe_rocm import _ck2stages_module

_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)
_ACTIVATIONS = ("silu", "gelu")
# The weight dtype is either the activation dtype (unquantized) or fp8. Both fp8
# encodings generate the same "f8" instances, so one stands for the pair here;
# which one is legal is the device's business, not the codegen's.
_WEIGHT_DTYPES = (None, torch.float8_e4m3fn)

# _ck2stages_module reads aiter.jit.core.AITER_CSRC_DIR. A ROCm box without the
# aiter package is a supported state, so skip rather than error there.
# _aiter_importable actually imports aiter, aiter_meta and aiter.jit.core, so a
# half-installed aiter -- which find_spec reports as present -- skips too.
requires_aiter_import = pytest.mark.skipif(
    not _aiter_importable(), reason="needs a working aiter install"
)


def test_lib_name_defaults_to_the_config_key():
    assert AiterModule("module_moe_sorting").lib_name == "module_moe_sorting"


def test_md_name_overrides_the_lib_name():
    m = AiterModule("module_moe_ck2stages", md_name="module_moe_ck2stages_b16_silu")
    assert m.lib_name == "module_moe_ck2stages_b16_silu"


@pytest.mark.parametrize(
    "bad", ["", "../escape", "-levil", "has/slash", ".hidden", "trailing\n"]
)
def test_unusable_lib_names_are_rejected(bad):
    """An empty md_name is rejected, not treated as "unset".

    Falling back to the unspecialized name would put two specializations on one
    ``lib*.so`` -- the second build skipped, the first silently reused.
    """
    with pytest.raises(ValueError, match="not usable as a library name"):
        AiterModule("module_moe_ck2stages", md_name=bad)
    with pytest.raises(ValueError, match="not usable as a library name"):
        AiterModule(bad)


@requires_aiter_import
def test_specializations_do_not_collide():
    """Every (dtype, activation, weight dtype) must map to a distinct cached lib."""
    combos = list(itertools.product(_ACTIVATION_DTYPES, _WEIGHT_DTYPES, _ACTIVATIONS))
    names = [_ck2stages_module(dt, wt or dt, act).lib_name for dt, wt, act in combos]
    assert len(set(names)) == len(combos), names
    assert all(n != "module_moe_ck2stages" for n in names), names


@requires_aiter_import
@pytest.mark.parametrize("dtype", _ACTIVATION_DTYPES)
@pytest.mark.parametrize("weight_dtype", _WEIGHT_DTYPES)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_blob_gen_cmd_is_format_safe(dtype, weight_dtype, activation):
    """AITER runs ``blob_gen_cmd.format(blob_dir)``: exactly one ``{}``, no others."""
    cmd = _ck2stages_module(dtype, weight_dtype or dtype, activation).blob_gen_cmd
    assert cmd.count("{") == 1 and cmd.count("}") == 1
    assert cmd.format("/tmp/blobs").endswith("--working_path /tmp/blobs")


@requires_aiter_import
@pytest.mark.parametrize("dtype", _ACTIVATION_DTYPES)
@pytest.mark.parametrize("weight_dtype", _WEIGHT_DTYPES)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_codegen_flags_match_the_shim(dtype, weight_dtype, activation):
    """The generated instances must match what the C++ shim asks CK for.

    ``-m 2`` because the shim passes sorted_weights to stage 2, the quant tag
    because that selects which QuantType the shim may pass, and the dtype tags
    because there is one build per configuration. A mismatch is a lookup miss at
    runtime, not a build failure.
    """
    tag = {torch.bfloat16: "b16", torch.float16: "f16"}[dtype]
    weight_tag = "f8" if weight_dtype is not None else tag
    quant = "per_token" if weight_dtype is not None else "no"
    cmd = _ck2stages_module(dtype, weight_dtype or dtype, activation).blob_gen_cmd
    for flag in (
        f"-b {weight_tag}",
        f"-c {tag}",
        f"-q {quant}",
        f"-act {activation}",
        "-m 2",
    ):
        assert flag in cmd, f"{flag!r} missing from {cmd!r}"
    # -a is dead: gen_instances.py derives the activation dtype from -b.
    assert " -a " not in cmd


@requires_aiter_import
def test_both_fp8_encodings_share_one_build():
    """e4m3fn and e4m3fnuz differ on the device, not in the generated code.

    Two libs would double a multi-minute CK build for nothing -- and only one of
    them is ever loadable on a given machine anyway.
    """
    names = {
        _ck2stages_module(torch.bfloat16, wt, "silu").lib_name
        for wt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    }
    assert len(names) == 1, names


def test_flags_link_every_module_once(monkeypatch):
    import flashinfer.jit.aiter_source as src

    built = []
    monkeypatch.setattr(src, "ensure_aiter_lib", lambda m: built.append(m.lib_name))
    monkeypatch.setattr(src, "_aiter_csrc_include_dir", lambda: "/fake/csrc/include")
    monkeypatch.setattr(src, "_aiter_libs_dir", lambda: "/fake/libs")

    _, ldflags = aiter_jitspec_flags(
        "module_moe_sorting", AiterModule("module_moe_ck2stages", md_name="spec_a")
    )

    assert built == ["module_moe_sorting", "spec_a"]
    assert ldflags.count("-L/fake/libs") == 1
    assert [f for f in ldflags if f.startswith("-l")] == [
        "-lmodule_moe_sorting",
        "-lspec_a",
    ]
    # rpath must come along, or the module loads whatever the loader finds first.
    assert "-Wl,-rpath,/fake/libs" in ldflags


def test_flags_needs_at_least_one_module():
    with pytest.raises(ValueError, match="at least one AITER module"):
        aiter_jitspec_flags()
