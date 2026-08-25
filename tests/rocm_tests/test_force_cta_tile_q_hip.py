# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FLASHINFER_ROCM_FORCE_CTA_TILE_Q, the CTA_TILE_Q measurement override.

The override is read once into a function-local static, so each value needs its
own process — hence subprocess rather than monkeypatch.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip, reason="FLASHINFER_ROCM_FORCE_CTA_TILE_Q is HIP-only"
)

# head_dim 256 is excluded deliberately: forcing bypasses the heuristic's guards,
# and 128 there is rejected by IsInvalid() while 16 needs 72 KB of smem.
_RUN = textwrap.dedent(
    """
    import torch, flashinfer
    qo_len, kv_len, heads, head_dim = 128, 512, 4, {head_dim}
    torch.manual_seed(0)
    q = torch.randn(qo_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")
    qf, kf, vf = q.float(), k.float(), v.float()
    s = torch.einsum("qhd,khd->hqk", qf, kf) / (head_dim ** 0.5)
    mask = torch.arange(qo_len, device=q.device)[:, None] + (kv_len - qo_len) < \\
        torch.arange(kv_len, device=q.device)[None, :]
    p = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
    ref = torch.einsum("hqk,khd->qhd", p, vf)
    err = (out.float() - ref).abs().max().item()
    assert err < 0.06, f"max_abs_err={{err}}"
    print("OK", err)
    """
)


def _run(source, **env):
    # A None value unsets the variable rather than overriding it. Without this the
    # unset-case test inherits an exported override, exercises the forced path and
    # still passes — verifying the opposite of what it claims.
    child = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-c", source],
        env={k: v for k, v in child.items() if v is not None},
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.parametrize("tile", ["16", "64", "128"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_forced_tile_matches_reference(tile, head_dim):
    """Every forced tile must agree with a torch reference, including the 128 arm
    that the HIP heuristic never selects on its own."""
    proc = _run(_RUN.format(head_dim=head_dim), FLASHINFER_ROCM_FORCE_CTA_TILE_Q=tile)
    assert proc.returncode == 0, (
        f"tile={tile} hd={head_dim}\n{proc.stdout}\n{proc.stderr}"
    )
    assert "OK" in proc.stdout


@pytest.mark.parametrize(
    "bad",
    # " 64"/"+64"/"\t64" are accepted by a bare strtoul, which would still reject
    # "64 " — the asymmetry is the bug, not the leniency.
    [
        "0",
        "32",
        "64abc",
        "64,128",
        "-1",
        "0x80",
        " ",
        "abc",
        " 64",
        "+64",
        "64 ",
        "\t64",
    ],
)
def test_invalid_override_is_rejected(bad):
    """A typo must fail loudly. Silently benchmarking a different tile than the one
    requested is the failure this harness exists to prevent."""
    proc = _run(_RUN.format(head_dim=128), FLASHINFER_ROCM_FORCE_CTA_TILE_Q=bad)
    assert proc.returncode != 0, f"{bad!r} was accepted:\n{proc.stdout}"
    assert "FLASHINFER_ROCM_FORCE_CTA_TILE_Q" in proc.stderr


@pytest.mark.parametrize("value", [None, ""])
def test_unset_override_uses_the_heuristic(value):
    """Absent the variable the heuristic is untouched; an empty value is also unset.

    None explicitly unsets, so an exported override in the caller's shell cannot
    turn this into a second test of the forced path.
    """
    proc = _run(_RUN.format(head_dim=128), FLASHINFER_ROCM_FORCE_CTA_TILE_Q=value)
    assert proc.returncode == 0, f"{value!r}\n{proc.stdout}\n{proc.stderr}"
