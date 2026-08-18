# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-architecture capability knowledge for the ROCm backends.

This module owns *which architectures are validated for which operations*.
That is deliberately distinct from :mod:`flashinfer.hip_utils`, which owns
*which architectures we compile for*, and from :mod:`flashinfer.aiter_utils`,
which owns *whether the AITER package is importable*.

.. important::
   Do not add a module-level ``import torch``; keep every torch import
   function-local.

   ``hip_utils`` imports this module at module scope, and ``hip_utils`` is used
   very early -- ``tests/conftest.py`` calls it to pick a GPU *before* pinning
   ``HIP_VISIBLE_DEVICES``, and it must do so without touching the HIP runtime
   (hence its use of ``rocminfo`` rather than ``torch.cuda``). Staying torch-free
   is what keeps this module usable on that path.
"""

__all__ = ["normalize_arch"]


def normalize_arch(gcn_arch_name: str) -> str:
    """Strip ROCm feature qualifiers from a GPU architecture name.

    ROCm reports architectures with trailing feature flags, e.g. torch's
    ``gcnArchName`` yields ``"gfx942:sramecc+:xnack-"``. Comparisons against
    :data:`~flashinfer.hip_utils.FLASHINFER_SUPPORTED_ROCM_ARCHS` need the bare
    ``"gfx942"``.

    This is the single place that transformation happens. It replaces several
    open-coded variants that did not agree; notably a ``re.match(r"(gfx\\d+)")``
    form that truncated letter-suffixed architectures (``gfx90a`` -> ``gfx90``),
    naming an architecture that does not exist.

    Args:
        gcn_arch_name: An architecture name, with or without qualifiers.

    Returns:
        The architecture without qualifiers, e.g. ``"gfx942"``. Input that
        contains no qualifier is returned unchanged (modulo surrounding
        whitespace), so this is safe to apply to already-normalized values.
    """
    return gcn_arch_name.split(":", 1)[0].strip()
