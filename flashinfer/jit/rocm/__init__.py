# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ROCm-only halves of the JIT machinery.

Each module here holds the body of an ``elif IS_HIP:`` arm that would otherwise
sit in an upstream file. The arms' module-level constants stay upstream, where
their ``type: ignore[no-redef]`` markers have to be.
"""
