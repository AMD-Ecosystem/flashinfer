# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root conftest: xdist worker count and per-worker GPU pinning.

Both halves live here rather than under ``tests/`` because xdist resolves
``-n auto`` from the *initial* conftests, so a subdirectory conftest is not
loaded in time for any invocation that does not name that subdirectory.
"""

import os


def _worker_gpu_index(worker_idx: int, supported):
    """Card to pin an xdist worker to, or None if there is nothing to pin to.

    Workers wrap around when they outnumber the cards. Using worker_idx itself
    names a device that need not exist, and a child process inheriting that
    HIP_VISIBLE_DEVICES sees no GPU at all.
    """
    return supported[worker_idx % len(supported)] if supported else None


# pytest_xdist_auto_num_workers is only available when pytest-xdist is installed
# and loaded (i.e. when -n / --dist is passed). Guard the definition so that
# running without xdist does not produce an "unknown hook" PluginValidationError.
try:
    import xdist  # noqa: F401

    def pytest_xdist_auto_num_workers(config):
        """Return the recommended worker count for 'pytest -n auto'.

        Halved from one-per-physical-card. One worker per physical card
        still produces sporadic HSA / HIPBLAS failures across the wider
        ROCm test suite (rope, single_prefill, logits_cap) under residual
        concurrent CPX load even with --reruns 2; halving eliminates them
        reliably at a ~1.6x wall-time cost. Users who want every device
        used can pass an explicit -n N -- the hook is firstresult, so it
        also shadows xdist's PYTEST_XDIST_AUTO_NUM_WORKERS.
        """
        from flashinfer.rocm.hip_utils import get_physical_card_device_indices

        n_physical = len(get_physical_card_device_indices())
        if n_physical == 0:
            raise RuntimeError(
                "pytest -n auto: no FlashInfer-supported AMD GPUs detected. "
                "Check HIP_VISIBLE_DEVICES or ROCm installation."
            )
        return max(1, n_physical // 2)

except ImportError:
    pass


# Pin this worker to a card. PYTEST_XDIST_WORKER ("gw0", "gw1", ...) is injected
# into each worker subprocess by pytest-xdist before any Python code runs.
# get_physical_card_device_indices() spreads workers one per *supported* physical
# card; the torch fallback below guarantees neither.
#
# This does not re-scope *this* process -- importing flashinfer initializes HIP,
# so torch has already latched the full device list. What it scopes is the child
# processes tests spawn, which is where a bad index is fatal.
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER", "")
if _xdist_worker.startswith("gw"):
    import torch

    from flashinfer.rocm.hip_utils import get_physical_card_device_indices

    # rocminfo is how supported cards are identified; when it reports none --
    # missing binary, or no supported GPU -- fall back to what torch can see
    # rather than to worker indices that need not name anything.
    _cards = get_physical_card_device_indices() or tuple(
        range(torch.cuda.device_count())
    )
    _gpu_index = _worker_gpu_index(int(_xdist_worker[2:]), _cards)
    if _gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(_gpu_index)
