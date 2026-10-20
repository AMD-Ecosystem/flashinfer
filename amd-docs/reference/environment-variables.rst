.. meta::
  :description: FlashInfer on ROCm environment variables and runtime helpers
  :keywords: FlashInfer, ROCm, environment variables, AITER, logging, JIT, runtime

.. _flashinfer-environment-variables:

********************************************************************
Environment variables
********************************************************************

FlashInfer on ROCm reads the variables on this page at import time or when an
operation runs. Set a variable that is read at import before ``import flashinfer``.
In a shell, export it for the process that starts Python:

.. code-block:: bash

   export FLASHINFER_LOGGING_LEVEL=DEBUG

Build-time variables, including ``FLASHINFER_ROCM_ARCH_LIST``,
``PYTORCH_ROCM_ARCH``, ``FLASHINFER_JIT_VERBOSE``, ``FLASHINFER_EXTRA_LDFLAGS``,
``FLASHINFER_EXTRA_CFLAGS``, ``FLASHINFER_EXTRA_CUDAFLAGS``,
``FLASHINFER_OWN_HEADERS_NON_SYSTEM``, ``MAX_JOBS``, and
``FLASHINFER_JIT_DEBUG``, are documented in
`CONTRIBUTING.md <https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md>`__.

Compilation and kernel cache
====================================================================

These variables control ``torch.compile``, JIT compilation, and the on-disk kernel cache.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 28 12 60

   * - Environment variable
     - Default
     - Description
   * - ``FLASHINFER_USE_TORCH_CUSTOM_OPS``
     - ``0``
     - Set to ``1`` before importing ``flashinfer`` to wrap kernels as PyTorch custom operations so ``torch.compile`` can trace them. Requires PyTorch 2.4 or later and adds a small dispatch cost on each call. When the variable is unset or ``0``, ``torch.compile`` raises an error if a trace enters a FlashInfer operation.
   * - ``FLASHINFER_WORKSPACE_BASE``
     - ``$HOME``
     - Parent directory of the JIT cache, ``.cache/flashinfer/``. Point this at fast local disk when the home directory is on a network filesystem. Use an absolute path. A relative path is resolved from the working directory, and a leading ``~`` is treated as a directory name.
   * - ``FLASHINFER_DISABLE_JIT``
     - Unset
     - Any non-empty value disables JIT compilation, including the string ``0``. FlashInfer then loads only ahead-of-time kernels and kernels already in the JIT cache. It will not compile a missing kernel, and it will not load a module that exists only under ``~/.cache/flashinfer``. AITER shims are not shipped prebuilt, so a call that selects AITER raises ``MissingJITCacheError``.
   * - ``FLASHINFER_DISABLE_VERSION_CHECK``
     - Unset
     - Any non-empty value skips the version check between FlashInfer and the JIT-cache package.
   * - ``FLASHINFER_DISABLE_AOT_ARCH_CHECK``
     - Unset
     - Any non-empty value, including ``0``, allows a JIT-cache kernel to run when its architecture does not match the current GPU. By default, a mismatch is discarded with a warning and the kernel is JIT-compiled for the current GPU.
   * - ``FLASHINFER_TRACE_APPLY``
     - Unset
     - Set to ``1`` to apply the kernel substitution mapping named by ``FLASHINFER_TRACE_APPLY_PATH`` at import. ``0`` leaves substitution off. Any other value logs a warning and leaves it off. If substitution cannot be enabled, FlashInfer logs a warning and continues to import.

AITER
====================================================================

These variables control the `AITER <https://github.com/ROCm/aiter>`__ backend and the kernels it selects. For operation coverage, see the :doc:`kernel support matrix <../what-is-flashinfer>`.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 28 16 56

   * - Environment variable
     - Default
     - Description
   * - ``FLASHINFER_AITER_STRICT``
     - ``0``
     - Set to ``1`` to raise an error when AITER cannot serve a page size directly, or when an AITER kernel fails to initialize for ``backend="auto"``. The default is to use another supported path.
   * - ``FLASHINFER_ARCH_ALLOW_KNOWN_BAD``
     - ``0``
     - Set to ``1`` to run an operation and backend that the capability table marks as broken on the current toolchain. This release does not mark any combination that way. Soft-capped causal prefill on gfx950 (MI350X and MI355X) keeps its own routing.
   * - ``FLASHINFER_HIP_FUSED_CASCADE``
     - ``0``
     - Applies to ``MultiLevelCascadeAttentionWrapper`` only. Set to ``1`` before importing ``flashinfer`` to pass each level's partial state into the next prefill call. The value is read once at import. AITER levels and the shared-prefix cascade wrappers do not use this variable.
   * - ``AITER_JIT_DIR``
     - AITER default
     - Directory of AITER shared libraries that the FlashInfer C++ shim loads. Overrides the path recorded when FlashInfer was built.
   * - ``AITER_ASM_DIR``
     - Set by ``import aiter``
     - Directory of AITER assembly code objects, initialized by ``import aiter``. This variable does not disable assembly dispatch or force single prefill to use CK Tile.
   * - ``GPU_ARCHS``
     - Autodetected
     - GPU architecture list for AITER JIT compilation. If you leave it unset, a shim build may set it from ``FLASHINFER_ROCM_ARCH_LIST`` for that build. If you set it yourself, FlashInfer restores your value after the build.

Logging
====================================================================

``FLASHINFER_LOGGING_LEVEL`` controls the FlashInfer logger. ``FLASHINFER_LOGLEVEL`` and ``FLASHINFER_LOGDEST`` control a separate API-call log.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 28 12 60

   * - Environment variable
     - Default
     - Description
   * - ``FLASHINFER_LOGGING_LEVEL``
     - ``INFO``
     - Logger verbosity. Typical values are ``DEBUG``, ``INFO``, and ``WARNING``. This level covers AITER fallback messages and JIT build messages.
   * - ``FLASHINFER_LOGLEVEL``
     - ``0``
     - API-call log detail. ``1`` logs function names. ``3`` adds inputs and outputs. ``5`` adds tensor statistics. ``10`` also writes tensors to disk.
   * - ``FLASHINFER_LOGDEST``
     - ``stdout``
     - Destination of the API-call log: ``stdout``, ``stderr``, or a file path. In a file path, ``%i`` is replaced with the process ID.

ROCm installation
====================================================================

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 28 12 60

   * - Environment variable
     - Default
     - Description
   * - ``ROCM_PATH`` or ``ROCM_HOME``
     - ``/opt/rocm``
     - ROCm installation prefix used by ``flashinfer.rocm.hip_utils``. Set one of these only when ROCm is installed somewhere else.

Runtime helpers
====================================================================

Two helpers report whether the current process can use AITER and whether the PyTorch build matches the ROCm installation. Importing ``flashinfer`` already runs the ROCm compatibility check.

``is_aiter_supported`` returns ``True`` when the device is gfx942 (MI300X or MI325X) or gfx950 (MI350X or MI355X) and PyTorch is a ROCm build. It does not import the ``aiter`` package. Catch ``ImportError`` around ``import aiter`` when you also need to confirm that the package is installed.

``check_torch_rocm_compatibility`` warns when the ROCm version reported by PyTorch differs from the ROCm version on the system. It raises ``RuntimeError`` when PyTorch has no ROCm build, and ``ImportError`` when PyTorch is not installed.

PyTorch addresses AMD GPUs with the device string ``"cuda"``.

.. code-block:: python

   import torch

   from flashinfer.rocm.aiter_utils import is_aiter_supported
   from flashinfer.rocm.hip_utils import check_torch_rocm_compatibility

   device = torch.device("cuda")
   aiter_ready = is_aiter_supported(device)
   check_torch_rocm_compatibility()
