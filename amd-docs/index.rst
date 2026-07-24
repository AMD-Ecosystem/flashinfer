.. meta::
  :description: FlashInfer documentation
  :keywords: FlashInfer, ROCm, documentation, deep learning, framework, GPU

.. _flashinfer-documentation-index:

********************************************************************
FlashInfer on ROCm documentation
********************************************************************

Accelerate LLM attention and decoding kernels with FlashInfer on ROCm for
AMD Instinct GPUs, enabling high-throughput batch and streaming generation
for real-time applications like multilingual chat and code completion.

`FlashInfer <https://docs.flashinfer.ai/index.html>`__ is a library and kernel generator 
for Large Language Models (LLMs) that provides a high-performance implementation of
kernels for graphic processing units (GPUs). FlashInfer focuses on LLM serving and inference,
as well as advanced performance across diverse scenarios.

FlashInfer on ROCm includes capabilities such as load balancing, 
sparse and dense attention optimizations, and single and batch decode, alongside
prefill for high‑performance execution on AMD Instinct MI300X, MI325X, and MI355X GPUs.
FlashInfer features highly efficient attention kernels, load-balanced scheduling,
and memory-optimized techniques, while supporting customized attention variants.
It’s compatible with ``torch.compile``, and offers high-performance LLM-specific
operators, with easy integration through PyTorch, and C++ APIs.

.. note::

  The ROCm port of FlashInfer is under active development, and some features are not yet available. 
  For the most up-to-date feature support matrix, refer to the ``README`` in the 
  `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__ repository.

FlashInfer is part of the `AMD LLM Extension toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.03/>`__.

The FlashInfer public repository is located at `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install FlashInfer <install/flashinfer-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a FlashInfer example <examples/flashinfer-examples>`

  .. grid-item-card:: Reference

      * `API reference (upstream) <https://docs.flashinfer.ai/>`__

To contribute to the documentation, refer to
`Contributing to FlashInfer <https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md>`__.

You can find licensing information on the :doc:`Licensing <about/license>` page.
