.. meta::
  :description: What is FlashInfer?
  :keywords: FlashInfer, documentation, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-flashinfer:

********************************************************************
What is FlashInfer?
********************************************************************

`FlashInfer <https://docs.flashinfer.ai/index.html>`__ is a library and kernel generator 
for Large Language Models (LLMs) that provides a high-performance implementation of kernels
for graphic processing units (GPUs). FlashInfer focuses on LLM serving and inference, as well 
as advanced performance across diverse scenarios.

FlashInfer features highly efficient attention kernels, load-balanced scheduling, and memory-optimized 
techniques, while supporting customized attention variants. It’s compatible with ``torch.compile``, and 
offers high-performance LLM-specific operators, with easy integration through PyTorch, and C++ APIs.

Features and use cases
====================================================================

.. note::

  The ROCm port of FlashInfer is under active development, and some features are not yet available. 
  For the most up-to-date feature support matrix, refer to the ``README`` in the 
  `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__ repository.

FlashInfer provides the following key features:

- **High-Performance Attention Kernels:** Delivers optimized prefill and
  decode kernels for transformer attention to minimize latency on ROCm GPUs.

- **Paged KV Cache Management:** Implements efficient KV cache layouts and
  eviction strategies to sustain long-context, high-throughput generation.

- **Kernel Generator and Fusion:** Provides a flexible kernel generation
  approach with fused operations to reduce memory transfers and overhead.

- **Streaming and Batched Inference:** Supports both real-time streaming
  generation and large-batch workloads with dynamic sequence handling.

- **Model Compatibility:** Works with common attention variants (e.g., RoPE,
  multi-query/group-query attention) to accelerate modern LLMs.

FlashInfer is commonly used in the following scenarios:

- **Real-Time Serving:** Power low-latency chat, agent, and code-completion
  systems with optimized attention paths.

- **Large-Scale Batch Generation:** Accelerate batched inference for content
  creation, retrieval-augmented generation, and evaluation jobs.

- **Throughput-Critical Pipelines:** Reduce end-to-end latency in model
  serving stacks and microservices.

- **Research on Kernel Efficiency:** Prototype new attention and cache
  strategies to push inference performance on AMD Instinct GPUs.

For currently supported use cases and recommendations, refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/search.html?q=flashinfer>`__, 
where you can search for examples and best practices to optimize your workloads on AMD GPUs.

Why FlashInfer?
====================================================================

FlashInfer is well suited for LLM inference acceleration for the following reasons:

- Its **specialized attention kernels** target the critical path of decoding
  to deliver substantial latency and throughput gains.

- **Paged cache and fused operations** reduce memory bandwidth pressure
  and improve efficiency at long sequence lengths.

- **Flexible integration** allows drop-in acceleration within existing
  inference stacks and services.

- **Optimized for GPUs** with ROCm builds to leverage AMD Instinct hardware
  effectively in production environments.