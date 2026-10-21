.. meta::
  :description: What is FlashInfer?
  :keywords: FlashInfer, documentation, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-flashinfer:

********************************************************************
What is FlashInfer?
********************************************************************

`FlashInfer <https://docs.flashinfer.ai/index.html>`__ is a library and kernel generator
for Large Language Models (LLMs) that provides a high-performance implementation of kernels
for graphics processing units (GPUs). FlashInfer focuses on LLM serving and inference, as well
as advanced performance across diverse scenarios.

FlashInfer features highly efficient attention kernels, load-balanced scheduling, and memory-optimized
techniques, while supporting customized attention variants. It’s compatible with ``torch.compile``, and
offers high-performance LLM-specific operators, with easy integration through PyTorch, and C++ APIs.

FlashInfer on ROCm supports both stages of LLM inference:

- **Prefill** – processes the input prompt to construct KV caches and activations.
- **Decode** – generates tokens sequentially using previously computed states.

Why FlashInfer?
====================================================================

FlashInfer is well suited for LLM inference acceleration for the following reasons:

- Its specialized attention kernels target the critical path of decoding
  to deliver substantial latency and throughput gains.

- Paged cache and fused operations reduce memory bandwidth pressure
  and improve efficiency at long sequence lengths.

- Flexible integration allows drop-in acceleration within existing
  inference stacks and services.

- Optimized for GPUs with ROCm builds to leverage AMD Instinct hardware
  effectively in production environments.

Features and use cases
====================================================================

FlashInfer provides the following key features:

- **High-Performance Attention Kernels:** Delivers optimized prefill and
  decode kernels for transformer attention to minimize latency on ROCm GPUs.

- **Paged KV Cache Management:** Implements efficient KV cache layouts and
  eviction strategies to sustain long-context, high-throughput generation.

- **Kernel Generator and Fusion:** Provides a flexible kernel generation
  approach with fused operations to reduce memory transfers and overhead.

- **Streaming and Batched Inference:** Supports both real-time streaming
  generation and large-batch workloads with dynamic sequence handling.

- **Model Compatibility:** Works with common attention variants (for example, RoPE
  and multi-query/group-query attention) to accelerate modern LLMs.

FlashInfer on ROCm also includes performance-enhancing features:

- Load balancing across GPU compute units
- Sparse and dense attention optimizations
- Batch and single‑sequence decode
- High‑throughput prefill optimized for AMD Instinct MI300X, MI325X, and MI355X GPUs

FlashInfer is commonly used in the following scenarios:

- **Real-Time Serving:** Power low-latency chat, agent, and code-completion
  systems with optimized attention paths.

- **Large-Scale Batch Generation:** Accelerate batched inference for content
  creation, retrieval-augmented generation, and evaluation jobs.

- **Throughput-Critical Pipelines:** Reduce end-to-end latency in model
  serving stacks and microservices.

- **Research on Kernel Efficiency:** Prototype new attention and cache
  strategies to improve inference performance on AMD Instinct GPUs.

For currently supported use cases and recommendations, refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/search.html?q=flashinfer>`__,
where you can search for examples and best practices to optimize your workloads on AMD GPUs.

Kernel support matrix
====================================================================

FlashInfer on ROCm supports the following attention modes:

- **Multi-Head Attention (MHA):** Each attention head has its own keys and values.
- **Grouped-Query Attention (GQA):** Groups of query heads share keys and values, reducing KV cache size and memory traffic.
- **Multi-Query Attention (MQA):** All query heads share a single key and value stream, minimizing KV cache size.

The `AITER <https://github.com/ROCm/aiter>`__ backend is the first-class citizen in
FlashInfer. Where supported, operations run on AITER and fall back to in-tree HIP
kernels otherwise. For APIs exposing a ``backend`` argument, its default is
``"auto"``. Single/batch prefill and batch decode use AITER only for compatible
calls and otherwise fall back to HIP. Other defaults are listed in the
**Default** column.

Where supported, select AITER with ``backend="aiter"`` or HIP with
``backend="fa2"`` (attention) or ``backend="native"`` (other routed operations).
``single_decode_with_kv_cache``, ``aiter_fused_moe``, and the cascade wrappers
do not accept ``backend``; cascade attention levels are routed automatically.

All operations listed below are supported on AMD Instinct MI300X, MI325X, and MI355X GPUs.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 22 8 8 12 50

   * - Operation
     - HIP
     - AITER
     - Default
     - Notes
   * - Single decode
     - ✅
     - —
     - HIP
     - MHA, GQA, and MQA; FP8 KV cache
   * - Batch decode
     - ✅
     - ✅
     - AITER
     - MHA, GQA, and MQA; sliding window; FP8 KV cache; graph capture
   * - Single prefill
     - ✅
     - ✅
     - AITER
     - MHA, GQA, and MQA; sliding window; custom attention masks (HIP); FP8 KV cache on HIP. AITER requires matching FP16 or BF16 queries and KV.
   * - Batch prefill
     - ✅
     - ✅
     - AITER
     - Paged and ragged KV cache; sliding window; custom attention masks (HIP); FP8 KV cache on HIP, and on AITER's paged route at head dimension 128 or 256
   * - Block-sparse attention
     - ✅
     - —
     - HIP
     - Fixed and variable block sizes
   * - Cascade attention
     - ✅
     - ✅
     - HIP
     - Shared-prefix attention; each level is routed independently
   * - POD attention
     - ✅
     - —
     - HIP
     - Fused prefill and decode
   * - Multi-head latent attention (MLA)
     - —
     - ✅
     - AITER
     - DeepSeek V2 and V3 layout; FP16 and BF16
   * - Fused MoE
     - —
     - ✅
     - AITER
     - FP16, BF16, and FP8 weights
   * - RoPE
     - ✅
     - ✅
     - HIP
     - LLaMA and LLaMA 3.1 scaling; fused RoPE with FP8 quantization and paged KV append
   * - Append paged KV cache
     - ✅
     - ✅
     - HIP
     - FP8 KV cache
   * - RMSNorm and fused add RMSNorm
     - ✅
     - ✅
     - HIP
     - Includes Gemma RMSNorm variants (HIP only)
   * - Activation
     - ✅
     - ✅
     - HIP
     - SiLU and GELU with fused gating (GELU is HIP only)
   * - Sampling
     - ✅
     - —
     - HIP
     - Top-K, Top-P, Min-P, online softmax, and sampling from logits
   * - Logits processor
     - ✅
     - —
     - HIP
     - Composable processing pipeline
   * - Quantization
     - ✅
     - —
     - HIP
     - ``packbits`` and ``segment_packbits``

For per-operation constraints, backend selection rules, and the list of
upstream modules that are not available on ROCm, refer to the
`support matrix <https://github.com/AMD-Ecosystem/flashinfer#support-matrix>`__
in the FlashInfer on ROCm ``README``.
