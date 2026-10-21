.. meta::
  :description: FlashInfer examples
  :keywords: FlashInfer, programming, ROCm, example, sample, tutorial

.. _run-a-flashinfer-example:

********************************************************************
Run a FlashInfer example
********************************************************************

AITER supports single and batch prefill, along with other operations listed in the :doc:`kernel support matrix <../what-is-flashinfer>`.
The ``auto`` backend calls the most optimized kernels from AITER or the in-tree HIP kernels, based on the benchmarking results.
To use AITER as the backend for any operation explicitly, use the ``backend="aiter"`` keyword argument to invoke the kernels.

1. Save the following code snippet to a Python script named ``flashinfer_example.py``.

   .. code-block:: python

      import torch
      import flashinfer

      # Configuration
      seq_len = 1024        # Prompt length
      num_qo_heads = 32     # Number of query/output heads
      num_kv_heads = 8      # Number of KV heads (GQA with 4:1 ratio)
      head_dim = 128

      # Create Q, K, V tensors (NHD layout: sequence, heads, dimension)
      q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
      k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
      v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

      # Run single prefill attention with causal masking
      output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")

2. Run the script to use FlashInfer.

   .. code-block:: bash

      python flashinfer_example.py
