.. meta::
  :description: FlashInfer examples
  :keywords: FlashInfer, programming, ROCm, example, sample, tutorial

.. _run-a-flashinfer-example:

********************************************************************
Run a FlashInfer example
********************************************************************

The examples folder in the `https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples <https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples>`__ repository
has example code that you can use to run FlashInfer. 

You also have the option to use the AITER backend for the prefill attention kernels.
The AITER backend is currently enabled for the ``single_prefill`` and ``batch_prefill`` kernels only.
To use AITER as the backend for these kernels, use the ``backend="aiter"`` keyword argument to invoke the kernels.

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
