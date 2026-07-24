.. meta::
  :description: FlashInfer examples
  :keywords: FlashInfer, programming, ROCm, example, sample, tutorial

.. _run-a-flashinfer-example:

********************************************************************
Run a FlashInfer example
********************************************************************

The examples folder in the `https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples <https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples>`__ repository has example code that you can run FlashInfer with.
You can save the following code snippet to a Python script once you have FlashInfer installed and run the script to try it out.

1. Save the following code snippet to a Python script named ``flashinfer_example.py``.

   .. code-block:: bash

      import torch
      import flashinfer

      kv_len = 2048
      num_kv_heads = 32
      head_dim = 128

      k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
      v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

      # decode attention

      num_qo_heads = 32
      q = torch.randn(num_qo_heads, head_dim).half().to(0)

      o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
      o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

2. Run the script to use FlashInfer.

   .. code-block:: bash

      python flashinfer_example.py