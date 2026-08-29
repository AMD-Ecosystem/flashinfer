# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end cover for a use_softmax=false custom variant on batch prefill.

That branch dispatches to the sum merge rather than the LSE merge, and the shape
below is chosen to split KV so the merge actually runs. The unit tests in
test_cascade_hip.py pin the sum kernels; this pins the dispatch that reaches them.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_HIP, reason="ROCm-only prefill path"
)

# ROCm's variant contract differs from the CUDA examples in
# tests/utils/test_jit_example.py: window_left is read unconditionally by
# prefill.cuh, and the math helpers live in math.
FLASH_SIGMOID_DECL = r"""
struct FlashSigmoid : AttentionVariantBase {
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float logits_scale_log2, sigmoid_bias_log2e;

  template <typename Params>
  __device__ __host__ FlashSigmoid(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    logits_scale_log2 = params.logits_scale * math::log2e;
    sigmoid_bias_log2e = params.sigmoid_bias * math::log2e;
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx,
                            kv_head_idx, {
    return math::ptx_rcp(
        1.f + math::ptx_exp2(-float(logits) * logits_scale_log2 -
                                        sigmoid_bias_log2e));
  })
};
"""


@pytest.mark.slow
def test_use_softmax_false_ragged_prefill_matches_reference():
    # b=8/s=256/h=8 with head_dim=128 splits KV into 2 chunks per row, so the
    # sum merge runs; wider head counts and longer sequences do not split.
    batch_size, seq_len, num_heads, head_dim = 8, 256, 8, 128

    jit_args = (
        "batch_prefill_flash_sigmoid_use_softmax_false",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        head_dim,
        head_dim,
        [],
        [],
        ["logits_scale", "sigmoid_bias"],
        ["double", "double"],
        "FlashSigmoid",
        FLASH_SIGMOID_DECL,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    torch.manual_seed(42)
    indptr = torch.arange(0, batch_size * seq_len + 1, seq_len, dtype=torch.int32)
    wrapper.plan(
        indptr,
        indptr,
        num_heads,
        num_heads,
        head_dim,
        causal=False,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    # split_kv is the last plan-info field. Whether this shape splits depends on
    # the CU count, and without a split the sum merge never runs and the test
    # would pass without covering anything.
    assert wrapper._plan_info[-1].item() == 1, (
        "shape did not split KV; merge unexercised"
    )

    shape = (batch_size * seq_len, num_heads, head_dim)
    q = torch.randn(shape, dtype=torch.float16, device="cuda")
    k = torch.randn(shape, dtype=torch.float16, device="cuda")
    v = torch.randn(shape, dtype=torch.float16, device="cuda")

    logits_scale = 1.0 / math.sqrt(head_dim)
    sigmoid_bias = 0.25
    out = wrapper.run(q, k, v, logits_scale, sigmoid_bias)

    assert torch.isfinite(out.float()).all(), "sum merge produced a non-finite value"

    batched = (batch_size, seq_len, num_heads, head_dim)
    p = torch.sigmoid(
        torch.einsum(
            "bmhd,bnhd->bhmn", q.view(batched).float(), k.view(batched).float()
        )
        * logits_scale
        + sigmoid_bias
    )
    ref = (
        torch.einsum("bhmn,bnhd->bmhd", p, v.view(batched).float())
        .half()
        .reshape(shape)
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
