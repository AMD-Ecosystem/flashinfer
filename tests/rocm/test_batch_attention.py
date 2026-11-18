# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BatchAttention (persistent / holistic kernel) on ROCm.

Port of upstream tests/attention/test_batch_attention.py. Upstream compares
against BatchPrefillWithPagedKVCacheWrapper; here the reference is plain fp32
torch, so a mismatch can only come from the kernel under test. The workloads are
upstream's _build_seq_len_configs, and the tolerance is upstream's
(rtol = atol = 1e-2 on both out and the base-2 lse). Upstream's full cartesian
product (23,040 cases) is cut to every axis value on a small mixed batch, plus
every workload mix at head_dim 128.
"""

import math

import numpy as np
import pytest
import torch
from jit_utils import gen_persistent_batch_attention_modules

import flashinfer

DEVICE = "cuda:0"
LOG2E = 1.4426950408889634


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_persistent_batch_attention_modules(
            [torch.float16, torch.bfloat16],  # q_dtypes
            [torch.float16, torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [False, True],  # use_logits_soft_cap
        ),
        verbose=False,
    )
    yield


def _build_seq_len_configs():
    """Upstream's workloads: lists of (kv_len, qo_len)."""
    np.random.seed(42)
    configs = [
        [(146, 146)],
        [(67, 67)],
        [(8190, 7939)],
        [(2048, 1)] * 77,  # decode-only
        [(4099, 129)] * 2,  # prefill-only
        [(600, 1)] * 132 * 2 + [(5000, 3)] * 128,
        [(1024, 1)] * 100 + [(8192, 17)] * 8,  # speculative decode
        [(766, 2)] * 99 + [(1024, 512)] * 1,  # chunked prefill
        [(2, 235)] + [(1, 13353)],  # real workload
    ]
    bsz, stride, sparsity = 256, 16, 0.05
    full_kv_len = np.random.randint(1000, 11000, size=bsz)
    seq_len = []
    for i in range(bsz):
        if i % stride == 0:
            seq_len.append((int(full_kv_len[i]), stride + 1))
        else:
            seq_len.append((int(full_kv_len[i] * sparsity), 1))
    configs.append(seq_len)
    return configs


SEQ_LEN_CONFIGS = _build_seq_len_configs()
SEQ_LEN_IDS = [
    "146x146",
    "67x67",
    "8190x7939",
    "decode",
    "prefill",
    "mixed",
    "spec-decode",
    "chunked-prefill",
    "real",
    "random",
]
# The two largest mixes run with one KV head to bound the reference's memory.
BIG_SEQ_LEN_IDS = {"8190x7939", "random"}
SMALL_MIXED = [
    (146, 146),
    (67, 67),
    (2048, 1),
    (33, 1),
    (300, 17),
    (600, 1),
    (1500, 3),
]


def _indptr(lens):
    zero = torch.zeros(1, dtype=torch.int32, device=lens.device)
    return torch.cat([zero, torch.cumsum(lens, 0)]).int()


def _make_inputs(
    kv_lens,
    qo_lens,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    dtype,
    noncontiguous_q=False,
):
    dev = torch.device(DEVICE)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32, device=dev)
    q_indptr = _indptr(q_lens)
    kv_indptr = _indptr((seq_lens + page_size - 1) // page_size)
    num_blocks = int(kv_indptr[-1].item())
    num_tokens = int(q_indptr[-1].item())
    if noncontiguous_q:
        q = torch.rand(num_tokens, num_qo_heads, head_dim * 2, dtype=dtype, device=dev)
        q = torch.chunk(q, 2, dim=-1)[0]
    else:
        q = torch.rand(num_tokens, num_qo_heads, head_dim, dtype=dtype, device=dev)
    if layout == "NHD":
        shape = (num_blocks, 2, page_size, num_kv_heads, head_dim)
    else:
        shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)
    kv_data = torch.randn(*shape, dtype=dtype, device=dev)
    # Scattered pages, so the page table is actually exercised.
    kv_indices = torch.randperm(num_blocks, device=dev).int()
    return q, kv_data, q_indptr, kv_indptr, kv_indices, seq_lens


def _reference(
    q,
    kv_data,
    q_indptr,
    kv_indptr,
    kv_indices,
    kv_lens,
    layout,
    causal,
    sm_scale,
    soft_cap,
    v_scale,
    row_chunk=512,
):
    """fp32 attention per request with a base-2 lse.

    Rows that see no key (causal, kv_len < qo_len) give out 0 and lse -inf.
    """
    num_qo_heads, head_dim = q.shape[1], q.shape[2]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.full(q.shape[:2], -math.inf, dtype=torch.float32, device=q.device)
    q_ind, kv_ind, lens = q_indptr.tolist(), kv_indptr.tolist(), kv_lens.tolist()
    for i in range(len(lens)):
        qs, qe, kv_len = q_ind[i], q_ind[i + 1], lens[i]
        if qe == qs or kv_len == 0:
            continue
        pages = kv_indices[kv_ind[i] : kv_ind[i + 1]].long()
        kv = kv_data[pages]  # [pages, 2, ...]
        if layout == "HND":
            kv = kv.transpose(2, 3)  # -> [pages, 2, page_size, H, D]
        num_kv_heads = kv.shape[3]
        k = kv[:, 0].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        v = kv[:, 1].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        group = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        q_len = qe - qs
        key_pos = torch.arange(kv_len, device=q.device)
        for r0 in range(0, q_len, row_chunk):
            r1 = min(q_len, r0 + row_chunk)
            rows = slice(qs + r0, qs + r1)
            s = torch.einsum("qhd,khd->hqk", q[rows].float(), k) * sm_scale
            if soft_cap > 0:
                s = soft_cap * torch.tanh(s / soft_cap)
            if causal:
                row_pos = torch.arange(r0, r1, device=q.device)
                visible = key_pos[None, :] <= (kv_len - q_len + row_pos)[:, None]
                s = s.masked_fill(~visible[None], -math.inf)
            m = s.amax(-1, keepdim=True)
            has_key = torch.isfinite(m)
            m = torch.where(has_key, m, torch.zeros_like(m))
            p = torch.exp(s - m)
            d = p.sum(-1, keepdim=True)
            o = torch.einsum("hqk,khd->qhd", p / d.clamp_min(1e-30), v) * v_scale
            row_lse = ((torch.log(d) + m) * LOG2E).squeeze(-1).transpose(0, 1)
            has_key_q = has_key.squeeze(-1).transpose(0, 1)
            out[rows] = torch.where(has_key_q[..., None], o, torch.zeros_like(o))
            lse[rows] = torch.where(has_key_q, row_lse, lse[rows])
    return out, lse


def _run_and_check(
    kv_lens,
    qo_lens,
    page_size=16,
    num_kv_heads=1,
    gqa_group_size=1,
    head_dim=128,
    v_scale=None,
    causal=True,
    layout="NHD",
    dtype=torch.bfloat16,
    logits_soft_cap=0.0,
    noncontiguous_q=False,
):
    num_qo_heads = num_kv_heads * gqa_group_size
    q, kv_data, q_indptr, kv_indptr, kv_indices, seq_lens = _make_inputs(
        kv_lens,
        qo_lens,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        layout,
        dtype,
        noncontiguous_q,
    )
    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out, lse = wrapper.run(q, kv_data, v_scale=v_scale, logits_soft_cap=logits_soft_cap)
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(
        q,
        kv_data,
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens,
        layout,
        causal,
        1.0 / math.sqrt(head_dim),
        logits_soft_cap,
        1.0 if v_scale is None else v_scale,
    )
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)


# Every upstream workload mix, both masks, both dtypes.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seq_len_id, seq_len_pairs", list(zip(SEQ_LEN_IDS, SEQ_LEN_CONFIGS, strict=True))
)
def test_batch_attention_workloads(seq_len_id, seq_len_pairs, causal, dtype):
    _run_and_check(
        [p[0] for p in seq_len_pairs],
        [p[1] for p in seq_len_pairs],
        page_size=16,
        num_kv_heads=1 if seq_len_id in BIG_SEQ_LEN_IDS else 4,
        gqa_group_size=4,
        causal=causal,
        dtype=dtype,
    )


# Every value of upstream's remaining axes, on a small mixed batch. Layout,
# v_scale, soft cap and dtype ride along so each of their values is covered
# without the full product.
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 7, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batch_attention_axes(
    head_dim, page_size, num_kv_heads, gqa_group_size, causal
):
    use_bf16 = (num_kv_heads + gqa_group_size + page_size) % 2 == 0
    _run_and_check(
        [p[0] for p in SMALL_MIXED],
        [p[1] for p in SMALL_MIXED],
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        head_dim=head_dim,
        v_scale=2.0 if gqa_group_size == 4 else None,
        causal=causal,
        layout="HND" if gqa_group_size % 2 else "NHD",
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        logits_soft_cap=50.0 if page_size == 8 else 0.0,
    )


def test_batch_attention_with_noncontiguous_q():
    _run_and_check(
        [146], [146], page_size=1, head_dim=64, causal=True, noncontiguous_q=True
    )


# Few long decode requests: the planner splits their KV across CTAs inside the
# 16-row task, which exercises the partial writes and the merge launch that
# follows the attention launch.
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "seq_len_pairs",
    [[(20000, 1)] * 3, [(30000, 1)] + [(100, 1)] * 20, [(25000, 2), (9000, 40)]],
    ids=["3-long-decodes", "1-long-20-short", "long-decode+prefill"],
)
def test_batch_attention_split_kv(seq_len_pairs, head_dim):
    _run_and_check(
        [p[0] for p in seq_len_pairs],
        [p[1] for p in seq_len_pairs],
        num_kv_heads=1,
        gqa_group_size=4,
        head_dim=head_dim,
        dtype=torch.float16,
    )


def test_batch_attention_graph_replays_across_batch_shapes():
    """One captured graph serves a re-planned, differently shaped batch."""
    head_dim, num_kv_heads, gqa, page_size = 128, 2, 4, 16
    dtype = torch.float16
    num_qo_heads = num_kv_heads * gqa
    batches = {
        "a": [(2048, 1)] * 40 + [(700, 300)] * 2,
        "b": [(333, 5)] * 9 + [(9000, 1)] * 3 + [(1500, 777)],
    }
    inputs = {
        name: _make_inputs(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            page_size,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            "NHD",
            dtype,
        )
        for name, pairs in batches.items()
    }
    # Static buffers sized for the larger batch; the graph keeps their addresses.
    max_tokens = max(x[0].shape[0] for x in inputs.values())
    max_blocks = max(x[1].shape[0] for x in inputs.values())
    q = torch.zeros(max_tokens, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    kv = torch.zeros(
        max_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    kv_indices = torch.zeros(max_blocks, dtype=torch.int32, device=DEVICE)
    out = torch.empty_like(q)
    lse = torch.empty(max_tokens, num_qo_heads, dtype=torch.float32, device=DEVICE)

    wrapper = flashinfer.BatchAttention(kv_layout="NHD")

    def load_and_plan(name):
        q_n, kv_n, q_indptr, kv_indptr, kv_indices_n, seq_lens = inputs[name]
        q[: q_n.shape[0]].copy_(q_n)
        kv[: kv_n.shape[0]].copy_(kv_n)
        kv_indices[: kv_indices_n.shape[0]].copy_(kv_indices_n)
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            seq_lens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    def check(name):
        q_n, kv_n, q_indptr, kv_indptr, kv_indices_n, seq_lens = inputs[name]
        n = q_n.shape[0]
        out_ref, lse_ref = _reference(
            q_n,
            kv_n,
            q_indptr,
            kv_indptr,
            kv_indices_n,
            seq_lens,
            "NHD",
            True,
            1.0 / math.sqrt(head_dim),
            0.0,
            1.0,
        )
        torch.testing.assert_close(out[:n].float(), out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse[:n], lse_ref, rtol=1e-2, atol=1e-2)

    load_and_plan("a")
    wrapper.run(q, kv, out=out, lse=lse)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out, lse=lse)

    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check("a")

    load_and_plan("b")  # only the contents of the plan's buffers change
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check("b")


# ------------------------------- refusals --------------------------------- #


def _tiny_plan(wrapper, dtype=torch.float16, kv_dtype=None, **kwargs):
    q_indptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    kv_lens = torch.tensor([16], dtype=torch.int32, device=DEVICE)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        4,  # num_qo_heads
        4,  # num_kv_heads
        128,  # head_dim_qk
        128,  # head_dim_vo
        16,  # page_size
        q_data_type=dtype,
        kv_data_type=kv_dtype or dtype,
        **kwargs,
    )


def _tiny_inputs():
    q = torch.randn(1, 4, 128, dtype=torch.float16, device=DEVICE)
    kv = torch.randn(1, 2, 16, 4, 128, dtype=torch.float16, device=DEVICE)
    return q, kv


def test_refuses_profiler():
    with pytest.raises(NotImplementedError, match="use_profiler"):
        _tiny_plan(flashinfer.BatchAttention(), use_profiler=True)


def test_refuses_fp8_kv_cache():
    with pytest.raises(NotImplementedError, match="fp8 and NVFP4"):
        _tiny_plan(flashinfer.BatchAttention(), kv_dtype=torch.float8_e4m3fnuz)


def test_refuses_nvfp4_scale_factors():
    wrapper = flashinfer.BatchAttention()
    _tiny_plan(wrapper)
    q, kv = _tiny_inputs()
    sf = torch.ones(1, 2, 16, 4, 8, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(RuntimeError, match="NVFP4"):
        wrapper.run(q, kv, kv_cache_sf=sf)


def test_refuses_soft_cap_planned_but_not_run():
    wrapper = flashinfer.BatchAttention()
    _tiny_plan(wrapper, logits_soft_cap=30.0)
    q, kv = _tiny_inputs()
    with pytest.raises(RuntimeError, match="logits_soft_cap"):
        wrapper.run(q, kv)


def test_refuses_attention_sink_wrapper():
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(NotImplementedError, match="attention sinks"):
        flashinfer.BatchAttentionWithAttentionSinkWrapper(workspace, backend="fa2")
