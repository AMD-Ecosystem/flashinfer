"""
Copyright (c) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Acceptance test + timing for graph-capturable AITER decode (plan §3 #1).

Verifies that AITER PA v1 decode, captured once at a maximum sequence length,
replays *correctly* for shorter sequences (the kernel early-exits per-seq on
context_lens), and times aiter-under-graph vs fa2-under-graph.

Run:
    python benchmarks/rocm_benchmarks/bench_decode_graph_hip.py
"""

import os
import sys
import time

import torch

import flashinfer
from flashinfer.aiter_utils import is_aiter_supported

# Fixed problem geometry (Llama-70B TP1-ish decode). BATCH/CAP_SEQ overridable
# via env for quick scaling checks: FI_GRAPH_BATCH, FI_GRAPH_CAP_SEQ.
BATCH = int(os.environ.get("FI_GRAPH_BATCH", "16"))
PAGE = 16
NUM_QO, NUM_KV, HD = 32, 8, 128
DTYPE = torch.bfloat16
CAP_SEQ = int(
    os.environ.get("FI_GRAPH_CAP_SEQ", "4096")
)  # capacity captured; replays <= this
CAP_PAGES_PER_SEQ = (CAP_SEQ + PAGE - 1) // PAGE
TOTAL_PAGES = BATCH * CAP_PAGES_PER_SEQ
DEVICE = torch.device("cuda")


def _layout_for(seq_len: int):
    """Uniform per-seq kv_len=seq_len over the fixed page pool.

    Each sequence keeps a stable reserved block of CAP_PAGES_PER_SEQ pages; a
    shorter seq_len uses only the first `npages` of its block. This mirrors a
    real fixed-capacity paged-KV pool (stable per-seq page mapping) rather than
    repacking pages when seq_len < CAP_SEQ.
    """
    npages = (seq_len + PAGE - 1) // PAGE
    last = seq_len - (npages - 1) * PAGE
    indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=DEVICE) * npages
    base = (torch.arange(BATCH, device=DEVICE) * CAP_PAGES_PER_SEQ).view(-1, 1)
    offs = torch.arange(npages, device=DEVICE).view(1, -1)
    indices = (base + offs).reshape(-1).to(torch.int32)
    last_page = torch.full((BATCH,), last, dtype=torch.int32, device=DEVICE)
    return indptr, indices, last_page


def _reference(q, kv, seq_len, backend):
    """Eager (non-graph) wrapper result for the same inputs."""
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend=backend)
    indptr, indices, last_page = _layout_for(seq_len)
    w.plan(
        indptr,
        indices,
        last_page,
        NUM_QO,
        NUM_KV,
        HD,
        PAGE,
        pos_encoding_mode="NONE",
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    return w.run(q, kv)


def _make_graph_wrapper(backend):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    indptr_buf = torch.empty(BATCH + 1, dtype=torch.int32, device=DEVICE)
    indices_buf = torch.empty(TOTAL_PAGES, dtype=torch.int32, device=DEVICE)
    last_page_buf = torch.empty(BATCH, dtype=torch.int32, device=DEVICE)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        use_cuda_graph=True,
        backend=backend,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_buf,
    )
    return w


def _capture(w, q_static, kv):
    """plan() at capacity, then capture run() into a static output."""
    indptr, indices, last_page = _layout_for(CAP_SEQ)
    w.plan(
        indptr,
        indices,
        last_page,
        NUM_QO,
        NUM_KV,
        HD,
        PAGE,
        pos_encoding_mode="NONE",
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    # warmup (also triggers dlopen / first alloc outside capture)
    for _ in range(3):
        w.run(q_static, kv)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_static = w.run(q_static, kv)
    return g, out_static


def main():
    if not is_aiter_supported(DEVICE):
        print("AITER not supported on this device. Exiting.")
        return

    torch.manual_seed(0)
    kv = torch.randn(TOTAL_PAGES, 2, PAGE, NUM_KV, HD, dtype=DTYPE, device=DEVICE)
    q_static = torch.randn(BATCH, NUM_QO, HD, dtype=DTYPE, device=DEVICE)

    print(f"Backend resolves under graph capture (batch={BATCH}, cap_seq={CAP_SEQ}):")
    w = _make_graph_wrapper("aiter")
    g, out_static = _capture(w, q_static, kv)
    print(f"  captured backend = {w._backend!r}")
    assert w._backend == "aiter", f"expected aiter under graph, got {w._backend!r}"

    print("\nReplay correctness across seq_len <= capacity (vs eager aiter):")
    ok = True
    for seq_len in [512, 1024, 2048, 4096]:
        # fresh q contents for this step
        q_new = torch.randn(BATCH, NUM_QO, HD, dtype=DTYPE, device=DEVICE)
        q_static.copy_(q_new)
        # update fixed buffers to the real (shorter) layout — NOT captured
        indptr, indices, last_page = _layout_for(seq_len)
        w.plan(
            indptr,
            indices,
            last_page,
            NUM_QO,
            NUM_KV,
            HD,
            PAGE,
            pos_encoding_mode="NONE",
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )
        g.replay()
        torch.cuda.synchronize()
        ref = _reference(q_new, kv, seq_len, "aiter")
        max_diff = (out_static.float() - ref.float()).abs().max().item()
        good = torch.allclose(out_static, ref, atol=2e-2, rtol=2e-2)
        ok = ok and good
        print(
            f"  seq_len={seq_len:>5d}  max|graph-eager|={max_diff:.4f}  "
            f"{'PASS' if good else 'FAIL'}"
        )

    print(f"\nOverall correctness: {'PASS' if ok else 'FAIL'}")

    # ── timing: aiter-under-graph vs fa2-under-graph (replay only) ──────────
    print(f"\nUnder-graph replay latency (seq_len=4096, batch={BATCH}):")
    for backend in ["fa2", "aiter"]:
        wl = _make_graph_wrapper(backend)
        gl, _ = _capture(wl, q_static, kv)
        # set a mid-size real layout
        indptr, indices, last_page = _layout_for(4096)
        wl.plan(
            indptr,
            indices,
            last_page,
            NUM_QO,
            NUM_KV,
            HD,
            PAGE,
            pos_encoding_mode="NONE",
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )
        for _ in range(5):
            gl.replay()
        torch.cuda.synchronize()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            gl.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        print(f"  {backend:<5s}  {ms:.4f} ms / replay")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
