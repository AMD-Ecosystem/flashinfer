# `torch.compile` on ROCm

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer` to
register the kernels as `torch.library` custom ops so TorchDynamo can trace
them (PyTorch ≥ 2.4). Each eager call pays a host-side dispatch cost, measured
at 21 µs and 11 µs per single prefill on our MI300X and MI355X hosts. Measured
in bf16 on gfx942 and gfx950 (identical) on the fa2 and AITER backends, each
compiled result checked against eager:

| Op | Flag off | Flag on | Flag on, `fullgraph=True` |
|---|---|---|---|
| Paged and ragged prefill, batch decode `run()` | fails | compiles | compiles¹ |
| Single prefill, `aiter_fused_moe` | fails | compiles | fails |
| MLA | compiles | compiles | fails |
| `append_paged_kv_cache`, `append_paged_mla_kv_cache` | fails | compiles | compiles² |

¹ Except AITER decode with `return_lse=True`, which breaks the graph to build
its fa2 LSE plan eagerly; use `backend="fa2"` there.
² Except `append_paged_kv_cache(backend="aiter")`, which breaks the graph.

Ops outside the table were not measured beyond the known issue below; sampling
ops that take a `generator`, POD, `jit_args` custom variants, the
fp8-quantizing norms and AITER's opt-in norm, RoPE and activation backends are
not registered. Call `plan()` outside the compiled region: its ops are not
registered either. With the flag off the error is often torch's `RuntimeError
when making fake tensor call` rather than a message naming the flag.

**Known issue:** under every `dynamic` setting, PyTorch 2.12 can miscompile an
op that writes into a view of a larger tensor, silently or with a GPU fault.
That covers an `out=` slice, an in-place norm on part of a buffer, K and V
caches carved from one allocation (a layer of a multi-layer pool, the halves of
a `[2, pages, ...]` tensor), and the single `[pages, 2, ...]` cache
`append_paged_kv_cache` accepts, which it splits into K and V views itself.
Allocate every tensor an op writes as its own buffer up front, rather than
slicing it from a larger one, and pass append two separately allocated caches;
a `.clone()` per call is not a fix, since the write never reaches the original.
In-place RoPE on `qkv.split` views and MLA's `torch.split(buf, [512, 64],
dim=-1)` cache layout measured exact, so keep those. Under
`mode="reduce-overhead"`, a KV cache passed as an argument is copied into the
CUDA graph on every replay (measured with AITER decode); register it as a
module buffer (`persistent=False`) instead.
