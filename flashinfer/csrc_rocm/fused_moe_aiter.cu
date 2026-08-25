// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch entry point for AITER's CK two-stage fused MoE. FlashInfer links the
// symbol-visible AITER modules (see flashinfer/jit/aiter_source.py) and calls the
// kernels directly — no Python `import aiter` at runtime.
//
// Routing is the caller's: this takes topk_ids/topk_weights and runs
// moe_sorting -> stage1 (gate/up + activation) -> stage2 (down + weighted sum).
// The routing weights are applied in stage2, matching the `-m 2` (mulWeightStage2)
// instances the JIT spec generates.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

#include <optional>
#include <string>

// AITER's public headers (moe_sorting.h, moe_ck.h) pull in <torch/extension.h> →
// full pybind11, which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor
// is at::Tensor, so forward-declare the entry points; the linker resolves them
// against the symbol-visible AITER .so. Both are at global namespace.
void moe_sorting_fwd(at::Tensor& topk_ids, at::Tensor& topk_weights, at::Tensor& sorted_token_ids,
                     at::Tensor& sorted_weights, at::Tensor& sorted_expert_ids,
                     at::Tensor& num_valid_ids, at::Tensor& moe_buf, int num_experts, int unit_size,
                     std::optional<at::Tensor> local_expert_mask,
                     std::optional<at::Tensor> num_local_tokens, int dispatch_policy);

void ck_moe_stage1(at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2,
                   at::Tensor& sorted_token_ids, at::Tensor& sorted_expert_ids,
                   at::Tensor& num_valid_ids, at::Tensor& out, int topk, std::string& kernelName,
                   std::optional<at::Tensor> w1_scale, std::optional<at::Tensor> a1_scale,
                   std::optional<int> block_m, std::optional<at::Tensor> sorted_weights,
                   int quant_type, int activation, std::optional<int> splitk, bool nt,
                   std::optional<std::string> dst_type);

void ck_moe_stage2(at::Tensor& inter_states, at::Tensor& w1, at::Tensor& w2,
                   at::Tensor& sorted_token_ids, at::Tensor& sorted_expert_ids,
                   at::Tensor& num_valid_ids, at::Tensor& out, int topk, std::string& kernelName,
                   std::optional<at::Tensor> w2_scale, std::optional<at::Tensor> a2_scale,
                   std::optional<int> block_m, std::optional<at::Tensor> sorted_weights,
                   int quant_type, int activation, std::optional<int> splitk, bool nt,
                   std::optional<std::string> dst_type);

namespace {

// aiter_enum.h: ActivationType { No = -1, Silu = 0, Gelu = 1, Swiglu = 2 }.
constexpr int kActivationSilu = 0;
constexpr int kActivationGelu = 1;
// aiter_enum.h: QuantType { No = 0, ... }.
constexpr int kQuantNone = 0;

// The CK heuristic dispatch (ck2stages_moe_stage{1,2}_heuristic_dispatch.hpp)
// enumerates exactly these and TORCH_CHECKs otherwise; reject here so the error
// names the argument rather than surfacing from inside CK.
bool is_supported_block_m(int64_t block_m) {
  return block_m == 32 || block_m == 64 || block_m == 128;
}

// Byte ranges of two tensors' occupied memory intersect.
bool overlaps(const at::Tensor& a, const at::Tensor& b) {
  if (!a.has_storage() || !b.has_storage() || a.device() != b.device()) return false;
  const auto* a0 = static_cast<const char*>(a.data_ptr());
  const auto* b0 = static_cast<const char*>(b.data_ptr());
  return a0 < b0 + b.nbytes() && b0 < a0 + a.nbytes();
}

}  // namespace

// hidden_states: [m, model_dim]
// w1:            [num_experts, 2 * inter_dim, model_dim]  (gate and up, concatenated)
// w2:            [num_experts, model_dim, inter_dim]
// topk_ids:      [m, topk]  int32, every value in [0, num_experts)
// topk_weights:  [m, topk]  float32
// out:           [m, model_dim]
//
// Caller precondition: topk_ids values are in range. Checking on device would
// cost a synchronize per call, so out-of-range ids (a -1 drop marker, or global
// ids against a local expert-parallel shard) index w1/w2 out of bounds instead.
void fused_moe_aiter(at::Tensor out, at::Tensor hidden_states, at::Tensor w1, at::Tensor w2,
                     at::Tensor topk_ids, at::Tensor topk_weights, int64_t block_m,
                     int64_t activation) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(hidden_states.device());

  TORCH_CHECK(activation == kActivationSilu || activation == kActivationGelu,
              "fused_moe_aiter: activation must be silu (0) or gelu (1), got ", activation);
  TORCH_CHECK(is_supported_block_m(block_m), "fused_moe_aiter: block_m must be 32, 64 or 128, got ",
              block_m);

  const auto dtype = hidden_states.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
              "fused_moe_aiter: hidden_states must be bfloat16 or float16, got ", dtype);
  TORCH_CHECK(w1.scalar_type() == dtype && w2.scalar_type() == dtype,
              "fused_moe_aiter: w1/w2 dtype must match hidden_states (", dtype, "), got ",
              w1.scalar_type(), " and ", w2.scalar_type());
  TORCH_CHECK(out.scalar_type() == dtype, "fused_moe_aiter: out dtype must match hidden_states (",
              dtype, "), got ", out.scalar_type());

  TORCH_CHECK(hidden_states.dim() == 2, "fused_moe_aiter: hidden_states must be 2-D, got ",
              hidden_states.dim(), "-D");
  TORCH_CHECK(w1.dim() == 3 && w2.dim() == 3, "fused_moe_aiter: w1 and w2 must be 3-D, got ",
              w1.dim(), "-D and ", w2.dim(), "-D");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2,
              "fused_moe_aiter: topk_ids and topk_weights must be 2-D");

  TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "fused_moe_aiter: topk_ids must be int32, got ",
              topk_ids.scalar_type());
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
              "fused_moe_aiter: topk_weights must be float32, got ", topk_weights.scalar_type());

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t model_dim = hidden_states.size(1);
  const int64_t num_experts = w1.size(0);
  const int64_t inter_dim = w2.size(2);
  const int64_t topk = topk_ids.size(1);

  TORCH_CHECK(w1.size(1) == 2 * inter_dim,
              "fused_moe_aiter: expected w1 [E, 2 * inter_dim, model_dim] with inter_dim=",
              inter_dim, " from w2, got w1.size(1)=", w1.size(1));
  TORCH_CHECK(w1.size(2) == model_dim, "fused_moe_aiter: w1.size(2)=", w1.size(2),
              " must equal model_dim=", model_dim);
  TORCH_CHECK(w2.size(0) == num_experts, "fused_moe_aiter: w1 has ", num_experts,
              " experts but w2 has ", w2.size(0));
  TORCH_CHECK(w2.size(1) == model_dim, "fused_moe_aiter: w2.size(1)=", w2.size(1),
              " must equal model_dim=", model_dim);
  TORCH_CHECK(topk_ids.size(0) == num_tokens && topk_weights.size(0) == num_tokens,
              "fused_moe_aiter: topk_ids/topk_weights must have ", num_tokens, " rows");
  TORCH_CHECK(topk_weights.size(1) == topk, "fused_moe_aiter: topk_weights has ",
              topk_weights.size(1), " columns but topk_ids has ", topk);
  TORCH_CHECK(topk <= num_experts, "fused_moe_aiter: topk=", topk,
              " exceeds num_experts=", num_experts);
  TORCH_CHECK(out.size(0) == num_tokens && out.size(1) == model_dim,
              "fused_moe_aiter: out must be [", num_tokens, ", ", model_dim, "]");

  TORCH_CHECK(hidden_states.is_contiguous() && w1.is_contiguous() && w2.is_contiguous(),
              "fused_moe_aiter: hidden_states, w1 and w2 must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              "fused_moe_aiter: topk_ids and topk_weights must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "fused_moe_aiter: out must be contiguous");

  // Every tensor is read or written by kernels launched on hidden_states' device.
  const auto device = hidden_states.device();
  TORCH_CHECK(w1.device() == device && w2.device() == device && topk_ids.device() == device &&
                  topk_weights.device() == device && out.device() == device,
              "fused_moe_aiter: every tensor must be on ", device, "; got w1=", w1.device(),
              " w2=", w2.device(), " topk_ids=", topk_ids.device(),
              " topk_weights=", topk_weights.device(), " out=", out.device());

  // out doubles as moe_sorting_fwd's zero-filled accumulation buffer, so it is
  // cleared before stage 1 reads the activations. Aliasing an input would feed
  // stage 1 zeros and return an all-zero result with no error.
  TORCH_CHECK(!overlaps(out, hidden_states),
              "fused_moe_aiter: out must not overlap hidden_states (out is zero-filled "
              "before the activations are read); pass a separate tensor");
  TORCH_CHECK(!overlaps(out, w1) && !overlaps(out, w2),
              "fused_moe_aiter: out must not overlap w1 or w2");

  // Sorting buffer sizes, mirroring aiter/fused_moe.py::moe_sorting. Note
  // num_valid_ids is 2 int32 even though moe_sorting.h comments it as [1].
  const int64_t max_num_tokens_padded = topk_ids.numel() + num_experts * block_m - topk;
  const int64_t max_num_m_blocks = (max_num_tokens_padded + block_m - 1) / block_m;

  const auto i32 = hidden_states.options().dtype(at::kInt);
  const auto f32 = hidden_states.options().dtype(at::kFloat);

  at::Tensor sorted_token_ids = at::empty({max_num_tokens_padded}, i32);
  at::Tensor sorted_weights = at::empty({max_num_tokens_padded}, f32);
  at::Tensor sorted_expert_ids = at::empty({max_num_m_blocks}, i32);
  at::Tensor num_valid_ids = at::empty({2}, i32);

  // moe_sorting_fwd zero-fills moe_buf, and stage2 accumulates the per-expert
  // contributions into it — so `out` is the sorting buffer, not a separate one.
  moe_sorting_fwd(topk_ids, topk_weights, sorted_token_ids, sorted_weights, sorted_expert_ids,
                  num_valid_ids, out, static_cast<int>(num_experts), static_cast<int>(block_m),
                  /*local_expert_mask=*/std::nullopt, /*num_local_tokens=*/std::nullopt,
                  /*dispatch_policy=*/0);

  at::Tensor inter_states =
      at::empty({num_tokens, topk, inter_dim}, hidden_states.options().dtype(dtype));

  // Empty kernelName selects AITER's heuristic dispatch (moe_dispatch in
  // gemm_moe_ck2stages.cu); a tuned name would come from aiter's tuned_fmoe.csv.
  std::string kernel_name;
  const int topk_i32 = static_cast<int>(topk);
  const int block_m_i32 = static_cast<int>(block_m);
  const int activation_i32 = static_cast<int>(activation);

  ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids,
                inter_states, topk_i32, kernel_name, /*w1_scale=*/std::nullopt,
                /*a1_scale=*/std::nullopt, block_m_i32, /*sorted_weights=*/std::nullopt, kQuantNone,
                activation_i32, /*splitk=*/1, /*nt=*/false,
                /*dst_type=*/std::nullopt);

  ck_moe_stage2(inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out,
                topk_i32, kernel_name, /*w2_scale=*/std::nullopt, /*a2_scale=*/std::nullopt,
                block_m_i32, sorted_weights, kQuantNone, activation_i32, /*splitk=*/1,
                /*nt=*/false, /*dst_type=*/std::nullopt);
}
