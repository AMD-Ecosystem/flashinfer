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
//
// With fp8 expert weights the shim also quantizes the activations per token,
// before each GEMM, since CK's per_Token instances take fp8 on both operands.
// Stage 1 still writes bf16/fp16 (gemm_moe_ck2stages.cu TORCH_CHECKs it), so the
// intermediate is quantized again between the stages.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>

// AITER's public headers (moe_sorting.h, moe_ck.h) pull in <torch/extension.h> →
// full pybind11, which clashes with FlashInfer's -DPy_LIMITED_API. torch::Tensor
// is at::Tensor, so forward-declare the entry points; the linker resolves them
// against the symbol-visible AITER .so. These three are at global namespace.
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

#ifdef FLASHINFER_MOE_AITER_PER_TOKEN
// Unlike the three above, this one AITER declares inside `namespace aiter`.
namespace aiter {
void dynamic_per_token_scaled_quant(at::Tensor& out, at::Tensor const& input, at::Tensor& scales,
                                    std::optional<at::Tensor> scale_ub, bool shuffle_scale,
                                    std::optional<at::Tensor> num_rows, int num_rows_factor);
}  // namespace aiter
#endif

namespace {

// aiter_enum.h: ActivationType { No = -1, Silu = 0, Gelu = 1, Swiglu = 2 }.
constexpr int kActivationSilu = 0;
constexpr int kActivationGelu = 1;
// aiter_enum.h: QuantType { No = 0, per_Tensor = 1, per_Token = 2, ... }.
constexpr int kQuantNone = 0;
constexpr int kQuantPerToken = 2;

bool is_fp8(at::ScalarType t) { return t == at::kFloat8_e4m3fn || t == at::kFloat8_e4m3fnuz; }

#ifdef FLASHINFER_MOE_AITER_PER_TOKEN
// Quantize `x` row-wise into `fp8`, returning the values and their [.., 1] fp32
// scales -- the shape AITER's own fused_moe hands to CK.
std::pair<at::Tensor, at::Tensor> quantize_per_token(const at::Tensor& x, at::ScalarType fp8,
                                                     int64_t num_rows_factor) {
  auto scale_sizes = x.sizes().vec();
  scale_sizes.back() = 1;
  at::Tensor q = at::empty(x.sizes(), x.options().dtype(fp8));
  at::Tensor scale = at::empty(scale_sizes, x.options().dtype(at::kFloat));
  aiter::dynamic_per_token_scaled_quant(q, x, scale, /*scale_ub=*/std::nullopt,
                                        /*shuffle_scale=*/false, /*num_rows=*/std::nullopt,
                                        static_cast<int>(num_rows_factor));
  return {q, scale};
}
#endif

void check_weight_scale(const at::Tensor& scale, const char* name, int64_t num_experts,
                        int64_t rows, const at::Device& device) {
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "fused_moe_aiter: ", name,
              " must be float32, got ", scale.scalar_type());
  TORCH_CHECK(scale.dim() == 3 && scale.size(0) == num_experts && scale.size(1) == rows &&
                  scale.size(2) == 1,
              "fused_moe_aiter: ", name, " must be [", num_experts, ", ", rows, ", 1], got ",
              scale.sizes());
  TORCH_CHECK(scale.is_contiguous(), "fused_moe_aiter: ", name, " must be contiguous");
  TORCH_CHECK(scale.device() == device, "fused_moe_aiter: ", name, " must be on ", device, ", got ",
              scale.device());
}

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
// w1_scale:      [num_experts, 2 * inter_dim, 1] float32, fp8 weights only
// w2_scale:      [num_experts, model_dim, 1]     float32, fp8 weights only
//
// Caller precondition: topk_ids values are in range. Checking on device would
// cost a synchronize per call, so out-of-range ids (a -1 drop marker, or global
// ids against a local expert-parallel shard) index w1/w2 out of bounds instead.
void fused_moe_aiter(at::Tensor out, at::Tensor hidden_states, at::Tensor w1, at::Tensor w2,
                     at::Tensor topk_ids, at::Tensor topk_weights, int64_t block_m,
                     int64_t activation, std::optional<at::Tensor> w1_scale,
                     std::optional<at::Tensor> w2_scale) {
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(hidden_states.device());

  TORCH_CHECK(activation == kActivationSilu || activation == kActivationGelu,
              "fused_moe_aiter: activation must be silu (0) or gelu (1), got ", activation);
  TORCH_CHECK(is_supported_block_m(block_m), "fused_moe_aiter: block_m must be 32, 64 or 128, got ",
              block_m);

  const auto dtype = hidden_states.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
              "fused_moe_aiter: hidden_states must be bfloat16 or float16, got ", dtype);
  const bool quantized = is_fp8(w1.scalar_type());
  TORCH_CHECK(w1.scalar_type() == w2.scalar_type(),
              "fused_moe_aiter: w1 and w2 must have the same dtype, got ", w1.scalar_type(),
              " and ", w2.scalar_type());
  TORCH_CHECK(quantized || w1.scalar_type() == dtype,
              "fused_moe_aiter: w1/w2 must be fp8 or match hidden_states (", dtype, "), got ",
              w1.scalar_type());
  TORCH_CHECK(out.scalar_type() == dtype, "fused_moe_aiter: out dtype must match hidden_states (",
              dtype, "), got ", out.scalar_type());
#ifndef FLASHINFER_MOE_AITER_PER_TOKEN
  // Unreachable from Python, which picks the module from the weight dtype. Here
  // so the quantized path below needs no #else arm.
  TORCH_CHECK(!quantized, "fused_moe_aiter: this module was built without fp8 support");
#endif

  // CK gates *all* scaling on the scale pointers being non-null together, so a
  // half-supplied pair silently runs an unscaled GEMM rather than failing.
  TORCH_CHECK(w1_scale.has_value() == quantized && w2_scale.has_value() == quantized,
              "fused_moe_aiter: w1_scale and w2_scale must both be given for fp8 weights "
              "and both omitted otherwise; got w1_scale=",
              w1_scale.has_value(), " w2_scale=", w2_scale.has_value(), " for weight dtype ",
              w1.scalar_type());

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
  // topk==0 reaches a division by zero inside AITER and raises SIGFPE, which
  // kills the process rather than surfacing as an exception.
  TORCH_CHECK(topk >= 1, "fused_moe_aiter: topk must be at least 1, got ", topk);
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

  if (quantized) {
    check_weight_scale(*w1_scale, "w1_scale", num_experts, 2 * inter_dim, device);
    check_weight_scale(*w2_scale, "w2_scale", num_experts, model_dim, device);
    // Same reason as the weights: both are read after out has been zero-filled.
    TORCH_CHECK(!overlaps(out, *w1_scale) && !overlaps(out, *w2_scale),
                "fused_moe_aiter: out must not overlap w1_scale or w2_scale");
  }

  // An expert-parallel rank can legitimately be routed no tokens. AITER's
  // kernels launch a zero-sized grid for that and leave a sticky HIP error that
  // surfaces on some later, unrelated op.
  if (num_tokens == 0) {
    out.zero_();
    return;
  }

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
  const int quant_type = quantized ? kQuantPerToken : kQuantNone;

  at::Tensor stage1_in = hidden_states;
  std::optional<at::Tensor> a1_scale;
#ifdef FLASHINFER_MOE_AITER_PER_TOKEN
  if (quantized) {
    std::tie(stage1_in, a1_scale) =
        quantize_per_token(hidden_states, w1.scalar_type(), /*num_rows_factor=*/1);
  }
#endif

  ck_moe_stage1(stage1_in, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, inter_states,
                topk_i32, kernel_name, w1_scale, a1_scale, block_m_i32,
                /*sorted_weights=*/std::nullopt, quant_type, activation_i32, /*splitk=*/1,
                /*nt=*/false,
                /*dst_type=*/std::nullopt);

  at::Tensor stage2_in = inter_states;
  std::optional<at::Tensor> a2_scale;
#ifdef FLASHINFER_MOE_AITER_PER_TOKEN
  if (quantized) {
    // The [m, topk, inter_dim] shape is what gives each (token, expert) row its
    // own scale; num_rows_factor only applies alongside an explicit num_rows,
    // which expert parallelism would pass and this shim does not.
    std::tie(stage2_in, a2_scale) =
        quantize_per_token(inter_states, w2.scalar_type(), /*num_rows_factor=*/topk);
  }
#endif

  ck_moe_stage2(stage2_in, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out,
                topk_i32, kernel_name, w2_scale, a2_scale, block_m_i32, sorted_weights, quant_type,
                activation_i32, /*splitk=*/1,
                /*nt=*/false, /*dst_type=*/std::nullopt);
}
