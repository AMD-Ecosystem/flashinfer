// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pytorch_extension_utils.h"
#include "single_prefill_config.inc"

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  int64_t mask_mode_code, int64_t layout,
                                  int64_t window_left ADDITIONAL_FUNC_PARAMS);

// Named so flashinfer/rocm/torch_compile.py can mark the written arguments by name;
// the tail is generated with ADDITIONAL_FUNC_PARAMS. CONSERVATIVE: a named schema
// otherwise declares this void op side-effect free, and torch.jit.trace drops it.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(
      torch::schema("run(Tensor q, Tensor k, Tensor v, Tensor tmp, Tensor o, Tensor? maybe_lse, "
                    "int mask_mode_code, int layout, int window_left" ADDITIONAL_FUNC_PARAMS_SCHEMA
                    ") -> ()",
                    c10::AliasAnalysisKind::CONSERVATIVE),
      single_prefill_with_kv_cache);
}
