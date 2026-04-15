// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: Inline assembly primitives for AMD MI300X (gfx942 / CDNA3).
//
// Wrappers for ISA-level instructions that the HIP compiler either cannot
// emit or emits incorrectly without explicit guidance.  Every wrapper is
// __forceinline__ so that -mllvm -amdgpu-function-calls=false applies and
// no device function call overhead is introduced.

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__)
#include <cstdint>
#endif

#if defined(__HIPCC__) || defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_runtime.h>
#endif

namespace flashinfer {
namespace cdna3 {
namespace asm_primitives {

// ---------------------------------------------------------------------------
// Scheduler control
// ---------------------------------------------------------------------------

// Scheduling barrier: prevents the compiler back-end from reordering
// instructions across this point.  mask=0 fences all instruction classes.
// The mask must be a compile-time constant (hardware requirement).
template <uint32_t mask = 0>
__device__ __forceinline__ void sched_barrier() {
  __builtin_amdgcn_sched_barrier(mask);
}

// ---------------------------------------------------------------------------
// Synchronisation / wait counts
// ---------------------------------------------------------------------------

// Wait until outstanding VMEM (global memory) operations fall to n or fewer.
__device__ __forceinline__ void s_waitcnt_vmcnt(uint32_t n) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(n) : "memory");
}

// Wait until outstanding LDS (lgkm) operations fall to n or fewer.
__device__ __forceinline__ void s_waitcnt_lgkmcnt(uint32_t n) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(n) : "memory");
}

// Workgroup-level barrier (equivalent to __syncthreads on CUDA).
__device__ __forceinline__ void s_barrier() { __syncthreads(); }

// ---------------------------------------------------------------------------
// MFMA wrapper: v_mfma_f32_32x32x8f16 (native gfx942 instruction)
//
// Computes: C[16 floats] += A[4 halves] * B[4 halves]  (32x32x8 tile)
// All 64 threads collectively compute a 32x32 FP32 output tile.
//
// Per-thread register layout (gfx942):
//   A-operand: 4 fp16 (2 x uint32). Thread t: row = t%32, k_base = (t/32)*4
//   B-operand: 4 fp16 (2 x uint32). Thread t: col = t%32, k_base = (t/32)*4
//   C-output:  16 fp32. Thread t: col = t%32,
//              rows = {(t/32)*16+0..3, (t/32)*16+4..7,
//                      (t/32)*16+8..11, (t/32)*16+12..15}
// ---------------------------------------------------------------------------

using f16x4_t = _Float16 __attribute__((ext_vector_type(4)));
using f32x4_t = float __attribute__((ext_vector_type(4)));
using f32x16_t = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ f32x16_t mfma_f32_32x32x8_f16(f16x4_t a, f16x4_t b, f32x16_t c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx90a__))
  return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
#else
  return c;
#endif
}

// Vector-typed overload: takes f32x16_t& directly (preferred for correct VGPR allocation).
__device__ __forceinline__ void mfma_f32_32x32x8_f16_vec(f32x16_t& c, const uint32_t* a,
                                                         const uint32_t* b) {
  f16x4_t av = *reinterpret_cast<const f16x4_t*>(a);
  f16x4_t bv = *reinterpret_cast<const f16x4_t*>(b);
  c = mfma_f32_32x32x8_f16(av, bv, c);
}

// ---------------------------------------------------------------------------
// sched_group_barrier: fine-grained instruction scheduling hint
//
// Tells the hardware scheduler to allow exactly `count` instructions of type
// `mask` to issue before this barrier retires.  Used to interleave DS reads
// with MFMA instructions for optimal pipeline utilization.
//
// Common masks:
//   0x008 = MFMA instructions
//   0x100 = DS_READ (LDS read) instructions
//   0x200 = DS_WRITE (LDS write) instructions
//   0x020 = VMEM_READ (global load) instructions
// ---------------------------------------------------------------------------

static constexpr uint32_t kSchedMFMA = 0x008;
static constexpr uint32_t kSchedDSRead = 0x100;
static constexpr uint32_t kSchedDSWrite = 0x200;
static constexpr uint32_t kSchedVMEMRead = 0x020;

template <uint32_t mask, uint32_t count>
__device__ __forceinline__ void sched_group_barrier() {
  __builtin_amdgcn_sched_group_barrier(mask, count, 0);
}

// ---------------------------------------------------------------------------
// MFMA 32x32 output register-to-row mapping (gfx942).
//
// For v_mfma_f32_32x32x8f16, register (block, v) maps to row:
//   row[4:0] = {v[2], block, v[3], v[1], v[0]}
// ---------------------------------------------------------------------------

__device__ __forceinline__ int mfma_32x32_row(int block, int v) {
  return ((v >> 2) << 3) | (block << 2) | (v & 3);
}

// ---------------------------------------------------------------------------
// Fast exp2 approximation (replaces 1/4-rate v_exp_f32 with full-rate VALU)
//
// Degree-3 polynomial over [-0.5, 0.5] with integer exponent decomposition.
// Error: ~0.5 ULP max, negligible for attention score exponentiation.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fast_exp2(float x) {
  if (x < -126.f) return 0.f;
  if (x > 128.f) return 1.0f / 0.0f;  // +inf
  float n = floorf(x + 0.5f);
  float f = x - n;
  int ni = static_cast<int>(n);
  float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * 0.0555041f));
  int32_t pi;
  __builtin_memcpy(&pi, &p, sizeof(pi));
  pi += (ni << 23);
  float result;
  __builtin_memcpy(&result, &pi, sizeof(result));
  return result;
}

}  // namespace asm_primitives
}  // namespace cdna3
}  // namespace flashinfer
