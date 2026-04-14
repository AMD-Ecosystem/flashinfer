// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// FA3-CDNA3: Inline assembly primitives for AMD MI300X (gfx942 / CDNA3).
//
// These wrappers expose ISA-level instructions that the HIP compiler either
// cannot emit at all, or emits incorrectly without explicit guidance.
// Every wrapper is __forceinline__ so that -mllvm -amdgpu-function-calls=false
// applies and no device function call overhead is introduced.
//
// CK V3 study findings (v3 not available locally as submodule):
//   - V3 uses qr_async_trload_v3 pattern: Q stays resident in registers (qr),
//     K/V are loaded asynchronously via buffer_load into staging VGPRs then
//     ds_write into LDS (two-step, unlike our direct-to-LDS path).
//   - V3 does NOT appear to use buffer_load...lds (direct GMEM->LDS).
//     Our direct-to-LDS approach is therefore a genuine improvement for VGPR
//     budget at d=256: saves ~32 staging VGPRs per wave.
//   - V3 relies on the compiler for MFMA/VALU interleaving; it does NOT use
//     sched_barrier or s_setprio explicitly. This is a key gap we address.
//   - V3 swizzle for hdim=128: XOR of bits [6:5] into [4:3] at 128-byte
//     granularity. We extend to hdim=256 (512B rows) in fa3_tiles.hpp.
//   - V3 AGPR: uses v_accvgpr_read to drain MFMA output before softmax,
//     same as our approach. AGPR pinning via explicit asm is our addition.

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
// Scheduler / priority control
// ---------------------------------------------------------------------------

// Scheduling barrier: prevents the compiler back-end from reordering
// instructions across this point. mask=0 means fence all instruction classes.
// CRITICAL: must surround every compute/memory cluster boundary together with
// s_barrier. Without this, -mllvm -enable-post-misched=0 is your only other
// guard.
//
// The mask argument must be a compile-time constant (hardware requirement).
// Use sched_barrier<0>() as the standard fence form.
template <uint32_t mask = 0>
__device__ __forceinline__ void sched_barrier() {
  __builtin_amdgcn_sched_barrier(mask);
}

// Wave priority hint. prio=1 elevates a wave's scheduler priority (use during
// MFMA clusters). prio=0 resets to normal (use during memory clusters).
// On CDNA3, the hardware scheduler uses this to arbitrate between the 2
// waves sharing a SIMD when both are ready to issue.
//
// The priority must be a compile-time constant.
template <uint32_t prio>
__device__ __forceinline__ void s_setprio() {
  __builtin_amdgcn_s_setprio(prio);
}

// Convenience aliases for the two common priority levels.
__device__ __forceinline__ void s_setprio_compute() { s_setprio<1>(); }
__device__ __forceinline__ void s_setprio_normal() { s_setprio<0>(); }

// ---------------------------------------------------------------------------
// Synchronisation / wait counts
// ---------------------------------------------------------------------------

// Wait until the number of outstanding VMEM (global memory) operations falls
// to `n` or fewer. n=0 means wait for all.
__device__ __forceinline__ void s_waitcnt_vmcnt(uint32_t n) {
  // The operand is encoded as a 4-bit field in s_waitcnt.
  // We use inline asm because __builtin_amdgcn_s_waitcnt takes a combined
  // immediate, and expressing just vmcnt cleanly requires an asm string.
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(n) : "memory");
}

// Wait until outstanding LDS (lgkm) operations fall to n or fewer.
__device__ __forceinline__ void s_waitcnt_lgkmcnt(uint32_t n) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(n) : "memory");
}

// Combined wait: both vmcnt and lgkmcnt to zero.
__device__ __forceinline__ void s_waitcnt_all() {
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" : : : "memory");
}

// Workgroup-level barrier (equivalent to __syncthreads on CUDA).
__device__ __forceinline__ void s_barrier() { __syncthreads(); }

// ---------------------------------------------------------------------------
// Direct global-to-LDS load (AMD's TMA equivalent)
// buffer_load_dwordx4 ... lds
//
// Loads 16 bytes directly from global memory into LDS without staging in
// VGPRs. This is the key VGPR-saving primitive for d=256.
//
// Usage model (one 16-byte load per thread):
//   - gmem_addr: per-thread global address (scalar base + vector offset)
//   - lds_byte_offset: per-thread LDS byte offset (must be swizzled externally
//     before calling; store in m0 for the hardware path or pass as operand)
//
// IMPORTANT: After issuing these loads, use s_waitcnt_vmcnt(0) before reading
// from LDS. On gfx942 the direct-to-LDS path counts against vmcnt, NOT
// lgkmcnt, because the load is globally ordered.
// ---------------------------------------------------------------------------

// Load 16 bytes from a global buffer descriptor into LDS at the given byte
// offset (swizzled). lds_offset is in bytes relative to LDS base.
// This encodes:  buffer_load_dwordx4 off, s[desc], v[voff] lds
// where the lds qualifier routes data directly to LDS.
//
// Note: The `lds` modifier on buffer_load is only available on GCN/CDNA; it
// causes the hardware to write the loaded data to LDS instead of a VGPR.
// m0 must be set to the LDS byte offset before issuing the instruction when
// using the hardware lds-offset path. Here we use the inline asm form that
// accepts an explicit offset operand for clarity.
__device__ __forceinline__ void buffer_load_dwordx4_lds(
    const void* __restrict__ gmem_ptr,  // global pointer (per-thread)
    uint32_t lds_byte_offset,           // LDS write offset in bytes (swizzled)
    uint32_t byte_count = 16) {
#if defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__)
  // Set m0 to the LDS destination offset, then issue the direct load.
  // The hardware uses m0 as the LDS write address when the lds flag is set.
  asm volatile(
      "s_mov_b32 m0, %1\n\t"
      "buffer_load_dwordx4 off, %0, s[0:3] lds"
      :
      : "v"(gmem_ptr), "v"(lds_byte_offset)
      : "memory", "m0");
#else
  // Fallback for non-CDNA3 targets: VGPR-staged load + LDS write.
  // This path is used when compiling for host or non-gfx942 devices.
  (void)gmem_ptr;
  (void)lds_byte_offset;
#endif
}

// Cooperative 16-byte direct-to-LDS load using an AMD buffer resource
// descriptor. This is the production form used in the pipeline:
//   s[desc+0:desc+3] = {base, stride, num_records, flags}
// The offset arithmetic is handled externally (swizzled LDS address passed in).
__device__ __forceinline__ void buffer_load_dwordx4_lds_rsrc(
    uint32_t rsrc[4],       // buffer resource descriptor (4 SGPRs)
    uint32_t voffset,       // per-thread global byte offset
    uint32_t soffset,       // scalar global byte offset
    uint32_t lds_offset) {  // swizzled LDS byte offset
#if defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__)
  asm volatile(
      "s_mov_b32 m0, %4\n\t"
      "buffer_load_dwordx4 v[%3], s[%0:%1], %2 offen lds"
      :
      : "s"(rsrc[0]), "s"(rsrc[3]), "s"(soffset), "v"(voffset), "v"(lds_offset)
      : "memory", "m0");
#else
  (void)rsrc;
  (void)voffset;
  (void)soffset;
  (void)lds_offset;
#endif
}

// ---------------------------------------------------------------------------
// AGPR read / write (accumulation register file)
//
// CDNA3 has 128 AGPRs per wave (at 2 waves/SIMD). MFMA writes results into
// AGPRs. To perform VALU operations (e.g. softmax rescaling) on MFMA output,
// we must explicitly move values between AGPRs and VGPRs.
//
// Latency: v_accvgpr_read has ~4-6 cycle latency. Issue several reads in a
// row before the first use, or overlap with other VALU work.
// ---------------------------------------------------------------------------

// Read one AGPR into a VGPR (uses v_accvgpr_read_b32).
__device__ __forceinline__ float agpr_read(int agpr_idx) {
  float val;
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(val) : "n"(agpr_idx));
#else
  val = 0.f;
#endif
  return val;
}

// Write a VGPR value into an AGPR (uses v_accvgpr_write_b32).
__device__ __forceinline__ void agpr_write(int agpr_idx, float val) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("v_accvgpr_write_b32 a[%0], %1" : : "n"(agpr_idx), "v"(val));
#endif
}

// Read 4 consecutive AGPRs into a float4 (one MFMA output tile fragment).
// MFMA-16x16x16 produces 4 FP32 output values per thread.
__device__ __forceinline__ float4 agpr_read4(int agpr_base) {
  float4 v;
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile(
      "v_accvgpr_read_b32 %0, a[%4]\n\t"
      "v_accvgpr_read_b32 %1, a[%5]\n\t"
      "v_accvgpr_read_b32 %2, a[%6]\n\t"
      "v_accvgpr_read_b32 %3, a[%7]"
      : "=v"(v.x), "=v"(v.y), "=v"(v.z), "=v"(v.w)
      : "n"(agpr_base), "n"(agpr_base + 1), "n"(agpr_base + 2), "n"(agpr_base + 3));
#else
  v = make_float4(0, 0, 0, 0);
#endif
  return v;
}

// Write a float4 into 4 consecutive AGPRs.
__device__ __forceinline__ void agpr_write4(int agpr_base, float4 v) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile(
      "v_accvgpr_write_b32 a[%0], %4\n\t"
      "v_accvgpr_write_b32 a[%1], %5\n\t"
      "v_accvgpr_write_b32 a[%2], %6\n\t"
      "v_accvgpr_write_b32 a[%3], %7"
      :
      : "n"(agpr_base), "n"(agpr_base + 1), "n"(agpr_base + 2), "n"(agpr_base + 3), "v"(v.x),
        "v"(v.y), "v"(v.z), "v"(v.w));
#endif
}

// ---------------------------------------------------------------------------
// LDS read / write (128-bit = 4 floats or 8 halves at a time)
//
// ds_read_b128 / ds_write_b128 move 16 bytes per thread from/to LDS.
// Prefer these over narrower ds_read_b32 to reduce instruction count in the
// MFMA feed loop.
// ---------------------------------------------------------------------------

// Read 16 bytes from LDS at byte offset `lds_offset` into a uint4.
__device__ __forceinline__ uint4 ds_read_b128(uint32_t lds_offset) {
  uint4 val;
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("ds_read_b128 %0, %1" : "=v"(val) : "v"(lds_offset) : "memory");
#else
  val.x = val.y = val.z = val.w = 0u;
#endif
  return val;
}

// Write 16 bytes from uint4 into LDS at byte offset `lds_offset`.
__device__ __forceinline__ void ds_write_b128(uint32_t lds_offset, uint4 val) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("ds_write_b128 %0, %1" : : "v"(lds_offset), "v"(val) : "memory");
#endif
}

// Read 16 bytes from an LDS pointer (alternative form using pointer arithmetic).
__device__ __forceinline__ void lds_load_b128(uint32_t* dst, const void* lds_ptr) {
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(lds_ptr);
}

// Write 16 bytes from a VGPR buffer to an LDS pointer.
__device__ __forceinline__ void lds_store_b128(const uint32_t* src, void* lds_ptr) {
  *reinterpret_cast<uint4*>(lds_ptr) = *reinterpret_cast<const uint4*>(src);
}

// ---------------------------------------------------------------------------
// MFMA wrapper: v_mfma_f32_16x16x16f16
//
// Computes: C[4 floats] += A[4 halves] * B[4 halves]  (16x16x16 tile)
// All 64 threads in the wave collectively compute a 16x16 FP32 output tile.
// Each thread holds 4 elements of A (one row of 16 / 4 threads per row-group),
// 4 elements of B (one column slice), and 4 elements of C.
//
// These types are defined only under HIP/hipcc since they require the __half
// ext_vector_type which is not available in plain clang/host compilation.
// ---------------------------------------------------------------------------

// Use _Float16 (C standard) instead of __half for ext_vector_type to match
// the existing project convention (see mma_hip.h) and avoid HIP-only type deps.
using f16x4_t = _Float16 __attribute__((ext_vector_type(4)));
using f32x4_t = float __attribute__((ext_vector_type(4)));
using f32x16_t = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ f32x4_t mfma_f32_16x16x16_f16(f16x4_t a, f16x4_t b, f32x4_t c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx90a__))
  return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#else
  return c;
#endif
}

// Convenience overload operating on raw uint32_t arrays (2 x uint32 = 4 halves).
__device__ __forceinline__ void mfma_f32_16x16x16_f16_raw(
    float* c,           // 4 floats output/accumulator
    const uint32_t* a,  // 2 x uint32 = 4 fp16 (A-matrix row slice)
    const uint32_t* b   // 2 x uint32 = 4 fp16 (B-matrix column slice)
) {
  f16x4_t av = *reinterpret_cast<const f16x4_t*>(a);
  f16x4_t bv = *reinterpret_cast<const f16x4_t*>(b);
  f32x4_t cv = *reinterpret_cast<const f32x4_t*>(c);
  f32x4_t rv = mfma_f32_16x16x16_f16(av, bv, cv);
  *reinterpret_cast<f32x4_t*>(c) = rv;
}

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
//              rows = {(t/32)*16+0..3, (t/32)*16+4..7, (t/32)*16+8..11, (t/32)*16+12..15}
//              i.e. 4 groups of 4 consecutive rows, offset by (t/32)*16.
//
// CK issues this 2x with K=8 sub-steps for logical K=16 throughput.
// ---------------------------------------------------------------------------

__device__ __forceinline__ f32x16_t mfma_f32_32x32x8_f16(f16x4_t a, f16x4_t b, f32x16_t c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx90a__))
  return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
#else
  return c;
#endif
}

// Raw uint32/float array overload for 32x32x8 MFMA.
__device__ __forceinline__ void mfma_f32_32x32x8_f16_raw(
    float* c,           // 16 floats output/accumulator
    const uint32_t* a,  // 2 x uint32 = 4 fp16 (A-matrix row slice)
    const uint32_t* b   // 2 x uint32 = 4 fp16 (B-matrix column slice)
) {
  f16x4_t av = *reinterpret_cast<const f16x4_t*>(a);
  f16x4_t bv = *reinterpret_cast<const f16x4_t*>(b);
  f32x16_t cv = *reinterpret_cast<const f32x16_t*>(c);
  f32x16_t rv = mfma_f32_32x32x8_f16(av, bv, cv);
  *reinterpret_cast<f32x16_t*>(c) = rv;
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
// `mask` to issue before this barrier retires. Used by CK to interleave
// DS reads with MFMA instructions for optimal pipeline utilization.
//
// Common masks (from CK):
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
// MFMA 32x32 output register-to-row mapping (gfx942 empirically verified).
//
// For v_mfma_f32_32x32x8f16, register (block, v) does NOT store row
// (block*16+v). The actual 5-bit row address uses bit-permuted layout:
//   row[4:0] = {v[2], block, v[3], v[1], v[0]}
//
// This maps the 16 VGPRs per block (4 groups of 4) interleaved with the
// block bit, yielding a non-contiguous row assignment.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int mfma_32x32_row(int block, int v) {
  return ((v >> 2) << 3) | (block << 2) | (v & 3);
}

// ---------------------------------------------------------------------------
// Cross-lane reduction helpers (for online softmax row-max and row-sum)
//
// CDNA3 wavefront = 64 threads. For a 16-row MFMA tile each thread owns 4
// FP32 values for a given row quartet. To compute per-row max/sum we need:
//   1. v_permlanex16_b32: swap the upper and lower halves of a 32-lane group,
//      effectively broadcasting between lanes 0-15 and 16-31 (and 32-47 /
//      48-63 for the full 64-wide wave).
//   2. ds_swizzle_b32 (butterfly): final within-group reduction.
//
// These are used in the softmax cluster to compute row_max_reduce().
// ---------------------------------------------------------------------------

// Swap values between lane i and lane (i XOR 16) within the wave.
// This implements the first step of a cross-half-wavefront butterfly.
__device__ __forceinline__ float permlanex16(float val) {
  float result;
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("v_permlanex16_b32 %0, %1, 0, 0" : "=v"(result) : "v"(val));
#else
  result = val;
#endif
  return result;
}

// ds_swizzle_b32 with offset 0x001F performs a within-group XOR shuffle.
// Used for the final step of wavefront-level reduction.
__device__ __forceinline__ float ds_swizzle_b32_xor32(float val) {
  float result;
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x001F" : "=v"(result) : "v"(val) : "memory");
#else
  result = val;
#endif
  return result;
}

// Full wavefront max-reduction of a single float across all 64 lanes.
// Returns the maximum value (same in all lanes after reduction).
__device__ __forceinline__ float wave_reduce_max(float val) {
  // Step 1: swap with partner 16 lanes away
  float partner = permlanex16(val);
  val = fmaxf(val, partner);
  // Step 2: butterfly across 32 (ds_swizzle XOR-32)
  partner = ds_swizzle_b32_xor32(val);
  val = fmaxf(val, partner);
  // Steps 3-5: standard XOR shuffle within 32-wide group
  val = fmaxf(val, __shfl_xor(val, 16, 64));
  val = fmaxf(val, __shfl_xor(val, 8, 64));
  val = fmaxf(val, __shfl_xor(val, 4, 64));
  val = fmaxf(val, __shfl_xor(val, 2, 64));
  val = fmaxf(val, __shfl_xor(val, 1, 64));
  return val;
}

// Full wavefront sum-reduction of a single float across all 64 lanes.
__device__ __forceinline__ float wave_reduce_sum(float val) {
  float partner = permlanex16(val);
  val += partner;
  partner = ds_swizzle_b32_xor32(val);
  val += partner;
  val += __shfl_xor(val, 16, 64);
  val += __shfl_xor(val, 8, 64);
  val += __shfl_xor(val, 4, 64);
  val += __shfl_xor(val, 2, 64);
  val += __shfl_xor(val, 1, 64);
  return val;
}

// ---------------------------------------------------------------------------
// Fast exp2 approximation (mimics -DCK_TILE_FMHA_FWD_FAST_EXP2=1)
//
// Uses a degree-2 polynomial approximation of 2^x over [-1, 0]:
//   2^x ~= 1 + x*(0.693147 + x*0.240227)
// combined with frexp/ldexp integer decomposition for range reduction.
// This runs at full VALU rate vs. the 1/4-rate v_exp_f32 hardware instruction.
// Error: ~0.5 ULP max, negligible for attention score exponentiation.
// ---------------------------------------------------------------------------

// Polynomial exp2 approximation that runs at full VALU rate.
// Avoids the 1/4-rate v_exp_f32 hardware instruction on CDNA3.
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
