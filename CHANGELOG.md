# v0.5.3+amd.2

## Added

- Add Jupyter notebook tutorial for using amd-flashinfer on ROCm (#213) @diptorupd
- Add ROCm profiler module (`flashinfer/rocm_profiler/`) and FA2 single-prefill benchmark driver using rocprofv3 (#205) @diptorupd
- Gate `torch.compile` integration behind `FLASHINFER_USE_TORCH_CUSTOM_OPS`, with HIP pytest coverage (#210) @demandal25
- Support CUDA-graph capture on the AITER batch-decode path via an explicit `backend="aiter"`: capture at a maximum sequence length and replay for shorter sequences (the launch grid and `.so` variant are fixed at capture-time shapes; the kernel early-exits per sequence on `context_lens`). Previously `backend="aiter"` raised under `use_cuda_graph=True`. `backend="auto"` continues to use the in-tree `fa2` kernel under capture (fa2's graph path is capacity-based and capture-order-independent). Adds `benchmarks/rocm_benchmarks/bench_decode_graph_hip.py` and HIP graph-replay tests. @demandal25

## Changed

- FA2 prefill on HIP: improve occupancy and throughput on CDNA3 via LDS-aware `CTA_TILE_Q` selection and shared-memory budget capping (#209) @diptorupd
- Add `k128B_16Row` swizzle mode for prefill shared memory to reduce LDS bank conflicts on CDNA3 (#207) @diptorupd

## Fixed

- HIP FA2 on CDNA3: correct bfloat16 row-sum MFMA intrinsic selection and MFMA C/D row indexing for custom masks with GQA (#214) @subhajitdchow
- Editable installs: ensure JIT sees current headers under `include/` with scikit-build-core redirect mode (symlink at install + `get_include_paths` fix) (#208) @diptorupd

## Maintenance

- Refresh README @demandal25

---

**Contributors**: @diptorupd, @demandal25, @subhajitdchow

**Summary**: This point release focuses on ROCm prefill correctness and performance, developer tooling, and install ergonomics. It fixes silent bf16 softmax scaling and custom-mask/GQA mis-indexing in HIP FA2, improves CDNA3 occupancy and LDS bank behavior for FA2 prefill, adds an opt-in `torch.compile` path, a rocprofv3-based profiling kit, and a notebook tutorial. Editable installs now pick up header edits without reinstalling.

---

# v0.5.3+amd.1

## Updated Upstream

- Updated to upstream v0.5.3 tag of FlashInfer (#173) @diptorupd

## Added

- Port new upstream RoPE kernels to ROCm: fused RoPE + FP8 quantization and paged KV cache append (#196) @diptorupd
- Add test_hip_utils (#197) @rtmadduri
- Add AITER version check to FlashInfer and configure AITER during Docker build (#191) @rtmadduri
- Add AITER_ROCM_VERSION (#193) @rtmadduri
- Enable gfx950 support for FlashInfer+ROCm (#188) @rtmadduri
- Add prefill and decode aliases to sys.modules (#185) @diptorupd
- Support environments built by TheRock build system (#172) @eppaneamd
- Run tests in parallel with pytest-xdist and multi-GPU scheduling (#176) @diptorupd
- JIT core and env: IS_HIP block with cpp_ext_hip, JitSpecRegistry, and amd-jit-cache integration (#173) @diptorupd

## Changed

- Update decode/prefill_rocm to v0.5.3 API; expand HIP exports (#173) @diptorupd
- Re-HIPify sampling kernels (#173) @diptorupd
- Update and reorganize HIP tests (#173) @diptorupd
- amd-flashinfer-jit-cache: align version with v0.5.3 (#173) @diptorupd
- Fix README post v0.5.3 upgrade (#179) @diptorupd

## Fixed

- Fix AITER prefill graph-capture path + update page-size support (#170) @rtmadduri
- Add support for both CUDA and HIP generator headers in sampling.cu (#180) @eppaneamd
- Move HIP_VISIBLE_DEVICES setting to top-level conftests.py (#183) @diptorupd
- Fix HIP import errors blocking canary test on ROCm (pynvml lazy import, fp4 guard, modules_hip generate_additional_params, mypy) (#173) @diptorupd
- Fix flaky Sampling Tests (#189) @rtmadduri
- Reduce sampling threshold from 0.99 to 0.98 (#190) @diptorupd
- Improve test_logits_processor_hip tolerances (#192) @rtmadduri
- Fix arch check (#187) @diptorupd

## Maintenance

- Fixes devcontainer Dockerfile (#195) @diptorupd
- Update dev Dockerfile to install AITER (#194) @diptorupd
- Infra: improve dockerfile.rocm ci (#186) @diptorupd
- Fix linter issues identified by pre-commit (#184) @diptorupd
- chore: Update coverage include list (#182) @diptorupd
- Fix the coverage include list workflow (#178) @diptorupd

---

**Contributors**: @diptorupd, @rtmadduri, @eppaneamd

**Summary**: This release rebases amd-flashinfer onto the upstream v0.5.3 tag. It adds a full port of the fused RoPE + FP8 quantization and paged KV cache append kernels to ROCm, gfx950 (MI350) support, and AITER version checking with Docker build integration. JIT and env layers now use a dedicated IS_HIP path with cpp_ext_hip and amd-jit-cache. Fixes include AITER prefill graph-capture and page-size handling, HIP/CUDA generator header compatibility in sampling, and ROCm import/canary test fixes. Testing improvements include parallel pytest runs with multi-GPU scheduling, test_hip_utils, and TheRock build system support.

---

# v0.3.1+amd.1

## Updated Upstream

- Updated to upstream v0.3.1 tag of FlashInfer (#156) @diptorupd

## Added

- Add AITER backend support for FlashInfer SinglePrefill (#167) @rtmadduri
- Add AITER backend support for FlashInfer BatchPrefill (#161) @rtmadduri
- Port sampling module (OnlineSoftmax / SamplingFromLogits) to HIP (#102, #163) @Debasis Mandal, @diptorupd
- Port quantization module to ROCm/HIP (#145) @diptorupd
- Enable activation kernels on v0.3.1 API (#165) @diptorupd
- Add ROCm-specific logits_processor test case (#166) @diptorupd
- Add cuda graph support for paged batch prefill (#135, #138) @Debasis Mandal
- Add device_utils for HIP/CUDA device identification (#149) @diptorupd
- Initial infrastructure for AMD-specific code coverage automation (#128) @diptorupd

## Changed

- Port over upstream's latest AOT infrastructure to amd-flashinfer (#123) @diptorupd
- Isolate HIP kernels in dedicated csrc_rocm directory (#144) @diptorupd
- Remove CUDA sections from pytorch_hip (#143) @diptorupd
- Refactor sampling.cuh to unified CUDA/HIP header (#147) @diptorupd
- Refactor pytorch.py to increase coverage (#132) @Debasis Mandal
- Port HIP unit tests to v0.3.1 API @diptorupd
- Updates coverage script to identify unmodified but tested modules (#157) @diptorupd
- Update README with published docker images (#137) @Debasis Mandal

## Removed

- Remove cascade attention support (not supported on ROCm) (#129, #130) @Debasis Mandal

## Fixed

- Fix device contextmanager to use per-call context instead of setting default globally (#146) @diptorupd
- Fix minor issues in pytorch_hip.py @diptorupd

## Maintenance

- Tech debt reduction: remove superficial diffs and unused code (#152, #153) @diptorupd
- Update pre-commit hooks with AMD-specific configuration @diptorupd

---

**Contributors**: @diptorupd, @Debasis Mandal, @rtmadduri

**Summary**: This release rebases amd-flashinfer onto the upstream v0.3.1 tag and adds significant new functionality. Key additions include AITER backend support for both single and batch prefill, full sampling and quantization module ports to ROCm/HIP, and CUDA graph support for paged batch prefill. Infrastructure improvements include isolation of HIP kernels into a dedicated `csrc_rocm` directory, updated AOT build infrastructure, and initial AMD-specific code coverage tooling.

# v0.2.5+amd.2

## Added

- Make single prefill example script standalone (#104) @Debasis Mandal
- Run tests/test_non_contiguous_prefill.py on HIP (#96) @Debasis Mandal
- Run tests/test_activation.py on HIP (#98) @Debasis Mandal
- Add more PyTests from upstream to ROCm CI (#93) @Debasis Mandal
- Unify pytests for batch prefill for HIP and add to pyproject.toml to run on CI (#90) @Debasis Mandal
- Port over BatchPrefillWithPagedKVCacheDevice kernel to HIP (#63) @rtmadduri
- Batch prefill example script (#58) @Debasis Mandal
- Port over BatchPrefillWithRaggedKVCache to HIP (#50) @rtmadduri
- Add test_batch_prefill.cpp to HIP (#43) @rtmadduri
- PyTest script for single prefill (#48) @Debasis Mandal
- Add tests for LSE in single prefill example script (#46) @Debasis Mandal
- Enable FP8 support for Flashinfer ROCm decode kernels on CDNA3 (#40) @rtmadduri
- Single prefill example script (#36) @Debasis Mandal
- Support only gfx942 arch for ROCm (#45) @Debasis Mandal
- Enable JIT Installation for Prefill (#22) @rtmadduri
- Enable AOT Installation for Prefill (#21) @rtmadduri
- Adds a HIPified version of the SinglePrefillWithKVCacheDevice kernel (#31) @diptorupd
- Copy CUDA versions of prefill into attention/generic (#30) @diptorupd
- Port changes to vec_dtypes from prefill branch to amd-integration (#9) @diptorupd
- Add bit-width specific masks (#7) @diptorupd
- Feature/layout transformation function to transform from A matrix layout to B matrix layout for MFMA (#5) @diptorupd

## Changed

- Updated changelog @diptorupd
- Update README Wheel installation section (#112) @diptorupd
- Change pyproject.toml to generate revised whl name (#113) @rtmadduri
- Update README with clearer usage guide (#72) @diptorupd
- Change Swizzle mode from kLinear to k128B for prefill (#103) @Debasis Mandal
- Update logging for using JIT in absence of AOT (#94) @Debasis Mandal
- Improvements to Python packaging infrastructure (#76) @diptorupd
- pytest Configuration Improvements for ROCm/HIP Testing (#65) @diptorupd
- Refactor single prefill tests (#51) @Debasis Mandal
- Update CHANGELOG for v0.2.5+rocm.0.2 release (#34) @rtmadduri
- Decode feature chunking logic and shared mem optimization (#25) @rtmadduri
- Update hyperlinks in the TOC of README (#33) @Debasis Mandal
- Update build instructions for C++ tests in README.md (#27) @Debasis Mandal
- Incorporate gpu_iface changes needed for prefill (#14) @diptorupd
- Update CMake infra to run HIP CXX tests using top-level cmake (#10) @diptorupd
- Updates to permuted_smem.cuh (#8) @diptorupd

## Removed

- Removes leftover src and all tvm bindings (#99) @diptorupd
- Remove verbose CMake installation messages for editable JIT (#97) @Debasis Mandal
- Chore: Refactors the codebase to remove libflashinfer (#88) @diptorupd
- Remove xfail markers about HIP support from pytests (#92) @Debasis Mandal
- Chore: Reduce tech debt by removing CUDA sections from generic/prefill.cuh (#87) @diptorupd
- Removes the test_transpose_4x4_half_registers (#11) @diptorupd

## Fixed

- Add custom ROCm version scheme to fix wheels version naming (#110) @diptorupd
- Fix datatypes for HIP when using customized attention kernels (#111) @Debasis Mandal
- Fix partition-kv=True case and memory allocation issues in batch prefill (#89) @Debasis Mandal
- Fixes the single prefill kernel dispatch for HEAD_DIM_QK values gt. 64 (#86) @diptorupd
- Fix/threadblock_sync_mdo (#62) @diptorupd
- Fix batch prefill example script for ragged kv cache (#73) @Debasis Mandal
- Fixes to the single prefill dispatch for HIP devices (#64) @diptorupd
- Skip failing C++ tests and fix mma_debug_utils (#59) @diptorupd
- Fix Log-sum-exp (LSE) write back for single prefill kernels for CDNA3 (#42) @diptorupd
- Implemented fix for the write_o_reg_gmem kernel (#39) @diptorupd
- Fix few more leftover SPDX headers (#38) @diptorupd
- Fix SPDX headers for AMD authored files (#37) @diptorupd
- Improvements to the S-matrix (s_frag) materialization to LDS for debugging (#20) @diptorupd
- fix the pipe at the end of a table (#19) @demandal25
- Fix some compiler warnings in Cxx unit tests (#13) @diptorupd
- Adds debug utility functions for CDNA3 MMA ops. (#3) @diptorupd
- Fixes fragment loading to properly pack 16b values into a 32b register (#2) @diptorupd

## Maintenance

- Update rocm+torch versions in Dev Dockerfile (#118) @Debasis Mandal
- Update Dockerfile (#116) @rtmadduri
- Update Dockerfile.rocm_ci (#78) @diptorupd
- Fix Dockerfile.rocm_ci (#77) @diptorupd
- Removes hardcoded mamba env name from Docker image (#71) @diptorupd
- Fix/dockerfile rocm.ci (#70) @diptorupd
- Change base image to rocm/dev-ubuntu (#69) @diptorupd
- Revert tar changes from Dockerfile (#68) @rtmadduri
- Update ROCm CI Dockerfile (#67) @rtmadduri
- Update devcontainer dockerfile to rocm7.0.2 (#24) @Debasis Mandal
- Enhance devcontainer Dockerfile for ROCm (#28) @Debasis Mandal
- Update/devcontainer dockerfile (#23) @diptorupd
- Update Dockerfile.rocm_ci (#6) @Clint

## Miscellaneous

- Mark copied git repo as a safe directory (#114) @diptorupd
- Disable test_custom_allreduce (#108) @rtmadduri
- Ignore nfs related files in git tracking (#75) @Debasis Mandal
- Update .clangd to support HIP specific flags (#74) @Debasis Mandal
- Add clangd configuration (#32) @Debasis Mandal
- Lint README.md with suggestions from markdown linter (#17) @demandal25

---

**Contributors**: @diptorupd, @Debasis Mandal, @rtmadduri, @demandal25, @Clint

**Summary**: This release brings full prefill kernel support to ROCm, including single and batch prefill with paged and ragged KV cache. Major performance improvements include k128B swizzle mode and FP8 support. Significant infrastructure improvements include complete pytest coverage, improved build system, and updated to ROCm 7.1.1 / PyTorch 2.8.0.

# v0.2.5+rocm.1

## Added

- Decode feature chunking logic and shared mem optimization (#25) @Madduri, Rishi

# v0.2.5+rocm

The initial technical preview release of a ROCm port of FlashInfer. The release
has the port of the decode kernels and some infrastructure changes.

## Added

- A port of the decode kernels to HIP. (#38, #34) @Madduri, Rishi
- Add norm, page and rope to jit build infra (#46) @Madduri, Rishi
- Initial gpu interoperability interface for HIP/CUDA. (#22) @diptorupd
- Initial CDMA3 MFMA asbtractions (#62, #64, #68) @Madduri, Rishi
- Port build system to scikit-build-core. (#14) @diptorupd
