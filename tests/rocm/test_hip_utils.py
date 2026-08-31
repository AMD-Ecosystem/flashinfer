# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest tests for flashinfer/hip_utils.py

Covers every public function using unittest.mock so that no real ROCm
installation, GPU hardware, or external tools are required.
"""

import contextlib
import subprocess
import warnings
from unittest.mock import MagicMock, patch

import pytest

from flashinfer.rocm import hip_utils
from flashinfer.rocm.hip_utils import (
    FLASHINFER_SUPPORTED_ROCM_ARCHS,
    check_torch_rocm_compatibility,
    get_available_gpu_count,
    get_rocm_home,
    get_supported_device_indices,
    get_system_rocm_version_from_hipconfig,
    is_therock_build,
    resolve_target_archs,
    validate_flashinfer_rocm_arch,
    validate_rocm_arch,
)


@pytest.fixture(autouse=True)
def _clear_arch_detection_cache():
    """Reset the process-cached architecture probe around every test in this file.

    ``_detected_supported_archs`` caches rocminfo's answer for the process, so a
    test that patches ``rocminfo_gpu_agents`` can otherwise be shadowed by
    whatever a previous test -- or the real hardware, via package import -- put
    there first, and silently assert against the wrong architecture. Module-wide
    rather than on the one class that resolves: several tests here reach the
    probe indirectly through ``validate_rocm_arch(arch_list=None)``, and any test
    added later that patches it inherits the same hazard.

    Clearing on the way out as well keeps a fake from leaking into the rest of
    the session.
    """
    hip_utils._detected_supported_archs.cache_clear()
    yield
    hip_utils._detected_supported_archs.cache_clear()


# get_rocm_home
class TestGetRocmHome:
    def test_rocm_path_env_var(self, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", "/custom/rocm")
        monkeypatch.delenv("ROCM_HOME", raising=False)
        assert get_rocm_home() == "/custom/rocm"

    def test_rocm_home_env_var_fallback(self, monkeypatch):
        monkeypatch.delenv("ROCM_PATH", raising=False)
        monkeypatch.setenv("ROCM_HOME", "/home/rocm")
        assert get_rocm_home() == "/home/rocm"

    def test_rocm_path_takes_priority_over_rocm_home(self, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", "/path/rocm")
        monkeypatch.setenv("ROCM_HOME", "/home/rocm")
        assert get_rocm_home() == "/path/rocm"

    def test_default_path_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("ROCM_PATH", raising=False)
        monkeypatch.delenv("ROCM_HOME", raising=False)
        assert get_rocm_home() == "/opt/rocm"


# is_therock_build
class TestIsTheRockBuild:
    def test_returns_true_when_rocm_sdk_has_version(self):
        rocm_sdk_mock = MagicMock()
        rocm_sdk_mock.__version__ = "7.1.0"
        with patch.dict("sys.modules", {"rocm_sdk": rocm_sdk_mock}):
            assert is_therock_build() is True

    def test_falls_through_when_rocm_sdk_has_no_version_attr(self, tmp_path):
        rocm_sdk_mock = MagicMock(spec=[])  # no __version__
        manifest = tmp_path / "share" / "therock" / "therock_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.touch()
        with (
            patch.dict("sys.modules", {"rocm_sdk": rocm_sdk_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_rocm_home", return_value=str(tmp_path)
            ),
        ):
            assert is_therock_build() is True

    def test_falls_through_when_rocm_sdk_version_is_empty(self, tmp_path):
        rocm_sdk_mock = MagicMock()
        rocm_sdk_mock.__version__ = ""
        with (
            patch.dict("sys.modules", {"rocm_sdk": rocm_sdk_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_rocm_home", return_value=str(tmp_path)
            ),
        ):
            # no manifest → False
            assert is_therock_build() is False

    def test_manifest_file_exists(self, tmp_path):
        manifest = tmp_path / "share" / "therock" / "therock_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.touch()
        with patch.dict("sys.modules", {"rocm_sdk": None}):
            # Simulate ImportError by removing the key
            import sys

            sys.modules.pop("rocm_sdk", None)
            with patch(
                "flashinfer.rocm.hip_utils.get_rocm_home", return_value=str(tmp_path)
            ):
                assert is_therock_build() is True

    def test_manifest_file_missing_and_no_rocm_sdk(self, tmp_path):
        with (
            patch.dict("sys.modules", {"rocm_sdk": None}),
            patch(
                "flashinfer.rocm.hip_utils.get_rocm_home", return_value=str(tmp_path)
            ),
        ):
            assert is_therock_build() is False


# get_system_rocm_version_from_hipconfig
class TestGetSystemRocmVersionFromHipconfig:
    def _run_result(self, stdout, returncode=0):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        return result

    @pytest.mark.parametrize(
        "stdout,expected",
        [
            ("7.1.0\n", "7.1.0"),
            ("7.13.26183-83e9908b71\n", "7.13.26183"),
            ("7.13\n", "7.13"),
        ],
    )
    def test_parses_version_string(self, stdout, expected):
        with patch("subprocess.run", return_value=self._run_result(stdout)):
            assert get_system_rocm_version_from_hipconfig() == expected

    def test_returns_none_on_nonzero_returncode(self):
        with patch("subprocess.run", return_value=self._run_result("", returncode=1)):
            assert get_system_rocm_version_from_hipconfig() is None

    def test_returns_none_when_hipconfig_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert get_system_rocm_version_from_hipconfig() is None

    def test_returns_none_on_timeout(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("hipconfig", 5)
        ):
            assert get_system_rocm_version_from_hipconfig() is None


# resolve_target_archs
class TestResolveTargetArchs:
    """The single resolver every build path now consults.

    Before it existed, three call sites answered "what are we building for"
    independently and could disagree: on a gfx950 host,
    ``validate_flashinfer_rocm_arch(arch_list=None)`` returned ``{"gfx942"}``
    while ``CompilationContext`` emitted ``--offload-arch=gfx950``.
    """

    def _agents(self, *archs):
        return patch(
            "flashinfer.rocm.hip_utils.rocminfo_gpu_agents",
            return_value=tuple((arch, "") for arch in archs),
        )

    def test_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950")
        with self._agents("gfx950"):
            assert resolve_target_archs("gfx942") == "gfx942"

    def test_env_var_beats_detection(self, monkeypatch):
        """An explicit request is honoured even when it is not what is plugged in
        -- cross-compiling for the other architecture must stay possible."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
        with self._agents("gfx950"):
            assert resolve_target_archs() == "gfx942"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Qualifiers: what torch's gcnArchName looks like, so an operator
            # copying from `rocminfo` or a torch error message pastes this shape.
            ("gfx950:sramecc+:xnack-", "gfx950"),
            # ';' is documented for this same variable by jit/aiter_source.py.
            # The two consumers must not disagree about their own env var.
            ("gfx942;gfx950", "gfx942,gfx950"),
            ("gfx942; gfx950", "gfx942,gfx950"),
            # Empty tokens would otherwise reach the validators as "" and be
            # reported as an unsupported architecture.
            ("gfx942,,gfx950", "gfx942,gfx950"),
            ("gfx942, gfx950 ", "gfx942,gfx950"),
            # Duplicates collapse; first occurrence sets the order, which is what
            # lands on the hipcc command line.
            ("gfx950,gfx942,gfx950", "gfx950,gfx942"),
            # Already canonical input must round-trip untouched.
            ("gfx942,gfx950", "gfx942,gfx950"),
        ],
    )
    def test_env_var_is_canonicalized(self, monkeypatch, raw, expected):
        """The validators split on ',' only and match tokens verbatim, so an
        unnormalized value is a hard failure, not an untidiness: "gfx942;gfx950"
        arrives as one token, matches nothing, and validate_flashinfer_rocm_arch
        raises "does not support any of the requested ROCm architectures"."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", raw)
        with self._agents("gfx950"):
            assert resolve_target_archs() == expected

    def test_explicit_argument_is_canonicalized(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._agents("gfx950"):
            assert resolve_target_archs("gfx942:xnack-;gfx950") == "gfx942,gfx950"

    def test_unknown_archs_survive_normalization(self, monkeypatch):
        """Only syntax is normalized. Dropping an unrecognized arch here would
        turn the validators' clear error into a build that quietly targets less
        than was asked for."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx900:xnack-;gfx942")
        with self._agents("gfx950"):
            assert resolve_target_archs() == "gfx900,gfx942"

    @pytest.mark.parametrize("raw", [";;", "  ", ",", " ; , "])
    def test_a_value_that_normalizes_away_falls_through(self, monkeypatch, raw):
        """Returning "" would reach the validators as a single empty token and
        fail as an unsupported architecture; detection is the better answer."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", raw)
        with self._agents("gfx950"):
            assert resolve_target_archs() == "gfx950"

    def test_detects_the_running_device(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._agents("gfx950"):
            assert resolve_target_archs() == "gfx950"

    def test_detects_every_distinct_supported_arch(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._agents("gfx942", "gfx950", "gfx942"):
            assert resolve_target_archs() == "gfx942,gfx950"

    def test_ignores_unsupported_agents(self, monkeypatch):
        """An integrated GPU alongside the dGPU must not widen the target set."""
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._agents("gfx1035", "gfx950"):
            assert resolve_target_archs() == "gfx950"

    def test_no_device_keeps_the_existing_default_and_warns(self, monkeypatch, caplog):
        """A GPU-less host keeps today's answer, loudly.

        Widening this to every supported architecture was tried and reverted: it
        desynchronizes the AITER shim, which takes exactly one architecture and
        picks ``env_archs[0]`` when no device is visible, so a fat list ships
        gfx950 kernels beside a gfx942-only shim. Making the GPU-less default
        *correct* rather than merely consistent is a separate change.
        """
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._agents(), caplog.at_level("WARNING"):
            assert resolve_target_archs() == "gfx942"
        assert "FLASHINFER_ROCM_ARCH_LIST" in caplog.text

    def test_gpuless_fallback_matches_the_aiter_shim_default(self):
        """The two are resolved independently on a GPU-less host; if they ever
        diverge, the shim is built for a different architecture than the kernels
        it ships beside and faults at run time."""
        from flashinfer.jit.rocm.aiter_source import _DEFAULT_BUILD_ARCH

        assert hip_utils._GPULESS_FALLBACK_ARCH == _DEFAULT_BUILD_ARCH

    def test_detection_is_cached_per_process(self):
        """Restores a cache the refactor dropped: the previous implementation
        reached rocminfo through the cached ``get_supported_device_indices``,
        while ``rocminfo_gpu_agents`` is uncached by design."""
        with patch(
            "flashinfer.rocm.hip_utils.rocminfo_gpu_agents",
            return_value=(("gfx950", ""),),
        ) as probe:
            hip_utils._detected_supported_archs()
            hip_utils._detected_supported_archs()
        assert probe.call_count == 1

    def test_agrees_with_the_compilation_context(self, monkeypatch):
        """The property the whole change exists for: the validator and the thing
        that emits --offload-arch must not disagree."""
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        import torch.utils.cpp_extension as torch_cpp_ext

        from flashinfer.rocm.compilation_context import CompilationContext

        # Both sides must see the same PyTorch. CompilationContext imports the
        # real torch.utils.cpp_extension and validates against it, while the
        # direct call below is handed _FakeCppExt -- so without this patch the
        # assertion depends on the installed wheel. The wheel here is a fat build
        # advertising gfx950, which is why that went unnoticed; an arch-specific
        # build (gfx942-only) would make CompilationContext() raise "PyTorch does
        # not support the following architectures" and fail this test for a
        # reason that has nothing to do with resolver agreement.
        with (
            self._agents("gfx950"),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version",
                return_value="7.1.0",
            ),
            patch.object(
                torch_cpp_ext,
                "_get_rocm_arch_flags",
                _FakeCppExt._get_rocm_arch_flags,
            ),
        ):
            _, validated = validate_flashinfer_rocm_arch(
                arch_list=None, torch_cpp_ext_module=_FakeCppExt(), verbose=False
            )
            assert validated == CompilationContext().TARGET_ROCM_ARCHS


class _FakeCppExt:
    """Stands in for torch.utils.cpp_extension: claims both archs are built in."""

    @staticmethod
    def _get_rocm_arch_flags():
        return [f"--offload-arch={a}" for a in FLASHINFER_SUPPORTED_ROCM_ARCHS]


class TestValidateRocmArch:
    def _patch_rocm_version(self, version):
        return patch(
            "flashinfer.rocm.hip_utils.get_system_rocm_version", return_value=version
        )

    def test_valid_arch_returns_arch_list(self):
        with self._patch_rocm_version("7.1.0"):
            result = validate_rocm_arch(arch_list="gfx942")
            assert result == "gfx942"

    def test_multiple_valid_archs(self):
        with self._patch_rocm_version("7.1.0"):
            result = validate_rocm_arch(arch_list="gfx942,gfx950")
            assert result == "gfx942,gfx950"

    def test_raises_when_rocm_not_detected(self):
        with (
            self._patch_rocm_version(None),
            pytest.raises(RuntimeError, match="Could not detect ROCm installation"),
        ):
            validate_rocm_arch(arch_list="gfx942")

    def test_raises_for_unknown_rocm_version(self):
        with (
            self._patch_rocm_version("5.0.0"),
            pytest.raises(RuntimeError, match="not recognized in the ROCm"),
        ):
            validate_rocm_arch(arch_list="gfx942")

    def test_raises_when_all_archs_unsupported(self):
        # gfx950 is only supported from ROCm 7.x
        with (
            self._patch_rocm_version("6.4.0"),
            pytest.raises(RuntimeError, match="does not support any"),
        ):
            validate_rocm_arch(arch_list="gfx950")

    def test_warns_and_filters_partially_unsupported_archs(self):
        with self._patch_rocm_version("6.4.0"):
            # gfx942 is supported in 6.4; gfx950 is not
            with pytest.warns(UserWarning, match="does not support"):
                result = validate_rocm_arch(arch_list="gfx942,gfx950")
            assert result == "gfx942"

    def test_reads_arch_from_env_when_none_given(self, monkeypatch):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
        with self._patch_rocm_version("7.1.0"):
            assert validate_rocm_arch(arch_list=None) == "gfx942"

    def test_falls_back_to_the_running_device_not_a_hard_coded_arch(self, monkeypatch):
        """With no argument and no env var, follow the hardware.

        This used to assert ``== "gfx942"``, encoding the literal that made
        ``validate_flashinfer_rocm_arch(arch_list=None)`` answer ``gfx942`` on a
        gfx950 device while CompilationContext compiled for gfx950. A test that
        pins a wrong constant is how the constant survives, so it is now pinned
        to the detected architecture instead.
        """
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with (
            self._patch_rocm_version("7.1.0"),
            patch(
                "flashinfer.rocm.hip_utils.rocminfo_gpu_agents",
                return_value=(("gfx950", "AMD Instinct MI350X"),),
            ),
        ):
            assert validate_rocm_arch(arch_list=None) == "gfx950"

    def test_verbose_prints_message(self, capsys):
        with self._patch_rocm_version("7.1.0"):
            validate_rocm_arch(arch_list="gfx942", verbose=True)
            captured = capsys.readouterr()
            assert "7.1.0" in captured.out
            assert "gfx942" in captured.out

    @pytest.mark.parametrize("version", ["7.3.0", "7.2.0", "7.1.0", "7.0.0"])
    def test_rocm_7x_supports_gfx950(self, version):
        with self._patch_rocm_version(version):
            assert validate_rocm_arch(arch_list="gfx950") == "gfx950"

    @pytest.mark.parametrize("version", ["7.13.26183", "7.13.0", "7.12.0", "7.11.0"])
    def test_therock_versions_support_gfx950(self, version):
        with self._patch_rocm_version(version):
            assert validate_rocm_arch(arch_list="gfx950") == "gfx950"

    @pytest.mark.parametrize("version", ["6.4.0", "6.3.0"])
    def test_rocm_6x_supports_gfx942_not_gfx950(self, version):
        with self._patch_rocm_version(version):
            assert validate_rocm_arch(arch_list="gfx942") == "gfx942"
            with pytest.raises(RuntimeError):
                validate_rocm_arch(arch_list="gfx950")


# validate_flashinfer_rocm_arch
class TestValidateFlashinferRocmArch:
    def _patch_validate_rocm_arch(self, return_value):
        return patch(
            "flashinfer.rocm.hip_utils.validate_rocm_arch", return_value=return_value
        )

    def test_returns_flags_and_set_for_supported_arch(self):
        with self._patch_validate_rocm_arch("gfx942"):
            flags, arch_set = validate_flashinfer_rocm_arch(arch_list="gfx942")
        assert flags == ["--offload-arch=gfx942"]
        assert arch_set == {"gfx942"}

    def test_multiple_supported_archs(self):
        with self._patch_validate_rocm_arch("gfx942,gfx950"):
            flags, arch_set = validate_flashinfer_rocm_arch(arch_list="gfx942,gfx950")
        assert set(flags) == {"--offload-arch=gfx942", "--offload-arch=gfx950"}
        assert arch_set == {"gfx942", "gfx950"}

    def test_raises_when_no_flashinfer_supported_arch(self):
        # gfx90a passes system ROCm check but is not in FLASHINFER_SUPPORTED_ROCM_ARCHS
        with (
            self._patch_validate_rocm_arch("gfx90a"),
            pytest.raises(RuntimeError, match="FlashInfer does not support any"),
        ):
            validate_flashinfer_rocm_arch(arch_list="gfx90a")

    def test_warns_and_filters_when_some_archs_unsupported_by_flashinfer(self):
        # gfx942 supported, gfx90a not supported by FlashInfer
        with (
            self._patch_validate_rocm_arch("gfx942,gfx90a"),
            pytest.warns(UserWarning, match="FlashInfer does not support"),
        ):
            flags, arch_set = validate_flashinfer_rocm_arch(arch_list="gfx942,gfx90a")
        assert flags == ["--offload-arch=gfx942"]
        assert arch_set == {"gfx942"}

    def test_pytorch_validation_passes_when_all_flags_present(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx942;gfx950")
        torch_cpp_ext = MagicMock()
        torch_cpp_ext._get_rocm_arch_flags.return_value = [
            "--offload-arch=gfx942",
            "--offload-arch=gfx950",
        ]
        with self._patch_validate_rocm_arch("gfx942"):
            flags, arch_set = validate_flashinfer_rocm_arch(
                arch_list="gfx942", torch_cpp_ext_module=torch_cpp_ext
            )
        assert flags == ["--offload-arch=gfx942"]

    def test_pytorch_validation_raises_when_flag_missing(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx950")
        torch_cpp_ext = MagicMock()
        torch_cpp_ext._get_rocm_arch_flags.return_value = ["--offload-arch=gfx950"]
        with (
            self._patch_validate_rocm_arch("gfx942"),
            pytest.raises(RuntimeError, match="PYTORCH_ROCM_ARCH excludes"),
        ):
            validate_flashinfer_rocm_arch(
                arch_list="gfx942", torch_cpp_ext_module=torch_cpp_ext
            )

    @pytest.mark.parametrize(
        "torch_flags",
        [
            pytest.param(["--offload-arch=", "-fno-gpu-rdc"], id="no-visible-device"),
            pytest.param(["--offload-arch=gfx942"], id="cross-compile-from-gfx942"),
        ],
    )
    def test_pytorch_validation_skipped_without_pytorch_rocm_arch(
        self, monkeypatch, torch_flags
    ):
        """Unset PYTORCH_ROCM_ARCH makes torch report visible cards, not its build.

        Enforcing against that breaks a GPU-free import and any cross-compile,
        so the check only applies when the variable pins the set explicitly.
        """
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
        torch_cpp_ext = MagicMock()
        torch_cpp_ext._get_rocm_arch_flags.return_value = torch_flags
        with self._patch_validate_rocm_arch("gfx950"):
            flags, arch_set = validate_flashinfer_rocm_arch(
                arch_list="gfx950", torch_cpp_ext_module=torch_cpp_ext
            )
        assert flags == ["--offload-arch=gfx950"]
        assert arch_set == {"gfx950"}

    def test_reads_arch_from_env_when_none_given(self, monkeypatch):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
        with self._patch_validate_rocm_arch("gfx942"):
            flags, arch_set = validate_flashinfer_rocm_arch(arch_list=None)
        assert arch_set == {"gfx942"}

    def test_defaults_to_gfx942_when_no_env_no_arg(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        with self._patch_validate_rocm_arch("gfx942"):
            flags, arch_set = validate_flashinfer_rocm_arch(arch_list=None)
        assert arch_set == {"gfx942"}

    def test_verbose_prints_message(self, capsys):
        with self._patch_validate_rocm_arch("gfx942"):
            validate_flashinfer_rocm_arch(arch_list="gfx942", verbose=True)
        captured = capsys.readouterr()
        assert "gfx942" in captured.out


# get_available_gpu_count
class TestGetAvailableGpuCount:
    """
    get_available_gpu_count() does ``import torch`` inside the function body,
    so we inject a mock via sys.modules to avoid requiring a real torch install.
    """

    def _make_torch_mock(self, device_count):
        torch_mock = MagicMock()
        torch_mock.cuda.device_count.return_value = device_count
        return torch_mock

    def test_returns_device_count(self):
        with patch.dict("sys.modules", {"torch": self._make_torch_mock(4)}):
            assert get_available_gpu_count() == 4

    def test_returns_zero_when_no_gpus(self):
        with patch.dict("sys.modules", {"torch": self._make_torch_mock(0)}):
            assert get_available_gpu_count() == 0

    def test_delegates_to_torch_cuda_device_count(self):
        torch_mock = self._make_torch_mock(8)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = get_available_gpu_count()
        torch_mock.cuda.device_count.assert_called_once()
        assert result == 8


# rocminfo output template with configurable agent sections
_ROCMINFO_HEADER = "ROCm Agent Enumeration\n"

_ROCMINFO_CPU_AGENT = """\
Agent 1
  Name:                    CPU
  Device Type:             CPU
"""

_ROCMINFO_GPU_AGENT_TEMPLATE = """\
Agent {idx}
  Name:                    {name}
  Device Type:             GPU
"""


def _make_rocminfo_output(*gpu_names, cpu_first=True):
    """Build a synthetic rocminfo output string."""
    lines = [_ROCMINFO_HEADER]
    agent_idx = 1
    if cpu_first:
        lines.append(_ROCMINFO_CPU_AGENT.replace("Agent 1", f"Agent {agent_idx}"))
        agent_idx += 1
    for name in gpu_names:
        lines.append(_ROCMINFO_GPU_AGENT_TEMPLATE.format(idx=agent_idx, name=name))
        agent_idx += 1
    return "".join(lines)


class TestGetSupportedDeviceIndices:
    """Each test clears the functools.cache to avoid cross-test contamination."""

    def setup_method(self):
        get_supported_device_indices.cache_clear()

    def teardown_method(self):
        get_supported_device_indices.cache_clear()

    def _run_result(self, stdout, returncode=0):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        return result

    def test_single_supported_gpu(self):
        output = _make_rocminfo_output("gfx942")
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert indices == (0,)

    def test_two_supported_gpus(self):
        output = _make_rocminfo_output("gfx942", "gfx950")
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert indices == (0, 1)

    def test_unsupported_gpu_excluded(self):
        # gfx90a is not in FLASHINFER_SUPPORTED_ROCM_ARCHS
        output = _make_rocminfo_output("gfx90a")
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert indices == ()

    def test_mixed_supported_and_unsupported(self):
        # GPU 0: gfx942 (supported), GPU 1: gfx90a (unsupported), GPU 2: gfx950 (supported)
        output = _make_rocminfo_output("gfx942", "gfx90a", "gfx950")
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert indices == (0, 2)

    def test_no_gpus_returns_empty_tuple(self):
        output = _ROCMINFO_HEADER + _ROCMINFO_CPU_AGENT
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert indices == ()

    def test_rocminfo_nonzero_returncode_returns_empty(self):
        with patch("subprocess.run", return_value=self._run_result("", returncode=1)):
            indices = get_supported_device_indices()
        assert indices == ()

    def test_rocminfo_not_found_returns_empty(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            indices = get_supported_device_indices()
        assert indices == ()

    def test_rocminfo_timeout_returns_empty(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("rocminfo", 10)
        ):
            indices = get_supported_device_indices()
        assert indices == ()

    def test_result_is_cached(self):
        output = _make_rocminfo_output("gfx942")
        with patch("subprocess.run", return_value=self._run_result(output)) as mock_run:
            get_supported_device_indices()
            get_supported_device_indices()
        # rocminfo should only be called once due to caching
        mock_run.assert_called_once()

    def test_returns_tuple_type(self):
        output = _make_rocminfo_output("gfx942")
        with patch("subprocess.run", return_value=self._run_result(output)):
            indices = get_supported_device_indices()
        assert isinstance(indices, tuple)


# check_torch_rocm_compatibility
def _make_torch_mock(hip=None):
    """
    Build a minimal ``torch`` mock with a ``version`` sub-object whose ``hip``
    attribute is set to *hip*.

    ``from torch import version`` inside the function under test resolves at
    call time from ``sys.modules["torch"]``, so we inject the mock there via
    ``patch.dict``.
    """
    version_mock = MagicMock()
    if hip is None:
        version_mock.hip = None
    else:
        version_mock.hip = hip
    torch_mock = MagicMock()
    torch_mock.version = version_mock
    return torch_mock, version_mock


class TestCheckTorchRocmCompatibility:
    def test_raises_when_hip_is_none(self):
        torch_mock, _ = _make_torch_mock(hip=None)
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            pytest.raises(RuntimeError, match="does NOT have ROCm support"),
        ):
            check_torch_rocm_compatibility()

    def test_raises_when_hip_attribute_missing(self):
        torch_mock = MagicMock()
        # version object with no 'hip' attribute at all
        torch_mock.version = MagicMock(spec=[])
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            pytest.raises(RuntimeError, match="does NOT have ROCm support"),
        ):
            check_torch_rocm_compatibility()

    def test_no_warning_when_system_rocm_undetectable(self):
        torch_mock, _ = _make_torch_mock(hip="6.4.0")
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version", return_value=None
            ),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_torch_rocm_compatibility()
            assert len(w) == 0

    def test_no_warning_when_versions_match(self):
        torch_mock, _ = _make_torch_mock(hip="6.4.0")
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version",
                return_value="6.4.2",
            ),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_torch_rocm_compatibility()
            assert len(w) == 0

    def test_warns_on_major_minor_mismatch(self):
        torch_mock, _ = _make_torch_mock(hip="6.4.0")
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version",
                return_value="7.1.0",
            ),
            pytest.warns(RuntimeWarning, match="version mismatch"),
        ):
            check_torch_rocm_compatibility()

    def test_warning_contains_both_versions(self):
        torch_mock, _ = _make_torch_mock(hip="6.4.0")
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version",
                return_value="7.1.0",
            ),
        ):
            with pytest.warns(RuntimeWarning) as record:
                check_torch_rocm_compatibility()
            message = str(record[0].message)
            assert "7.1.0" in message
            assert "6.4" in message

    def test_patch_version_difference_does_not_warn(self):
        # Same major.minor but different patch: 6.4.0 vs 6.4.2 → no warning
        torch_mock, _ = _make_torch_mock(hip="6.4.0")
        with (
            patch.dict("sys.modules", {"torch": torch_mock}),
            patch(
                "flashinfer.rocm.hip_utils.get_system_rocm_version",
                return_value="6.4.2",
            ),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_torch_rocm_compatibility()
            runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warns) == 0

    def test_error_message_contains_install_instructions(self):
        torch_mock, _ = _make_torch_mock(hip=None)
        with patch.dict("sys.modules", {"torch": torch_mock}):
            with pytest.raises(RuntimeError) as exc_info:
                check_torch_rocm_compatibility()
            msg = str(exc_info.value)
            assert "pip install torch" in msg
            assert "repo.radeon.com" in msg


# FLASHINFER_SUPPORTED_ROCM_ARCHS constant
class TestFlashinferSupportedRocmArchs:
    def test_constant_is_a_list(self):
        assert isinstance(FLASHINFER_SUPPORTED_ROCM_ARCHS, list)

    def test_constant_contains_gfx942(self):
        assert "gfx942" in FLASHINFER_SUPPORTED_ROCM_ARCHS

    def test_constant_contains_gfx950(self):
        assert "gfx950" in FLASHINFER_SUPPORTED_ROCM_ARCHS

    def test_constant_is_non_empty(self):
        assert len(FLASHINFER_SUPPORTED_ROCM_ARCHS) > 0

    def test_all_entries_start_with_gfx(self):
        for arch in FLASHINFER_SUPPORTED_ROCM_ARCHS:
            assert arch.startswith("gfx"), f"Unexpected arch: {arch}"


def _completed(stdout="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


class TestRocmVersionProbes:
    """Each probe returns a version or None; none may raise or hang the build."""

    def test_info_file_reports_the_first_three_components(self, tmp_path):
        info = tmp_path / ".info"
        info.mkdir()
        # Four dot-separated fields: a three-field input leaves the [:3] slice
        # a no-op, so it would not pin the truncation this test is named for.
        (info / "version").write_text("7.2.0.60002-42\n")
        with patch.object(hip_utils, "get_rocm_home", return_value=str(tmp_path)):
            assert hip_utils.get_system_rocm_version_from_info_file() == "7.2.0"

    def test_missing_info_file_is_not_an_error(self, tmp_path):
        with patch.object(hip_utils, "get_rocm_home", return_value=str(tmp_path)):
            assert hip_utils.get_system_rocm_version_from_info_file() is None

    def test_amd_smi_version_is_parsed_out_of_the_banner(self):
        with patch.object(
            subprocess, "run", return_value=_completed("ROCm version: 7.1.0\n")
        ):
            assert hip_utils.get_system_rocm_version_from_amd_smi() == "7.1.0"

    @pytest.mark.parametrize(
        "outcome",
        [
            _completed("ROCm version: 7.1.0\n", returncode=1),
            _completed("no version here\n"),
        ],
    )
    def test_amd_smi_failure_or_unparsable_output_yields_none(self, outcome):
        with patch.object(subprocess, "run", return_value=outcome):
            assert hip_utils.get_system_rocm_version_from_amd_smi() is None

    @pytest.mark.parametrize(
        "raised", [FileNotFoundError, subprocess.TimeoutExpired("amd-smi", 5)]
    )
    def test_amd_smi_absent_or_hung_yields_none(self, raised):
        with patch.object(subprocess, "run", side_effect=raised):
            assert hip_utils.get_system_rocm_version_from_amd_smi() is None

    def test_dpkg_version_is_read_from_the_rocm_core_row_only(self):
        """dpkg -l prints a header and every matched package, so the version has
        to be anchored to rocm-core rather than to the first triple on the page."""
        listing = (
            "||/ Name        Version      Architecture Description\n"
            "+++-===========-============-============-==========\n"
            "ii  libfoo      1.2.3-1      amd64        unrelated\n"
            "ii  rocm-core   7.2.0-1      amd64        ROCm core\n"
        )
        with patch.object(subprocess, "run", return_value=_completed(listing)):
            assert hip_utils.get_system_rocm_version_from_dpkg() == "7.2.0"

    def test_dpkg_absent_yields_none(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            assert hip_utils.get_system_rocm_version_from_dpkg() is None

    def test_dpkg_without_the_package_yields_none(self):
        with patch.object(subprocess, "run", return_value=_completed("", returncode=1)):
            assert hip_utils.get_system_rocm_version_from_dpkg() is None


def _named_probe(name, value):
    """A stand-in probe carrying a real __name__; the ladder prints it."""
    probe = MagicMock(return_value=value)
    probe.__name__ = name
    return probe


class TestRocmVersionLadder:
    def test_therock_build_asks_hipconfig_and_nothing_else(self):
        with (
            patch.object(hip_utils, "is_therock_build", return_value=True),
            patch.object(
                hip_utils,
                "get_system_rocm_version_from_hipconfig",
                return_value="7.9.0",
            ) as hipconfig,
            patch.object(
                hip_utils, "get_system_rocm_version_from_info_file"
            ) as info_file,
        ):
            assert hip_utils.get_system_rocm_version() == "7.9.0"

        hipconfig.assert_called_once()
        info_file.assert_not_called()

    def test_the_first_probe_that_answers_wins(self, capsys):
        with (
            patch.object(hip_utils, "is_therock_build", return_value=False),
            patch.object(
                hip_utils,
                "get_system_rocm_version_from_info_file",
                return_value="7.2.0",
            ),
            patch.object(hip_utils, "get_system_rocm_version_from_amd_smi") as amd_smi,
        ):
            assert hip_utils.get_system_rocm_version() == "7.2.0"

        amd_smi.assert_not_called()
        assert "Trying next method" not in capsys.readouterr().out

    def test_a_silent_probe_names_itself_and_falls_through(self, capsys):
        with (
            patch.object(hip_utils, "is_therock_build", return_value=False),
            patch.object(
                hip_utils,
                "get_system_rocm_version_from_info_file",
                _named_probe("get_system_rocm_version_from_info_file", None),
            ),
            patch.object(
                hip_utils,
                "get_system_rocm_version_from_amd_smi",
                _named_probe("get_system_rocm_version_from_amd_smi", "7.1.0"),
            ),
        ):
            assert hip_utils.get_system_rocm_version() == "7.1.0"

        assert "get_system_rocm_version_from_info_file" in capsys.readouterr().out

    def test_all_probes_silent_yields_none(self):
        names = (
            "get_system_rocm_version_from_info_file",
            "get_system_rocm_version_from_amd_smi",
            "get_system_rocm_version_from_dpkg",
            "get_system_rocm_version_from_hipconfig",
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(hip_utils, "is_therock_build", return_value=False)
            )
            for name in names:
                stack.enter_context(
                    patch.object(hip_utils, name, _named_probe(name, None))
                )
            assert hip_utils.get_system_rocm_version() is None


def _vram_json(**cards):
    import json

    return json.dumps(
        {f"card{i}": {"VRAM Total Memory (B)": str(v)} for i, v in cards.items()}
    )


@pytest.fixture
def physical_cards():
    """Call get_physical_card_device_indices uncached, with a fixed device list."""
    hip_utils.get_physical_card_device_indices.cache_clear()

    def _call(supported, run_result=None, run_error=None):
        kwargs = (
            {"side_effect": run_error} if run_error else {"return_value": run_result}
        )
        with (
            patch.object(
                hip_utils, "get_supported_device_indices", return_value=supported
            ),
            patch.object(subprocess, "run", **kwargs),
        ):
            hip_utils.get_physical_card_device_indices.cache_clear()
            return hip_utils.get_physical_card_device_indices()

    yield _call
    hip_utils.get_physical_card_device_indices.cache_clear()


class TestPhysicalCardIndices:
    """CPX splits one card into several devices; allocating on each concurrently
    exhausts the shared VRAM, so callers need one index per physical card."""

    def test_no_supported_devices_is_empty(self, physical_cards):
        assert physical_cards(()) == ()

    def test_identical_vram_keeps_every_device(self, physical_cards):
        result = physical_cards((0, 1), _completed(_vram_json(**{"0": 200, "1": 200})))
        assert result == (0, 1)

    def test_cpx_partitions_collapse_to_the_full_capacity_device(self, physical_cards):
        # card0 reports the whole card; card1 and card2 are its partitions.
        result = physical_cards(
            (0, 1, 2), _completed(_vram_json(**{"0": 200, "1": 50, "2": 50}))
        )
        assert result == (0,)

    @pytest.mark.parametrize(
        "run_result, run_error",
        [
            (_completed("", returncode=1), None),
            (_completed("not json"), None),
            (None, FileNotFoundError),
            (None, subprocess.TimeoutExpired("rocm-smi", 10)),
        ],
    )
    def test_unusable_rocm_smi_falls_back_to_the_supported_list(
        self, physical_cards, run_result, run_error
    ):
        assert physical_cards((0, 1), run_result, run_error) == (0, 1)

    def test_unparsable_card_entries_are_skipped(self, physical_cards):
        import json

        payload = json.dumps(
            {
                "card0": {"VRAM Total Memory (B)": "200"},
                "cardX": {"VRAM Total Memory (B)": "200"},
                "card1": {"no such key": "1"},
                "system": {"VRAM Total Memory (B)": "999"},
            }
        )
        # Device 1 reported nothing readable, so it is dropped rather than
        # assumed to be a full card.
        assert physical_cards((0, 1), _completed(payload)) == (0,)

    def test_no_vram_reported_for_any_supported_device_falls_back(self, physical_cards):
        assert physical_cards((3, 4), _completed(_vram_json(**{"0": 200}))) == (3, 4)
