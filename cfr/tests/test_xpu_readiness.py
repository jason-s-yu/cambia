"""Regression tests for XPU (Intel Arc via torch's xpu backend) readiness.

There is no XPU device in this environment, so every test here either drives
the device-string plumbing directly (cpu/xpu never require a real device) or
monkeypatches ``torch.cuda``/``torch.xpu`` attributes to simulate an
xpu-capable or xpu-absent host. cambia-333.
"""

import os
import re

import numpy as np
import pytest
import torch

from src.cfr import gpu_safety
from src.cfr.prtcfr_trainer import _accel_rng_save, _accel_rng_restore


def _read_source(relative_path: str) -> str:
    """Read a source file relative to cfr/src/."""
    base = os.path.join(os.path.dirname(__file__), "..", "src")
    with open(os.path.join(base, relative_path)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# gpu_safety.py: device_type dispatch (cuda vs xpu vs unavailable)
# ---------------------------------------------------------------------------


class _FakeAccel:
    """Stand-in for torch.cuda / torch.xpu, mirroring the memory/OOM API both
    backends expose (mem_get_info, set_per_process_memory_fraction, empty_cache).
    """

    def __init__(self, free_bytes: int, total_bytes: int, available: bool = True):
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.available = available
        self.empty_cache_calls = 0
        self.set_fraction_calls = []

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def mem_get_info(self, dev):
        return (self.free_bytes, self.total_bytes)

    def set_per_process_memory_fraction(self, frac, dev):
        self.set_fraction_calls.append((frac, dev))

    def empty_cache(self):
        self.empty_cache_calls += 1


@pytest.fixture
def fake_xpu(monkeypatch):
    fake = _FakeAccel(free_bytes=4 * gpu_safety._GB, total_bytes=8 * gpu_safety._GB)
    monkeypatch.setattr(torch, "xpu", fake, raising=False)
    return fake


class TestAccelModuleDispatch:
    def test_cuda_device_type_routes_to_torch_cuda(self):
        if torch.cuda.is_available():
            assert gpu_safety._accel_module("cuda") is torch.cuda
        else:
            assert gpu_safety._accel_module("cuda") is None

    def test_xpu_device_type_routes_to_torch_xpu(self, fake_xpu):
        assert gpu_safety._accel_module("xpu") is fake_xpu

    def test_xpu_unavailable_returns_none(self, fake_xpu):
        fake_xpu.available = False
        assert gpu_safety._accel_module("xpu") is None

    def test_cpu_device_type_returns_none(self):
        assert gpu_safety._accel_module("cpu") is None

    def test_accel_available_matches_module_dispatch(self, fake_xpu):
        assert gpu_safety.accel_available("xpu") is True
        fake_xpu.available = False
        assert gpu_safety.accel_available("xpu") is False

    def test_cuda_available_is_backward_compatible_alias(self):
        assert gpu_safety.cuda_available() == gpu_safety.accel_available("cuda")


class TestVramHelpersDispatchByDeviceType:
    def test_cuda_mem_info_uses_xpu_module_when_requested(self, fake_xpu):
        info = gpu_safety.cuda_mem_info(device_type="xpu")
        assert info == (fake_xpu.free_bytes, fake_xpu.total_bytes)

    def test_cuda_mem_info_none_when_xpu_unavailable(self, fake_xpu):
        fake_xpu.available = False
        assert gpu_safety.cuda_mem_info(device_type="xpu") is None

    def test_require_free_vram_noop_when_xpu_unavailable(self, fake_xpu):
        fake_xpu.available = False
        # Must return immediately without raising, matching the CUDA-absent
        # no-op contract exercised in test_gpu_safety.py.
        gpu_safety.require_free_vram(9999.0, wait_s=0.0, device_type="xpu")

    def test_require_free_vram_passes_when_budget_met_on_xpu(self, fake_xpu):
        gpu_safety.require_free_vram(1.0, wait_s=0.0, device_type="xpu")

    def test_require_free_vram_raises_when_budget_not_met_on_xpu(self, fake_xpu):
        fake_xpu.free_bytes = 0
        with pytest.raises(RuntimeError, match="insufficient free VRAM"):
            gpu_safety.require_free_vram(
                9999.0, wait_s=0.0, poll_s=0.0, device_type="xpu"
            )

    def test_cap_process_vram_dispatches_to_xpu_module(self, fake_xpu):
        gpu_safety.cap_process_vram(2.0, device_type="xpu")
        assert fake_xpu.set_fraction_calls, (
            "cap_process_vram(device_type='xpu') must call "
            "set_per_process_memory_fraction on torch.xpu, not torch.cuda"
        )

    def test_cap_process_vram_noop_when_xpu_unavailable(self, fake_xpu):
        fake_xpu.available = False
        gpu_safety.cap_process_vram(2.0, device_type="xpu")
        assert fake_xpu.set_fraction_calls == []


class TestOomRetryDispatchByDeviceType:
    def test_noop_passthrough_when_xpu_unavailable(self, fake_xpu):
        fake_xpu.available = False

        @gpu_safety.oom_retry(retries=2, device_type="xpu")
        def f(x):
            return x + 1

        assert f(41) == 42

    def test_retries_and_recovers_on_shared_out_of_memory_error(self, fake_xpu):
        calls = {"n": 0}

        @gpu_safety.oom_retry(retries=2, backoff_s=0.0, device_type="xpu")
        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise torch.OutOfMemoryError("simulated OOM")
            return "ok"

        assert flaky() == "ok"
        assert fake_xpu.empty_cache_calls == 1

    def test_reraises_as_runtimeerror_after_exhausting_retries(self, fake_xpu):
        @gpu_safety.oom_retry(retries=1, backoff_s=0.0, device_type="xpu")
        def always_ooms():
            raise torch.OutOfMemoryError("simulated OOM")

        with pytest.raises(RuntimeError, match="xpu OOM"):
            always_ooms()


# ---------------------------------------------------------------------------
# prtcfr_trainer.py: accelerator RNG save/restore (resume_state.json)
# ---------------------------------------------------------------------------


class _FakeAccelRng:
    def __init__(self, available: bool = True):
        self.available = available
        self.restored_states = None

    def is_available(self):
        return self.available

    def get_rng_state_all(self):
        return [torch.tensor([1, 2, 3], dtype=torch.uint8)]

    def set_rng_state_all(self, states):
        self.restored_states = states


class TestAccelRngSaveRestore:
    def test_cpu_device_saves_nothing(self):
        assert _accel_rng_save("cpu") == {}

    def test_xpu_device_saves_xpu_key_only(self, monkeypatch):
        fake = _FakeAccelRng()
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        result = _accel_rng_save("xpu")
        assert "torch_xpu_rng" in result
        assert "torch_cuda_rng" not in result
        assert isinstance(result["torch_xpu_rng"], list)

    def test_xpu_unavailable_saves_nothing(self, monkeypatch):
        fake = _FakeAccelRng(available=False)
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        assert _accel_rng_save("xpu") == {}

    def test_cuda_device_saves_cuda_key_only(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_rng_state_all",
            lambda: [torch.tensor([9], dtype=torch.uint8)],
        )
        result = _accel_rng_save("cuda")
        assert "torch_cuda_rng" in result
        assert "torch_xpu_rng" not in result

    def test_cuda_colon_index_device_string_dispatches_by_kind(self, monkeypatch):
        # self.device is often a "cuda:0"-shaped string; only the part before
        # ":" should select the backend.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_rng_state_all",
            lambda: [torch.tensor([9], dtype=torch.uint8)],
        )
        result = _accel_rng_save("cuda:0")
        assert "torch_cuda_rng" in result

    def test_restore_dispatches_to_xpu_when_key_present(self, monkeypatch):
        fake = _FakeAccelRng()
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        _accel_rng_restore(
            {"torch_xpu_rng": [np.zeros(4, dtype=np.uint8).tobytes().hex()]}
        )
        assert fake.restored_states is not None

    def test_restore_noop_without_either_key(self, monkeypatch):
        fake = _FakeAccelRng()
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        # Backward-compat: a CPU-only or pre-accelerator-RNG resume_state.json
        # has neither key; restore must not raise or call set_rng_state_all.
        _accel_rng_restore({"iteration": 3})
        assert fake.restored_states is None

    def test_restore_skips_xpu_key_when_xpu_unavailable(self, monkeypatch):
        fake = _FakeAccelRng(available=False)
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        _accel_rng_restore(
            {"torch_xpu_rng": [np.zeros(4, dtype=np.uint8).tobytes().hex()]}
        )
        assert fake.restored_states is None

    def test_save_then_restore_roundtrip_on_xpu(self, monkeypatch):
        fake = _FakeAccelRng()
        monkeypatch.setattr(torch, "xpu", fake, raising=False)
        saved = _accel_rng_save("xpu")
        fake.restored_states = None
        _accel_rng_restore(saved)
        assert fake.restored_states is not None
        assert len(fake.restored_states) == len(saved["torch_xpu_rng"])


# ---------------------------------------------------------------------------
# prtcfr_net.py / prtcfr_critic.py: device defaults no longer hardcode "cuda"
# ---------------------------------------------------------------------------


class TestPrtcfrNetDeviceDefaults:
    def test_net_with_no_device_arg_defaults_to_cpu(self):
        from src.cfr.prtcfr_net import PRTCFRNet

        net = PRTCFRNet(
            vocab_size=8, embed_dim=4, hidden_dim=8, num_layers=1, head_hidden_dim=8
        )
        assert net.device == torch.device("cpu")

    def test_build_prtcfr_net_without_config_defaults_to_cpu(self):
        from src.cfr.prtcfr_net import build_prtcfr_net

        net = build_prtcfr_net(config=None, device=None)
        assert net.device == torch.device("cpu")

    def test_critic_net_with_no_device_arg_defaults_to_cpu(self):
        from src.cfr.prtcfr_critic import PRTCFRCriticNet

        net = PRTCFRCriticNet(
            vocab_size=8, embed_dim=4, hidden_dim=8, num_layers=1, head_hidden_dim=8
        )
        assert net.device == torch.device("cpu")

    def test_build_prtcfr_critic_net_without_config_defaults_to_cpu(self):
        from src.cfr.prtcfr_critic import build_prtcfr_critic_net

        net = build_prtcfr_critic_net(config=None, device=None)
        assert net.device == torch.device("cpu")

    @pytest.mark.parametrize("relpath", ["cfr/prtcfr_net.py", "cfr/prtcfr_critic.py"])
    def test_source_has_no_cuda_literal(self, relpath):
        source = _read_source(relpath)
        assert "cuda" not in source.lower(), (
            f"{relpath} must not hardcode a cuda-specific default; every "
            "production call site passes device explicitly, so the safe "
            "universal default is None -> cpu (see PRTCFRNet.__init__)."
        )


# ---------------------------------------------------------------------------
# deep_trainer.py: source-level dispatch checks (the trainer itself needs a
# heavy fixture to instantiate, so the pin_memory/profiler branches -- which
# only run inside a live training step -- are verified at the source level,
# matching TestResolveDeviceSource's existing pattern in test_device_config.py).
# ---------------------------------------------------------------------------


class TestDeepTrainerXpuDispatch:
    def _source(self):
        return _read_source("cfr/deep_trainer.py")

    def test_gradscaler_already_generalized_via_device_type(self):
        # B4: torch.amp.GradScaler(device, ...) accepts any device-type
        # string (cuda/cpu/xpu) in this torch version; no cuda literal here.
        assert "torch.amp.GradScaler(self.device.type" in self._source()

    def test_autocast_already_generalized_via_device_type(self):
        source = self._source()
        assert source.count("torch.amp.autocast(self.device.type") >= 2

    def test_pin_memory_prefetch_gate_is_not_cuda_only(self):
        source = self._source()
        assert 'if self.device.type != "cpu":' in source
        assert (
            'if self.device.type == "cuda":\n            prefetch_queue' not in source
        ), "pin_memory/non_blocking prefetch path regressed to a cuda-only gate"

    def test_profiler_activity_dispatches_both_cuda_and_xpu(self):
        source = self._source()
        assert "ProfilerActivity.CUDA" in source
        assert "ProfilerActivity.XPU" in source
        # The two must be paired (if/elif), not two independent unconditional
        # appends -- assert the xpu branch immediately follows the cuda one.
        match = re.search(
            r'if self\.device\.type == "cuda":\s*\n\s*_prof_activities\.append\('
            r"_torch_profiler\.ProfilerActivity\.CUDA\)\s*\n\s*elif self\.device\.type"
            r' == "xpu":\s*\n\s*_prof_activities\.append\(_torch_profiler\.ProfilerActivity\.XPU\)',
            source,
        )
        assert match, "expected an if/elif cuda/xpu pair around ProfilerActivity"


# ---------------------------------------------------------------------------
# cli.py: device resolution reuse (no forked cuda-only auto-detect logic)
# ---------------------------------------------------------------------------


class TestCliDeviceResolutionReuse:
    def _source(self):
        return _read_source("cli.py")

    def test_desca_reuses_shared_resolve_device(self):
        source = self._source()
        # cambia-333: train_desca used to inline its own cuda->xpu->cpu
        # detection instead of reusing deep_trainer._resolve_device (the
        # canonical resolver), which meant it silently missed the
        # explicit-xpu-unavailable RuntimeError path prtcfr already got.
        assert source.count("from .cfr.deep_trainer import _resolve_device") >= 2

    def test_desca_no_longer_forks_its_own_cuda_xpu_detection(self):
        source = self._source()
        assert (
            'if _torch.cuda.is_available():\n                _device = "cuda"'
            not in source
        )

    def test_prtcfr_device_help_text_mentions_xpu(self):
        source = self._source()
        assert (
            "Training device: auto, cpu, cuda, xpu (default: config prt_cfr.device)"
            in source
        )

    def test_vram_safety_gate_covers_xpu_not_just_cuda(self):
        source = self._source()
        assert "gpu_safety.accel_available(_accel_type)" in source
        assert '_accel_type in ("cuda", "xpu")' in source
        # Regression guard: the old gate only ever fired for a "cuda"-prefixed
        # device string, silently skipping the VRAM failsafe for xpu runs.
        assert (
            'if _device.startswith("cuda") and gpu_safety.cuda_available():' not in source
        )


# ---------------------------------------------------------------------------
# Reachable-path files: no stray unguarded "cuda" literal outside the
# device-dispatch helpers and the resolver. Grep-style source assertion: any
# line mentioning "cuda" in these files must match one of the allowed
# substrings below, so a regression (a reintroduced cuda-only branch) fails
# loudly instead of silently degrading xpu behavior.
# ---------------------------------------------------------------------------

_ALLOWED_CUDA_SUBSTRINGS = {
    "cfr/gpu_safety.py": [
        "cuda and xpu",
        '"cuda" by default',
        "torch.cuda``/``torch.xpu``",
        'device_type == "cuda"',
        "torch.cuda if torch.cuda.is_available()",
        'device_type: str = "cuda"',
        "def cuda_available",
        'accel_available("cuda")',
        "def cuda_mem_info",
        "cuda_mem_info(device, device_type)",
        "both cuda and xpu allocator OOMs",
        "torch.cuda.OutOfMemoryError",
    ],
    "cfr/deep_trainer.py": [
        '"auto" = cuda if available',
        "(FP16) on CUDA",
        "CUDA graph optimization",
        "torch.cuda.is_available():",
        'return "cuda"',
        "CUDA and XPU",
        "cuda supports",
        'if self.device.type == "cuda":',
        "ProfilerActivity.CUDA",
    ],
    "cfr/prtcfr_trainer.py": [
        "(cuda or xpu)",
        "pre-xpu CUDA-only",
        'kind == "cuda" and torch.cuda.is_available()',
        '"torch_cuda_rng"',
        "torch.cuda.get_rng_state_all()",
        "torch.cuda.set_rng_state_all",
        "torch_cuda_rng",
        "cuda_rng and torch.cuda.is_available()",
        "for s in cuda_rng",
    ],
    # cambia-342: cfr/src/benchmarks dev-tooling surface. These benchmarks
    # were deliberately skipped by the cambia-329 sweep; this closes that gap
    # so a reintroduced cuda-only default/dispatch fails loudly here too.
    "benchmarks/__init__.py": [],
    "benchmarks/memory_bench.py": [],
    "benchmarks/reporting.py": [],
    "benchmarks/worker_scaling.py": [],
    "benchmarks/desca_bench.py": [
        "auto|cpu|cuda|mps|xpu",
        'device.type == "cuda"',
        "torch.cuda.get_device_properties",
        "torch.cuda.synchronize()",
        "torch.cuda.is_bf16_supported()",
        "cuda -> xpu -> cpu",
        '"auto", "cpu", "cuda", "mps", "xpu"',
        "CUDA tensor cores",
        "on CUDA",
    ],
    "benchmarks/e2e_bench.py": [
        '"cpu" or "cuda"',
        'accel_available("cuda")',
    ],
    "benchmarks/es_bench.py": [
        "cpu/cuda",
    ],
    "benchmarks/traversal_bench.py": [
        "cpu/cuda",
    ],
    "benchmarks/network_bench.py": [
        "'cuda' or 'xpu'",
        '("cuda", "xpu")',
        'kind == "cuda"',
        "torch.cuda.synchronize()",
        "torch.cuda.reset_peak_memory_stats()",
        "torch.cuda.max_memory_allocated()",
        "torch.cuda.get_device_name(0)",
        'accel_available("cuda")',
        '"cpu" or "cuda"',
        "cuda -> xpu -> cpu",
        '"cpu", "cuda", or "both"',
    ],
    "benchmarks/runner.py": [
        '"cpu", "cuda", etc.',
        "cuda -> xpu -> cpu",
        "cuda-only auto-detect",
        'accel_available("cuda")',
        'accel_kind == "cuda"',
        "torch.cuda.get_device_name(0)",
    ],
}

# cli.py: scope the check to the reachable train_deep/train_desca/train_prtcfr
# command bodies rather than the whole file. The cli.py benchmark_* typer
# command wiring itself stays out of scope here; the benchmarks/*.py library
# code it dispatches into is covered separately (see _ALLOWED_CUDA_SUBSTRINGS
# above, cambia-342).
_CLI_ALLOWED_CUDA_SUBSTRINGS = [
    "auto, cpu, cuda, xpu",
    "cuda -> xpu",
    "cuda-only auto-detect",
    '"cuda:0"',
    'in ("cuda", "xpu")',
    '"cuda" if gpu else "cpu"',  # deprecated --gpu/--no-gpu flag: cuda-only by definition
    "on CUDA/XPU",  # --amp/--compile help text; the actual gate is device.type != "cpu"
    "concretely (cuda ->",
]


class TestNoStrayCudaLiterals:
    @pytest.mark.parametrize("relpath", sorted(_ALLOWED_CUDA_SUBSTRINGS))
    def test_file_has_no_unallowed_cuda_line(self, relpath):
        source = _read_source(relpath)
        allowed = _ALLOWED_CUDA_SUBSTRINGS[relpath]
        offenders = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            if "cuda" not in line.lower():
                continue
            if not any(sub.lower() in line.lower() for sub in allowed):
                offenders.append(f"{relpath}:{lineno}: {line.strip()}")
        assert not offenders, (
            "unallowed 'cuda' literal(s) found -- generalize for xpu or add "
            "to the allowlist if deliberate:\n" + "\n".join(offenders)
        )

    def test_cli_reachable_commands_have_no_unallowed_cuda_line(self):
        source = _read_source("cli.py")
        lines = source.splitlines()

        def _command_span(def_name: str):
            start = next(
                i for i, l in enumerate(lines) if l.startswith(f"def {def_name}(")
            )
            end = start + 1
            while end < len(lines) and not lines[end].startswith("def "):
                end += 1
            return start, end

        offenders = []
        for cmd in ("train_deep", "train_desca", "train_prtcfr"):
            start, end = _command_span(cmd)
            for lineno in range(start, end):
                line = lines[lineno]
                if "cuda" not in line.lower():
                    continue
                if not any(
                    sub.lower() in line.lower() for sub in _CLI_ALLOWED_CUDA_SUBSTRINGS
                ):
                    offenders.append(f"cli.py:{lineno + 1} ({cmd}): {line.strip()}")
        assert not offenders, (
            "unallowed 'cuda' literal(s) found in a reachable train command -- "
            "generalize for xpu or add to the allowlist if deliberate:\n"
            + "\n".join(offenders)
        )
