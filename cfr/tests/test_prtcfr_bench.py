"""tests/test_prtcfr_bench.py

Scoped smoke tests for scripts/prtcfr_bench.py (the X3 throughput bench
harness, v0.4 Phase 2 S1W7). Fast, CPU-only, tiny K -- the real X3 measurement
(K=8192, m=4, Go backend, GPU) is a one-off operator invocation captured in
the sprint's X3 verdict artifact, not part of the scoped suite.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")

from scripts.prtcfr_bench import (  # noqa: E402
    ContentionMonitor,
    Heartbeat,
    _cuda_warm_or_timeout,
    _replicate_reservoir_to_size,
    build_arg_parser,
    build_bench_config,
    contention_snapshot,
    gate_verdict,
    human_summary,
    run_cell,
    run_cell_with_oom_backoff,
    run_fit_scale_probe,
)


def _tiny_args(**overrides) -> argparse.Namespace:
    args = build_arg_parser().parse_args(
        [
            "--k-games",
            "4",
            "--m-rollouts",
            "1",
            "--batch-size",
            "16",
            "--train-steps",
            "5",
            "--seq-cap",
            "128",
            "--backend",
            "python",
            "--device",
            "cpu",
            "--max-trajectory-steps",
            "40",
            "--no-wait-for-clean-host",
        ]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Contention snapshot
# ---------------------------------------------------------------------------


def test_contention_snapshot_has_load_and_gpu_keys():
    snap = contention_snapshot()
    assert "load1" in snap and "gpu" in snap and "timestamp" in snap


def test_contention_monitor_collects_samples_over_a_short_window():
    with ContentionMonitor(interval_s=0.05) as mon:
        import time

        time.sleep(0.2)
    summary = mon.summary()
    assert summary["samples"] >= 1


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def test_build_bench_config_overrides_apply(tmp_path):
    args = _tiny_args()
    cfg = build_bench_config(args)
    assert cfg.k_games_per_iter == 4
    assert cfg.m_rollouts == 1
    assert cfg.batch_size == 16
    assert cfg.train_steps == 5
    assert cfg.seq_cap == 128
    assert cfg.backend == "python"
    assert cfg.device == "cpu"
    # critic/stability disabled by default (out of X3 gate scope).
    assert cfg.critic_enabled is False
    assert cfg.stability_enabled is False
    # Non-overridden production fields still come from the real config.
    assert cfg.num_players == 2


# ---------------------------------------------------------------------------
# End-to-end tiny cell (python backend, CPU)
# ---------------------------------------------------------------------------


def test_run_cell_tiny_python_backend_cpu(tmp_path):
    args = _tiny_args(run_dir=str(tmp_path / "run"))
    result = run_cell(args)

    assert result["gen_seconds"] >= 0.0
    assert result["fit_seconds"] >= 0.0
    assert result["config"]["k_games"] == 4
    assert result["config"]["backend"] == "python"

    # Profile buckets are present and sum consistently with the phase totals.
    # Each bucket is independently rounded to 3dp before being stored, so the
    # sum can differ from the (also independently rounded) total by a few
    # thousandths; the buckets are exact (unrounded) by construction as
    # remainders of gen_seconds/fit_seconds, so a millisecond-scale tolerance
    # is the rounding-only slack, not a correctness gap.
    p = result["profile"]
    gen_sum = (
        p["gen"]["ffi_s"]
        + p["gen"]["inference_s"]
        + p["gen"]["reservoir_write_s"]
        + p["gen"]["other_s"]
    )
    assert gen_sum == pytest.approx(p["gen"]["total_s"], abs=5e-3)
    fit_sum = (
        p["fit"]["reservoir_sample_s"]
        + p["fit"]["forward_s"]
        + p["fit"]["backward_opt_misc_s"]
    )
    assert fit_sum == pytest.approx(p["fit"]["total_s"], abs=5e-3)

    # At least some FFI and inference calls happened during generation.
    assert p["call_counts"].get("ffi", 0) > 0
    assert p["call_counts"].get("inference", 0) > 0

    # Contention annotation present.
    assert "host_load1_before" in result["contention"]
    assert "during_summary" in result["contention"]


def test_run_cell_with_oom_backoff_passthrough_when_no_oom(tmp_path):
    """No CUDA/OOM path exercised on CPU; the wrapper must simply pass the
    result through unchanged."""
    args = _tiny_args(run_dir=str(tmp_path / "run2"))
    result = run_cell_with_oom_backoff(args)
    assert result["gen_seconds"] >= 0.0
    assert result["fit_seconds"] >= 0.0


def test_gate_verdict_pass_and_fail():
    result = {"gen_seconds": 10.0, "fit_seconds": 200.0}
    v = gate_verdict(result, gate_gen_s=90.0, gate_fit_s=120.0)
    assert v["gen"] == "PASS"
    assert v["fit"] == "FAIL"


def test_human_summary_contains_gate_lines(tmp_path):
    args = _tiny_args(run_dir=str(tmp_path / "run3"))
    result = run_cell(args)
    text = human_summary(result, gate_gen_s=90.0, gate_fit_s=120.0)
    assert "gen:" in text and "fit:" in text
    assert "PASS" in text or "FAIL" in text


# ---------------------------------------------------------------------------
# Fit-scale probe
# ---------------------------------------------------------------------------


def test_replicate_reservoir_to_size_grows_and_preserves_content(tmp_path):
    from src.disk_reservoir import DiskReservoir
    from src.encoding import NUM_ACTIONS

    disk = DiskReservoir(
        path=str(tmp_path / "r0"),
        capacity=10_000,
        seq_cap=64,
        target_dim=NUM_ACTIONS,
        has_mask=True,
        seed=0,
    )
    tgt = __import__("numpy").zeros(NUM_ACTIONS, dtype="float32")
    mask = __import__("numpy").zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True
    from src.reservoir import ReservoirSample

    for toks in ([1, 5, 6], [1, 5, 7, 8]):
        disk.add(
            ReservoirSample(features=toks, target=tgt, action_mask=mask, iteration=1)
        )
    assert len(disk) == 2

    _replicate_reservoir_to_size(disk, 20)
    assert len(disk) >= 20

    batch = disk.sample_batch(len(disk))
    # Every row's natural length must be one of the two original lengths (3 or 4):
    # replication only ever re-inserts real rows, never synthesizes new content.
    assert set(int(x) for x in batch.lengths) <= {3, 4}


def test_replicate_reservoir_to_size_noop_when_already_at_target(tmp_path):
    from src.disk_reservoir import DiskReservoir
    from src.encoding import NUM_ACTIONS

    disk = DiskReservoir(
        path=str(tmp_path / "r1"),
        capacity=10_000,
        seq_cap=64,
        target_dim=NUM_ACTIONS,
        has_mask=True,
        seed=0,
    )
    assert len(disk) == 0
    _replicate_reservoir_to_size(disk, 5)  # empty reservoir: no-op, must not crash
    assert len(disk) == 0


def test_run_fit_scale_probe_tiny_python_backend_cpu(tmp_path):
    args = _tiny_args(
        run_dir=str(tmp_path / "fitprobe"),
        batch_size=24,
        train_steps=3,
    )
    result = run_fit_scale_probe(args, seed_k_games=4)

    assert result["mode"] == "fit_scale_probe"
    assert result["fit_seconds"] >= 0.0
    assert result["config"]["batch_size"] == 24
    assert result["config"]["train_steps"] == 3
    # Reservoirs replicated up to at least the requested per-player share.
    sizes = result["config"]["reservoir_sizes_after_replication"]
    assert all(v >= 1 for v in sizes.values())
    assert "caveat" in result and "REAL" in result["caveat"]


# ---------------------------------------------------------------------------
# Heartbeat + fail-fast CUDA init (hang localization, added after the K=2
# GPU probe hung silently under contention -- see the S1W7 verdict).
# ---------------------------------------------------------------------------


def test_heartbeat_tick_throttles_force_always_prints(capsys):
    hb = Heartbeat(min_interval_s=1000.0)  # effectively never elapses in-test
    hb.force("first")
    hb.tick("suppressed")  # too soon after force(); must not print
    hb.force("second")  # force always prints regardless of throttle
    out = capsys.readouterr().out
    assert out.count("heartbeat") == 2
    assert "first" in out and "second" in out
    assert "suppressed" not in out


def test_cuda_warm_or_timeout_is_noop_on_cpu():
    # Must return immediately and never raise when device is not CUDA.
    _cuda_warm_or_timeout("cpu", timeout_s=1.0)


def test_cuda_warm_or_timeout_raises_timeout_when_warm_fn_hangs(monkeypatch):
    """Simulate a wedged CUDA context (as observed with tgx at ~100% SM
    util): torch.zeros/synchronize block forever. The helper must raise
    TimeoutError promptly rather than hang."""
    import time as _time

    import scripts.prtcfr_bench as bench_mod

    class _HangingTorch:
        class cuda:
            @staticmethod
            def synchronize():
                _time.sleep(5.0)

        @staticmethod
        def zeros(*a, **kw):
            return object()

    monkeypatch.setitem(sys.modules, "torch", _HangingTorch)
    with pytest.raises(TimeoutError):
        bench_mod._cuda_warm_or_timeout("cuda", timeout_s=0.2)
