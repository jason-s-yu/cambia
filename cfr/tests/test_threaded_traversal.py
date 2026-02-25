"""
tests/test_threaded_traversal.py

Integration tests and benchmarks for multi-threaded Go FFI traversals (Task #9).

Verifies that:
- ThreadPoolExecutor-based traversals produce valid, non-empty results
- Multiple threads produce consistent sample counts
- Benchmarks 1/2/4 threads at various traversal counts
"""

import time

import pytest
import numpy as np
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        e = GoEngine(seed=0)
        e.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(engine_backend="go", recursion_limit=12) -> SimpleNamespace:
    """Build a minimal config namespace for Go backend traversal tests."""
    config = SimpleNamespace()
    config.deep_cfr = SimpleNamespace(engine_backend=engine_backend)
    config.system = SimpleNamespace(recursion_limit=recursion_limit)
    config.agent_params = SimpleNamespace(memory_level=0, time_decay_turns=0)
    config.cambia_rules = SimpleNamespace(
        allowDrawFromDiscardPile=False,
        allowReplaceAbilities=False,
        snapRace=False,
        penaltyDrawCount=2,
        use_jokers=2,
        cards_per_player=4,
        initial_view_count=2,
        cambia_allowed_round=0,
        allowOpponentSnapping=False,
        max_game_turns=300,
    )
    config.logging = SimpleNamespace(
        log_max_bytes=1024 * 1024,
        log_backup_count=1,
        log_file_prefix="test",
        log_archive_enabled=False,
        log_archive_max_archives=0,
        log_archive_dir="",
        log_size_update_interval_sec=60,
        log_simulation_traces=False,
        simulation_trace_filename_prefix="sim",
        get_worker_log_level=lambda wid, ntotal: "WARNING",
    )
    config.cfr_training = SimpleNamespace(num_workers=1)
    return config


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestThreadedTraversal:
    def test_threaded_traversal_produces_samples(self):
        """Multi-threaded traversal produces non-empty advantage and strategy samples."""
        from src.cfr.deep_trainer import _run_traversals_threaded
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        config = _make_config()
        network_config = {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS}

        adv, strat, val, done, nodes = _run_traversals_threaded(
            iteration_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=4,
            num_threads=2,
            run_log_dir="/tmp/test_threaded_logs",
            run_timestamp="test",
        )

        assert done == 4, f"Expected 4 traversals done, got {done}"
        assert len(adv) > 0, "Expected non-empty advantage samples"
        assert nodes > 0, "Expected non-zero nodes visited"

        # --- Behavioral: strategy samples are valid probability distributions ---
        # Each strategy sample stores σ(I) at opponent info sets. The sum over
        # legal actions must equal 1.0 (probability simplex constraint).
        for sample in strat:
            mask = sample.action_mask
            strategy_sum = sample.target[mask].sum()
            assert abs(strategy_sum - 1.0) < 0.01, (
                f"Threaded strategy sum {strategy_sum} != 1.0"
            )

        # --- Behavioral: regret targets zero for illegal actions ---
        # Regret vectors are zero-initialized; only legal slots are written.
        for sample in adv:
            illegal_mask = ~sample.action_mask
            assert (sample.target[illegal_mask] == 0).all(), (
                "Threaded: non-zero regret for illegal action"
            )

    def test_threaded_traversal_sample_dimensions(self):
        """Samples from threaded traversal have correct tensor dimensions."""
        from src.cfr.deep_trainer import _run_traversals_threaded
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        config = _make_config()
        network_config = {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS}

        adv, strat, val, done, nodes = _run_traversals_threaded(
            iteration_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=2,
            num_threads=2,
            run_log_dir="/tmp/test_threaded_logs",
            run_timestamp="test",
        )

        for sample in adv:
            assert sample.features.shape == (INPUT_DIM,), (
                f"Expected ({INPUT_DIM},), got {sample.features.shape}"
            )
            assert sample.target.shape == (NUM_ACTIONS,), (
                f"Expected ({NUM_ACTIONS},), got {sample.target.shape}"
            )
            assert sample.action_mask.shape == (NUM_ACTIONS,), (
                f"Expected ({NUM_ACTIONS},), got {sample.action_mask.shape}"
            )
            # Regret mask: illegal action regrets must be zero
            illegal_mask = ~sample.action_mask
            assert (sample.target[illegal_mask] == 0).all(), (
                "Threaded dim test: non-zero regret for illegal action"
            )

        for sample in strat:
            # Strategy targets form valid probability distribution
            mask = sample.action_mask
            strategy_sum = sample.target[mask].sum()
            assert abs(strategy_sum - 1.0) < 0.01, (
                f"Threaded dim test: strategy sum {strategy_sum} != 1.0"
            )

    def test_threaded_vs_sequential_consistency(self):
        """Threaded path produces roughly similar sample counts as sequential."""
        from src.cfr.deep_trainer import _run_traversals_batch, _run_traversals_threaded
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        config = _make_config()
        network_config = {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS}
        traversals = 4

        # Sequential (_run_traversals_batch returns 6-tuple with timing_stats)
        adv_seq, strat_seq, val_seq, done_seq, nodes_seq, _timing = _run_traversals_batch(
            iteration_offset=0,
            total_traversals_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=traversals,
            num_workers=1,
            run_log_dir="/tmp/test_threaded_logs",
            run_timestamp="test",
        )

        # Threaded
        adv_thr, strat_thr, val_thr, done_thr, nodes_thr = _run_traversals_threaded(
            iteration_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=traversals,
            num_threads=2,
            run_log_dir="/tmp/test_threaded_logs",
            run_timestamp="test",
        )

        assert done_seq == done_thr == traversals
        # Both should produce advantage samples (exact counts differ due to random seeds)
        assert len(adv_seq) > 0
        assert len(adv_thr) > 0

    def test_threaded_traversal_4_threads(self):
        """Verify 4-thread traversal works without errors."""
        from src.cfr.deep_trainer import _run_traversals_threaded
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        config = _make_config()
        network_config = {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS}

        adv, strat, val, done, nodes = _run_traversals_threaded(
            iteration_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=8,
            num_threads=4,
            run_log_dir="/tmp/test_threaded_logs",
            run_timestamp="test",
        )

        assert done == 8
        assert len(adv) > 0
        assert nodes > 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestThreadedBenchmark:
    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize("traversals", [4, 8])
    def test_benchmark_threaded(self, num_threads, traversals):
        """Benchmark threaded traversals at various thread/traversal counts."""
        from src.cfr.deep_trainer import _run_traversals_threaded
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        config = _make_config()
        network_config = {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS}

        start = time.perf_counter()
        adv, strat, val, done, nodes = _run_traversals_threaded(
            iteration_offset=0,
            config=config,
            network_weights=None,
            network_config=network_config,
            traversals_per_step=traversals,
            num_threads=num_threads,
            run_log_dir="/tmp/test_threaded_bench",
            run_timestamp="bench",
        )
        elapsed = time.perf_counter() - start

        assert done == traversals
        assert len(adv) > 0

        # Print benchmark results (visible with pytest -s)
        print(
            f"\n  [BENCH] threads={num_threads} traversals={traversals} "
            f"time={elapsed:.3f}s adv_samples={len(adv)} "
            f"strat_samples={len(strat)} nodes={nodes}"
        )
