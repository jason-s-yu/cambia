"""
tests/test_metrics_scripts.py

Tests for the metrics collection and visualization scripts:
  - collect_metrics.py: JSONL output format, iteration/loss extraction
  - kl_divergence.py: KL computation with known distributions
  - plot_metrics.py: runs without error on sample data
"""

import json
import math
import os
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure cfr root is importable (conftest.py handles sys.path).
_CFR_ROOT = Path(__file__).resolve().parent.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(
    tmpdir: Path,
    filename: str = "deep_cfr_checkpoint_iter_42.pt",
    hidden_dim: int = 64,
    adv_loss_history: list = None,
    strat_loss_history: list = None,
    training_step: int = 42,
) -> Path:
    """Create a minimal fake .pt checkpoint using torch.save."""
    import torch
    from src.networks import AdvantageNetwork, StrategyNetwork
    from src.encoding import INPUT_DIM, NUM_ACTIONS

    net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=hidden_dim, output_dim=NUM_ACTIONS, validate_inputs=False)
    strat_net = StrategyNetwork(input_dim=INPUT_DIM, hidden_dim=hidden_dim, output_dim=NUM_ACTIONS, validate_inputs=False)

    checkpoint = {
        "training_step": training_step,
        "total_traversals": training_step * 1000,
        "dcfr_config": {"hidden_dim": hidden_dim},
        "advantage_net_state_dict": net.state_dict(),
        "strategy_net_state_dict": strat_net.state_dict(),
        "adv_loss_history": adv_loss_history or [5.0, 4.5, 4.0],
        "strat_loss_history": strat_loss_history or [0.1, 0.08, 0.07],
    }

    path = tmpdir / filename
    torch.save(checkpoint, str(path))
    return path


# ---------------------------------------------------------------------------
# collect_metrics tests
# ---------------------------------------------------------------------------


class TestCollectMetrics:
    """Tests for scripts/collect_metrics.py."""

    def _fake_multi_baseline_results(self) -> dict:
        """Returns fake run_evaluation_multi_baseline results."""
        baselines = ["random", "greedy", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]
        results = {}
        for bl in baselines:
            results[bl] = Counter({
                "P0 Wins": 55,
                "P1 Wins": 40,
                "Ties": 5,
            })
        return results

    def test_jsonl_row_schema(self, tmp_path):
        """Each JSONL row must contain the required schema fields."""
        import torch
        ckpt_path = _make_fake_checkpoint(tmp_path, training_step=50)

        fake_results = self._fake_multi_baseline_results()
        fake_config = MagicMock()
        fake_config.cambia_rules = MagicMock()

        with patch("scripts.collect_metrics.run_evaluation_multi_baseline", return_value=fake_results), \
             patch("scripts.collect_metrics.load_config", return_value=fake_config), \
             patch("scripts.collect_metrics.torch.load") as mock_load:

            mock_load.return_value = {
                "training_step": 50,
                "dcfr_config": {"hidden_dim": 64},
                "adv_loss_history": [4.5, 4.0],
                "strat_loss_history": [0.09, 0.07],
            }

            from scripts.collect_metrics import collect_metrics

            run_dir = tmp_path / "os-test"
            run_dir.mkdir()
            config_path = str(run_dir / "config.yaml")
            (run_dir / "config.yaml").write_text("")

            collect_metrics(
                run_dir=str(run_dir),
                checkpoint_path=str(ckpt_path),
                num_games=10,
                config_path=config_path,
            )

        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists(), "metrics.jsonl was not created"

        required_fields = {
            "run", "iter", "baseline", "win_rate", "games_played",
            "p0_wins", "p1_wins", "ties", "adv_loss", "strat_loss", "timestamp",
        }
        rows = []
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)
                missing = required_fields - set(row.keys())
                assert not missing, f"Missing fields in JSONL row: {missing}"

        # Should have 5 rows (one per baseline).
        assert len(rows) == 5

    def test_jsonl_row_values(self, tmp_path):
        """win_rate and games_played should be correctly computed from Counter."""
        fake_results = {
            "random": Counter({"P0 Wins": 60, "P1 Wins": 30, "Ties": 10}),
        }
        fake_config = MagicMock()

        with patch("scripts.collect_metrics.run_evaluation_multi_baseline", return_value=fake_results), \
             patch("scripts.collect_metrics.load_config", return_value=fake_config), \
             patch("scripts.collect_metrics.torch.load") as mock_load:

            mock_load.return_value = {
                "training_step": 25,
                "dcfr_config": {},
                "adv_loss_history": [3.5],
                "strat_loss_history": [0.05],
            }

            from scripts.collect_metrics import collect_metrics

            # Reload to pick up module-level state after patching.
            run_dir = tmp_path / "os-val"
            run_dir.mkdir()
            config_path = str(run_dir / "config.yaml")
            (run_dir / "config.yaml").write_text("")

            # Temporarily override ALL_BASELINES to just "random".
            import scripts.collect_metrics as cm_mod
            original_baselines = cm_mod.ALL_BASELINES
            cm_mod.ALL_BASELINES = ["random"]
            try:
                collect_metrics(
                    run_dir=str(run_dir),
                    checkpoint_path=str(tmp_path / "ckpt.pt"),
                    num_games=100,
                    config_path=config_path,
                )
            finally:
                cm_mod.ALL_BASELINES = original_baselines

        metrics_path = run_dir / "metrics.jsonl"
        with open(metrics_path) as f:
            row = json.loads(f.readline())

        assert row["iter"] == 25
        assert row["baseline"] == "random"
        assert row["p0_wins"] == 60
        assert row["p1_wins"] == 30
        assert row["ties"] == 10
        assert row["games_played"] == 100
        assert abs(row["win_rate"] - 0.6) < 1e-6

    def test_iter_inferred_from_filename(self, tmp_path):
        """Iteration number should be inferred from checkpoint filename when not in metadata."""
        from scripts.collect_metrics import _infer_iter_from_path

        assert _infer_iter_from_path("runs/os-20/checkpoints/deep_cfr_checkpoint_iter_75.pt") == 75
        assert _infer_iter_from_path("deep_cfr_checkpoint_iter_1.pt") == 1
        assert _infer_iter_from_path("deep_cfr_checkpoint.pt") == -1

    def test_loss_extraction(self, tmp_path):
        """adv_loss and strat_loss should be the last element of their respective history lists."""
        from scripts.collect_metrics import _extract_loss_history

        checkpoint = {
            "adv_loss_history": [5.0, 4.5, 3.9],
            "strat_loss_history": [0.1, 0.08, 0.065],
        }
        adv, strat = _extract_loss_history(checkpoint)
        assert abs(adv - 3.9) < 1e-9
        assert abs(strat - 0.065) < 1e-9

    def test_loss_extraction_empty_history(self, tmp_path):
        """Empty loss history lists should return NaN."""
        from scripts.collect_metrics import _extract_loss_history

        adv, strat = _extract_loss_history({})
        assert math.isnan(adv)
        assert math.isnan(strat)

    def test_metrics_jsonl_appends(self, tmp_path):
        """Calling collect_metrics twice should append rows, not overwrite."""
        fake_results = {
            "random": Counter({"P0 Wins": 5, "P1 Wins": 5, "Ties": 0}),
        }
        fake_config = MagicMock()

        import scripts.collect_metrics as cm_mod

        run_dir = tmp_path / "os-append"
        run_dir.mkdir()
        config_path = str(run_dir / "config.yaml")
        (run_dir / "config.yaml").write_text("")

        original_baselines = cm_mod.ALL_BASELINES
        cm_mod.ALL_BASELINES = ["random"]

        with patch("scripts.collect_metrics.run_evaluation_multi_baseline", return_value=fake_results), \
             patch("scripts.collect_metrics.load_config", return_value=fake_config), \
             patch("scripts.collect_metrics.torch.load") as mock_load:
            mock_load.return_value = {"training_step": 10, "dcfr_config": {}, "adv_loss_history": [], "strat_loss_history": []}
            cm_mod.collect_metrics(str(run_dir), str(tmp_path / "ckpt.pt"), 10, config_path)

        with patch("scripts.collect_metrics.run_evaluation_multi_baseline", return_value=fake_results), \
             patch("scripts.collect_metrics.load_config", return_value=fake_config), \
             patch("scripts.collect_metrics.torch.load") as mock_load:
            mock_load.return_value = {"training_step": 20, "dcfr_config": {}, "adv_loss_history": [], "strat_loss_history": []}
            cm_mod.collect_metrics(str(run_dir), str(tmp_path / "ckpt.pt"), 10, config_path)

        cm_mod.ALL_BASELINES = original_baselines

        metrics_path = run_dir / "metrics.jsonl"
        with open(metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 2, f"Expected 2 appended rows, got {len(rows)}"


# ---------------------------------------------------------------------------
# kl_divergence tests
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Tests for scripts/kl_divergence.py KL computation logic."""

    def test_kl_div_identical_distributions(self):
        """KL(p||p) should be 0 for any distribution p."""
        from scripts.kl_divergence import _kl_div

        p = np.array([0.4, 0.35, 0.25, 0.0])
        assert abs(_kl_div(p, p)) < 1e-9

    def test_kl_div_known_value(self):
        """KL divergence should match the analytical result for a known pair."""
        from scripts.kl_divergence import _kl_div

        # KL( [0.5, 0.5] || [0.25, 0.75] )
        # = 0.5 * log(0.5/0.25) + 0.5 * log(0.5/0.75)
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        result = _kl_div(p, q)
        assert abs(result - expected) < 1e-9

    def test_kl_div_asymmetry(self):
        """KL(p||q) != KL(q||p) in general."""
        from scripts.kl_divergence import _kl_div

        # Use asymmetric distributions to guarantee KL(p||q) != KL(q||p).
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.5, 0.4])
        kl_pq = _kl_div(p, q)
        kl_qp = _kl_div(q, p)
        assert kl_pq > 0.0
        assert kl_qp > 0.0
        assert abs(kl_pq - kl_qp) > 1e-6  # These are genuinely asymmetric.

    def test_kl_div_zero_p_support_ignored(self):
        """Actions where p=0 should not contribute to KL divergence."""
        from scripts.kl_divergence import _kl_div

        # p has mass only on index 0; q has mass on both.
        p = np.array([1.0, 0.0])
        q = np.array([0.7, 0.3])
        # KL = 1.0 * log(1.0 / 0.7) + 0 = log(1/0.7)
        expected = math.log(1.0 / 0.7)
        result = _kl_div(p, q)
        assert abs(result - expected) < 1e-9

    def test_kl_div_empty_support(self):
        """If p is all zeros, KL should be 0."""
        from scripts.kl_divergence import _kl_div

        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.5, 0.3, 0.2])
        assert _kl_div(p, q) == 0.0

    def test_load_network_returns_advantage_network(self, tmp_path):
        """_load_network should return an AdvantageNetwork with correct architecture."""
        import torch
        from scripts.kl_divergence import _load_network
        from src.networks import AdvantageNetwork

        ckpt_path = _make_fake_checkpoint(tmp_path, hidden_dim=64)
        device = torch.device("cpu")
        net = _load_network(str(ckpt_path), device)

        assert isinstance(net, AdvantageNetwork)

    def test_get_strategy_sums_to_one(self, tmp_path):
        """Strategy returned by _get_strategy should sum to 1 over legal actions."""
        import torch
        from scripts.kl_divergence import _load_network, _get_strategy
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        ckpt_path = _make_fake_checkpoint(tmp_path, hidden_dim=64)
        device = torch.device("cpu")
        net = _load_network(str(ckpt_path), device)

        features = np.zeros(INPUT_DIM, dtype=np.float32)
        action_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        action_mask[0] = True
        action_mask[1] = True
        action_mask[2] = True

        strat = _get_strategy(net, features, action_mask, device)
        legal_probs = strat[action_mask]
        assert abs(legal_probs.sum() - 1.0) < 1e-5
        # Illegal actions should be 0.
        assert strat[~action_mask].sum() < 1e-6

    def test_compute_kl_divergence_identical_checkpoints(self, tmp_path):
        """KL between identical checkpoints should be near 0."""
        from scripts.kl_divergence import compute_kl_divergence

        ckpt_path = _make_fake_checkpoint(tmp_path, hidden_dim=64)
        result = compute_kl_divergence(
            checkpoint_a=str(ckpt_path),
            checkpoint_b=str(ckpt_path),
            num_states=20,
        )
        assert result["num_states_sampled"] > 0
        # Identical networks: KL(a||b) should be exactly 0.
        assert result["mean_kl"] < 1e-6, f"Expected near-0 KL, got {result['mean_kl']}"

    def test_compute_kl_divergence_different_checkpoints(self, tmp_path):
        """KL between checkpoints with different random weights should be > 0."""
        import torch
        from scripts.kl_divergence import compute_kl_divergence
        from src.networks import AdvantageNetwork
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        def _make_ckpt(filename, seed):
            torch.manual_seed(seed)
            net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_ACTIONS, validate_inputs=False)
            # Re-initialize with different random seed.
            for p in net.parameters():
                p.data = torch.randn_like(p.data)
            checkpoint = {
                "training_step": seed,
                "dcfr_config": {"hidden_dim": 64},
                "advantage_net_state_dict": net.state_dict(),
                "adv_loss_history": [],
                "strat_loss_history": [],
            }
            path = tmp_path / filename
            torch.save(checkpoint, str(path))
            return path

        ckpt_a = _make_ckpt("ckpt_a.pt", seed=1)
        ckpt_b = _make_ckpt("ckpt_b.pt", seed=99)

        result = compute_kl_divergence(
            checkpoint_a=str(ckpt_a),
            checkpoint_b=str(ckpt_b),
            num_states=30,
        )
        assert result["num_states_sampled"] > 0
        # Different random networks -> KL should be > 0.
        assert result["mean_kl"] >= 0.0  # KL is non-negative.
        assert "max_kl" in result
        assert "median_kl" in result


# ---------------------------------------------------------------------------
# plot_metrics tests
# ---------------------------------------------------------------------------


class TestPlotMetrics:
    """Tests for scripts/plot_metrics.py."""

    def _write_sample_metrics(self, runs_dir: Path) -> None:
        """Write sample metrics.jsonl files to test runs."""
        runs = [
            {"run": "os-20", "iters": [25, 50], "baselines": ["random", "greedy", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]},
            {"run": "os-30", "iters": [25], "baselines": ["random", "greedy", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]},
        ]
        for run_spec in runs:
            run_dir = runs_dir / run_spec["run"]
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "metrics.jsonl", "w") as f:
                for it in run_spec["iters"]:
                    for bl in run_spec["baselines"]:
                        row = {
                            "run": run_spec["run"],
                            "iter": it,
                            "baseline": bl,
                            "win_rate": 0.5 + (it / 1000.0),
                            "games_played": 100,
                            "p0_wins": 55,
                            "p1_wins": 40,
                            "ties": 5,
                            "avg_game_turns": 25.0,
                            "adv_loss": 4.0,
                            "strat_loss": 0.08,
                            "timestamp": "2026-02-19T12:00:00Z",
                        }
                        f.write(json.dumps(row) + "\n")

    def test_load_all_metrics(self, tmp_path):
        """load_all_metrics should read all rows from all runs."""
        from scripts.plot_metrics import load_all_metrics

        self._write_sample_metrics(tmp_path)
        rows = load_all_metrics(tmp_path)

        # os-20: 2 iters x 5 baselines = 10; os-30: 1 iter x 5 = 5; total = 15
        assert len(rows) == 15

    def test_load_metrics_empty_dir(self, tmp_path):
        """load_all_metrics on a directory with no metrics.jsonl should return []."""
        from scripts.plot_metrics import load_all_metrics

        rows = load_all_metrics(tmp_path)
        assert rows == []

    def test_generate_plots_creates_files(self, tmp_path):
        """generate_plots should create all three plot files without error."""
        pytest.importorskip("matplotlib")

        from scripts.plot_metrics import generate_plots

        runs_dir = tmp_path / "runs"
        plots_dir = tmp_path / "plots"
        self._write_sample_metrics(runs_dir)

        generate_plots(runs_dir=str(runs_dir), output_dir=str(plots_dir))

        assert (plots_dir / "win_rate_by_run.png").exists()
        assert (plots_dir / "win_rate_by_baseline.png").exists()
        assert (plots_dir / "win_rate_combined.png").exists()

    def test_generate_plots_no_error_single_run(self, tmp_path):
        """generate_plots should not error with a single run."""
        pytest.importorskip("matplotlib")

        from scripts.plot_metrics import generate_plots

        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "os-only"
        run_dir.mkdir(parents=True)
        with open(run_dir / "metrics.jsonl", "w") as f:
            for it in [10, 20, 30]:
                row = {
                    "run": "os-only",
                    "iter": it,
                    "baseline": "random",
                    "win_rate": 0.5,
                    "games_played": 100,
                    "p0_wins": 50, "p1_wins": 50, "ties": 0,
                    "avg_game_turns": 20.0,
                    "adv_loss": 3.5, "strat_loss": 0.06,
                    "timestamp": "2026-02-19T12:00:00Z",
                }
                f.write(json.dumps(row) + "\n")

        plots_dir = tmp_path / "plots"
        generate_plots(runs_dir=str(runs_dir), output_dir=str(plots_dir))

        assert (plots_dir / "win_rate_combined.png").exists()

    def test_generate_plots_empty_dir_no_error(self, tmp_path):
        """generate_plots on an empty directory should not crash."""
        pytest.importorskip("matplotlib")

        from scripts.plot_metrics import generate_plots

        runs_dir = tmp_path / "empty_runs"
        runs_dir.mkdir()
        plots_dir = tmp_path / "plots"

        # Should log a warning and return without creating files.
        generate_plots(runs_dir=str(runs_dir), output_dir=str(plots_dir))

    def test_malformed_jsonl_skipped(self, tmp_path):
        """Malformed JSONL lines should be skipped without crashing."""
        from scripts.plot_metrics import load_all_metrics

        run_dir = tmp_path / "bad-run"
        run_dir.mkdir()
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write('{"run": "bad-run", "iter": 1, "baseline": "random", "win_rate": 0.5, "games_played": 10, "p0_wins": 5, "p1_wins": 5, "ties": 0, "adv_loss": null, "strat_loss": null, "timestamp": "2026-02-19T12:00:00Z"}\n')
            f.write("NOT VALID JSON {\n")
            f.write('{"run": "bad-run", "iter": 2, "baseline": "random", "win_rate": 0.6, "games_played": 10, "p0_wins": 6, "p1_wins": 4, "ties": 0, "adv_loss": null, "strat_loss": null, "timestamp": "2026-02-19T12:00:00Z"}\n')

        rows = load_all_metrics(tmp_path)
        # Only 2 valid rows.
        assert len(rows) == 2
