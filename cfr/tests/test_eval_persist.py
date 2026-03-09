"""
tests/test_eval_persist.py

Tests for persist_eval_results() and CLI run-dir detection logic.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_CFR_ROOT = Path(__file__).resolve().parent.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))


class TestPersistEvalResults:
    def test_writes_metrics_jsonl(self, tmp_path):
        """persist_eval_results creates metrics.jsonl with correct schema."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 70, "P1 Wins": 25, "Ties": 5})
        results.stats = {
            "avg_game_turns": 15.2,
            "avg_score_margin": 8.1,
            "t1_cambia_rate": 0.35,
        }

        persist_eval_results(
            run_dir=str(run_dir),
            iteration=100,
            results_map={"random_no_cambia": results},
        )

        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists()
        with open(metrics_path) as f:
            row = json.loads(f.readline())
        assert row["run"] == "test-run"
        assert row["iter"] == 100
        assert row["baseline"] == "random_no_cambia"
        assert abs(row["win_rate"] - 0.70) < 0.01
        assert row["games_played"] == 100
        assert row["p0_wins"] == 70
        assert row["p1_wins"] == 25
        assert row["ties"] == 5
        assert row["timestamp"]  # non-empty
        assert row["ci_low"] is not None
        assert row["ci_high"] is not None
        assert abs(row["avg_game_turns"] - 15.2) < 0.01
        assert abs(row["t1_cambia_rate"] - 0.35) < 0.001
        assert abs(row["avg_score_margin"] - 8.1) < 0.01

    def test_appends_not_overwrites(self, tmp_path):
        """Multiple calls append, not overwrite."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}

        persist_eval_results(str(run_dir), 100, {"random": results})
        persist_eval_results(str(run_dir), 200, {"random": results})

        with open(run_dir / "metrics.jsonl") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 2

    def test_creates_evaluations_dir(self, tmp_path):
        """persist_eval_results creates evaluations/iter_N/ directory."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}

        persist_eval_results(str(run_dir), 100, {"random": results})
        assert (run_dir / "evaluations" / "iter_100").is_dir()

    def test_handles_empty_results(self, tmp_path):
        """persist_eval_results handles empty results gracefully."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        # Should not crash
        persist_eval_results(str(run_dir), 100, {})

    def test_multiple_baselines(self, tmp_path):
        """persist_eval_results writes one row per baseline."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        r1 = Counter({"P0 Wins": 70, "P1 Wins": 30})
        r1.stats = {}
        r2 = Counter({"P0 Wins": 60, "P1 Wins": 40})
        r2.stats = {}

        persist_eval_results(str(run_dir), 100, {"random": r1, "greedy": r2})

        with open(run_dir / "metrics.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 2
        baselines = {r["baseline"] for r in rows}
        assert baselines == {"random", "greedy"}

    def test_custom_run_name(self, tmp_path):
        """run_name param overrides directory basename."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}

        persist_eval_results(
            str(run_dir), 100, {"random": results}, run_name="custom-name"
        )

        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert row["run"] == "custom-name"

    def test_loss_values_passed_through(self, tmp_path):
        """adv_loss and strat_loss params appear in JSONL output."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}

        persist_eval_results(
            str(run_dir),
            100,
            {"random": results},
            adv_loss=0.54,
            strat_loss=1.13,
        )

        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert abs(row["adv_loss"] - 0.54) < 0.001
        assert abs(row["strat_loss"] - 1.13) < 0.001

    def test_nan_loss_becomes_none(self, tmp_path):
        """NaN loss values are written as null in JSONL."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 50})
        results.stats = {}

        persist_eval_results(
            str(run_dir),
            100,
            {"random": results},
            adv_loss=float("nan"),
            strat_loss=float("nan"),
        )

        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert row["adv_loss"] is None
        assert row["strat_loss"] is None

    def test_maxturnties_included_in_ties(self, tmp_path):
        """MaxTurnTies are added to ties count."""
        from src.evaluate_agents import persist_eval_results

        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        results = Counter({"P0 Wins": 50, "P1 Wins": 40, "Ties": 5, "MaxTurnTies": 5})
        results.stats = {}

        persist_eval_results(str(run_dir), 100, {"random": results})

        with open(run_dir / "metrics.jsonl") as f:
            row = json.loads(f.readline())
        assert row["ties"] == 10
        assert row["games_played"] == 100


class TestRunDirDetection:
    """Test that CLI correctly detects run dir from checkpoint path."""

    def test_checkpoint_in_run_dir_detected(self, tmp_path):
        """When checkpoint is at runs/foo/checkpoints/bar.pt, run_dir = runs/foo/."""
        run_dir = tmp_path / "runs" / "my-run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text("num_players: 2\n")

        ckpt = ckpt_dir / "deep_cfr_checkpoint_iter_100.pt"
        ckpt.touch()

        # Test the detection logic
        p = Path(str(ckpt))
        detected = None
        if p.parent.name == "checkpoints" and (p.parent.parent / "config.yaml").exists():
            detected = p.parent.parent
        assert detected == run_dir

    def test_checkpoint_not_in_run_dir(self, tmp_path):
        """When checkpoint is in a random directory, run_dir is not detected."""
        ckpt = tmp_path / "some_checkpoint.pt"
        ckpt.touch()

        p = Path(str(ckpt))
        detected = None
        if p.parent.name == "checkpoints" and (p.parent.parent / "config.yaml").exists():
            detected = p.parent.parent
        assert detected is None

    def test_extract_iter_from_checkpoint_name(self):
        """Iteration number correctly extracted from checkpoint filenames."""
        patterns = [
            ("deep_cfr_checkpoint_iter_100.pt", 100),
            ("sog_checkpoint_sog_epoch_50.pt", 50),
            ("gtcfr_checkpoint_epoch_200.pt", 200),
            ("rebel_checkpoint_iter_1.pt", 1),
        ]
        for filename, expected in patterns:
            m = re.search(r"(?:epoch|iter)_(\d+)", filename)
            assert m is not None, f"No match for {filename}"
            assert int(m.group(1)) == expected


class TestCheckpointDiscovery:
    """Test checkpoint glob and selection logic used by run-dir mode."""

    def test_latest_selects_highest_epoch(self, tmp_path):
        """--latest picks the checkpoint with the highest epoch number."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for i in [50, 100, 200, 150]:
            (ckpt_dir / f"sog_checkpoint_sog_epoch_{i}.pt").touch()

        all_ckpts = sorted(ckpt_dir.glob("*sog*epoch_*.pt"))

        def extract_num(p):
            m = re.search(r"(?:epoch|iter)_(\d+)", p.name)
            return int(m.group(1)) if m else 0

        all_ckpts.sort(key=extract_num)
        latest_ckpt = all_ckpts[-1]
        assert "epoch_200" in latest_ckpt.name

    def test_epoch_selects_specific(self, tmp_path):
        """--epoch N picks the checkpoint matching that epoch."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for i in [50, 100, 200]:
            (ckpt_dir / f"gtcfr_checkpoint_epoch_{i}.pt").touch()

        target_epoch = 100
        all_ckpts = sorted(ckpt_dir.glob("*gtcfr*epoch_*.pt"))
        matches = [
            p for p in all_ckpts
            if f"epoch_{target_epoch}.pt" in p.name
        ]
        assert len(matches) == 1
        assert "epoch_100" in matches[0].name

    def test_iter_pattern_fallback(self, tmp_path):
        """Falls back to iter_ pattern when epoch_ not found."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for i in [25, 50, 75]:
            (ckpt_dir / f"deep_cfr_checkpoint_iter_{i}.pt").touch()

        # First try epoch pattern (no matches)
        found = sorted(ckpt_dir.glob("*deep_cfr*epoch_*.pt"))
        assert len(found) == 0
        # Then try iter pattern
        found = sorted(ckpt_dir.glob("*deep_cfr*iter_*.pt"))
        assert len(found) == 3
