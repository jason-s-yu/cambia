"""
tests/test_psro_integration.py

Integration tests for PSRO wiring into trainer, config, CLI, and conftest stub.
"""

import sys
import types
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPSROConfigFields:
    """PSRO fields in config.py DeepCfrConfig dataclass."""

    def test_config_defaults(self):
        """PSRO config fields have correct defaults."""
        from src.config import DeepCfrConfig

        cfg = DeepCfrConfig()
        assert cfg.use_psro is False
        assert cfg.psro_population_size == 15
        assert cfg.psro_eval_games == 200
        assert cfg.psro_checkpoint_interval == 50
        assert cfg.psro_heuristic_types == "random,greedy,memory_heuristic"

    def test_config_from_yaml(self):
        """PSRO fields parsed from YAML config."""
        import yaml
        from src.config import load_config

        yaml_content = {
            "deep_cfr": {
                "use_psro": True,
                "psro_population_size": 10,
                "psro_eval_games": 100,
                "psro_checkpoint_interval": 25,
                "psro_heuristic_types": "random,greedy",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg is not None
            assert cfg.deep_cfr.use_psro is True
            assert cfg.deep_cfr.psro_population_size == 10
            assert cfg.deep_cfr.psro_eval_games == 100
            assert cfg.deep_cfr.psro_checkpoint_interval == 25
            assert cfg.deep_cfr.psro_heuristic_types == "random,greedy"
        finally:
            os.unlink(tmp_path)

    def test_config_defaults_in_yaml(self):
        """PSRO defaults are used when YAML does not include PSRO section."""
        import yaml
        from src.config import load_config

        yaml_content = {"deep_cfr": {"hidden_dim": 128}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg is not None
            assert cfg.deep_cfr.use_psro is False
            assert cfg.deep_cfr.psro_population_size == 15
        finally:
            os.unlink(tmp_path)


class TestPSROTrainerConfig:
    """PSRO fields in DeepCFRConfig (internal trainer config)."""

    def test_trainer_config_defaults(self):
        """Trainer DeepCFRConfig has PSRO fields with correct defaults."""
        from src.cfr.deep_trainer import DeepCFRConfig

        dcfr = DeepCFRConfig()
        assert dcfr.use_psro is False
        assert dcfr.psro_population_size == 15
        assert dcfr.psro_eval_games == 200
        assert dcfr.psro_checkpoint_interval == 50
        assert dcfr.psro_heuristic_types == "random,greedy,memory_heuristic"

    def test_from_yaml_config_forwards_psro_fields(self):
        """from_yaml_config correctly forwards PSRO fields from Config."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.config import Config, DeepCfrConfig as YamlCfr, CambiaRulesConfig

        mock_deep_cfg = YamlCfr(
            use_psro=True,
            psro_population_size=8,
            psro_eval_games=50,
            psro_checkpoint_interval=20,
            psro_heuristic_types="random",
        )
        mock_config = MagicMock(spec=Config)
        mock_config.deep_cfr = mock_deep_cfg
        mock_config.cambia_rules = CambiaRulesConfig()
        mock_config.cfr_training = MagicMock()
        mock_config.cfr_training.num_iterations = 100

        dcfr = DeepCFRConfig.from_yaml_config(mock_config)
        assert dcfr.use_psro is True
        assert dcfr.psro_population_size == 8
        assert dcfr.psro_eval_games == 50
        assert dcfr.psro_checkpoint_interval == 20
        assert dcfr.psro_heuristic_types == "random"

    def test_from_yaml_config_psro_override(self):
        """CLI override can force use_psro=True."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.config import Config, DeepCfrConfig as YamlCfr, CambiaRulesConfig

        mock_deep_cfg = YamlCfr(use_psro=False)
        mock_config = MagicMock(spec=Config)
        mock_config.deep_cfr = mock_deep_cfg
        mock_config.cambia_rules = CambiaRulesConfig()
        mock_config.cfr_training = MagicMock()
        mock_config.cfr_training.num_iterations = 100

        dcfr = DeepCFRConfig.from_yaml_config(mock_config, use_psro=True)
        assert dcfr.use_psro is True


class TestPSROTrainerIntegration:
    """PSRO oracle wiring into DeepCFRTrainer."""

    def _make_trainer(self, use_psro=False, **kwargs):
        """Helper to construct a minimal DeepCFRTrainer."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.config import Config, CambiaRulesConfig

        mock_config = MagicMock(spec=Config)
        mock_config.cambia_rules = CambiaRulesConfig()
        mock_config.persistence = MagicMock()
        mock_config.persistence.agent_data_save_path = "/tmp/test_ckpt.pt"
        mock_config.cfr_training = MagicMock()
        mock_config.cfr_training.num_iterations = 10
        mock_config.cfr_training.num_workers = 1

        dcfr_config = DeepCFRConfig(
            device="cpu",
            pipeline_training=False,
            use_amp=False,
            use_psro=use_psro,
            **kwargs,
        )

        trainer = DeepCFRTrainer(config=mock_config, deep_cfr_config=dcfr_config)
        return trainer

    def test_trainer_no_oracle_by_default(self):
        """When use_psro=False (default), no oracle is created."""
        trainer = self._make_trainer(use_psro=False)
        assert trainer._psro_oracle is None

    def test_trainer_creates_oracle_when_enabled(self):
        """When use_psro=True, trainer initializes a PSROOracle."""
        from src.cfr.psro import PSROOracle

        trainer = self._make_trainer(use_psro=True, psro_population_size=5)
        assert trainer._psro_oracle is not None
        assert isinstance(trainer._psro_oracle, PSROOracle)

    def test_trainer_oracle_population_size(self):
        """Oracle respects psro_population_size from config."""
        trainer = self._make_trainer(use_psro=True, psro_population_size=7)
        assert trainer._psro_oracle.max_checkpoints == 7

    def test_trainer_oracle_heuristic_types(self):
        """Oracle receives parsed heuristic types."""
        trainer = self._make_trainer(
            use_psro=True, psro_heuristic_types="random,greedy"
        )
        heuristic_types = [
            m.agent_type for m in trainer._psro_oracle._heuristics
        ]
        assert "random" in heuristic_types
        assert "greedy" in heuristic_types
        assert len(heuristic_types) == 2

    def test_trainer_oracle_heuristic_types_default(self):
        """Oracle uses 3 default heuristics."""
        trainer = self._make_trainer(use_psro=True)
        assert len(trainer._psro_oracle._heuristics) == 3


class TestPSROCheckpointState:
    """PSRO state save/load via PSROOracle."""

    def test_save_and_load_state(self):
        """save_state and load_state round-trip correctly."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
        oracle.add_checkpoint("/tmp/fake_ckpt.pt", iteration=10)
        oracle.add_checkpoint("/tmp/fake_ckpt2.pt", iteration=20)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            oracle.save_state(tmp_path)
            # Load into a fresh oracle
            oracle2 = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
            oracle2.load_state(tmp_path)

            assert oracle2.checkpoint_count == 2
            assert oracle2._checkpoints[0].iteration == 10
            assert oracle2._checkpoints[1].iteration == 20
            assert oracle2.max_checkpoints == 5
        finally:
            os.unlink(tmp_path)

    def test_save_state_json_structure(self):
        """save_state produces valid JSON with expected keys."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=3, heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/tmp/ckpt.pt", iteration=5)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            oracle.save_state(tmp_path)
            with open(tmp_path) as f:
                state = json.load(f)

            assert "max_checkpoints" in state
            assert "checkpoints" in state
            assert "heuristics" in state
            assert state["max_checkpoints"] == 3
            assert len(state["checkpoints"]) == 1
            assert state["checkpoints"][0]["iteration"] == 5
        finally:
            os.unlink(tmp_path)


class TestPSROOracle:
    """Unit tests for PSROOracle population management."""

    def test_add_checkpoint_increases_size(self):
        """Adding a checkpoint increases population size."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
        initial_size = oracle.size
        oracle.add_checkpoint("/tmp/ckpt.pt", iteration=1)
        assert oracle.size == initial_size + 1

    def test_rolling_window_evicts_oldest(self):
        """When max exceeded, oldest checkpoint is evicted (FIFO)."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=2, heuristic_types=[])
        oracle.add_checkpoint("/tmp/ckpt1.pt", iteration=1)
        oracle.add_checkpoint("/tmp/ckpt2.pt", iteration=2)
        evicted = oracle.add_checkpoint("/tmp/ckpt3.pt", iteration=3)

        assert evicted == "/tmp/ckpt1.pt"
        assert oracle.checkpoint_count == 2
        iters = [m.iteration for m in oracle._checkpoints]
        assert 1 not in iters
        assert 2 in iters
        assert 3 in iters

    def test_sample_opponents_returns_correct_count(self):
        """sample_opponents returns requested number of opponents."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/tmp/ckpt.pt", iteration=1)
        opponents = oracle.sample_opponents(3)
        assert len(opponents) == 3

    def test_population_includes_heuristics(self):
        """Population always includes heuristic members."""
        from src.cfr.psro import PSROOracle

        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random", "greedy"])
        assert any(m.is_heuristic for m in oracle.population)
        assert len([m for m in oracle.population if m.is_heuristic]) == 2

    def test_plackett_luce_ratings_shape(self):
        """_plackett_luce_ratings returns array of correct length."""
        from src.cfr.psro import PSROOracle, PopulationMember
        import numpy as np

        oracle = PSROOracle(max_checkpoints=5, heuristic_types=[])
        pop = [PopulationMember(path="", iteration=i) for i in range(3)]
        # Simple orderings: player 0 always wins
        orderings = [[0, 1, 2], [0, 2, 1], [0, 1, 2]]
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert len(ratings) == 3
        # Player 0 should have highest rating
        assert ratings[0] == max(ratings)


class TestPSROCLI:
    """PSRO CLI command registration and basic invocation."""

    def test_psro_command_exists(self):
        """The 'psro' train subcommand is registered in the CLI."""
        from typer.testing import CliRunner
        from src.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "psro" in result.output

    def test_psro_subcommand_help(self):
        """psro subcommand shows help text."""
        from typer.testing import CliRunner
        from src.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "psro", "--help"])
        assert result.exit_code == 0
        assert "psro" in result.output.lower() or "PSRO" in result.output


class TestPSROConftestStub:
    """PSRO fields in conftest _DeepCfrConfig stub."""

    def test_stub_has_psro_fields(self):
        """The conftest stub DeepCfrConfig has all PSRO fields."""
        from src.config import DeepCfrConfig

        cfg = DeepCfrConfig()
        # These must all exist (not raise AttributeError)
        assert hasattr(cfg, "use_psro")
        assert hasattr(cfg, "psro_population_size")
        assert hasattr(cfg, "psro_eval_games")
        assert hasattr(cfg, "psro_checkpoint_interval")
        assert hasattr(cfg, "psro_heuristic_types")

    def test_stub_psro_defaults(self):
        """The conftest stub has correct PSRO default values."""
        from src.config import DeepCfrConfig

        cfg = DeepCfrConfig()
        assert cfg.use_psro is False
        assert cfg.psro_population_size == 15
        assert cfg.psro_eval_games == 200
        assert cfg.psro_checkpoint_interval == 50
        assert cfg.psro_heuristic_types == "random,greedy,memory_heuristic"
