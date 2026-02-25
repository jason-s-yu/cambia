"""
tests/conftest.py

Shared fixtures and bootstrap logic for all tests.

The project's config.py is currently incomplete (missing dataclass import
and several Config sub-classes). We inject a minimal stub before any
src.* imports so that modules like agent_state and encoding can be loaded.
"""

import sys
import types
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Stub out src.config ---
# Only inject the stub if config hasn't been loaded yet or is broken.
_config_mod = sys.modules.get("src.config")
if _config_mod is None or not hasattr(_config_mod, "Config"):
    _config_stub = types.ModuleType("src.config")

    class _StubConfig:
        """Minimal placeholder for Config and its sub-classes."""
        pass

    from dataclasses import dataclass as _dataclass, field as _field

    # CambiaRulesConfig needs real defaults because CambiaGameState reads them
    @_dataclass
    class _CambiaRulesConfig:
        """Stub for CambiaRulesConfig with real game defaults."""
        allowDrawFromDiscardPile: bool = False
        allowReplaceAbilities: bool = False
        snapRace: bool = False
        penaltyDrawCount: int = 2
        use_jokers: int = 2
        cards_per_player: int = 4
        initial_view_count: int = 2
        cambia_allowed_round: int = 0
        allowOpponentSnapping: bool = False
        max_game_turns: int = 300
        lockCallerHand: bool = True

    @_dataclass
    class _DeepCfrConfig:
        """Stub for DeepCfrConfig with real defaults."""
        hidden_dim: int = 256
        dropout: float = 0.1
        learning_rate: float = 1e-3
        batch_size: int = 2048
        train_steps_per_iteration: int = 4000
        alpha: float = 1.5
        traversals_per_step: int = 1000
        advantage_buffer_capacity: int = 2_000_000
        strategy_buffer_capacity: int = 2_000_000
        save_interval: int = 10
        device: str = "cpu"
        sampling_method: str = "outcome"
        exploration_epsilon: float = 0.6
        engine_backend: str = "python"
        es_validation_interval: int = 10
        es_validation_depth: int = 10
        es_validation_traversals: int = 1000
        pipeline_training: bool = True
        use_amp: bool = False
        use_compile: bool = False
        num_traversal_threads: int = 1
        validate_inputs: bool = True
        traversal_depth_limit: int = 0
        # ESCHER fields
        traversal_method: str = "outcome"
        value_hidden_dim: int = 512
        value_learning_rate: float = 1e-3
        value_buffer_capacity: int = 2_000_000
        batch_counterfactuals: bool = True
        # DEPRECATED: ReBeL fields retained for checkpoint backward compat only.
        # ReBeL/PBS subgame solving is mathematically unsound for N-player FFA with continuous beliefs.
        rebel_subgame_depth: int = 4
        rebel_cfr_iterations: int = 200
        rebel_value_hidden_dim: int = 1024
        rebel_policy_hidden_dim: int = 512
        rebel_value_learning_rate: float = 1e-3
        rebel_policy_learning_rate: float = 1e-3
        rebel_value_buffer_capacity: int = 500_000
        rebel_policy_buffer_capacity: int = 500_000
        rebel_games_per_epoch: int = 100
        rebel_epochs: int = 500

        # SD-CFR fields
        use_sd_cfr: bool = False
        sd_cfr_max_snapshots: int = 200
        sd_cfr_snapshot_weighting: str = "linear"
        num_hidden_layers: int = 3
        use_residual: bool = True
        use_ema: bool = True  # EMA serving weights for O(1) SD-CFR inference
        enable_traversal_profiling: bool = False
        profiling_jsonl_path: str = ""
        profile_step: Optional[int] = None
        encoding_mode: str = "legacy"  # "legacy" (222-dim) or "ep_pbs" (200-dim)
        # Memory archetype fields
        memory_archetype: str = "perfect"
        memory_decay_lambda: float = 0.1
        memory_capacity: int = 3

        # N-player fields
        num_players: int = 2
        # QRE fields
        qre_lambda_start: float = 0.5
        qre_lambda_end: float = 0.05
        qre_anneal_fraction: float = 0.6
        # PSRO fields
        use_psro: bool = False
        psro_population_size: int = 15
        psro_eval_games: int = 200
        psro_checkpoint_interval: int = 50
        psro_heuristic_types: str = "random,greedy,memory_heuristic"

    def _load_config(path: str):
        """Stub load_config: delegate to the real load_config implementation."""
        try:
            # Temporarily remove stub to import real module
            _saved = sys.modules.pop("src.config", None)
            import importlib
            _real_mod = importlib.import_module("src.config")
            result = _real_mod.load_config(path)
            # Restore stub
            if _saved is not None:
                sys.modules["src.config"] = _saved
            return result
        except Exception:
            return None

    _config_stub.Config = _StubConfig
    _config_stub.CambiaRulesConfig = _CambiaRulesConfig
    _config_stub.CfrTrainingConfig = _StubConfig
    _config_stub.AgentParamsConfig = _StubConfig
    _config_stub.ApiConfig = _StubConfig
    _config_stub.SystemConfig = _StubConfig
    _config_stub.CfrPlusParamsConfig = _StubConfig
    _config_stub.PersistenceConfig = _StubConfig
    _config_stub.LoggingConfig = _StubConfig
    _config_stub.AgentsConfig = _StubConfig
    _config_stub.AnalysisConfig = _StubConfig
    _config_stub.DeepCfrConfig = _DeepCfrConfig
    _config_stub.load_config = _load_config

    sys.modules["src.config"] = _config_stub

# ---------------------------------------------------------------------------
# Deterministic seeds fixture
# ---------------------------------------------------------------------------
import os
import pytest


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Fix all random seeds when CAMBIA_DETERMINISTIC=1 env var is set.

    Activate with: CAMBIA_DETERMINISTIC=1 pytest tests/
    """
    if os.environ.get("CAMBIA_DETERMINISTIC") == "1":
        import random as _random
        import numpy as _np
        import torch as _torch

        _torch.manual_seed(42)
        _np.random.seed(42)
        _random.seed(42)
        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    yield
    # Reset deterministic mode after test
    if os.environ.get("CAMBIA_DETERMINISTIC") == "1":
        import torch as _torch

        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(False)
            except Exception:
                pass
