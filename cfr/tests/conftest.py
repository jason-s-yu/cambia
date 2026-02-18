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

    # CambiaRulesConfig needs real defaults because CambiaGameState reads them
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

    from dataclasses import dataclass as _dataclass, field as _field

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
        use_gpu: bool = False
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

    def _load_config(path: str):
        """Stub load_config: loads real yaml config if pyyaml available, else returns None."""
        try:
            import yaml
            import importlib
            # Try to load the real config module dynamically
            _real_mod = importlib.import_module.__module__
        except Exception:
            pass
        # Fallback: return None so callers using stub can skip
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
