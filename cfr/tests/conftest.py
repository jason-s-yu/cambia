"""
tests/conftest.py

Shared fixtures and bootstrap logic for all tests.

Injects a minimal stub for src.config so that modules like agent_state and
encoding can be loaded before the real config module is fully initialised.
"""

import sys
import types
from pathlib import Path
from typing import List, Optional, Union

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Stub out src.config ---
# Only inject the stub if config hasn't been loaded yet or is broken.
_config_mod = sys.modules.get("src.config")
if _config_mod is None or not hasattr(_config_mod, "Config"):
    _config_stub = types.ModuleType("src.config")

    from pydantic import BaseModel as _BaseModel, Field as _Field, ConfigDict as _ConfigDict

    class _StubConfig(_BaseModel):
        """Minimal placeholder for Config and its sub-classes."""
        model_config = _ConfigDict(extra="allow")

    # CambiaRulesConfig needs real defaults because CambiaGameState reads them
    class _CambiaRulesConfig(_BaseModel):
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
        num_decks: int = 1

    class _DeepCfrConfig(_BaseModel):
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
        max_tasks_per_child: Optional[Union[int, str]] = "auto"
        worker_memory_budget_pct: float = 0.10
        # ESCHER fields
        traversal_method: str = "outcome"
        value_hidden_dim: int = 512
        value_learning_rate: float = 1e-3
        value_buffer_capacity: int = 2_000_000
        batch_counterfactuals: bool = True
        # DEPRECATED: ReBeL fields retained for checkpoint backward compat only.
        rebel_subgame_depth: int = 4
        rebel_cfr_iterations: int = 200
        rebel_value_hidden_dim: int = 1024
        rebel_policy_hidden_dim: int = 512
        rebel_value_learning_rate: float = 1e-3
        rebel_policy_learning_rate: float = 1e-3
        rebel_value_buffer_capacity: int = 500_000
        rebel_policy_buffer_capacity: int = 500_000
        rebel_games_per_epoch: int = 100
        rebel_epochs: int = 10

        # SD-CFR fields
        use_sd_cfr: bool = False
        sd_cfr_max_snapshots: int = 200
        sd_cfr_snapshot_weighting: str = "linear"
        num_hidden_layers: int = 3
        use_residual: bool = True
        network_type: str = "residual"
        use_pos_embed: bool = True
        use_ema: bool = True
        enable_traversal_profiling: bool = False
        profiling_jsonl_path: str = ""
        profile_step: Optional[List[int]] = None
        encoding_mode: str = "legacy"
        encoding_layout: str = "auto"
        encoding_version: int = 1
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
        target_buffer_passes: float = 0.0
        value_target_buffer_passes: float = 2.0

        # GT-CFR fields
        gtcfr_expansion_budget: int = 100
        gtcfr_c_puct: float = 2.0
        gtcfr_cfr_iters_per_expansion: int = 10
        gtcfr_expansion_k: int = 3
        gtcfr_cvpn_hidden_dim: int = 512
        gtcfr_cvpn_num_blocks: int = 4
        gtcfr_cvpn_learning_rate: float = 3e-4
        gtcfr_value_loss_weight: float = 1.0
        gtcfr_policy_loss_weight: float = 5.0
        gtcfr_buffer_capacity: int = 1_000_000
        gtcfr_games_per_epoch: int = 50
        gtcfr_epochs: int = 20
        gtcfr_exploration_epsilon: float = 0.05
        gtcfr_warm_start_rebel_checkpoint: str = ""

        # CVPN gradient control
        cvpn_detach_policy_grad: bool = False

        # SoG fields
        sog_train_budget: int = 50
        sog_eval_budget: int = 200
        sog_c_puct: float = 2.0
        sog_cfr_iters_per_expansion: int = 10
        sog_max_persist_depth: int = 8
        sog_max_persist_handles: int = 512
        sog_safety_margin: float = 0.01
        sog_games_per_epoch: int = 50
        sog_epochs: int = 1000
        sog_exploration_epsilon: float = 0.05
        sog_warm_start_checkpoint: str = ""

    class _StallDetectionConfig(_BaseModel):
        """Stub for StallDetectionConfig."""
        window_size_iters: int = 50
        num_windows: int = 5
        max_iter_abs: int = 3000

    class _DESCAConfig(_BaseModel):
        """Stub for DESCAConfig."""
        encoding_version: int = 2
        hidden_dim: int = 512
        num_abstract_actions: int = 32
        iterations: int = 1000
        traversals_per_iter: int = 2000
        minibatch: int = 1024
        lr: float = 3.0e-4
        weight_decay: float = 1.0e-4
        grad_clip: float = 1.0
        dcfr_alpha: float = 1.5
        apcfr_asymmetry: float = 0.9
        buffer_capacity: int = 2_000_000
        checkpoint_every: int = 50
        eval_every: int = 50
        warmup_iters: int = 50
        inner_update: str = "apcfr_plus"
        stall_detection: _StallDetectionConfig = _Field(default_factory=_StallDetectionConfig)

    class _PRTCFRConfig(_BaseModel):
        """Stub for PRTCFRConfig (Phase 1 X2 PRT-CFR)."""
        gru_vocab_size: int = 325
        gru_embed_dim: int = 64
        gru_hidden_dim: int = 256
        gru_num_layers: int = 2
        gru_dropout: float = 0.1
        head_hidden_dim: int = 256
        seq_cap: int = 256
        m_rollouts: int = 4
        k_games_per_iter: int = 200
        iterations: int = 100
        lr: float = 1.0e-3
        batch_size: int = 1024
        train_steps_per_iter: int = 256
        buffer_capacity: int = 2_000_000
        weight_decay: float = 0.0
        grad_clip: float = 1.0
        warm_start: bool = False
        snapshot_weighting: str = "linear"
        # Stability + LR schedule (defaults match the tiny trainer's getattr
        # fallbacks; production values come from the run config, not here).
        lr_min: float = 0.0
        lr_schedule: str = "restart"
        reanchor_every: int = 0
        stability_enabled: bool = False
        stability_eval_every: int = 10
        stability_patience: int = 3
        stability_rel_tolerance: float = 0.15
        stability_min_iters: int = 10
        stability_metric_mode: str = "min"
        stability_metric_name: str = "nashconv"
        # Production trainer (Phase 2 S1W5).
        train_steps: int = 3000
        reservoir_capacity: int = 20_000_000
        reservoir_dir: object = None
        snapshot_dir: object = None
        num_players: int = 2
        max_trajectory_steps: int = 4000
        backend: str = "go"
        critic_enabled: bool = True
        critic_capacity: int = 200_000
        critic_steps_per_iter: int = 500
        critic_batch_size: int = 512
        critic_lr: float = 1.0e-3
        critic_held_out_fraction: float = 0.1
        seed: int = 0
        device: str = "cuda"

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

    def _resolve_config_yaml(path):
        """Stub resolve_config_yaml: delegate to the real implementation.

        The PRT-CFR (and DESCA) CLI materializes the run-dir config via
        ``resolve_config_yaml``; the stub delegates like ``_load_config`` so a
        direct CLI-function call under the stub still resolves _base/_rule_profile.
        """
        _saved = sys.modules.pop("src.config", None)
        import importlib
        _real_mod = importlib.import_module("src.config")
        try:
            return _real_mod.resolve_config_yaml(path)
        finally:
            if _saved is not None:
                sys.modules["src.config"] = _saved

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
    _config_stub.StallDetectionConfig = _StallDetectionConfig
    _config_stub.DESCAConfig = _DESCAConfig
    _config_stub.PRTCFRConfig = _PRTCFRConfig
    _config_stub.load_config = _load_config
    _config_stub.resolve_config_yaml = _resolve_config_yaml

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
