"""src/config.py"""

from typing import List, Dict, TypeVar, Optional, Union, Any, Literal
import os
import logging
import re
import warnings
import yaml
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

T = TypeVar("T")

log = logging.getLogger(__name__)


# --- Helpers ---


def get_nested(data: Dict, keys: List[str], default: T) -> T:
    """Safely retrieve a nested value from a dict. DEPRECATED."""
    warnings.warn(
        "get_nested() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current.get(key)
        else:
            return default
    if current is None and default is not None:
        return default
    return current  # type: ignore


def parse_human_readable_size(size_str: Union[str, int]) -> int:
    """Parses a human-readable size string (e.g., '1GB', '500MB', '1024') into bytes."""
    if isinstance(size_str, int):
        return size_str
    if not isinstance(size_str, str):
        raise ValueError(f"Invalid size format: {size_str}. Must be int or string.")

    size_str = size_str.upper().strip()
    match = re.fullmatch(r"(\d+)\s*(KB|MB|GB|TB)?", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "KB":
        value *= 1024
    elif unit == "MB":
        value *= 1024**2
    elif unit == "GB":
        value *= 1024**3
    elif unit == "TB":
        value *= 1024**4
    return value


def parse_num_workers(num_workers: Union[str, int]) -> int:
    """Parse the number of workers, ensuring it's a positive integer."""
    if isinstance(num_workers, int):
        if num_workers < 0:
            if num_workers == 0:
                cpu_count = os.cpu_count()
                return max(1, (cpu_count - 1)) if cpu_count else 1
            raise ValueError("num_workers must be a positive integer or 0 for auto.")
        return max(1, num_workers)
    if isinstance(num_workers, str):
        if num_workers.lower() == "auto":
            cpu_count = os.cpu_count()
            return max(1, (cpu_count - 1)) if cpu_count else 1
        try:
            val = int(num_workers)
            if val < 0:
                if val == 0:
                    cpu_count = os.cpu_count()
                    return max(1, (cpu_count - 1)) if cpu_count else 1
                raise ValueError("num_workers must be a positive integer or 0 for auto.")
            return max(1, val)
        except ValueError:
            raise ValueError(
                f"num_workers string '{num_workers}' must be 'auto' or an integer."
            ) from None
    raise ValueError(
        f"num_workers must be 'auto' or a positive integer. Got: {num_workers}"
    )


def validate_num_players(num_players: int) -> int:
    """Validate a num_players config value against the engine's supported range.

    Mirrors engine.HouseRules.Validate() (engine/rules.go): NumPlayers must be
    in [2, MaxPlayers=8]. Config-time validation here, rather than only
    Go-side FFI rejection, gives a clear error at config load instead of a
    late failure the first time a GoEngine is constructed (cambia-542 F3).
    """
    from src.constants import N_PLAYER_MAX_PLAYERS  # noqa: PLC0415 (avoid import cycle)

    if not (2 <= num_players <= N_PLAYER_MAX_PLAYERS):
        raise ValueError(
            f"num_players must be between 2 and {N_PLAYER_MAX_PLAYERS} "
            f"(MaxPlayers), got {num_players}."
        )
    return num_players


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge dicts; override wins. Non-dict values replaced."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# --- Base model ---


class _CambiaBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_keys(cls, values: Any) -> Any:
        if isinstance(values, dict):
            known = set(cls.model_fields.keys())
            # Convert "CambiaRulesConfig" → "cambia_rules" for readable warning
            raw_name = re.sub(r"Config$", "", cls.__name__)
            section_name = re.sub(r"(?<!^)(?=[A-Z])", "_", raw_name).lower()
            for key in values:
                if key not in known:
                    log.warning(
                        "Unknown %s key '%s' — will be ignored", section_name, key
                    )
        return values


# --- Configuration classes ---


class ApiConfig(_CambiaBaseModel):
    """API Client Settings (for online play)."""

    base_url: str = "http://localhost:8080"
    auth: Dict[str, str] = Field(default_factory=dict)


class SystemConfig(_CambiaBaseModel):
    """General system-level settings."""

    recursion_limit: int = 10000


class AnalysisConfig(_CambiaBaseModel):
    """Parameters for analysis tools, like exploitability calculation."""

    exploitability_num_workers: int = 1

    @field_validator("exploitability_num_workers", mode="before")
    @classmethod
    def _parse_workers(cls, v: Any) -> int:
        return parse_num_workers(v)


class CfrTrainingConfig(_CambiaBaseModel):
    """Parameters controlling the CFR training process."""

    num_iterations: int = 100000
    save_interval: int = 5000
    pruning_enabled: bool = True
    pruning_threshold: float = 1.0e-6
    exploitability_interval: int = 1000
    exploitability_interval_seconds: int = 0
    num_workers: int = 1

    @field_validator("num_workers", mode="before")
    @classmethod
    def _parse_num_workers(cls, v: Any) -> int:
        return parse_num_workers(v)


class CfrPlusParamsConfig(_CambiaBaseModel):
    """Parameters specific to CFR+ algorithm variants."""

    weighted_averaging_enabled: bool = True
    averaging_delay: int = 100


class AgentParamsConfig(_CambiaBaseModel):
    """Settings defining CFR agent behavior (memory, abstraction)."""

    memory_level: int = 1
    time_decay_turns: int = 3


class CambiaRulesConfig(_CambiaBaseModel):
    """Defines the specific game rules of Cambia."""

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
    # Optional reduced-deck rank subset for tractable tabular ground truth
    # (research experiment E1). None = full 13-rank deck. When set (e.g.
    # ["A","2","3","4","K"]), the engine deals only those non-joker ranks.
    deck_ranks: Optional[List[str]] = None


class PersistenceConfig(_CambiaBaseModel):
    """Configuration for saving and loading trained agent data."""

    agent_data_save_path: str = "runs/default/checkpoints/deep_cfr_checkpoint.pt"


class WorkerLogOverrideConfig(_CambiaBaseModel):
    """Specifies a logging level override for specific worker IDs."""

    worker_ids: List[int] = Field(default_factory=list)
    level: str = "INFO"


class WorkerLoggingConfig(_CambiaBaseModel):
    """Configuration for worker-specific logging levels."""

    default_level: str = "INFO"
    sequential_rules: List[Union[str, Dict[str, int]]] = Field(default_factory=list)
    overrides: List[WorkerLogOverrideConfig] = Field(default_factory=list)


class LoggingConfig(_CambiaBaseModel):
    """Settings for configuring logging behavior."""

    log_level_file: str = "DEBUG"
    log_level_console: str = "ERROR"
    log_dir: str = "logs"
    log_file_prefix: str = "cambia"
    log_max_bytes: int = 9 * 1024 * 1024
    log_backup_count: int = 999
    worker_config: Optional[WorkerLoggingConfig] = None
    log_archive_enabled: bool = False
    log_archive_max_archives: int = 10
    log_archive_dir: str = "archives"
    log_size_update_interval_sec: int = 60
    log_simulation_traces: bool = False
    simulation_trace_filename_prefix: str = "simulation_traces"

    @field_validator("log_max_bytes", mode="before")
    @classmethod
    def _parse_log_max_bytes(cls, v: Any) -> int:
        return parse_human_readable_size(v)

    def get_worker_log_level(self, worker_id: int, num_total_workers: int) -> str:
        """Determines the log level for a specific worker based on the configuration."""
        if self.worker_config is None:
            return self.log_level_file

        if self.worker_config.overrides:
            for override_rule in self.worker_config.overrides:
                if worker_id in override_rule.worker_ids:
                    return override_rule.level.upper()

        current_worker_idx_in_rules = 0
        if self.worker_config.sequential_rules:
            for rule_entry in self.worker_config.sequential_rules:
                level_to_apply: str
                num_workers_for_rule = 1

                if isinstance(rule_entry, str):
                    level_to_apply = rule_entry.upper()
                elif isinstance(rule_entry, dict):
                    if len(rule_entry) != 1:
                        logging.warning(
                            "Invalid sequential_rule format: %s. Skipping.", rule_entry
                        )
                        continue
                    level_to_apply = list(rule_entry.keys())[0].upper()
                    num_workers_for_rule = list(rule_entry.values())[0]
                    if (
                        not isinstance(num_workers_for_rule, int)
                        or num_workers_for_rule < 1
                    ):
                        logging.warning(
                            "Invalid count in sequential_rule: %s. Using 1.", rule_entry
                        )
                        num_workers_for_rule = 1
                else:
                    logging.warning(
                        "Unknown sequential_rule type: %s. Skipping.", rule_entry
                    )
                    continue

                if (
                    current_worker_idx_in_rules
                    <= worker_id
                    < current_worker_idx_in_rules + num_workers_for_rule
                ):
                    return level_to_apply
                current_worker_idx_in_rules += num_workers_for_rule

        if self.worker_config.default_level:
            return self.worker_config.default_level.upper()

        return self.log_level_file.upper()


# --- Deep CFR Configuration ---


class DeepCfrConfig(_CambiaBaseModel):
    """Configuration for Deep CFR training."""

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
    # "auto" | "cpu" | "cuda" | "xpu" (Intel Arc via torch's xpu backend, no
    # IPEX). "auto" resolves at trainer init: cuda -> xpu -> cpu.
    device: str = "auto"
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

    # ESCHER traversal method
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

    # GT-CFR configuration
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

    # SoG (Student of Games) configuration
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

    # SD-CFR configuration
    use_sd_cfr: bool = False
    sd_cfr_max_snapshots: int = 200
    sd_cfr_snapshot_weighting: str = "linear"
    num_hidden_layers: int = 3
    use_residual: bool = True
    network_type: str = "residual"
    use_pos_embed: bool = True
    use_ema: bool = True

    # Profiling
    enable_traversal_profiling: bool = False
    profiling_jsonl_path: str = ""
    profile_step: Optional[List[int]] = None

    # Encoding
    encoding_mode: str = "legacy"
    encoding_layout: str = "auto"
    encoding_version: int = 1  # 1 = legacy 224-dim, 2 = 257-dim (Phase 0 DESCA)

    # Memory archetype
    memory_archetype: str = "perfect"
    memory_decay_lambda: float = 0.1
    memory_capacity: int = 3

    # N-player configuration
    num_players: int = 2

    @field_validator("num_players")
    @classmethod
    def _validate_num_players(cls, v: int) -> int:
        return validate_num_players(v)

    # QRE regularization
    qre_lambda_start: float = 0.5
    qre_lambda_end: float = 0.05
    qre_anneal_fraction: float = 0.6

    # PSRO configuration
    use_psro: bool = False
    psro_population_size: int = 15
    psro_eval_games: int = 200
    psro_checkpoint_interval: int = 50
    psro_heuristic_types: str = "random,greedy,memory_heuristic"

    # Adaptive training steps
    target_buffer_passes: float = 0.0
    value_target_buffer_passes: float = 2.0


# --- DESCA Configuration ---


class StallDetectionConfig(_CambiaBaseModel):
    """Stall detection parameters for DESCA training."""

    window_size_iters: int = 50
    num_windows: int = 5
    max_iter_abs: int = 3000


class DESCAConfig(_CambiaBaseModel):
    """Configuration for DESCA (Dense ESCHER + Semantic Action Abstraction) training."""

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
    inner_update: Literal["apcfr_plus", "rm_plus"] = "apcfr_plus"
    use_bf16_inference: bool = True
    # SGD steps per training iteration for each head. Trainer reads these via
    # _cfg_get with these defaults; exposing them on the model lets tiny-game /
    # fast configs override them (the defaults are tuned for full Cambia).
    regret_sgd_steps: int = 2000
    strategy_sgd_steps: int = 2000
    value_sgd_steps: int = 1000
    stall_detection: StallDetectionConfig = Field(default_factory=StallDetectionConfig)


# --- PRT-CFR Configuration ---


class PRTCFRConfig(_CambiaBaseModel):
    """Configuration for PRT-CFR (Perfect-Recall Trajectory CFR) training.

    Minimal Phase 1 (X2 gate) surface: the GRU sequence net dims, the m-rollout
    MC estimator, the from-scratch refit loop, and SD-CFR snapshot weighting. The
    net dims are pinned by the Phase 1 Sprint 1 interface contract (shared with
    the X2 scorer); changing them breaks snapshot/checkpoint compatibility.
    """

    # GRU sequence encoder (pinned). Vocab = sequence_encoding.VOCAB_SIZE; 327
    # after the cambia-564 race-resolution block was appended (325 base, +1
    # cambia-529 peek, +1 cambia-564 race). Growing it invalidates prior
    # checkpoints (the token embedding row count changed): acceptable, X4 is future work.
    gru_vocab_size: int = 327
    gru_embed_dim: int = 64
    gru_hidden_dim: int = 256
    gru_num_layers: int = 2
    gru_dropout: float = 0.1
    head_hidden_dim: int = 256
    seq_cap: int = 256

    # Estimator + training loop.
    m_rollouts: int = 4
    k_games_per_iter: int = 200
    iterations: int = 100
    lr: float = 1.0e-3
    batch_size: int = 1024
    train_steps_per_iter: int = 256
    buffer_capacity: int = 2_000_000
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Refit init: False re-inits the net each iteration (Brown 2019 from-scratch);
    # True warm-starts from the previous iterate and fine-tunes to convergence.
    # SD-CFR-valid either way (the fit's fixed point R^t is init-independent given
    # convergence); warm-start reaches R^t with fewer steps (the underfit fix).
    warm_start: bool = False

    # SD-CFR snapshot realization. Only "linear" (w_t = t) is implemented for X2.
    snapshot_weighting: str = "linear"

    # --- Late-training stability + LR schedule (Phase 2 S1W1). ---
    # These field DEFAULTS deliberately match the tiny trainer's prior getattr
    # fallbacks (schedule "restart", stability OFF) so the X2 tiny-game paths and
    # their tests stay byte-for-byte identical. The PRODUCTION defaults
    # (warm_start + global_cosine + stability ON, design-overview Trainer
    # conventions amended 2026-07-07) are set in the production run config YAML
    # (config/prtcfr_production.yaml), NOT flipped here -- flipping the model
    # defaults would break the tiny gate reproduction (e.g. stability-OFF tests)
    # and force the 256-wide tiny reservoir to a 12288-wide allocation.
    lr_min: float = 0.0
    lr_schedule: str = "restart"  # "restart" | "global_cosine"
    reanchor_every: int = 0
    stability_enabled: bool = False
    stability_eval_every: int = 10
    stability_patience: int = 3
    stability_rel_tolerance: float = 0.15
    stability_min_iters: int = 10
    stability_metric_mode: str = "min"
    stability_metric_name: str = "nashconv"
    # Early-stop rule (cambia-341). "divergence" (default) is the patience/
    # tolerance rule above and reproduces every existing run byte-for-byte;
    # "plateau" stops on trailing-window relative-improvement stagnation
    # instead (for future gate runs that flatten without diverging -- see
    # prtcfr_stability module docstring). Additive: switching stop_mode does
    # not change the other stability_* fields' meaning.
    stability_stop_mode: str = "divergence"  # "divergence" | "plateau"
    stability_plateau_window_iters: int = 50
    stability_plateau_step_iters: int = 10
    stability_plateau_rel_improvement: float = 0.005

    # --- Production trainer (Phase 2 S1W5): full-game PRT-CFR. ---
    # Read only by PRTCFRProductionTrainer (the tiny trainer never touches
    # these), so production-scale defaults here are safe for the tiny paths.
    # A production run points seq_cap/k_games_per_iter/batch_size/warm_start/
    # lr_schedule/stability_enabled at the production values via the run config.
    train_steps: int = 3000  # per-iteration fit budget (p2 sec 2.3)
    reservoir_capacity: int = 20_000_000  # per-player disk reservoir rows (AC5)
    reservoir_dir: Optional[str] = None  # override; default <run_dir>/reservoir
    snapshot_dir: Optional[str] = None  # override; default <run_dir>/snapshots
    num_players: int = 2

    @field_validator("num_players")
    @classmethod
    def _validate_num_players(cls, v: int) -> int:
        return validate_num_players(v)

    max_trajectory_steps: int = 4000
    backend: str = "go"  # production GameDriver backend: "go" | "python"
    # --- Batched incremental production generation (S1W15). ---
    # gen_batched=True routes production generation through the batched
    # incremental PRTCFRInferenceService (all live games + their m rollouts
    # share one carried-hidden forward per decision tick) instead of the
    # per-decision full-prefix NetProductionSigma; it is the X3 gen remedy and
    # the production default. False keeps the original sequential path (kept for
    # the semantic-equivalence gate and as a fallback).
    gen_batched: bool = True
    # Concurrent games per scheduler chunk. Bounds peak simultaneously-live
    # drivers (~gen_chunk_games * max_legal * m_rollouts rollout clones); keep
    # it under the Go handle pool ceiling (maxGames=32768, raised from 2048 in
    # cambia-534) with margin. The 32768 ceiling is mechanism, not policy: this
    # default stays 64 (production value pending the X3 P4/P5 GPU-window
    # validation), and X-cells set it per run-config.
    gen_chunk_games: int = 64
    # Inference precision for the batched sigma service: "bf16" (throughput
    # default, p2-redesign sec 6) or "fp32" (used by the equivalence gate). The
    # carry-vs-reencode identity holds at either precision; bf16 is an
    # independent approximation of the fp32 sigma.
    infer_dtype: str = "bf16"
    # V_phi critic outside the regret path (S1W6 wiring).
    critic_enabled: bool = True
    critic_capacity: int = 200_000
    critic_steps_per_iter: int = 500
    critic_batch_size: int = 512
    critic_lr: float = 1.0e-3
    critic_held_out_fraction: float = 0.1

    # --- In-loop X4 battery (Phase 2 S2W1): Tier-A LBR fast lane. ---
    # Read only by build_production_battery_eval_fn (the trainer's in-loop
    # stability trend); the tiny trainer never touches these. Small fast-lane
    # sizes so the every-N-iters battery stays a small fraction of gen+fit; a
    # production run tunes them per device (see the S2 budget-branch note).
    # battery_lbr_games -> Tier-A sampled_lbr num_infosets (P0 infosets sampled);
    # battery_lbr_depth  -> br_rollouts_per_infoset (BR-estimate rollouts/action).
    battery_lbr_games: int = 64
    battery_lbr_depth: int = 8

    seed: int = 0
    # "cpu" | "cuda" | "xpu" (Intel Arc via torch's xpu backend) | "auto"
    # (resolved at CLI launch: cuda -> xpu -> cpu; see cli.py train_prtcfr).
    device: str = "cuda"

    # --- Tiny-gate trainer (X2 NashConv gate). ---
    # True routes cli.train_prtcfr to PRTCFRTinyTrainer (the tiny_solver
    # ground-truth NashConv gate) instead of PRTCFRProductionTrainer. Default
    # False leaves every existing config on the production full-game path.
    tiny_gate: bool = False
    # Full-state warm-start source for a NEW run: a resume_state.json (or its
    # run dir) for a bit-exact continuation, or a legacy snapshot/checkpoint .pt
    # for net + iteration only. Distinct from ``warm_start`` (the per-iteration
    # refit-init policy). None starts fresh. The serving harness surfaces its
    # spec ``warm_start`` field here as an absolute path render rail.
    #
    # Semantics (cambia-374): a bare .pt carries net weights + iteration only,
    # so warm-starting from one restarts the reservoir EMPTY and fine-tunes the
    # imported net on regret targets from a tiny immature buffer -- a fresh run
    # with a net prior, NOT a continuation. Measured effect: ~0.2-NashConv
    # snapshots that dominate the linear SD-CFR mixture. Full mode (a run dir
    # or its resume_state.json) restores net + reservoir and is the only
    # state-faithful continuation path; verdict/ruled continuations must use
    # it. warm_start_net_only_ok gates the net-only path explicitly.
    warm_start_path: Optional[str] = None
    # Explicit opt-in required to warm-start from a bare .pt (see warm_start_path
    # above). False (default): _resolve_warm_start raises when the path
    # classifies as net-only, naming the fix (point at the run dir) instead of
    # silently producing a fresh-run-with-net-prior snapshot lineage. Full-mode
    # warm start and --resume are never affected by this flag.
    warm_start_net_only_ok: bool = False


# --- Baseline Agent Configuration ---


class GreedyAgentConfig(_CambiaBaseModel):
    """Configuration specific to the Greedy baseline agent."""

    cambia_call_threshold: int = 5


class AgentsConfig(_CambiaBaseModel):
    """Configuration for baseline agents (used for evaluation)."""

    greedy_agent: GreedyAgentConfig = Field(default_factory=GreedyAgentConfig)


# --- Main Config Class ---


class Config(_CambiaBaseModel):
    """Root configuration object."""

    config_schema_version: int = 1
    api: ApiConfig = Field(default_factory=ApiConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    cfr_training: CfrTrainingConfig = Field(default_factory=CfrTrainingConfig)
    cfr_plus_params: CfrPlusParamsConfig = Field(default_factory=CfrPlusParamsConfig)
    agent_params: AgentParamsConfig = Field(default_factory=AgentParamsConfig)
    cambia_rules: CambiaRulesConfig = Field(default_factory=CambiaRulesConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    deep_cfr: DeepCfrConfig = Field(default_factory=DeepCfrConfig)
    desca: Optional[DESCAConfig] = None
    prt_cfr: Optional[PRTCFRConfig] = None
    _source_path: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _check_schema_version(cls, values: Any) -> Any:
        if isinstance(values, dict):
            version = values.get("config_schema_version", 1)
            if isinstance(version, int) and version > 1:
                log.warning(
                    "Config schema version %d is newer than supported version 1. "
                    "Some fields may be ignored.",
                    version,
                )
        return values


def resolve_config_yaml(
    config_path: Union[str, Path], _seen: Optional[set] = None
) -> dict:
    """Read a config YAML and return the merged raw dict (resolves _base + _rule_profile).

    _base resolution searches: (1) next to config_path, (2) cfr/config/. The fallback
    lets eval load run-dir configs that retain a _base reference even though the base
    file lives in cfr/config/, not in the run dir.

    _base chains resolve recursively (base-of-base and deeper), with cycle detection.
    Silently dropping a nested _base once trained the full 54-card game under a
    tiny-gate config (X2R incident, cambia-518): the chain must merge completely or
    fail loudly.
    """
    config_path = Path(config_path)
    resolved_self = config_path.resolve()
    _seen = set() if _seen is None else _seen
    if resolved_self in _seen:
        raise ValueError(
            f"_base cycle detected: {config_path} already visited in this chain"
        )
    _seen.add(resolved_self)
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_path = raw.pop("_base", None)
    if base_path:
        candidates = [
            config_path.parent / base_path,
            Path(__file__).parent.parent / "config" / base_path,
        ]
        base_resolved = next((c for c in candidates if c.exists()), None)
        if base_resolved is None:
            raise FileNotFoundError(
                f"_base '{base_path}' referenced by {config_path} not found in: "
                f"{[str(c) for c in candidates]}"
            )
        base_raw = resolve_config_yaml(base_resolved, _seen)
        raw = _deep_merge(base_raw, raw)
    rule_profile = raw.pop("_rule_profile", None)
    if rule_profile:
        config_dir = config_path.parent
        profile_path = config_dir / "rule_profiles" / f"{rule_profile}.yaml"
        if not profile_path.exists():
            profile_path = (
                Path(__file__).parent.parent
                / "config"
                / "rule_profiles"
                / f"{rule_profile}.yaml"
            )
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as pf:
                profile_raw = yaml.safe_load(pf) or {}
            raw["cambia_rules"] = _deep_merge(profile_raw, raw.get("cambia_rules", {}))
        else:
            log.warning("Rule profile '%s' not found at %s", rule_profile, profile_path)
    return raw


def load_config(config_path: str = "config.yaml") -> Config:
    """Loads configuration from a YAML file."""
    try:
        raw = resolve_config_yaml(config_path)
        deep = raw.get("deep_cfr", {})
        if "use_gpu" in deep and "device" not in deep:
            deep["device"] = "cuda" if deep.pop("use_gpu") else "cpu"
        cfg = Config.model_validate(raw)
        cfg._source_path = os.path.abspath(config_path)
        return cfg
    except FileNotFoundError:
        log.warning(
            "Config file '%s' not found. Using default configuration.", config_path
        )
        return Config()
    except (TypeError, KeyError, AttributeError, yaml.YAMLError, ValueError) as e:
        log.exception(
            "Error loading config file '%s': %s. Using default config.",
            config_path,
            e,
        )
        return Config()
    except IOError as e:
        log.exception(
            "IOError loading config file '%s': %s. Using default config.",
            config_path,
            e,
        )
        return Config()
