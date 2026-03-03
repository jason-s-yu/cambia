"""src/config.py"""

from typing import List, Dict, TypeVar, Optional, Union, Any
import os
import logging
import re
import warnings
import yaml
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

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
                    log.warning("Unknown %s key '%s' — will be ignored", section_name, key)
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

    # Memory archetype
    memory_archetype: str = "perfect"
    memory_decay_lambda: float = 0.1
    memory_capacity: int = 3

    # N-player configuration
    num_players: int = 2

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


def load_config(config_path: str = "config.yaml") -> Config:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        base_path = raw.pop("_base", None)
        if base_path:
            abs_base = Path(config_path).parent / base_path
            with open(abs_base) as bf:
                base_raw = yaml.safe_load(bf) or {}
            base_raw.pop("_base", None)
            raw = _deep_merge(base_raw, raw)
        # _rule_profile support
        rule_profile = raw.pop("_rule_profile", None)
        if rule_profile:
            config_dir = Path(config_path).parent
            profile_path = config_dir / "rule_profiles" / f"{rule_profile}.yaml"
            if not profile_path.exists():
                profile_path = (
                    Path(__file__).parent.parent / "config" / "rule_profiles" / f"{rule_profile}.yaml"
                )
            if profile_path.exists():
                with open(profile_path) as pf:
                    profile_raw = yaml.safe_load(pf) or {}
                raw["cambia_rules"] = _deep_merge(profile_raw, raw.get("cambia_rules", {}))
            else:
                log.warning("Rule profile '%s' not found at %s", rule_profile, profile_path)
        # use_gpu backward compat
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
