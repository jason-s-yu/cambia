"""
src/cfr/desca_trainer.py

DESCA training loop: three reservoir buffers, three networks, APCFR+ / RM+
inner update, DCFR+ weighting on regret, Linear CFR weighting on strategy,
checkpoints, run_db integration, Tier 4 runtime monitors, and iteration
milestone logging.

Public surface (stable from day 1 of Stream B so Streams A, C, and D can
integrate without churn):

    from cfr.src.cfr.desca_trainer import DESCATrainer

    trainer = DESCATrainer(
        config,                         # Pydantic DESCAConfig or dict
        regret_net,                     # cfr.src.desca_networks.RegretNetwork
        avg_strategy_net,               # cfr.src.desca_networks.AvgStrategyNetwork
        history_value_net,              # cfr.src.desca_networks.HistoryValueNetwork
        env_factory,                    # () -> (engine, agents) for a fresh game
        *,
        device=None,                    # torch.device or string
        checkpoint_path=None,           # "runs/<name>/checkpoints/desca_checkpoint.pt"
        seed=None,                      # rng seed (optional)
    )
    trainer.train(num_iterations=None)  # overrides config.iterations if set

Attributes (read-only after __init__):
    trainer.iteration                   # current CFR iteration counter
    trainer.regret_buffer               # ReservoirBuffer(input_dim=257, target_dim=32)
    trainer.strategy_buffer             # ReservoirBuffer(input_dim=257, target_dim=32)
    trainer.value_buffer                # ReservoirBuffer(input_dim=377, target_dim=1)

Checkpoint layout (runs/<name>/checkpoints/desca_checkpoint_iter_N.pt):
    {
        "regret_state_dict": ...,
        "avg_strategy_state_dict": ...,
        "history_value_state_dict": ...,
        "desca_state_dict": {
            "iteration": int,
            "inner_update": "apcfr_plus" | "rm_plus",
            "config": <dict>,
            "buffer_rng_state": tuple,
        },
        "algorithm": "desca",
        "iteration": int,
    }

See spec Section 8 for the authoritative algorithm description and
``.docs/v3/phase1-dense-escher/contract.md`` for the full deliverable list.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..action_abstraction import NUM_ABSTRACT_ACTIONS_2P
from ..persistence import atomic_torch_save
from ..reservoir import ReservoirBuffer, ReservoirSample
from .desca_worker import (
    FEATURE_DIM,
    OMNISCIENT_DIM_2P,
    VALUE_INPUT_DIM,
    run_desca_iteration,
)

try:
    from .. import run_db as _run_db  # type: ignore[attr-defined]
    _RUN_DB_AVAILABLE = True
except Exception:
    _run_db = None  # type: ignore[assignment]
    _RUN_DB_AVAILABLE = False

logger = logging.getLogger(__name__)

_APCFR_FALLBACK_WARNED: bool = False


# ---------------------------------------------------------------------------
# Config compatibility layer
# ---------------------------------------------------------------------------


_DEFAULT_STALL = {
    "window_size_iters": 50,
    "num_windows": 5,
    "max_iter_abs": 3000,
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Uniform accessor for Pydantic models, dataclasses, and plain dicts."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _coerce_stall_config(raw: Any) -> Dict[str, int]:
    """Normalize stall-detection config into a dict of {window_size_iters,
    num_windows, max_iter_abs}."""
    out = dict(_DEFAULT_STALL)
    if raw is None:
        return out
    for key in out:
        val = _cfg_get(raw, key, out[key])
        try:
            out[key] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def _cfg_as_dict(cfg: Any) -> Dict[str, Any]:
    """Best-effort serialization of config for the checkpoint payload."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "model_dump"):
        try:
            return cfg.model_dump()
        except Exception:
            pass
    if hasattr(cfg, "dict"):
        try:
            return cfg.dict()
        except Exception:
            pass
    if is_dataclass(cfg):
        return asdict(cfg)
    out = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            v = getattr(cfg, key)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            # Only keep JSON-friendly primitives; nested configs get recursed.
            if isinstance(v, (int, float, bool, str, list, tuple)) or v is None:
                out[key] = v
            else:
                out[key] = _cfg_as_dict(v)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Loss + weighting helpers
# ---------------------------------------------------------------------------


def _dcfr_plus_weight(iteration: int, alpha: float = 1.5) -> float:
    """DCFR+ discounting: w_iter = iter^alpha / (iter^alpha + 1).

    Per spec Section 8.3 default alpha = 1.5.
    """
    it = max(1, int(iteration))
    num = float(it) ** float(alpha)
    return num / (num + 1.0)


def _regret_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"],
    iter_weights: "torch.Tensor",
) -> "torch.Tensor":
    """DCFR+ weighted masked MSE on regret predictions.

    Args:
        pred: (B, A)
        target: (B, A)
        mask: (B, A) bool or None
        iter_weights: (B,) per-sample DCFR+ weights

    Returns:
        Scalar mean loss.
    """
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask.float()
    per_sample = diff.sum(dim=1)
    if mask is not None:
        denom = mask.float().sum(dim=1).clamp_min(1.0)
        per_sample = per_sample / denom
    weighted = per_sample * iter_weights
    return weighted.mean()


def _strategy_loss(
    pred_probs: "torch.Tensor",
    target: "torch.Tensor",
    mask: Optional["torch.Tensor"],
    iter_weights: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Linear-CFR weighted KL divergence from target -> pred distributions.

    KL(target || pred) summed over abstract-action support, then weighted by
    per-sample iteration weights.
    """
    p = pred_probs.clamp_min(eps)
    q = target.clamp_min(eps)
    kl = target * (torch.log(q) - torch.log(p))
    if mask is not None:
        kl = kl * mask.float()
    per_sample = kl.sum(dim=1)
    weighted = per_sample * iter_weights
    return weighted.mean()


def _value_loss(
    pred: "torch.Tensor", target: "torch.Tensor"
) -> "torch.Tensor":
    """Plain MSE for V_omni."""
    return ((pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# APCFR+ wrapper
# ---------------------------------------------------------------------------


class _PrevGradStore:
    """Per-parameter cache of the previous step's (post-clip) gradient.

    Keyed by ``id(parameter)``: APCFR+ uses this across successive SGD steps
    within a single fit call to implement the optimistic-gradient
    extrapolation. The cache is cleared at the top of each fit call to bound
    staleness (the prev-grad prediction is only useful when consecutive batches
    share the same broad loss surface, i.e. within one fit pass).
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: Dict[int, "torch.Tensor"] = {}

    def get(self, p: "torch.nn.Parameter") -> Optional["torch.Tensor"]:
        return self._cache.get(id(p))

    def put(self, p: "torch.nn.Parameter", g: "torch.Tensor") -> None:
        self._cache[id(p)] = g.detach().clone()

    def clear(self) -> None:
        self._cache.clear()


def _apcfr_plus_step(
    optimizer: "torch.optim.Optimizer",
    params: Sequence["torch.nn.Parameter"],
    loss: "torch.Tensor",
    asymmetry: float,
    grad_clip: float,
    prev_store: _PrevGradStore,
) -> float:
    """APCFR+ optimistic-gradient extrapolation (Meng et al., Mar 2025).

    Implements the predictive-gradient variant from Meng et al.'s APCFR+:
    at step t, use the optimistic extrapolation
    ``g_tilde = g_t + asymmetry * (g_t - g_{t-1})`` as the update direction.
    This is mathematically the lookahead used in Optimistic / Predictive
    Gradient methods (e.g. Rakhlin & Sridharan 2013) and gives APCFR+ its
    tighter regret bound over plain RM+ on adversarial loss sequences near
    the origin. When ``asymmetry = 0``, the step degenerates to RM+; when
    ``asymmetry = 1``, the prediction assumes the next gradient will drift
    by the same vector the last one did.

    The prev_store is cleared at the top of each fit call by the trainer to
    bound stale-prediction risk: prev grads are only a useful predictor of
    the next grad while we are sampling from the same reservoir snapshot.

    Args:
        optimizer: The torch optimizer wrapping ``params``.
        params: Iterable of parameters to step. Must be the same concrete
            objects across calls within a fit pass so ``id(p)`` lookups hit.
        loss: Scalar loss tensor with requires_grad.
        asymmetry: Optimistic extrapolation coefficient in [0, 1].
            0 = RM+ (no extrapolation); 1 = full optimistic lookahead.
        grad_clip: Max grad-norm for clip_grad_norm_ prior to extrapolation.
        prev_store: Per-head ``_PrevGradStore`` with prev-iter gradients.

    Returns:
        Post-clip, pre-extrapolation grad norm (for runtime monitoring).
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip).item()

    alpha = float(asymmetry)
    for p in params:
        if p.grad is None:
            continue
        # Snapshot the clipped gradient BEFORE extrapolation so we cache
        # g_t (not g_tilde) for the next step.
        raw = p.grad.detach().clone()
        prev = prev_store.get(p)
        if prev is not None and alpha != 0.0:
            # p.grad <- g_t + alpha * (g_t - g_{t-1})
            p.grad.add_(raw - prev, alpha=alpha)
        prev_store.put(p, raw)

    optimizer.step()
    return float(grad_norm)


def _rm_plus_step(
    optimizer: "torch.optim.Optimizer",
    params: Sequence["torch.nn.Parameter"],
    loss: "torch.Tensor",
    grad_clip: float,
) -> float:
    """Plain RM+ update: backward, clip, step."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip).item()
    optimizer.step()
    return float(grad_norm)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DESCATrainer:
    """Main DESCA training loop (spec v3.1 Section 8).

    Args:
        config: ``DESCAConfig`` Pydantic model, a dataclass, or a dict with
            equivalent fields. Required fields:
              encoding_version, hidden_dim, num_abstract_actions, iterations,
              traversals_per_iter, minibatch, lr, weight_decay, grad_clip,
              dcfr_alpha, apcfr_asymmetry, buffer_capacity,
              checkpoint_every, eval_every, warmup_iters, inner_update,
              stall_detection.{window_size_iters, num_windows, max_iter_abs}.
        regret_net: RegretNetwork (257 -> 32).
        avg_strategy_net: AvgStrategyNetwork (257 -> 32 masked softmax).
        history_value_net: HistoryValueNetwork (257 + 120 -> 1 in 2P).
        env_factory: Callable producing a fresh (engine, agents) pair per
            traversal. Signature: ``(rng=None) -> (engine, agents)``.
        device: Torch device string or ``torch.device``. Defaults to CPU.
        checkpoint_path: Base checkpoint path (".pt"). Iteration checkpoints
            are saved as ``{base}_iter_{N}.pt``.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        config: Any,
        regret_net: "nn.Module",
        avg_strategy_net: "nn.Module",
        history_value_net: "nn.Module",
        env_factory: Callable,
        *,
        device: Any = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for DESCATrainer")

        self.config = config
        self.env_factory = env_factory
        self.checkpoint_path = checkpoint_path or "runs/desca-scratch/checkpoints/desca_checkpoint.pt"

        # Device
        if device is None:
            device = "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device

        # Networks
        self.regret_net = regret_net.to(self.device)
        self.avg_strategy_net = avg_strategy_net.to(self.device)
        self.history_value_net = history_value_net.to(self.device)

        # Config fields (with reasonable defaults)
        self.num_abstract_actions = int(_cfg_get(config, "num_abstract_actions", NUM_ABSTRACT_ACTIONS_2P))
        if self.num_abstract_actions != NUM_ABSTRACT_ACTIONS_2P:
            raise ValueError(
                f"DESCAConfig.num_abstract_actions={self.num_abstract_actions} "
                f"does not match action_abstraction.NUM_ABSTRACT_ACTIONS_2P={NUM_ABSTRACT_ACTIONS_2P}"
            )
        self.total_iterations = int(_cfg_get(config, "iterations", 1000))
        self.traversals_per_iter = int(_cfg_get(config, "traversals_per_iter", 500))
        self.minibatch = int(_cfg_get(config, "minibatch", 1024))
        self.lr = float(_cfg_get(config, "lr", 3e-4))
        self.weight_decay = float(_cfg_get(config, "weight_decay", 1e-4))
        self.grad_clip = float(_cfg_get(config, "grad_clip", 1.0))
        self.dcfr_alpha = float(_cfg_get(config, "dcfr_alpha", 1.5))
        self.apcfr_asymmetry = float(_cfg_get(config, "apcfr_asymmetry", 0.9))
        self.buffer_capacity = int(_cfg_get(config, "buffer_capacity", 2_000_000))
        self.checkpoint_every = int(_cfg_get(config, "checkpoint_every", 50))
        self.eval_every = int(_cfg_get(config, "eval_every", 50))
        self.warmup_iters = int(_cfg_get(config, "warmup_iters", 50))
        self.inner_update = str(_cfg_get(config, "inner_update", "apcfr_plus"))
        if self.inner_update not in ("apcfr_plus", "rm_plus"):
            raise ValueError(
                f"inner_update must be 'apcfr_plus' or 'rm_plus'; got {self.inner_update!r}"
            )

        self.stall_cfg = _coerce_stall_config(_cfg_get(config, "stall_detection", None))
        self.use_bf16_inference: bool = bool(_cfg_get(config, "use_bf16_inference", True))

        # Reservoir buffers (per contract item 3.2). Regret + strategy share
        # (input_dim=257, target_dim=32). Value buffer uses (input=377, target=1).
        self.regret_buffer = ReservoirBuffer(
            capacity=self.buffer_capacity,
            input_dim=FEATURE_DIM,
            target_dim=self.num_abstract_actions,
            has_mask=True,
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=self.buffer_capacity,
            input_dim=FEATURE_DIM,
            target_dim=self.num_abstract_actions,
            has_mask=True,
        )
        self.value_buffer = ReservoirBuffer(
            capacity=self.buffer_capacity,
            input_dim=VALUE_INPUT_DIM,
            target_dim=1,
            has_mask=False,
        )

        # Optimizers
        self.regret_opt = torch.optim.AdamW(
            self.regret_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.strategy_opt = torch.optim.AdamW(
            self.avg_strategy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.value_opt = torch.optim.AdamW(
            self.history_value_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # RNG / iteration state
        self.iteration: int = 0
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
                random.seed(int(seed))
            except Exception:
                pass

        # Monitor history
        self._mean_imp_history: List[Tuple[int, float]] = []
        self._regret_target_std_history: List[float] = []
        self._v_loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._window_means: List[float] = []

        # Runtime SGD step counts per contract 3.3
        self._regret_steps_per_iter = int(_cfg_get(config, "regret_sgd_steps", 2000))
        self._strategy_steps_per_iter = int(_cfg_get(config, "strategy_sgd_steps", 2000))
        self._value_steps_per_iter = int(_cfg_get(config, "value_sgd_steps", 1000))

        # APCFR+ prev-grad stores (per Meng et al. Mar 2025 extrapolation
        # variant). Cleared at the top of each fit call. V_omni stays on
        # plain RM+ because it's a regression task with no predictive regime.
        self._regret_prev_grad = _PrevGradStore()
        self._strategy_prev_grad = _PrevGradStore()

        # DCFR+ weight LUT. Lazily extended up to the current max iteration
        # seen in a minibatch. Avoids the Python list-comp + math.pow per SGD
        # step in _fit_regret; see _get_dcfr_weights().
        self._dcfr_weight_lut: Optional[torch.Tensor] = None

        # run_db init (non-fatal)
        self._db_run_id: Optional[int] = None
        self._db_conn = None
        self._init_run_db()

    # ------------------------------------------------------------------
    # run_db
    # ------------------------------------------------------------------

    def _init_run_db(self) -> None:
        if not _RUN_DB_AVAILABLE:
            return
        try:
            ckpt_path = self.checkpoint_path
            save_dir = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else "."
            db_path = str(Path(save_dir).parent.parent / "cambia_runs.db")
            self._db_conn = _run_db.get_db(db_path)
            run_name = (
                Path(save_dir).parent.name
                if Path(save_dir).name == "checkpoints"
                else Path(save_dir).name
            )
            run_dir = (
                Path(save_dir).parent
                if Path(save_dir).name == "checkpoints"
                else Path(save_dir)
            )
            config_yaml = None
            config_dict: Dict[str, Any] = {"algorithm": "desca"}
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                try:
                    config_yaml = config_path.read_text(encoding="utf-8")
                    import yaml  # type: ignore[import-not-found]
                    loaded = yaml.safe_load(config_yaml) or {}
                    if isinstance(loaded, dict):
                        config_dict.update(loaded)
                except Exception:
                    pass
            algorithm = _run_db.infer_algorithm(config_dict) or "desca"
            self._db_run_id = _run_db.upsert_run(
                self._db_conn,
                name=run_name,
                algorithm=algorithm,
                config_yaml=config_yaml,
                config_dict=config_dict,
                status="running",
            )
            logger.info("DESCA run_db registered '%s' (id=%d)", run_name, self._db_run_id)
        except Exception as e:
            logger.debug("DESCA run_db init failed (non-fatal): %s", e)
            self._db_run_id = None
            self._db_conn = None

    # ------------------------------------------------------------------
    # Public training entry
    # ------------------------------------------------------------------

    def train(self, num_iterations: Optional[int] = None) -> None:
        """Run the DESCA training loop.

        Args:
            num_iterations: Overrides ``config.iterations`` when provided.
        """
        total = int(num_iterations) if num_iterations is not None else self.total_iterations
        safety_cap = int(self.stall_cfg["max_iter_abs"])

        for _ in range(total):
            self.iteration += 1
            if self.iteration > safety_cap:
                logger.warning(
                    "DESCA safety cap reached at iter %d; halting.", self.iteration
                )
                break

            self._run_iteration()

            if self._detect_stall():
                logger.warning("DESCA stall detected at iter %d; halting.", self.iteration)
                break

        # Final checkpoint
        try:
            self.save_checkpoint(final=True)
        except Exception as e:
            logger.warning("DESCA final checkpoint failed: %s", e)

        if self._db_conn is not None and self._db_run_id is not None:
            try:
                _run_db.update_run_status(self._db_conn, self._db_run_id, "completed")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-iteration driver
    # ------------------------------------------------------------------

    def _run_iteration(self) -> None:
        warmup = self.iteration <= self.warmup_iters
        logger.info(
            "DESCA iter %d start (warmup=%s, inner_update=%s)",
            self.iteration, warmup, self.inner_update,
        )

        # 1. Traversals per player
        for p in range(2):
            result = run_desca_iteration(
                self.env_factory,
                updating_player=p,
                regret_net=self.regret_net,
                avg_strategy_net=self.avg_strategy_net,
                history_value_net=self.history_value_net,
                iteration=self.iteration,
                traversals=self.traversals_per_iter,
                device=self.device,
                rng=self.rng,
                warmup=warmup,
                use_bf16=self.use_bf16_inference,
            )
            self._ingest_samples(
                result.regret_samples, result.strategy_samples, result.value_samples
            )
            logger.debug(
                "DESCA iter %d player %d: nodes=%d, terminals=%d, errors=%d",
                self.iteration, p, result.nodes_visited,
                result.terminals_reached, result.errors,
            )

        # 2. Fit networks (skipped if buffers are empty).
        regret_grad = self._fit_regret()
        strategy_grad = self._fit_strategy()
        value_grad = self._fit_value()

        # 3. Runtime monitors (Tier 4, spec 10.4). Fire every 10 iters.
        if self.iteration % 10 == 0:
            self._runtime_monitors(regret_grad, strategy_grad, value_grad)

        # 4. Iteration milestones (spec 10.5).
        self._iteration_milestone_log()

        # 5. Checkpoints.
        if self.iteration % self.checkpoint_every == 0:
            try:
                self.save_checkpoint()
            except Exception as e:
                logger.warning("DESCA checkpoint save failed at iter %d: %s",
                               self.iteration, e)

    # ------------------------------------------------------------------
    # Buffer ingestion
    # ------------------------------------------------------------------

    def _ingest_samples(
        self,
        regret: List[ReservoirSample],
        strategy: List[ReservoirSample],
        value: List[ReservoirSample],
    ) -> None:
        for s in regret:
            self.regret_buffer.add(s)
        for s in strategy:
            self.strategy_buffer.add(s)
        for s in value:
            self.value_buffer.add(s)

        # Diagnostic: track std of regret targets for runtime monitor.
        if regret:
            try:
                stacked = np.stack([s.target for s in regret])
                self._regret_target_std_history.append(float(np.std(stacked)))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Network fitting
    # ------------------------------------------------------------------

    def _fit_regret(self) -> float:
        if len(self.regret_buffer) == 0:
            return 0.0
        # Clear APCFR+ prev-grad cache so the extrapolation only chains within
        # this fit call. Stale prev grads across iterations would extrapolate
        # from a different reservoir distribution and amplify noise.
        self._regret_prev_grad.clear()
        last_grad = 0.0
        for _ in range(self._regret_steps_per_iter):
            batch = self.regret_buffer.sample_batch(self.minibatch)
            if len(batch) == 0:
                break
            x = torch.from_numpy(batch.features).to(self.device)
            y = torch.from_numpy(batch.targets).to(self.device)
            mask = (
                torch.from_numpy(batch.masks).to(self.device)
                if batch.masks is not None
                else None
            )
            iter_t = torch.from_numpy(
                np.asarray(batch.iterations, dtype=np.int64)
            ).to(self.device)
            w_iter = self._get_dcfr_weights(iter_t)
            pred = self.regret_net(x)
            loss = _regret_loss(pred, y, mask, w_iter)
            last_grad = self._run_step(
                self.regret_opt,
                self.regret_net.parameters(),
                loss,
                self._regret_prev_grad,
            )
        return last_grad

    def _fit_strategy(self) -> float:
        if len(self.strategy_buffer) == 0:
            return 0.0
        self._strategy_prev_grad.clear()
        last_grad = 0.0
        for _ in range(self._strategy_steps_per_iter):
            batch = self.strategy_buffer.sample_batch(self.minibatch)
            if len(batch) == 0:
                break
            x = torch.from_numpy(batch.features).to(self.device)
            y = torch.from_numpy(batch.targets).to(self.device)
            mask = (
                torch.from_numpy(batch.masks).to(self.device)
                if batch.masks is not None
                else None
            )
            w_iter = torch.tensor(
                [float(it) for it in batch.iterations],
                dtype=torch.float32,
                device=self.device,
            )
            if mask is None:
                raise RuntimeError(
                    "AvgStrategyNetwork requires action_mask but reservoir batch had None."
                    " strategy_buffer must be created with has_mask=True."
                )
            pred = self.avg_strategy_net(x, mask)
            loss = _strategy_loss(pred, y, mask, w_iter)
            last_grad = self._run_step(
                self.strategy_opt,
                self.avg_strategy_net.parameters(),
                loss,
                self._strategy_prev_grad,
            )
        return last_grad

    def _fit_value(self) -> float:
        if len(self.value_buffer) == 0:
            return 0.0
        last_grad = 0.0
        step_losses: List[float] = []
        for _ in range(self._value_steps_per_iter):
            batch = self.value_buffer.sample_batch(self.minibatch)
            if len(batch) == 0:
                break
            x_combined = torch.from_numpy(batch.features).to(self.device)
            y = torch.from_numpy(batch.targets).to(self.device)
            # Split into fair + omniscient.
            x_fair = x_combined[:, :FEATURE_DIM]
            x_omni = x_combined[:, FEATURE_DIM:]
            try:
                pred = self.history_value_net(x_fair, x_omni)
            except TypeError:
                pred = self.history_value_net(x_combined)
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            loss = _value_loss(pred, y)
            step_losses.append(float(loss.detach().cpu().item()))
            # V_omni fit always uses plain SGD-like Adam step (no APCFR+).
            last_grad = _rm_plus_step(
                self.value_opt,
                list(self.history_value_net.parameters()),
                loss,
                self.grad_clip,
            )
        avg_loss = float(np.mean(step_losses[-20:])) if step_losses else 0.0
        self._v_loss_history.append(avg_loss)
        return last_grad

    def _run_step(
        self,
        optimizer: "torch.optim.Optimizer",
        params: Iterable["torch.nn.Parameter"],
        loss: "torch.Tensor",
        prev_store: Optional[_PrevGradStore] = None,
    ) -> float:
        """Dispatch to APCFR+ or RM+ per config.

        ``prev_store`` is required for the APCFR+ path to cache prev-step
        gradients. The regret and strategy fit functions pass their own
        per-head stores; the V_omni fit calls ``_rm_plus_step`` directly and
        never reaches here.
        """
        params_list = list(params)
        if self.inner_update == "apcfr_plus":
            if prev_store is None:
                global _APCFR_FALLBACK_WARNED
                if not _APCFR_FALLBACK_WARNED:
                    _APCFR_FALLBACK_WARNED = True
                    logger.warning(
                        "APCFR+ requested but prev_store is None; falling back to RM+."
                        " This indicates a config wiring bug."
                    )
                return _rm_plus_step(optimizer, params_list, loss, self.grad_clip)
            return _apcfr_plus_step(
                optimizer,
                params_list,
                loss,
                self.apcfr_asymmetry,
                self.grad_clip,
                prev_store,
            )
        return _rm_plus_step(optimizer, params_list, loss, self.grad_clip)

    def _get_dcfr_weights(self, iterations: "torch.Tensor") -> "torch.Tensor":
        """Return DCFR+ weights for each iteration index via a precomputed LUT.

        LUT entry i = (i+1)^alpha / ((i+1)^alpha + 1), indexed 0-based so
        that iteration 1 maps to index 0, iteration N maps to index N-1.

        Lazily extends the LUT whenever ``iterations.max()`` exceeds the
        current size. Extension is O(new_size) and happens at most once per
        new maximum iteration seen across SGD steps.

        Args:
            iterations: 1-D integer tensor of CFR iteration indices (1-based).

        Returns:
            Float32 tensor of same shape as ``iterations``, on ``self.device``.

        Numerical guarantee: values match ``_dcfr_plus_weight(it, alpha)``
        within ``atol=1e-7`` over the range 1..3000 (the safety cap).
        """
        if torch is None:
            raise RuntimeError("torch required for _get_dcfr_weights")

        max_iter = int(iterations.max().item())
        needed_size = max_iter  # LUT[i] covers iteration i+1; need at least max_iter entries.

        if self._dcfr_weight_lut is None or self._dcfr_weight_lut.shape[0] < needed_size:
            # Extend to the next power of 2 past needed_size (or at least 1024)
            # to avoid repeated small re-allocations during warm-up.
            new_size = max(1024, needed_size)
            # LUT formula: lut[i] = (i+1)^alpha / ((i+1)^alpha + 1)
            indices = torch.arange(1, new_size + 1, dtype=torch.float64)
            powered = indices.pow(self.dcfr_alpha)
            lut = (powered / (powered + 1.0)).to(torch.float32).to(self.device)
            self._dcfr_weight_lut = lut

        # iterations are 1-based; LUT is 0-indexed (LUT[0] = w(iter=1)).
        indices_0based = (iterations.long() - 1).clamp(min=0)
        return self._dcfr_weight_lut[indices_0based]

    # ------------------------------------------------------------------
    # Tier 4 monitors and milestones
    # ------------------------------------------------------------------

    def _runtime_monitors(
        self, regret_grad: float, strategy_grad: float, value_grad: float
    ) -> None:
        """Fire Tier 4 runtime monitors per spec Section 10.4.

        This logs warnings on threshold violations. The authoritative halt
        criterion is ``_detect_stall``; individual monitors are informative.
        """
        # Regret target variance
        recent_std = (
            self._regret_target_std_history[-3:]
            if len(self._regret_target_std_history) >= 3
            else self._regret_target_std_history
        )
        if recent_std:
            if all(s < 0.01 for s in recent_std) and len(recent_std) >= 3:
                logger.error(
                    "Monitor: regret target std < 0.01 for 3 windows (values=%s)",
                    recent_std,
                )

        # Gradient norms
        for name, g in (
            ("regret", regret_grad),
            ("strategy", strategy_grad),
            ("value", value_grad),
        ):
            if g > 0.0 and not (1e-6 <= g <= 100.0):
                logger.warning(
                    "Monitor: %s grad_norm %.6g outside [1e-6, 100] at iter %d",
                    name, g, self.iteration,
                )
        self._grad_norm_history.append(
            float((regret_grad + strategy_grad + value_grad) / 3.0)
        )

        # V_omni loss (spec requires < 0.1 by iter 50)
        if self._v_loss_history and self.iteration >= 50:
            if self._v_loss_history[-1] > 0.1:
                logger.warning(
                    "Monitor: V_omni loss %.4f > 0.1 at iter %d (spec target ≤ 0.1 by iter 50)",
                    self._v_loss_history[-1], self.iteration,
                )

    def _iteration_milestone_log(self) -> None:
        """Log Tier 4 iteration milestones per spec 10.5."""
        milestone_map = {20: "early smoke", 100: "coverage",
                         200: "H2H trend", 300: "mean_imp target"}
        label = milestone_map.get(self.iteration)
        if label is None:
            return
        logger.info(
            "DESCA milestone iter %d (%s): regret_steps=%d, V_loss_last=%s",
            self.iteration, label, self._regret_steps_per_iter,
            f"{self._v_loss_history[-1]:.4f}" if self._v_loss_history else "n/a",
        )

    def _detect_stall(self) -> bool:
        """Return True when stall detection fires per spec 10.5.

        Stall = no >= 1pp mean_imp window-over-window improvement for
        ``num_windows`` consecutive windows, AND iteration >= 500.
        """
        min_iter_floor = 500
        if self.iteration < min_iter_floor:
            return False
        if len(self._window_means) < self.stall_cfg["num_windows"] + 1:
            return False
        tail = self._window_means[-(self.stall_cfg["num_windows"] + 1):]
        deltas = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        return all(d < 0.01 for d in deltas)

    def record_mean_imp(self, mean_imp: float) -> None:
        """Record a mean_imp measurement for window-based stall detection.

        Callers (e.g. eval_watcher integration) call this after each eval to
        update the window history.
        """
        self._mean_imp_history.append((self.iteration, float(mean_imp)))
        window_size = int(self.stall_cfg["window_size_iters"])
        if window_size <= 0:
            return
        # Recompute window means.
        windows: Dict[int, List[float]] = {}
        for (it, m) in self._mean_imp_history:
            w = it // window_size
            windows.setdefault(w, []).append(m)
        self._window_means = [float(np.mean(windows[k])) for k in sorted(windows.keys())]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, final: bool = False) -> str:
        """Save state to ``<base>_iter_<iteration>.pt`` and register with run_db."""
        base_dir = os.path.dirname(self.checkpoint_path) or "."
        os.makedirs(base_dir, exist_ok=True)
        base = os.path.splitext(self.checkpoint_path)[0]
        iter_path = f"{base}_iter_{self.iteration}.pt"

        cfg_dict = _cfg_as_dict(self.config)
        cfg_dict.setdefault("algorithm", "desca")
        cfg_dict.setdefault("inner_update", self.inner_update)

        payload = {
            "algorithm": "desca",
            "iteration": self.iteration,
            "regret_state_dict": self.regret_net.state_dict(),
            "avg_strategy_state_dict": self.avg_strategy_net.state_dict(),
            "history_value_state_dict": self.history_value_net.state_dict(),
            "regret_optimizer_state_dict": self.regret_opt.state_dict(),
            "strategy_optimizer_state_dict": self.strategy_opt.state_dict(),
            "value_optimizer_state_dict": self.value_opt.state_dict(),
            "desca_config": {
                "hidden_dim": int(_cfg_get(self.config, "hidden_dim", 512)),
                "encoding_dim": FEATURE_DIM,
                "num_abstract_actions": self.num_abstract_actions,
                "omniscient_dim": OMNISCIENT_DIM_2P,
                "inner_update": self.inner_update,
            },
            "desca_state_dict": {
                "iteration": self.iteration,
                "inner_update": self.inner_update,
                "config": cfg_dict,
                "rng_state": self.rng.bit_generator.state,
                "v_loss_history": self._v_loss_history[-50:],
                "regret_target_std_history": self._regret_target_std_history[-50:],
            },
        }

        atomic_torch_save(payload, iter_path)
        # Also write the canonical latest pointer.
        try:
            atomic_torch_save(payload, self.checkpoint_path)
        except Exception:
            pass
        logger.info("DESCA checkpoint saved to %s%s", iter_path, " (final)" if final else "")

        if self._db_conn is not None and self._db_run_id is not None:
            try:
                _run_db.register_checkpoint(
                    self._db_conn, self._db_run_id, self.iteration, iter_path
                )
                _run_db.compute_retention_flags(self._db_conn, self._db_run_id)
            except Exception as e:
                logger.debug("DESCA run_db register_checkpoint failed: %s", e)

        return iter_path

    def load_checkpoint(self, path: str) -> None:
        """Load state from a checkpoint previously saved by ``save_checkpoint``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.regret_net.load_state_dict(ckpt["regret_state_dict"])
        self.avg_strategy_net.load_state_dict(ckpt["avg_strategy_state_dict"])
        self.history_value_net.load_state_dict(ckpt["history_value_state_dict"])
        if "regret_optimizer_state_dict" in ckpt:
            self.regret_opt.load_state_dict(ckpt["regret_optimizer_state_dict"])
        if "strategy_optimizer_state_dict" in ckpt:
            self.strategy_opt.load_state_dict(ckpt["strategy_optimizer_state_dict"])
        if "value_optimizer_state_dict" in ckpt:
            self.value_opt.load_state_dict(ckpt["value_optimizer_state_dict"])
        desca_state = ckpt.get("desca_state_dict", {})
        self.iteration = int(desca_state.get("iteration", ckpt.get("iteration", 0)))
        if "rng_state" in desca_state:
            try:
                self.rng.bit_generator.state = desca_state["rng_state"]
            except Exception:
                pass


__all__ = ["DESCATrainer"]
