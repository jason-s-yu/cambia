"""src/cfr/prtcfr_trainer.py

Minimal neural PRT-CFR trainer for the Phase 1 X2 gate (tiny {A,6} 2-card game).

Iteration loop (v0.4 trainer conventions, RC-E remainder / RC-F):
  - collect K external-sampling traversals into a uniform reservoir that
    ACCUMULATES across iterations (every iteration's samples persist, tagged with
    the iteration t);
  - REFIT: re-initialize PRTCFRNet every iteration (Brown 2019 from-scratch
    convention, the default) OR warm-start from the previous iterate when
    config.warm_start is set; Adam + cosine-decayed lr inside the fit;
  - loss = MSE SUMMED over legal actions per sample (the /num_legal starvation is
    deleted), with normalized linear sample weighting w = t_sample/mean(t_batch);
  - gradient clip applied LAST (after unscale, before optimizer step);
  - no APCFR+, no DCFR+ discounting; warm-start is config-gated (default off).

After each fit:
  - SD-CFR snapshot prtcfr_snapshot_iter_{t}.pt = {encoder_state_dict,
    head_state_dict, iteration} (decision 4: the average strategy is realized
    exactly by snapshot sampling, so every iterate's regret net is preserved);
  - a rolling checkpoint prtcfr_checkpoint.pt = {encoder_state_dict,
    head_state_dict, config, iteration}.

The current iterate sigma^t (used by the worker for rollouts and trajectory
sampling) is the regret-matched strategy of the CURRENT net. On the tiny tree it
is precomputed once per iteration over all decision nodes (one batched forward),
so the worker's per-node sigma lookup is O(1) and exactly the fixed sigma^t.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..encoding import NUM_ACTIONS, action_to_index, encode_action_mask
from ..reservoir import ReservoirBuffer
from ..sequence_encoding import SEQ_CAP
from .prtcfr_net import (
    PRTCFRNet,
    _regret_match,
    build_prtcfr_net,
    tiny_node_to_token_array,
    tiny_node_to_tokens,
)
from .prtcfr_stability import BestSnapshotController, write_deployable_manifest
from .prtcfr_worker import PRTCFRWorker

logger = logging.getLogger(__name__)


def _peak_lr_for_iter(
    lr: float, lr_min: float, t: int, total_iters: int, schedule: str
) -> float:
    """Per-iteration peak LR for the fit at iteration ``t`` (1-based).

    ``schedule="restart"`` (default) returns ``lr`` unchanged every iteration --
    the original per-iteration cosine warm-restart to the same peak, which keeps
    the effective step size constant across iterations. Near equilibrium the
    regret targets are MC-noise-dominated (the estimator floor does not shrink as
    the strategy converges), so a warm-started net keeps taking full-size steps
    that fit that noise; the linear reservoir/SD-CFR recency weighting then
    amplifies the late overfit into the divergence.

    ``schedule="global_cosine"`` decays the per-iteration PEAK across the whole
    run (cosine from ``lr`` at t=1 to ``lr_min`` at t=total_iters), a
    Robbins-Monro decreasing step size: late iterations take small steps and can
    no longer overfit the near-equilibrium noise, while early iterations still
    converge fast. The within-iteration cosine (to ``lr_min``) is unchanged.
    """
    if schedule == "restart":
        return lr
    if schedule == "global_cosine":
        if total_iters <= 1:
            return lr
        frac = (t - 1) / (total_iters - 1)
        frac = min(max(frac, 0.0), 1.0)
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


# ---------------------------------------------------------------------------
# sigma^t provider: regret-matched strategy of the current net, precomputed once
# per iteration over the tiny tree's decision nodes.
# ---------------------------------------------------------------------------


def _collect_decision_nodes(root) -> List[object]:
    """All Decision nodes reachable from ``root`` (DFS), de-duplicated by identity."""
    seen = set()
    out: List[object] = []

    def rec(node):
        if node.kind == "T":
            return
        if node.kind == "C":
            for c in node.children:
                rec(c)
            return
        if id(node) not in seen:
            seen.add(id(node))
            out.append(node)
        for c in node.children:
            rec(c)

    rec(root)
    return out


class NetSigmaProvider:
    """Per-iteration sigma^t over the tiny tree's decision nodes.

    Built once from the current net: one batched forward over every decision
    node's padded token array yields raw advantages, regret-matched (with the
    node's 146-mask) to a strategy, then restricted to the node's legal-action
    order (node.actions). ``policy(node)`` returns the length-nA vector the worker
    needs. The strategy is FIXED for the iteration (identity-keyed cache).
    """

    def __init__(
        self,
        net: PRTCFRNet,
        decision_nodes: List[object],
        seq_cap: int = SEQ_CAP,
        infer_batch: int = 4096,
    ):
        self.seq_cap = seq_cap
        # Per-node legal action index lists (into the 146 space), node.actions order.
        self._global_idx: Dict[int, List[int]] = {}
        feats = np.empty((len(decision_nodes), seq_cap), dtype=np.int64)
        masks = np.zeros((len(decision_nodes), NUM_ACTIONS), dtype=bool)
        for r, node in enumerate(decision_nodes):
            feats[r] = tiny_node_to_token_array(node, seq_cap=seq_cap)
            gidx = [action_to_index(a) for a in node.actions]
            self._global_idx[id(node)] = gidx
            masks[r, gidx] = True

        device = net.device
        net.eval()
        strat_rows = np.empty((len(decision_nodes), NUM_ACTIONS), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(decision_nodes), infer_batch):
                e = min(s + infer_batch, len(decision_nodes))
                tk = torch.from_numpy(feats[s:e]).to(device)
                mk = torch.from_numpy(masks[s:e]).to(device)
                adv = net.raw_advantages(tk, mk)
                strat = _regret_match(adv, mk)
                strat_rows[s:e] = strat.detach().to("cpu", dtype=torch.float64).numpy()

        # Cache the per-node legal-order probability vector.
        self._policy: Dict[int, np.ndarray] = {}
        for r, node in enumerate(decision_nodes):
            gidx = self._global_idx[id(node)]
            vec = strat_rows[r, gidx].astype(np.float64)
            tot = vec.sum()
            if tot <= 0:
                vec = np.full(len(gidx), 1.0 / len(gidx), dtype=np.float64)
            else:
                vec = vec / tot
            self._policy[id(node)] = vec

    def policy(self, node) -> np.ndarray:
        v = self._policy.get(id(node))
        if v is None:
            n = len(node.actions)
            return np.full(n, 1.0 / n, dtype=np.float64)
        return v


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class PRTCFRTrainState:
    """Lightweight record of a completed iteration (returned for logging/tests)."""

    iteration: int
    samples_added: int
    buffer_size: int
    fit_loss: float
    snapshot_path: str


def _fit_from_scratch(
    net: PRTCFRNet,
    buf: ReservoirBuffer,
    lr: float,
    batch_size: int,
    num_steps: int,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    lr_min: float = 0.0,
) -> float:
    """Refit ``net`` (already freshly initialized) on the reservoir.

    Loss = normalized-linear-weighted, masked-SUM MSE over legal actions. Adam +
    cosine-decayed lr across ``num_steps`` (floored at ``lr_min``, default 0.0 =
    the original decay-to-zero). Gradient clip applied LAST. Returns the mean
    weighted loss over executed steps.
    """
    if len(buf) == 0:
        return 0.0
    device = net.device
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_steps, 1), eta_min=lr_min
    )

    total_loss = 0.0
    steps = 0
    for _step in range(num_steps):
        batch = buf.sample_batch(batch_size)
        if not batch:
            break
        # features are token ids stored as float32 in the reservoir; cast to long.
        feats = torch.from_numpy(batch.features).to(device).long()
        targets = torch.from_numpy(batch.targets).float().to(device)
        masks = torch.from_numpy(batch.masks).to(device)
        iters = torch.from_numpy(batch.iterations.astype(np.float32)).to(device)

        # Normalized LINEAR weighting: w = t / mean(t) over the batch.
        weights = iters / iters.mean().clamp(min=1e-8)

        optimizer.zero_grad(set_to_none=True)
        preds = net.raw_advantages(feats, masks)
        masked_preds = preds.masked_fill(~masks, 0.0)
        masked_targets = targets.masked_fill(~masks, 0.0)
        # SUM over legal actions (no /num_legal); mean over the (weighted) batch.
        per_sample = ((masked_preds - masked_targets) ** 2).sum(dim=1)
        loss = (weights * per_sample).mean()

        loss.backward()
        # Clip applied LAST, immediately before the optimizer step.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


class PRTCFRTinyTrainer:
    """Drives PRT-CFR training on the tiny_solver tree to produce X2 snapshots."""

    def __init__(
        self,
        root,
        config,
        snapshot_dir: str,
        net_factory: Optional[Callable[[], PRTCFRNet]] = None,
        eval_fn: Optional[Callable[["PRTCFRTinyTrainer", int], float]] = None,
    ):
        self.root = root
        self.config = config
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

        self.seq_cap = int(getattr(config, "seq_cap", SEQ_CAP))
        self.m_rollouts = int(getattr(config, "m_rollouts", 4))
        self.k_games = int(getattr(config, "k_games_per_iter", 200))
        self.iterations = int(getattr(config, "iterations", 100))
        self.lr = float(getattr(config, "lr", 1e-3))
        self.batch_size = int(getattr(config, "batch_size", 1024))
        self.buffer_capacity = int(getattr(config, "buffer_capacity", 2_000_000))
        self.weight_decay = float(getattr(config, "weight_decay", 0.0))
        self.grad_clip = float(getattr(config, "grad_clip", 1.0))
        self.train_steps = int(getattr(config, "train_steps_per_iter", 256))
        self.warm_start = bool(getattr(config, "warm_start", False))
        self.seed = int(getattr(config, "seed", 0))
        self.device = getattr(config, "device", "cuda")

        # --- Late-training stability (all config-gated, defaults reproduce the
        # Phase 1 gate byte-for-byte). ---
        # Optimizer-side fix for the warm-start blow-up: a global (across-run) LR
        # decay so late near-equilibrium iterations take small steps.
        self.lr_schedule = str(getattr(config, "lr_schedule", "restart"))
        self.lr_min = float(getattr(config, "lr_min", 0.0))
        # Periodic re-anchor: re-initialize the net from scratch every N iters even
        # under warm_start (0 = never; breaks the warm-start error accumulation).
        self.reanchor_every = int(getattr(config, "reanchor_every", 0))
        # Best-snapshot / early-stop controller over the exploitability trend.
        self.stability_enabled = bool(getattr(config, "stability_enabled", False))
        self.stability_eval_every = int(getattr(config, "stability_eval_every", 10))
        self.stability_patience = int(getattr(config, "stability_patience", 3))
        self.stability_rel_tolerance = float(
            getattr(config, "stability_rel_tolerance", 0.15)
        )
        self.stability_min_iters = int(
            getattr(config, "stability_min_iters", self.stability_eval_every)
        )
        self.stability_metric_mode = str(getattr(config, "stability_metric_mode", "min"))
        self.stability_metric_name = str(
            getattr(config, "stability_metric_name", "nashconv")
        )
        self.eval_fn = eval_fn
        self.controller: Optional[BestSnapshotController] = None
        if self.stability_enabled:
            self.controller = BestSnapshotController(
                rel_tolerance=self.stability_rel_tolerance,
                patience=self.stability_patience,
                min_iters=self.stability_min_iters,
                mode=self.stability_metric_mode,
            )

        self._net_factory = net_factory or (
            lambda: build_prtcfr_net(config, device=self.device)
        )
        # Reservoir feature width = token width (seq_cap); target = 146; masked.
        self.buffer = ReservoirBuffer(
            capacity=self.buffer_capacity,
            input_dim=self.seq_cap,
            target_dim=NUM_ACTIONS,
            has_mask=True,
        )
        self.decision_nodes = _collect_decision_nodes(root)
        # Current net (refit each iteration). Initialized once; re-created per
        # iteration inside train() so iteration 1's sigma uses a fresh init.
        self.net: Optional[PRTCFRNet] = None

    def snapshot_path(self, t: int) -> str:
        return os.path.join(self.snapshot_dir, f"prtcfr_snapshot_iter_{t}.pt")

    def checkpoint_path(self) -> str:
        return os.path.join(self.snapshot_dir, "prtcfr_checkpoint.pt")

    def _config_dict(self) -> dict:
        if hasattr(self.config, "model_dump"):
            try:
                return self.config.model_dump()
            except Exception:
                pass
        return {
            "seq_cap": self.seq_cap,
            "m_rollouts": self.m_rollouts,
            "k_games_per_iter": self.k_games,
            "iterations": self.iterations,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
        }

    def _save_snapshot(self, t: int) -> str:
        path = self.snapshot_path(t)
        torch.save(
            {
                "encoder_state_dict": self.net.encoder_state_dict(),
                "head_state_dict": self.net.head_state_dict(),
                "iteration": t,
            },
            path,
        )
        return path

    def _save_checkpoint(self, t: int) -> None:
        torch.save(
            {
                "encoder_state_dict": self.net.encoder_state_dict(),
                "head_state_dict": self.net.head_state_dict(),
                "config": self._config_dict(),
                "iteration": t,
            },
            self.checkpoint_path(),
        )

    def run_iteration(self, t: int) -> PRTCFRTrainState:
        """One PRT-CFR iteration: traverse under sigma^t, accumulate samples,
        refit a fresh net from scratch, snapshot + checkpoint.

        CFR recurrence: sigma^t is the regret-matched strategy of the regret net
        trained at iteration t-1 (the net approximates the cumulative regret R^t
        via the linear t-weighting on the accumulated reservoir, so regret-match of
        it is the current iterate). At t=1 there is no prior net, so sigma^1 is the
        regret-match of a freshly initialized net (effectively near-uniform). The
        fit re-inits each iteration by default (Brown 2019); with config.warm_start
        it fine-tunes the previous iterate's net instead (same fixed point R^t
        given convergence, fewer steps to reach it).
        """
        # sigma^t: previous iteration's fitted net (uniform-ish fresh net at t=1).
        if self.net is None:
            self.net = self._net_factory()
        sigma = NetSigmaProvider(self.net, self.decision_nodes, seq_cap=self.seq_cap)

        worker = PRTCFRWorker(
            self.root,
            sigma.policy,
            m_rollouts=self.m_rollouts,
            seq_cap=self.seq_cap,
            seed=self.seed + t * 1_000_003,
        )

        added = 0
        for k in range(self.k_games):
            traverser = (t + k) % 2  # alternate traverser across traversals
            added += worker.traverse(traverser, t, self.buffer)

        # Refit on the accumulated, t-weighted reservoir; the result is
        # sigma^{t+1}'s source and the iteration-t SD-CFR snapshot. From-scratch
        # (the default) re-inits each iteration; warm-start keeps the previous
        # iterate's net and fine-tunes it. Both converge to the same fixed-point
        # regret map R^t (init only changes the optimization path, given the fit
        # converges); warm-start reaches it with fewer steps -- the from-scratch
        # underfit fix the capacity probe pointed at.
        # Re-init from scratch when warm_start is off (Brown 2019), or when the
        # periodic re-anchor fires (breaks warm-start error accumulation).
        reanchor = self.reanchor_every > 0 and t % self.reanchor_every == 0
        if (not self.warm_start) or reanchor:
            self.net = self._net_factory()
        peak_lr = _peak_lr_for_iter(
            self.lr, self.lr_min, t, self.iterations, self.lr_schedule
        )
        loss = _fit_from_scratch(
            self.net,
            self.buffer,
            lr=peak_lr,
            batch_size=self.batch_size,
            num_steps=self.train_steps,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            lr_min=self.lr_min,
        )

        snap = self._save_snapshot(t)
        self._save_checkpoint(t)
        return PRTCFRTrainState(
            iteration=t,
            samples_added=added,
            buffer_size=len(self.buffer),
            fit_loss=loss,
            snapshot_path=snap,
        )

    def _stability_check_due(self, t: int, n: int) -> bool:
        return t == 1 or t % self.stability_eval_every == 0 or t == n

    def train(self, iterations: Optional[int] = None) -> List[PRTCFRTrainState]:
        """Run ``iterations`` (default config.iterations) PRT-CFR iterations.

        With ``stability_enabled`` and an ``eval_fn``, the trend metric is scored
        at the stability cadence, fed to the best-snapshot controller, and the
        deployable manifest is (re)written each check; training early-stops once
        the metric has risen past the tolerance band for ``patience`` checks. The
        deployable snapshot set pins to ``[1 .. best_iteration]`` so the served
        SD-CFR average excludes the diverged tail regardless.
        """
        n = iterations if iterations is not None else self.iterations
        # The global LR schedule spans the run actually requested.
        self.iterations = n
        history: List[PRTCFRTrainState] = []
        written_iters: List[int] = []
        for t in range(1, n + 1):
            st = self.run_iteration(t)
            history.append(st)
            written_iters.append(t)
            logger.info(
                "[prtcfr] iter=%d samples+=%d buffer=%d fit_loss=%.5f snapshot=%s",
                st.iteration, st.samples_added, st.buffer_size, st.fit_loss,
                os.path.basename(st.snapshot_path),
            )

            if not self.stability_enabled or self.controller is None:
                continue
            if not self._stability_check_due(t, n):
                continue
            if self.eval_fn is None:
                # No trend metric available: keep the whole set deployable.
                write_deployable_manifest(
                    self.snapshot_dir, self.controller, written_iters,
                    metric_name=self.stability_metric_name, stopped_early=False,
                )
                continue
            metric = float(self.eval_fn(self, t))
            decision = self.controller.update(t, metric)
            logger.info(
                "[prtcfr] stability iter=%d %s=%.5f best_iter=%d best=%.5f "
                "worse_streak=%d stop=%s",
                t, self.stability_metric_name, metric, decision.best_iteration,
                decision.best_metric, decision.num_worse_since_best,
                decision.should_stop,
            )
            write_deployable_manifest(
                self.snapshot_dir, self.controller, written_iters,
                metric_name=self.stability_metric_name,
                stopped_early=decision.should_stop,
            )
            if decision.should_stop:
                logger.info(
                    "[prtcfr] early-stop at iter=%d; deployable window pinned to "
                    "[1..%d] (%s=%.5f)",
                    t, decision.best_iteration, self.stability_metric_name,
                    decision.best_metric,
                )
                break
        return history


def train_tiny_prtcfr(
    config_path: str,
    snapshot_dir: str,
    iterations: Optional[int] = None,
    deals: int = 5,
    seed0: int = 0,
    seq_cap: int = SEQ_CAP,
    config_overrides: Optional[dict] = None,
) -> List[PRTCFRTrainState]:
    """Build the tiny perfect-recall tree and run PRT-CFR training on it.

    The function @chief calls to launch the X2 training run. Builds the tiny
    {A,6} 2-card tree (the X1-validated substrate) with tokenize=True, constructs
    a PRTCFRConfig (config's prt_cfr block overlaid with ``config_overrides``),
    trains, and writes snapshots + a rolling checkpoint to ``snapshot_dir``.
    Returns the per-iteration training history.

    Example (from cfr/):
        python -c "from src.cfr.prtcfr_trainer import train_tiny_prtcfr; \
            train_tiny_prtcfr('config/tiny_2card_plateau.yaml', \
            'runs/v0.4-prtcfr-pilot/snapshots', iterations=100)"
    """
    from src.config import load_config, PRTCFRConfig

    full_cfg = load_config(config_path)
    prt_cfg = getattr(full_cfg, "prt_cfr", None)
    if prt_cfg is None:
        prt_cfg = PRTCFRConfig()
    if config_overrides:
        merged = prt_cfg.model_dump()
        merged.update(config_overrides)
        prt_cfg = PRTCFRConfig(**merged)
    # seq_cap argument overrides the config field if explicitly different.
    if seq_cap is not None:
        prt_cfg = PRTCFRConfig(**{**prt_cfg.model_dump(), "seq_cap": seq_cap})

    # Build the tiny tree with tokenization (single-sourced seq_tokens on nodes).
    from tools.tiny_solver import build_tree

    root, _isets, n_nodes, aborted = build_tree(
        full_cfg,
        deals,
        seed0,
        2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=True,
        seq_cap=prt_cfg.seq_cap,
    )
    logger.info(
        "[prtcfr] tiny tree built: nodes~%d aborted_deals=%d seq_cap=%d",
        n_nodes, aborted, prt_cfg.seq_cap,
    )

    trainer = PRTCFRTinyTrainer(root, prt_cfg, snapshot_dir)
    return trainer.train(iterations=iterations)
