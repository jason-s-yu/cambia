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

import glob
import json
import logging
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..constants import ActionCallCambia
from ..encoding import NUM_ACTIONS, action_to_index, encode_action_mask
from ..reservoir import ColumnarBatch, ReservoirBuffer, ReservoirSample
from ..sequence_encoding import PAD_ID, SEQ_CAP, TOKENIZER_VERSION, VOCAB_SIZE
from .prtcfr_net import (
    PRTCFRNet,
    _regret_match,
    build_prtcfr_net,
    pad_tokens,
    tiny_node_to_token_array,
    tiny_node_to_tokens,
)
from .prtcfr_stability import BestSnapshotController, write_deployable_manifest
from .prtcfr_worker import (
    PRODUCTION_SEQ_CAP,
    GameDriver,
    IncrementalSigmaManager,
    PRTCFRBatchedProductionWorker,
    PRTCFRProductionWorker,
    PRTCFRWorker,
    new_production_driver,
)

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
    violation_box: Optional[List[int]] = None,
) -> float:
    """Refit ``net`` (already freshly initialized) on the reservoir.

    Loss = normalized-linear-weighted, masked-SUM MSE over legal actions. Adam +
    cosine-decayed lr across ``num_steps`` (floored at ``lr_min``, default 0.0 =
    the original decay-to-zero). Gradient clip applied LAST. Returns the mean
    weighted loss over executed steps.

    ``violation_box`` (optional single-element ``[int]``): when supplied, each
    fit step whose PRE-clip total grad norm exceeds ``grad_clip`` increments
    ``violation_box[0]`` (the AC2 grad-norm-violation counter, S2W1). Default
    None leaves the counter untouched, so the tiny trainer's call site is
    byte-for-byte unchanged.
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
        # clip_grad_norm_ returns the PRE-clip total norm; a value above
        # grad_clip means clipping fired -> one AC2 grad-norm violation.
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        if violation_box is not None and float(total_norm) > grad_clip:
            violation_box[0] += 1
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Tiny-gate full-state persistence (bit-exact warm-start / resume).
# ---------------------------------------------------------------------------
#
# The tiny trainer's cross-iteration state, and where each piece is captured so
# a fresh trainer can continue iteration (j+1) exactly as an uninterrupted run
# would:
#   - net weights -> the rolling checkpoint prtcfr_checkpoint.pt (Phase-1 dict).
#   - reservoir contents + seen_count -> <run_dir>/reservoir.npz (ReservoirBuffer
#     .save/.load; the in-RAM buffer has no dedicated RNG, so nothing else there).
#   - BestSnapshotController fields -> resume_state.json (via _controller_to_dict).
#   - GLOBAL RNG -> resume_state.json. The fit consumes numpy (np.random.choice
#     in ReservoirBuffer.sample_batch) and torch (GRU inter-layer dropout, plus
#     net init on from-scratch/reanchor) every iteration; buffer.add draws python
#     `random` only at capacity. The worker seeds its OWN random.Random per t, so
#     it needs no capture. All three global streams + the accelerator RNG are
#     stored so the refit stream is reproduced bit-for-bit.
#   - iteration counter + written-snapshot list -> resume_state.json.
# The peak LR and the per-iteration optimizer/scheduler are pure functions of t,
# self.iterations, and the schedule (both are rebuilt each iteration), so they
# carry no state beyond the iteration counter and the requested total.

TINY_RESUME_SCHEMA_VERSION = 1

# Token-embedding row count -> tokenizer version, for naming a resume-time vintage
# mismatch (cambia-612). The embedding grew 325 (v1) -> 326 (v2, cambia-529 peek)
# -> 327 (v3, cambia-564 race); the current live vocab is VOCAB_SIZE. Only the
# vocab-changing bumps appear here (F1's post-draw reshuffle was vocab-invariant).
_VOCAB_ROWS_TO_TOKENIZER_VERSION = {325: 1, 326: 2, 327: 3}


def _assert_resume_tokenizer_compatible(payload: Dict[str, Any]) -> None:
    """Refuse to resume from a checkpoint whose token embedding was trained under a
    different tokenizer vocab, raising a clear tokenizer-version provenance error
    instead of the raw ``load_state_dict`` size mismatch that surfaces deeper in
    ``load_encoder_head`` (cambia-612). Shape-driven (mirrors prtcfr_eval._load_net,
    cambia-341): the checkpoint's vocab is read from the saved ``embed.weight`` row
    count, so no stamped field is required and the pinned checkpoint format is
    untouched. A same-vocab checkpoint passes through unchanged."""
    enc = payload.get("encoder_state_dict") or {}
    emb = enc.get("embed.weight")
    if emb is None:
        return  # unexpected shape; let the real load surface it
    ckpt_vocab = int(emb.shape[0])
    if ckpt_vocab == VOCAB_SIZE:
        return
    ckpt_ver = _VOCAB_ROWS_TO_TOKENIZER_VERSION.get(ckpt_vocab)
    ckpt_desc = f"v{ckpt_ver}" if ckpt_ver else f"vocab={ckpt_vocab}"
    raise PRTCFRResumeError(
        f"TOKENIZER-VERSION MISMATCH on resume: this checkpoint's token embedding "
        f"has {ckpt_vocab} rows ({ckpt_desc}) but the live tokenizer is "
        f"v{TOKENIZER_VERSION} ({VOCAB_SIZE} rows). Resuming would fine-tune a net "
        f"on a token stream it was not built for. Resume with the training-era "
        f"code, or start a fresh run under the current tokenizer. Refusing to load "
        f"(cambia-612)."
    )


def _global_rng_save() -> Dict[str, Any]:
    """Capture the process-global numpy + python `random` + torch RNG streams.

    JSON-safe. numpy's MT19937 key array and python's internal-state tuple are
    stored as plain int lists; torch's CPU generator state as a hex byte string.
    The accelerator (cuda or xpu) RNG is captured separately via _accel_rng_save.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "numpy_rng": [
            np_state[0],
            [int(x) for x in np_state[1]],
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ],
        "python_rng": [
            int(py_state[0]),
            [int(x) for x in py_state[1]],
            None if py_state[2] is None else float(py_state[2]),
        ],
        "torch_rng": torch.random.get_rng_state().numpy().tobytes().hex(),
    }


def _global_rng_restore(state: Dict[str, Any]) -> None:
    """Restore the streams captured by _global_rng_save, then the accelerator RNG.

    Restored LAST in the load path so any RNG the load itself might touch does
    not leak into the resumed stream.
    """
    np_rng = state.get("numpy_rng")
    if np_rng is not None:
        np.random.set_state(
            (
                np_rng[0],
                np.array(np_rng[1], dtype=np.uint32),
                int(np_rng[2]),
                int(np_rng[3]),
                float(np_rng[4]),
            )
        )
    py_rng = state.get("python_rng")
    if py_rng is not None:
        random.setstate((int(py_rng[0]), tuple(int(x) for x in py_rng[1]), py_rng[2]))
    torch_hex = state.get("torch_rng")
    if torch_hex is not None:
        torch_bytes = np.frombuffer(bytes.fromhex(torch_hex), dtype=np.uint8).copy()
        torch.random.set_rng_state(torch.from_numpy(torch_bytes))
    _accel_rng_restore(state)


class PRTCFRTinyTrainer:
    """Drives PRT-CFR training on the tiny_solver tree to produce X2 snapshots.

    Two operating modes, chosen at construction:
      - Legacy (only ``snapshot_dir`` given): snapshots + rolling checkpoint go
        to ``snapshot_dir``; no run_db, no resume_state.json, no reservoir save.
        Byte-for-byte the original behavior the X2 tests and the scratch driver
        depend on.
      - Run-dir (``run_dir`` given): the harness-runnable path. Snapshots ->
        ``<run_dir>/snapshots``; the production-format resume_state.json +
        reservoir.npz are written every iteration so ``--resume`` and
        ``warm_start_path`` continuations hold; run_db journaling activates when
        ``run_name`` is supplied.
    """

    def __init__(
        self,
        root,
        config,
        snapshot_dir: Optional[str] = None,
        net_factory: Optional[Callable[[], PRTCFRNet]] = None,
        eval_fn: Optional[Callable[["PRTCFRTinyTrainer", int], float]] = None,
        run_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        db_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        config_dict: Optional[dict] = None,
        warm_start_path: Optional[str] = None,
    ):
        self.root = root
        self.config = config
        # Run-dir mode (harness-runnable) vs legacy snapshot-dir mode. When
        # run_dir is given, snapshots default to <run_dir>/snapshots and the
        # resume_state.json / reservoir.npz / run_db machinery activates.
        self.run_dir = run_dir
        if snapshot_dir is None:
            if run_dir is None:
                raise ValueError("PRTCFRTinyTrainer needs snapshot_dir or run_dir")
            snapshot_dir = os.path.join(run_dir, "snapshots")
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
        # Persistence (resume_state.json + reservoir.npz every iteration) is a
        # run-dir feature; legacy snapshot-dir-only construction stays side-effect
        # identical to the original trainer.
        self._persist_resume = run_dir is not None
        self.warm_start_path = (
            warm_start_path
            if warm_start_path is not None
            else getattr(config, "warm_start_path", None)
        )
        # cambia-374: explicit opt-in to warm-start from a bare .pt (net weights
        # + iteration only, empty reservoir). See _resolve_warm_start.
        self.warm_start_net_only_ok = bool(
            getattr(config, "warm_start_net_only_ok", False)
        )

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
        # "cpu" is the safe universal fallback for a config object that lacks
        # a device attribute entirely (duck-typed test configs); the real
        # production path always sets config.device via _resolve_device
        # before constructing this trainer, so this default is not hit there.
        self.device = getattr(config, "device", "cpu")

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
        self.stability_stop_mode = str(
            getattr(config, "stability_stop_mode", "divergence")
        )
        self.stability_plateau_window_iters = int(
            getattr(config, "stability_plateau_window_iters", 50)
        )
        self.stability_plateau_step_iters = int(
            getattr(config, "stability_plateau_step_iters", 10)
        )
        self.stability_plateau_rel_improvement = float(
            getattr(config, "stability_plateau_rel_improvement", 0.005)
        )
        self.eval_fn = eval_fn
        self.controller: Optional[BestSnapshotController] = None
        if self.stability_enabled:
            self.controller = BestSnapshotController(
                rel_tolerance=self.stability_rel_tolerance,
                patience=self.stability_patience,
                min_iters=self.stability_min_iters,
                mode=self.stability_metric_mode,
                stop_mode=self.stability_stop_mode,
                plateau_window_iters=self.stability_plateau_window_iters,
                plateau_step_iters=self.stability_plateau_step_iters,
                plateau_rel_improvement=self.stability_plateau_rel_improvement,
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
        # Iterations whose snapshot is on disk (drives the deployable manifest);
        # restored on resume/warm-start so the SD-CFR window survives.
        self._written_iters: List[int] = []

        # run_db journaling (run-dir mode with a run_name only; never fatal).
        self._db_conn = None
        self._db_run_id: Optional[int] = None
        # iteration -> checkpoints.id, populated by _register_checkpoint_in_db;
        # lets the stability-best mirror (cambia-390) flip is_best without a
        # re-query, since a checkpoint for iteration t is always registered
        # (run_iteration) before that iteration's stability check runs (train).
        self._db_ckpt_ids: Dict[int, int] = {}
        if self._persist_resume and run_name is not None:
            self._init_run_db(db_path, run_name, config_yaml, config_dict)

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

    # -- run_db journaling (run-dir mode) -----------------------------------

    def _init_run_db(self, db_path, run_name, config_yaml, config_dict) -> None:
        """Register the run, mirroring PRTCFRProductionTrainer._init_run_db so
        tiny-gate runs journal identically (CAMBIA_RUN_DB honored)."""
        try:
            from .. import run_db as _run_db
        except Exception:  # pragma: no cover - run_db optional
            return
        try:
            if db_path is None:
                db_path = os.environ.get("CAMBIA_RUN_DB") or os.path.join(
                    os.path.dirname(os.path.abspath(self.run_dir)), "cambia_runs.db"
                )
            self._db_conn = _run_db.get_db(db_path)
            algorithm = "prt-cfr"
            if config_dict:
                try:
                    algorithm = _run_db.infer_algorithm(config_dict)
                except Exception:
                    algorithm = "prt-cfr"
            self._db_run_id = _run_db.upsert_run(
                self._db_conn,
                name=run_name,
                algorithm=algorithm,
                config_yaml=config_yaml,
                config_dict=config_dict,
                status="running",
            )
            logger.info(
                "[prtcfr-tiny] run_db: registered '%s' (id=%s)",
                run_name,
                self._db_run_id,
            )
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr-tiny] run_db init failed (non-fatal): %s", e)
            self._db_conn = None
            self._db_run_id = None

    def _register_checkpoint_in_db(self, t: int, path: str) -> None:
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            ckpt_id = _run_db.register_checkpoint(self._db_conn, self._db_run_id, t, path)
            self._db_ckpt_ids[t] = ckpt_id
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr-tiny] register_checkpoint failed (non-fatal): %s", e)

    def _record_stability_best_in_db(self, t: int, metric: float) -> None:
        """Mirror a new stability-controller best into run_db (cambia-390).

        Called only when the controller's ``update`` reports ``is_best`` for
        iteration t, so the mode-aware (min/max) comparison already happened
        in the controller; this just persists its pointer -- exclusive
        checkpoints.is_best plus runs.best_metric_*. Runs every stability
        eval (not just at stop), so a resume/interrupt after any eval still
        leaves run_db reflecting the best known at that point."""
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            _run_db.set_best_metric(
                self._db_conn,
                self._db_run_id,
                self.stability_metric_name,
                metric,
                t,
            )
            ckpt_id = self._db_ckpt_ids.get(t)
            if ckpt_id is not None:
                _run_db.mark_best_checkpoint(self._db_conn, self._db_run_id, ckpt_id)
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr-tiny] stability best journal failed (non-fatal): %s", e)

    def _record_nashconv_in_db(self, t: int, value: float) -> None:
        """Journal one per-eval NashConv as a reconcile-visible eval_results row.

        Uses the ``nashconv`` baseline: the reconciler replays eval_results, and
        recompute_best_metric aggregates only MEAN_IMP_BASELINES, so a nashconv
        row is synced to the hub yet never pollutes mean_imp. The exploitability
        value rides in win_rate (the generic numeric slot); on the tiny tree it
        is well inside [0, 1]."""
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            _run_db.insert_eval_result(
                self._db_conn,
                self._db_run_id,
                None,
                {
                    "iteration": int(t),
                    "baseline": _run_db.STABILITY_NASHCONV_BASELINE,
                    "win_rate": float(value),
                },
            )
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr-tiny] nashconv journal failed (non-fatal): %s", e)

    def _update_db_status(self, status: str) -> None:
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            _run_db.update_run_status(self._db_conn, self._db_run_id, status)
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        """Close the run_db connection (idempotent)."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:  # pragma: no cover
                pass
            self._db_conn = None

    # -- full-state persistence (bit-exact resume / warm-start) -------------

    def resume_state_path(self) -> str:
        base = self.run_dir if self.run_dir is not None else self.snapshot_dir
        return os.path.join(base, "resume_state.json")

    def reservoir_path(self) -> str:
        base = self.run_dir if self.run_dir is not None else self.snapshot_dir
        return os.path.join(base, "reservoir.npz")

    def _save_reservoir(self) -> None:
        # ReservoirBuffer.save appends .npz; strip it so we do not get .npz.npz.
        path = self.reservoir_path()
        stem = path[:-4] if path.endswith(".npz") else path
        self.buffer.save(stem)

    def _save_resume_state(self, t: int) -> None:
        """Persist the iteration-``t`` resume point (written each iteration).

        Captured AFTER the snapshot/checkpoint + reservoir save AND iteration
        ``t``'s own stability check (cambia-341: ``train()`` runs the stability
        block before calling this), so the on-disk reservoir, RNG, and
        controller state are all as of the FULL end of iteration ``t`` --
        including any ``controller.update`` iteration ``t`` itself triggered.
        Saving before that update (the pre-cambia-341 order) meant a resume at
        an eval-due iteration silently dropped that iteration's own controller
        update: the next iteration's save would eventually pick it up in
        memory, but a resume or a run ending on an eval-due iteration lost it
        for good. Written via temp file + atomic rename so an interruption
        mid-write never leaves a torn resume_state.json. ``total_iterations``
        records the horizon the global LR schedule spanned; a continuation
        that keeps it (and the schedule) identical is bit-exact, one that
        changes it (the ruled 530->1000 flat-lr extension) is an intentional
        schedule change, not bit-exact."""
        state = {
            "schema": TINY_RESUME_SCHEMA_VERSION,
            "iteration": int(t),
            "total_iterations": int(self.iterations),
            "snapshots": [int(i) for i in self._written_iters],
            "controller": (
                _controller_to_dict(self.controller)
                if self.controller is not None
                else None
            ),
        }
        state.update(_global_rng_save())
        state.update(_accel_rng_save(self.device))
        path = self.resume_state_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)

    def _restore_net_from_checkpoint(self, ckpt_path: str) -> int:
        """Load net weights from a rolling checkpoint or per-iteration snapshot.

        Both formats carry ``encoder_state_dict``/``head_state_dict``/
        ``iteration``; the rolling checkpoint adds ``config``. Returns the stored
        iteration."""
        if self.net is None:
            self.net = self._net_factory()
        # weights_only=True: the pinned payload is plain tensors/ints/dict; a
        # poisoned pickle rsync-written under runs/ cannot execute on load
        # (cambia-552). Compatible with existing checkpoints (prtcfr_mixture
        # already loads this shape under weights_only=True).
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        _assert_resume_tokenizer_compatible(payload)
        self.net.load_encoder_head(
            payload["encoder_state_dict"], payload["head_state_dict"]
        )
        return int(payload.get("iteration", 0))

    def _load_full_state(
        self, resume_state_file: str, checkpoint_file: str, reservoir_file: Optional[str]
    ) -> int:
        """Bit-exact restore: net + reservoir + RNG + controller + counter.

        Returns the last completed iteration; the loop resumes at it+1. RNG is
        restored LAST so nothing the load touches leaks into the resumed stream."""
        with open(resume_state_file, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        schema = int(state.get("schema", 0))
        if schema != TINY_RESUME_SCHEMA_VERSION:
            raise PRTCFRResumeError(
                f"cannot resume: resume_state.json schema {schema} != expected "
                f"{TINY_RESUME_SCHEMA_VERSION}"
            )
        ckpt_iter = self._restore_net_from_checkpoint(checkpoint_file)
        rs_iter = int(state["iteration"])
        if ckpt_iter != rs_iter:
            raise PRTCFRResumeError(
                f"cannot resume: rolling checkpoint iteration {ckpt_iter} != "
                f"resume_state iteration {rs_iter} (interrupted between the "
                f"checkpoint save and the resume_state commit; discard the "
                f"partial iteration before resuming)"
            )
        if reservoir_file and os.path.exists(reservoir_file):
            self.buffer.load(reservoir_file)
        if self.controller is not None and state.get("controller") is not None:
            self.controller = _controller_from_dict(state["controller"])
        self._written_iters = [int(i) for i in state.get("snapshots", [])]
        _global_rng_restore(state)
        logger.info(
            "[prtcfr-tiny] full-state restore from iter=%d: buffer=%d snapshots=%s",
            rs_iter,
            len(self.buffer),
            self._written_iters,
        )
        return rs_iter

    def _full_state_dir_hint(self, pt_path: str) -> Optional[str]:
        """If the bare .pt's own directory, or its parent run dir, already
        carries a full resume_state.json + reservoir.npz pair, return that
        directory so the net-only guard error can point straight at it."""
        own_dir = os.path.dirname(os.path.abspath(pt_path))
        for d in (own_dir, os.path.dirname(own_dir)):
            rs = os.path.join(d, "resume_state.json")
            res = os.path.join(d, "reservoir.npz")
            if os.path.exists(rs) and os.path.exists(res):
                return d
        return None

    def _guard_net_only_warm_start(self, pt_path: str) -> None:
        """Raise unless warm_start_net_only_ok is set (cambia-374).

        A bare-.pt warm start carries net weights + iteration only, so the
        reservoir restarts EMPTY and the trainer fine-tunes the imported net on
        regret targets from a tiny immature buffer -- that is a fresh run with
        a net prior, never a state-faithful continuation (measured: ~0.2
        NashConv-quality snapshots that go on to dominate the linear SD-CFR
        mixture). Full-mode warm start (a run dir / resume_state.json) already
        restores net + reservoir and is the correct continuation path."""
        if self.warm_start_net_only_ok:
            return
        msg = (
            f"warm_start_path {pt_path!r} resolves to a bare checkpoint "
            f"(net-only warm start, cambia-374): the reservoir restarts EMPTY "
            f"and the trainer fine-tunes on regret targets from a tiny immature "
            f"buffer, producing a fresh run with a net prior, not a "
            f"state-faithful continuation. Point warm_start_path at the source "
            f"RUN DIR (or its resume_state.json) for a state-faithful "
            f"continuation that restores net + reservoir, or set "
            f"prt_cfr.warm_start_net_only_ok: true to proceed deliberately."
        )
        hint = self._full_state_dir_hint(pt_path)
        if hint is not None:
            msg += f" Full state exists at {hint!r}; point warm_start_path there."
        raise PRTCFRResumeError(msg)

    def _resolve_warm_start(self, path: str):
        """Classify a warm_start_path into (mode, files).

        mode "full" -> (resume_state.json, rolling checkpoint, reservoir.npz) for
        a bit-exact continuation; mode "net" -> a bare .pt for net + iteration
        only (the legacy iter-530 snapshot case: it carries encoder/head/iteration
        and NOTHING else, so the reservoir starts empty and RNG seeds from the
        ambient stream -- NOT bit-exact). mode "net" raises unless
        config.warm_start_net_only_ok is set (cambia-374): unguarded net-only
        warm start silently produces a fresh-run-with-net-prior lineage rather
        than the continuation callers ask for."""
        if os.path.isdir(path):
            rs = os.path.join(path, "resume_state.json")
            ckpt = os.path.join(path, "snapshots", "prtcfr_checkpoint.pt")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(path, "prtcfr_checkpoint.pt")
            if os.path.exists(rs) and os.path.exists(ckpt):
                res = os.path.join(path, "reservoir.npz")
                return "full", (rs, ckpt, res if os.path.exists(res) else None)
            if os.path.exists(ckpt):
                self._guard_net_only_warm_start(ckpt)
                return "net", (ckpt,)
            raise PRTCFRResumeError(
                f"warm_start_path dir {path!r} has no resume_state.json+checkpoint "
                f"and no rolling checkpoint"
            )
        if os.path.basename(path) == "resume_state.json":
            base = os.path.dirname(path)
            ckpt = os.path.join(base, "snapshots", "prtcfr_checkpoint.pt")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(base, "prtcfr_checkpoint.pt")
            res = os.path.join(base, "reservoir.npz")
            return "full", (path, ckpt, res if os.path.exists(res) else None)
        # A bare .pt (per-iteration snapshot or rolling checkpoint): net + iter.
        self._guard_net_only_warm_start(path)
        return "net", (path,)

    def _import_prior_snapshots(self, src_dir: str, upto_t: int) -> List[int]:
        """Copy prtcfr_snapshot_iter_{i}.pt (i <= upto_t) from a source snapshot
        dir into this run's snapshot dir when they are not already there, so the
        continuation's SD-CFR average spans [1 .. N], not just the new iters.

        Registers every present iter in run_db via the same helper run_iteration
        uses for natively-written checkpoints (cambia-389): otherwise an imported
        snapshot is ledger-listed (resume_state.json's snapshots list) but has no
        checkpoints row, so the harness pull include-set -- derived from run_db
        checkpoint rows -- silently skips it.

        Returns the sorted iters now present (source + already-local)."""
        present = set()
        pat = re.compile(r"prtcfr_snapshot_iter_(\d+)\.pt$")
        if os.path.abspath(src_dir) != os.path.abspath(self.snapshot_dir):
            for fp in glob.glob(os.path.join(src_dir, "prtcfr_snapshot_iter_*.pt")):
                m = pat.search(os.path.basename(fp))
                if not m:
                    continue
                it = int(m.group(1))
                if it > upto_t:
                    continue
                dst = self.snapshot_path(it)
                if not os.path.exists(dst):
                    shutil.copy2(fp, dst)
                present.add(it)
        for fp in glob.glob(os.path.join(self.snapshot_dir, "prtcfr_snapshot_iter_*.pt")):
            m = pat.search(os.path.basename(fp))
            if m and int(m.group(1)) <= upto_t:
                present.add(int(m.group(1)))
        for it in present:
            self._register_checkpoint_in_db(it, self.snapshot_path(it))
        return sorted(present)

    def _warm_start_from_path(self) -> int:
        """Seed a fresh run from warm_start_path. Returns the iteration to
        continue AFTER (loop starts at it+1).

        Semantics (cambia-374): mode "full" restores net + reservoir + RNG and
        is the only state-faithful continuation; verdict/ruled continuations
        must use it. Mode "net" restores net weights + iteration only and is a
        fresh run with a net prior, not a continuation -- _resolve_warm_start
        raises on it unless config.warm_start_net_only_ok is explicitly set."""
        mode, files = self._resolve_warm_start(self.warm_start_path)
        if mode == "full":
            rs, ckpt, res = files
            last_t = self._load_full_state(rs, ckpt, res)
            self._import_prior_snapshots(os.path.dirname(ckpt), last_t)
            logger.info("[prtcfr-tiny] warm-start (full) from %s at iter=%d", rs, last_t)
            return last_t
        (ckpt,) = files
        last_t = self._restore_net_from_checkpoint(ckpt)
        self._written_iters = self._import_prior_snapshots(os.path.dirname(ckpt), last_t)
        logger.warning(
            "[prtcfr-tiny] warm-start (net-only) from %s at iter=%d: net weights "
            "+ prior snapshots imported, but the reservoir starts EMPTY and RNG is "
            "the ambient stream -- NOT a bit-exact continuation (the source carries "
            "net weights + iteration only)",
            ckpt,
            last_t,
        )
        return last_t

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
        if self._persist_resume:
            self._register_checkpoint_in_db(t, snap)
            self._save_reservoir()
        return PRTCFRTrainState(
            iteration=t,
            samples_added=added,
            buffer_size=len(self.buffer),
            fit_loss=loss,
            snapshot_path=snap,
        )

    def _stability_check_due(self, t: int, n: int) -> bool:
        return t == 1 or t % self.stability_eval_every == 0 or t == n

    def train(
        self, iterations: Optional[int] = None, resume: bool = False
    ) -> List[PRTCFRTrainState]:
        """Run ``iterations`` (default config.iterations) PRT-CFR iterations.

        With ``stability_enabled`` and an ``eval_fn``, the trend metric is scored
        at the stability cadence, fed to the best-snapshot controller, and the
        deployable manifest is (re)written each check; training early-stops once
        the metric has risen past the tolerance band for ``patience`` checks. The
        deployable snapshot set pins to ``[1 .. best_iteration]`` so the served
        SD-CFR average excludes the diverged tail regardless.

        Continuation (run-dir mode only): ``resume=True`` reloads this run's own
        resume_state.json + rolling checkpoint + reservoir.npz and continues in
        place at ``t+1`` (harness SIGKILL/restart semantics). Otherwise, a
        ``warm_start_path`` on the config seeds a fresh run from ANOTHER run's
        saved state -- full (bit-exact) when it points at a resume_state.json/run
        dir, net + iteration only when it points at a bare snapshot .pt. In run-dir
        mode resume_state.json + reservoir.npz are written every iteration.
        """
        n = iterations if iterations is not None else self.iterations
        # The global LR schedule spans the run actually requested.
        self.iterations = n

        start_t = 1
        if resume:
            if not self._persist_resume:
                raise PRTCFRResumeError("resume requires run-dir mode")
            rs = self.resume_state_path()
            ckpt = self.checkpoint_path()
            if not os.path.exists(rs) or not os.path.exists(ckpt):
                raise PRTCFRResumeError(f"cannot resume: missing {rs} or {ckpt}")
            last_t = self._load_full_state(rs, ckpt, self.reservoir_path())
            start_t = last_t + 1
        elif self.warm_start_path:
            start_t = self._warm_start_from_path() + 1

        history: List[PRTCFRTrainState] = []
        try:
            for t in range(start_t, n + 1):
                st = self.run_iteration(t)
                history.append(st)
                self._written_iters.append(t)
                logger.info(
                    "[prtcfr] iter=%d samples+=%d buffer=%d fit_loss=%.5f snapshot=%s",
                    st.iteration,
                    st.samples_added,
                    st.buffer_size,
                    st.fit_loss,
                    os.path.basename(st.snapshot_path),
                )

                # Stability check BEFORE the resume_state commit (cambia-341):
                # see _save_resume_state's docstring. Any controller update for
                # iteration t must land in-memory before t's resume_state.json
                # is written, or a resume at t never sees t's own update.
                stop_early = False
                if (
                    self.stability_enabled
                    and self.controller is not None
                    and (self._stability_check_due(t, n))
                ):
                    if self.eval_fn is None:
                        # No trend metric available: keep the whole set deployable.
                        write_deployable_manifest(
                            self.snapshot_dir,
                            self.controller,
                            self._written_iters,
                            metric_name=self.stability_metric_name,
                            stopped_early=False,
                        )
                    else:
                        metric = float(self.eval_fn(self, t))
                        self._record_nashconv_in_db(t, metric)
                        decision = self.controller.update(t, metric)
                        if decision.is_best:
                            self._record_stability_best_in_db(t, metric)
                        logger.info(
                            "[prtcfr] stability iter=%d %s=%.5f best_iter=%d best=%.5f "
                            "worse_streak=%d stop=%s",
                            t,
                            self.stability_metric_name,
                            metric,
                            decision.best_iteration,
                            decision.best_metric,
                            decision.num_worse_since_best,
                            decision.should_stop,
                        )
                        write_deployable_manifest(
                            self.snapshot_dir,
                            self.controller,
                            self._written_iters,
                            metric_name=self.stability_metric_name,
                            stopped_early=decision.should_stop,
                        )
                        if decision.should_stop:
                            logger.info(
                                "[prtcfr] early-stop at iter=%d; deployable window "
                                "pinned to [1..%d] (%s=%.5f)",
                                t,
                                decision.best_iteration,
                                self.stability_metric_name,
                                decision.best_metric,
                            )
                            stop_early = True

                if self._persist_resume:
                    self._save_resume_state(t)
                if stop_early:
                    break
        except KeyboardInterrupt:
            self._update_db_status("interrupted")
            raise
        except Exception:
            self._update_db_status("failed")
            raise
        self._update_db_status("completed")
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
        n_nodes,
        aborted,
        prt_cfg.seq_cap,
    )

    trainer = PRTCFRTinyTrainer(root, prt_cfg, snapshot_dir)
    return trainer.train(iterations=iterations)


def build_tiny_nashconv_eval_fn(
    root,
    device: str = "cpu",
    seq_cap: int = SEQ_CAP,
    chunk_size: int = 2048,
) -> Callable[["PRTCFRTinyTrainer", int], float]:
    """Ground-truth NashConv eval_fn for the tiny gate's stability controller.

    Returns ``eval_fn(trainer, t) -> nashconv`` scoring the SD-CFR linear-weighted
    average of the snapshots on disk in ``trainer.snapshot_dir``. A single
    IncrementalPolicyAccumulator is reused across calls: each snapshot is folded
    in ONCE ever (linear over the horizon, not quadratic), matching the technique
    the S1W11 launcher prototyped. The metric is exploitability on the tiny
    perfect-recall tree, the X2 gate's arbiter."""
    from .prtcfr_eval import IncrementalPolicyAccumulator, _load_net, discover_snapshots
    from tools.tiny_solver import exploitability

    acc = IncrementalPolicyAccumulator(
        root, weighting="linear", seq_cap=seq_cap, chunk_size=chunk_size
    )
    seen: set = set()

    def eval_fn(trainer: "PRTCFRTinyTrainer", t: int) -> float:
        new = []
        for it, fp in discover_snapshots(trainer.snapshot_dir):
            if it > t or it in seen:
                continue
            seen.add(it)
            new.append((it, _load_net(fp, device=device)))
        acc.accumulate(sorted(new, key=lambda p: p[0]))
        nashconv, _components = exploitability(root, acc.policy())
        return float(nashconv)

    return eval_fn


# ===========================================================================
# Production trainer (Phase 2 S1W5): full-game PRT-CFR
# ===========================================================================
#
# Ties every merged Phase-2 component into one iteration loop:
#   generate K single-trajectory ESCHER traversals (prtcfr_worker,
#   traverser alternating, one game per traversal through an injectable
#   driver_factory -- the Go substrate by default)
#     -> append traverser regret samples to per-player DiskReservoirs (p2 sec
#        2.2: reservoir B_i per traverser)
#     -> refit ONE shared regret net on both reservoirs (production trainer
#        conventions, design-overview amended 2026-07-07: warm_start +
#        global_cosine + stability controller are the production defaults, set
#        in the run config; masked-SUM loss + normalized-linear weighting +
#        clip-last, reusing ``_fit_from_scratch``)
#     -> interleave the V_phi critic fit (outside the regret path) and log its
#        held-out MSE vs the constant-predictor baseline
#     -> snapshot ``prtcfr_snapshot_iter_{t}.pt`` + rolling
#        ``prtcfr_checkpoint.pt`` (Phase-1 dict format; rolling adds config)
#     -> BestSnapshotController manifest when an eval_fn is supplied.
#
# The sigma^t provider is stateless (one batched forward per query from the
# current net's regret-matched strategy). The incremental hidden-state carry
# (PRTCFRInferenceService) is an X3-ladder throughput swap owned elsewhere;
# this trainer never rewires the worker.


# A per-game driver builder: seed -> a fresh GameDriver. The default delegates
# to prtcfr_worker.new_production_driver (Go substrate); tests inject a
# python-backend or scripted factory.
DriverFactory = Callable[[int], GameDriver]


class NetProductionSigma:
    """Stateless sigma^t for the production worker: the current regret net's
    regret-matched strategy over the 146 global actions.

    Signature matches ``prtcfr_worker.ProductionPolicyFn`` -- ``(tokens, mask)
    -> (146,) probability vector`` -- so it composes at the same call sites as
    the fixed uniform b_i. One forward per query (no incremental carry): the
    batched-carry inference service (``PRTCFRInferenceService``) is the X3
    throughput swap owned elsewhere; at this stage a direct forward is the
    contract (design-overview / task S1W5). The net is put in ``eval()`` and
    never trained through here; a new iterate is a new provider instance,
    matching SD-CFR's per-iteration regret-net turnover.
    """

    def __init__(self, net: PRTCFRNet, seq_cap: int = PRODUCTION_SEQ_CAP):
        self.net = net.eval()
        self.seq_cap = int(seq_cap)

    def __call__(self, tokens: List[int], legal_mask: np.ndarray) -> np.ndarray:
        toks = list(tokens) if tokens else [PAD_ID]
        if len(toks) > self.seq_cap:
            # The driver already caps at seq_cap; this only guards a caller that
            # somehow hands a longer prefix (keep-most-recent, matching
            # pad_tokens / encode_observation_sequence overflow semantics).
            toks = toks[-self.seq_cap :]
        device = self.net.device
        t = torch.as_tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        mask_arr = np.asarray(legal_mask, dtype=bool)
        m = torch.as_tensor(mask_arr, device=device).unsqueeze(0)
        with torch.no_grad():
            strat = self.net.strategy_from_tokens(t, m)  # (1, 146)
        return strat[0].detach().to("cpu", dtype=torch.float64).numpy()


def _merge_columnar_batches(batches: List[ColumnarBatch]) -> Optional[ColumnarBatch]:
    """Concatenate ColumnarBatches from the per-player reservoirs into one
    batch for the shared-net fit.

    Each producer pads its features to the longest row in ITS batch, so the
    merge re-pads every sub-batch to the global max width with PAD_ID (0 --
    ``DiskReservoir``/``pad_tokens`` pad with 0, and net.encode recovers real
    lengths via the PAD_ID mask, so the extra zeros are inert).
    """
    batches = [b for b in batches if len(b) > 0]
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]
    max_w = max(int(b.features.shape[1]) for b in batches)
    feats = []
    for b in batches:
        f = b.features
        if f.shape[1] < max_w:
            padw = np.full((f.shape[0], max_w - f.shape[1]), PAD_ID, dtype=f.dtype)
            f = np.concatenate([f, padw], axis=1)
        feats.append(f)
    features = np.concatenate(feats, axis=0)
    targets = np.concatenate([b.targets for b in batches], axis=0)
    masks = (
        np.concatenate([b.masks for b in batches], axis=0)
        if all(b.masks is not None for b in batches)
        else None
    )
    iterations = np.concatenate([b.iterations for b in batches], axis=0)
    lengths = (
        np.concatenate([b.lengths for b in batches], axis=0)
        if all(b.lengths is not None for b in batches)
        else None
    )
    return ColumnarBatch(
        features=features,
        targets=targets,
        masks=masks,
        iterations=iterations,
        lengths=lengths,
    )


class _UnpaddingReservoir:
    """Impedance-match between the production worker (pads) and the DiskReservoir
    (ragged).

    ``PRTCFRProductionWorker.traverse`` stores each regret sample as
    ``pad_tokens(tokens_h, seq_cap)`` -- a FIXED-WIDTH seq_cap row, correct for
    the in-RAM ReservoirBuffer the tiny path uses. The DiskReservoir instead
    wants the RAGGED natural-length row (``add_batch`` contract: "NOT padded"):
    storing the padded 12288-wide row would consume seq_cap tokens per sample
    and blow the 20M-row reservoir to ~457GB, the exact case the ragged design
    exists to avoid (disk_reservoir.py docstring; p2 sec 6 targets ~12-25GB).

    This adapter strips the trailing PAD_ID before delegating to the disk
    reservoir. PAD_ID (0) is reserved as padding only -- real token bodies begin
    with BOS and never contain an interior 0 -- so the trailing run is pure
    right-padding and trimming it recovers the exact natural length. Everything
    else (len / sample_batch / save / load) delegates straight through.
    """

    def __init__(self, reservoir: Any):
        self._r = reservoir

    def add(self, sample: ReservoirSample) -> None:
        feats = np.asarray(sample.features)
        nz = np.nonzero(feats != PAD_ID)[0]
        natural = feats[: int(nz[-1]) + 1] if nz.size else feats[:0]
        self._r.add(
            ReservoirSample(
                features=natural,
                target=sample.target,
                action_mask=sample.action_mask,
                iteration=sample.iteration,
            )
        )

    def __len__(self) -> int:
        return len(self._r)

    def sample_batch(self, batch_size: int) -> ColumnarBatch:
        return self._r.sample_batch(batch_size)

    def save(self, *a, **k):
        return self._r.save(*a, **k)

    def load(self, *a, **k):
        return self._r.load(*a, **k)

    @property
    def raw(self) -> Any:
        return self._r


class _MultiReservoirSampler:
    """Presents the per-player DiskReservoirs as one ``sample_batch`` source so
    ``_fit_from_scratch`` fits the SINGLE shared regret net on both.

    Single shared net across seats is the default (p2 sec 2.3 fits R_theta on
    B_i; here unified) because PRT-CFR's token stream is SEAT-RELATIVE -- it
    encodes the acting seat's own observation-action history, not an absolute
    seat id -- so one net generalizes across both reservoirs without a per-seat
    head. Each fit step draws a batch split across reservoirs in proportion to
    their current sizes.
    """

    def __init__(self, reservoirs: List[Any], rng: Optional[np.random.Generator] = None):
        self._rs = list(reservoirs)
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return sum(len(r) for r in self._rs)

    def sample_batch(self, batch_size: int) -> ColumnarBatch:
        sizes = [len(r) for r in self._rs]
        total = sum(sizes)
        if total == 0:
            return ColumnarBatch(
                features=np.empty((0, 0), dtype=np.int16),
                targets=np.empty((0, NUM_ACTIONS), dtype=np.float32),
                masks=np.empty((0, NUM_ACTIONS), dtype=bool),
                iterations=np.empty(0, dtype=np.int64),
                lengths=np.empty(0, dtype=np.int64),
            )
        # Proportional allocation; remainder to the largest reservoir.
        alloc = [int(batch_size * s // total) for s in sizes]
        deficit = batch_size - sum(alloc)
        if deficit > 0:
            alloc[int(np.argmax(sizes))] += deficit
        sub = [r.sample_batch(a) for r, a in zip(self._rs, alloc) if a > 0]
        merged = _merge_columnar_batches(sub)
        if merged is None:
            # Every allocation landed on an empty reservoir; fall back to the
            # non-empty one at the full batch size.
            idx = int(np.argmax(sizes))
            return self._rs[idx].sample_batch(batch_size)
        return merged


# ---------------------------------------------------------------------------
# Resume-from-disk (S1T3): resume_state.json + reservoir/net/RNG reload.
# ---------------------------------------------------------------------------
#
# resume_state.json is written every iteration next to the run's rolling
# checkpoint + per-player reservoirs; --resume reloads all three so training
# continues at t+1 with the exact stream a non-interrupted run would produce:
#   - net: the rolling ``prtcfr_checkpoint.pt`` (the fitted iterate at t).
#   - reservoirs: each per-player DiskReservoir's on-disk bookkeeping (count,
#     seen_count, pool cursors, and its own sampling RNG) via ``.load()``.
#   - RNG: ``self._fit_rng`` (numpy Generator state) + the torch global RNG.
#     Per-iteration worker/game seeds are a deterministic function of ``t``
#     (prtcfr_worker seeds every traversal/rollout from ``self.seed + t*...``),
#     so they need no persistence -- resuming at the right ``t`` reproduces them.
#   - controller: the BestSnapshotController's public fields, so the deployable
#     window and early-stop streak survive the interruption.

RESUME_SCHEMA_VERSION = 1


class PRTCFRResumeError(RuntimeError):
    """Raised when ``train(resume=True)`` finds no resumable state on disk."""


def _controller_to_dict(controller: BestSnapshotController) -> dict:
    """Serialize a BestSnapshotController's public fields (JSON-safe).

    ``best_metric`` starts at +/-inf (mode-dependent) before the first check;
    inf has no JSON literal, so it is stored as ``None`` and reconstructed from
    ``mode`` on load.
    """
    bm = controller.best_metric
    return {
        "rel_tolerance": controller.rel_tolerance,
        "patience": controller.patience,
        "min_iters": controller.min_iters,
        "mode": controller.mode,
        "stop_mode": controller.stop_mode,
        "plateau_window_iters": controller.plateau_window_iters,
        "plateau_step_iters": controller.plateau_step_iters,
        "plateau_rel_improvement": controller.plateau_rel_improvement,
        "best_iteration": controller.best_iteration,
        "best_metric": bm if math.isfinite(bm) else None,
        "num_worse_since_best": controller.num_worse_since_best,
        "stopped": controller.stopped,
        "history": list(controller.history),
    }


def _controller_from_dict(d: dict) -> BestSnapshotController:
    """Rebuild a BestSnapshotController from ``_controller_to_dict`` output.

    ``stop_mode``/``plateau_*`` default to the class's divergence-mode
    defaults via ``.get`` so a resume_state.json written before cambia-341
    (no plateau fields) restores as divergence mode unchanged.
    """
    c = BestSnapshotController(
        rel_tolerance=float(d["rel_tolerance"]),
        patience=int(d["patience"]),
        min_iters=int(d["min_iters"]),
        mode=str(d["mode"]),
        stop_mode=str(d.get("stop_mode", "divergence")),
        plateau_window_iters=int(d.get("plateau_window_iters", 50)),
        plateau_step_iters=int(d.get("plateau_step_iters", 10)),
        plateau_rel_improvement=float(d.get("plateau_rel_improvement", 0.005)),
    )
    # __post_init__ set best_metric to the mode's inf sentinel; a stored None
    # means "no best yet" (keep that sentinel), else restore the saved value.
    bm = d.get("best_metric")
    if bm is not None:
        c.best_metric = float(bm)
    c.best_iteration = int(d.get("best_iteration", 0))
    c.num_worse_since_best = int(d.get("num_worse_since_best", 0))
    c.stopped = bool(d.get("stopped", False))
    c.history = list(d.get("history", []))
    return c


def _accel_rng_save(device: Any) -> Dict[str, List[str]]:
    """Capture the accelerator RNG state (cuda or xpu) for resume_state.json.

    Returns an empty dict for cpu devices or when the matching backend is
    unavailable, so a CPU-trained run's resume_state.json carries no
    accelerator RNG key at all -- the omission (not a null), matching the
    pre-xpu CUDA-only behavior this generalizes.
    """
    kind = str(device).split(":")[0]
    if kind == "cuda" and torch.cuda.is_available():
        return {
            "torch_cuda_rng": [
                s.numpy().tobytes().hex() for s in torch.cuda.get_rng_state_all()
            ]
        }
    if kind == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return {
            "torch_xpu_rng": [
                s.numpy().tobytes().hex() for s in torch.xpu.get_rng_state_all()
            ]
        }
    return {}


def _accel_rng_restore(state: dict) -> None:
    """Restore accelerator RNG state saved by ``_accel_rng_save``, if present.

    A resume_state.json written by a CPU run, or a pre-xpu-RNG version of this
    trainer, simply has neither key: restore is CPU-only in that case, which
    is the documented backward-compatible fallback, not an error.
    """
    cuda_rng = state.get("torch_cuda_rng")
    if cuda_rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                torch.from_numpy(np.frombuffer(bytes.fromhex(s), dtype=np.uint8).copy())
                for s in cuda_rng
            ]
        )
    xpu_rng = state.get("torch_xpu_rng")
    if xpu_rng and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_rng_state_all(
            [
                torch.from_numpy(np.frombuffer(bytes.fromhex(s), dtype=np.uint8).copy())
                for s in xpu_rng
            ]
        )


@dataclass
class PRTCFRProductionTrainState:
    """Per-iteration record (returned for logging/tests; one metrics.jsonl row)."""

    iteration: int
    samples_added: Dict[int, int]
    buffer_sizes: Dict[int, int]
    fit_loss: float
    peak_lr: float
    critic_held_out_mse: float
    critic_constant_baseline_mse: float
    critic_ratio: float
    gen_seconds: float
    fit_seconds: float
    snapshot_path: str
    # AC2 in-loop battery quantities (S2W1). Additive; existing consumers/tests
    # read the row by name, so these trail the pre-existing fields and default
    # to no-op values for any construction that omits them.
    #   t1_cambia_rate      : fraction of this iteration's K generation games
    #                         whose FIRST applied decision is a Cambia call.
    #   tier_a_lbr          : Tier-A LBR fast-lane exploitability of the SD-CFR
    #                         mixture over snapshots [1..t]; NaN off the battery
    #                         cadence (only scored when eval_fn runs).
    #   grad_norm_violations: fit steps whose pre-clip grad norm exceeded
    #                         grad_clip, summed over this iteration's fit.
    t1_cambia_rate: float = float("nan")
    tier_a_lbr: float = float("nan")
    grad_norm_violations: int = 0


# Global action index of ActionCallCambia in the 146-action space. Turn-1
# Cambia = this action being the first decision applied in a generation game.
_CALL_CAMBIA_INDEX = action_to_index(ActionCallCambia())


class _GenTurn1Observer:
    """Transparent GameDriver wrapper that taps the FIRST applied main-trajectory
    action of one generation game (turn-1 Cambia detection, S2W1).

    The production worker advances the real trajectory by calling ``apply`` on
    THIS top-level driver, while pricing rollouts and per-action children run on
    ``clone()``s. ``clone()`` therefore returns the INNER driver's clone
    (unwrapped) so only the single real trajectory is observed, never the
    rollout copies. Every other attribute (``tokens``/``current_player``/
    ``is_terminal``/``legal_actions``/``close``/``.engine``/``.game`` for the
    critic value-sink) delegates to the inner driver, so the wrap is behaviorally
    invisible to the sampler and the critic tap.
    """

    __slots__ = ("_inner", "_stats", "_recorded")

    def __init__(self, inner, stats: Dict[str, int]):
        self._inner = inner
        self._stats = stats
        self._recorded = False
        stats["games"] += 1

    def apply(self, action) -> bool:
        ok = self._inner.apply(action)
        if ok and not self._recorded:
            # First decision that actually advanced the game: the game's turn-1
            # move. Count it iff it is the Cambia call.
            self._recorded = True
            idx = (
                int(action)
                if isinstance(action, (int, np.integer))
                else action_to_index(action)
            )
            if idx == _CALL_CAMBIA_INDEX:
                self._stats["t1_cambia"] += 1
        return ok

    def clone(self):
        # Unwrapped: rollout / child-pricing clones must NOT be observed.
        return self._inner.clone()

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name):
        # Delegate everything not overridden above to the wrapped driver
        # (tokens, current_player, is_terminal, legal_actions, utility, and the
        # critic value-sink's .engine/.game/_get_all_cards_unsafe probes).
        return getattr(self._inner, name)


class PRTCFRProductionTrainer:
    """Full-game PRT-CFR production trainer (additive; the tiny trainer above is
    untouched).

    Parameters
    ----------
    config : PRTCFRConfig-like
        Read via getattr so a plain SimpleNamespace works in tests. Production
        values (warm_start, global_cosine, stability_enabled, K=8192,
        seq_cap=PRODUCTION_SEQ_CAP, reservoir_capacity=20M) come from the run
        config; the model defaults stay tiny-safe.
    run_dir : str
        The run directory. Snapshots -> ``<run_dir>/snapshots`` (unless
        config.snapshot_dir), reservoirs -> ``<run_dir>/reservoir`` (unless
        config.reservoir_dir), metrics -> ``<run_dir>/metrics.jsonl``.
    driver_factory : optional (seed) -> GameDriver
        Defaults to ``new_production_driver`` on the configured backend (Go by
        default). Tests inject a python-backend or scripted factory. The trainer
        owns each per-game driver and ``close()``s it after the traversal.
    net_factory : optional () -> PRTCFRNet
        Defaults to ``build_prtcfr_net(config)``.
    eval_fn : optional (trainer, iteration) -> float
        Trend metric for the BestSnapshotController (e.g. exploitability). None
        (the S1W5 default) means no early-stop scoring; the deployable manifest,
        when stability is enabled, keeps the whole set deployable.
    db_path : optional str
        run_db location; defaults to ``<run_dir>/../cambia_runs.db``.
    """

    def __init__(
        self,
        config,
        run_dir: str,
        driver_factory: Optional[DriverFactory] = None,
        net_factory: Optional[Callable[[], PRTCFRNet]] = None,
        eval_fn: Optional[Callable[["PRTCFRProductionTrainer", int], float]] = None,
        db_path: Optional[str] = None,
        run_name: Optional[str] = None,
        config_yaml: Optional[str] = None,
        config_dict: Optional[dict] = None,
    ):
        self.config = config
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.seq_cap = int(getattr(config, "seq_cap", PRODUCTION_SEQ_CAP))
        self.m_rollouts = int(getattr(config, "m_rollouts", 4))
        self.k_games = int(getattr(config, "k_games_per_iter", 8192))
        self.iterations = int(getattr(config, "iterations", 300))
        self.lr = float(getattr(config, "lr", 1e-3))
        self.lr_min = float(getattr(config, "lr_min", 0.0))
        self.lr_schedule = str(getattr(config, "lr_schedule", "global_cosine"))
        self.batch_size = int(getattr(config, "batch_size", 8192))
        self.train_steps = int(getattr(config, "train_steps", 3000))
        self.weight_decay = float(getattr(config, "weight_decay", 0.0))
        self.grad_clip = float(getattr(config, "grad_clip", 1.0))
        self.warm_start = bool(getattr(config, "warm_start", True))
        self.reanchor_every = int(getattr(config, "reanchor_every", 0))
        self.num_players = int(getattr(config, "num_players", 2))
        self.max_trajectory_steps = int(getattr(config, "max_trajectory_steps", 4000))
        self.backend = str(getattr(config, "backend", "go"))
        self.seed = int(getattr(config, "seed", 0))
        # "cpu" is the safe universal fallback for a config object that lacks
        # a device attribute entirely (duck-typed test configs); the real
        # production path always sets config.device via _resolve_device
        # before constructing this trainer, so this default is not hit there.
        self.device = getattr(config, "device", "cpu")
        self.reservoir_capacity = int(getattr(config, "reservoir_capacity", 20_000_000))
        # Batched incremental generation (S1W15, the X3 gen remedy).
        self.gen_batched = bool(getattr(config, "gen_batched", True))
        self.gen_chunk_games = int(getattr(config, "gen_chunk_games", 64))
        self.infer_dtype = str(getattr(config, "infer_dtype", "bf16"))

        # Stability controller (production default ON; config-gated).
        self.stability_enabled = bool(getattr(config, "stability_enabled", True))
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
        self.stability_stop_mode = str(
            getattr(config, "stability_stop_mode", "divergence")
        )
        self.stability_plateau_window_iters = int(
            getattr(config, "stability_plateau_window_iters", 50)
        )
        self.stability_plateau_step_iters = int(
            getattr(config, "stability_plateau_step_iters", 10)
        )
        self.stability_plateau_rel_improvement = float(
            getattr(config, "stability_plateau_rel_improvement", 0.005)
        )
        self.eval_fn = eval_fn
        self.controller: Optional[BestSnapshotController] = None
        if self.stability_enabled:
            self.controller = BestSnapshotController(
                rel_tolerance=self.stability_rel_tolerance,
                patience=self.stability_patience,
                min_iters=self.stability_min_iters,
                mode=self.stability_metric_mode,
                stop_mode=self.stability_stop_mode,
                plateau_window_iters=self.stability_plateau_window_iters,
                plateau_step_iters=self.stability_plateau_step_iters,
                plateau_rel_improvement=self.stability_plateau_rel_improvement,
            )

        # Directory layout.
        self.snapshot_dir = str(
            getattr(config, "snapshot_dir", None) or os.path.join(run_dir, "snapshots")
        )
        self.reservoir_root = str(
            getattr(config, "reservoir_dir", None) or os.path.join(run_dir, "reservoir")
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.reservoir_root, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")

        self._driver_factory: DriverFactory = driver_factory or (
            lambda seed: new_production_driver(
                seed, num_players=self.num_players, backend=self.backend
            )
        )
        self._net_factory = net_factory or (
            lambda: build_prtcfr_net(config, device=self.device)
        )

        # Per-player DiskReservoirs B_i (p2 sec 2.2). Ragged token storage,
        # disk-backed; the shared net fits on both via _MultiReservoirSampler.
        from ..disk_reservoir import DiskReservoir

        self.reservoirs: Dict[int, Any] = {}
        for p in range(self.num_players):
            disk = DiskReservoir(
                path=os.path.join(self.reservoir_root, f"player_{p}"),
                capacity=self.reservoir_capacity,
                seq_cap=self.seq_cap,
                target_dim=NUM_ACTIONS,
                has_mask=True,
                seed=self.seed + 101 + p,
            )
            # Unpad the worker's fixed-width samples into ragged disk rows.
            self.reservoirs[p] = _UnpaddingReservoir(disk)
        self._fit_rng = np.random.default_rng(self.seed + 202)

        # V_phi critic (outside the regret path, S1W6). Optional.
        self.critic_enabled = bool(getattr(config, "critic_enabled", True))
        self._critic_net = None
        self._critic_trainer = None
        self._critic_reservoir = None
        self._critic_sink = None
        self.critic_steps = int(getattr(config, "critic_steps_per_iter", 500))
        self.critic_batch_size = int(getattr(config, "critic_batch_size", 512))
        if self.critic_enabled:
            self._init_critic()

        self.net: Optional[PRTCFRNet] = None
        self._history: List[PRTCFRProductionTrainState] = []
        self._written_iters: List[int] = []

        # AC2 battery stash (S2W1). Written each iteration for the metrics row:
        # t1_cambia_rate by the generation tap; tier_a_lbr by the battery eval_fn
        # at the stability cadence (NaN off-cadence). Public attrs so the eval_fn
        # can stash onto the trainer per the pinned S2W1 interface.
        self.t1_cambia_rate: float = float("nan")
        self.tier_a_lbr: float = float("nan")
        self._gen_stats: Dict[str, int] = {"games": 0, "t1_cambia": 0}

        # run_db registration (optional, never fatal).
        self._db_conn = None
        self._db_run_id: Optional[int] = None
        # iteration -> checkpoints.id, populated by _register_checkpoint_in_db;
        # lets the stability-best mirror (cambia-390) flip is_best without a
        # re-query, since a checkpoint for iteration t is always registered
        # (run_iteration) before that iteration's stability check runs (train).
        self._db_ckpt_ids: Dict[int, int] = {}
        self._init_run_db(db_path, run_name, config_yaml, config_dict)

    # -- setup helpers ------------------------------------------------------

    def _init_critic(self) -> None:
        from .prtcfr_critic import (
            CriticReservoir,
            CriticReservoirSink,
            CriticTrainer,
            build_prtcfr_critic_net,
        )

        self._critic_net = build_prtcfr_critic_net(
            self.config, num_players=self.num_players, device=self.device
        )
        self._critic_trainer = CriticTrainer(
            self._critic_net, lr=float(getattr(self.config, "critic_lr", 1e-3))
        )
        self._critic_reservoir = CriticReservoir(
            capacity=int(getattr(self.config, "critic_capacity", 200_000)),
            held_out_fraction=float(
                getattr(self.config, "critic_held_out_fraction", 0.1)
            ),
            seq_cap=self.seq_cap,
            num_players=self.num_players,
            seed=self.seed + 303,
        )
        self._critic_sink = CriticReservoirSink(
            self._critic_reservoir, num_players=self.num_players, seq_cap=self.seq_cap
        )

    def _make_value_sink(self):
        """Wrap the critic sink so it works with BOTH driver backends.

        ``CriticReservoirSink`` -> ``omniscient_features_from_driver`` accepts a
        ``_get_all_cards_unsafe`` holder or a ``.game`` holder. The Go driver
        (S1W13) exposes its GoEngine as ``.engine`` (which HAS
        ``_get_all_cards_unsafe``); the Python stub exposes ``.game``. Passing
        ``getattr(driver, "engine", driver)`` hands the resolver an object it
        recognizes in either case, using only the single-source public function
        (no bespoke omniscient extraction here).
        """
        if not self.critic_enabled or self._critic_sink is None:
            return None
        sink = self._critic_sink

        def value_sink(tokens_h, driver, pooled_mean, iteration):
            source = getattr(driver, "engine", driver)
            sink(tokens_h, source, pooled_mean, iteration)

        return value_sink

    def _init_run_db(self, db_path, run_name, config_yaml, config_dict) -> None:
        try:
            from .. import run_db as _run_db
        except Exception:  # pragma: no cover - run_db optional
            return
        try:
            if db_path is None:
                # CAMBIA_RUN_DB (serving harness, design 4.2) redirects the
                # journal to the per-run-dir run_db.sqlite the pull loop
                # syncs; otherwise the local default is the runs-dir sibling.
                db_path = os.environ.get("CAMBIA_RUN_DB") or os.path.join(
                    os.path.dirname(os.path.abspath(self.run_dir)), "cambia_runs.db"
                )
            self._db_conn = _run_db.get_db(db_path)
            name = run_name or os.path.basename(os.path.normpath(self.run_dir))
            algorithm = "prt-cfr"
            if config_dict:
                try:
                    algorithm = _run_db.infer_algorithm(config_dict)
                except Exception:
                    algorithm = "prt-cfr"
            self._db_run_id = _run_db.upsert_run(
                self._db_conn,
                name=name,
                algorithm=algorithm,
                config_yaml=config_yaml,
                config_dict=config_dict,
                status="running",
            )
            logger.info("[prtcfr] run_db: registered '%s' (id=%s)", name, self._db_run_id)
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr] run_db init failed (non-fatal): %s", e)
            self._db_conn = None
            self._db_run_id = None

    # -- paths + persistence ------------------------------------------------

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
            "train_steps": self.train_steps,
            "reservoir_capacity": self.reservoir_capacity,
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
        """Write the rolling checkpoint via a temp file + atomic rename, mirroring
        ``_save_resume_state``'s pattern, so an interruption mid-write never
        leaves a torn ``prtcfr_checkpoint.pt`` (a torn checkpoint that happens to
        still deserialize with stale/partial tensor data would silently corrupt
        the next resume or eval)."""
        path = self.checkpoint_path()
        tmp = path + ".tmp"
        torch.save(
            {
                "encoder_state_dict": self.net.encoder_state_dict(),
                "head_state_dict": self.net.head_state_dict(),
                "config": self._config_dict(),
                "iteration": t,
            },
            tmp,
        )
        os.replace(tmp, path)

    def _register_checkpoint_in_db(self, t: int, path: str) -> None:
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            ckpt_id = _run_db.register_checkpoint(self._db_conn, self._db_run_id, t, path)
            self._db_ckpt_ids[t] = ckpt_id
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr] register_checkpoint failed (non-fatal): %s", e)

    def _record_stability_best_in_db(self, t: int, metric: float) -> None:
        """Mirror a new stability-controller best into run_db (cambia-390).

        Called only when the controller's ``update`` reports ``is_best`` for
        iteration t, so the mode-aware (min/max) comparison already happened
        in the controller; this just persists its pointer -- exclusive
        checkpoints.is_best plus runs.best_metric_*. Runs every stability
        eval (not just at stop), so a resume/interrupt after any eval still
        leaves run_db reflecting the best known at that point."""
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            _run_db.set_best_metric(
                self._db_conn,
                self._db_run_id,
                self.stability_metric_name,
                metric,
                t,
            )
            ckpt_id = self._db_ckpt_ids.get(t)
            if ckpt_id is not None:
                _run_db.mark_best_checkpoint(self._db_conn, self._db_run_id, ckpt_id)
        except Exception as e:  # pragma: no cover - never fatal
            logger.debug("[prtcfr] stability best journal failed (non-fatal): %s", e)

    def _update_db_status(self, status: str) -> None:
        if self._db_conn is None or self._db_run_id is None:
            return
        try:
            from .. import run_db as _run_db

            _run_db.update_run_status(self._db_conn, self._db_run_id, status)
        except Exception:  # pragma: no cover
            pass

    def save_reservoirs(self) -> None:
        for r in self.reservoirs.values():
            try:
                r.save()
            except Exception as e:  # pragma: no cover
                logger.warning("[prtcfr] reservoir save failed (non-fatal): %s", e)

    # -- resume-from-disk ---------------------------------------------------

    def resume_state_path(self) -> str:
        return os.path.join(self.run_dir, "resume_state.json")

    def _save_resume_state(self, t: int) -> None:
        """Persist the iteration-``t`` resume point (written each iteration).

        Captured AFTER the iteration's snapshot/checkpoint + reservoir save AND
        iteration ``t``'s own stability check (cambia-341: ``train()`` runs the
        stability block before calling this), so the on-disk reservoir, RNG,
        and controller state are all as of the FULL end of iteration ``t`` --
        including any ``controller.update`` iteration ``t`` itself triggered.
        Saving before that update (the pre-cambia-341 order) meant a resume at
        an eval-due iteration silently dropped that iteration's own controller
        update: the next iteration's save would eventually pick it up in
        memory, but a resume or a run ending on an eval-due iteration lost it
        for good. Written via a temp file + atomic rename so an interruption
        mid-write never leaves a torn resume_state.json.

        ``torch_cuda_rng``/``torch_xpu_rng`` is populated only when this
        trainer's device matches that accelerator and it is available; it is
        omitted (not merely null) otherwise, so a CPU-trained run's
        resume_state.json carries no accelerator RNG key at all. Older
        resume_state.json files written before this field existed simply lack
        the key, which ``_load_resume_state`` treats as "nothing to restore"
        (CPU-only restore, backward compatible).
        """
        state = {
            "schema": RESUME_SCHEMA_VERSION,
            "iteration": int(t),
            "snapshots": [int(i) for i in self._written_iters],
            "numpy_rng": self._fit_rng.bit_generator.state,
            "torch_rng": torch.random.get_rng_state().numpy().tobytes().hex(),
            "controller": (
                _controller_to_dict(self.controller)
                if self.controller is not None
                else None
            ),
        }
        state.update(_accel_rng_save(self.device))
        path = self.resume_state_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)

    def _load_resume_state(self) -> int:
        """Restore net + reservoirs + RNG + controller from disk for ``--resume``.

        Returns the last completed iteration ``t`` (the loop resumes at ``t+1``).
        Raises :class:`PRTCFRResumeError` when the run has no resumable
        checkpoint, leaving the run directory otherwise untouched.
        """
        ckpt = self.checkpoint_path()
        rs_path = self.resume_state_path()
        if not os.path.isfile(ckpt):
            raise PRTCFRResumeError(f"cannot resume: no rolling checkpoint at {ckpt}")
        if not os.path.isfile(rs_path):
            raise PRTCFRResumeError(f"cannot resume: no resume_state.json at {rs_path}")
        with open(rs_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        schema = int(state.get("schema", 0))
        if schema != RESUME_SCHEMA_VERSION:
            raise PRTCFRResumeError(
                f"cannot resume: resume_state.json schema {schema} != expected "
                f"{RESUME_SCHEMA_VERSION}"
            )

        # 1. Rolling checkpoint net (the fitted iterate at t). Load onto CPU then
        # copy into the (possibly non-CPU) params -- load_state_dict handles the
        # cross-device copy, so this is device-agnostic.
        if self.net is None:
            self.net = self._net_factory()
        # weights_only=True: the rolling checkpoint is plain tensors + a
        # model_dump() config dict + ints; hardened against a poisoned pickle
        # under runs/ (cambia-552). Load-compatible with in-flight X2R runs
        # (prtcfr_mixture already reads this exact shape under weights_only=True).
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        # resume_state.json is the commit marker written LAST each iteration; the
        # rolling checkpoint + reservoirs are saved just before it (inside
        # run_iteration). An interrupt in that narrow gap leaves the checkpoint
        # one iteration ahead of resume_state, so replaying resume_state's
        # iteration would warm-start from the wrong net and double-count the
        # reservoir. Cross-check the two and refuse rather than corrupt.
        ckpt_iter = int(payload.get("iteration", -1))
        rs_iter = int(state["iteration"])
        if ckpt_iter != rs_iter:
            raise PRTCFRResumeError(
                f"cannot resume: rolling checkpoint iteration {ckpt_iter} != "
                f"resume_state iteration {rs_iter} (interrupted between the "
                f"checkpoint save and the resume_state commit; the partial "
                f"iteration must be discarded before resuming)"
            )
        _assert_resume_tokenizer_compatible(payload)
        self.net.load_encoder_head(
            payload["encoder_state_dict"], payload["head_state_dict"]
        )

        # 2. Per-player DiskReservoirs (through the _UnpaddingReservoir adapter):
        # restores count / seen_count / pool cursors / each reservoir's own RNG.
        for p in range(self.num_players):
            self.reservoirs[p].load()

        # 3. RNG: the fit-side numpy Generator + the torch global (CPU) RNG,
        # plus the accelerator RNG (cuda or xpu) when this resume_state.json
        # carries it. A resume_state.json written by a CPU run (or by a
        # pre-accelerator-RNG version of this trainer) simply has neither
        # "torch_cuda_rng" nor "torch_xpu_rng" key: restore is CPU-only in
        # that case, which is the documented backward-compatible fallback,
        # not an error.
        self._fit_rng.bit_generator.state = state["numpy_rng"]
        torch_bytes = np.frombuffer(
            bytes.fromhex(state["torch_rng"]), dtype=np.uint8
        ).copy()
        torch.random.set_rng_state(torch.from_numpy(torch_bytes))
        _accel_rng_restore(state)

        # 4. Controller + the written-snapshot list (drives the deployable
        # manifest). ``self._history`` stays empty: train() returns only the
        # resumed portion, so the caller sees the loop start at t+1.
        if self.controller is not None and state.get("controller") is not None:
            self.controller = _controller_from_dict(state["controller"])
        self._written_iters = [int(i) for i in state.get("snapshots", [])]

        last_t = int(state["iteration"])
        logger.info(
            "[prtcfr-prod] resumed from iter=%d: reservoirs=%s snapshots=%s",
            last_t,
            {p: len(self.reservoirs[p]) for p in range(self.num_players)},
            self._written_iters,
        )
        return last_t

    # -- the iteration ------------------------------------------------------

    def _infer_torch_dtype(self):
        import torch

        return (
            torch.float32 if self.infer_dtype in ("fp32", "float32") else torch.bfloat16
        )

    def _generate_sequential(self, t: int, added: Dict[int, int], value_sink) -> None:
        """Original per-decision full-prefix generation (NetProductionSigma):
        one un-batched, non-carried GRU forward per decision and per rollout
        step. Kept as a fallback and the equivalence-gate reference; the
        production default is the batched path below (config.gen_batched)."""
        sigma = NetProductionSigma(self.net, seq_cap=self.seq_cap)
        worker = PRTCFRProductionWorker(
            sigma,
            m_rollouts=self.m_rollouts,
            seq_cap=self.seq_cap,
            seed=self.seed + t * 7_000_003,
            max_trajectory_steps=self.max_trajectory_steps,
            value_sink=value_sink,
        )
        for k in range(self.k_games):
            traverser = (t + k) % self.num_players
            game_seed = self.seed + t * 1_000_003 + k
            driver = _GenTurn1Observer(self._driver_factory(game_seed), self._gen_stats)
            try:
                n = worker.traverse(driver, traverser, t, self.reservoirs[traverser])
                added[traverser] += n
            finally:
                driver.close()

    def _generate_batched(self, t: int, added: Dict[int, int], value_sink) -> None:
        """Batched incremental generation (S1W15, the X3 gen remedy): every
        live game and its m CRN rollouts that reach a decision on a scheduler
        tick share ONE carried-hidden GRU forward via the
        PRTCFRInferenceService. Games run in chunks of ``gen_chunk_games`` to
        bound peak simultaneously-live drivers under the Go handle pool; one
        service (a frozen bf16/fp32 snapshot of sigma^t) + manager is reused
        across chunks (each game's streams are dropped when its chunk finishes,
        so no hidden state leaks).
        """
        from .prtcfr_infer import PRTCFRInferenceService

        service = PRTCFRInferenceService(
            self.net, device=str(self.device), dtype=self._infer_torch_dtype()
        )
        mgr = IncrementalSigmaManager(service, num_players=self.num_players)
        worker = PRTCFRBatchedProductionWorker(
            m_rollouts=self.m_rollouts,
            seq_cap=self.seq_cap,
            max_trajectory_steps=self.max_trajectory_steps,
            value_sink=value_sink,
        )
        chunk = max(1, self.gen_chunk_games)
        for start in range(0, self.k_games, chunk):
            end = min(start + chunk, self.k_games)
            specs: List[Dict[str, Any]] = []
            for k in range(start, end):
                traverser = (t + k) % self.num_players
                game_seed = self.seed + t * 1_000_003 + k
                rng_seed = self.seed + t * 7_000_003 + k * 2_000_029
                driver = _GenTurn1Observer(
                    self._driver_factory(game_seed), self._gen_stats
                )
                specs.append(
                    {
                        "seed": rng_seed,
                        "driver": driver,
                        "traverser": traverser,
                        "iteration": t,
                        "buf": self.reservoirs[traverser],
                    }
                )
            try:
                chunk_added = worker.generate(specs, mgr)
                for gi, spec in enumerate(specs):
                    added[spec["traverser"]] += chunk_added[gi]
            finally:
                for spec in specs:
                    spec["driver"].close()

    def run_iteration(self, t: int) -> PRTCFRProductionTrainState:
        """One production PRT-CFR iteration.

        sigma^t is the regret-matched strategy of the net trained at t-1 (a
        fresh near-uniform net at t=1). K single-trajectory ESCHER traversals
        (traverser alternating) append regret samples to the per-player
        reservoirs; the shared net is then refit on both. A snapshot + rolling
        checkpoint are written and the critic is fit (outside the regret path).
        """
        if self.net is None:
            self.net = self._net_factory()
        value_sink = self._make_value_sink()

        added = {p: 0 for p in range(self.num_players)}
        # Reset the turn-1 Cambia tap for this iteration's own K games; the
        # generation drivers are wrapped in _GenTurn1Observer, which fills it.
        self._gen_stats = {"games": 0, "t1_cambia": 0}
        gen_start = time.perf_counter()
        if self.gen_batched:
            self._generate_batched(t, added, value_sink)
        else:
            self._generate_sequential(t, added, value_sink)
        gen_seconds = time.perf_counter() - gen_start
        games = self._gen_stats["games"]
        self.t1_cambia_rate = (
            self._gen_stats["t1_cambia"] / games if games > 0 else float("nan")
        )

        # Refit the SHARED net on both reservoirs. From-scratch re-init when
        # warm_start is off or a periodic re-anchor fires; else fine-tune the
        # previous iterate. peak LR follows the global schedule.
        reanchor = self.reanchor_every > 0 and t % self.reanchor_every == 0
        if (not self.warm_start) or reanchor:
            self.net = self._net_factory()
        peak_lr = _peak_lr_for_iter(
            self.lr, self.lr_min, t, self.iterations, self.lr_schedule
        )
        sampler = _MultiReservoirSampler(
            [self.reservoirs[p] for p in range(self.num_players)], rng=self._fit_rng
        )
        fit_start = time.perf_counter()
        grad_viol = [0]
        loss = _fit_from_scratch(
            self.net,
            sampler,
            lr=peak_lr,
            batch_size=self.batch_size,
            num_steps=self.train_steps,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            lr_min=self.lr_min,
            violation_box=grad_viol,
        )
        fit_seconds = time.perf_counter() - fit_start

        # Critic fit (outside the regret path); held-out MSE vs constant baseline.
        c_mse = float("nan")
        c_base = float("nan")
        c_ratio = float("nan")
        if self.critic_enabled and self._critic_trainer is not None:
            cm = self._critic_trainer.fit(
                self._critic_reservoir,
                steps=self.critic_steps,
                batch_size=self.critic_batch_size,
            )
            c_mse = cm.held_out_mse
            c_base = cm.constant_baseline_mse
            c_ratio = cm.ratio

        snap = self._save_snapshot(t)
        self._save_checkpoint(t)
        self._register_checkpoint_in_db(t, snap)
        self.save_reservoirs()

        return PRTCFRProductionTrainState(
            iteration=t,
            samples_added=dict(added),
            buffer_sizes={p: len(self.reservoirs[p]) for p in range(self.num_players)},
            fit_loss=loss,
            peak_lr=peak_lr,
            critic_held_out_mse=c_mse,
            critic_constant_baseline_mse=c_base,
            critic_ratio=c_ratio,
            gen_seconds=gen_seconds,
            fit_seconds=fit_seconds,
            snapshot_path=snap,
            t1_cambia_rate=self.t1_cambia_rate,
            grad_norm_violations=grad_viol[0],
            # tier_a_lbr is scored by the battery eval_fn at the stability
            # cadence (train() sets st.tier_a_lbr from its return); NaN here.
            tier_a_lbr=float("nan"),
        )

    def _write_metrics_row(self, st: PRTCFRProductionTrainState) -> None:
        row = {
            "iteration": st.iteration,
            "samples_added": {str(p): int(n) for p, n in st.samples_added.items()},
            "buffer_sizes": {str(p): int(n) for p, n in st.buffer_sizes.items()},
            "fit_loss": _json_num(st.fit_loss),
            "peak_lr": _json_num(st.peak_lr),
            "critic_held_out_mse": _json_num(st.critic_held_out_mse),
            "critic_constant_baseline_mse": _json_num(st.critic_constant_baseline_mse),
            "critic_ratio": _json_num(st.critic_ratio),
            "gen_seconds": _json_num(st.gen_seconds),
            "fit_seconds": _json_num(st.fit_seconds),
            # AC2 in-loop battery fields (S2W1). tier_a_lbr is null off the
            # battery cadence; t1_cambia_rate/grad_norm_violations are per-iter.
            "t1_cambia_rate": _json_num(st.t1_cambia_rate),
            "tier_a_lbr": _json_num(st.tier_a_lbr),
            "grad_norm_violations": int(st.grad_norm_violations),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _stability_check_due(self, t: int, n: int) -> bool:
        return t == 1 or t % self.stability_eval_every == 0 or t == n

    def train(
        self, iterations: Optional[int] = None, resume: bool = False
    ) -> List[PRTCFRProductionTrainState]:
        """Run ``iterations`` (default config.iterations) production iterations.

        Each iteration appends a metrics.jsonl row and rewrites
        ``resume_state.json``. With stability enabled the trend metric is scored
        at the cadence (via ``eval_fn`` when supplied), the deployable manifest
        is (re)written, and training early-stops once the metric rises past
        tolerance for ``patience`` checks; the deployable window pins to
        ``[1 .. best_iteration]`` so the served SD-CFR average excludes any
        diverged tail. run_db status transitions to completed / interrupted /
        failed.

        With ``resume=True`` the trainer reloads the rolling checkpoint, the
        per-player reservoirs, the RNG state, and the controller from disk (see
        :meth:`_load_resume_state`) and continues at ``t+1``; the returned
        history covers only the resumed iterations, so the loop is observed to
        start at ``t+1``. A resume with no checkpoint raises
        :class:`PRTCFRResumeError`. The run_db row is reused by name (the
        by-name upsert in ``__init__`` never creates a duplicate on resume).
        """
        n = iterations if iterations is not None else self.iterations
        self.iterations = n  # global LR schedule spans the run actually requested
        start_t = 1
        if resume:
            last_t = self._load_resume_state()
            start_t = last_t + 1
            if start_t > n:
                logger.info(
                    "[prtcfr-prod] resume: last iter=%d already at/past n=%d; "
                    "nothing to run",
                    last_t,
                    n,
                )
        try:
            for t in range(start_t, n + 1):
                st = self.run_iteration(t)
                self._history.append(st)
                self._written_iters.append(t)
                logger.info(
                    "[prtcfr-prod] iter=%d added=%s buffers=%s fit_loss=%.5f "
                    "peak_lr=%.2e critic_mse=%.4f/%.4f gen=%.1fs fit=%.1fs",
                    st.iteration,
                    st.samples_added,
                    st.buffer_sizes,
                    st.fit_loss,
                    st.peak_lr,
                    st.critic_held_out_mse,
                    st.critic_constant_baseline_mse,
                    st.gen_seconds,
                    st.fit_seconds,
                )

                # Stability check BEFORE the resume_state commit (cambia-341):
                # see _save_resume_state's docstring. Any controller update for
                # iteration t must land in-memory before t's resume_state.json
                # is written, or a resume at t never sees t's own update.
                stop_early = False
                if (
                    self.stability_enabled
                    and self.controller is not None
                    and (self._stability_check_due(t, n))
                ):
                    if self.eval_fn is None:
                        write_deployable_manifest(
                            self.snapshot_dir,
                            self.controller,
                            self._written_iters,
                            metric_name=self.stability_metric_name,
                            stopped_early=False,
                        )
                    else:
                        metric = float(self.eval_fn(self, t))
                        # The battery eval_fn returns the Tier-A LBR and stashes
                        # it on the trainer; record it in this iteration's row.
                        st.tier_a_lbr = metric
                        decision = self.controller.update(t, metric)
                        if decision.is_best:
                            self._record_stability_best_in_db(t, metric)
                        logger.info(
                            "[prtcfr-prod] stability iter=%d %s=%.5f best_iter=%d "
                            "best=%.5f worse_streak=%d stop=%s",
                            t,
                            self.stability_metric_name,
                            metric,
                            decision.best_iteration,
                            decision.best_metric,
                            decision.num_worse_since_best,
                            decision.should_stop,
                        )
                        write_deployable_manifest(
                            self.snapshot_dir,
                            self.controller,
                            self._written_iters,
                            metric_name=self.stability_metric_name,
                            stopped_early=decision.should_stop,
                        )
                        if decision.should_stop:
                            logger.info(
                                "[prtcfr-prod] early-stop at iter=%d; deployable "
                                "window pinned to [1..%d]",
                                t,
                                decision.best_iteration,
                            )
                            stop_early = True

                # Metrics row is written AFTER the stability block so the same
                # row carries this iteration's tier_a_lbr (scored by the battery
                # eval_fn just above); it commits alongside resume_state, whose
                # cambia-341 ordering the row now shares.
                self._write_metrics_row(st)
                self._save_resume_state(t)
                if stop_early:
                    break
        except KeyboardInterrupt:
            # Do NOT save_reservoirs() here. The last COMPLETED iteration's
            # reservoir state was already flushed to disk inside that
            # iteration's own run_iteration() call, before resume_state.json
            # committed it. An interrupt mid-iteration-t (e.g. during
            # generation, after samples were already added to the in-memory
            # reservoir but before this iteration's own checkpoint/reservoir
            # save) must leave the on-disk reservoir at its last-committed
            # (t-1) state: flushing here would persist iteration t's partial
            # in-memory mutations while checkpoint + resume_state.json stay at
            # t-1, and a later --resume would replay iteration t from scratch
            # on top of that partial state, double-counting its samples.
            logger.warning("[prtcfr-prod] interrupted; last snapshot is on disk")
            self._update_db_status("interrupted")
            raise
        except Exception:
            # Same rule as the KeyboardInterrupt handler above: a partial
            # iteration's in-memory reservoir mutations must not be flushed
            # past the last-committed (t-1) on-disk state, or --resume
            # double-counts iteration t's samples.
            self._update_db_status("failed")
            raise
        self.save_reservoirs()
        self._update_db_status("completed")
        return self._history

    def close(self) -> None:
        """Close the run_db connection (idempotent).

        Deliberately does NOT flush reservoirs: close() runs in the CLI's
        finally block, so flushing here would persist a partial iteration's
        mutations after an interrupt or crash (the double-count path the
        abnormal-exit handlers in train() refuse). Reservoir state is owned
        by the per-iteration commit inside train().
        """
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:  # pragma: no cover
                pass
            self._db_conn = None


def _json_num(x: float):
    """JSON-safe number: NaN/inf -> None (JSON has no NaN literal)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf
