"""src/cfr/prtcfr_stability.py

Late-training stability for PRT-CFR.

Warm-start PRT-CFR reaches the X2 gate window then diverges: the fit loss and the
SD-CFR NashConv climb together once the accumulating reservoir grows past the
point where a fixed per-iteration fit budget re-converges the warm net after each
cosine-LR restart, and SD-CFR's linear (w_t=t) snapshot weighting then amplifies
the diverged tail into the served average. Two orthogonal, additive guards live
here; both are config-gated in the trainer and off by default so the Phase 1
gate reproduces byte-for-byte when disabled:

  1. BestSnapshotController -- tracks a scalar trend metric (exploitability where
     a scorer is available, else any cheap online proxy) across evaluation
     checkpoints, records the best iteration, and signals early-stop after
     ``patience`` consecutive checks worse than ``best * (1 + rel_tolerance)``.
     The DEPLOYABLE snapshot set is pinned to iters ``[1 .. best_iteration]`` so
     the served SD-CFR average never includes the diverged tail even if training
     ran past it.

  2. Deployable manifest (``prtcfr_deployable.json``) -- the on-disk record of
     the deployable window: ``{best_iteration, best_metric, deployable_iters,
     metric_name, stopped_early, mode, checks}``. The eval path reads it (via
     ``read_deployable_iters``) to restrict the SD-CFR average to the
     pre-divergence window.

Neither guard changes the training dynamics; they select which snapshots the
average serves. The optimizer-side fix for the blow-up itself (LR floor / no
per-iteration restart) is applied in the trainer's fit path, separately gated.

``BestSnapshotController.stop_mode`` selects the early-stop rule:

  - ``"divergence"`` (default) -- the ``patience``-consecutive-worse-than-
    tolerance rule described above. Byte-for-byte the original (only) rule;
    every existing caller that does not set ``stop_mode`` is unaffected.
  - ``"plateau"`` -- stops once the metric's relative improvement over the
    trailing ``plateau_window_iters`` iterations, normalized to a rate per
    ``plateau_step_iters``, drops below ``plateau_rel_improvement``. Intended
    for future gate runs that plateau without diverging (cambia-341): the
    divergence rule never fires on a trajectory that flattens instead of
    turning back up, so a run with no floor in its trajectory fit can run to
    the iteration budget with no early-stop signal. Config-gated and additive;
    it does not change ``"divergence"`` mode's behavior.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEPLOYABLE_MANIFEST = "prtcfr_deployable.json"


@dataclass
class ControllerDecision:
    """Result of feeding one (iteration, metric) check to the controller."""

    iteration: int
    metric: float
    is_best: bool
    best_iteration: int
    best_metric: float
    should_stop: bool
    num_worse_since_best: int


@dataclass
class BestSnapshotController:
    """Best-metric tracker with trend-based early stop for the deployable window.

    The metric is a scalar where lower is better by default (``mode="min"``, e.g.
    exploitability / NashConv / fit loss); ``mode="max"`` flips the comparisons
    for win-rate-style signals. ``update`` is called once per evaluation
    checkpoint with the trend metric at that iteration.

    Early-stop fires when the metric has been worse than ``best * (1 +
    rel_tolerance)`` (for ``mode="min"``; ``best * (1 - rel_tolerance)`` for
    ``mode="max"``) on ``patience`` consecutive checks AND at least ``min_iters``
    iterations have elapsed. ``rel_tolerance`` absorbs the eval-to-eval noise
    floor so a single noisy check does not trip the guard.

    The controller never mutates training; it only records ``best_iteration`` and
    reports ``should_stop``. The deployable window is ``[1 .. best_iteration]``.
    """

    rel_tolerance: float = 0.15
    patience: int = 3
    min_iters: int = 1
    mode: str = "min"

    # Early-stop rule selector. "divergence" (default) is the patience/tolerance
    # rule above; "plateau" is the trailing-window relative-improvement rule
    # (see module docstring). Additive: "divergence" behavior is unchanged.
    stop_mode: str = "divergence"
    plateau_window_iters: int = 50
    plateau_step_iters: int = 10
    plateau_rel_improvement: float = 0.005

    best_iteration: int = 0
    best_metric: float = math.inf
    num_worse_since_best: int = 0
    stopped: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.stop_mode not in ("divergence", "plateau"):
            raise ValueError(
                f"stop_mode must be 'divergence' or 'plateau', got {self.stop_mode!r}"
            )
        if self.mode == "max":
            self.best_metric = -math.inf

    def _is_better(self, metric: float, ref: float) -> bool:
        return metric < ref if self.mode == "min" else metric > ref

    def _worse_than_tolerance(self, metric: float) -> bool:
        if not math.isfinite(self.best_metric):
            return False
        if self.mode == "min":
            thresh = self.best_metric * (1.0 + self.rel_tolerance)
            # additive guard so a near-zero best does not make the band vanish
            thresh = max(thresh, self.best_metric + 1e-9)
            return metric > thresh
        thresh = self.best_metric * (1.0 - self.rel_tolerance)
        return metric < thresh

    def update(self, iteration: int, metric: float) -> ControllerDecision:
        """Feed one evaluation checkpoint; return the current decision."""
        metric = float(metric)
        self.history.append({"iteration": float(iteration), "metric": metric})

        is_best = self._is_better(metric, self.best_metric)
        if is_best:
            self.best_metric = metric
            self.best_iteration = iteration
            self.num_worse_since_best = 0
        elif self._worse_than_tolerance(metric):
            self.num_worse_since_best += 1
        else:
            # within the tolerance band of best: neither a new best nor a
            # divergence step. Reset the streak so only a SUSTAINED rise trips it.
            self.num_worse_since_best = 0

        if self.stop_mode == "plateau":
            trigger = self._plateau_triggered(iteration)
        else:
            trigger = self.num_worse_since_best >= self.patience
        should_stop = not self.stopped and iteration >= self.min_iters and trigger
        if should_stop:
            self.stopped = True

        return ControllerDecision(
            iteration=iteration,
            metric=metric,
            is_best=is_best,
            best_iteration=self.best_iteration,
            best_metric=self.best_metric,
            should_stop=should_stop,
            num_worse_since_best=self.num_worse_since_best,
        )

    def _plateau_triggered(self, iteration: int) -> bool:
        """True when the trailing-window relative improvement rate has
        dropped below ``plateau_rel_improvement``.

        Reference point: the latest recorded check at or before ``iteration -
        plateau_window_iters`` (the history is append-ordered by increasing
        iteration, so the last qualifying entry scanned is the closest one not
        exceeding the window). Returns False until such a reference exists --
        a run younger than the window never plateau-stops. The relative
        improvement over ``[ref_iteration, iteration]`` is normalized to a
        rate per ``plateau_step_iters`` (matching the eval cadence the window
        was sized against, e.g. 50 iters / 10-iter eval steps = 5 steps) so
        the threshold reads as "0.5% per 10-iteration step" regardless of how
        many checks actually landed inside the window.
        """
        threshold_iter = iteration - self.plateau_window_iters
        ref: Optional[Dict[str, float]] = None
        for entry in self.history:
            if entry["iteration"] <= threshold_iter:
                ref = entry
            else:
                break
        if ref is None:
            return False
        elapsed = iteration - ref["iteration"]
        if elapsed <= 0 or self.plateau_step_iters <= 0:
            return False
        ref_metric = ref["metric"]
        if abs(ref_metric) < 1e-12:
            return False
        if self.mode == "min":
            rel_improvement = (ref_metric - self.history[-1]["metric"]) / abs(ref_metric)
        else:
            rel_improvement = (self.history[-1]["metric"] - ref_metric) / abs(ref_metric)
        rate = rel_improvement / (elapsed / self.plateau_step_iters)
        return rate < self.plateau_rel_improvement

    def deployable_iters(self, all_iters: List[int]) -> List[int]:
        """Snapshots to serve: those at or before ``best_iteration``.

        ``all_iters`` is every snapshot iteration written so far. With no best
        recorded yet (no checks) the whole set is deployable.
        """
        if self.best_iteration <= 0:
            return sorted(all_iters)
        return sorted(i for i in all_iters if i <= self.best_iteration)


def write_deployable_manifest(
    snapshot_dir: str,
    controller: BestSnapshotController,
    all_iters: List[int],
    metric_name: str = "nashconv",
    stopped_early: bool = False,
) -> str:
    """Write ``prtcfr_deployable.json`` recording the deployable window."""
    deployable = controller.deployable_iters(all_iters)
    payload = {
        "best_iteration": controller.best_iteration,
        "best_metric": (
            controller.best_metric if math.isfinite(controller.best_metric) else None
        ),
        "metric_name": metric_name,
        "mode": controller.mode,
        "deployable_iters": deployable,
        "all_iters": sorted(all_iters),
        "stopped_early": bool(stopped_early),
        "stop_mode": controller.stop_mode,
        "rel_tolerance": controller.rel_tolerance,
        "patience": controller.patience,
        "checks": controller.history,
    }
    path = os.path.join(snapshot_dir, DEPLOYABLE_MANIFEST)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    return path


def read_deployable_manifest(snapshot_dir: str) -> Optional[dict]:
    """Load ``prtcfr_deployable.json`` if present, else None."""
    path = os.path.join(snapshot_dir, DEPLOYABLE_MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def read_deployable_iters(snapshot_dir: str) -> Optional[List[int]]:
    """The deployable snapshot iterations recorded in the manifest, or None.

    The eval path calls this to restrict the SD-CFR average to the pre-divergence
    window; a ``None`` return means no manifest (serve every snapshot, the
    unpinned default).
    """
    manifest = read_deployable_manifest(snapshot_dir)
    if manifest is None:
        return None
    iters = manifest.get("deployable_iters")
    if not iters:
        return None
    return sorted(int(i) for i in iters)
