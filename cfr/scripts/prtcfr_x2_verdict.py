"""cfr/scripts/prtcfr_x2_verdict.py

X2 re-specified gate verdict (v0.4 Phase 2): renders the X2 gate status or
verdict from the FROZEN pre-registration
(.docs/v0.4/phase2-throughput-pilot/x2-respec-preregistration.md, cambia-516/517)
and its amendment A3 (C-rep continuation past the ~330 flat, hub note cambia-632).

Encoding the frozen rules exactly is the point of this tool: a deviation poisons a
gate verdict. Every threshold and formula below is transcribed from the
pre-registration; none is a CLI knob, so a run of this script cannot silently
re-narrate a bar.

The tool is STAGE-AWARE. It mirrors the staged procedure and emits only the
status the data at hand authorizes:

  (a) C-rep pre-stop: trajectory status only; rule 1 (variance band B) is pending
      on C-rep's frozen plateau stop.
  (b) C-rep frozen stop recorded: compute B = |NashConv(C-rep) - 0.08895| / 0.08895
      (rule 1 / A3-1) and the rule-2 floor-move threshold max(B, 0.10).
  (c) A3 continuation: evaluate A3-3 (a post-stop checkpoint at iteration <= 700
      with NashConv <= bar_respec resolves "no floor at or above the bar") vs A3-4
      (no such checkpoint and the data reaches iteration 700 -> "floor at or above
      the bar"; the 700 window is the verdict input even if the run drifts past it,
      A3-5).
  (d) C0/C1 cells present (the A3-4 fork): rule 2 moved-the-floor per cell,
      measured at each cell's A3 continuation read-out floor (the post-stop
      minimum NashConv up to iteration 700), NOT the shared ~330 flat the
      plateau stop sits inside; then rule 4 (INDETERMINATE, C2 auto-queued only
      if BOTH C0 and C1 moved) or rule 5 (FAIL if neither moved).
  (e) Confirm seeds present: rule 3 PASS requires the passing cell <= bar_respec
      at its A3-3 read-out with a descent monotone under the stop rule, plus two
      additional seeds plateauing <= 1.25 * bar_respec, each under the A3 regime.

The plateau stop rule is NOT reimplemented here. The frozen-stop iteration is
derived by replaying a BestSnapshotController (src/cfr/prtcfr_stability.py)
offline over the (iteration, NashConv) series, ALWAYS with the frozen params: a
run's own recorded controller params are never adopted (a misconfigured run must
not be able to re-narrate its plateau stop). The run's recorded stop
(resume_state.json latch, cross-checked against logs/training.log) is read too; a
disagreement between the frozen replay and the record is reported loudly and
refuses to resolve into a verdict, EXCEPT that A3-3's authorized divergence stop
mode (recorded stopped=False on a clean descent) is not treated as a discrepancy
against the plateau replay. A run whose controller recorded non-frozen stop-rule
params is flagged as drift.

Data source is the run directory, per cell. The (iteration, NashConv) series and
the stop latch come from ``<run_dir>/resume_state.json`` (controller.history and
controller.stopped), cross-checked against ``<run_dir>/logs/training.log``. run_db
(``run_db.sqlite``) does NOT carry NashConv, so it is not a series source. The
convergence-audit CSV is a test fixture only, never a production input.

Usage:
  cd cfr && python scripts/prtcfr_x2_verdict.py runs/v0.4-x2r-crep-xpu
  cd cfr && python scripts/prtcfr_x2_verdict.py runs/v0.4-x2r-crep-xpu \
      --c0-dir runs/v0.4-x2r-c0-xpu --c1-dir runs/v0.4-x2r-c1-xpu \
      --confirm-seed-dir runs/seed2 --confirm-seed-dir runs/seed3 \
      --out runs/v0.4-x2r-crep-xpu/x2_verdict.json

Exit codes: 0 PASS, 1 FAIL, 2 INDETERMINATE, 3 PENDING (insufficient data),
4 DISCREPANCY (replay vs recorded stop disagree; no verdict rendered).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_stability import BestSnapshotController  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (provenance: x2-respec-preregistration.md, verbatim). None of
# these is a CLI override; the whole point of the tool is to hold them fixed.
# ---------------------------------------------------------------------------

# Fresh-run plateau NashConv; the fixed reference the variance band and the
# floor-move test are relative to. Rule 1 writes "0.08895" literally (the
# two-sig-fig value in the frozen text); the exact record is 0.08894883814554927.
# Provenance: Baseline facts, x2-respec-preregistration.md; run
# v0.4-x2-fresh-1000-xpu, plateau-stop at iter 330.
BASELINE_NASHCONV = 0.08895

# X2 gate bar on the corrected {A,6} tree. bar_respec = 0.0340 * U rounded to two
# significant figures, U = exact uniform-over-legal-actions NashConv.
# Provenance: Bar derivation + Results 2026-07-16 (cambia-517). Mirrors
# prtcfr_eval.X2_NASHCONV_BAR (consistency checked at import; drift warns, not
# fails).
BAR_RESPEC = 0.057

# Exact uniform-policy NashConv, recorded for the bar provenance only.
U_UNIFORM = 1.6727709190672155

# Rule 3 / A3-3 confirm-seed threshold: two additional seeds must plateau at or
# below 1.25 * bar_respec. 1.25 * 0.057 = 0.07125 (the frozen text displays the
# rounded 0.0713; the load-bearing quantity is 1.25 * bar_respec).
CONFIRM_SEED_FACTOR = 1.25
CONFIRM_SEED_BAR = CONFIRM_SEED_FACTOR * BAR_RESPEC  # 0.07125

# Rule 2 floor-move threshold floor: a cell "moved the floor" iff its plateau
# NashConv improves on the baseline by more than max(B, 0.10) relative.
FLOOR_MOVE_MIN_REL = 0.10

# A3-3/A3-4 read-out window upper bound. A crossing at or before iteration 700
# counts; a first crossing after 700 does not (A3-5: the 700 window is the verdict
# input even if the run drifts past it).
A3_READOUT_MAX_ITER = 700

# Frozen plateau stop-rule parameters (config/x2_tiny_gate.yaml: 50-iter window,
# <0.5% per 10-iter step, patience 5, min_iters 10, mode min). These are the ONLY
# params used to replay the stop: a loaded run's own controller params are never
# adopted (adopting them would let a misconfigured run re-narrate its plateau stop
# and poison rules 1/2). A run's recorded params are inspected for drift instead
# (_param_drift), and its stop_mode is read to decide whether the recorded stop is
# comparable to this frozen plateau replay (A3-3 permits a divergence stop mode).
FROZEN_STOP_PARAMS: Dict[str, Any] = {
    "mode": "min",
    "stop_mode": "plateau",
    "plateau_window_iters": 50,
    "plateau_step_iters": 10,
    "plateau_rel_improvement": 0.005,
    "patience": 5,
    "min_iters": 10,
    "rel_tolerance": 0.005,
}

# Verdict states and their process exit codes.
_EXIT_CODES = {
    "PASS": 0,
    "FAIL": 1,
    "INDETERMINATE": 2,
    "PENDING": 3,
    "DISCREPANCY": 4,
}


# ---------------------------------------------------------------------------
# Import-time consistency check: BAR_RESPEC must match prtcfr_eval.X2_NASHCONV_BAR.
# Read textually from the source file so this tool does not pull torch/engine just
# to compare a scalar. Drift warns (not fails), per the design contract.
# ---------------------------------------------------------------------------


def _eval_bar_from_source() -> Optional[float]:
    """Parse ``X2_NASHCONV_BAR = <float>`` from src/cfr/prtcfr_eval.py.

    Returns the float, or None if the file or the assignment cannot be read. A
    textual read avoids importing prtcfr_eval (which pulls torch/engine).
    """
    src = Path(__file__).resolve().parent.parent / "src" / "cfr" / "prtcfr_eval.py"
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^X2_NASHCONV_BAR\s*=\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _check_bar_consistency() -> Tuple[Optional[float], bool]:
    """Compare BAR_RESPEC to the scorer's X2_NASHCONV_BAR; warn on drift."""
    eval_bar = _eval_bar_from_source()
    if eval_bar is None:
        warnings.warn(
            "could not read X2_NASHCONV_BAR from src/cfr/prtcfr_eval.py; skipping "
            "the bar-consistency check (this verdict still uses the frozen "
            "bar_respec = %.4f)" % BAR_RESPEC,
            stacklevel=2,
        )
        return None, True
    consistent = abs(eval_bar - BAR_RESPEC) < 1e-12
    if not consistent:
        warnings.warn(
            "BAR DRIFT: this tool's frozen bar_respec = %.6f but "
            "prtcfr_eval.X2_NASHCONV_BAR = %.6f. The scorer and the verdict tool "
            "disagree on the X2 bar; reconcile before acting on this verdict."
            % (BAR_RESPEC, eval_bar),
            stacklevel=2,
        )
    return eval_bar, consistent


_EVAL_BAR, _EVAL_BAR_CONSISTENT = _check_bar_consistency()


# ---------------------------------------------------------------------------
# Frozen-stop replay (reuses BestSnapshotController; never reimplements the rule)
# ---------------------------------------------------------------------------


def replay_frozen_stop(
    series: List[Tuple[int, float]], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Replay the plateau stop rule over a (iteration, NashConv) series.

    Constructs the run's own BestSnapshotController and feeds the series in
    iteration order, returning the first ``should_stop`` firing. Pure list/dict
    logic in the controller: no torch/engine, fully offline.

    Returns ``{stopped, stop_iter, stop_nashconv, best_iter, best_metric}``.
    """
    controller = BestSnapshotController(
        mode=params.get("mode", "min"),
        stop_mode="plateau",
        plateau_window_iters=int(params.get("plateau_window_iters", 50)),
        plateau_step_iters=int(params.get("plateau_step_iters", 10)),
        plateau_rel_improvement=float(params.get("plateau_rel_improvement", 0.005)),
        patience=int(params.get("patience", 5)),
        min_iters=int(params.get("min_iters", 10)),
        rel_tolerance=float(params.get("rel_tolerance", 0.005)),
    )
    stop_iter: Optional[int] = None
    stop_nc: Optional[float] = None
    for it, nc in series:
        decision = controller.update(int(it), float(nc))
        if decision.should_stop:
            stop_iter = int(it)
            stop_nc = float(nc)
            break
    return {
        "stopped": stop_iter is not None,
        "stop_iter": stop_iter,
        "stop_nashconv": stop_nc,
        "best_iter": controller.best_iteration,
        "best_metric": (
            controller.best_metric if controller.best_metric != float("inf") else None
        ),
    }


def _descent_monotone_under_stop_rule(
    series: List[Tuple[int, float]], decision_iter: int, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Operational reading of rule 3's "descent monotone under the stop rule".

    The frozen text does not pin a formal monotonicity test. This uses the stop
    rule's OWN divergence detector: replay a divergence-mode controller with the
    run's rel_tolerance/patience over [1 .. decision_iter]; the descent is
    "monotone under the stop rule" iff no sustained divergence excursion fired
    (no ``patience`` consecutive checks worse than best * (1 + rel_tolerance)) and
    the decision-point NashConv is the minimum over that prefix. Both properties
    are what the stop rule assumes when it pins the deployable window to the best
    iteration. Single-step wiggles inside the tolerance band do not break it; a
    genuine oscillation (a sustained rise) does. Evidence is returned so the call
    is auditable rather than opaque.
    """
    prefix = [(it, nc) for it, nc in series if it <= decision_iter]
    controller = BestSnapshotController(
        mode=params.get("mode", "min"),
        stop_mode="divergence",
        rel_tolerance=float(params.get("rel_tolerance", 0.005)),
        patience=int(params.get("patience", 5)),
        min_iters=int(params.get("min_iters", 10)),
    )
    divergence_fired_at: Optional[int] = None
    max_worse_streak = 0
    for it, nc in prefix:
        decision = controller.update(int(it), float(nc))
        max_worse_streak = max(max_worse_streak, decision.num_worse_since_best)
        if decision.should_stop and divergence_fired_at is None:
            divergence_fired_at = int(it)
    prefix_min = min((nc for _, nc in prefix), default=None)
    decision_nc = next((nc for it, nc in prefix if it == decision_iter), None)
    ends_at_min = (
        prefix_min is not None
        and decision_nc is not None
        and abs(decision_nc - prefix_min) < 1e-15
    )
    monotone = divergence_fired_at is None and ends_at_min
    return {
        "monotone": bool(monotone),
        "divergence_fired_at": divergence_fired_at,
        "max_worse_streak": max_worse_streak,
        "prefix_min_nashconv": prefix_min,
        "decision_nashconv": decision_nc,
        "ends_at_prefix_min": bool(ends_at_min),
    }


# ---------------------------------------------------------------------------
# Cell read-out
# ---------------------------------------------------------------------------


@dataclass
class CellReadout:
    """Everything the frozen rules need from one run directory's series."""

    name: str
    run_dir: Optional[str] = None
    series: List[Tuple[int, float]] = field(default_factory=list)
    stop_params: Dict[str, Any] = field(default_factory=lambda: dict(FROZEN_STOP_PARAMS))

    # replayed frozen plateau stop
    replay_stopped: bool = False
    replay_stop_iter: Optional[int] = None
    replay_stop_nashconv: Optional[float] = None
    replay_best_iter: int = 0
    replay_best_metric: Optional[float] = None

    # recorded stop (resume_state.json latch, cross-checked against training.log)
    recorded_stopped: Optional[bool] = None
    recorded_stop_iter: Optional[int] = None
    recorded_best_iter: Optional[int] = None
    discrepancy: Optional[str] = None
    # stop_mode the run itself used (A3-3 permits divergence); "plateau" by
    # default so a synthetic cell is treated as plateau-governed.
    recorded_stop_mode: str = "plateau"
    # non-frozen stop-rule params the run recorded, if any (drift signal).
    param_drift: Optional[str] = None
    # cross-checks that could not run (e.g. training.log absent); informational.
    crosscheck_note: Optional[str] = None

    # A3 read-out window
    max_iter: Optional[int] = None
    reaches_readout_end: bool = False
    bar_cross_iter: Optional[int] = None
    bar_cross_nashconv: Optional[float] = None
    confirm_cross_iter: Optional[int] = None
    confirm_cross_nashconv: Optional[float] = None
    readout_min_iter: Optional[int] = None
    readout_min_nashconv: Optional[float] = None
    descent_monotone: Optional[bool] = None
    monotone_evidence: Dict[str, Any] = field(default_factory=dict)

    load_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": self.run_dir,
            "n_points": len(self.series),
            "first_iter": self.series[0][0] if self.series else None,
            "last_iter": self.series[-1][0] if self.series else None,
            "last_nashconv": self.series[-1][1] if self.series else None,
            "replay_stopped": self.replay_stopped,
            "replay_stop_iter": self.replay_stop_iter,
            "replay_stop_nashconv": self.replay_stop_nashconv,
            "replay_best_iter": self.replay_best_iter,
            "recorded_stopped": self.recorded_stopped,
            "recorded_stop_iter": self.recorded_stop_iter,
            "recorded_stop_mode": self.recorded_stop_mode,
            "discrepancy": self.discrepancy,
            "param_drift": self.param_drift,
            "crosscheck_note": self.crosscheck_note,
            "max_iter": self.max_iter,
            "reaches_readout_end": self.reaches_readout_end,
            "bar_cross_iter": self.bar_cross_iter,
            "bar_cross_nashconv": self.bar_cross_nashconv,
            "confirm_cross_iter": self.confirm_cross_iter,
            "confirm_cross_nashconv": self.confirm_cross_nashconv,
            "readout_min_iter": self.readout_min_iter,
            "readout_min_nashconv": self.readout_min_nashconv,
            "descent_monotone": self.descent_monotone,
            "monotone_evidence": self.monotone_evidence,
            "load_error": self.load_error,
        }

    # A3 predicates -----------------------------------------------------------

    def crossed_bar(self) -> bool:
        """A3-3: a checkpoint at iteration <= 700 records NashConv <= bar_respec."""
        return self.bar_cross_iter is not None

    def floored(self) -> bool:
        """A3-4: no <= 700 bar crossing and the data reaches iteration 700."""
        return self.bar_cross_iter is None and self.reaches_readout_end

    def confirm_passes(self) -> bool:
        """Confirm seed: a checkpoint at iteration <= 700 <= 1.25 * bar_respec."""
        return self.confirm_cross_iter is not None


def build_cell_readout(
    name: str,
    series: List[Tuple[int, float]],
    stop_params: Optional[Dict[str, Any]] = None,
    recorded_stopped: Optional[bool] = None,
    recorded_stop_iter: Optional[int] = None,
    recorded_best_iter: Optional[int] = None,
    run_dir: Optional[str] = None,
) -> CellReadout:
    """Compute a CellReadout from a raw (iteration, NashConv) series.

    Runs the frozen-stop replay, derives the A3 read-out fields, and (when a
    recorded stop is supplied) checks it against the replay.

    ``stop_params`` is the run's OWN recorded controller params. They are never
    adopted for the replay (the replay always uses the frozen params); they are
    inspected for drift and their stop_mode is read to decide whether the
    recorded stop is comparable to the frozen plateau replay.
    """
    frozen = dict(FROZEN_STOP_PARAMS)
    run_params = dict(stop_params) if stop_params else {}
    series = sorted(((int(it), float(nc)) for it, nc in series), key=lambda p: p[0])

    cell = CellReadout(name=name, run_dir=run_dir, series=series, stop_params=frozen)
    cell.recorded_stop_mode = str(run_params.get("stop_mode", "plateau"))
    cell.param_drift = _param_drift(run_params)
    if not series:
        cell.load_error = "empty series"
        return cell

    replay = replay_frozen_stop(series, frozen)
    cell.replay_stopped = replay["stopped"]
    cell.replay_stop_iter = replay["stop_iter"]
    cell.replay_stop_nashconv = replay["stop_nashconv"]
    cell.replay_best_iter = replay["best_iter"]
    cell.replay_best_metric = replay["best_metric"]

    cell.recorded_stopped = recorded_stopped
    cell.recorded_stop_iter = recorded_stop_iter
    cell.recorded_best_iter = recorded_best_iter
    cell.discrepancy = _stop_discrepancy(cell)

    cell.max_iter = series[-1][0]
    cell.reaches_readout_end = cell.max_iter >= A3_READOUT_MAX_ITER

    # Read-out region: the continuation after the plateau stop, up to iteration
    # 700. "After the stop" places the crossing physically past the ~330 flat;
    # every pre-stop checkpoint is far above the bar, so restricting to
    # iteration > stop_iter never excludes a real crossing. When the run has not
    # yet stopped there is no read-out.
    if cell.replay_stopped and cell.replay_stop_iter is not None:
        readout = [
            (it, nc)
            for it, nc in series
            if cell.replay_stop_iter < it <= A3_READOUT_MAX_ITER
        ]
    else:
        readout = []

    for it, nc in readout:
        if cell.bar_cross_iter is None and nc <= BAR_RESPEC:
            cell.bar_cross_iter, cell.bar_cross_nashconv = it, nc
        if cell.confirm_cross_iter is None and nc <= CONFIRM_SEED_BAR:
            cell.confirm_cross_iter, cell.confirm_cross_nashconv = it, nc
    if readout:
        cell.readout_min_iter, cell.readout_min_nashconv = min(
            readout, key=lambda p: p[1]
        )

    if cell.bar_cross_iter is not None:
        evidence = _descent_monotone_under_stop_rule(series, cell.bar_cross_iter, frozen)
        cell.descent_monotone = evidence["monotone"]
        cell.monotone_evidence = evidence

    return cell


def _param_drift(run_params: Dict[str, Any]) -> Optional[str]:
    """Describe any frozen stop-rule param the run recorded with a non-frozen
    value, else None.

    The replay always uses the frozen params, so a drifted run is still scored on
    the frozen rule; this only reports that the run itself did not follow the
    frozen stop rule (so its own recorded stop is untrustworthy). ``stop_mode`` is
    exempt: A3-3 authorizes a divergence stop mode (or a latched controller) for
    the read-out regime of confirm seeds and resubmitted C0/C1.
    """
    drifts = []
    for key, frozen_val in FROZEN_STOP_PARAMS.items():
        if key == "stop_mode" or key not in run_params:
            continue
        run_val = run_params[key]
        if isinstance(frozen_val, float) or isinstance(run_val, float):
            same = abs(float(run_val) - float(frozen_val)) < 1e-12
        else:
            same = run_val == frozen_val
        if not same:
            drifts.append("%s=%s (frozen %s)" % (key, run_val, frozen_val))
    if not drifts:
        return None
    return (
        "run controller recorded non-frozen stop-rule params: %s; the replay uses "
        "the frozen params, but the run did not follow the frozen stop rule"
        % ", ".join(drifts)
    )


def _stop_discrepancy(cell: CellReadout) -> Optional[str]:
    """Loud description of any replay-vs-record disagreement, else None."""
    problems = []
    if cell.param_drift:
        problems.append(cell.param_drift)
    # The recorded stopped flag and stop iteration cross-check the frozen plateau
    # replay only when the run itself ran the plateau rule. A3-3 permits a
    # divergence stop mode (or a latched controller) whose recorded stop is not
    # comparable to the plateau replay: on a clean descent the divergence rule
    # never fires (recorded stopped=False) while the plateau replay fires at the
    # ~330 flat (replay stopped=True). Comparing the two there manufactures a
    # false discrepancy, so the flag/iter checks run only under plateau mode.
    if cell.recorded_stop_mode == "plateau":
        if (
            cell.recorded_stopped is not None
            and cell.recorded_stopped != cell.replay_stopped
        ):
            problems.append(
                "recorded stopped=%s but offline replay of the frozen stop rule "
                "yields stopped=%s" % (cell.recorded_stopped, cell.replay_stopped)
            )
        if (
            cell.recorded_stop_iter is not None
            and cell.replay_stopped
            and cell.recorded_stop_iter != cell.replay_stop_iter
        ):
            problems.append(
                "recorded stop iteration=%s but offline replay fires at iteration=%s"
                % (cell.recorded_stop_iter, cell.replay_stop_iter)
            )
    if not problems:
        return None
    return "; ".join(problems)


# ---------------------------------------------------------------------------
# Run-directory loader (resume_state.json series + latch, training.log crosscheck)
# ---------------------------------------------------------------------------


def _parse_training_log_stop(log_path: Path) -> Tuple[Optional[int], Optional[bool]]:
    """Recorded stop from logs/training.log: (stop_iter, saw_stop).

    Prefers the terminal ``early-stop at iter=N`` line; falls back to the first
    ``stability iter=N ... stop=True`` line. Returns (None, None) when the log is
    absent, (None, False) when present with no stop recorded.
    """
    if not log_path.is_file():
        return None, None
    stop_iter: Optional[int] = None
    saw_stop = False
    terminal_iter: Optional[int] = None
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m_term = re.search(r"early-stop at iter=(\d+)", line)
            if m_term:
                terminal_iter = int(m_term.group(1))
                saw_stop = True
                continue
            m = re.search(r"stability iter=(\d+).*stop=(True|False)", line)
            if m and m.group(2) == "True" and stop_iter is None:
                stop_iter = int(m.group(1))
                saw_stop = True
    return (terminal_iter if terminal_iter is not None else stop_iter, saw_stop)


def load_cell(run_dir: str, name: str) -> CellReadout:
    """Load a CellReadout from a run directory.

    Series and stop latch come from ``resume_state.json`` (controller.history and
    controller.stopped); the stop iteration is cross-checked against
    ``logs/training.log``. Missing or malformed inputs are recorded as a
    ``load_error`` rather than raised, so a stage-aware verdict can still report
    which cell's data is absent.
    """
    path = Path(run_dir)
    cell = CellReadout(name=name, run_dir=str(path))
    resume_path = path / "resume_state.json"
    if not resume_path.is_file():
        cell.load_error = "no resume_state.json in %s" % run_dir
        return cell
    try:
        with open(resume_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError) as exc:
        cell.load_error = "could not read resume_state.json: %s" % exc
        return cell

    controller = state.get("controller") or {}
    history = controller.get("history") or []
    series = [(int(h["iteration"]), float(h["metric"])) for h in history]
    if not series:
        cell.load_error = "resume_state.json controller.history is empty"
        return cell

    # The frozen stop replay and the A3 read-out both assume the full trajectory
    # from iteration 1. A truncated (resume / net-only warm-start) history that
    # begins mid-run would replay the plateau stop over the tail and emit a
    # plausible-looking but wrong verdict (including a negative floor-move
    # target). Sanctioned A3 in-place resume preserves the whole history, so a
    # mid-run first iteration signals a wrong run directory: refuse it.
    first_iter = min(it for it, _ in series)
    if first_iter > 1:
        cell.load_error = (
            "resume_state.json controller.history begins at iteration %d, not 1; a "
            "truncated or warm-start history is not a valid X2 series source (the "
            "frozen stop replay and the A3 read-out assume the full trajectory from "
            "iteration 1). Point the tool at the full-history run directory." % first_iter
        )
        return cell

    params = {k: controller[k] for k in FROZEN_STOP_PARAMS if k in controller}
    log_stop_iter, log_saw_stop = _parse_training_log_stop(path / "logs" / "training.log")

    cell = build_cell_readout(
        name=name,
        series=series,
        stop_params=params,
        recorded_stopped=controller.get("stopped"),
        recorded_stop_iter=log_stop_iter,
        recorded_best_iter=controller.get("best_iteration"),
        run_dir=str(path),
    )
    # When logs/training.log is absent the stop-iteration cross-check cannot run
    # (resume_state records only the stopped flag and best_iteration, not the
    # stop-fire iteration). Record a note so the verdict is not presented as
    # fully cross-checked, rather than silently dropping the check.
    if log_saw_stop is None:
        cell.crosscheck_note = (
            "logs/training.log absent: the stop-iteration cross-check was not "
            "performed; only the resume_state stopped flag was checked against the "
            "frozen plateau replay"
        )
    return cell


# ---------------------------------------------------------------------------
# Frozen decision rules
# ---------------------------------------------------------------------------


def variance_band(crep: CellReadout) -> float:
    """Rule 1 / A3-1: B = |NashConv(C-rep at its frozen stop) - 0.08895| / 0.08895."""
    assert crep.replay_stop_nashconv is not None
    return abs(crep.replay_stop_nashconv - BASELINE_NASHCONV) / BASELINE_NASHCONV


def floor_move_threshold(band_b: float) -> float:
    """Rule 2 threshold: max(B, 0.10) relative improvement over the baseline."""
    return max(band_b, FLOOR_MOVE_MIN_REL)


def _floor_move_nashconv(cell: CellReadout) -> Optional[float]:
    """The NashConv rule 2 reads as a cell's floor.

    Under amendment A3 the capacity/coverage cells are continued past the ~330
    transient flat, so the floor is the minimum over the post-stop A3 read-out
    window (iteration <= 700), NOT the plateau-stop value. The plateau stop sits
    inside the flat that A3's basis says every cell shares at ~0.088; reading it
    would score the mechanism on the region A3 exists to look past, drive rule 4
    (INDETERMINATE) out of reach, and misfire rule 5 (FAIL) on a cell that in
    fact descended far in continuation. Falls back to the plateau-stop value when
    a cell has no continuation read-out (a base-pre-registration cell that
    stopped without being continued); None when neither is defined.
    """
    if cell.readout_min_nashconv is not None:
        return cell.readout_min_nashconv
    return cell.replay_stop_nashconv


def moved_the_floor(cell: CellReadout, threshold_rel: float) -> Optional[bool]:
    """Rule 2: a cell moved the floor iff its floor NashConv improves on the
    baseline by more than ``threshold_rel`` relative.

    The floor NashConv is the A3 continuation read-out minimum (post ~330 flat,
    iteration <= 700); see _floor_move_nashconv. Returns None when the floor is
    undefined (the cell has neither an A3 read-out nor a plateau stop).
    """
    floor_nc = _floor_move_nashconv(cell)
    if floor_nc is None:
        return None
    improvement = (BASELINE_NASHCONV - floor_nc) / BASELINE_NASHCONV
    return improvement > threshold_rel


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------


def compute_verdict(
    crep_dir: Optional[str] = None,
    c0_dir: Optional[str] = None,
    c1_dir: Optional[str] = None,
    c2_dir: Optional[str] = None,
    confirm_seed_dirs: Optional[List[str]] = None,
    *,
    crep: Optional[CellReadout] = None,
    c0: Optional[CellReadout] = None,
    c1: Optional[CellReadout] = None,
    c2: Optional[CellReadout] = None,
    confirm_seeds: Optional[List[CellReadout]] = None,
) -> Dict[str, Any]:
    """Render the X2 status/verdict.

    Cells may be passed as run-directory paths (loaded here) or as pre-built
    CellReadout objects (the test path). Directory arguments take precedence when
    both are given for the same cell.
    """
    if crep_dir is not None:
        crep = load_cell(crep_dir, "C-rep")
    if c0_dir is not None:
        c0 = load_cell(c0_dir, "C0")
    if c1_dir is not None:
        c1 = load_cell(c1_dir, "C1")
    if c2_dir is not None:
        c2 = load_cell(c2_dir, "C2")
    if confirm_seed_dirs:
        confirm_seeds = [
            load_cell(d, "seed-%d" % (i + 2)) for i, d in enumerate(confirm_seed_dirs)
        ]
    confirm_seeds = confirm_seeds or []

    cells: Dict[str, Any] = {}
    for cell in [crep, c0, c1, c2, *confirm_seeds]:
        if cell is not None:
            cells[cell.name] = cell.to_dict()

    verdict: Dict[str, Any] = {
        "tool": "prtcfr_x2_verdict",
        "constants": {
            "baseline_nashconv": BASELINE_NASHCONV,
            "bar_respec": BAR_RESPEC,
            "u_uniform": U_UNIFORM,
            "confirm_seed_bar": CONFIRM_SEED_BAR,
            "floor_move_min_rel": FLOOR_MOVE_MIN_REL,
            "a3_readout_max_iter": A3_READOUT_MAX_ITER,
            "eval_bar_constant": _EVAL_BAR,
            "eval_bar_consistent": _EVAL_BAR_CONSISTENT,
        },
        "cells": cells,
        "rule_1_variance_band": None,
        "rule_2_floor_move": None,
        "a3_readout": {},
        "rule_3_pass": None,
        "rule_4_5": None,
        "stage": None,
        "pending": [],
        "discrepancies": [],
        "crosscheck_notes": [],
        "overall_verdict": None,
        "notes": [],
    }

    # Load / discrepancy gate ------------------------------------------------
    for cell in [crep, c0, c1, c2, *confirm_seeds]:
        if cell is None:
            continue
        if cell.discrepancy:
            verdict["discrepancies"].append("%s: %s" % (cell.name, cell.discrepancy))
        if cell.crosscheck_note:
            verdict["crosscheck_notes"].append(
                "%s: %s" % (cell.name, cell.crosscheck_note)
            )
    if verdict["discrepancies"]:
        verdict["stage"] = "discrepancy"
        verdict["overall_verdict"] = "DISCREPANCY"
        verdict["notes"].append(
            "Replay of the frozen stop rule disagrees with the recorded stop. "
            "Refusing to render a verdict until the disagreement is resolved."
        )
        return verdict

    if crep is None or crep.load_error or not crep.series:
        verdict["stage"] = "a"
        verdict["overall_verdict"] = "PENDING"
        reason = (
            crep.load_error
            if crep is not None and crep.load_error
            else "C-rep run directory not provided or carries no series"
        )
        verdict["pending"].append("rule 1 (variance band B): %s" % reason)
        return verdict

    # Stage (a): C-rep pre-stop ---------------------------------------------
    crep_recorded_stop = crep.recorded_stopped is True
    if not crep.replay_stopped and not crep_recorded_stop:
        verdict["stage"] = "a"
        verdict["overall_verdict"] = "PENDING"
        verdict["rule_1_variance_band"] = {
            "available": False,
            "note": (
                "C-rep has not reached its frozen plateau stop; last checkpoint "
                "iter=%s nashconv=%.6f. B (rule 1 / A3-1) is measured at the stop."
                % (crep.series[-1][0], crep.series[-1][1])
            ),
        }
        verdict["pending"].append(
            "rule 1 (variance band B): pending on C-rep frozen plateau stop"
        )
        verdict["pending"].append(
            "A3-3/A3-4 read-out: pending on C-rep continuation to iteration %d"
            % A3_READOUT_MAX_ITER
        )
        return verdict

    # Stage (b): C-rep frozen stop recorded -> rule 1 (B) + rule 2 threshold --
    band_b = variance_band(crep)
    threshold_rel = floor_move_threshold(band_b)
    floor_move_target = BASELINE_NASHCONV * (1.0 - threshold_rel)
    verdict["rule_1_variance_band"] = {
        "available": True,
        "crep_stop_iter": crep.replay_stop_iter,
        "crep_stop_nashconv": crep.replay_stop_nashconv,
        "baseline_nashconv": BASELINE_NASHCONV,
        "B": band_b,
    }
    verdict["rule_2_floor_move"] = {
        "threshold_rel": threshold_rel,
        "threshold_source": "max(B, 0.10)",
        "floor_move_target_nashconv": floor_move_target,
        "cells": {},
    }

    # A3 read-out per cell ---------------------------------------------------
    for cell in [crep, c0, c1, c2, *confirm_seeds]:
        if cell is None or cell.load_error or not cell.series:
            continue
        if cell.crossed_bar():
            status = "A3-3-bar-cross"
        elif cell.floored():
            status = "A3-4-floored"
        else:
            status = "pending"
        verdict["a3_readout"][cell.name] = {
            "status": status,
            "bar_cross_iter": cell.bar_cross_iter,
            "bar_cross_nashconv": cell.bar_cross_nashconv,
            "readout_min_iter": cell.readout_min_iter,
            "readout_min_nashconv": cell.readout_min_nashconv,
            "reaches_readout_end": cell.reaches_readout_end,
        }

    # Passing cell: first provided cell that crosses the bar within <= 700 -----
    passing_cell = None
    for cell in [crep, c0, c1, c2]:
        if cell is not None and not cell.load_error and cell.crossed_bar():
            passing_cell = cell
            break

    if passing_cell is not None:
        return _finish_pass_path(verdict, passing_cell, confirm_seeds)

    # No bar crossing anywhere. Is C-rep floored (A3-4) or still pending? ------
    if not crep.floored():
        verdict["stage"] = "c"
        verdict["overall_verdict"] = "PENDING"
        verdict["pending"].append(
            "A3-3/A3-4: C-rep continuation has not reached iteration %d and has "
            "not crossed bar_respec; the floor read-out is undecided"
            % A3_READOUT_MAX_ITER
        )
        return verdict

    # A3-4 fork: C-rep floored -> capacity/coverage cells C0/C1 (rules 2/4/5) --
    return _finish_c0c1_path(verdict, c0, c1, c2, threshold_rel)


def _finish_pass_path(
    verdict: Dict[str, Any],
    passing_cell: CellReadout,
    confirm_seeds: List[CellReadout],
) -> Dict[str, Any]:
    """Rule 3 PASS assessment: bar-cross + monotone descent + two confirm seeds."""
    verdict["stage"] = "e"
    monotone = bool(passing_cell.descent_monotone)
    passed_seeds = [s for s in confirm_seeds if s.confirm_passes()]
    pending_seeds = [
        s for s in confirm_seeds if not s.confirm_passes() and not s.reaches_readout_end
    ]
    n_needed = 2
    verdict["rule_3_pass"] = {
        "passing_cell": passing_cell.name,
        "bar_cross_iter": passing_cell.bar_cross_iter,
        "bar_cross_nashconv": passing_cell.bar_cross_nashconv,
        "descent_monotone": monotone,
        "monotone_evidence": passing_cell.monotone_evidence,
        "confirm_seed_bar": CONFIRM_SEED_BAR,
        "confirm_seeds_needed": n_needed,
        "confirm_seeds_passed": [s.name for s in passed_seeds],
        "confirm_seeds_pending": [s.name for s in pending_seeds],
    }

    if not monotone:
        verdict["overall_verdict"] = "PENDING"
        verdict["pending"].append(
            "rule 3 PASS: %s crossed bar_respec but the descent is not monotone "
            "under the stop rule (divergence detector fired at iter=%s); a "
            "bar-cross via an oscillating trajectory does not satisfy rule 3"
            % (
                passing_cell.name,
                passing_cell.monotone_evidence.get("divergence_fired_at"),
            )
        )
        return verdict

    if len(passed_seeds) >= n_needed:
        verdict["overall_verdict"] = "PASS"
        verdict["notes"].append(
            "A3-3 resolved: %s reached NashConv %.6f <= bar_respec %.4f at iter %d "
            "(no floor at or above the bar). Two confirm seeds plateaued "
            "<= 1.25 * bar_respec. X2 PASS."
            % (
                passing_cell.name,
                passing_cell.bar_cross_nashconv,
                BAR_RESPEC,
                passing_cell.bar_cross_iter,
            )
        )
        return verdict

    verdict["overall_verdict"] = "PENDING"
    verdict["pending"].append(
        "rule 3 PASS: %s crossed bar_respec (A3-3 resolved: no floor at or above "
        "the bar) but only %d of %d confirm seeds have plateaued <= 1.25 * "
        "bar_respec (%.5f); PASS is pending the remaining confirm seed(s)"
        % (passing_cell.name, len(passed_seeds), n_needed, CONFIRM_SEED_BAR)
    )
    return verdict


def _finish_c0c1_path(
    verdict: Dict[str, Any],
    c0: Optional[CellReadout],
    c1: Optional[CellReadout],
    c2: Optional[CellReadout],
    threshold_rel: float,
) -> Dict[str, Any]:
    """Rules 2/4/5 over the A3-4 capacity/coverage fork (C-rep floored)."""
    verdict["stage"] = "d"

    # Both C0 and C1 must have reached their <= 700 read-out end (A3-4 floor
    # read-out) with no bar crossing before rules 2/4/5 can be applied. A missing
    # or still-running cell keeps the verdict PENDING, never a premature FAIL.
    missing = []
    for cell, label in [(c0, "C0"), (c1, "C1")]:
        if cell is None or cell.load_error or not cell.series:
            missing.append(
                "%s: %s"
                % (label, (cell.load_error if cell is not None else "not provided"))
            )
        elif not cell.reaches_readout_end:
            missing.append(
                "%s: read-out has not reached iteration %d (last iter=%s)"
                % (label, A3_READOUT_MAX_ITER, cell.max_iter)
            )
    if missing:
        verdict["overall_verdict"] = "PENDING"
        verdict["pending"].append(
            "rules 2/4/5 (A3-4 fork): C-rep floored at or above the bar; the "
            "capacity/coverage cells are not both read out to iteration %d -> %s"
            % (A3_READOUT_MAX_ITER, "; ".join(missing))
        )
        return verdict

    moved_c0 = moved_the_floor(c0, threshold_rel)
    moved_c1 = moved_the_floor(c1, threshold_rel)
    verdict["rule_2_floor_move"]["cells"] = {
        "C0": {
            "plateau_stop_iter": c0.replay_stop_iter,
            "plateau_stop_nashconv": c0.replay_stop_nashconv,
            "readout_floor_iter": c0.readout_min_iter,
            "readout_floor_nashconv": _floor_move_nashconv(c0),
            "moved_floor": moved_c0,
        },
        "C1": {
            "plateau_stop_iter": c1.replay_stop_iter,
            "plateau_stop_nashconv": c1.replay_stop_nashconv,
            "readout_floor_iter": c1.readout_min_iter,
            "readout_floor_nashconv": _floor_move_nashconv(c1),
            "moved_floor": moved_c1,
        },
    }

    both_moved = bool(moved_c0) and bool(moved_c1)
    either_moved = bool(moved_c0) or bool(moved_c1)
    c2_auto_queued = both_moved

    rule_4_5 = {
        "c0_moved_floor": moved_c0,
        "c1_moved_floor": moved_c1,
        "c2_auto_queued": c2_auto_queued,
    }

    if not either_moved:
        # Rule 5: neither C0 nor C1 moved the floor -> mechanism-defect FAIL.
        rule_4_5["rule"] = 5
        verdict["rule_4_5"] = rule_4_5
        verdict["overall_verdict"] = "FAIL"
        verdict["notes"].append(
            "Rule 5: neither C0 nor C1 moved the floor (> max(B, 0.10) = %.4f "
            "relative over baseline %.5f), and no cell reached bar_respec. X2 FAIL "
            "(mechanism-defect presumption); Phase 1 reopens as a defect hunt."
            % (threshold_rel, BASELINE_NASHCONV)
        )
        return verdict

    # Rule 4: at least one of C0/C1 moved but no cell reached the bar.
    rule_4_5["rule"] = 4
    verdict["rule_4_5"] = rule_4_5
    verdict["overall_verdict"] = "INDETERMINATE"
    if c2_auto_queued and c2 is not None and not c2.load_error and c2.crossed_bar():
        # C2 already run and crossed: escalate to the PASS path (rule 3 governs).
        return _finish_pass_path(verdict, c2, [])
    if c2_auto_queued:
        verdict["notes"].append(
            "Rule 4: BOTH C0 and C1 moved the floor but no cell reached bar_respec. "
            "C2 (C0 + C1 combined) is auto-queued. If C2 still misses the bar the "
            "question is a capacity/sample-budget decision for the user, not a "
            "soundness ruling."
        )
    else:
        moved_name = "C0" if moved_c0 else "C1"
        verdict["notes"].append(
            "Rule 4: %s moved the floor but no cell reached bar_respec; the other "
            "capacity/coverage cell did not move it, so C2 is NOT auto-queued "
            "(auto-queue requires both)." % moved_name
        )
    return verdict


# ---------------------------------------------------------------------------
# Human-readable summary + pre-registration Results append block
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], places: int = 6) -> str:
    if value is None:
        return "n/a"
    return ("%%.%df" % places) % value


def human_summary(verdict: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("X2 re-spec gate verdict (frozen pre-registration + amendment A3)")
    c = verdict["constants"]
    lines.append(
        "constants: baseline=%.5f bar_respec=%.4f confirm_bar=%.5f "
        "floor-move floor=max(B,%.2f) read-out<=iter %d"
        % (
            c["baseline_nashconv"],
            c["bar_respec"],
            c["confirm_seed_bar"],
            c["floor_move_min_rel"],
            c["a3_readout_max_iter"],
        )
    )
    if not c["eval_bar_consistent"]:
        lines.append(
            "  WARNING: bar drift vs prtcfr_eval.X2_NASHCONV_BAR=%s"
            % c["eval_bar_constant"]
        )
    lines.append("stage: %s" % verdict["stage"])
    lines.append("")

    if verdict["discrepancies"]:
        lines.append("!! STOP-RULE DISCREPANCY (no verdict rendered):")
        for d in verdict["discrepancies"]:
            lines.append("   %s" % d)
        lines.append("")

    if verdict.get("crosscheck_notes"):
        lines.append("cross-check notes (verdict not fully cross-checked):")
        for n in verdict["crosscheck_notes"]:
            lines.append("   %s" % n)
        lines.append("")

    for name, cell in verdict["cells"].items():
        if cell.get("load_error"):
            lines.append("cell %-7s LOAD ERROR: %s" % (name, cell["load_error"]))
            continue
        lines.append(
            "cell %-7s n=%d iters[%s..%s] last_nc=%s stop@%s(nc=%s) crossed_bar@%s"
            % (
                name,
                cell["n_points"],
                cell["first_iter"],
                cell["last_iter"],
                _fmt(cell["last_nashconv"]),
                cell["replay_stop_iter"],
                _fmt(cell["replay_stop_nashconv"]),
                cell["bar_cross_iter"],
            )
        )
    lines.append("")

    r1 = verdict["rule_1_variance_band"]
    if r1 and r1.get("available"):
        lines.append(
            "rule 1: B = |%.6f - %.5f| / %.5f = %.5f"
            % (
                r1["crep_stop_nashconv"],
                BASELINE_NASHCONV,
                BASELINE_NASHCONV,
                r1["B"],
            )
        )
    elif r1:
        lines.append("rule 1: %s" % r1.get("note"))

    r2 = verdict["rule_2_floor_move"]
    if r2:
        lines.append(
            "rule 2: floor-move threshold = %s = %.5f -> a cell moves the floor "
            "iff plateau NashConv < %.6f"
            % (
                r2["threshold_source"],
                r2["threshold_rel"],
                r2["floor_move_target_nashconv"],
            )
        )
        for cname, cinfo in (r2.get("cells") or {}).items():
            lines.append(
                "  %s A3 read-out floor nc=%s (plateau-stop nc=%s) moved_floor=%s"
                % (
                    cname,
                    _fmt(cinfo["readout_floor_nashconv"]),
                    _fmt(cinfo["plateau_stop_nashconv"]),
                    cinfo["moved_floor"],
                )
            )

    if verdict["a3_readout"]:
        lines.append("")
        lines.append("A3 read-out (post-stop, iteration <= %d):" % A3_READOUT_MAX_ITER)
        for cname, info in verdict["a3_readout"].items():
            lines.append(
                "  %-7s %s (bar-cross@%s nc=%s; min@%s nc=%s)"
                % (
                    cname,
                    info["status"],
                    info["bar_cross_iter"],
                    _fmt(info["bar_cross_nashconv"]),
                    info["readout_min_iter"],
                    _fmt(info["readout_min_nashconv"]),
                )
            )

    r3 = verdict["rule_3_pass"]
    if r3:
        lines.append("")
        lines.append(
            "rule 3: passing cell=%s bar-cross@%s (nc=%s) monotone=%s "
            "confirm-seeds passed=%s/%d pending=%s"
            % (
                r3["passing_cell"],
                r3["bar_cross_iter"],
                _fmt(r3["bar_cross_nashconv"]),
                r3["descent_monotone"],
                len(r3["confirm_seeds_passed"]),
                r3["confirm_seeds_needed"],
                r3["confirm_seeds_pending"] or "-",
            )
        )

    r45 = verdict["rule_4_5"]
    if r45:
        lines.append("")
        lines.append(
            "rule %d: C0 moved=%s C1 moved=%s C2 auto-queued=%s"
            % (
                r45["rule"],
                r45["c0_moved_floor"],
                r45["c1_moved_floor"],
                r45["c2_auto_queued"],
            )
        )

    if verdict["pending"]:
        lines.append("")
        lines.append("PENDING:")
        for p in verdict["pending"]:
            lines.append("  - %s" % p)
    if verdict["notes"]:
        lines.append("")
        for n in verdict["notes"]:
            lines.append(n)

    lines.append("")
    lines.append("OVERALL: %s" % verdict["overall_verdict"])
    return "\n".join(lines)


def results_append_block(verdict: Dict[str, Any]) -> str:
    """A markdown block ready to paste under the pre-registration's Results
    section. This is EMITTED ONLY; the tool never writes into the frozen doc.
    """
    lines: List[str] = []
    lines.append(
        "<!-- append under x2-respec-preregistration.md '## Results'; do "
        "not edit the frozen rules -->"
    )
    lines.append(
        "- (DATE, CITE): X2 verdict = %s (stage %s)."
        % (verdict["overall_verdict"], verdict["stage"])
    )
    r1 = verdict["rule_1_variance_band"]
    if r1 and r1.get("available"):
        lines.append(
            "  Rule 1: NashConv(C-rep) = %.6f at frozen plateau stop iter %s; "
            "B = %.5f." % (r1["crep_stop_nashconv"], r1["crep_stop_iter"], r1["B"])
        )
    r2 = verdict["rule_2_floor_move"]
    if r2:
        lines.append(
            "  Rule 2: floor-move threshold max(B, 0.10) = %.5f (plateau NashConv "
            "must be < %.6f to move the floor)."
            % (r2["threshold_rel"], r2["floor_move_target_nashconv"])
        )
        for cname, cinfo in (r2.get("cells") or {}).items():
            lines.append(
                "    %s: A3 read-out floor NashConv %s (plateau-stop NashConv %s), "
                "moved floor = %s."
                % (
                    cname,
                    _fmt(cinfo["readout_floor_nashconv"]),
                    _fmt(cinfo["plateau_stop_nashconv"]),
                    cinfo["moved_floor"],
                )
            )
    for cname, info in verdict["a3_readout"].items():
        lines.append(
            "  A3 read-out %s: %s (bar-cross iter %s nc %s; read-out min iter %s "
            "nc %s)."
            % (
                cname,
                info["status"],
                info["bar_cross_iter"],
                _fmt(info["bar_cross_nashconv"]),
                info["readout_min_iter"],
                _fmt(info["readout_min_nashconv"]),
            )
        )
    r3 = verdict["rule_3_pass"]
    if r3:
        lines.append(
            "  Rule 3: passing cell %s reached bar_respec at iter %s (nc %s); "
            "descent monotone = %s; confirm seeds passed %d/%d."
            % (
                r3["passing_cell"],
                r3["bar_cross_iter"],
                _fmt(r3["bar_cross_nashconv"]),
                r3["descent_monotone"],
                len(r3["confirm_seeds_passed"]),
                r3["confirm_seeds_needed"],
            )
        )
    r45 = verdict["rule_4_5"]
    if r45:
        lines.append(
            "  Rule %d: C0 moved floor = %s, C1 moved floor = %s, C2 auto-queued "
            "= %s."
            % (
                r45["rule"],
                r45["c0_moved_floor"],
                r45["c1_moved_floor"],
                r45["c2_auto_queued"],
            )
        )
    for p in verdict["pending"]:
        lines.append("  PENDING: %s" % p)
    for d in verdict["discrepancies"]:
        lines.append("  DISCREPANCY: %s" % d)
    for n in verdict.get("crosscheck_notes", []):
        lines.append("  CROSS-CHECK NOT PERFORMED: %s" % n)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "PRT-CFR X2 re-spec gate verdict: renders the frozen "
            "pre-registration (x2-respec-preregistration.md) + amendment A3 "
            "status/verdict from per-cell run directories. Stage-aware: emits "
            "only what the data authorizes. Frozen constants are not CLI knobs."
        )
    )
    ap.add_argument(
        "crep_dir",
        help="C-rep run directory (variance-band cell; required).",
    )
    ap.add_argument("--c0-dir", default=None, help="C0 (coverage-restore) run dir.")
    ap.add_argument("--c1-dir", default=None, help="C1 (capacity-scaling) run dir.")
    ap.add_argument("--c2-dir", default=None, help="C2 (C0+C1 combined) run dir.")
    ap.add_argument(
        "--confirm-seed-dir",
        action="append",
        default=None,
        dest="confirm_seed_dirs",
        help="Confirm-seed run dir (repeatable; rule 3 needs two passing).",
    )
    ap.add_argument("--out", default=None, help="JSON verdict output path.")
    ap.add_argument(
        "--no-results-block",
        action="store_true",
        help="Suppress the pre-registration Results append block on stdout.",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verdict = compute_verdict(
        crep_dir=args.crep_dir,
        c0_dir=args.c0_dir,
        c1_dir=args.c1_dir,
        c2_dir=args.c2_dir,
        confirm_seed_dirs=args.confirm_seed_dirs,
    )
    print(human_summary(verdict))
    if not args.no_results_block:
        print("")
        print("--- pre-registration Results append block (emit only) ---")
        print(results_append_block(verdict))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, indent=2, sort_keys=False)
        print("\n[prtcfr-x2-verdict] wrote %s" % args.out)
    return _EXIT_CODES.get(verdict["overall_verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
