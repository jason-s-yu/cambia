"""src/cfr/prtcfr_battery.py

Production in-loop X4 battery (Phase 2 S2W1): the Tier-A LBR fast-lane eval_fn
for ``PRTCFRProductionTrainer``'s stability controller.

``build_production_battery_eval_fn`` returns ``eval_fn(trainer, t) -> float``.
At each stability-cadence iteration the trainer already calls this with the
current iteration ``t``; the returned value is the stability trend metric
(``stability_metric_mode="min"``, lower is better). The eval_fn:

  (a) constructs the SAME SD-CFR snapshot mixture the final eval serves
      (``PRTCFRAgentWrapper`` over a ``PRTCFRMixture``), but pinned to snapshots
      ``[1..t]`` -- the grown in-loop window -- rather than the deployable
      manifest's ``[1..best]``. The manifest pins to ``best_iteration`` after
      the first cadence, which would flatten the trend once ``best`` stops
      advancing; the AC2 slope test needs the mixture measured AS OF iter ``t``,
      so the window is rebuilt directly from the snapshots on disk. The mixture
      object and policy path are byte-identical to the final eval; only the iter
      selection differs.
  (b) runs the Tier-A LBR fast lane (``sampled_lbr``) against that mixture at a
      small game/rollout count (``battery_lbr_games`` / ``battery_lbr_depth``) --
      NOT the post-hoc mean_imp(5) floor.
  (c) returns the LBR exploitability as the stability trend.
  (d) stashes ``tier_a_lbr`` on the trainer for the metrics row. The turn-1
      Cambia rate is tapped separately in the trainer's generation loop (from
      the iteration's own K games) and is already on the trainer as
      ``trainer.t1_cambia_rate`` by the time this runs; held-out critic MSE is
      emitted by the trainer directly. This function adds only ``tier_a_lbr``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prtcfr_trainer import PRTCFRProductionTrainer

logger = logging.getLogger(__name__)

# Fixed CRN seed for the fast lane. Holding the deck/opponent sampling stream
# constant across iterations makes the LBR trend reflect POLICY change, not
# resampled trajectory noise (the AC2 slope is measured over iters 100-300).
_DEFAULT_LBR_SEED = 20_240_710


def _build_upto_mixture(trainer, t: int, device: str, weighting: str, seq_cap: int):
    """Build the SD-CFR mixture over the snapshots written so far with iter <= t.

    Uses ``trainer._written_iters`` (the trainer's own record of which iters
    have snapshots on disk, ``[1..t]`` at the cadence) rather than the deployable
    manifest, so the in-loop trend spans the full grown window. Snapshots load
    lazily (cambia-249) via the mixture's per-iter loader closures.
    """
    from .prtcfr_mixture import PRTCFRMixture

    iters = sorted(i for i in getattr(trainer, "_written_iters", []) if 1 <= i <= t)
    kept, loaders = [], []
    for i in iters:
        snap = os.path.join(trainer.snapshot_dir, f"prtcfr_snapshot_iter_{i}.pt")
        if os.path.isfile(snap):
            kept.append(i)
            loaders.append(PRTCFRMixture._make_loader(snap, device))

    if loaders:
        return PRTCFRMixture(
            kept, loaders, weighting=weighting, seq_cap=seq_cap, device=device
        )

    # Defensive fallback (unreachable in normal operation: run_iteration writes
    # snapshot_iter_t.pt before this eval runs). Build the single-file mixture
    # the standard construction falls back to.
    return PRTCFRMixture.from_checkpoint(
        trainer.snapshot_path(t),
        device=device,
        weighting=weighting,
        seq_cap=seq_cap,
        max_iter=t,
    )


def build_production_battery_eval_fn(
    eval_config,
    device: str = "cpu",
    weighting: str = "linear",
    lbr_games: Optional[int] = None,
    lbr_depth: Optional[int] = None,
    lbr_seed: int = _DEFAULT_LBR_SEED,
    player_id: int = 0,
) -> Callable[["PRTCFRProductionTrainer", int], float]:
    """Build the Tier-A LBR fast-lane eval_fn for the production trainer.

    Args:
        eval_config: a Config-like exposing ``cambia_rules`` (and ``agents`` for
            the strong-opponent fallbacks). Governs the LBR games' house rules;
            the trainer's PRT-CFR config carries only the net/training knobs.
        device: device the mixture snapshots and LBR net queries run on.
        weighting: SD-CFR snapshot weighting ("linear" = ``w_t = t``, the served
            average-strategy weighting; the final eval's default).
        lbr_games: Tier-A ``num_infosets`` (P0 infosets sampled). None reads
            ``trainer.config.battery_lbr_games``.
        lbr_depth: Tier-A ``br_rollouts_per_infoset`` (BR-estimate rollouts per
            action). None reads ``trainer.config.battery_lbr_depth``.
        lbr_seed: fixed CRN seed for the fast lane (cross-iteration stability).
        player_id: seat the mixture agent plays (LBR always measures P0).

    Returns:
        ``eval_fn(trainer, t) -> float`` returning the Tier-A LBR exploitability
        of the ``[1..t]`` mixture and stashing it on ``trainer.tier_a_lbr``.
    """
    from src.cfr.prtcfr_worker import PRODUCTION_SEQ_CAP
    from src.cfr.sampled_lbr import sampled_lbr
    from src.evaluate_agents import PRTCFRAgentWrapper

    def eval_fn(trainer, t: int) -> float:
        games = int(
            lbr_games
            if lbr_games is not None
            else getattr(trainer.config, "battery_lbr_games", 64)
        )
        depth = int(
            lbr_depth
            if lbr_depth is not None
            else getattr(trainer.config, "battery_lbr_depth", 8)
        )

        # Same wrapper the final eval serves; its manifest-derived mixture is
        # immediately replaced with the [1..t] window (see _build_upto_mixture).
        wrapper = PRTCFRAgentWrapper(
            player_id=player_id,
            config=eval_config,
            checkpoint_path=trainer.snapshot_path(t),
            device=device,
            weighting=weighting,
        )
        wrapper._mixture = _build_upto_mixture(
            trainer, t, device, weighting, PRODUCTION_SEQ_CAP
        )

        result = sampled_lbr(
            wrapper,
            eval_config,
            num_infosets=games,
            br_rollouts_per_infoset=depth,
            seed=lbr_seed,
        )
        lbr = float(result.get("exploitability", 0.0))
        # (d) stash on the trainer for the metrics row (per the S2W1 interface).
        trainer.tier_a_lbr = lbr
        logger.info(
            "[prtcfr-battery] iter=%d tier_a_lbr=%.6f (games=%d depth=%d "
            "infosets_sampled=%d window=[1..%d])",
            t,
            lbr,
            games,
            depth,
            int(result.get("num_infosets_sampled", 0)),
            t,
        )
        return lbr

    return eval_fn
