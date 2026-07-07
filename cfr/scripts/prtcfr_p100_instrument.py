"""
cfr/scripts/prtcfr_p100_instrument.py

P100 (and P50/P90/P99/mean) instrumentation for PRT-CFR's production sequence
cap. Pins the value the v0.4 Phase 2 window-semantics decision note recommends
(SEQ_CAP raised so keep-most-recent truncation structurally never fires) and
gives S1W4 (disk reservoir) its per-row allocation bound.

Method: play full 2-player games under the PRODUCTION rule profile
(config/rule_profiles/competitive.yaml: allowDrawFromDiscardPile=True,
allowOpponentSnapping=True, max_game_turns=300 -- the real engine turn cap, the
structural ceiling on how long a game, and therefore a token stream, can run),
using a uniform-random legal-action policy (the same b_i the ESCHER sampler's
traverser actually uses, and a reasonable proxy for an early/untrained sigma^t).
Two cohorts:

  natural     : normal random play, including CallCambia whenever legal. This is
                the realistic mixture (most games end promptly).
  avoid_cambia: skip CallCambia whenever a non-Cambia legal action exists, so
                games run to their natural non-Cambia-terminated length -- the
                honest worst case bounded only by the engine's own turn cap.
                Mirrors tests/test_sequence_tokenizer.py's established stress
                convention.

For each game, the RAW (untruncated, seq_cap=10**9, strict=False since we WANT
the true unbounded length here, not an exception) token length is computed per
player via encode_observation_sequence -- exactly the function production call
sites use with strict=True at the recommended cap.

Run (CPU-only, no torch needed):
  cd cfr && python3 scripts/prtcfr_p100_instrument.py --n-games 3000
  cd cfr && python3 scripts/prtcfr_p100_instrument.py --n-games 3000 --avoid-cambia-only
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CambiaRulesConfig, Config  # noqa: E402
from src.constants import NUM_PLAYERS, ActionCallCambia  # noqa: E402
import src.sequence_encoding as se  # noqa: E402
from src.cfr.prtcfr_worker import (  # noqa: E402
    PythonEngineGameDriver,
    _legal_mask,
    _sample_and_apply,
    uniform_policy_production,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from tests.test_cross_engine_samples import _setup_python_game_matching_go
except ImportError:  # pragma: no cover
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
    )
    from test_cross_engine_samples import _setup_python_game_matching_go  # type: ignore


def _production_rule_config() -> Config:
    """Production rule profile: config/rule_profiles/competitive.yaml values.

    Matches the Go engine defaults (memory: AllowDrawFromDiscard=true,
    AllowOpponentSnapping=true) and the real production turn cap (300), not the
    46-turn value some unit-test helpers use for faster fixtures.
    """
    cfg = Config()
    cfg.cambia_rules = CambiaRulesConfig()
    cfg.cambia_rules.allowDrawFromDiscardPile = True
    cfg.cambia_rules.allowReplaceAbilities = True
    cfg.cambia_rules.allowOpponentSnapping = True
    cfg.cambia_rules.max_game_turns = 300
    cfg.cambia_rules.lockCallerHand = False
    return cfg


def _play_one_game(seed: int, avoid_cambia: bool, max_steps: int) -> List[int]:
    """Play one full 2P game under the production rule profile; return the RAW
    (untruncated) per-player token-sequence lengths (len == NUM_PLAYERS).

    Drives via PythonEngineGameDriver + _sample_and_apply (the same production
    driver/retry path prtcfr_worker.py uses) rather than a raw apply_action
    loop, so a nominally-"legal" action the engine rejects for the current
    pending sub-decision (a pre-existing engine/ability-pending-chain
    interaction; see PythonEngineGameDriver.apply's docstring) is resampled
    instead of silently recorded as a phantom event that never happened --
    otherwise this instrumentation would measure inflated/corrupted lengths.
    """
    game = _setup_python_game_matching_go(seed)
    game.house_rules = _production_rule_config().cambia_rules
    rng = random.Random(seed)

    init_hands = {p: list(game.players[p].hand) for p in range(NUM_PLAYERS)}
    init_peeks = {
        p: tuple(game.players[p].initial_peek_indices) for p in range(NUM_PLAYERS)
    }
    driver = PythonEngineGameDriver(game, init_hands, init_peeks, seq_cap=10**9)

    for _ in range(max_steps):
        if driver.is_terminal():
            break
        actor = driver.current_player()
        if actor == -1:
            break
        legal = driver.legal_actions()
        if not legal:
            break
        pool = legal
        if avoid_cambia:
            non_cambia = [a for a in legal if not isinstance(a, ActionCallCambia)]
            if non_cambia:
                pool = non_cambia
        mask = _legal_mask(pool)
        probs = uniform_policy_production([], mask)
        try:
            _sample_and_apply(driver, pool, probs, rng)
        except RuntimeError:
            break  # true engine stall (DriverStuckError); stop this game

    lengths = []
    for observer in range(NUM_PLAYERS):
        seq = se.encode_observation_sequence(
            init_hands[observer],
            init_peeks[observer],
            driver.obs_streams[observer],
            observer,
            seq_cap=10**9,
            strict=False,
        )
        lengths.append(len(seq))
    return lengths


def _percentiles(xs: List[int]) -> dict:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> int:
        if n == 0:
            return 0
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        return xs_sorted[idx]

    return {
        "n": n,
        "mean": statistics.mean(xs_sorted) if xs_sorted else 0.0,
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "p100_max": xs_sorted[-1] if xs_sorted else 0,
    }


def run(n_games: int, seed0: int, avoid_cambia_only: bool, max_steps: int) -> dict:
    cohorts = (
        {"avoid_cambia": True}
        if avoid_cambia_only
        else {
            "natural": False,
            "avoid_cambia": True,
        }
    )
    report = {}
    t0 = time.time()
    for name, avoid in cohorts.items():
        lengths: List[int] = []
        for i in range(n_games):
            lengths.extend(_play_one_game(seed0 + i, avoid, max_steps))
        report[name] = _percentiles(lengths)
    report["_wall_s"] = time.time() - t0
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--avoid-cambia-only", action="store_true")
    args = ap.parse_args()

    report = run(args.n_games, args.seed0, args.avoid_cambia_only, args.max_steps)
    wall = report.pop("_wall_s")
    for cohort, stats in report.items():
        print(
            f"[{cohort}] n={stats['n']} mean={stats['mean']:.1f} "
            f"p50={stats['p50']} p90={stats['p90']} p99={stats['p99']} "
            f"p100_max={stats['p100_max']}"
        )
    worst_p100 = max(stats["p100_max"] for stats in report.values())
    print(f"\nwall clock: {wall:.1f}s")
    print(f"worst-cohort P100 (max observed): {worst_p100}")
    print(
        "recommended production seq_cap: raise above this max with margin "
        f"(e.g. next of {{1536, 2048, 3072}} clearing {worst_p100})"
    )


if __name__ == "__main__":
    main()
