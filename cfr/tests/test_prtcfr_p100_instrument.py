"""tests/test_prtcfr_p100_instrument.py

Scoped smoke tests for scripts/prtcfr_p100_instrument.py -- the P100
instrumentation that pins PRODUCTION_SEQ_CAP (v0.4 Phase 2 window-semantics
decision, S1W3 stage 2). Fast, small-N checks only; the real calibration run
(3000+ games) is a one-off operator invocation, not part of the scoped suite
(see the module docstring for the actual run command and the numbers that
pinned PRODUCTION_SEQ_CAP=8192 in src/cfr/prtcfr_worker.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.prtcfr_p100_instrument import (  # noqa: E402
    _percentiles,
    _play_one_game,
    run,
)


def test_percentiles_basic():
    stats = _percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert stats["n"] == 10
    assert stats["p50"] in (5, 6)
    assert stats["p100_max"] == 10
    assert stats["mean"] == 5.5


def test_percentiles_empty():
    stats = _percentiles([])
    assert stats["n"] == 0
    assert stats["p100_max"] == 0


def test_play_one_game_natural_returns_two_lengths():
    lengths = _play_one_game(seed=0, avoid_cambia=False, max_steps=500)
    assert len(lengths) == 2  # NUM_PLAYERS
    assert all(isinstance(x, int) and x > 0 for x in lengths)


def test_play_one_game_avoid_cambia_runs_longer_than_natural_on_average():
    """avoid_cambia games are structurally bounded only by the 300-turn engine
    cap and should, on average over a handful of seeds, produce longer raw
    token sequences than natural play."""
    natural_lengths = []
    avoid_lengths = []
    for seed in range(6):
        natural_lengths.extend(
            _play_one_game(seed=100 + seed, avoid_cambia=False, max_steps=500)
        )
        avoid_lengths.extend(
            _play_one_game(seed=100 + seed, avoid_cambia=True, max_steps=2000)
        )
    assert sum(avoid_lengths) / len(avoid_lengths) > sum(natural_lengths) / len(
        natural_lengths
    )


def test_run_small_n_end_to_end():
    report = run(n_games=3, seed0=999_000, avoid_cambia_only=False, max_steps=500)
    assert "_wall_s" in report
    assert "natural" in report and "avoid_cambia" in report
    for cohort in ("natural", "avoid_cambia"):
        stats = report[cohort]
        assert stats["n"] == 6  # 3 games x 2 players
        assert stats["p100_max"] >= stats["p50"] >= 0
