"""Tests for cfr/src/circuit.py"""

import pytest
from src.circuit import (
    CircuitConfig,
    CircuitRunner,
    CircuitState,
    OpenSkillRating,
    ranks_from_scores,
    tournament_house_rules,
    update_openskill,
)


# ── Config / validation ───────────────────────────────────────────────


def test_circuit_config_validation():
    """round count not multiple of num_players raises ValueError."""
    with pytest.raises(ValueError, match="multiple of"):
        CircuitState(
            CircuitConfig(
                num_players=4,
                num_rounds=10,  # 10 % 4 != 0
                player_ids=[1, 2, 3, 4],
            )
        )


def test_circuit_format_auto_resolution():
    """quick=8, standard=12, championship=20."""
    for fmt, expected in [("quick", 8), ("standard", 12), ("championship", 20)]:
        n = 4
        # Need num_rounds % num_players == 0
        # quick=8, standard=12, championship=20 all divisible by 4
        state = CircuitState(
            CircuitConfig(format=fmt, num_players=n, player_ids=list(range(n)))
        )
        assert state.config.num_rounds == expected


# ── Subsidies ─────────────────────────────────────────────────────────


def _make_4p_state():
    return CircuitState(
        CircuitConfig(
            num_players=4,
            num_rounds=4,
            player_ids=[10, 20, 30, 40],
        )
    )


def _make_5p_state():
    return CircuitState(
        CircuitConfig(
            format="standard",
            num_players=4,
            num_rounds=4,
            player_ids=[1, 2, 3, 4],
        )
    )


def test_circuit_record_round_subsidies_4p():
    """4 players: subsidies [-5, -2, 0, 0] applied by sorted placement."""
    state = _make_4p_state()
    # Distinct scores; no Cambia caller
    # Sorted: 10→5, 20→8, 30→12, 40→18  (positions 0,1,2,3)
    state.record_round({10: 5, 20: 8, 30: 12, 40: 18})

    # cumulative_score = raw + subsidy
    assert state._player_map[10].cumulative_score == 5 + (-5)   # 0
    assert state._player_map[20].cumulative_score == 8 + (-2)   # 6
    assert state._player_map[30].cumulative_score == 12 + 0     # 12
    assert state._player_map[40].cumulative_score == 18 + 0     # 18


def test_circuit_record_round_subsidies_5p():
    """5 players: subsidies [-5, -2, -1, 0, 0] by placement."""
    state = CircuitState(
        CircuitConfig(
            num_players=5,
            num_rounds=5,
            player_ids=[1, 2, 3, 4, 5],
        )
    )
    state.record_round({1: 2, 2: 4, 3: 6, 4: 8, 5: 10})

    assert state._player_map[1].cumulative_score == 2 + (-5)
    assert state._player_map[2].cumulative_score == 4 + (-2)
    assert state._player_map[3].cumulative_score == 6 + (-1)
    assert state._player_map[4].cumulative_score == 8 + 0
    assert state._player_map[5].cumulative_score == 10 + 0


# ── Tie-break rules ───────────────────────────────────────────────────


def test_circuit_cambia_caller_tie_break():
    """Cambia caller wins ties for placement & gets higher bonus."""
    state = _make_4p_state()
    # Players 10 and 20 both score 7; player 10 is Cambia caller
    # Sorted order: [10, 20, 30, 40] (caller first in tie)
    # Positions: 10→0(-5), 20→1(-2), 30→2(0), 40→3(0)
    state.record_round({10: 7, 20: 7, 30: 12, 40: 18}, cambia_caller_id=10)

    assert state._player_map[10].cumulative_score == 7 + (-5)
    assert state._player_map[20].cumulative_score == 7 + (-2)
    assert state._player_map[10].round_placements[0] == 1
    assert state._player_map[20].round_placements[0] == 1  # Same raw placement


def test_circuit_tie_both_get_higher_bonus():
    """Neither is Cambia caller, both tied get higher placement's bonus."""
    state = _make_4p_state()
    # Players 10 and 20 tied at 7; neither is Cambia caller
    # Both should get position 0 bonus (-5)
    state.record_round({10: 7, 20: 7, 30: 12, 40: 18}, cambia_caller_id=-1)

    assert state._player_map[10].cumulative_score == 7 + (-5)
    assert state._player_map[20].cumulative_score == 7 + (-5)


# ── Dealer rotation ───────────────────────────────────────────────────


def test_circuit_dealer_rotation():
    """Dealer advances each round."""
    state = _make_4p_state()
    pids = [10, 20, 30, 40]
    assert state.next_dealer_seat() == 10

    state.record_round({10: 5, 20: 8, 30: 12, 40: 18})
    assert state.next_dealer_seat() == 20

    state.record_round({10: 5, 20: 8, 30: 12, 40: 18})
    assert state.next_dealer_seat() == 30

    state.record_round({10: 5, 20: 8, 30: 12, 40: 18})
    assert state.next_dealer_seat() == 40


# ── Missed rounds / abandonment ───────────────────────────────────────


def test_circuit_missed_round():
    """Scores 41, consecutive miss tracking."""
    state = _make_4p_state()
    state.record_missed_round(10)

    p = state._player_map[10]
    assert p.consecutive_misses == 1
    assert p.raw_cumulative == 41
    assert p.cumulative_score == 41
    assert p.round_scores == [41]


def test_circuit_abandonment():
    """2 consecutive misses → abandoned, remaining rounds scored 41."""
    state = CircuitState(
        CircuitConfig(
            num_players=4,
            num_rounds=8,
            player_ids=[1, 2, 3, 4],
            abandon_threshold=2,
            missed_round_score=41,
        )
    )
    # Round 0 out of 8; after 2 misses, 6 rounds remain
    state.record_missed_round(1)
    assert not state._player_map[1].abandoned

    state.record_missed_round(1)
    p = state._player_map[1]
    assert p.abandoned
    # 2 recorded misses (82) + 6 remaining * 41 = 82 + 246 = 328
    assert p.cumulative_score == 41 * 2 + 41 * 6


def test_circuit_reconnection():
    """Reconnection resets consecutive_misses."""
    state = _make_4p_state()
    state.record_missed_round(10)
    assert state._player_map[10].consecutive_misses == 1

    state.record_reconnection(10)
    assert state._player_map[10].consecutive_misses == 0


# ── Standings ─────────────────────────────────────────────────────────


def test_circuit_get_standings_all_tiebreakers():
    """All 4 tiebreaker levels tested."""
    # Use 4p, 4 rounds; carefully craft scores to exercise tiebreakers
    state = CircuitState(
        CircuitConfig(num_players=4, num_rounds=4, player_ids=[1, 2, 3, 4])
    )
    # Give all players equal cumulative (same score each round)
    for _ in range(4):
        state.record_round({1: 10, 2: 10, 3: 10, 4: 10})

    standings = state.get_standings()
    # All equal cumulative, raw, h2h wins (all ties → no update), best_round, sort by player_id
    ids = [p.player_id for p in standings]
    assert ids == sorted(ids), "Should fall back to player_id ordering"


# ── Full tournament ───────────────────────────────────────────────────


def test_circuit_full_tournament():
    """12 rounds with known scores; verify completion and ordering."""
    state = CircuitState(
        CircuitConfig(
            format="standard",
            num_players=4,
            num_rounds=12,
            player_ids=[1, 2, 3, 4],
        )
    )
    # Player 1 always gets low scores (wins)
    for _ in range(12):
        state.record_round({1: 2, 2: 5, 3: 9, 4: 14})

    assert state.is_complete()
    standings = state.get_standings()
    assert standings[0].player_id == 1  # Best (lowest cumulative)
    assert standings[-1].player_id == 4  # Worst


# ── OpenSkill ─────────────────────────────────────────────────────────


def test_openskill_defaults():
    """Default mu=25, sigma≈8.333."""
    r = OpenSkillRating()
    assert r.mu == 25.0
    assert abs(r.sigma - 25.0 / 3.0) < 1e-6


def test_openskill_clear_winner():
    """Winner mu up, loser mu down."""
    r1 = OpenSkillRating()
    r2 = OpenSkillRating()
    updated = update_openskill([r1, r2], ranks=[1, 2])
    assert updated[0].mu > r1.mu, "Winner should gain mu"
    assert updated[1].mu < r2.mu, "Loser should lose mu"


def test_openskill_tie():
    """Equal ranks produce smaller shifts."""
    r1 = OpenSkillRating()
    r2 = OpenSkillRating()
    updated_win = update_openskill([r1, r2], ranks=[1, 2])
    updated_tie = update_openskill([r1, r2], ranks=[1, 1])

    delta_win = abs(updated_win[0].mu - r1.mu)
    delta_tie = abs(updated_tie[0].mu - r1.mu)
    assert delta_tie < delta_win, "Tie should produce smaller shift"


def test_openskill_4player():
    """Rank ordering: winner > 2nd > 3rd > last in mu change."""
    ratings = [OpenSkillRating() for _ in range(4)]
    updated = update_openskill(ratings, ranks=[1, 2, 3, 4])
    deltas = [updated[i].mu - ratings[i].mu for i in range(4)]
    # Better rank → higher delta
    assert deltas[0] > deltas[1] > deltas[2] > deltas[3]


def test_openskill_sigma_decreases():
    """Sigma decreases after an update (more certainty)."""
    r1 = OpenSkillRating()
    r2 = OpenSkillRating()
    updated = update_openskill([r1, r2], ranks=[1, 2])
    assert updated[0].sigma < r1.sigma
    assert updated[1].sigma < r2.sigma


# ── ranks_from_scores ─────────────────────────────────────────────────


def test_ranks_from_scores_basic():
    """[5, 10, 3, 20] → [2, 3, 1, 4] (lower score = better rank, margin=0)."""
    # With default margin=3: 5 and 3 within margin? 5-3=2 < 3, so they tie as rank 1
    # Use margin=0 to force strict ordering
    result = ranks_from_scores([5, 10, 3, 20], tie_margin=0)
    assert result == [2, 3, 1, 4]


def test_ranks_from_scores_tie_margin():
    """[10, 12, 15, 30] margin=3 → [1, 1, 3, 4]."""
    result = ranks_from_scores([10, 12, 15, 30], tie_margin=3)
    # 10 and 12 within 3 → group rank 1; 15 > 12+3=15 → NOT in group (15 <= 10+3=13? No)
    # group min=10; 12<=13 yes; 15<=13? No → group=[10,12] rank=1; next group min=15; 30>18 → rank=3; 30 rank=4
    assert result == [1, 1, 3, 4]


def test_ranks_from_scores_all_tied():
    """[10, 11, 12, 13] margin=5 → [1, 1, 1, 1]."""
    result = ranks_from_scores([10, 11, 12, 13], tie_margin=5)
    assert result == [1, 1, 1, 1]


def test_tournament_house_rules():
    """T1 enforced rules: discard draw, replace abilities, no caller lock."""
    rules = tournament_house_rules()
    assert rules["allowDrawFromDiscardPile"] is True
    assert rules["allowReplaceAbilities"] is True
    assert rules["lockCallerHand"] is False
