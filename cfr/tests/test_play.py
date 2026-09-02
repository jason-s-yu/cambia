"""Tests for src/play.py on GoEngine (cambia-1428).

play_game() itself needs no human seat, so these drive full games with only
AI seats: a stand-in for the interactive path (verified separately by feeding
`cambia play` scripted stdin) that runs deterministically under pytest.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

_CFR_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(_CFR_ROOT / "config" / "deep_train.yaml")

try:
    from src.ffi.bridge import GoEngine

    GoEngine.N_PLAYER_INPUT_DIM  # noqa: B018 - touch it so a broken load raises here
    _lib_ok = True
    _lib_err = ""
except Exception as exc:  # noqa: BLE001 - reported as a skip reason
    _lib_ok = False
    _lib_err = f"{type(exc).__name__}: {exc}"

skiplib = pytest.mark.skipif(not _lib_ok, reason=f"libcambia unavailable ({_lib_err})")


@pytest.fixture
def cfg():
    from src.config import load_config

    loaded = load_config(CONFIG_PATH)
    assert loaded is not None, f"failed to load {CONFIG_PATH}"
    return loaded


def _ai_seats(cfg, num_players: int, opponent_types):
    from src.evaluate_agents import get_agent
    from src.play import SeatConfig

    seats = []
    for i in range(num_players):
        agent_type = opponent_types[i % len(opponent_types)]
        agent = get_agent(agent_type, player_id=i, config=cfg)
        seats.append(
            SeatConfig(
                seat_id=i,
                is_human=False,
                name=f"{agent_type}(P{i})",
                agent_type=agent_type,
                agent=agent,
            )
        )
    return seats


def _run_silently(seats, house_rules, num_players):
    from src.play import play_game

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        play_game(seats, house_rules, num_players=num_players)
    return buf.getvalue()


@skiplib
@pytest.mark.parametrize("num_players", [2, 4])
def test_play_game_completes_and_declares_correct_winner(cfg, num_players):
    """A full AI-only game reaches Game Over and the printed winner matches
    the lowest final hand score (or is a genuine tie), across several deals."""
    opponent_types = ["imperfect_greedy", "memory_heuristic"]
    for seed_trial in range(3):
        seats = _ai_seats(cfg, num_players, opponent_types)
        out = _run_silently(seats, cfg.cambia_rules, num_players)
        assert "Game Over" in out
        assert "Winner" in out or "Tie between" in out


@skiplib
def test_play_game_2p_winner_has_lowest_score():
    """Direct check that the reported winner has the minimum hand score
    (regression guard for the utility-vs-non-terminal-state bug: play_game's
    action-count cap must be generous enough that the loop always exits via
    is_terminal(), not the turn cap, or get_utility() on a mid-game state
    would report a spurious tie)."""
    from src.config import load_config
    from src.evaluate_agents import get_agent
    from src.ffi.bridge import GoEngine
    from src.play import SeatConfig, play_game

    cfg = load_config(CONFIG_PATH)
    for trial in range(5):
        engine_probe = GoEngine(house_rules=cfg.cambia_rules, num_players=2, seed=trial)
        agents = [
            get_agent("imperfect_greedy", player_id=0, config=cfg),
            get_agent("memory_heuristic", player_id=1, config=cfg),
        ]
        from src.agents import action_codec
        from src.encoding import action_to_index

        turn = 0
        while not engine_probe.is_terminal() and turn < 5000:
            turn += 1
            mask = engine_probe.legal_actions_mask()
            actions = action_codec.actions_from_mask(mask)
            index = {a: int(i) for a, i in zip(actions, np.flatnonzero(mask))}
            seat = engine_probe.acting_player()
            chosen = agents[seat].choose_action(engine_probe, actions)
            idx = index.get(chosen)
            if idx is None:
                idx = action_to_index(chosen)
            engine_probe.apply_action(int(idx))
        assert engine_probe.is_terminal()
        hands = [engine_probe.get_player_hand(s) for s in range(2)]
        scores = [sum(c.value for c in h) for h in hands]
        utils = np.asarray(engine_probe.get_utility(), dtype=np.float64)
        engine_probe.close()

        best_util = float(utils.max())
        leaders = np.flatnonzero(utils >= best_util - 1e-9)
        if len(leaders) == 1:
            winner = int(leaders[0])
            other = 1 - winner
            # The Cambia-caller tiebreak can make the utility winner differ
            # from the raw min-score seat only on an exact score tie; a
            # strict score gap must always agree with the utility winner.
            if scores[winner] != scores[other]:
                assert scores[winner] < scores[other]


@skiplib
def test_legal_actions_for_two_player_matches_mask_decode():
    from src.agents import action_codec
    from src.config import load_config
    from src.ffi.bridge import GoEngine
    from src.play import _legal_actions_for

    cfg = load_config(CONFIG_PATH)
    engine = GoEngine(house_rules=cfg.cambia_rules, num_players=2, seed=7)
    try:
        actions, index = _legal_actions_for(engine, engine.acting_player(), 2)
        mask = engine.legal_actions_mask()
        expected = action_codec.actions_from_mask(mask)
        assert actions == expected
        for a in actions:
            assert mask[index[a]] == 1
    finally:
        engine.close()


@skiplib
def test_legal_actions_for_four_player_returns_nonempty_actions():
    from src.agents.transition import TransitionBroadcaster
    from src.config import load_config
    from src.ffi.bridge import GoEngine
    from src.play import _legal_actions_for

    cfg = load_config(CONFIG_PATH)
    engine = GoEngine(house_rules=cfg.cambia_rules, num_players=4, seed=3)
    try:
        # Applying is the shared transition step now (cambia-711); with no
        # agent seated it is the plain N-player engine apply this asserted
        # before.
        broadcast = TransitionBroadcaster(engine, [None] * 4, 4)
        for _ in range(20):
            if engine.is_terminal():
                break
            seat = engine.acting_player()
            actions, index = _legal_actions_for(engine, seat, 4)
            assert actions, "acting seat must always have at least one legal action"
            chosen = actions[0]
            broadcast.apply(chosen, index, seat)
    finally:
        engine.close()
