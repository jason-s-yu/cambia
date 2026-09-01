"""
tests/test_cfr_agent_wrapper_go_infoset_key.py

cambia-1488 part 1: the tabular CFRAgentWrapper (src/evaluate_agents.py) is documented as the one
wrapper the Go evaluation loop cannot drive, because its infoset key needs GamePhase -- a function
of who called Cambia -- and the FFI exported no cambia-caller accessor. Every other component of
the key (own-hand buckets, opponent belief, hand lengths, discard-top bucket, stockpile estimate)
was already reachable through GoEngine and GoAgentState; see CFRAgentWrapper's docstring.

cambia_game_cambia_caller (GoEngine.cambia_caller(), added alongside this test) closes that one
gap. This module proves it: for a handful of games driven entirely by GoEngine/GoAgentState, it
builds the exact 6-component tuple shape AgentState.get_infoset_key() returns --

    (own_hand_tuple, opp_belief_tuple, opp_count, discard_top_val, stockpile_est_val, game_phase_val)

-- using only Go-reachable state, reusing the production AgentState._estimate_stockpile /
_estimate_game_phase methods (fed Go-native scalars: stock_len(), cambia_caller(), turn_number())
rather than reimplementing that logic. Some games explicitly call Cambia partway through so both
the CAMBIA_CALLED and not-yet-called branches of GamePhase are exercised.

Scope note: this does NOT wire CFRAgentWrapper into evaluate_agents.py's `_GoEvalGame` -- that
class still lists CFRAgentWrapper in `_PYTHON_ONLY_WRAPPER_TYPES` and refuses it, because
CFRAgentWrapper's `initialize_state`/`update_state` methods are written against the Python
reference engine's observation objects, which is a separate, considerably larger integration than
the single accessor this ticket scopes. What this test proves is the thing CFRAgentWrapper's own
docstring names as the actual blocker: the infoset key's components all build off Go-only state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        engine = GoEngine.from_deck(list(range(54)))
        engine.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")

if go_available:
    from src.agent_state import AgentState
    from src.config import CambiaRulesConfig, Config, DeepCfrConfig
    from src.constants import GamePhase
    from src.evaluate_agents import CFRAgentWrapper, _PYTHON_ONLY_WRAPPER_TYPES
    from src.ffi.bridge import GoAgentState, GoEngine

ACTION_CALL_CAMBIA = 2  # engine/types.go ActionCallCambia (146- and 452-action spaces agree)


def _phase_estimator() -> "AgentState":
    """A lightweight AgentState used only for its _estimate_stockpile/_estimate_game_phase
    methods -- the production GamePhase logic, reused rather than reimplemented. Neither method
    reads anything beyond self.config, so no .initialize() call is needed."""
    cfg = Config(cambia_rules=CambiaRulesConfig(), deep_cfr=DeepCfrConfig())
    return AgentState(
        player_id=0,
        opponent_id=1,
        memory_level=0,
        time_decay_turns=0,
        initial_hand_size=4,
        config=cfg,
    )


def _build_infoset_key_tuple(
    engine: "GoEngine", agent: "GoAgentState", estimator: "AgentState"
) -> Tuple:
    """Builds the AgentState.get_infoset_key()-shaped 6-tuple entirely from GoEngine/GoAgentState
    accessors. own_hand_tuple and opp_belief_tuple use Go's own bucket encoding (0..9, 9=Unknown)
    rather than being converted to Python's CardBucket domain -- that conversion is not what this
    ticket is about, and every other component here was already reachable pre-cambia-1488, per
    CFRAgentWrapper's docstring. The point is game_phase_val, which is the one component that
    could not be computed before cambia_caller() existed."""
    own_hand_len, opp_hand_len = agent.get_hand_lens()
    own_hand_raw = agent.get_own_hand_buckets_and_seen()  # (MAX_HAND_SIZE, 3): bucket,seen,valid
    own_hand_tuple = tuple(int(own_hand_raw[i, 0]) for i in range(own_hand_len))

    opp_belief_raw = agent.get_opp_belief_buckets()  # (MAX_HAND_SIZE,) uint8
    opp_belief_tuple = tuple(int(opp_belief_raw[i]) for i in range(opp_hand_len))

    discard_top_val = engine.discard_top()
    stockpile_est_val = estimator._estimate_stockpile(engine.stock_len())
    cambia_caller = engine.cambia_caller()  # the cambia-1488 export
    game_phase_val = estimator._estimate_game_phase(
        engine.stock_len(), cambia_caller, engine.turn_number()
    )

    return (
        own_hand_tuple,
        opp_belief_tuple,
        opp_hand_len,
        discard_top_val,
        stockpile_est_val,
        game_phase_val,
    )


@skip_if_no_go
def test_cfr_agent_wrapper_is_still_documented_python_only():
    """Guards the scope note above: this test does not change _GoEvalGame's refusal."""
    assert CFRAgentWrapper in _PYTHON_ONLY_WRAPPER_TYPES


@skip_if_no_go
def test_infoset_key_builds_off_go_only_state_across_several_games():
    """Drives a handful of games entirely through GoEngine/GoAgentState, forcing Cambia calls in
    some of them, and builds the infoset key tuple at every decision point. Before cambia-1488,
    game_phase_val could not be computed on this backend at all once Cambia was called (no
    cambia-caller accessor existed); this proves it now can, for both branches."""
    hr = CambiaRulesConfig()
    hr.cambia_allowed_round = 0
    hr.allowDrawFromDiscardPile = True
    hr.allowOpponentSnapping = True

    saw_cambia_called_phase = False
    saw_not_called_phase = False
    tuples_built = 0

    for seed in range(6):
        force_cambia_at_step = 3 if seed % 2 == 0 else None
        with GoEngine(seed=seed, house_rules=hr) as engine:
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            estimator = _phase_estimator()
            try:
                step = 0
                while not engine.is_terminal() and step < 60:
                    mask = engine.legal_actions_mask()
                    legal = np.where(mask > 0)[0]
                    assert len(legal) > 0, f"seed {seed} step {step}: empty legal mask"

                    if (
                        force_cambia_at_step is not None
                        and step == force_cambia_at_step
                        and ACTION_CALL_CAMBIA in legal
                    ):
                        action = ACTION_CALL_CAMBIA
                    else:
                        action = int(np.random.default_rng(seed * 1000 + step).choice(legal))

                    engine.apply_action(action)
                    for a in agents:
                        a.update(engine)

                    key = _build_infoset_key_tuple(engine, agents[engine.acting_player()], estimator)
                    tuples_built += 1
                    assert isinstance(key, tuple) and len(key) == 6
                    assert isinstance(key[0], tuple)  # own_hand_tuple
                    assert isinstance(key[1], tuple)  # opp_belief_tuple
                    assert isinstance(key[2], int)  # opp_count
                    assert key[5] in GamePhase  # game_phase_val

                    if key[5] == GamePhase.CAMBIA_CALLED:
                        saw_cambia_called_phase = True
                    else:
                        saw_not_called_phase = True

                    step += 1
            finally:
                for a in agents:
                    a.close()

    assert tuples_built > 0, "no infoset-key tuples were built; the game loop never advanced"
    assert saw_not_called_phase, "never observed a pre-Cambia GamePhase across any game"
    assert saw_cambia_called_phase, (
        "never observed GamePhase.CAMBIA_CALLED despite forcing ActionCallCambia; "
        "cambia_caller() is not feeding _estimate_game_phase correctly"
    )


@skip_if_no_go
def test_game_phase_is_cambia_called_immediately_after_the_call():
    """Narrower, single-game version pinning the exact transition: game_phase_val flips to
    CAMBIA_CALLED on the very next infoset key built after ActionCallCambia, and cambia_caller()
    itself reports the calling seat throughout the rest of that game."""
    hr = CambiaRulesConfig()
    hr.cambia_allowed_round = 0

    with GoEngine(seed=42, house_rules=hr) as engine:
        estimator = _phase_estimator()
        assert engine.cambia_caller() is None
        before = estimator._estimate_game_phase(engine.stock_len(), None, engine.turn_number())
        assert before != GamePhase.CAMBIA_CALLED

        acting = engine.acting_player()
        mask = engine.legal_actions_mask()
        assert mask[ACTION_CALL_CAMBIA] == 1, "ActionCallCambia is not legal at turn 0"
        engine.apply_action(ACTION_CALL_CAMBIA)

        caller = engine.cambia_caller()
        assert caller == acting
        after = estimator._estimate_game_phase(engine.stock_len(), caller, engine.turn_number())
        assert after == GamePhase.CAMBIA_CALLED

        # Stays CAMBIA_CALLED regardless of stockpile as the game continues.
        for _ in range(5):
            if engine.is_terminal():
                break
            mask = engine.legal_actions_mask()
            legal = np.where(mask > 0)[0]
            engine.apply_action(int(legal[0]))
            caller = engine.cambia_caller()
            assert caller is not None
            phase = estimator._estimate_game_phase(
                engine.stock_len(), caller, engine.turn_number()
            )
            assert phase == GamePhase.CAMBIA_CALLED
