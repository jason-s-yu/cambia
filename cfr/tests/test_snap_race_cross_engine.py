"""
tests/test_snap_race_cross_engine.py

Cross-engine (Go vs Python) lockstep parity for the race-ON snap model
(cambia-564, snapRace=True). Both engines are driven from the same seed with a
Go-synced XorShift64 RNG and an identical deterministic action policy that
maximizes multi-committer snap windows. Because the race winner is drawn from
the SAME continuous RNG stream at the SAME point in both engines, a crossed
multi-committer window resolves to the SAME winner with the SAME loser
penalties.

Two parity levels are asserted at every step:
  1. STATE: acting player, legal-action set, stock length, discard-top bucket,
     both hand lengths, terminal.
  2. TOKEN STREAM (cambia-564 #1): each player's Go agent token body equals the
     Python-encoded observation body byte-for-byte, across race windows -- so the
     imperfect-info commit suppression and the public race-resolution frame match
     across engines and across observers. This is the deferred token layer landing.
"""

import pytest

from src.constants import NUM_PLAYERS, ActionPassSnap
from src.config import CambiaRulesConfig
from src.encoding import action_to_index
from src.abstraction import get_card_bucket
from src.game.engine import GoXorShift64Rng
import src.sequence_encoding as se

try:
    from tests.test_cross_engine_samples import (
        _setup_python_game_matching_go,
        go_available,
        skip_if_no_go,
    )
except ImportError:  # pragma: no cover - path fallback
    from test_cross_engine_samples import (  # type: ignore
        _setup_python_game_matching_go,
        go_available,
        skip_if_no_go,
    )

from src.cfr.worker import _create_observation, _filter_observation

if go_available:
    from src.ffi.bridge import (
        GoEngine,
        GoAgentState,
        apply_games_batch,
    )

# 2P action-index landmarks.
_DRAW_STOCKPILE = 0
_DISCARD_NO_ABILITY = 3
_PASS_SNAP = 97
_SNAP_DECISION_MIN = 98  # SnapOwn(0)
_SNAP_DECISION_MAX = 109  # SnapOpponent(last)
_SNAP_MOVE_MIN = 110
_SNAP_MOVE_MAX = 145


def _race_rules() -> CambiaRulesConfig:
    r = CambiaRulesConfig()
    r.allowDrawFromDiscardPile = True
    r.allowOpponentSnapping = True
    r.max_game_turns = 46
    r.snapRace = True
    return r


def _go_legal(eng) -> set:
    mask = eng.legal_actions_mask()
    return {i for i, v in enumerate(mask) if v}


def _py_legal(pygame) -> set:
    return {action_to_index(a) for a in pygame.get_legal_actions()}


def _install_go_synced_rng(pygame, seed: int) -> None:
    rng = GoXorShift64Rng(seed)
    rng.shuffle([0] * 54)
    rng.randint(0, NUM_PLAYERS - 1)
    pygame._rng = rng


def _py_discard_top_bucket(pygame):
    top = pygame.get_discard_top()
    if top is None:
        return None
    return get_card_bucket(top).value


def _py_body(init_hand, init_peek, obs_stream, observer):
    return se.encode_observation_sequence(
        init_hand, init_peek, obs_stream, observer, seq_cap=10**9, add_bos_eos=False
    )


_SNAP_OWN_MAX = 103  # SnapOwn(0..5) = 98..103; SnapOpponent = 104..109.


def _choose_action(common, prefer_opp: bool) -> int:
    """Deterministic, snap-maximizing policy over the shared legal set. prefer_opp
    biases the snap decision toward SnapOpponent (104-109) so the winning
    opponent-snap + pending-move token path is exercised too; otherwise toward
    SnapOwn (98-103)."""
    moves = [c for c in common if _SNAP_MOVE_MIN <= c <= _SNAP_MOVE_MAX]
    if moves:
        return min(moves)
    snaps = [c for c in common if _SNAP_DECISION_MIN <= c <= _SNAP_DECISION_MAX]
    if snaps:
        if prefer_opp:
            opp = [c for c in snaps if c > _SNAP_OWN_MAX]
            if opp:
                return min(opp)
        return min(snaps)
    if _DISCARD_NO_ABILITY in common:
        return _DISCARD_NO_ABILITY
    if _DRAW_STOCKPILE in common:
        return _DRAW_STOCKPILE
    if _PASS_SNAP in common:
        return _PASS_SNAP
    return min(common)


@skip_if_no_go
@pytest.mark.parametrize("prefer_opp", [False, True])
def test_race_on_cross_engine_state_and_token_parity(prefer_opp):
    race_windows_crossed = 0
    saw_race_frame = False
    race_marker_lo = se.RACE_FRAME_BASE
    race_marker_hi = se.RACE_FRAME_BASE + se.NUM_RACE_FRAME_IDS

    for seed in range(60):
        rules = _race_rules()
        pygame = _setup_python_game_matching_go(seed)
        pygame.house_rules = rules
        _install_go_synced_rng(pygame, seed)

        eng = GoEngine(seed=seed, house_rules=rules)
        a0 = GoAgentState(eng, 0)
        a1 = GoAgentState(eng, 1)
        agents = {0: a0, 1: a1}

        init_hands = {p: list(pygame.players[p].hand) for p in range(NUM_PLAYERS)}
        init_peeks = {
            p: tuple(pygame.players[p].initial_peek_indices) for p in range(NUM_PLAYERS)
        }
        obs_streams = {p: [] for p in range(NUM_PLAYERS)}

        # Initial state: empty streams => body is exactly the init-peek prefix.
        for observer in range(NUM_PLAYERS):
            go_body = agents[observer].tokens().tolist()
            py_body = _py_body(init_hands[observer], init_peeks[observer], [], observer)
            assert go_body == py_body, (
                f"seed {seed} obs {observer}: init-peek body mismatch\n"
                f"  Go={go_body}\n  Py={py_body}"
            )

        for step in range(400):
            if eng.is_terminal() or pygame.is_terminal():
                assert eng.is_terminal() == pygame.is_terminal(), (
                    f"seed {seed} step {step}: terminal mismatch"
                )
                break

            # --- State parity (pre-action) ---
            assert eng.acting_player() == pygame.get_acting_player(), (
                f"seed {seed} step {step}: acting-player mismatch"
            )
            go_set, py_set = _go_legal(eng), _py_legal(pygame)
            assert go_set == py_set, (
                f"seed {seed} step {step}: legal-set mismatch "
                f"go_only={sorted(go_set - py_set)} py_only={sorted(py_set - go_set)}"
            )
            assert eng.stock_len() == len(pygame.stockpile), (
                f"seed {seed} step {step}: stock-len mismatch"
            )
            assert eng.discard_top() == _py_discard_top_bucket(pygame), (
                f"seed {seed} step {step}: discard-top mismatch"
            )
            assert a0.get_hand_lens() == (
                len(pygame.players[0].hand),
                len(pygame.players[1].hand),
            ), f"seed {seed} step {step}: hand-len mismatch"

            common = sorted(go_set & py_set)
            if not common:
                break
            action_idx = _choose_action(common, prefer_opp)

            # Count a genuine multi-committer race (>= 2 willing committers).
            if (
                pygame.snap_phase_active
                and len(pygame.snap_potential_snappers) >= 2
                and pygame.snap_current_snapper_idx
                == len(pygame.snap_potential_snappers) - 1
            ):
                prior_willing = sum(
                    1
                    for c in pygame.snap_commits[
                        : len(pygame.snap_potential_snappers) - 1
                    ]
                    if c is not None and not isinstance(c, ActionPassSnap)
                )
                if prior_willing + (1 if action_idx >= _SNAP_DECISION_MIN else 0) >= 2:
                    race_windows_crossed += 1

            actor = pygame.get_acting_player()
            py_action = next(
                a for a in pygame.get_legal_actions() if action_to_index(a) == action_idx
            )

            # --- Apply identically ---
            pygame.apply_action(py_action)
            apply_games_batch([eng.handle], [a0.handle], [a1.handle], [action_idx])

            # --- Build the Python observation and compare token bodies ---
            snap_results = list(getattr(pygame, "snap_results_log", []) or [])
            full_obs = _create_observation(None, py_action, pygame, actor, snap_results)
            assert full_obs is not None, f"seed {seed} step {step}: obs build failed"
            for observer in range(NUM_PLAYERS):
                obs_streams[observer].append(_filter_observation(full_obs, observer))

            for observer in range(NUM_PLAYERS):
                go_body = agents[observer].tokens().tolist()
                py_body = _py_body(
                    init_hands[observer],
                    init_peeks[observer],
                    obs_streams[observer],
                    observer,
                )
                assert go_body == py_body, (
                    f"seed {seed} obs {observer} step {step} action {action_idx}: "
                    f"token body mismatch\n  Go ({len(go_body)})={go_body}\n"
                    f"  Py ({len(py_body)})={py_body}"
                )
                if any(race_marker_lo <= t < race_marker_hi for t in go_body):
                    saw_race_frame = True

    assert race_windows_crossed > 0, "no multi-committer race window exercised"
    assert saw_race_frame, "no race-resolution frame ever emitted in the token stream"
