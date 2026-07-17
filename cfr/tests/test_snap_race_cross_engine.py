"""
tests/test_snap_race_cross_engine.py

Cross-engine (Go vs Python) lockstep parity for the race-ON snap model
(cambia-564, snapRace=True). Both engines are driven from the same seed with a
Go-synced XorShift64 RNG and an identical deterministic action policy that
maximizes multi-committer snap windows. Because the race winner is drawn from
the SAME continuous RNG stream at the SAME point in both engines, a crossed
multi-committer window resolves to the SAME winner with the SAME loser
penalties; the test asserts full state parity (acting player, legal-action set,
stock length, discard top, both hand lengths, terminal) at every step and that
at least one winner+loser race window is actually exercised.

Token-stream parity is intentionally NOT asserted here: the imperfect-info
commit-suppression + race-resolution observation frame is the deferred follow-up
(cambia-564 #1, sequenced after the tokenizer merge). This test is state-level.
"""

import pytest

from src.constants import NUM_PLAYERS, ActionPassSnap
from src.config import CambiaRulesConfig
from src.encoding import action_to_index
from src.abstraction import get_card_bucket
from src.game.engine import GoXorShift64Rng

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
    """Pre-advance a GoXorShift64Rng to Go's post-Deal() state (53-card shuffle +
    starting-player pick), then install it so all later draws (reshuffles and the
    race winner draw) share Go's continuing stream."""
    rng = GoXorShift64Rng(seed)
    rng.shuffle([0] * 54)
    rng.randint(0, NUM_PLAYERS - 1)
    pygame._rng = rng


def _py_discard_top_bucket(pygame):
    """Map the Python discard top to the same card bucket Go's discard_top()
    reports (Go returns the bucket index, not a raw card index)."""
    top = pygame.get_discard_top()
    if top is None:
        return None
    return get_card_bucket(top).value


def _choose_action(common) -> int:
    """Deterministic, snap-maximizing policy over the shared legal set: resolve
    any pending snap move, then commit real snaps, then progress the game via
    draw/discard. Identical inputs -> identical action in both engines."""
    moves = [c for c in common if _SNAP_MOVE_MIN <= c <= _SNAP_MOVE_MAX]
    if moves:
        return min(moves)
    snaps = [c for c in common if _SNAP_DECISION_MIN <= c <= _SNAP_DECISION_MAX]
    if snaps:
        return min(snaps)
    if _DISCARD_NO_ABILITY in common:
        return _DISCARD_NO_ABILITY
    if _DRAW_STOCKPILE in common:
        return _DRAW_STOCKPILE
    if _PASS_SNAP in common:
        return _PASS_SNAP
    return min(common)


@skip_if_no_go
def test_race_on_cross_engine_state_parity():
    race_windows_crossed = 0

    for seed in range(60):
        rules = _race_rules()
        pygame = _setup_python_game_matching_go(seed)
        pygame.house_rules = rules
        _install_go_synced_rng(pygame, seed)

        eng = GoEngine(seed=seed, house_rules=rules)
        a0 = GoAgentState(eng, 0)
        a1 = GoAgentState(eng, 1)

        for step in range(400):
            gt, pt = eng.is_terminal(), pygame.is_terminal()
            assert gt == pt, f"seed {seed} step {step}: terminal mismatch Go={gt} Py={pt}"
            if gt:
                break

            # --- State parity assertions (pre-action) ---
            assert eng.acting_player() == pygame.get_acting_player(), (
                f"seed {seed} step {step}: acting-player mismatch"
            )
            go_set, py_set = _go_legal(eng), _py_legal(pygame)
            assert go_set == py_set, (
                f"seed {seed} step {step}: legal-set mismatch "
                f"go_only={sorted(go_set - py_set)} py_only={sorted(py_set - go_set)}"
            )
            assert eng.stock_len() == len(pygame.stockpile), (
                f"seed {seed} step {step}: stock-len mismatch "
                f"Go={eng.stock_len()} Py={len(pygame.stockpile)}"
            )
            assert eng.discard_top() == _py_discard_top_bucket(pygame), (
                f"seed {seed} step {step}: discard-top bucket mismatch "
                f"Go={eng.discard_top()} Py={_py_discard_top_bucket(pygame)}"
            )
            go_lens = a0.get_hand_lens()  # (P0, P1) absolute for observer 0
            py_lens = (len(pygame.players[0].hand), len(pygame.players[1].hand))
            assert go_lens == py_lens, (
                f"seed {seed} step {step}: hand-len mismatch Go={go_lens} Py={py_lens}"
            )

            common = sorted(go_set & py_set)
            if not common:
                break
            action_idx = _choose_action(common)

            # Count a genuine multi-committer race: the last committer of a window
            # with >= 2 potential snappers, where >= 2 committers are willing (a
            # winner plus at least one penalized loser).
            if (
                pygame.snap_phase_active
                and len(pygame.snap_potential_snappers) >= 2
                and pygame.snap_current_snapper_idx == len(pygame.snap_potential_snappers) - 1
            ):
                prior_willing = sum(
                    1
                    for c in pygame.snap_commits[: len(pygame.snap_potential_snappers) - 1]
                    if c is not None and not isinstance(c, ActionPassSnap)
                )
                cur_willing = 1 if action_idx >= _SNAP_DECISION_MIN else 0
                if prior_willing + cur_willing >= 2:
                    race_windows_crossed += 1

            # --- Apply identically to both engines ---
            py_action = next(
                a for a in pygame.get_legal_actions() if action_to_index(a) == action_idx
            )
            pygame.apply_action(py_action)
            apply_games_batch([eng.handle], [a0.handle], [a1.handle], [action_idx])

    assert race_windows_crossed > 0, (
        "no multi-committer race window was exercised across the seed set; "
        "the test is not covering the race resolution"
    )
