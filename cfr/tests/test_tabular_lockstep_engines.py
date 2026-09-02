"""
tests/test_tabular_lockstep_engines.py

Lockstep engine equivalence over the tabular traversal's own apply/undo
sequence (cambia-1782).

``src.cfr.worker``'s outcome-sampling traversal used to drive a Python
``CambiaGameState``, applying a sampled action and calling the undo callable
that came back. It now drives ``src.cfr.br_state.GoBrState``, applying an
action index and rewinding to a checkpoint. The two are only interchangeable if
the Go engine reaches the same state as the Python engine after every one of
those operations -- not just at the leaves, and not just on the way down.

So this records the exact sequence a real traversal performs (via
``run_cfr_simulation_worker`` on a pinned deal, so the sequence is the
production traversal's and not a walk invented here), replays it on a Python
``CambiaGameState`` dealt from the same deck, and compares the full observable
state after every apply and every rewind: both hands, the pile sizes, the
discard top, the acting seat, the decision context, the repr-sorted legal
action list, the turn number, the Cambia caller and the terminal utilities.

The rewind side is the half that a same-state-at-the-leaf comparison would
miss: a checkpoint that restored a subtly different state would leave the rest
of the traversal sampling from the wrong node, and every later infoset key would
be written against it.
"""

import logging
import random
import tempfile
from typing import Any, List, Tuple

import numpy as np
import pytest

from src.agents.action_codec import index_to_action
from src.config import load_config


def _go_available() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.ffi.bridge import GoEngine

            e = GoEngine(seed=0)
            e.close()
        return True
    except Exception:
        return False


skip_if_no_go = pytest.mark.skipif(
    not _go_available(), reason="libcambia.so not available"
)

_NUM_DEALS = 20
# The full standard deck, so the ability ranks are live, with the turn cap low
# enough that the stockpile never runs out: a reshuffle is the one place the two
# engines' deals legitimately diverge (their shuffle RNGs differ), which would
# make a lockstep comparison meaningless rather than failing (memory cambia-1492).
_MAX_GAME_TURNS = 24
# Under the shipped rules a uniform sampler calls Cambia almost immediately and
# the traversal bottoms out after about five actions, which is far too shallow a
# sequence to be worth comparing. Holding Cambia back and turning opponent
# snapping on takes the same 20 deals to roughly 880 applies and puts the snap
# phase, its penalties and its follow-up move in the compared sequence.
_CAMBIA_ALLOWED_ROUND = 6


def _config():
    cfg = load_config("config/production_train.yaml")
    cfg.cambia_rules.max_game_turns = _MAX_GAME_TURNS
    cfg.cambia_rules.cambia_allowed_round = _CAMBIA_ALLOWED_ROUND
    cfg.cambia_rules.allowOpponentSnapping = True
    cfg.logging.log_simulation_traces = False
    return cfg


def _go_snapshot(state) -> Tuple:
    """Everything the traversal can observe, off the Go engine."""
    terminal = state.is_terminal()
    engine = state.engine
    return (
        tuple(tuple(str(c) for c in state.hand(p)) for p in range(state.num_players())),
        engine.stock_len(),
        engine.discard_len(),
        str(engine.get_discard_top()),
        state.acting_player(),
        terminal,
        state.decision_context(),
        tuple(repr(a) for _, a in state.legal_actions()),
        engine.turn_number(),
        engine.cambia_caller(),
        tuple(state.utility(p) for p in range(state.num_players())) if terminal else None,
    )


def _py_snapshot(pygame) -> Tuple:
    """The same tuple, off the Python engine."""
    from src.analysis_tools import AnalysisTools

    terminal = pygame.is_terminal()
    return (
        tuple(tuple(str(c) for c in p.hand) for p in pygame.players),
        pygame.get_stockpile_size(),
        len(pygame.discard_pile),
        str(pygame.get_discard_top()),
        pygame.get_acting_player(),
        terminal,
        AnalysisTools._get_decision_context(pygame),
        tuple(repr(a) for a in sorted(pygame.get_legal_actions(), key=repr)),
        pygame.get_turn_number(),
        pygame.cambia_caller_id,
        (
            tuple(pygame.get_utility(p) for p in range(len(pygame.players)))
            if terminal
            else None
        ),
    )


def _record_traversal(cfg, deal, iteration: int) -> List[Tuple[str, Any, Tuple]]:
    """The (op, payload, post-op Go state) journal of one real traversal.

    The traversal runs through ``run_cfr_simulation_worker``, so the sequence is
    the production one; ``GoBrState.apply`` and ``.rewind`` are wrapped for the
    duration of the call to write the journal.
    """
    from src.cfr.br_state import GoBrState
    from src.cfr.worker import run_cfr_simulation_worker

    journal: List[Tuple[str, Any, Tuple]] = []
    real_apply, real_rewind = GoBrState.apply, GoBrState.rewind

    def apply(self, action_idx):
        real_apply(self, int(action_idx))
        journal.append(("apply", int(action_idx), _go_snapshot(self)))

    def rewind(self, cp):
        real_rewind(self, cp)
        journal.append(("rewind", self.action_prefix, _go_snapshot(self)))

    GoBrState.apply, GoBrState.rewind = apply, rewind
    try:
        result = run_cfr_simulation_worker(
            (iteration, cfg, {}, None, None, 0, tempfile.mkdtemp(), "lockstep"),
            deal=deal,
        )
    finally:
        GoBrState.apply, GoBrState.rewind = real_apply, real_rewind

    assert result is not None, "the traversal returned no result"
    assert result.stats.error_count == 0, (
        f"the traversal logged {result.stats.error_count} errors, so the recorded "
        "sequence is not a clean one"
    )
    return journal


@skip_if_no_go
def test_traversal_apply_and_undo_sequence_matches_python_engine():
    """Both engines agree after every apply and every undo the traversal makes."""
    from src.cfr.br_state import GoBrState, deal_spec_from_python_game
    from src.game.engine import CambiaGameState

    logging.disable(logging.CRITICAL)
    try:
        cfg = _config()
        applies = rewinds = 0

        for seed in range(_NUM_DEALS):
            pygame = CambiaGameState(
                house_rules=cfg.cambia_rules, _rng=random.Random(4000 + seed)
            )
            deal = deal_spec_from_python_game(pygame)

            np.random.seed(8000 + seed)
            journal = _record_traversal(cfg, deal, seed)
            assert journal, f"deal {seed}: the traversal applied nothing"

            # The deal itself has to match before any step comparison means
            # anything.
            fresh = GoBrState.new(cfg.cambia_rules, deal)
            try:
                assert _go_snapshot(fresh) == _py_snapshot(
                    pygame
                ), f"deal {seed}: the deals differ"
            finally:
                fresh.close()

            undo_stack = []
            for step, (op, payload, go_state) in enumerate(journal):
                if op == "apply":
                    action = index_to_action(payload)
                    _, undo = pygame.apply_action(action)
                    assert callable(undo), f"deal {seed} step {step}: no undo callable"
                    undo_stack.append(undo)
                    applies += 1
                else:
                    assert undo_stack, f"deal {seed} step {step}: rewind with no apply"
                    undo_stack.pop()()
                    rewinds += 1
                    assert len(undo_stack) == len(payload), (
                        f"deal {seed} step {step}: rewound to depth "
                        f"{len(undo_stack)} but the Go prefix is {len(payload)} long"
                    )
                assert _py_snapshot(pygame) == go_state, (
                    f"deal {seed} step {step}: engines diverged after {op} "
                    f"{payload!r}"
                )

        # A traversal that never sampled anything, or one whose rewinds were
        # optimised away, would make the comparison above vacuous.
        assert applies >= 20 * _NUM_DEALS, (
            f"only {applies} actions were compared across {_NUM_DEALS} deals; the "
            "traversals are bottoming out too early to be worth comparing"
        )
        assert rewinds == applies, "every applied action should have been rewound"
    finally:
        logging.disable(logging.NOTSET)


@skip_if_no_go
def test_recorded_sequence_returns_the_engine_to_the_deal():
    """The traversal leaves the engine where it started, as the undo chain did."""
    from src.cfr.br_state import GoBrState, deal_spec_from_python_game
    from src.game.engine import CambiaGameState

    logging.disable(logging.CRITICAL)
    try:
        cfg = _config()
        seen = []
        real_close = GoBrState.close

        def close(self):
            if not self._closed:
                seen.append(self.action_prefix)
            real_close(self)

        GoBrState.close = close
        try:
            pygame = CambiaGameState(
                house_rules=cfg.cambia_rules, _rng=random.Random(4321)
            )
            np.random.seed(999)
            from src.cfr.worker import run_cfr_simulation_worker

            run_cfr_simulation_worker(
                (
                    0,
                    cfg,
                    {},
                    None,
                    None,
                    0,
                    tempfile.mkdtemp(),
                    "lockstep",
                ),
                deal=deal_spec_from_python_game(pygame),
            )
        finally:
            GoBrState.close = real_close

        assert seen, "the traversal never closed its state"
        assert seen[-1] == (), f"the traversal ended at prefix {seen[-1]}, not the deal"
    finally:
        logging.disable(logging.NOTSET)
