"""
src/cfr/br_state.py

GoEngine search substrate for the tabular best-response / exploitability search
in ``src.analysis_tools`` (cambia-1428).

What moves and what does not
----------------------------
Only the RULES engine moves to Go. The tabular lane keys its policy table by
``InfosetKey``, built from the Python ``AgentState`` belief machinery
(``src.agent_state``), and ``src.cfr.worker`` writes that table with the same
machinery during training. Rebuilding the key off ``GoAgentState`` instead would
silently look up different entries than the ones training wrote, so the belief
layer stays exactly where it was: this module hands the unchanged Python
``AgentState`` the same ``AgentObservation`` stream it always consumed, read off
a ``GoEngine`` instead of a ``CambiaGameState``.

Branching
---------
The Python engine offered ``apply_action`` plus an undo callable. The Go engine
has no in-process undo, so a branch is explored under ``checkpoint`` /
``rewind``. The fast path is ``GoEngine.save()`` / ``restore()`` -- the
game-only checkpoint pair, not the token-inclusive ``bridge.state_save`` that
``src.cfr.lbr.GoSearchState`` uses, because this search carries no
``GoAgentState`` handles and so has no token stream to rewind. That pool holds
256 snapshots and a best-response recursion pins one per open depth level, so a
deep tree can exhaust it; ``rewind`` then falls back to replaying the node's
action prefix from the deal, which is slower and exactly as correct. Belief
rewinds the way it always did, by cloning the Python ``AgentState`` per branch.

Multiprocess
------------
An engine handle cannot cross a fork, so a state is described by a
``src.cfr.lbr.DealSpec`` plus the action-index prefix that reaches the node, and
a pool worker rebuilds its own engine by replaying that prefix (the same
record-and-replay idiom ``lbr.SampledInfoset`` uses; a Go deal is a pure
function of its seed or deck and the engine is deterministic given an action
sequence).

Snap results
------------
``AgentObservation.snap_results`` is the one field with no engine accessor: the
Go engine keeps the accumulated snap outcomes for its own tokenizer and exports
nothing. It is reconstructed here from the applied action plus the engine's
pre-apply ground truth, which is exact -- a snap succeeds iff the addressed
card's rank equals the open snap rank, and fails to a penalty otherwise (see
``src/game/_snap_mixin.py``'s ActionSnapOwn / ActionSnapOpponent branches, which
this mirrors field for field). The accumulate-and-clear lifetime of the Python
``snap_results_log`` is mirrored too; ``_SnapLogMirror`` documents each rule
against the Python clear point it reproduces, and
``tests/test_br_state_cross_engine.py`` walks both engines over 40 deals
comparing whole observation frames.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..agent_state import AgentObservation
from ..agents.action_codec import index_to_action
from ..constants import (
    ActionDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOwn,
    DecisionContext,
    GameAction,
)
from ..ffi.bridge import GoEngine
from .lbr import DealSpec

logger = logging.getLogger(__name__)

__all__ = ["DealSpec", "GoBrState", "deal_spec_from_python_game"]

#: Set once the snapshot-pool exhaustion fallback has been reported. A search
#: deep enough to hit it hits it at nearly every node, and one line per node is
#: millions of identical warnings.
_REPLAY_FALLBACK_WARNED = False


def deal_spec_from_python_game(game) -> DealSpec:
    """Pin the deal a Python ``CambiaGameState`` was dealt, for the Go engine.

    Used by the cross-engine equality harness: dealing both engines from one
    deck order is what makes the two searches comparable, since neither
    engine's seeded shuffle reproduces the other's.
    """
    from ..ffi.bridge import extract_deck_from_python_game

    deck, starting_player = extract_deck_from_python_game(game)
    return DealSpec(
        deck=tuple(int(c) for c in deck), starting_player=int(starting_player)
    )


class _SnapLogMirror:
    """Python's ``CambiaGameState.snap_results_log``, rebuilt beside the engine.

    ``AgentState.update`` reads six fields off each entry -- snapper,
    action_type, success, penalty, removed_own_index, removed_opponent_index --
    and ignores the rest, so those six are what this reproduces.

    The clear points matter as much as the entries, because the same entry is
    visible to every observation taken before the log is cleared. All three of
    Python's clears -- ``change_snap_start``, ``change_snap_end`` and
    ``_flush_snap_results_log`` -- land on the same observable edge, the one
    where Go's ``Snap.Active`` flips, so ``GoBrState.apply`` reproduces them with
    a single rule. A SnapOpponentMove that resumes the phase for a remaining
    snapper is deliberately NOT a clear on either engine: the accumulated block
    stays visible to that snapper's decision.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None) -> None:
        self._entries: List[Dict[str, Any]] = list(entries or [])

    def clone(self) -> "_SnapLogMirror":
        return _SnapLogMirror([dict(e) for e in self._entries])

    def entries(self) -> List[Dict[str, Any]]:
        """A copy, matching the Python builder's ``copy.deepcopy`` of the log."""
        return [dict(e) for e in self._entries]

    def clear(self) -> None:
        self._entries = []

    def append(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)


class Checkpoint(NamedTuple):
    """A rewind token. ``snap_h`` is None once the engine snapshot pool is full,
    in which case ``rewind`` replays ``prefix`` from the deal instead."""

    snap_h: Optional[int]
    prefix: Tuple[int, ...]
    snap_log: _SnapLogMirror


class GoBrState:
    """One ``GoEngine`` driven by the tabular best-response search.

    Owns its handle: ``close`` is idempotent and the handle pool is finite, so
    close in a ``finally`` or use the context manager.
    """

    __slots__ = (
        "engine",
        "deal",
        "house_rules",
        "_prefix",
        "_snap_log",
        "_num_players",
        "_closed",
    )

    def __init__(self, engine: GoEngine, deal: DealSpec, house_rules: Any) -> None:
        self.engine = engine
        self.deal = deal
        self.house_rules = house_rules
        self._prefix: List[int] = []
        self._snap_log = _SnapLogMirror()
        self._num_players = int(engine.num_players())
        self._closed = False

    @classmethod
    def new(
        cls,
        house_rules: Any,
        deal: DealSpec,
        action_prefix: Sequence[int] = (),
    ) -> "GoBrState":
        """Deal ``deal`` under ``house_rules``, then replay ``action_prefix``.

        The prefix is how a pool worker reconstructs the node its parent was
        sitting on: the deal is a pure function of the spec and the engine is
        deterministic given an action sequence, so replay lands on the identical
        state, snap-log mirror included.
        """
        if getattr(house_rules, "snapRace", False):
            raise ValueError(
                "GoBrState cannot drive snapRace=true: the race path resolves "
                "every committed snap at once and logs its outcomes through "
                "_resolve_snap_race, which the snap-result reconstruction here "
                "does not model. Run the best-response search under "
                "snapRace=false."
            )
        engine = cls._deal_engine(house_rules, deal)
        try:
            state = cls(engine, deal, house_rules)
            if state._num_players != 2:
                raise ValueError(
                    f"GoBrState is a 2-seat search substrate, got "
                    f"num_players={state._num_players}. It drives the 2-player "
                    "146-action space, whose apply paths resolve the opponent as "
                    "1-acting; above two seats that underflows and panics the Go "
                    "runtime. The tabular lane it serves is 2-seat throughout "
                    "(src.constants.NUM_PLAYERS). See cambia-1171."
                )
        except Exception:
            engine.close()
            raise
        try:
            for action_idx in action_prefix:
                state.apply(int(action_idx))
        except Exception:
            state.close()
            raise
        return state

    @staticmethod
    def _deal_engine(house_rules: Any, deal: DealSpec) -> GoEngine:
        if deal.deck is not None:
            return GoEngine.from_deck(
                list(deal.deck),
                starting_player=int(deal.starting_player),
                house_rules=house_rules,
            )
        return GoEngine(seed=int(deal.seed), house_rules=house_rules)

    # --- Identity, for handing this node to another process ---

    @property
    def action_prefix(self) -> Tuple[int, ...]:
        """The action indices applied since the deal, in order."""
        return tuple(self._prefix)

    # --- Read surface ---

    def num_players(self) -> int:
        return self._num_players

    def is_terminal(self) -> bool:
        return bool(self.engine.is_terminal())

    def utility(self, seat: int) -> float:
        return float(self.engine.get_utility()[seat])

    def acting_player(self) -> int:
        if self.engine.is_terminal():
            return -1
        return int(self.engine.acting_player())

    def decision_context(self) -> DecisionContext:
        """The acting seat's decision context.

        ``engine/legal.go``'s ``DecisionCtx`` partitions states exactly as
        ``AnalysisTools._get_decision_context`` did and its values are
        numerically identical to ``DecisionContext``'s, terminal included, so
        the mapping is the identity rather than a re-derivation off a pending
        record.
        """
        return DecisionContext(int(self.engine.decision_ctx()))

    def legal_actions(self) -> List[Tuple[int, GameAction]]:
        """Legal ``(action index, GameAction)`` pairs, sorted by ``repr``.

        The repr sort is not cosmetic: ``src.cfr.worker`` indexes each stored
        strategy vector by ``sorted(legal_actions, key=repr)``, so the search
        that reads that vector has to enumerate in the same order or it reads
        the probability of a different action. The engine's own ascending index
        order is a different order and would silently mis-index.
        """
        mask = self.engine.legal_actions_mask()
        pairs = [(int(i), index_to_action(int(i))) for i in np.flatnonzero(mask)]
        pairs.sort(key=lambda pair: repr(pair[1]))
        return pairs

    def hand(self, seat: int):
        """A seat's true hand, slot 0 first."""
        return self.engine.get_player_hand(seat)

    def initial_peek_indices(self) -> Tuple[int, ...]:
        """The slots dealt face-up to every seat at the start.

        ``PlayerState`` gives every seat ``tuple(range(initial_view_count))``,
        so the rules record is the whole story.
        """
        count = int(self.engine.get_house_rules().initial_view_count)
        return tuple(range(count))

    # --- Mutation + rewind ---

    def apply(self, action_idx: int) -> None:
        """Apply an action index, keeping the snap-log mirror in step.

        The snap bookkeeping reads the pre-apply state, so it has to run first;
        it is gated on an open snap window because that gate is one FFI call and
        the work behind it is several, and the search spends most of its edges
        nowhere near a snap.
        """
        snap_before = self.engine.get_snap_state()
        entry = None
        if (
            snap_before.active
            and self.decision_context() is DecisionContext.SNAP_DECISION
        ):
            entry = self._snap_entry(
                index_to_action(int(action_idx)),
                int(self.engine.acting_player()),
                snap_before,
            )

        self.engine.apply_action(int(action_idx))
        self._prefix.append(int(action_idx))

        snap_after = self.engine.get_snap_state()
        if entry is not None:
            self._snap_log.append(entry)
        if snap_after.active != snap_before.active:
            # A phase starting clears for the new phase (change_snap_start); a
            # phase ending clears inside the same apply that appended the last
            # snapper's entry (change_snap_end), which is why that entry never
            # reaches an observation. The SnapOpponentMove flush
            # (_flush_snap_results_log) is the same edge seen from the other
            # side: Python flushes only when no snapper remains, which is
            # exactly when Go's Snap.Active drops. A move that RESUMES the
            # phase for the next snapper deliberately keeps the block visible
            # on both engines, so no clear belongs there.
            self._snap_log.clear()

    def checkpoint(self) -> Checkpoint:
        """Take a rewind token for the current node.

        Prefers the engine's snapshot pool and falls back to prefix replay when
        it is full, so a deep recursion degrades in speed rather than failing.
        The fallback rebuilds the engine per rewind and is expensive enough to be
        worth knowing about, so it says so once per process.
        """
        global _REPLAY_FALLBACK_WARNED
        try:
            snap_h: Optional[int] = self.engine.save()
        except RuntimeError:
            snap_h = None
            if not _REPLAY_FALLBACK_WARNED:
                _REPLAY_FALLBACK_WARNED = True
                logger.warning(
                    "Best-response search exhausted the engine snapshot pool at "
                    "depth %d; rewinding by replaying the action prefix from the "
                    "deal instead. Correct but much slower.",
                    len(self._prefix),
                )
        return Checkpoint(snap_h, tuple(self._prefix), self._snap_log.clone())

    def rewind(self, cp: Checkpoint) -> None:
        """Restore the state the checkpoint was taken at."""
        if cp.snap_h is not None:
            self.engine.restore(cp.snap_h)
        else:
            self._replay(cp.prefix)
        self._prefix = list(cp.prefix)
        self._snap_log = cp.snap_log.clone()

    def release(self, cp: Checkpoint) -> None:
        """Release a checkpoint's engine snapshot, if it took one."""
        if cp.snap_h is not None:
            self.engine.free_snapshot(cp.snap_h)

    def _replay(self, prefix: Tuple[int, ...]) -> None:
        """Rebuild the engine from the deal and re-apply ``prefix``."""
        fresh = self._deal_engine(self.house_rules, self.deal)
        old = self.engine
        self.engine = fresh
        old.close()
        for action_idx in prefix:
            self.engine.apply_action(int(action_idx))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.engine.close()

    def __enter__(self) -> "GoBrState":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Observation ---

    def initial_observation(self) -> AgentObservation:
        """The pre-first-action frame ``AgentState.initialize`` consumes."""
        return self.observation(None, -1)

    def observation(
        self, action: Optional[GameAction], acting_player: int
    ) -> AgentObservation:
        """The post-action frame, field for field as the Python builder had it.

        ``peeked_cards`` is None here exactly as it was on the Python path: the
        best-response search never surfaced a peek to either belief, and
        ``_filter_observation_for_br`` nulled the field regardless.
        """
        engine = self.engine
        caller = engine.cambia_caller()
        return AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=engine.get_discard_top(),
            player_hand_sizes=[
                len(engine.get_hand_indices(i)) for i in range(self._num_players)
            ],
            stockpile_size=engine.stock_len(),
            drawn_card=self._drawn_card(action, acting_player),
            peeked_cards=None,
            snap_results=self._snap_log.entries(),
            did_cambia_get_called=caller is not None,
            who_called_cambia=caller,
            is_game_over=engine.is_terminal(),
            current_turn=engine.turn_number(),
        )

    # --- Internals ---

    def _drawn_card(self, action: Optional[GameAction], acting_player: int):
        """The card this action surfaced to its actor, or None.

        The same three cases the Python builder handled and no others: a
        discard puts the drawn card on top of the pile, a replace leaves it in
        the addressed slot, and a stockpile draw parks it on the pending record.
        """
        engine = self.engine
        if isinstance(action, ActionDiscard):
            return engine.get_discard_top()
        if isinstance(action, ActionReplace):
            if acting_player < 0:
                return None
            hand = engine.get_player_hand(acting_player)
            if 0 <= action.target_hand_index < len(hand):
                return hand[action.target_hand_index]
            logger.error(
                "BR Create Obs: ActionReplace index %d invalid for actor %d "
                "hand size %d",
                action.target_hand_index,
                acting_player,
                len(hand),
            )
            return None
        if isinstance(action, ActionDrawStockpile):
            pending = engine.get_pending()
            if pending.seat == acting_player:
                return pending.drawn_card
        return None

    def _snap_entry(
        self, action: GameAction, actor: int, snap
    ) -> Optional[Dict[str, Any]]:
        """The log entry a snap-phase action appends, decided pre-apply.

        Returns None for anything Python's ``_handle_snap_action`` does not
        log: the SnapOpponentMove that resolves a successful opponent snap goes
        through the pending-action handler, not this one.

        The ``snapped_card`` / ``attempted_card_str`` strings are carried even
        though ``AgentState.update`` reads neither, so the reconstructed frame is
        the Python frame rather than merely enough of it -- which is what lets
        tests/test_br_state_cross_engine.py compare whole observations instead of
        a hand-picked subset.
        """
        if not isinstance(action, (ActionPassSnap, ActionSnapOwn, ActionSnapOpponent)):
            return None

        target_rank = snap.rank
        success = False
        penalty = False
        snapped_card = None
        attempted_card_str: Optional[str] = None

        if isinstance(action, ActionSnapOwn):
            hand = self.engine.get_player_hand(actor)
            idx = action.own_card_hand_index
            if not (0 <= idx < len(hand)):
                penalty = True
                attempted_card_str = f"Invalid Index {idx}"
            elif hand[idx].rank == target_rank:
                success = True
                snapped_card = hand[idx]
            else:
                penalty = True
                attempted_card_str = str(hand[idx])
        elif isinstance(action, ActionSnapOpponent):
            rules = self.engine.get_house_rules()
            opp_hand = self.engine.get_player_hand(1 - actor)
            idx = action.opponent_target_hand_index
            if not rules.allow_opponent_snapping:
                penalty = True
                attempted_card_str = "Disallowed Action"
            elif len(self.engine.get_hand_indices(actor)) == 0:
                penalty = True
                attempted_card_str = "No cards to move"
            elif not (0 <= idx < len(opp_hand)):
                penalty = True
                attempted_card_str = f"Invalid Index {idx}"
            elif opp_hand[idx].rank == target_rank:
                success = True
                snapped_card = opp_hand[idx]
            else:
                penalty = True
                attempted_card_str = str(opp_hand[idx])

        entry: Dict[str, Any] = {
            "snapper": actor,
            "action_type": type(action).__name__,
            "target_rank": target_rank,
            "success": success,
            "penalty": penalty,
        }
        if snapped_card is not None:
            entry["snapped_card"] = str(snapped_card)
        if attempted_card_str is not None:
            entry["attempted_card_str"] = attempted_card_str
        if isinstance(action, ActionSnapOwn):
            entry["removed_own_index"] = action.own_card_hand_index if success else None
        if isinstance(action, ActionSnapOpponent):
            entry["removed_opponent_index"] = (
                action.opponent_target_hand_index if success else None
            )
        return entry
