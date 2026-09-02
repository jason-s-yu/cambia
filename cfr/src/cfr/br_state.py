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

One read a node
---------------
Every read below comes off a ``GameStateView``: one ``cambia_game_apply_and_read``
crossing hands back terminality, the acting seat, the decision context, the turn
number, the stockpile, the discard top, the Cambia caller, the snap window, the
pending record, the legal set and both hands together, and the apply that
produces a node shares that node's crossing (cambia-1902). Before this the
traversal paid about 21 crossings for every applied action, which is why the Go
engine bought it no throughput over the Python one.

The view is a snapshot, so this class drops it at every mutation --
``apply``, ``rewind``, ``_replay`` -- and takes a fresh one lazily. That is
correct exactly as long as nothing else moves ``self.engine`` behind its back;
the engine attribute is public for reads (the lockstep test reads pile lengths
straight off it), and a caller that applies an action through it instead of
through ``apply`` would leave a view describing the previous state.

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
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOwn,
    DecisionContext,
    GameAction,
)
from ..ffi.bridge import GameStateView, GoEngine, HouseRulesView
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

    A clear that drops entries the closing action itself produced hands them to
    ``closing`` first (cambia-1985). Those entries are the ones no observation
    would otherwise ever see, and the belief needs them: a successful own snap
    names the slot that left, and without it the belief truncates the hand from
    the end and keeps the removed card's bucket. They ride their own field on the
    observation rather than going back into the log, because the log is the
    tokenizer channel and Go emits public snap frames only while ``Snap.Active``.
    Entries an earlier observation already delivered are not carried over.
    """

    __slots__ = ("_entries", "_closing")

    def __init__(
        self,
        entries: Optional[List[Dict[str, Any]]] = None,
        closing: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._entries: List[Dict[str, Any]] = list(entries or [])
        self._closing: List[Dict[str, Any]] = list(closing or [])

    def clone(self) -> "_SnapLogMirror":
        return _SnapLogMirror(
            [dict(e) for e in self._entries], [dict(e) for e in self._closing]
        )

    def entries(self) -> List[Dict[str, Any]]:
        """A copy, matching the Python builder's ``copy.deepcopy`` of the log."""
        return [dict(e) for e in self._entries]

    def closing_entries(self) -> List[Dict[str, Any]]:
        """A copy of the entries the last window-closing action produced."""
        return [dict(e) for e in self._closing]

    def clear(self, closed_by: Optional[List[Dict[str, Any]]] = None) -> None:
        """Drop the log. ``closed_by`` names the entries the closing action just
        appended, which are handed to the belief instead of being lost."""
        self._entries = []
        self._closing = list(closed_by or [])

    def start_action(self) -> None:
        """Drop the previous action's closing entries.

        They are live for exactly the one observation taken after the action that
        produced them, the same lifetime the Python engine gives
        ``race_resolution``.
        """
        self._closing = []

    def append(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)


class Checkpoint(NamedTuple):
    """A rewind token. ``snap_h`` is None once the engine snapshot pool is full,
    in which case ``rewind`` replays ``prefix`` forward from the deepest live
    snapshot that lies on the way to it (the deal, when there is none)."""

    snap_h: Optional[int]
    prefix: Tuple[int, ...]
    snap_log: _SnapLogMirror
    king_swap: Optional[Tuple[int, int]]


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
        "_king_swap",
        "_live_snaps",
        "_num_players",
        "_closed",
        "_state",
        "_rules",
    )

    def __init__(self, engine: GoEngine, deal: DealSpec, house_rules: Any) -> None:
        self.engine = engine
        self.deal = deal
        self.house_rules = house_rules
        self._prefix: List[int] = []
        self._snap_log = _SnapLogMirror()
        self._king_swap: Optional[Tuple[int, int]] = None
        self._live_snaps: List[Tuple[Tuple[int, ...], int]] = []
        self._closed = False
        self._state: Optional[GameStateView] = None
        self._rules: Optional[HouseRulesView] = None
        self._num_players = self._view().num_players

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

    def _view(self) -> GameStateView:
        """This node's state, read in one crossing and reused until it moves."""
        state = self._state
        if state is None:
            state = self._state = self.engine.read_state()
        return state

    def _house_rules(self) -> HouseRulesView:
        """The game's rules, read once. They do not change over a game, and a
        replay rewind rebuilds the engine from the same ``house_rules``."""
        rules = self._rules
        if rules is None:
            rules = self._rules = self.engine.get_house_rules()
        return rules

    def num_players(self) -> int:
        return self._num_players

    def is_terminal(self) -> bool:
        return self._view().terminal

    def utility(self, seat: int) -> float:
        """A seat's utility. Only meaningful once the game is terminal.

        The batched read carries the utilities of a terminal state only, since
        scoring every hand is real work and a traversal asks for them nowhere
        else. The engine still answers for a live state, which is what the
        no-legal-actions error path in ``src.analysis_tools`` reads.
        """
        utils = self._view().utility
        if utils is None:
            return float(self.engine.get_utility()[seat])
        return float(utils[seat])

    def acting_player(self) -> int:
        return self._view().acting_player

    def decision_context(self) -> DecisionContext:
        """The acting seat's decision context.

        ``engine/legal.go``'s ``DecisionCtx`` partitions states exactly as
        ``AnalysisTools._get_decision_context`` did and its values are
        numerically identical to ``DecisionContext``'s, terminal included, so
        the mapping is the identity rather than a re-derivation off a pending
        record.
        """
        return DecisionContext(self._view().decision_ctx)

    def legal_actions(self) -> List[Tuple[int, GameAction]]:
        """Legal ``(action index, GameAction)`` pairs, sorted by ``repr``.

        The repr sort is not cosmetic: ``src.cfr.worker`` indexes each stored
        strategy vector by ``sorted(legal_actions, key=repr)``, so the search
        that reads that vector has to enumerate in the same order or it reads
        the probability of a different action. The engine's own ascending index
        order is a different order and would silently mis-index.
        """
        mask = self._view().legal_mask
        pairs = [(int(i), index_to_action(int(i))) for i in np.flatnonzero(mask)]
        pairs.sort(key=lambda pair: repr(pair[1]))
        return pairs

    def hand(self, seat: int):
        """A seat's true hand, slot 0 first."""
        return self._view().hand(seat)

    def initial_peek_indices(self) -> Tuple[int, ...]:
        """The slots dealt face-up to every seat at the start.

        ``PlayerState`` gives every seat ``tuple(range(initial_view_count))``,
        so the rules record is the whole story.
        """
        return tuple(range(int(self._house_rules().initial_view_count)))

    # --- Mutation + rewind ---

    def apply(self, action_idx: int) -> None:
        """Apply an action index, keeping the snap-log mirror in step.

        The snap bookkeeping reads the pre-apply state, so it has to run before
        the action lands; the pre-apply view is already in hand, since the node
        this action leaves was itself read in one crossing, so the whole of it
        is free here.
        """
        action = index_to_action(int(action_idx))
        self._snap_log.start_action()
        before = self._view()
        snap_before = before.snap
        entry = None
        if (
            snap_before.active
            and DecisionContext(before.decision_ctx) is DecisionContext.SNAP_DECISION
        ):
            entry = self._snap_entry(action, before)
        self._king_swap = self._pre_apply_king_swap(action, before)

        # Dropped before the call, not after: a rejected action raises out of
        # here, and a stale view would then outlive the failure.
        self._state = None
        self._state = self.engine.apply_and_read_state(int(action_idx))
        self._prefix.append(int(action_idx))

        snap_after = self._state.snap
        if entry is not None:
            self._snap_log.append(entry)
        if snap_after.active != snap_before.active:
            # A phase starting clears for the new phase (change_snap_start); a
            # phase ending clears inside the same apply that appended the last
            # snapper's entry (change_snap_end). The SnapOpponentMove flush
            # (_flush_snap_results_log) is the same edge seen from the other
            # side: Python flushes only when no snapper remains, which is
            # exactly when Go's Snap.Active drops. A move that RESUMES the
            # phase for the next snapper deliberately keeps the block visible
            # on both engines, so no clear belongs there.
            #
            # The entry this action appended is what the clear would otherwise
            # lose, so it goes to the belief on the closing channel instead
            # (cambia-1985). A phase START clears a block belonging to an
            # earlier window that every interested observation already saw, so
            # nothing is carried there.
            closed_by = [entry] if (entry is not None and not snap_after.active) else []
            self._snap_log.clear(closed_by)

    def checkpoint(self) -> Checkpoint:
        """Take a rewind token for the current node.

        Prefers the engine's snapshot pool and falls back to prefix replay when
        it is full, so a deep recursion degrades in speed rather than failing.
        The fallback rebuilds the engine per rewind and is expensive enough to be
        worth knowing about, so it says so once per process.
        """
        global _REPLAY_FALLBACK_WARNED
        prefix = tuple(self._prefix)
        try:
            snap_h: Optional[int] = self.engine.save()
        except RuntimeError:
            snap_h = None
            if not _REPLAY_FALLBACK_WARNED:
                _REPLAY_FALLBACK_WARNED = True
                logger.warning(
                    "Exhausted the engine snapshot pool at depth %d; rewinding "
                    "by replaying forward from the deepest live snapshot "
                    "instead. Correct but slower.",
                    len(prefix),
                )
        else:
            self._live_snaps.append((prefix, snap_h))
        return Checkpoint(snap_h, prefix, self._snap_log.clone(), self._king_swap)

    def rewind(self, cp: Checkpoint) -> None:
        """Restore the state the checkpoint was taken at."""
        self._state = None
        if cp.snap_h is not None:
            self.engine.restore(cp.snap_h)
        else:
            self._replay(cp.prefix)
        self._prefix = list(cp.prefix)
        self._snap_log = cp.snap_log.clone()
        self._king_swap = cp.king_swap

    def release(self, cp: Checkpoint) -> None:
        """Release a checkpoint's engine snapshot, if it took one."""
        if cp.snap_h is None:
            return
        self.engine.free_snapshot(cp.snap_h)
        for i in range(len(self._live_snaps) - 1, -1, -1):
            if self._live_snaps[i][1] == cp.snap_h:
                del self._live_snaps[i]
                break

    def _replay(self, prefix: Tuple[int, ...]) -> None:
        """Restore the closest live snapshot on the way to ``prefix``, then
        re-apply the rest.

        Every live snapshot sits on the path this state walked, so the deepest
        one that ``prefix`` extends is a valid starting point and replaying from
        it costs the tail rather than the whole game. With none to start from
        the engine is re-dealt, which is the same work the search used to do on
        every fallback rewind.
        """
        self._state = None
        start = 0
        base: Optional[int] = None
        for snap_prefix, snap_h in self._live_snaps:
            n = len(snap_prefix)
            if n <= len(prefix) and n >= start and prefix[:n] == snap_prefix:
                start, base = n, snap_h
        if base is not None:
            self.engine.restore(base)
        else:
            fresh = self._deal_engine(self.house_rules, self.deal)
            old = self.engine
            self.engine = fresh
            old.close()
        for action_idx in prefix[start:]:
            self.engine.apply_action(int(action_idx))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # The snapshot pool is global and finite, so a checkpoint an aborted
        # recursion never released would be lost for the life of the process.
        for _, snap_h in self._live_snaps:
            self.engine.free_snapshot(snap_h)
        self._live_snaps = []
        self.engine.close()

    def __enter__(self) -> "GoBrState":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Observation ---

    def initial_observation(self, ability_reveals: bool = False) -> AgentObservation:
        """The pre-first-action frame ``AgentState.initialize`` consumes."""
        return self.observation(None, -1, ability_reveals=ability_reveals)

    def observation(
        self,
        action: Optional[GameAction],
        acting_player: int,
        ability_reveals: bool = False,
    ) -> AgentObservation:
        """The post-action frame, field for field as the Python builder had it.

        ``ability_reveals`` selects which of the two frames the Python engine
        produced. Off (the default, and what ``src.analysis_tools``' search
        asks for) reproduces ``AnalysisTools._create_observation_for_br``: a
        peek reveals nothing to either belief and a King swap moves no known
        face. On reproduces the production frame
        ``src.cfr.worker._create_observation`` built, which the tabular
        traversal's belief layer needs -- the peeked cards, which
        ``_filter_observation`` then masks down to the peeker, and the King
        swap's slot pair, which is public and tells both beliefs which two
        faces travelled.

        The two frames are not interchangeable: the search's belief was
        verified against the reduced one, and handing it the fuller frame would
        move its exploitability numbers.
        """
        view = self._view()
        caller = view.cambia_caller
        return AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=view.discard_top,
            player_hand_sizes=[view.hand_len(i) for i in range(self._num_players)],
            stockpile_size=view.stock_len,
            drawn_card=self._drawn_card(action, acting_player, view),
            peeked_cards=(
                self._peeked_cards(action, acting_player, view)
                if ability_reveals
                else None
            ),
            snap_results=self._snap_log.entries(),
            closing_snap_results=self._snap_log.closing_entries(),
            did_cambia_get_called=caller is not None,
            who_called_cambia=caller,
            is_game_over=view.terminal,
            current_turn=view.turn_number,
            king_swap_indices=self._king_swap if ability_reveals else None,
        )

    # --- Internals ---

    def _drawn_card(
        self, action: Optional[GameAction], acting_player: int, view: GameStateView
    ):
        """The card this action surfaced to its actor, or None.

        The same three cases the Python builder handled and no others: a
        discard puts the drawn card on top of the pile, a replace leaves it in
        the addressed slot, and a stockpile draw parks it on the pending record.
        """
        if isinstance(action, ActionDiscard):
            return view.discard_top
        if isinstance(action, ActionReplace):
            if acting_player < 0:
                return None
            hand = view.hand(acting_player)
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
        if isinstance(action, (ActionDrawStockpile, ActionDrawDiscard)):
            pending = view.pending
            if pending.seat == acting_player:
                return pending.drawn_card
        return None

    def _pre_apply_king_swap(
        self, action: GameAction, view: GameStateView
    ) -> Optional[Tuple[int, int]]:
        """The (own slot, target slot) pair a King swap is about to move.

        Read before the action applies, because applying it clears the pending
        King record that names the two slots -- the same reason
        ``src.cfr.worker`` captured the pair off ``pending_action_data`` before
        calling ``apply_action``. Declining the swap moves nothing and records
        nothing.
        """
        if not isinstance(action, ActionAbilityKingSwapDecision):
            return None
        if not action.perform_swap:
            return None
        pending = view.pending
        if pending.own_slot is None or pending.target_slot is None:
            return None
        return (int(pending.own_slot), int(pending.target_slot))

    def _peeked_cards(
        self, action: Optional[GameAction], acting_player: int, view: GameStateView
    ):
        """The faces this ability turned up, keyed by (seat, hand slot).

        None for every other action, matching the production builder. The
        engine is read after the action applies, which is where the Python
        builder read too and is safe because none of the three look abilities
        moves a card: the King's swap is a separate later decision.
        """
        if acting_player < 0 or action is None:
            return None
        opponent = 1 - acting_player

        if isinstance(action, ActionAbilityPeekOwnSelect):
            hand = view.hand(acting_player)
            idx = action.target_hand_index
            if 0 <= idx < len(hand):
                return {(acting_player, idx): hand[idx]}
            logger.warning(
                "Create Obs: PeekOwn index %d invalid for hand size %d.",
                idx,
                len(hand),
            )
            return None

        if isinstance(action, ActionAbilityPeekOtherSelect):
            opp_hand = view.hand(opponent)
            idx = action.target_opponent_hand_index
            if 0 <= idx < len(opp_hand):
                return {(opponent, idx): opp_hand[idx]}
            logger.warning(
                "Create Obs: PeekOther index %d invalid for opp hand size %d.",
                idx,
                len(opp_hand),
            )
            return None

        if isinstance(action, ActionAbilityKingLookSelect):
            own_hand = view.hand(acting_player)
            opp_hand = view.hand(opponent)
            own_idx, opp_idx = action.own_hand_index, action.opponent_hand_index
            if 0 <= own_idx < len(own_hand) and 0 <= opp_idx < len(opp_hand):
                return {
                    (acting_player, own_idx): own_hand[own_idx],
                    (opponent, opp_idx): opp_hand[opp_idx],
                }
            logger.warning(
                "Create Obs: KingLook indices invalid. Own %s/%d, Opp %s/%d",
                own_idx,
                len(own_hand),
                opp_idx,
                len(opp_hand),
            )
            return None

        return None

    def _snap_entry(
        self, action: GameAction, view: GameStateView
    ) -> Optional[Dict[str, Any]]:
        """The log entry a snap-phase action appends, decided pre-apply.

        ``view`` is the pre-apply node: the snapper, the open window and both
        hands all come off it.

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

        actor = view.acting_player
        target_rank = view.snap.rank
        success = False
        penalty = False
        snapped_card = None
        attempted_card_str: Optional[str] = None

        if isinstance(action, ActionSnapOwn):
            hand = view.hand(actor)
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
            rules = self._house_rules()
            opp_hand = view.hand(1 - actor)
            idx = action.opponent_target_hand_index
            if not rules.allow_opponent_snapping:
                penalty = True
                attempted_card_str = "Disallowed Action"
            elif view.hand_len(actor) == 0:
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
