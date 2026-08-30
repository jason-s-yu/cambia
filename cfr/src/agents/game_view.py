"""
src/agents/game_view.py

GameView: the read-only game surface an evaluation agent is written against.

An agent sees the game through exactly two objects: a GameView for public and
ground-truth game state, and (for imperfect-information agents) a
GoAgentState for its own beliefs. Nothing else. The protocol below names every
accessor an agent may reach for; anything not on it is either not part of the
agent-facing contract or belongs to the belief surface.

Two implementations exist:

  - GoEngine (cfr/src/ffi/bridge.py) is the real one, backed by the Go engine.
  - PythonGameView wraps the Python reference CambiaGameState for the
    transition window, so an agent ported to this protocol can still be run
    and diffed against the Python engine while that engine is retired.

Both return the same decoded records (PendingInfo / SnapInfo /
HouseRulesView), which live next to the FFI decode in bridge.py and are
re-exported here so a consumer only needs this module.

Card identities in this surface are ground truth: get_player_hand returns what
a seat actually holds, which is what the perfect-information baselines and any
best-response search need. An imperfect-information agent reads its beliefs
off GoAgentState and consults this surface only at the moments the rules
reveal a card to it.
"""

from typing import List, Optional, Protocol, runtime_checkable

from ..card import Card
from ..constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionSnapOpponentMove,
)
from ..ffi.bridge import (
    DRAWN_FROM_DISCARD,
    DRAWN_FROM_STOCKPILE,
    PENDING_BLIND_SWAP,
    PENDING_DISCARD,
    PENDING_KING_DECISION,
    PENDING_KING_LOOK,
    PENDING_NONE,
    PENDING_PEEK_OTHER,
    PENDING_PEEK_OWN,
    PENDING_SNAP_MOVE,
    HouseRulesView,
    PendingInfo,
    SnapInfo,
)

__all__ = [
    "GameView",
    "PythonGameView",
    "PendingInfo",
    "SnapInfo",
    "HouseRulesView",
    "NO_PENDING",
    "NO_SNAP",
    "PENDING_NONE",
    "PENDING_DISCARD",
    "PENDING_PEEK_OWN",
    "PENDING_PEEK_OTHER",
    "PENDING_BLIND_SWAP",
    "PENDING_KING_LOOK",
    "PENDING_KING_DECISION",
    "PENDING_SNAP_MOVE",
    "DRAWN_FROM_STOCKPILE",
    "DRAWN_FROM_DISCARD",
]


# Shared empty records, so a caller can compare against them and neither
# implementation allocates on the common "nothing pending / no snap window"
# path.
NO_PENDING = PendingInfo(
    type=PENDING_NONE,
    seat=None,
    drawn_card=None,
    drawn_from=None,
    own_slot=None,
    target_slot=None,
    target_seat=None,
    own_card=None,
    target_card=None,
)

NO_SNAP = SnapInfo(
    active=False,
    rank=None,
    card=None,
    snapper_count=0,
    snapper_cursor=0,
    snapper_seat=None,
)


@runtime_checkable
class GameView(Protocol):
    """The complete read-only game surface available to an evaluation agent.

    Deliberately excludes anything that changes the game: an agent returns a
    chosen action to its caller and never applies one itself, so apply_action,
    save/restore and the legal-action masks are not part of this contract. The
    legal action set is handed to the agent alongside the view.
    """

    def num_players(self) -> int:
        """Number of seats in this game."""
        ...

    def acting_player(self) -> int:
        """Seat that must act now."""
        ...

    def turn_number(self) -> int:
        """Turns elapsed, starting at 0."""
        ...

    def is_terminal(self) -> bool:
        """True once the game has ended."""
        ...

    def stock_len(self) -> int:
        """Cards left in the stockpile."""
        ...

    def discard_len(self) -> int:
        """Cards in the discard pile. Cheaper than measuring get_discard_pile."""
        ...

    def get_player_hand(self, seat: int) -> List[Card]:
        """A seat's true hand, slot 0 first, truncated to its hand length."""
        ...

    def get_discard_pile(self) -> List[Card]:
        """The whole discard pile, bottom card first, so the last entry is the top."""
        ...

    def get_discard_top(self) -> Optional[Card]:
        """The top discard, or None if the pile is empty."""
        ...

    def get_pending(self) -> PendingInfo:
        """The pending decision the game is waiting on (see PendingInfo)."""
        ...

    def get_snap_state(self) -> SnapInfo:
        """The open snap window, if any (see SnapInfo)."""
        ...

    def get_house_rules(self) -> HouseRulesView:
        """The live house rules (see HouseRulesView)."""
        ...


# Python pending_action placeholder type -> engine.PendingType. The Python
# engine parks an ActionDiscard(use_ability=False) in pending_action to mean
# "a card has been drawn and not yet played", which is the Go engine's
# PendingDiscard.
_PENDING_TYPE_BY_ACTION = (
    (ActionDiscard, PENDING_DISCARD),
    (ActionAbilityPeekOwnSelect, PENDING_PEEK_OWN),
    (ActionAbilityPeekOtherSelect, PENDING_PEEK_OTHER),
    (ActionAbilityBlindSwapSelect, PENDING_BLIND_SWAP),
    (ActionAbilityKingLookSelect, PENDING_KING_LOOK),
    (ActionAbilityKingSwapDecision, PENDING_KING_DECISION),
    (ActionSnapOpponentMove, PENDING_SNAP_MOVE),
)

_DRAWN_FROM_BY_NAME = {
    "stockpile": DRAWN_FROM_STOCKPILE,
    "discard": DRAWN_FROM_DISCARD,
}


class PythonGameView:
    """GameView over the Python reference CambiaGameState.

    A thin, read-only adapter for the transition window: it renames and repacks
    what CambiaGameState already exposes as attributes and never mutates it. It
    exists so an agent rewritten against GameView can be run on either engine
    and the two diffed, and it goes away with the Python engine.

    Fields are mapped to match the Go implementation's contract exactly, not to
    be marginally more informative where the Python engine happens to know
    more. SnapInfo.card is the notable case: the Python engine keeps the card
    that opened the snap window, the Go engine keeps only its rank, so both
    report the top of the discard pile here and both report the opening card's
    rank in SnapInfo.rank.
    """

    __slots__ = ("_state",)

    def __init__(self, state) -> None:
        self._state = state

    @property
    def state(self):
        """The wrapped CambiaGameState. For the transition window only."""
        return self._state

    # --- Scalars ---

    def num_players(self) -> int:
        return int(self._state.num_players)

    def acting_player(self) -> int:
        return int(self._state.get_acting_player())

    def turn_number(self) -> int:
        return int(self._state.get_turn_number())

    def is_terminal(self) -> bool:
        return bool(self._state.is_terminal())

    def stock_len(self) -> int:
        return int(self._state.get_stockpile_size())

    def discard_len(self) -> int:
        return len(self._state.discard_pile)

    # --- Cards ---

    def get_player_hand(self, seat: int) -> List[Card]:
        return list(self._state.get_player_hand(seat))

    def get_discard_pile(self) -> List[Card]:
        return list(self._state.discard_pile)

    def get_discard_top(self) -> Optional[Card]:
        return self._state.get_discard_top()

    # --- Records ---

    def get_pending(self) -> PendingInfo:
        pending = getattr(self._state, "pending_action", None)
        if pending is None:
            return NO_PENDING

        ptype = PENDING_NONE
        for action_cls, mapped in _PENDING_TYPE_BY_ACTION:
            if isinstance(pending, action_cls):
                ptype = mapped
                break
        if ptype == PENDING_NONE:
            return NO_PENDING

        data = getattr(self._state, "pending_action_data", None) or {}
        seat = getattr(self._state, "pending_action_player", None)
        seat = None if seat is None else int(seat)

        target_slot = data.get("opp_idx")
        if ptype == PENDING_SNAP_MOVE:
            target_slot = data.get("target_empty_slot_index")

        # The Python engine's ability and snap actions carry no target-seat
        # field, so the target is the sole opponent and only defined at two
        # seats (see QueryMixin.get_opponent_index).
        target_seat = None
        if (
            seat is not None
            and self.num_players() == 2
            and ptype in (PENDING_KING_DECISION, PENDING_SNAP_MOVE)
        ):
            target_seat = 1 - seat

        return PendingInfo(
            type=ptype,
            seat=seat,
            drawn_card=_as_card(data.get("drawn_card")),
            drawn_from=_DRAWN_FROM_BY_NAME.get(data.get("drawn_from")),
            own_slot=_as_slot(data.get("own_idx")),
            target_slot=_as_slot(target_slot),
            target_seat=target_seat,
            own_card=_as_card(data.get("card1")),
            target_card=_as_card(data.get("card2")),
        )

    def get_snap_state(self) -> SnapInfo:
        if not getattr(self._state, "snap_phase_active", False):
            return NO_SNAP
        opening_card = getattr(self._state, "snap_discarded_card", None)
        snappers = list(getattr(self._state, "snap_potential_snappers", None) or [])
        cursor = int(getattr(self._state, "snap_current_snapper_idx", 0) or 0)
        return SnapInfo(
            active=True,
            rank=opening_card.rank if isinstance(opening_card, Card) else None,
            card=self._state.get_discard_top(),
            snapper_count=len(snappers),
            snapper_cursor=cursor,
            snapper_seat=(snappers[cursor] if 0 <= cursor < len(snappers) else None),
        )

    def get_house_rules(self) -> HouseRulesView:
        rules = self._state.house_rules
        return HouseRulesView(
            max_game_turns=int(getattr(rules, "max_game_turns", 0)),
            cards_per_player=int(rules.cards_per_player),
            cambia_allowed_round=int(rules.cambia_allowed_round),
            penalty_draw_count=int(rules.penaltyDrawCount),
            allow_draw_from_discard=bool(rules.allowDrawFromDiscardPile),
            allow_replace_abilities=bool(rules.allowReplaceAbilities),
            allow_opponent_snapping=bool(rules.allowOpponentSnapping),
            snap_race=bool(getattr(rules, "snapRace", False)),
            num_jokers=int(rules.use_jokers),
            lock_caller_hand=bool(getattr(rules, "lockCallerHand", True)),
            num_players=int(self._state.num_players),
            initial_view_count=int(rules.initial_view_count),
            num_decks=int(getattr(rules, "num_decks", 1)),
        )


def _as_card(value) -> Optional[Card]:
    """Return value if it is a Card, else None.

    The Python engine's pending_action_data is an untyped dict whose entries
    are absent, None, or occasionally a serialized string depending on the code
    path that filled it.
    """
    return value if isinstance(value, Card) else None


def _as_slot(value) -> Optional[int]:
    """Return value as a hand-slot index if it is one, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
