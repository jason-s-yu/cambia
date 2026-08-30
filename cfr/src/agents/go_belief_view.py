"""
src/agents/go_belief_view.py

GoBeliefView: the Python AgentState attribute surface, read off a GoAgentState.

``src/action_abstraction.py`` is written against three Python ``AgentState``
attributes -- ``own_hand``, ``opponent_belief`` and ``_current_game_turn`` -- and
reads them with ``getattr(..., None)``, so handing it a GoAgentState does not
fail: the abstraction silently collapses to its default bucket instead, which
would change an evaluated agent's action distribution without any error to
notice. This adapter closes that gap by projecting the Go belief getters into
exactly those three attributes.

Same projection the DESCA Go-backed trainer already uses
(``cli.py:_GoAgentStateAdapter``), so an agent abstracted at eval time is
abstracted the way it was during training. Values are cached and refreshed on
demand rather than per attribute read, because each refresh is three FFI
crossings.
"""

from dataclasses import dataclass
from typing import Dict

from ..constants import CardBucket

__all__ = ["GoBeliefView", "OwnSlotInfo"]

#: Go's BucketUnknown, and its empty/out-of-range slot sentinel. Python's
#: CardBucket.UNKNOWN carries a different numeric value (99), so the two do not
#: line up by value and must be mapped explicitly.
_GO_BUCKET_UNKNOWN = 9
_GO_SLOT_EMPTY = 0xFF


@dataclass
class OwnSlotInfo:
    """Stand-in for AgentState.KnownCardInfo.

    ``action_abstraction`` reads only ``bucket`` and ``last_seen_turn``. Card
    identity is deliberately absent: the Go belief holds a bucket, not a card,
    so exposing a Card attribute here would invent information.
    """

    bucket: CardBucket
    last_seen_turn: int


def _go_bucket_to_py(value: int) -> CardBucket:
    """Map a Go belief bucket byte to its Python CardBucket."""
    if value == _GO_SLOT_EMPTY or value == _GO_BUCKET_UNKNOWN:
        return CardBucket.UNKNOWN
    if 0 <= value <= 8:
        return CardBucket(value)
    return CardBucket.UNKNOWN


class GoBeliefView:
    """The three AgentState attributes ``action_abstraction`` reads, off Go."""

    __slots__ = ("_go_agent", "own_hand", "opponent_belief", "_current_game_turn")

    def __init__(self, go_agent) -> None:
        self._go_agent = go_agent
        self.own_hand: Dict[int, OwnSlotInfo] = {}
        self.opponent_belief: Dict[int, CardBucket] = {}
        self._current_game_turn: int = 0
        self.refresh()

    @property
    def go_agent(self):
        """The wrapped GoAgentState."""
        return self._go_agent

    def refresh(self) -> None:
        """Re-read own hand, opponent belief and turn counter from the Go agent."""
        ga = self._go_agent
        own_arr = ga.get_own_hand_buckets_and_seen()
        opp_arr = ga.get_opp_belief_buckets()
        own_len, opp_len = ga.get_hand_lens()

        self.own_hand = {
            s: OwnSlotInfo(
                bucket=_go_bucket_to_py(int(own_arr[s, 0])),
                last_seen_turn=int(own_arr[s, 1]),
            )
            for s in range(own_len)
        }
        self.opponent_belief = {
            s: _go_bucket_to_py(int(opp_arr[s])) for s in range(opp_len)
        }
        self._current_game_turn = int(ga.get_current_turn())
