"""src/sequence_encoding.py

Perfect-recall sequence tokenizer for PRT-CFR (v0.4).

This module turns a Cambia player's observation-action event stream into a flat
sequence of integer token IDs, one frame per event, suitable for consumption by
a GRU sequence encoder (built separately; this file is the tokenizer only).

Why perfect recall, and what the tokens must carry
---------------------------------------------------
The X1 keystone (cfr/tools/tiny_solver.py, --perfect-recall) proved that keying
CFR on the acting player's GENUINE perfect-recall information state drives the
{A,6} 2-card NashConv to ~0 (5.22e-06), versus the 0.0832 plateau under the
production imperfect-recall belief abstraction. The validated perfect-recall key
for player p is:

  ("PR",
   priv_init[p],         # p's peeked initial-hand cards: ((slot, card_repr), ...)
   tuple(priv_draw[p]),  # p's own stockpile draws, in order: (card_repr, ...)
   tuple(pub_path))      # FULL public reveal sequence for ALL players, in order:
                         #   ((actor, repr(action), repr(discard_top_after)), ...)

This tokenizer encodes EXACTLY that information content, in temporal order, so a
GRU reading the token sequence sees genuine perfect recall:

  - PRIVATE, p's own:
      * peeked initial-hand cards     -> a BOS-anchored prefix of (SLOT, CARD)
                                         frames, one per peeked slot.
      * own stockpile draws           -> the drawn card's identity, surfaced on
                                         p's own Discard/Replace event (the
                                         production observation filter reveals
                                         drawn_card to the actor exactly there;
                                         X1 captures the same card from
                                         pending_action_data at draw time -- same
                                         content, adjacent event).
  - PUBLIC, common knowledge (every player's turn):
      * the actor                     -> ACTOR token.
      * the action's public structure -> ACTION token (tag + slot/flag args; never
                                         card contents, matching repr(action)).
      * the resulting discard top     -> CARD token (post-action public reveal).
      * snap outcomes                 -> SNAP frames (actor, outcome, slot).

The token stream is a deterministic, reversible function of the per-player
filtered observation stream. The AC2 gate (tests/test_sequence_tokenizer.py)
replays seeded games and asserts the decoded token stream reconstructs the
per-player observation-action history losslessly (no information lost relative
to the X1 perfect-recall key content).

Input contract
--------------
Feed this tokenizer the per-player FILTERED observation, i.e.
``_filter_observation(_create_observation(prev, action, next, actor, snaps),
observer_id)`` from src/cfr/worker.py. That is the production information
boundary: drawn_card present only for the actor on Discard/Replace; peeked_cards
present only for the actor who peeked; everything else public. The tokenizer is
driven by the OBSERVATION stream, not by the belief abstraction.

Vocabulary stability
--------------------
Token IDs are assigned from fixed, documented base offsets (see the *_BASE
constants). New event kinds extend at the end; existing IDs never shift. The
design caps a sequence at SEQ_CAP=256 frames; full 2-player games run well under
this (see the gate test's max-length report). The truncation policy
(keep-most-recent) is documented on ``encode_observation_sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .card import Card
from .constants import (
    ALL_RANKS_STR,
    ALL_SUITS,
    JOKER_RANK_STR,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionCallCambia,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    ActionSnapOwn,
)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

#: Maximum hand-slot index the tokenizer can represent. Standard play deals 4;
#: tiny variants deal 2; penalty draws and N-player rules can grow a hand. 8 is a
#: safe ceiling for 2-player Cambia (the engine's 6-slot layout plus headroom).
MAX_SLOTS: int = 8

#: Number of SLOT ids = MAX_SLOTS concrete slots + one dedicated "no slot" id.
#: The "no slot" id lets snap outcomes that carry no concrete slot (penalty /
#: fail / success_other) round-trip distinctly from real slot index MAX_SLOTS-1.
NUM_SLOT_IDS: int = MAX_SLOTS + 1
#: Local id (within the SLOT block) reserved for "no slot / absent".
SLOT_NONE_LOCAL: int = MAX_SLOTS

#: Maximum actor id the tokenizer can represent (player indices 0..MAX_ACTORS-1).
#: 2-player today; sized to 8 so the vocabulary is stable for N-player extension.
MAX_ACTORS: int = 8

#: Sequence length cap (frames). Full 2-player games stay well below this.
SEQ_CAP: int = 256


# ---------------------------------------------------------------------------
# Card <-> id mapping (53 distinct identities + a "none/unknown" id)
# ---------------------------------------------------------------------------
# Cards are identified by (rank, suit) exactly as repr(Card) distinguishes them.
# The two physical jokers share (rank='R', suit=None) and are therefore a single
# identity -- the same collapse the X1 key makes via repr(card). Card ids are
# assigned deterministically: all suited ranks in (rank, suit) order, then the
# single joker, then a trailing "none" id for absent/unknown cards.

_SUITED_RANKS: List[str] = [r for r in ALL_RANKS_STR if r != JOKER_RANK_STR]

#: Ordered list of canonical (rank, suit) card identities. Index = card id.
CARD_IDENTITIES: List[Tuple[str, Optional[str]]] = [
    (rank, suit) for rank in _SUITED_RANKS for suit in ALL_SUITS
] + [(JOKER_RANK_STR, None)]

_CARD_TO_LOCAL: dict = {ident: i for i, ident in enumerate(CARD_IDENTITIES)}

#: Local id (within the card block) reserved for "no card / unknown".
CARD_NONE_LOCAL: int = len(CARD_IDENTITIES)

#: Number of card ids (distinct identities + the none/unknown id).
NUM_CARD_IDS: int = len(CARD_IDENTITIES) + 1  # 53 + 1 = 54


def _card_local_id(card: Optional[Card]) -> int:
    """Local card id within the CARD block; CARD_NONE_LOCAL for None/unknown."""
    if card is None:
        return CARD_NONE_LOCAL
    ident = (card.rank, card.suit)
    return _CARD_TO_LOCAL.get(ident, CARD_NONE_LOCAL)


def _local_card_id_to_card(local: int) -> Optional[Card]:
    """Inverse of ``_card_local_id`` for the round-trip decoder."""
    if local == CARD_NONE_LOCAL:
        return None
    rank, suit = CARD_IDENTITIES[local]
    return Card(rank=rank, suit=suit)


# ---------------------------------------------------------------------------
# Action <-> id mapping (action public structure: tag + slot/flag args)
# ---------------------------------------------------------------------------
# An ACTION token encodes the action's PUBLIC structure -- the tag plus any
# hand-slot indices and boolean flags -- exactly the information repr(action)
# carries, and never card contents. Hand-slot arguments are folded into the
# action id so a single token fixes the action and its targets; this is what
# makes the public path determine the legal-action count (the X1 property).
#
# Action ids are grouped by tag with a fixed per-tag stride. The stride covers
# every (slot,) or (slot, slot) or (flag,) argument combination a tag can take,
# bounded by MAX_SLOTS. New tags append after the last; existing ids never move.

# Argument arity per tag drives the per-tag id stride.
#   none      : single id (tag only)
#   slot1     : MAX_SLOTS ids (one hand-slot arg)
#   slot2     : MAX_SLOTS*MAX_SLOTS ids (two hand-slot args: own, opp)
#   flag      : 2 ids (a boolean arg)
_TAG_NONE = "none"
_TAG_SLOT1 = "slot1"
_TAG_SLOT2 = "slot2"
_TAG_FLAG = "flag"

# (tag-name, arity-kind). Order is FROZEN: ids are assigned by cumulative stride.
_ACTION_SPEC: List[Tuple[str, str]] = [
    ("draw_stockpile", _TAG_NONE),
    ("draw_discard", _TAG_NONE),
    ("call_cambia", _TAG_NONE),
    ("discard_ability", _TAG_NONE),  # ActionDiscard(use_ability=True)
    ("discard_no_ability", _TAG_NONE),  # ActionDiscard(use_ability=False)
    ("replace", _TAG_SLOT1),  # target_hand_index
    ("peek_own", _TAG_SLOT1),  # target_hand_index
    ("peek_other", _TAG_SLOT1),  # target_opponent_hand_index
    ("blind_swap", _TAG_SLOT2),  # (own_hand_index, opponent_hand_index)
    ("king_look", _TAG_SLOT2),  # (own_hand_index, opponent_hand_index)
    ("king_swap", _TAG_FLAG),  # perform_swap
    ("pass_snap", _TAG_NONE),
    ("snap_own", _TAG_SLOT1),  # own_card_hand_index
    ("snap_opp", _TAG_SLOT1),  # opponent_target_hand_index
    ("snap_opp_move", _TAG_SLOT2),  # (own_card_to_move, target_empty_slot)
]


def _tag_stride(kind: str) -> int:
    if kind == _TAG_NONE:
        return 1
    if kind == _TAG_FLAG:
        return 2
    if kind == _TAG_SLOT1:
        return MAX_SLOTS
    if kind == _TAG_SLOT2:
        return MAX_SLOTS * MAX_SLOTS
    raise ValueError(f"unknown action arity kind {kind!r}")


# Cumulative local offset of each tag within the ACTION block.
_ACTION_TAG_OFFSET: dict = {}
_ACTION_TAG_KIND: dict = {}
_acc = 0
for _name, _kind in _ACTION_SPEC:
    _ACTION_TAG_OFFSET[_name] = _acc
    _ACTION_TAG_KIND[_name] = _kind
    _acc += _tag_stride(_kind)

#: Number of action ids (sum of all per-tag strides).
NUM_ACTION_IDS: int = _acc


def _clamp_slot(idx: int) -> int:
    """Clamp a hand-slot index into [0, MAX_SLOTS-1] (defensive; ids stay valid)."""
    if idx < 0:
        return 0
    if idx >= MAX_SLOTS:
        return MAX_SLOTS - 1
    return idx


def _action_local_id(action: Any) -> Tuple[int, str]:
    """Local action id within the ACTION block, plus the resolved tag name.

    Encodes tag + slot/flag args. Returns (-1, "") for None (no main action).
    """
    if action is None:
        return -1, ""
    if isinstance(action, ActionDrawStockpile):
        name = "draw_stockpile"
        return _ACTION_TAG_OFFSET[name], name
    if isinstance(action, ActionDrawDiscard):
        name = "draw_discard"
        return _ACTION_TAG_OFFSET[name], name
    if isinstance(action, ActionCallCambia):
        name = "call_cambia"
        return _ACTION_TAG_OFFSET[name], name
    if isinstance(action, ActionDiscard):
        name = "discard_ability" if action.use_ability else "discard_no_ability"
        return _ACTION_TAG_OFFSET[name], name
    if isinstance(action, ActionReplace):
        name = "replace"
        return _ACTION_TAG_OFFSET[name] + _clamp_slot(action.target_hand_index), name
    if isinstance(action, ActionAbilityPeekOwnSelect):
        name = "peek_own"
        return _ACTION_TAG_OFFSET[name] + _clamp_slot(action.target_hand_index), name
    if isinstance(action, ActionAbilityPeekOtherSelect):
        name = "peek_other"
        return (
            _ACTION_TAG_OFFSET[name] + _clamp_slot(action.target_opponent_hand_index),
            name,
        )
    if isinstance(action, ActionAbilityBlindSwapSelect):
        name = "blind_swap"
        off = _clamp_slot(action.own_hand_index) * MAX_SLOTS + _clamp_slot(
            action.opponent_hand_index
        )
        return _ACTION_TAG_OFFSET[name] + off, name
    if isinstance(action, ActionAbilityKingLookSelect):
        name = "king_look"
        off = _clamp_slot(action.own_hand_index) * MAX_SLOTS + _clamp_slot(
            action.opponent_hand_index
        )
        return _ACTION_TAG_OFFSET[name] + off, name
    if isinstance(action, ActionAbilityKingSwapDecision):
        name = "king_swap"
        return _ACTION_TAG_OFFSET[name] + (1 if action.perform_swap else 0), name
    if isinstance(action, ActionPassSnap):
        name = "pass_snap"
        return _ACTION_TAG_OFFSET[name], name
    if isinstance(action, ActionSnapOwn):
        name = "snap_own"
        return (
            _ACTION_TAG_OFFSET[name] + _clamp_slot(action.own_card_hand_index),
            name,
        )
    if isinstance(action, ActionSnapOpponent):
        name = "snap_opp"
        return (
            _ACTION_TAG_OFFSET[name] + _clamp_slot(action.opponent_target_hand_index),
            name,
        )
    if isinstance(action, ActionSnapOpponentMove):
        name = "snap_opp_move"
        off = _clamp_slot(action.own_card_to_move_hand_index) * MAX_SLOTS + _clamp_slot(
            action.target_empty_slot_index
        )
        return _ACTION_TAG_OFFSET[name] + off, name
    raise ValueError(f"unhandled action type {type(action).__name__}")


def _decode_action_local_id(local: int) -> Any:
    """Inverse of ``_action_local_id``: reconstruct a GameAction from a local id."""
    # Find the tag whose [offset, offset+stride) range contains ``local``.
    name = None
    for _n, _k in _ACTION_SPEC:
        off = _ACTION_TAG_OFFSET[_n]
        if off <= local < off + _tag_stride(_k):
            name = _n
            break
    if name is None:
        raise ValueError(f"action local id {local} out of range")
    rel = local - _ACTION_TAG_OFFSET[name]
    if name == "draw_stockpile":
        return ActionDrawStockpile()
    if name == "draw_discard":
        return ActionDrawDiscard()
    if name == "call_cambia":
        return ActionCallCambia()
    if name == "discard_ability":
        return ActionDiscard(use_ability=True)
    if name == "discard_no_ability":
        return ActionDiscard(use_ability=False)
    if name == "replace":
        return ActionReplace(target_hand_index=rel)
    if name == "peek_own":
        return ActionAbilityPeekOwnSelect(target_hand_index=rel)
    if name == "peek_other":
        return ActionAbilityPeekOtherSelect(target_opponent_hand_index=rel)
    if name == "blind_swap":
        return ActionAbilityBlindSwapSelect(
            own_hand_index=rel // MAX_SLOTS, opponent_hand_index=rel % MAX_SLOTS
        )
    if name == "king_look":
        return ActionAbilityKingLookSelect(
            own_hand_index=rel // MAX_SLOTS, opponent_hand_index=rel % MAX_SLOTS
        )
    if name == "king_swap":
        return ActionAbilityKingSwapDecision(perform_swap=bool(rel))
    if name == "pass_snap":
        return ActionPassSnap()
    if name == "snap_own":
        return ActionSnapOwn(own_card_hand_index=rel)
    if name == "snap_opp":
        return ActionSnapOpponent(opponent_target_hand_index=rel)
    if name == "snap_opp_move":
        return ActionSnapOpponentMove(
            own_card_to_move_hand_index=rel // MAX_SLOTS,
            target_empty_slot_index=rel % MAX_SLOTS,
        )
    raise ValueError(f"unhandled action tag {name!r}")


# ---------------------------------------------------------------------------
# Snap-outcome <-> id mapping
# ---------------------------------------------------------------------------
# Snap results are public. Each result is encoded as a SNAP frame carrying the
# snapper (an ACTOR token), an OUTCOME token, and the affected slot (a SLOT
# token). Outcomes are a small fixed enum.

_SNAP_OUTCOMES: List[str] = [
    "penalty",  # snapper drew penalty cards
    "success_own",  # ActionSnapOwn success (removed own slot)
    "success_opp",  # ActionSnapOpponent success (removed opp slot)
    "success_other",  # success with an unrecognized/absent action_type
    "fail",  # explicit failure, no penalty
]
_SNAP_OUTCOME_TO_LOCAL: dict = {o: i for i, o in enumerate(_SNAP_OUTCOMES)}
NUM_SNAP_OUTCOME_IDS: int = len(_SNAP_OUTCOMES)


def _classify_snap(snap_info: dict) -> Tuple[str, int]:
    """Map a snap_results entry to (outcome_name, slot). slot=-1 when absent."""
    penalty = bool(snap_info.get("penalty", False))
    success = bool(snap_info.get("success", False))
    action_type = snap_info.get("action_type")
    if penalty:
        return "penalty", -1
    if success:
        if action_type == "ActionSnapOwn":
            # NOTE: no "or -1" fallback here -- removed_own_index legitimately
            # can be 0 (the first hand slot), and `0 or -1` would incorrectly
            # collapse a valid index-0 success to the "no slot" sentinel.
            idx = snap_info.get("removed_own_index")
            return "success_own", int(idx) if idx is not None else -1
        if action_type == "ActionSnapOpponent":
            idx = snap_info.get("removed_opponent_index")
            return "success_opp", int(idx) if idx is not None else -1
        return "success_other", -1
    return "fail", -1


# ---------------------------------------------------------------------------
# Frame-marker (event-kind) tokens
# ---------------------------------------------------------------------------
# Each event is encoded as a fixed-width FRAME headed by a marker token, so the
# decoder can segment the flat id stream without ambiguity.

_FRAME_KINDS: List[str] = [
    "init_peek",  # private: a peeked initial-hand slot. payload: SLOT, CARD
    "public",  # public turn event. payload: ACTOR, ACTION, CARD(discard top)
    "drawn",  # private: the observer's own drawn card. payload: CARD
    "snap",  # public snap outcome. payload: ACTOR, OUTCOME, SLOT
    "cambia",  # public: cambia called. payload: ACTOR(caller)
]
_FRAME_TO_LOCAL: dict = {k: i for i, k in enumerate(_FRAME_KINDS)}
NUM_FRAME_IDS: int = len(_FRAME_KINDS)


# ---------------------------------------------------------------------------
# Global token id layout (fixed base offsets; existing ids never shift)
# ---------------------------------------------------------------------------

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
SEP_ID: int = 3
NUM_SPECIAL: int = 4

# Block bases, assigned in a fixed order.
FRAME_BASE: int = NUM_SPECIAL
ACTOR_BASE: int = FRAME_BASE + NUM_FRAME_IDS
ACTION_BASE: int = ACTOR_BASE + MAX_ACTORS
CARD_BASE: int = ACTION_BASE + NUM_ACTION_IDS
SLOT_BASE: int = CARD_BASE + NUM_CARD_IDS
OUTCOME_BASE: int = SLOT_BASE + NUM_SLOT_IDS

#: Total vocabulary size.
VOCAB_SIZE: int = OUTCOME_BASE + NUM_SNAP_OUTCOME_IDS


# Encoders: local id (or sentinel) -> global token id.
def _frame_tok(kind: str) -> int:
    return FRAME_BASE + _FRAME_TO_LOCAL[kind]


def _actor_tok(actor: Optional[int]) -> int:
    # Actor -1 (the synthetic "no actor" of the initial observation) maps to the
    # last actor slot so it round-trips distinctly from real players.
    a = (MAX_ACTORS - 1) if actor is None or actor < 0 else actor
    if a >= MAX_ACTORS:
        a = MAX_ACTORS - 1
    return ACTOR_BASE + a


def _decode_actor_tok(tok: int) -> int:
    a = tok - ACTOR_BASE
    return -1 if a == (MAX_ACTORS - 1) else a


def _action_tok(local: int) -> int:
    return ACTION_BASE + local


def _card_tok(card: Optional[Card]) -> int:
    return CARD_BASE + _card_local_id(card)


def _decode_card_tok(tok: int) -> Optional[Card]:
    return _local_card_id_to_card(tok - CARD_BASE)


def _slot_tok(slot: int) -> int:
    return SLOT_BASE + _clamp_slot(slot)


def _slot_tok_signed(slot: Optional[int]) -> int:
    # Slot -1/absent maps to the dedicated SLOT_NONE id, distinct from every real
    # slot, so snap outcomes that carry no concrete slot (penalty / fail /
    # success_other) round-trip losslessly. success_own / success_opp always
    # carry a real slot.
    if slot is None or slot < 0:
        return SLOT_BASE + SLOT_NONE_LOCAL
    return SLOT_BASE + _clamp_slot(slot)


def _decode_slot_tok_signed(tok: int) -> int:
    """Inverse of ``_slot_tok_signed``: SLOT_NONE -> -1, else the slot index."""
    local = tok - SLOT_BASE
    return -1 if local == SLOT_NONE_LOCAL else local


def _outcome_tok(name: str) -> int:
    return OUTCOME_BASE + _SNAP_OUTCOME_TO_LOCAL[name]


# ---------------------------------------------------------------------------
# Decoded-frame container (what the round-trip decoder returns)
# ---------------------------------------------------------------------------


@dataclass
class DecodedEvent:
    """A reconstructed event from a token sequence (round-trip target).

    Mirrors the perfect-recall information content of one observation frame.
    """

    kind: str
    # init_peek
    peek_slot: Optional[int] = None
    peek_card: Optional[Card] = None
    # public
    actor: Optional[int] = None
    action: Any = None
    discard_top: Optional[Card] = None
    # drawn
    drawn_card: Optional[Card] = None
    # snap
    snap_actor: Optional[int] = None
    snap_outcome: Optional[str] = None
    snap_slot: Optional[int] = None
    # cambia
    cambia_caller: Optional[int] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initial_peek_frames(
    initial_hand: List[Card], initial_peek_indices: Tuple[int, ...]
) -> List[int]:
    """Token frames for the observer's private peeked initial hand.

    One ``init_peek`` frame (FRAME, SLOT, CARD) per peeked slot, in ascending
    slot order. This is the X1 ``priv_init[p]`` content. Emit these once at the
    start of an episode (after BOS), before any observation frames.
    """
    toks: List[int] = []
    for slot in sorted(initial_peek_indices):
        if slot >= len(initial_hand):
            continue
        card = initial_hand[slot]
        toks.append(_frame_tok("init_peek"))
        toks.append(_slot_tok(slot))
        toks.append(_card_tok(card))
    return toks


def observation_frame_groups(observation: Any, observer_id: int) -> List[List[int]]:
    """Per-frame token groups for one per-player FILTERED observation event.

    Returns a list of frames (each a token-list beginning with a FRAME marker)
    so callers can truncate at frame boundaries. Pass the observation already
    filtered for ``observer_id`` (see module docstring). Frames, in order:

      - a ``drawn`` frame iff this observation surfaces the observer's own drawn
        card (actor == observer, drawn_card present): the X1 ``priv_draw`` token.
      - a ``public`` frame: ACTOR, ACTION, CARD(discard top after the action).
        This is one entry of the X1 ``pub_path``.
      - a ``cambia`` frame iff cambia was just called (records the caller).
      - one ``snap`` frame per snap result: ACTOR(snapper), OUTCOME, SLOT.

    The initial observation (acting_player == -1, action is None) emits only the
    public frame with the NONE actor and a SEP action sentinel; this anchors the
    starting discard top, matching the X1 root.
    """
    groups: List[List[int]] = []
    actor = observation.acting_player
    action = observation.action

    # Private own-draw frame (X1 priv_draw).
    drawn = getattr(observation, "drawn_card", None)
    if drawn is not None and actor == observer_id:
        groups.append([_frame_tok("drawn"), _card_tok(drawn)])

    # Public turn frame (X1 pub_path entry).
    local, _name = _action_local_id(action)
    action_tok = SEP_ID if local < 0 else _action_tok(local)
    groups.append(
        [
            _frame_tok("public"),
            _actor_tok(actor),
            action_tok,
            _card_tok(observation.discard_top_card),
        ]
    )

    # Public cambia frame.
    if getattr(observation, "did_cambia_get_called", False) and isinstance(
        action, ActionCallCambia
    ):
        caller = getattr(observation, "who_called_cambia", None)
        groups.append(
            [_frame_tok("cambia"), _actor_tok(caller if caller is not None else actor)]
        )

    # Public snap frames.
    for snap_info in getattr(observation, "snap_results", None) or []:
        snapper = snap_info.get("snapper")
        outcome, slot = _classify_snap(snap_info)
        groups.append(
            [
                _frame_tok("snap"),
                _actor_tok(snapper),
                _outcome_tok(outcome),
                _slot_tok_signed(slot),
            ]
        )

    return groups


def observation_frames(observation: Any, observer_id: int) -> List[int]:
    """Flat token frames for one filtered observation (see observation_frame_groups)."""
    flat: List[int] = []
    for g in observation_frame_groups(observation, observer_id):
        flat.extend(g)
    return flat


class SequenceOverflowError(ValueError):
    """Raised by ``encode_observation_sequence(..., strict=True)`` when the raw
    (untruncated) sequence would exceed ``seq_cap``.

    Production PRT-CFR call sites condition on the FULL observation-action
    prefix (v0.4 Phase 2 window-semantics decision: option A, full-recall both
    sides via a non-firing cap). Silent keep-most-recent truncation there would
    merge distinct full-recall histories that happen to share a common suffix --
    exactly the RC-A/E1 recall-merge PRT-CFR exists to eliminate. ``seq_cap`` in
    production is therefore an ALLOCATION bound, not a window: if this fires, the
    fix is to raise ``seq_cap`` (re-run the P100 instrumentation), never to let
    truncation proceed. Tiny-game paths (X2 gate) keep the default
    ``strict=False`` truncating behavior unchanged.
    """


def encode_observation_sequence(
    initial_hand: List[Card],
    initial_peek_indices: Tuple[int, ...],
    observations: List[Any],
    observer_id: int,
    seq_cap: int = SEQ_CAP,
    add_bos_eos: bool = True,
    strict: bool = False,
) -> List[int]:
    """Encode a full per-player episode to a flat token-id sequence.

    Layout: [BOS] + init_peek frames + per-observation frames + [EOS].

    Truncation policy (FRAME-ALIGNED, keep-most-recent): if the assembled body
    exceeds the budget (``seq_cap`` minus BOS/EOS), whole OLDEST frames are
    dropped until it fits, so the kept suffix always starts on a frame marker and
    ``decode_sequence`` never sees a partial leading frame. The most recent
    frames are preserved (the GRU's final hidden state reflects the latest
    information). The private init_peek prefix is dropped only after all
    observation frames would otherwise have to go, i.e. it is kept preferentially
    as the observer's permanent private knowledge.

    Cap sizing note: the design cap is SEQ_CAP=256. Full 2-player games played to
    natural length exceed this (empirically mean ~726, worst-case ~1200 tokens at
    ~6 tokens/event over ~120 events); on such games this truncation drops the
    oldest events. Short games (and the {A,6} tiny game) stay well under the cap.
    Raising SEQ_CAP (or narrowing per-event frame width) is a downstream
    (GRU/trainer) decision; this tokenizer makes truncation lossless-decodable at
    any cap and the gate test reports the observed max length.

    ``strict``: when True, raise ``SequenceOverflowError`` instead of truncating
    if the raw (untruncated) body would exceed the ``seq_cap`` budget. Production
    PRT-CFR call sites (the raised, non-firing cap) pass ``strict=True`` so an
    overflow is a hard error, never a silent truncation (v0.4 Phase 2
    window-semantics decision, condition 1). Default False preserves the
    existing truncating behavior for tiny-game/X2 and any other caller that
    accepts a bounded window.
    """
    peek_groups: List[List[int]] = []
    for slot in sorted(initial_peek_indices):
        if slot < len(initial_hand):
            peek_groups.append(
                [_frame_tok("init_peek"), _slot_tok(slot), _card_tok(initial_hand[slot])]
            )

    obs_groups: List[List[int]] = []
    for obs in observations:
        obs_groups.extend(observation_frame_groups(obs, observer_id))

    budget = seq_cap - (2 if add_bos_eos else 0)
    if budget < 0:
        budget = 0

    # Frame-aligned keep-most-recent. Walk frames from newest to oldest, keeping
    # whole frames while they fit. Observation frames are preferred over the
    # init_peek prefix only in that the prefix is considered last (oldest).
    ordered = peek_groups + obs_groups  # oldest -> newest

    if strict:
        total_len = sum(len(g) for g in ordered)
        if total_len > budget:
            raise SequenceOverflowError(
                f"observation sequence length {total_len} exceeds strict cap "
                f"budget {budget} (seq_cap={seq_cap}); raise seq_cap rather than "
                f"truncate a production (full-recall) call site"
            )

    kept_rev: List[List[int]] = []
    used = 0
    for g in reversed(ordered):  # newest -> oldest
        if used + len(g) > budget:
            break
        kept_rev.append(g)
        used += len(g)
    kept = list(reversed(kept_rev))  # restore oldest -> newest

    seq: List[int] = []
    if add_bos_eos:
        seq.append(BOS_ID)
    for g in kept:
        seq.extend(g)
    if add_bos_eos:
        seq.append(EOS_ID)
    return seq


# ---------------------------------------------------------------------------
# Decoder (round-trip target for the AC2 gate)
# ---------------------------------------------------------------------------


def decode_sequence(tokens: List[int]) -> List[DecodedEvent]:
    """Decode a token sequence back into a list of ``DecodedEvent`` frames.

    Inverse of ``encode_observation_sequence`` at the information level: every
    field the encoder wrote is recovered. BOS/EOS/PAD are skipped. This is the
    function the AC2 gate uses to assert lossless round-trip against the
    perfect-recall key content.
    """
    events: List[DecodedEvent] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in (PAD_ID, BOS_ID, EOS_ID):
            i += 1
            continue
        if not (FRAME_BASE <= tok < FRAME_BASE + NUM_FRAME_IDS):
            raise ValueError(
                f"decode_sequence: expected a FRAME marker at pos {i}, got id {tok}"
            )
        kind = _FRAME_KINDS[tok - FRAME_BASE]
        if kind == "init_peek":
            slot = tokens[i + 1] - SLOT_BASE
            card = _decode_card_tok(tokens[i + 2])
            events.append(DecodedEvent(kind=kind, peek_slot=slot, peek_card=card))
            i += 3
        elif kind == "public":
            actor = _decode_actor_tok(tokens[i + 1])
            atok = tokens[i + 2]
            action = (
                None if atok == SEP_ID else _decode_action_local_id(atok - ACTION_BASE)
            )
            discard_top = _decode_card_tok(tokens[i + 3])
            events.append(
                DecodedEvent(
                    kind=kind, actor=actor, action=action, discard_top=discard_top
                )
            )
            i += 4
        elif kind == "drawn":
            card = _decode_card_tok(tokens[i + 1])
            events.append(DecodedEvent(kind=kind, drawn_card=card))
            i += 2
        elif kind == "snap":
            snapper = _decode_actor_tok(tokens[i + 1])
            outcome = _SNAP_OUTCOMES[tokens[i + 2] - OUTCOME_BASE]
            slot = _decode_slot_tok_signed(tokens[i + 3])
            events.append(
                DecodedEvent(
                    kind=kind,
                    snap_actor=snapper,
                    snap_outcome=outcome,
                    snap_slot=slot,
                )
            )
            i += 4
        elif kind == "cambia":
            caller = _decode_actor_tok(tokens[i + 1])
            events.append(DecodedEvent(kind=kind, cambia_caller=caller))
            i += 2
        else:  # pragma: no cover - frame table is exhaustive
            raise ValueError(f"decode_sequence: unhandled frame kind {kind!r}")
    return events


def vocab_summary() -> dict:
    """Return the vocabulary layout (sizes + base offsets) for documentation."""
    return {
        "VOCAB_SIZE": VOCAB_SIZE,
        "special": {"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID, "SEP": SEP_ID},
        "blocks": {
            "frame": (FRAME_BASE, NUM_FRAME_IDS),
            "actor": (ACTOR_BASE, MAX_ACTORS),
            "action": (ACTION_BASE, NUM_ACTION_IDS),
            "card": (CARD_BASE, NUM_CARD_IDS),
            "slot": (SLOT_BASE, NUM_SLOT_IDS),
            "outcome": (OUTCOME_BASE, NUM_SNAP_OUTCOME_IDS),
        },
        "MAX_SLOTS": MAX_SLOTS,
        "MAX_ACTORS": MAX_ACTORS,
        "SEQ_CAP": SEQ_CAP,
    }
