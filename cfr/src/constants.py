"""
src/constants.py

Defines core constants, enumerations, and action types for the Cambia game and CFR agent.

Includes card ranks/suits, abstraction buckets, game state estimates, and structured
definitions for all possible game actions.
"""

import enum
from typing import NamedTuple, Union, TypeAlias, TYPE_CHECKING

# Use TYPE_CHECKING to allow importing Card only for type hints
if TYPE_CHECKING:
    from .card import Card


# Card Ranks (String representation)
ACE = "A"
TWO = "2"
THREE = "3"
FOUR = "4"
FIVE = "5"
SIX = "6"
SEVEN = "7"
EIGHT = "8"
NINE = "9"

"""T represents ten"""
TEN = "T"
JACK = "J"
QUEEN = "Q"
KING = "K"

"""Joker is represented by the R string"""
JOKER_RANK_STR = "R"

NUMERIC_RANKS_STR = [TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN]
FACE_RANKS_STR = [JACK, QUEEN, KING]
ALL_RANKS_STR = [ACE] + NUMERIC_RANKS_STR + FACE_RANKS_STR + [JOKER_RANK_STR]

# Card Suits (String representation)
SPADES = "S"
HEARTS = "H"
DIAMONDS = "D"
CLUBS = "C"
RED_SUITS = [HEARTS, DIAMONDS]
BLACK_SUITS = [SPADES, CLUBS]
ALL_SUITS = [SPADES, HEARTS, DIAMONDS, CLUBS]


# --- Card Buckets Enum (Abstraction) ---
class CardBucket(enum.Enum):
    """
    Represents abstract categories (buckets) that cards are mapped into.

    Used for information set abstraction in the CFR agent's belief state.
    Buckets group cards by value and/or ability for simplification.
    """

    ZERO = 0  # Joker (Value: 0)
    NEG_KING = 1  # Red King (Value: -1, Ability: Look & Swap)
    ACE = 2  # Ace (Value: 1)
    LOW_NUM = 3  # 2, 3, 4 (Value: 2-4)
    MID_NUM = 4  # 5, 6 (Value: 5-6)
    PEEK_SELF = 5  # 7, 8 (Value: 7-8, Ability: Look Own)
    PEEK_OTHER = 6  # 9, T (Value: 9-10, Ability: Look Other's)
    SWAP_BLIND = 7  # Jack, Queen (Value: 11-12, Ability: Blind Swap)
    HIGH_KING = 8  # Black King (Value: 13, Ability: Look & Swap)
    UNKNOWN = 99  # Placeholder for unknown card belief


# --- Memory Decay Categories ---
class DecayCategory(enum.Enum):
    """
    Represents broader categories used when an agent's knowledge decays.

    If memory_level >= 1, specific CardBucket knowledge about an opponent's card
    can decay into one of these categories based on game events or time.
    """

    LIKELY_LOW = 100  # ZERO, NEG_KING, ACE, LOW_NUM
    LIKELY_MID = 101  # MID_NUM, PEEK_SELF
    LIKELY_HIGH = 102  # PEEK_OTHER, SWAP_BLIND, HIGH_KING
    UNKNOWN = CardBucket.UNKNOWN.value  # Re-use UNKNOWN state value for consistency


class DecisionContext(enum.Enum):
    """
    Enumerates the different contexts in which an agent needs to make a decision.

    Used as part of the InfosetKey to distinguish between similar game states
    that require different actions or strategies.
    """

    START_TURN = 0  # Choosing Draw Stockpile / Draw Discard / Call Cambia
    POST_DRAW = 1  # Choosing Discard (Ability/No) / Replace
    SNAP_DECISION = 2  # Choosing Pass Snap / Snap Own / Snap Opponent
    ABILITY_SELECT = 3  # Choosing target/decision for 7/8/9/T/J/Q/K ability
    SNAP_MOVE = 4  # Choosing which card to move after successful Snap Opponent
    TERMINAL = 5  # Although CFR stops before this, useful for completeness


# --- Game Phases / State Estimates ---
class GamePhase(enum.Enum):
    """
    Abstract representation of the current phase of the game.

    Used as part of the InfosetKey for state abstraction, typically based on
    factors like stockpile size or whether Cambia has been called.
    """

    START = 0  # Technically pre-first turn, maybe unused in infoset
    EARLY = 1  # Stockpile HIGH
    MID = 2  # Stockpile MEDIUM
    LATE = 3  # Stockpile LOW/EMPTY
    CAMBIA_CALLED = 4  # Cambia is active
    TERMINAL = 5  # Game is over


class StockpileEstimate(enum.Enum):
    """Abstract representation of the remaining stockpile size."""

    HIGH = 0
    MEDIUM = 1
    LOW = 2
    EMPTY = 3


class EpistemicTag(enum.IntEnum):
    """
    Epistemic tags for EP-PBS (Epistemic Public Belief State) encoding.

    Tracks who knows the identity of each card slot from a single agent's perspective.
    Used in the EP-PBS 200-dim encoding alongside slot identity features.
    """

    UNK = 0       # Unknown to both players
    PRIV_OWN = 1  # This agent privately knows
    PRIV_OPP = 2  # Opponent privately knows
    PUB = 3       # Common knowledge (both know)


# EP-PBS encoding constants
EP_PBS_INPUT_DIM = 200
EP_PBS_MAX_SLOTS = 12  # 6 own + 6 opp for 2P
EP_PBS_TAG_DIM = 4
EP_PBS_BUCKET_DIM = 9
EP_PBS_PUBLIC_DIM = 40
EP_PBS_MAX_ACTIVE_MASK = 3

# N-Player constants
N_PLAYER_INPUT_DIM = 580
N_PLAYER_NUM_ACTIONS = 452
N_PLAYER_MAX_PLAYERS = 6
N_PLAYER_MAX_SLOTS = 36  # 6 players × 6 cards
N_PLAYER_POWERSET_DIM = 216  # 36 slots × 6 bits
N_PLAYER_IDENTITY_DIM = 324  # 36 slots × 9 buckets
N_PLAYER_PUBLIC_DIM = 40

# Bucket midpoints for saliency eviction
_BUCKET_MIDPOINTS = {
    0: 0.0,    # BucketZero (Joker)
    1: -1.0,   # BucketNegKing
    2: 1.0,    # BucketAce
    3: 3.0,    # BucketLowNum (2-4)
    4: 5.5,    # BucketMidNum (5-6)
    5: 7.5,    # BucketPeekSelf (7-8)
    6: 9.5,    # BucketPeekOther (9-10)
    7: 11.5,   # BucketSwapBlind (J-Q)
    8: 13.0,   # BucketHighKing
}


def bucket_saliency(bucket: int) -> float:
    """Saliency = distance from neutral midpoint 4.5."""
    return abs(_BUCKET_MIDPOINTS.get(bucket, 4.5) - 4.5)


# --- Game Constants ---
INITIAL_HAND_SIZE = 4
"""Initial number of cards dealt to each player."""

NUM_PLAYERS = 2
"""Number of players in the game (fixed for 1v1 head-to-head)."""

# Define CardObject alias using a string forward reference
# This avoids importing the Card class directly here.
CardObject: TypeAlias = "Card"
"""Type alias for the Card class, used for clarity in type hints."""


# --- Structured Action Definitions ---
# Using NamedTuple for clarity, hashability, and type checking


class ActionDrawStockpile(NamedTuple):
    """Action: Draw a card from the top of the stockpile."""

    tag: str = "draw_stockpile"


class ActionDrawDiscard(NamedTuple):
    """Action: Draw the top card from the discard pile (if allowed by rules)."""

    tag: str = "draw_discard"


class ActionCallCambia(NamedTuple):
    """Action: Call "Cambia" to initiate the end-game sequence."""

    tag: str = "call_cambia"


class ActionDiscard(NamedTuple):
    """Action: Discard the card just drawn from stockpile/discard."""

    use_ability: bool  # Does the player intend to use the card's ability?
    tag: str = "discard"  # type discriminator — prevents hash collision with ActionReplace(0)


class ActionReplace(NamedTuple):
    """Action: Replace a card in hand with the card just drawn."""

    target_hand_index: int  # Index of the card in hand to replace
    tag: str = "replace"  # type discriminator


# --- Ability-related Actions (Sub-steps) ---
class ActionAbilityPeekOwnSelect(NamedTuple):
    """Action: Select own card to peek (resolves 7/8 ability)."""

    target_hand_index: int  # Index of own card to peek
    tag: str = "peek_own"  # type discriminator


class ActionAbilityPeekOtherSelect(NamedTuple):
    """Action: Select opponent's card to peek (resolves 9/T ability)."""

    target_opponent_hand_index: int  # Index of opponent's card to peek
    tag: str = "peek_other"  # type discriminator


class ActionAbilityBlindSwapSelect(NamedTuple):
    """Action: Select own and opponent's cards for blind swap (resolves J/Q ability)."""

    own_hand_index: int  # Index of own card to swap
    opponent_hand_index: int  # Index of opponent's card to swap
    tag: str = "blind_swap"  # type discriminator


class ActionAbilityKingLookSelect(NamedTuple):
    """Action: Select own and opponent's cards to look at (first part of K ability)."""

    own_hand_index: int  # Index of own card to look at
    opponent_hand_index: int  # Index of opponent's card to look at
    tag: str = "king_look"  # type discriminator


class ActionAbilityKingSwapDecision(NamedTuple):
    """Action: Decide whether to perform swap after looking (second part of K ability)."""

    perform_swap: bool  # True to swap the looked-at cards, False otherwise
    tag: str = "king_swap"  # type discriminator — prevents hash collision with ActionDiscard


# --- Snap Actions ---
# Represents the *attempt* to snap. Engine verifies and applies penalty/success.
class ActionPassSnap(NamedTuple):
    """Action: Explicitly pass the opportunity to snap."""

    tag: str = "pass_snap"


class ActionSnapOwn(NamedTuple):
    """Action: Attempt to snap by discarding a matching card from own hand."""

    own_card_hand_index: int  # Index of own card that matches the discard pile top
    tag: str = "snap_own"  # type discriminator


class ActionSnapOpponent(NamedTuple):
    """Action: Attempt to snap by targeting a matching card in opponent's hand."""

    opponent_target_hand_index: (
        int  # Index of *opponent's* card believed to match discard pile top
    )
    tag: str = "snap_opp"  # type discriminator


class ActionSnapOpponentMove(NamedTuple):
    """Action: Choose which own card to move to opponent's empty slot after successful opponent snap."""

    own_card_to_move_hand_index: int  # Index of own card to move
    target_empty_slot_index: int  # Index where opponent's card was removed
    tag: str = "snap_opp_move"  # type discriminator


# Union type for all possible actions
GameAction = Union[
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
]
"""A type alias representing any possible action a player can take in the game."""
