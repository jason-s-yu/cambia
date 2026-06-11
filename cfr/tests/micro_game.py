"""
tests/micro_game.py

Reduced-Cambia builder used by Phase 1's convergence tests.

Spec (Phase 0 contract item 8):
- 2 players
- 2-card hands each
- 20-card deck, ranks A-5 across four suits (no jokers, no abilities)
- Start discard is the top of the shuffled stockpile (matches the Go engine)

This sprint ships only the builder and a smoke test. Training runs against the
micro game are a Phase 1 deliverable.

Use from pytest:

    from tests.micro_game import build_micro_game
    game = build_micro_game(seed=0)

Import name path is `tests.micro_game` when invoked through pytest from the
`cfr/` directory.
"""

from __future__ import annotations

import random
from typing import List

import pytest

from src.card import Card
from src.config import CambiaRulesConfig
from src.constants import ACE, FIVE, FOUR, THREE, TWO, ALL_SUITS
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState


# Twenty-card reduced deck: A-5 in each of four suits. All these ranks have
# no ability on discard, so the micro game stays ability-free.
_MICRO_RANKS: List[str] = [ACE, TWO, THREE, FOUR, FIVE]


def build_micro_deck(rng: random.Random) -> List[Card]:
    """Build and shuffle a fresh 20-card micro deck."""
    deck = [Card(rank, suit) for rank in _MICRO_RANKS for suit in ALL_SUITS]
    rng.shuffle(deck)
    return deck


def build_micro_rules() -> CambiaRulesConfig:
    """Rules config tuned for the micro game (2-card hands, both peeked, no abilities)."""
    return CambiaRulesConfig(
        allowDrawFromDiscardPile=True,
        allowReplaceAbilities=False,
        snapRace=False,
        penaltyDrawCount=2,
        use_jokers=0,
        cards_per_player=2,
        initial_view_count=2,
        cambia_allowed_round=0,
        allowOpponentSnapping=False,
        max_game_turns=32,
        lockCallerHand=True,
        num_decks=1,
    )


def build_micro_game(seed: int = 0) -> CambiaGameState:
    """Build a reduced-Cambia game state from a deterministic seed.

    The returned game is ready for action selection: both players hold 2 cards
    they have already peeked, the stockpile contains 15 cards, and the discard
    pile has one starter card (mirroring the Go engine's Deal()).
    """
    rng = random.Random(seed)
    deck = build_micro_deck(rng)

    rules = build_micro_rules()
    players = [
        PlayerState(initial_peek_indices=tuple(range(rules.initial_view_count)))
        for _ in range(2)
    ]
    for _ in range(rules.cards_per_player):
        for p in players:
            p.hand.append(deck.pop())
    discard = [deck.pop()] if deck else []

    game_rng = random.Random(seed ^ 0xA5A5A5A5)
    return CambiaGameState(
        players=players,
        stockpile=deck,
        discard_pile=discard,
        current_player_index=game_rng.randint(0, 1),
        num_players=2,
        house_rules=rules,
        _rng=game_rng,
    )


# ---------------------------------------------------------------------------
# Smoke test: harness produces a valid game
# ---------------------------------------------------------------------------

def test_micro_game_builder_produces_valid_state():
    game = build_micro_game(seed=0)

    assert game.num_players == 2
    assert len(game.players) == 2
    for p in game.players:
        assert len(p.hand) == 2
    # 20 cards - 4 dealt - 1 to discard = 15 remaining in stockpile
    assert len(game.stockpile) == 15
    assert len(game.discard_pile) == 1
    assert not game.is_terminal()

    legal = game.get_legal_actions()
    assert len(legal) > 0, "Fresh micro game must have at least one legal action"


def test_micro_game_builder_is_deterministic():
    g1 = build_micro_game(seed=1234)
    g2 = build_micro_game(seed=1234)
    assert [c.rank for c in g1.stockpile] == [c.rank for c in g2.stockpile]
    assert [c.suit for c in g1.stockpile] == [c.suit for c in g2.stockpile]
    assert g1.current_player_index == g2.current_player_index


def test_micro_game_deck_has_no_ability_ranks():
    """None of the 20 cards should trigger abilities."""
    rng = random.Random(0)
    deck = build_micro_deck(rng)
    assert len(deck) == 20
    ability_ranks = {"7", "8", "9", "T", "J", "Q", "K"}
    for c in deck:
        assert c.rank not in ability_ranks, (
            f"Micro deck must be ability-free; found rank {c.rank!r}"
        )
