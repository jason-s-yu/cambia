"""Tier 1 tests for v3.1 Phase 0 Stream A FFI additions.

Covers:
- cambia_game_get_all_cards vs. a ground-truth derivation from the deal deck
  (100 randomized deals, 2P).
- cambia_agent_encode_eppbs_interleaved_v2 shape + posterior normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ffi.bridge import GoAgentState, GoEngine

# ---------------------------------------------------------------------------
# Card-index -> bucket helper (mirrors engine/agent/constants.go CardToBucket).
# ---------------------------------------------------------------------------

# Suit order in the Go engine's indexToCard: C=0, D=1, H=2, S=3. Jokers: 52, 53.
_SUIT_CLUBS, _SUIT_DIAMONDS, _SUIT_HEARTS, _SUIT_SPADES = 0, 1, 2, 3


def _card_idx_to_bucket(card_idx: int) -> int:
    """Map a canonical 0..53 card index to a CardBucket (0..8)."""
    if card_idx == 52 or card_idx == 53:
        return 0  # BucketZero (Joker)
    suit = card_idx // 13
    rank = card_idx % 13
    if rank == 0:
        return 2  # BucketAce
    if rank == 12:  # King
        return 1 if suit in (_SUIT_DIAMONDS, _SUIT_HEARTS) else 8
    if 1 <= rank <= 3:
        return 3  # 2-4 LowNum
    if rank in (4, 5):
        return 4  # 5-6 MidNum
    if rank in (6, 7):
        return 5  # 7-8 PeekSelf
    if rank in (8, 9):
        return 6  # 9-10 PeekOther
    if rank in (10, 11):
        return 7  # J-Q SwapBlind
    raise ValueError(f"unhandled rank {rank} for idx {card_idx}")


def _random_deck(rng: np.random.Generator, num_jokers: int = 2) -> list[int]:
    """Build a shuffled list of 52 + num_jokers card indices."""
    base = list(range(52))
    jokers = [52 + i for i in range(num_jokers)]
    deck = base + jokers
    rng.shuffle(deck)
    return deck


# ---------------------------------------------------------------------------
# get_all_cards
# ---------------------------------------------------------------------------


class TestGetAllCardsMatchesGoInspection:
    """cambia_game_get_all_cards must match the deck-derived ground truth
    on randomized deals. Round-robin dealing means Player p gets deck[c*N + p]
    for each card slot c.
    """

    @pytest.mark.parametrize("trial", range(100))
    def test_matches_deck_on_random_deals(self, trial: int) -> None:
        rng = np.random.default_rng(trial * 1009 + 7)
        num_players = 2
        cards_per_player = 4
        deck = _random_deck(rng, num_jokers=2)

        engine = GoEngine.from_deck(deck, starting_player=0, house_rules=None)
        try:
            cards = engine._get_all_cards_unsafe()
            assert cards.dtype == np.uint8
            assert cards.shape == (num_players * GoEngine.MAX_HAND_SIZE,)

            for p in range(num_players):
                for s in range(GoEngine.MAX_HAND_SIZE):
                    got = int(cards[p * GoEngine.MAX_HAND_SIZE + s])
                    if s < cards_per_player:
                        expected = _card_idx_to_bucket(deck[s * num_players + p])
                        assert got == expected, (
                            f"trial {trial}: p={p} s={s}: got bucket {got}, "
                            f"expected {expected} from deck idx {deck[s * num_players + p]}"
                        )
                    else:
                        assert (
                            got == 0xFF
                        ), f"trial {trial}: empty slot p={p} s={s} got {got}, want sentinel 0xFF"
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# encode_eppbs_interleaved_v2
# ---------------------------------------------------------------------------


class TestEPPBSV2EncodeShape:
    def test_v2_shape_and_v1_prefix(self) -> None:
        with GoEngine(seed=123) as engine:
            agent = GoAgentState(engine, player_id=0)
            try:
                v2 = agent.encode_eppbs_interleaved_v2(
                    decision_context=engine.decision_ctx(),
                    drawn_bucket=-1,
                )
                v1 = agent.encode_eppbs_interleaved(
                    decision_context=engine.decision_ctx(),
                    drawn_bucket=-1,
                )
            finally:
                agent.close()

        assert v2.shape == (GoEngine.EPPBS_V2_INPUT_DIM,)
        assert v2.dtype == np.float32
        assert v1.shape == (224,)

        # v1 prefix identical to first 224 dims of v2.
        np.testing.assert_allclose(v1, v2[:224], atol=1e-6)

    def test_posterior_sums_to_one(self) -> None:
        with GoEngine(seed=99) as engine:
            agent = GoAgentState(engine, player_id=0)
            try:
                v2 = agent.encode_eppbs_interleaved_v2(
                    decision_context=engine.decision_ctx(),
                    drawn_bucket=-1,
                )
            finally:
                agent.close()

        posterior = v2[224:233]
        assert posterior.min() >= 0
        assert abs(float(posterior.sum()) - 1.0) < 1e-5

    def test_action_history_initially_zero(self) -> None:
        with GoEngine(seed=555) as engine:
            agent = GoAgentState(engine, player_id=0)
            try:
                v2 = agent.encode_eppbs_interleaved_v2(
                    decision_context=engine.decision_ctx(),
                    drawn_bucket=-1,
                )
            finally:
                agent.close()

        window = v2[233:257]
        assert window.shape == (24,)
        assert float(window.sum()) == 0.0
