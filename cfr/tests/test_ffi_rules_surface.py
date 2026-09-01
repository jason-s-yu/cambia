"""
tests/test_ffi_rules_surface.py

The house-rules surface that crosses the FFI: the deck-rank mask (cambia-1478)
and the constructor bounds (cambia-1555).

cambia-1478: the bridge used to pass 12 of CambiaRulesConfig's 13 rule fields
to cambia_game_new_with_rules and drop deck_ranks, so tiny_2card_plateau.yaml
dealt a 2-rank 8-card game on the Python engine and a 13-rank 54-card game on
the Go engine. Every tiny-game Go measurement taken before the guard was a
different game than the one reported. The cross-engine cases below deal that
exact config on both engines and compare the composition.

cambia-1555: cards_per_player, use_jokers, num_decks and initial_view_count
were unbounded ints all the way into libcambia.so, where the Go engine has no
recover(). The cases below pin the two gates that now stand between a YAML typo
and that: the pydantic bounds on CambiaRulesConfig, and the constructors'
rejection of a rules record HouseRules.Validate refuses.

Requires libcambia.so to be built and available (skipped otherwise).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydantic import ValidationError

from src.config import CambiaRulesConfig, load_config
from src.ffi.bridge import ALL_DECK_RANKS_MASK, deck_rank_mask, effective_deck_rank_mask
from src.game.engine import CambiaGameState


def _real_config_module():
    """The real src.config, not conftest.py's stub.

    conftest installs a bounds-free CambiaRulesConfig stub under "src.config"
    so the heavy modules import cheaply, which is exactly the model these
    bounds are not on. Same pop-import-restore dance the stub's own
    load_config uses to reach the real implementation.
    """
    import importlib

    saved = sys.modules.pop("src.config", None)
    try:
        return importlib.import_module("src.config")
    finally:
        if saved is not None:
            sys.modules["src.config"] = saved


#: The real pydantic model the YAML loader builds, bounds and all.
RealCambiaRulesConfig = _real_config_module().CambiaRulesConfig

#: The tiny cause-isolation game whose deck the bridge used to drop.
TINY_CONFIG = str(Path(_PROJECT_ROOT) / "config" / "tiny_2card_plateau.yaml")


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        engine = GoEngine.from_deck(list(range(54)))
        engine.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")

if go_available:
    from src.ffi.bridge import GoEngine


class _RawRules:
    """A rules object that skips CambiaRulesConfig's own bounds.

    The bridge reads house rules by attribute, so this stands in for a caller
    that built its rules some other way, and lets the FFI rejection be tested
    on values CambiaRulesConfig now refuses outright.
    """

    def __init__(self, **overrides):
        self.allowDrawFromDiscardPile = False
        self.allowReplaceAbilities = False
        self.snapRace = False
        self.allowOpponentSnapping = False
        self.penaltyDrawCount = 2
        self.use_jokers = 2
        self.cards_per_player = 4
        self.initial_view_count = 2
        self.cambia_allowed_round = 0
        self.max_game_turns = 46
        self.lockCallerHand = True
        self.num_decks = 1
        self.num_players = 2
        self.deck_ranks: Optional[list] = None
        for key, value in overrides.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# deck_ranks -> DeckRanks mask
# ---------------------------------------------------------------------------


class TestDeckRankMask:
    def test_unset_encodes_as_the_all_ranks_sentinel(self):
        assert deck_rank_mask(None) == 0
        assert effective_deck_rank_mask(None) == ALL_DECK_RANKS_MASK

    def test_encodes_each_rank_at_its_engine_index(self):
        # A = bit 0, 6 = bit 5, K = bit 12, in the engine's rank order.
        assert deck_rank_mask(["A"]) == 0b1
        assert deck_rank_mask(["A", "6"]) == 0b100001
        assert deck_rank_mask(["K"]) == 1 << 12

    def test_order_does_not_change_the_mask(self):
        assert deck_rank_mask(["6", "A"]) == deck_rank_mask(["A", "6"])

    def test_every_suited_rank_is_the_full_mask(self):
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
        assert deck_rank_mask(ranks) == ALL_DECK_RANKS_MASK

    def test_refuses_an_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            deck_rank_mask([])

    def test_refuses_an_unknown_rank(self):
        with pytest.raises(ValueError, match="not a suited rank"):
            deck_rank_mask(["A", "11"])

    def test_refuses_the_joker_rank(self):
        # Jokers are governed by use_jokers; the mask covers suited ranks only.
        with pytest.raises(ValueError, match="not a suited rank"):
            deck_rank_mask(["A", "R"])

    def test_refuses_a_repeated_rank(self):
        # A set of bits cannot say "twice", but the Python deck builder would
        # deal that rank twice, so a repeat is a cross-engine divergence.
        with pytest.raises(ValueError, match="more than once"):
            deck_rank_mask(["A", "A"])


# ---------------------------------------------------------------------------
# The tiny game, dealt on both engines
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestTinyDeckCrossesTheFFI:
    """tiny_2card_plateau.yaml deals the same 2-rank game on either engine."""

    @pytest.fixture(scope="class")
    def rules(self) -> CambiaRulesConfig:
        return load_config(TINY_CONFIG).cambia_rules

    def test_the_config_still_names_two_ranks(self, rules):
        assert rules.deck_ranks == ["A", "6"]
        assert rules.use_jokers == 0
        assert rules.cards_per_player == 2

    def test_go_deals_only_the_configured_ranks(self, rules):
        engine = GoEngine(seed=7, house_rules=rules)
        try:
            dealt = [
                card for seat in range(2) for card in engine.get_player_hand(seat)
            ] + engine.get_discard_pile()
            assert {card.rank for card in dealt} <= {"A", "6"}
        finally:
            engine.close()

    def test_go_and_python_agree_on_the_deck_size(self, rules):
        python_state = CambiaGameState(house_rules=rules, seed=7)
        python_stock = len(python_state.stockpile)
        python_total = (
            python_stock
            + len(python_state.discard_pile)
            + sum(len(p.hand) for p in python_state.players)
        )

        engine = GoEngine(seed=7, house_rules=rules)
        try:
            go_stock = engine.stock_len()
            go_total = (
                go_stock
                + len(engine.get_discard_pile())
                + sum(len(engine.get_player_hand(s)) for s in range(2))
            )
        finally:
            engine.close()

        # 2 ranks x 4 suits, no jokers: 8 cards, 4 dealt, 1 flipped, 3 left.
        assert python_total == 8
        assert go_total == python_total
        assert go_stock == python_stock == 3

    def test_every_go_deal_comes_out_of_the_same_eight_card_deck(self, rules):
        # The stockpile's contents are not exposed, so the composition is
        # pinned across seeds instead: every card the deal reveals is one of
        # the 8 the Python engine builds, and none is dealt twice.
        python_deck = {
            (card.rank, card.suit)
            for card in (
                CambiaGameState(house_rules=rules, seed=7).stockpile
                + CambiaGameState(house_rules=rules, seed=7).discard_pile
                + [
                    c
                    for p in CambiaGameState(house_rules=rules, seed=7).players
                    for c in p.hand
                ]
            )
        }
        assert len(python_deck) == 8

        for seed in range(1, 25):
            engine = GoEngine(seed=seed, house_rules=rules)
            try:
                revealed = [
                    (card.rank, card.suit)
                    for seat in range(2)
                    for card in engine.get_player_hand(seat)
                ] + [(c.rank, c.suit) for c in engine.get_discard_pile()]
                stock = engine.stock_len()
            finally:
                engine.close()

            assert len(revealed) == len(set(revealed)), f"seed {seed} dealt a duplicate"
            assert set(revealed) <= python_deck, f"seed {seed} dealt an off-deck card"
            assert stock == 3, f"seed {seed} left {stock} in the stockpile, want 3"

    def test_the_rules_read_back_the_mask(self, rules):
        engine = GoEngine(seed=7, house_rules=rules)
        try:
            view = engine.get_house_rules()
        finally:
            engine.close()
        assert view.deck_rank_mask == deck_rank_mask(["A", "6"])

    def test_a_full_deck_reads_back_every_rank(self):
        engine = GoEngine(seed=7, house_rules=CambiaRulesConfig())
        try:
            view = engine.get_house_rules()
        finally:
            engine.close()
        assert view.deck_rank_mask == ALL_DECK_RANKS_MASK

    def test_search_state_no_longer_refuses_the_config(self, rules):
        # The cambia-1427 interim guard refused any deck_ranks config on the
        # search path, since dealing it on Go measured a different game.
        from src.cfr.lbr import GoSearchState

        state = GoSearchState.new(rules, seed=7)
        try:
            hand = state.engine.get_player_hand(0)
        finally:
            state.close()
        assert {card.rank for card in hand} <= {"A", "6"}


# ---------------------------------------------------------------------------
# Config bounds (refused before libcambia.so ever sees them)
# ---------------------------------------------------------------------------


class TestRulesConfigBounds:
    @pytest.mark.parametrize("cards", [0, 7, 255])
    def test_refuses_an_impossible_hand_size(self, cards):
        with pytest.raises(ValidationError):
            RealCambiaRulesConfig(cards_per_player=cards)

    @pytest.mark.parametrize("cards", [1, 4, 6])
    def test_accepts_a_hand_the_engine_can_hold(self, cards):
        assert RealCambiaRulesConfig(cards_per_player=cards, initial_view_count=1)

    @pytest.mark.parametrize("jokers", [3, 255])
    def test_refuses_more_jokers_than_a_deck_holds(self, jokers):
        with pytest.raises(ValidationError):
            RealCambiaRulesConfig(use_jokers=jokers)

    @pytest.mark.parametrize("jokers", [0, 1, 2])
    def test_accepts_every_valid_joker_count(self, jokers):
        assert RealCambiaRulesConfig(use_jokers=jokers)

    @pytest.mark.parametrize("decks", [5, 255])
    def test_refuses_more_decks_than_the_stockpile_holds(self, decks):
        with pytest.raises(ValidationError):
            RealCambiaRulesConfig(num_decks=decks)

    @pytest.mark.parametrize("decks", [0, 1, 4])
    def test_accepts_every_valid_deck_count(self, decks):
        assert RealCambiaRulesConfig(num_decks=decks)

    def test_refuses_a_peek_wider_than_the_hand(self):
        with pytest.raises(ValidationError):
            RealCambiaRulesConfig(cards_per_player=2, initial_view_count=3)

    def test_accepts_a_peek_at_the_whole_hand(self):
        assert RealCambiaRulesConfig(cards_per_player=2, initial_view_count=2)

    def test_the_shipped_tiny_config_still_loads(self):
        assert load_config(TINY_CONFIG).cambia_rules.cards_per_player == 2


# ---------------------------------------------------------------------------
# FFI rejection (the gate behind the config bounds)
# ---------------------------------------------------------------------------


@skip_if_no_go
# _RawRules is deliberately not a CambiaRulesConfig; the bridge's advisory
# warning about that is the point of the stand-in, not a finding.
@pytest.mark.filterwarnings("ignore:house_rules should be CambiaRulesConfig")
class TestConstructorRejectsOutOfRangeRules:
    """Every case here previously panicked or silently mis-dealt inside the
    shared library; the constructor returns -1 and the bridge raises."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"cards_per_player": 7},
            {"cards_per_player": 0, "initial_view_count": 0},
            {"initial_view_count": 5},
            {"use_jokers": 3},
            {"num_decks": 5},
            {"use_jokers": 3, "num_decks": 4},
            {"num_players": 9},
            {"num_players": 1},
        ],
    )
    def test_rejects(self, overrides):
        with pytest.raises(RuntimeError):
            GoEngine(seed=1, house_rules=_RawRules(**overrides))

    def test_rejects_a_deck_too_small_for_the_deal(self):
        # One rank, no jokers: 4 cards against a 2x2 deal plus the opening flip.
        with pytest.raises(RuntimeError):
            GoEngine(
                seed=1,
                house_rules=_RawRules(
                    deck_ranks=["A"],
                    use_jokers=0,
                    cards_per_player=2,
                    initial_view_count=1,
                ),
            )

    def test_accepts_the_same_deal_once_the_deck_covers_it(self):
        engine = GoEngine(
            seed=1,
            house_rules=_RawRules(
                deck_ranks=["A", "6"],
                use_jokers=0,
                cards_per_player=2,
                initial_view_count=1,
            ),
        )
        engine.close()

    def test_from_deck_rejects_a_starting_seat_off_the_table(self):
        with pytest.raises(RuntimeError):
            GoEngine.from_deck(
                list(range(54)), starting_player=2, house_rules=_RawRules()
            )

    def test_from_deck_rejects_a_deck_too_short_for_the_deal(self):
        # 2 seats x 4 cards plus the opening flip needs 9 cards.
        with pytest.raises(RuntimeError):
            GoEngine.from_deck(list(range(8)), house_rules=_RawRules())
