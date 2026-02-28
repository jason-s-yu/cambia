"""tests/test_multideck.py — Tests for NumDecks / multi-deck support."""
import pytest
from src.card import Card, create_standard_deck
from src.config import CambiaRulesConfig


# ---------------------------------------------------------------------------
# Unit tests: create_standard_deck
# ---------------------------------------------------------------------------


def test_single_deck_default():
    deck = create_standard_deck(include_jokers=2, num_decks=1)
    assert len(deck) == 54


def test_single_deck_no_jokers():
    deck = create_standard_deck(include_jokers=0, num_decks=1)
    assert len(deck) == 52


def test_two_decks_with_jokers():
    deck = create_standard_deck(include_jokers=2, num_decks=2)
    assert len(deck) == 108


def test_two_decks_no_jokers():
    deck = create_standard_deck(include_jokers=0, num_decks=2)
    assert len(deck) == 104


def test_four_decks_with_jokers():
    deck = create_standard_deck(include_jokers=2, num_decks=4)
    assert len(deck) == 216


def test_four_decks_no_jokers():
    deck = create_standard_deck(include_jokers=0, num_decks=4)
    assert len(deck) == 208


def test_duplicate_cards_in_multi_deck():
    """NumDecks=2 must produce exactly 2 copies of each non-joker card."""
    deck = create_standard_deck(include_jokers=0, num_decks=2)
    from collections import Counter
    counts = Counter((c.rank, c.suit) for c in deck)
    for card_key, count in counts.items():
        assert count == 2, f"Card {card_key} appears {count} times, expected 2"


def test_default_num_decks_unchanged():
    """create_standard_deck() without num_decks still returns 54 cards."""
    deck = create_standard_deck()
    assert len(deck) == 54


# ---------------------------------------------------------------------------
# CambiaRulesConfig interface
# ---------------------------------------------------------------------------


def test_cambia_rules_config_has_num_decks():
    config = CambiaRulesConfig()
    assert hasattr(config, "num_decks")
    assert config.num_decks == 1


def test_cambia_rules_config_num_decks_settable():
    config = CambiaRulesConfig(num_decks=2)
    assert config.num_decks == 2


# ---------------------------------------------------------------------------
# Python engine
# ---------------------------------------------------------------------------


def test_python_engine_single_deck_default():
    """Python engine default: 54-card deck."""
    from src.game.engine import CambiaGameState
    rules = CambiaRulesConfig(use_jokers=2, num_decks=1)
    g = CambiaGameState(house_rules=rules)
    assert len(g.stockpile) + sum(len(p.hand) for p in g.players) + len(g.discard_pile) == 54


def test_python_engine_two_decks():
    """Python engine NumDecks=2: 108-card pool."""
    from src.game.engine import CambiaGameState
    rules = CambiaRulesConfig(use_jokers=2, num_decks=2)
    g = CambiaGameState(house_rules=rules)
    total = len(g.stockpile) + sum(len(p.hand) for p in g.players) + len(g.discard_pile)
    assert total == 108


def test_python_engine_four_decks_no_jokers():
    """Python engine NumDecks=4, no jokers: 208-card pool."""
    from src.game.engine import CambiaGameState
    rules = CambiaRulesConfig(use_jokers=0, num_decks=4)
    g = CambiaGameState(house_rules=rules)
    total = len(g.stockpile) + sum(len(p.hand) for p in g.players) + len(g.discard_pile)
    assert total == 208


# ---------------------------------------------------------------------------
# Go FFI (bridge)
# ---------------------------------------------------------------------------


def test_go_engine_single_deck_stock_len():
    """Go engine NumDecks=1: StockLen after deal = 54 - 2*4 - 1 = 45."""
    pytest.importorskip("cffi")
    from src.ffi.bridge import GoEngine
    rules = CambiaRulesConfig(use_jokers=2, num_decks=1, cards_per_player=4)
    g = GoEngine(seed=42, house_rules=rules)
    stock_len = g._lib.cambia_game_stock_len(g._game_h)
    assert stock_len == 45  # 54 - 2*4 - 1 = 45
    g.close()


def test_go_engine_two_decks_stock_len():
    """Go engine NumDecks=2: StockLen after deal = 108 - 2*4 - 1 = 99."""
    pytest.importorskip("cffi")
    from src.ffi.bridge import GoEngine
    rules = CambiaRulesConfig(use_jokers=2, num_decks=2, cards_per_player=4)
    g = GoEngine(seed=42, house_rules=rules)
    stock_len = g._lib.cambia_game_stock_len(g._game_h)
    assert stock_len == 99  # 108 - 2*4 - 1 = 99
    g.close()


def test_go_engine_four_decks_stock_len():
    """Go engine NumDecks=4: StockLen after deal = 216 - 2*4 - 1 = 207."""
    pytest.importorskip("cffi")
    from src.ffi.bridge import GoEngine
    rules = CambiaRulesConfig(use_jokers=2, num_decks=4, cards_per_player=4)
    g = GoEngine(seed=42, house_rules=rules)
    stock_len = g._lib.cambia_game_stock_len(g._game_h)
    assert stock_len == 207  # 216 - 2*4 - 1 = 207
    g.close()


def test_go_engine_default_unchanged():
    """Go engine default (num_decks=1): StockLen after deal = 45."""
    pytest.importorskip("cffi")
    from src.ffi.bridge import GoEngine
    rules = CambiaRulesConfig()
    g = GoEngine(seed=42, house_rules=rules)
    stock_len = g._lib.cambia_game_stock_len(g._game_h)
    assert stock_len == 45  # 54 - 2*4 - 1 = 45
    g.close()
