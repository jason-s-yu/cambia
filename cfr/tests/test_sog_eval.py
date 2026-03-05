"""
tests/test_sog_eval.py

Tests for SoG eval harness: GoEngine.from_deck, card mapping, SoGAgentWrapper.
FFI-dependent tests are skipped if libcambia.so is not available.
"""

import numpy as np
import pytest

try:
    from src.ffi.bridge import GoEngine, python_card_to_go_index, extract_deck_from_python_game

    HAS_GO = True
except Exception:
    HAS_GO = False

skipgo = pytest.mark.skipif(not HAS_GO, reason="libcambia.so not available")


# ---------------------------------------------------------------------------
# Card mapping (no FFI required)
# ---------------------------------------------------------------------------


def test_python_card_to_go_index_ace_clubs():
    """Ace of Clubs → index 0 (C=0, rank A=0, 0*13+0=0)."""
    from src.card import Card
    card = Card(rank="A", suit="C")
    assert python_card_to_go_index(card) == 0


def test_python_card_to_go_index_king_spades():
    """King of Spades → index 51 (S=3, rank K=12, 3*13+12=51)."""
    from src.card import Card
    card = Card(rank="K", suit="S")
    assert python_card_to_go_index(card) == 51


def test_python_card_to_go_index_ace_hearts():
    """Ace of Hearts → index 26 (H=2, rank A=0, 2*13+0=26)."""
    from src.card import Card
    card = Card(rank="A", suit="H")
    assert python_card_to_go_index(card) == 26


def test_python_card_to_go_index_ten_diamonds():
    """Ten of Diamonds → index 22 (D=1, rank T=9, 1*13+9=22)."""
    from src.card import Card
    card = Card(rank="T", suit="D")
    assert python_card_to_go_index(card) == 22


def test_python_card_to_go_index_joker():
    """Joker → index 52."""
    from src.card import Card
    card = Card(rank="R")
    assert python_card_to_go_index(card) == 52


def test_card_index_coverage():
    """All 52 standard cards produce unique indices in [0, 51]."""
    from src.card import Card
    from src.constants import ALL_SUITS, ALL_RANKS_STR, JOKER_RANK_STR
    indices = set()
    for suit in ALL_SUITS:
        for rank in ALL_RANKS_STR:
            if rank == JOKER_RANK_STR:
                continue
            idx = python_card_to_go_index(Card(rank=rank, suit=suit))
            assert 0 <= idx <= 51, f"Index {idx} out of range for {rank}{suit}"
            indices.add(idx)
    assert len(indices) == 52, "Duplicate card indices found"


# ---------------------------------------------------------------------------
# GoEngine.from_deck (FFI required)
# ---------------------------------------------------------------------------


@skipgo
def test_from_deck_creates_valid_engine():
    """GoEngine.from_deck produces a non-terminal engine with correct stock len."""
    from src.card import Card, create_standard_deck
    import random

    # Build a 54-card deck and shuffle it
    deck = create_standard_deck(include_jokers=2)
    random.shuffle(deck)

    # Reconstruct as deal-order indices: 2P × 4 cards + 1 discard + 46 remaining
    # We'll set up the indices manually
    joker_count = 0

    def to_idx(card):
        nonlocal joker_count
        if card.rank == "R":
            idx = 52 + min(joker_count, 1)
            joker_count += 1
            return idx
        from src.ffi.bridge import SUIT_OFFSET, RANK_VALUE
        return SUIT_OFFSET[card.suit] * 13 + RANK_VALUE[card.rank]

    # deck order: p0c0, p1c0, p0c1, p1c1, p0c2, p1c2, p0c3, p1c3, discard, stock...
    deck_indices = [to_idx(c) for c in deck]

    with GoEngine.from_deck(deck_indices, starting_player=0) as eng:
        assert eng.handle >= 0
        assert not eng.is_terminal()
        # After dealing 8 cards + 1 discard from 54: 45 remaining
        assert eng.stock_len() == 45


@skipgo
def test_from_deck_legal_actions_nonempty():
    """GoEngine.from_deck has legal actions at turn start."""
    from src.card import create_standard_deck
    import random

    deck = create_standard_deck(include_jokers=2)
    random.shuffle(deck)

    joker_count = 0

    def to_idx(card):
        nonlocal joker_count
        if card.rank == "R":
            idx = 52 + min(joker_count, 1)
            joker_count += 1
            return idx
        from src.ffi.bridge import SUIT_OFFSET, RANK_VALUE
        return SUIT_OFFSET[card.suit] * 13 + RANK_VALUE[card.rank]

    deck_indices = [to_idx(c) for c in deck]

    with GoEngine.from_deck(deck_indices, starting_player=0) as eng:
        mask = eng.legal_actions_mask()
        assert mask.sum() > 0, "No legal actions at game start"


@skipgo
def test_extract_and_from_deck_roundtrip():
    """extract_deck_from_python_game + from_deck preserves stock_len."""
    from src.game.engine import CambiaGameState

    class _MinimalRules:
        use_jokers = 2
        cards_per_player = 4
        initial_view_count = 2
        num_decks = 1
        allowDrawFromDiscardPile = False
        allowReplaceAbilities = False
        allowOpponentSnapping = False
        snapRace = False
        penaltyDrawCount = 2
        lockCallerHand = True
        max_game_turns = 46
        cambia_allowed_round = 0
        num_players = 2

    game = CambiaGameState(house_rules=_MinimalRules())
    deck_indices, starting_player = extract_deck_from_python_game(game)

    # Total deck = 54, dealt 8 + 1 discard = 45 remaining
    assert len(deck_indices) == 54, f"Expected 54 deck indices, got {len(deck_indices)}"

    with GoEngine.from_deck(deck_indices, starting_player) as eng:
        assert eng.stock_len() == len(game.stockpile), (
            f"Go stockpile {eng.stock_len()} != Python stockpile {len(game.stockpile)}"
        )
        assert not eng.is_terminal()


@skipgo
def test_from_deck_action_applies():
    """Applying the first legal action to a from_deck engine succeeds."""
    from src.card import create_standard_deck
    import random

    deck = create_standard_deck(include_jokers=2)
    random.shuffle(deck)

    joker_count = 0

    def to_idx(card):
        nonlocal joker_count
        if card.rank == "R":
            idx = 52 + min(joker_count, 1)
            joker_count += 1
            return idx
        from src.ffi.bridge import SUIT_OFFSET, RANK_VALUE
        return SUIT_OFFSET[card.suit] * 13 + RANK_VALUE[card.rank]

    deck_indices = [to_idx(c) for c in deck]

    with GoEngine.from_deck(deck_indices, starting_player=0) as eng:
        mask = eng.legal_actions_mask()
        legal_indices = np.where(mask)[0]
        assert len(legal_indices) > 0
        # Apply first legal action (should not raise)
        eng.apply_action(int(legal_indices[0]))


# ---------------------------------------------------------------------------
# SoGAgentWrapper (no FFI required for init)
# ---------------------------------------------------------------------------


def _make_config(eval_budget: int = 3):
    """Minimal mock Config for SoGAgentWrapper."""
    cfg = type("Config", (), {})()

    rules = type("CambiaRulesConfig", (), {})()
    rules.allowDrawFromDiscardPile = False
    rules.allowReplaceAbilities = False
    rules.snapRace = False
    rules.penaltyDrawCount = 2
    rules.use_jokers = 0
    rules.cards_per_player = 4
    rules.initial_view_count = 2
    rules.cambia_allowed_round = 0
    rules.allowOpponentSnapping = False
    rules.max_game_turns = 100
    rules.lockCallerHand = True
    rules.num_decks = 1
    rules.num_players = 2
    cfg.cambia_rules = rules

    agent_params = type("AgentParamsConfig", (), {})()
    agent_params.memory_level = 1
    agent_params.time_decay_turns = 10
    cfg.agent_params = agent_params

    agents_cfg = type("AgentsConfig", (), {})()
    agents_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_cambia_threshold = 5
    cfg.agents = agents_cfg

    deep_cfr = type("DeepCfrConfig", (), {})()
    deep_cfr.gtcfr_expansion_budget = eval_budget
    deep_cfr.gtcfr_cvpn_hidden_dim = 64
    deep_cfr.gtcfr_cvpn_num_blocks = 1
    deep_cfr.gtcfr_c_puct = 2.0
    deep_cfr.gtcfr_cfr_iters_per_expansion = 2
    deep_cfr.gtcfr_buffer_capacity = 100
    deep_cfr.gtcfr_value_loss_weight = 1.0
    deep_cfr.gtcfr_policy_loss_weight = 1.0
    deep_cfr.gtcfr_cvpn_learning_rate = 3e-4
    deep_cfr.gtcfr_games_per_epoch = 1
    deep_cfr.gtcfr_epochs = 1
    deep_cfr.batch_size = 4
    cfg.deep_cfr = deep_cfr

    return cfg


def test_sog_agent_wrapper_instantiates():
    """SoGAgentWrapper initializes without error."""
    from src.evaluate_agents import SoGAgentWrapper
    config = _make_config()
    wrapper = SoGAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")
    assert wrapper._cvpn is not None
    assert not wrapper._cvpn.training  # eval mode


def test_sog_agent_wrapper_choose_action_fallback():
    """choose_action falls back to random when no GoEngine (libcambia absent)."""
    from src.evaluate_agents import SoGAgentWrapper
    from src.constants import ActionDrawStockpile, ActionCallCambia

    config = _make_config()
    wrapper = SoGAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")
    # No initialize_state called → _go_engine is None → falls back to CVPN-only → random
    legal_actions = {ActionDrawStockpile(), ActionCallCambia()}
    action = wrapper.choose_action(game_state=None, legal_actions=legal_actions)
    assert action in legal_actions


def test_sog_inference_wrapper_instantiates():
    """SoGInferenceAgentWrapper is a GTCFRAgentWrapper subclass."""
    from src.evaluate_agents import SoGInferenceAgentWrapper, GTCFRAgentWrapper
    config = _make_config()
    wrapper = SoGInferenceAgentWrapper(player_id=1, config=config, checkpoint_path="", device="cpu")
    assert isinstance(wrapper, GTCFRAgentWrapper)
    assert wrapper._cvpn is not None
