"""Tests for baseline agents: RandomAgent, GreedyAgent, ImperfectGreedyAgent,
MemoryHeuristicAgent, AggressiveSnapAgent."""

import pytest
from typing import Set, Optional
from dataclasses import dataclass, field

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import (
    BaseAgent,
    RandomAgent,
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    UNKNOWN_CARD_EXPECTED_VALUE,
)
from src.constants import (
    GameAction,
    ActionDrawStockpile,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
)
from src.card import Card


# ---------------------------------------------------------------------------
# Minimal config stubs (avoid importing yaml-dependent Config)
# ---------------------------------------------------------------------------


@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(default_factory=_GreedyAgentConfig)


@dataclass
class _CambiaRulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 300


@dataclass
class _TestConfig:
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)
    cambia_rules: _CambiaRulesConfig = field(default_factory=_CambiaRulesConfig)


def make_config(**kwargs) -> _TestConfig:
    return _TestConfig(**kwargs)


# Use the real CambiaRulesConfig from engine (already a dataclass)
# CambiaGameState uses CambiaRulesConfig from src.config, but the stub in conftest
# provides a compatible class. We just need our agent config wrapper.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> _TestConfig:
    """Config with default settings."""
    return make_config()


def _make_rules(**kwargs):
    """Create a CambiaRulesConfig from the stub (conftest-injected) module."""
    from src.config import CambiaRulesConfig as _RC
    # The stub class may not accept kwargs; fall back to setting attrs manually.
    try:
        obj = _RC(**kwargs)
    except TypeError:
        obj = _RC()
        for k, v in kwargs.items():
            setattr(obj, k, v)
    return obj


def make_game(rules=None) -> CambiaGameState:
    """Create a fresh game state. Uses stub CambiaRulesConfig if none given."""
    if rules is None:
        rules = _make_rules()
    return CambiaGameState(house_rules=rules)


ALL_AGENT_CLASSES = [
    RandomAgent,
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
]


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestAgentInitialization:
    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_init_player_0(self, AgentClass, default_config):
        agent = AgentClass(player_id=0, config=default_config)
        assert agent.player_id == 0
        assert agent.opponent_id == 1

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_init_player_1(self, AgentClass, default_config):
        agent = AgentClass(player_id=1, config=default_config)
        assert agent.player_id == 1
        assert agent.opponent_id == 0

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_is_base_agent(self, AgentClass, default_config):
        agent = AgentClass(player_id=0, config=default_config)
        assert isinstance(agent, BaseAgent)


# ---------------------------------------------------------------------------
# Empty legal actions tests
# ---------------------------------------------------------------------------


class TestEmptyLegalActions:
    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_raises_on_empty_legal_actions(self, AgentClass, default_config):
        agent = AgentClass(player_id=0, config=default_config)
        game = make_game()
        with pytest.raises((ValueError, Exception)):
            agent.choose_action(game, set())


# ---------------------------------------------------------------------------
# Legal action validity tests (50+ random game states)
# ---------------------------------------------------------------------------


def run_agent_through_game(agent: BaseAgent, num_turns: int = 50) -> int:
    """
    Run an agent through up to num_turns of a game, returning the number of
    valid actions taken. Raises on invalid action choices.
    """
    rules = _make_rules(max_game_turns=num_turns)
    game = make_game(rules)
    turns_taken = 0

    while not game.is_terminal() and turns_taken < num_turns:
        acting = game.get_acting_player()
        if acting == -1:
            break

        legal_actions = game.get_legal_actions()
        if not legal_actions:
            break

        if acting != agent.player_id:
            # Opponent's turn: use random agent
            import random
            action = random.choice(list(legal_actions))
        else:
            action = agent.choose_action(game, legal_actions)
            # Validate action is in legal set
            assert action in legal_actions, (
                f"Agent {type(agent).__name__} chose illegal action {action} "
                f"from legal set {legal_actions}"
            )

        game.apply_action(action)
        turns_taken += 1

    return turns_taken


class TestLegalActionValidity:
    """Each agent must always select a legal action across varied game states."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_agent_legal_actions(self, seed, default_config):
        import random
        random.seed(seed)
        agent = RandomAgent(player_id=0, config=default_config)
        turns = run_agent_through_game(agent, num_turns=60)
        assert turns > 0

    @pytest.mark.parametrize("seed", range(20))
    def test_greedy_agent_legal_actions(self, seed, default_config):
        import random
        random.seed(seed)
        agent = GreedyAgent(player_id=0, config=default_config)
        turns = run_agent_through_game(agent, num_turns=60)
        assert turns > 0

    @pytest.mark.parametrize("seed", range(20))
    def test_imperfect_greedy_legal_actions(self, seed, default_config):
        import random
        random.seed(seed)
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        turns = run_agent_through_game(agent, num_turns=60)
        assert turns > 0

    @pytest.mark.parametrize("seed", range(20))
    def test_memory_heuristic_legal_actions(self, seed, default_config):
        import random
        random.seed(seed)
        agent = MemoryHeuristicAgent(player_id=0, config=default_config)
        turns = run_agent_through_game(agent, num_turns=60)
        assert turns > 0

    @pytest.mark.parametrize("seed", range(20))
    def test_aggressive_snap_legal_actions(self, seed, default_config):
        import random
        random.seed(seed)
        agent = AggressiveSnapAgent(player_id=0, config=default_config)
        turns = run_agent_through_game(agent, num_turns=60)
        assert turns > 0


# ---------------------------------------------------------------------------
# No crash tests across varied game states
# ---------------------------------------------------------------------------


def run_full_games(AgentClass, num_games: int = 5, max_turns: int = 200):
    """Run multiple complete games with the agent as P0, random as P1."""
    import random
    config = make_config()
    for seed in range(num_games):
        random.seed(seed * 17 + 3)
        agent = AgentClass(player_id=0, config=config)
        rules = _make_rules(max_game_turns=max_turns)
        game = make_game(rules)
        turns = 0
        while not game.is_terminal() and turns < max_turns:
            acting = game.get_acting_player()
            if acting == -1:
                break
            legal = game.get_legal_actions()
            if not legal:
                break
            if acting == 0:
                action = agent.choose_action(game, legal)
            else:
                action = random.choice(list(legal))
            game.apply_action(action)
            turns += 1


class TestNoCrash:
    """Agents should not crash across a variety of game scenarios."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_no_crash_full_games(self, AgentClass, default_config):
        run_full_games(AgentClass, num_games=10, max_turns=200)


# ---------------------------------------------------------------------------
# Memory model tests (imperfect agents)
# ---------------------------------------------------------------------------


class TestImperfectMemoryMixin:
    """Test the internal memory model of imperfect info agents."""

    def test_initial_memory_peeks_first_two(self, default_config):
        """After init, own_memory should have first 2 cards known, rest unknown."""
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        # initial_view_count defaults to 2
        my_hand = game.get_player_hand(0)
        for i in range(len(my_hand)):
            if i < 2:
                assert agent.own_memory[i] is not None, f"Slot {i} should be known"
            else:
                assert agent.own_memory[i] is None, f"Slot {i} should be unknown"

    def test_estimate_hand_value_with_known_cards(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        # Force known values
        agent.own_memory = {0: 5, 1: 3, 2: None, 3: None}
        estimated = agent._estimate_own_hand_value()
        expected = 5 + 3 + 2 * UNKNOWN_CARD_EXPECTED_VALUE
        assert abs(estimated - expected) < 0.01

    def test_find_highest_known_slot(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        agent.own_memory = {0: 5, 1: 12, 2: None, 3: 2}
        highest = agent._find_highest_known_own_slot()
        assert highest == 1

    def test_find_unknown_slot(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        agent.own_memory = {0: 5, 1: 12, 2: None, 3: 2}
        unknown = agent._find_unknown_own_slot()
        assert unknown == 2

    def test_mark_own_unknown(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        agent.own_memory[0] = 7
        agent._mark_own_unknown(0)
        assert agent.own_memory[0] is None

    def test_update_memory_replace_own(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        new_card = Card(rank="3", suit="H")
        agent._update_memory_replace_own(0, new_card)
        assert agent.own_memory[0] == 3

    def test_all_unknown_estimate(self, default_config):
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        agent.own_memory = {0: None, 1: None, 2: None, 3: None}
        estimated = agent._estimate_own_hand_value()
        assert abs(estimated - 4 * UNKNOWN_CARD_EXPECTED_VALUE) < 0.01


# ---------------------------------------------------------------------------
# Snap phase tests
# ---------------------------------------------------------------------------


class TestSnapPhaseHandling:
    """Agents should handle snap phase without crashing and return legal actions."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_handles_snap_phase_without_crash(self, AgentClass, default_config):
        """Run games with snap enabled and ensure no crashes."""
        import random
        random.seed(42)
        # Enable snapping
        rules = _make_rules(
            allowOpponentSnapping=True,
            snapRace=False,
            max_game_turns=150,
        )
        agent = AgentClass(player_id=0, config=default_config)
        config = make_config()

        game = make_game(rules)
        turns = 0
        while not game.is_terminal() and turns < 150:
            acting = game.get_acting_player()
            if acting == -1:
                break
            legal = game.get_legal_actions()
            if not legal:
                break

            if acting == 0:
                action = agent.choose_action(game, legal)
                assert action in legal, (
                    f"{AgentClass.__name__} chose illegal action {action}"
                )
            else:
                action = random.choice(list(legal))

            game.apply_action(action)
            turns += 1


# ---------------------------------------------------------------------------
# Ability phase tests
# ---------------------------------------------------------------------------


class TestAbilityPhaseHandling:
    """Agents should handle all ability phases without crashing."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_handles_ability_phases(self, AgentClass, default_config):
        """Run multiple games and ensure ability phases are handled."""
        import random
        config = make_config()
        for seed in range(5):
            random.seed(seed * 31)
            agent = AgentClass(player_id=0, config=config)
            rules = _make_rules(
                allowReplaceAbilities=True,
                max_game_turns=200,
            )
            game = make_game(rules)
            turns = 0
            while not game.is_terminal() and turns < 200:
                acting = game.get_acting_player()
                if acting == -1:
                    break
                legal = game.get_legal_actions()
                if not legal:
                    break
                if acting == 0:
                    action = agent.choose_action(game, legal)
                    assert action in legal
                else:
                    action = random.choice(list(legal))
                game.apply_action(action)
                turns += 1


# ---------------------------------------------------------------------------
# Specific behavior tests
# ---------------------------------------------------------------------------


class TestAgentBehaviors:
    """Test specific decision-making behaviors."""

    def test_random_agent_is_nondeterministic(self, default_config):
        """RandomAgent should occasionally make different choices."""
        import random
        agent = RandomAgent(player_id=0, config=default_config)
        game = make_game()
        legal = game.get_legal_actions()
        choices = set()
        for _ in range(50):
            choices.add(agent.choose_action(game, legal))
        # With 2+ legal actions, we expect multiple choices
        if len(legal) > 1:
            assert len(choices) > 1

    def test_greedy_agent_calls_cambia_with_low_hand(self, default_config):
        """GreedyAgent should call Cambia when hand value is below threshold."""
        from src.card import Card
        agent = GreedyAgent(player_id=0, config=default_config)
        game = make_game()

        # Manually set player 0's hand to all low cards
        game.players[0].hand = [
            Card(rank="A", suit="H"),  # value 1
            Card(rank="A", suit="D"),  # value 1
            Card(rank="2", suit="H"),  # value 2
            Card(rank="A", suit="S"),  # value 1
        ]
        # total = 5, threshold = 5 → should call Cambia
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        action = agent.choose_action(game, legal)
        assert action == ActionCallCambia()

    def test_imperfect_greedy_calls_cambia_based_on_estimate(self, default_config):
        """ImperfectGreedyAgent should call Cambia when estimated hand is low."""
        agent = ImperfectGreedyAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        # Force low estimated hand value
        agent.own_memory = {0: 1, 1: 1, 2: 1, 3: 1}  # total = 4
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        action = agent.choose_action(game, legal)
        assert action == ActionCallCambia()

    def test_aggressive_snap_calls_cambia_with_small_hand(self, default_config):
        """AggressiveSnapAgent should call Cambia when hand has <= 2 cards."""
        agent = AggressiveSnapAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        # Force hand to 2 cards
        agent.own_memory = {0: 10, 1: 8}
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        action = agent.choose_action(game, legal)
        assert action == ActionCallCambia()

    def test_imperfect_agents_dont_snap_unknown_own_cards(self, default_config):
        """Imperfect agents should not snap own cards they haven't seen."""
        for AgentClass in [ImperfectGreedyAgent, MemoryHeuristicAgent, AggressiveSnapAgent]:
            agent = AgentClass(player_id=0, config=default_config)
            game = make_game()
            agent._ensure_initialized(game)

            # All own cards unknown — no snap should happen
            agent.own_memory = {0: None, 1: None, 2: None, 3: None}

            snap_own = ActionSnapOwn(own_card_hand_index=0)
            pass_snap = ActionPassSnap()
            legal = {snap_own, pass_snap}

            # Simulate snap phase
            game.snap_phase_active = True
            action = agent.choose_action(game, legal)
            # Should pass since card at slot 0 is unknown
            assert action == pass_snap, (
                f"{AgentClass.__name__} should pass snap for unknown card, chose {action}"
            )
            game.snap_phase_active = False

    def test_memory_heuristic_always_draws_stockpile(self, default_config):
        """MemoryHeuristicAgent should prefer stockpile over discard."""
        agent = MemoryHeuristicAgent(player_id=0, config=default_config)
        game = make_game()
        # Give a discard action too
        from src.constants import ActionDrawDiscard
        legal = {ActionDrawStockpile(), ActionDrawDiscard()}
        action = agent.choose_action(game, legal)
        assert action == ActionDrawStockpile()

    def test_aggressive_snap_snaps_known_opponent_card(self, default_config):
        """AggressiveSnapAgent should snap opponent cards it knows match discard."""
        from src.card import Card
        agent = AggressiveSnapAgent(player_id=0, config=default_config)
        game = make_game()
        agent._ensure_initialized(game)

        # Set discard top to a 5H
        discard_card = Card(rank="5", suit="H")
        game.discard_pile = [discard_card]

        # Set snap discard card
        game.snap_discarded_card = discard_card

        # Agent knows opponent slot 2 has a 5
        agent.opponent_memory = {0: None, 1: None, 2: 5, 3: None}

        snap_opp = ActionSnapOpponent(opponent_target_hand_index=2)
        pass_snap = ActionPassSnap()
        legal = {snap_opp, pass_snap}

        game.snap_phase_active = True
        action = agent.choose_action(game, legal)
        assert action == snap_opp, (
            f"AggressiveSnapAgent should snap opponent's known card, chose {action}"
        )
        game.snap_phase_active = False
