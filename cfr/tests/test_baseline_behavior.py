"""Behavioral tests for baseline agents.

These tests verify that agents make expected decisions in constructed scenarios,
not just that they return legal actions. They catch:
1. State reset bugs: agents must reinitialize for new games
2. Decision quality: Cambia calling, ability usage, snap correctness
3. Memory model: agents must track what they've seen correctly
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.game.engine import CambiaGameState
from src.config import load_config
from src.constants import (
    ActionCallCambia,
    ActionDrawStockpile,
    ActionDiscard,
    ActionReplace,
    ActionSnapOwn,
    ActionPassSnap,
)
from src.agents.baseline_agents import (
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    HumanPlayerAgent,
    RandomAgent,
    RandomNoCambiaAgent,
    GreedyAgent,
)


@pytest.fixture
def config():
    return load_config("runs/eppbs-2p/config.yaml")


def play_one_game(agent, config, max_steps=100):
    """Play one game and return (steps, terminal, cambia_step).

    Uses RandomNoCambiaAgent as opponent to avoid random immediate Cambia calls.
    """
    gs = CambiaGameState(house_rules=config.cambia_rules)
    other = RandomNoCambiaAgent(1 - agent.player_id, config)
    agents = [None, None]
    agents[agent.player_id] = agent
    agents[1 - agent.player_id] = other

    cambia_step = None
    for step in range(max_steps):
        if gs.is_terminal():
            return step, True, cambia_step
        pid = gs.get_acting_player()
        legal = gs.get_legal_actions()
        action = agents[pid].choose_action(gs, legal)
        if isinstance(action, ActionCallCambia):
            cambia_step = step
        gs.apply_action(action)
    return max_steps, gs.is_terminal(), cambia_step


class TestAgentStateReset:
    """Verify agents properly reinitialize for new game states."""

    @pytest.mark.parametrize("AgentClass", [
        ImperfectGreedyAgent, MemoryHeuristicAgent,
        AggressiveSnapAgent, HumanPlayerAgent,
    ])
    def test_memory_resets_on_new_game(self, config, AgentClass):
        """Agent must re-initialize memory when game_state changes."""
        agent = AgentClass(0, config)

        # Game 1: initialize and play a few steps
        gs1 = CambiaGameState(house_rules=config.cambia_rules)
        legal1 = gs1.get_legal_actions()
        agent.choose_action(gs1, legal1)

        # Record game 1 memory
        game1_memory = dict(agent.own_memory)

        # Game 2: completely new game state
        gs2 = CambiaGameState(house_rules=config.cambia_rules)
        legal2 = gs2.get_legal_actions()
        agent.choose_action(gs2, legal2)

        # Memory must reflect game 2's cards, not game 1's
        game2_memory = dict(agent.own_memory)

        # Memories should differ unless extremely unlikely identical deal
        # But more importantly, the known slots should match actual cards
        hand = gs2.get_player_hand(0)
        for slot, val in game2_memory.items():
            if val is not None:
                assert val == hand[slot].value, (
                    f"Agent memory at slot {slot} is {val} but actual card is "
                    f"{hand[slot].value}. Memory not reset for new game!"
                )

    @pytest.mark.parametrize("AgentClass", [
        ImperfectGreedyAgent, MemoryHeuristicAgent,
        AggressiveSnapAgent, HumanPlayerAgent,
    ])
    def test_multi_game_memory_correctness(self, config, AgentClass):
        """Agent memory must match actual cards in EVERY new game, not just the first."""
        agent = AgentClass(0, config)

        for game_num in range(5):
            gs = CambiaGameState(house_rules=config.cambia_rules)
            legal = gs.get_legal_actions()
            agent.choose_action(gs, legal)

            # After first action in each game, known slots must match actual hand
            hand = gs.get_player_hand(0)
            for slot, val in agent.own_memory.items():
                if val is not None:
                    assert val == hand[slot].value, (
                        f"Game {game_num}: agent memory[{slot}]={val} but "
                        f"actual card is {hand[slot]}. Memory not reset!"
                    )

    @pytest.mark.parametrize("AgentClass", [
        ImperfectGreedyAgent, MemoryHeuristicAgent,
        AggressiveSnapAgent, HumanPlayerAgent,
    ])
    def test_game_id_tracking(self, config, AgentClass):
        """Agent must track game identity and reinit on new game state."""
        agent = AgentClass(0, config)

        gs1 = CambiaGameState(house_rules=config.cambia_rules)
        agent.choose_action(gs1, gs1.get_legal_actions())
        id1 = agent._last_game_id

        gs2 = CambiaGameState(house_rules=config.cambia_rules)
        agent.choose_action(gs2, gs2.get_legal_actions())
        id2 = agent._last_game_id

        assert id1 != id2, "Agent _last_game_id should change between different games"
        assert id2 == id(gs2), "Agent _last_game_id should match current game state"


class TestCambiaDecisionQuality:
    """Verify Cambia is called at appropriate times."""

    @pytest.mark.parametrize("AgentClass", [
        ImperfectGreedyAgent, MemoryHeuristicAgent,
        AggressiveSnapAgent, HumanPlayerAgent,
    ])
    def test_no_cambia_with_only_initial_peek(self, config, AgentClass):
        """Agent should not call Cambia knowing only 2 of 4 cards (initial peek)."""
        agent = AgentClass(0, config)
        gs = CambiaGameState(house_rules=config.cambia_rules)

        # Initialize the agent's memory
        legal = gs.get_legal_actions()
        assert ActionCallCambia() in legal, "Cambia should be legal at game start"

        # Agent should NOT call Cambia on first action (only knows 2/4 cards)
        action = agent.choose_action(gs, legal)
        assert not isinstance(action, ActionCallCambia), (
            f"{AgentClass.__name__} called Cambia on first action with only "
            f"initial peek (2/4 cards known). Memory: {agent.own_memory}"
        )

    def test_cambia_with_all_low_known(self, config):
        """If agent somehow knows all 4 cards are low, it should call Cambia."""
        agent = ImperfectGreedyAgent(0, config)
        gs = CambiaGameState(house_rules=config.cambia_rules)

        # Force-initialize and manually set memory to all low cards
        agent._ensure_initialized(gs)
        for slot in agent.own_memory:
            agent.own_memory[slot] = 1  # All Aces
            agent.own_rank_memory[slot] = "A"

        legal = gs.get_legal_actions()
        if ActionCallCambia() in legal:
            action = agent.choose_action(gs, legal)
            assert isinstance(action, ActionCallCambia), (
                "Agent should call Cambia when all 4 cards are known to be Aces "
                f"(total=4, threshold=5). Chose: {action}"
            )


class TestAbilityUsage:
    """Verify agents actually use card abilities."""

    def test_imperfect_greedy_uses_abilities(self, config):
        """ImperfectGreedyAgent should use abilities when available."""
        # Play 50 games and check that abilities are used at least once
        agent = ImperfectGreedyAgent(0, config)
        ability_used = False

        for _ in range(50):
            gs = CambiaGameState(house_rules=config.cambia_rules)
            for step in range(100):
                if gs.is_terminal():
                    break
                pid = gs.get_acting_player()
                legal = gs.get_legal_actions()
                if pid == 0:
                    action = agent.choose_action(gs, legal)
                    if isinstance(action, ActionDiscard) and action.use_ability:
                        ability_used = True
                        break
                else:
                    # Opponent just draws and discards
                    if ActionDrawStockpile() in legal:
                        gs.apply_action(ActionDrawStockpile())
                        continue
                    gs.apply_action(next(iter(legal)))
                    continue
                gs.apply_action(action)
            if ability_used:
                break

        assert ability_used, (
            "ImperfectGreedyAgent never used abilities in 50 games. "
            "Cards 7/8/9/10/K should trigger ability usage."
        )


class TestEvalHarnessAgentReset:
    """Integration test: verify the eval harness properly handles agent reuse."""

    def test_eval_multi_game_consistency(self, config):
        """run_evaluation results should be consistent regardless of game count."""
        from src.evaluate_agents import run_evaluation

        # Run 100 games
        c1 = run_evaluation(
            config_path="runs/eppbs-2p/config.yaml",
            agent1_type="imperfect_greedy",
            agent2_type="random_no_cambia",
            num_games=100,
            strategy_path=None, checkpoint_path=None, device="cpu",
        )
        stats1 = getattr(c1, "stats", {})
        avg_turns_100 = stats1.get("avg_game_turns", 0)

        # Games against random_no_cambia should have substantial length
        # (random_no_cambia never calls Cambia, so imperfect_greedy controls timing)
        assert avg_turns_100 > 15, (
            f"Average game turns = {avg_turns_100:.1f} in 100 games vs random_no_cambia. "
            f"Expected >15 turns. Agents may not be resetting between games."
        )

    def test_win_rate_vs_random_no_cambia(self, config):
        """Imperfect agents should clearly beat random_no_cambia (>60%)."""
        from src.evaluate_agents import run_evaluation

        c = run_evaluation(
            config_path="runs/eppbs-2p/config.yaml",
            agent1_type="imperfect_greedy",
            agent2_type="random_no_cambia",
            num_games=500,
            strategy_path=None, checkpoint_path=None, device="cpu",
        )
        decided = c.get("P0 Wins", 0) + c.get("P1 Wins", 0)
        wr = c.get("P0 Wins", 0) / decided if decided else 0
        # Imperfect greedy should dominate random_no_cambia (>60%)
        # If it's near 50%, something is wrong with agent behavior
        assert wr > 0.60, (
            f"ImperfectGreedyAgent only wins {wr*100:.1f}% vs random_no_cambia. "
            f"Expected >60%. Agent may not be gaining information advantage."
        )
