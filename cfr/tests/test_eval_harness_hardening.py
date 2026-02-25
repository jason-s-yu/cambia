"""
test_eval_harness_hardening.py

Tests that the eval harness (run_evaluation) and baseline agents handle multi-game
reuse correctly. These tests would have caught the stale-memory bug in baseline agents:
if an agent fails to reinitialize between games, memory from game N bleeds into game N+1,
causing immediate Cambia calls (turn < 5) and heavily skewed win rates.

Three test categories:
  1. Cross-Game Eval Integration  — game-level stats remain valid over a full eval run
  2. Agent Strength Ordering      — canary assertions on known dominance relationships
  3. Multi-Step Memory Verification — unit-level memory correctness across games
"""

import pytest
from src.evaluate_agents import run_evaluation
from src.game.engine import CambiaGameState
from src.agents.baseline_agents import (
    ImperfectGreedyAgent,
    RandomNoCambiaAgent,
)
from src.config import load_config
from src.card import Card
from src.constants import ActionReplace

CONFIG_PATH = "runs/eppbs-2p/config.yaml"


# ---------------------------------------------------------------------------
# Category 1: Cross-Game Eval Integration
# ---------------------------------------------------------------------------


class TestCrossGameEvalIntegration:
    """Game statistics must remain valid over a full evaluation run.

    A stale-memory bug causes agents to carry hand knowledge from the previous
    game into the next, immediately calling Cambia on turn 1-3 and collapsing
    avg_game_turns. These tests catch that regression.
    """

    @pytest.mark.slow
    def test_game_turns_stable_across_eval(self):
        """avg_game_turns must be > 15 over a 100-game evaluation.

        Stale memory causes immediate Cambia calls (turns < 5 per game).
        With max_turns=46 and no stale-memory bug, avg should be well above 15.
        Failure means agents are terminating games immediately — a hallmark of the
        stale-memory bug where initial peek memory from a prior game is mistakenly
        reused, convincing the agent it already has a winning hand.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="imperfect_greedy",
            agent2_type="random_no_cambia",
            num_games=100,
            strategy_path=None,
        )
        avg_turns = results.stats.get("avg_game_turns", 0)
        assert avg_turns > 15, (
            f"avg_game_turns={avg_turns:.1f} is too low — agents may be carrying stale "
            f"memory across games, triggering immediate Cambia calls. "
            f"Expected >15 turns; stale-memory bug produces <5."
        )

    @pytest.mark.slow
    def test_seat_rotation_balanced(self):
        """Decisive outcomes must be non-zero over 200 games.

        Both agents should win at least some games. If one agent never wins
        (0 wins over all non-tie games), that agent is effectively broken —
        most likely due to stale memory causing irrational decisions on every game.

        Note: most games end in MaxTurnTies (~72%) because both agents play
        conservatively. The thresholds are conservative: P0 must win >5% of all
        scored games, and P1 must win at least 1 game out of 200.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="imperfect_greedy",
            agent2_type="random_no_cambia",
            num_games=200,
            strategy_path=None,
        )
        p0 = results.get("P0 Wins", 0)
        p1 = results.get("P1 Wins", 0)
        scored = p0 + p1 + results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        assert scored > 0, "No games scored — evaluation loop produced no results."

        p0_rate = p0 / scored
        assert p0_rate > 0.05, (
            f"P0 win rate {p0_rate:.2%} is below 5% — imperfect_greedy should win "
            f"at least some games vs random_no_cambia. Near-zero wins suggests the "
            f"agent is calling Cambia immediately (stale-memory bug)."
        )
        assert p1 >= 1, (
            f"P1 (random_no_cambia) won 0 games out of {scored} scored. "
            f"A working random agent should win occasionally. "
            f"Zero P1 wins may indicate P1 memory is corrupted from prior games."
        )


# ---------------------------------------------------------------------------
# Category 2: Agent Strength Ordering Invariant
# ---------------------------------------------------------------------------


class TestAgentStrengthOrdering:
    """Known dominance relationships among baseline agents.

    If any ordering breaks, agent logic is wrong. These act as canaries:
    breaking them after a code change signals a regression in agent behavior
    or in the eval harness itself.

    Win rates are computed over ALL scored games (including MaxTurnTies in denominator).
    Most games in conservative matchups end in MaxTurnTies, so the thresholds are
    calibrated against empirically measured steady-state win rates.
    """

    @pytest.mark.slow
    def test_greedy_dominates_random_no_cambia(self):
        """GreedyAgent (perfect info) must beat RandomNoCambiaAgent >80% of the time.

        Greedy has access to the true game state and makes optimal replace decisions.
        Against a fully random opponent, greedy calls Cambia aggressively and wins
        most decisive games. Empirical rate: ~88-90%.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="greedy",
            agent2_type="random_no_cambia",
            num_games=500,
            strategy_path=None,
        )
        p0 = results.get("P0 Wins", 0)
        scored = (
            p0
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
        )
        assert scored > 0, "No games scored."
        win_rate = p0 / scored
        assert win_rate > 0.80, (
            f"GreedyAgent win rate {win_rate:.2%} vs random_no_cambia is below 80%. "
            f"Greedy has perfect information and should dominate a random agent. "
            f"Empirical floor is ~88%. This may indicate a broken game loop, "
            f"agent initialization bug, or stale memory from a prior game."
        )

    @pytest.mark.slow
    def test_imperfect_beats_random_no_cambia(self):
        """ImperfectGreedyAgent must win >10% of scored games vs RandomNoCambiaAgent.

        Most games end in MaxTurnTies (~72%) because both agents play conservatively.
        When games do finish decisively, imperfect_greedy wins more often.
        Empirical P0 win rate (over all scored games): ~22-26%.
        The threshold of >10% is conservative but catches the stale-memory bug,
        which drops P0 wins to near zero by causing immediate Cambia calls every game.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="imperfect_greedy",
            agent2_type="random_no_cambia",
            num_games=500,
            strategy_path=None,
        )
        p0 = results.get("P0 Wins", 0)
        scored = (
            p0
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
        )
        assert scored > 0, "No games scored."
        win_rate = p0 / scored
        assert win_rate > 0.10, (
            f"ImperfectGreedyAgent win rate {win_rate:.2%} vs random_no_cambia is below 10%. "
            f"Empirical floor is ~22%. Near-zero P0 wins suggests stale memory causing "
            f"immediate Cambia calls every game (stale initial peek values)."
        )

    @pytest.mark.slow
    def test_imperfect_beats_random(self):
        """ImperfectGreedyAgent must beat vanilla RandomAgent >55% of the time.

        RandomAgent includes Cambia in its action set, causing early game termination
        frequently. This reduces the advantage of rational play compared to random_no_cambia.
        Empirical P0 win rate: ~68-72%.
        The threshold of >55% is conservative and catches the stale-memory bug.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="imperfect_greedy",
            agent2_type="random",
            num_games=500,
            strategy_path=None,
        )
        p0 = results.get("P0 Wins", 0)
        scored = (
            p0
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
        )
        assert scored > 0, "No games scored."
        win_rate = p0 / scored
        assert win_rate > 0.55, (
            f"ImperfectGreedyAgent win rate {win_rate:.2%} vs random is below 55%. "
            f"Empirical floor is ~68%. Even accounting for random Cambia calls, "
            f"imperfect_greedy should maintain a clear edge. This may indicate stale "
            f"memory or broken agent logic."
        )

    @pytest.mark.slow
    def test_memory_heuristic_beats_random_no_cambia(self):
        """MemoryHeuristicAgent must win >8% of scored games vs RandomNoCambiaAgent.

        Most games end in MaxTurnTies (~80%) because MemoryHeuristic plays very
        conservatively. Empirical P0 win rate: ~17%.
        The threshold of >8% catches the stale-memory bug while accounting for the
        high MaxTurnTie rate in this matchup.
        """
        results = run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="memory_heuristic",
            agent2_type="random_no_cambia",
            num_games=500,
            strategy_path=None,
        )
        p0 = results.get("P0 Wins", 0)
        scored = (
            p0
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
        )
        assert scored > 0, "No games scored."
        win_rate = p0 / scored
        assert win_rate > 0.08, (
            f"MemoryHeuristicAgent win rate {win_rate:.2%} vs random_no_cambia is below 8%. "
            f"Empirical floor is ~17%. Near-zero wins suggests stale memory corrupting "
            f"hand-value estimates and Cambia-call decisions."
        )


# ---------------------------------------------------------------------------
# Category 3: Multi-Step Memory Verification
# ---------------------------------------------------------------------------


class TestMultiStepMemory:
    """Unit-level verification that agent memory is correctly maintained.

    These tests manually step through games to inspect internal memory state.
    They verify:
      - Memory updates correctly when the agent replaces a card
      - Memory is fresh at the start of every game (no cross-game contamination)
      - Known memory slots match actual cards throughout a game
    """

    def _make_config(self):
        return load_config(CONFIG_PATH)

    def test_memory_updates_on_replace(self):
        """After agent performs Replace, own_memory for that slot must equal the new card value.

        If _update_memory_replace_own is not called, the memory slot keeps stale data
        from the previous card, causing incorrect Cambia threshold decisions.
        """
        config = self._make_config()
        agent = ImperfectGreedyAgent(player_id=0, config=config)
        opponent = RandomNoCambiaAgent(player_id=1, config=config)
        agents = [agent, opponent]

        for _attempt in range(50):
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            steps = 0
            while not game_state.is_terminal() and steps < 300:
                pid = game_state.get_acting_player()
                if pid == -1:
                    break
                legal = game_state.get_legal_actions()
                if not legal:
                    break
                action = agents[pid].choose_action(game_state, legal)
                if pid == 0 and isinstance(action, ActionReplace):
                    drawn_card = game_state.pending_action_data.get("drawn_card")
                    if drawn_card is not None:
                        replaced_slot = action.target_hand_index
                        replace_card_value = drawn_card.value
                        game_state.apply_action(action)
                        # Memory must be updated immediately after replace
                        mem_val = agent.own_memory.get(replaced_slot)
                        assert mem_val == replace_card_value, (
                            f"After Replace at slot {replaced_slot} with card value "
                            f"{replace_card_value}, agent.own_memory[{replaced_slot}]="
                            f"{mem_val}. _update_memory_replace_own was not called."
                        )
                        return  # Test passed
                else:
                    game_state.apply_action(action)
                steps += 1

        # Replace may not occur in conservative games — not a failure
        pytest.skip("No Replace action observed in 50 game attempts")

    def test_memory_correct_across_5_games(self):
        """In each of 5 sequential games, initial peek memory must match actual dealt cards.

        Verifies:
        1. Memory is re-initialized at the start of each new game (_needs_reinit check).
        2. Initial peek slots (slots 0..initial_view_count-1) reflect the new game's cards
           right after _init_memory runs (after the agent's first action).
        3. Memory does not carry stale values from a previous game into the next.

        Strategy: play one action, then check that peek slots match the game's actual
        dealt cards (which are stable for the first few turns before any Replace occurs).
        """
        config = self._make_config()
        agent = ImperfectGreedyAgent(player_id=0, config=config)
        opponent = RandomNoCambiaAgent(player_id=1, config=config)
        agents = [agent, opponent]
        peek_count = config.cambia_rules.initial_view_count

        for game_num in range(5):
            game_state = CambiaGameState(house_rules=config.cambia_rules)

            # Step until it's agent's (P0) turn, then check memory right after _init_memory
            init_checked = False
            steps = 0
            while not game_state.is_terminal() and steps < 500:
                p = game_state.get_acting_player()
                if p == -1:
                    break
                la = game_state.get_legal_actions()
                if not la:
                    break
                act = agents[p].choose_action(game_state, la)
                if p == 0 and not init_checked:
                    # Memory has just been initialized — check peek slots before applying
                    my_hand = game_state.get_player_hand(0)
                    for slot in range(min(peek_count, len(my_hand))):
                        actual_card = my_hand[slot]
                        if isinstance(actual_card, Card):
                            mem_val = agent.own_memory.get(slot)
                            assert mem_val == actual_card.value, (
                                f"Game {game_num + 1}: slot {slot} memory={mem_val} "
                                f"but actual card value={actual_card.value}. "
                                f"Memory not re-initialized for this game "
                                f"(possible cross-game contamination from game {game_num})."
                            )
                    init_checked = True
                game_state.apply_action(act)
                steps += 1

    def test_no_cross_game_contamination(self):
        """Memory state from game N must not appear in game N+1.

        After game 1 finishes, create a fresh game and verify that the initial peek
        slots reflect the new game's cards, not the previous game's values.
        If `_needs_reinit` or `_init_memory` is broken, game 2's memory will show
        game 1's card values at the peek slots.
        """
        config = self._make_config()
        agent = ImperfectGreedyAgent(player_id=0, config=config)
        opponent = RandomNoCambiaAgent(player_id=1, config=config)
        agents = [agent, opponent]

        # --- Game 1: play to completion ---
        # Keep a reference to game1 alive until after game2 is created.
        # This prevents Python from reusing game1's memory address for game2,
        # which would cause _needs_reinit (id-based check) to return False and
        # mask a real cross-game contamination bug.
        game1 = CambiaGameState(house_rules=config.cambia_rules)
        steps = 0
        while not game1.is_terminal() and steps < 500:
            p = game1.get_acting_player()
            if p == -1:
                break
            la = game1.get_legal_actions()
            if not la:
                break
            act = agents[p].choose_action(game1, la)
            game1.apply_action(act)
            steps += 1

        # Record the peek-slot memory values at end of game 1
        game1_peek_values = {
            slot: agent.own_memory.get(slot)
            for slot in range(config.cambia_rules.initial_view_count)
        }

        # --- Game 2: initialize on a fresh game state ---
        # game1 still referenced here — prevents id reuse.
        game2 = CambiaGameState(house_rules=config.cambia_rules)
        assert id(game2) != id(game1), (
            "game2 got the same id as game1 — id-based reinit check would be skipped. "
            "Keep game1 alive until after game2 is fully initialized."
        )

        # Step until agent (P0) gets to act in game2
        init_checked = False
        steps2 = 0
        while not game2.is_terminal() and steps2 < 100:
            p2 = game2.get_acting_player()
            if p2 == -1:
                break
            la2 = game2.get_legal_actions()
            if not la2:
                break
            act2 = agents[p2].choose_action(game2, la2)
            if p2 == 0 and not init_checked:
                # Memory has just been re-initialized for game2 — check peek slots
                my_hand2 = game2.get_player_hand(0)
                peek_count = config.cambia_rules.initial_view_count
                for slot in range(min(peek_count, len(my_hand2))):
                    actual_card = my_hand2[slot]
                    if isinstance(actual_card, Card):
                        mem_val = agent.own_memory.get(slot)
                        assert mem_val == actual_card.value, (
                            f"Game 2 slot {slot}: memory={mem_val}, actual={actual_card.value}. "
                            f"Game 1 memory at this slot was {game1_peek_values.get(slot)}. "
                            f"Cross-game contamination detected — memory was not re-initialized "
                            f"for game 2 (stale values from game 1 persist)."
                        )
                init_checked = True
                break  # Checked — don't need to continue
            game2.apply_action(act2)
            steps2 += 1

        # Ensure we found a turn where agent acted
        assert init_checked, (
            "Agent (P0) never got to act in game 2's first 100 steps — "
            "could not verify memory re-initialization."
        )

        # Keep game1 alive until here to guarantee no id reuse above
        _ = id(game1)
