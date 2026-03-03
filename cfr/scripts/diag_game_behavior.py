#!/usr/bin/env python3
"""
Behavioral profiling diagnostic: plays thousands of games and logs every action,
per-turn hand scores, ability utilization, Cambia timing, and slot replacement
preferences. Compares a trained Deep CFR agent against a random_no_cambia baseline,
both playing as P0 vs imperfect_greedy as P1.

Usage:
    cd cfr && python scripts/diag_game_behavior.py
"""

import sys
import os
import copy
import random
import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure cfr/src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.game.engine import CambiaGameState
from src.evaluate_agents import (
    DeepCFRAgentWrapper,
    NeuralAgentWrapper,
    get_agent,
    AGENT_REGISTRY,
)
from src.agents.baseline_agents import BaseAgent
from src.agent_state import AgentObservation
from src.constants import (
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
    GameAction,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    JACK,
    QUEEN,
    KING,
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

def classify_action(action: GameAction) -> str:
    """Return a short category string for any game action."""
    name = type(action).__name__
    mapping = {
        "ActionDrawStockpile": "draw_stock",
        "ActionDrawDiscard": "draw_discard",
        "ActionCallCambia": "cambia",
        "ActionPassSnap": "pass_snap",
        "ActionSnapOwn": "snap_own",
        "ActionSnapOpponent": "snap_opp",
        "ActionSnapOpponentMove": "snap_move",
    }
    if name in mapping:
        return mapping[name]
    if name == "ActionReplace":
        slot = action.target_hand_index
        return f"replace_{slot}"
    if name == "ActionDiscard":
        return "discard_ability" if action.use_ability else "discard_no_ability"
    if name == "ActionAbilityPeekOwnSelect":
        return "peek_own"
    if name == "ActionAbilityPeekOtherSelect":
        return "peek_other"
    if name == "ActionAbilityBlindSwapSelect":
        return "blind_swap"
    if name == "ActionAbilityKingLookSelect":
        return "king_look"
    if name == "ActionAbilityKingSwapDecision":
        return "king_swap_decision"
    return name


def hand_score(game_state: CambiaGameState, player_id: int) -> int:
    """Sum of card values in a player's hand."""
    return sum(c.value for c in game_state.players[player_id].hand)


def hand_size(game_state: CambiaGameState, player_id: int) -> int:
    return len(game_state.players[player_id].hand)


# ---------------------------------------------------------------------------
# Profile data structures
# ---------------------------------------------------------------------------

class BehaviorProfile:
    """Accumulates behavioral statistics across many games."""

    def __init__(self, label: str):
        self.label = label
        self.action_counts: Counter = Counter()        # action_category -> count
        self.p0_score_by_turn: Dict[int, List[int]] = defaultdict(list)
        self.p1_score_by_turn: Dict[int, List[int]] = defaultdict(list)
        self.p0_hand_size_by_turn: Dict[int, List[int]] = defaultdict(list)
        self.cambia_turns: List[int] = []              # turn when P0 called Cambia
        self.replace_slots: Counter = Counter()        # slot index -> count
        # Ability utilization: how often P0 uses vs skips ability
        self.ability_opportunities: int = 0            # times P0 discarded an ability card
        self.ability_used: int = 0                     # times P0 chose use_ability=True
        # Per-rank ability tracking
        self.ability_opp_by_rank: Counter = Counter()  # rank -> opportunities
        self.ability_use_by_rank: Counter = Counter()  # rank -> used
        self.wins: int = 0
        self.losses: int = 0
        self.ties: int = 0
        self.errors: int = 0
        self.total_turns: List[int] = []
        self.final_score_p0: List[int] = []
        self.final_score_p1: List[int] = []
        # Track drawn card rank when discard decision happens
        self._pending_drawn_rank: Optional[str] = None


# ---------------------------------------------------------------------------
# Game loop with full action logging
# ---------------------------------------------------------------------------

def play_games(
    p0_type: str,
    p1_type: str,
    config_path: str,
    checkpoint_path: Optional[str],
    num_games: int,
    profile: BehaviorProfile,
):
    """Play num_games and populate profile."""
    config = load_config(config_path)

    for game_idx in range(num_games):
        try:
            _play_one_game(p0_type, p1_type, config, checkpoint_path, profile, game_idx)
        except Exception as e:
            logger.error("Game %d error: %s", game_idx, e)
            profile.errors += 1


def _play_one_game(
    p0_type: str,
    p1_type: str,
    config,
    checkpoint_path: Optional[str],
    profile: BehaviorProfile,
    game_idx: int,
):
    # Create agents
    p0_kwargs = {}
    if p0_type in ("deep_cfr", "escher", "sd_cfr"):
        p0_kwargs = {"checkpoint_path": checkpoint_path, "device": "cpu"}
    p0_agent = get_agent(p0_type, player_id=0, config=config, **p0_kwargs)
    p1_agent = get_agent(p1_type, player_id=1, config=config)
    agents = [p0_agent, p1_agent]

    game_state = CambiaGameState(house_rules=config.cambia_rules)

    # Initialize stateful agents
    for agent in agents:
        if isinstance(agent, NeuralAgentWrapper):
            agent.initialize_state(game_state)

    max_turns = config.cambia_rules.max_game_turns if config.cambia_rules.max_game_turns > 0 else 500
    turn = 0
    game_turn_number = 0  # coarse turn (one per P0 start-of-turn)

    # Record initial scores
    profile.p0_score_by_turn[0].append(hand_score(game_state, 0))
    profile.p1_score_by_turn[0].append(hand_score(game_state, 1))
    profile.p0_hand_size_by_turn[0].append(hand_size(game_state, 0))

    while not game_state.is_terminal() and turn < max_turns:
        turn += 1
        acting_player = game_state.get_acting_player()
        if acting_player == -1:
            break

        legal_actions = game_state.get_legal_actions()
        if not legal_actions:
            if game_state.is_terminal():
                break
            profile.errors += 1
            break

        # Track if this is a start-of-turn draw (coarse turn counter)
        action_is_start = any(
            isinstance(a, (ActionDrawStockpile, ActionDrawDiscard, ActionCallCambia))
            for a in legal_actions
        )
        if acting_player == 0 and action_is_start:
            game_turn_number += 1

        # Check for ability tracking: if P0 is making a discard decision,
        # record the drawn card rank
        if acting_player == 0 and game_state.pending_action is not None:
            if isinstance(game_state.pending_action, ActionDiscard):
                drawn_card = game_state.pending_action_data.get("drawn_card")
                if drawn_card and drawn_card.rank in (SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING):
                    profile._pending_drawn_rank = drawn_card.rank
                else:
                    profile._pending_drawn_rank = None
            else:
                profile._pending_drawn_rank = None

        # Agent chooses action
        current_agent = agents[acting_player]
        chosen_action = current_agent.choose_action(game_state, legal_actions)

        # --- Log P0 actions ---
        if acting_player == 0:
            cat = classify_action(chosen_action)
            profile.action_counts[cat] += 1

            # Cambia timing
            if isinstance(chosen_action, ActionCallCambia):
                profile.cambia_turns.append(game_turn_number)

            # Replace slot tracking
            if isinstance(chosen_action, ActionReplace):
                profile.replace_slots[chosen_action.target_hand_index] += 1

            # Ability utilization
            if isinstance(chosen_action, ActionDiscard):
                rank = profile._pending_drawn_rank
                if rank is not None:
                    profile.ability_opportunities += 1
                    profile.ability_opp_by_rank[rank] += 1
                    if chosen_action.use_ability:
                        profile.ability_used += 1
                        profile.ability_use_by_rank[rank] += 1
                    profile._pending_drawn_rank = None

        # Apply action
        state_delta, undo_info = game_state.apply_action(chosen_action)

        # Update stateful agents
        has_stateful = any(isinstance(a, NeuralAgentWrapper) for a in agents)
        if has_stateful:
            obs = _make_observation(game_state, chosen_action, acting_player)
            if obs:
                for agent in agents:
                    if isinstance(agent, NeuralAgentWrapper):
                        agent.update_state(obs)

        # Record per-turn scores (after each start-of-turn action by P0)
        if acting_player == 0 and action_is_start and game_turn_number <= 50:
            profile.p0_score_by_turn[game_turn_number].append(hand_score(game_state, 0))
            profile.p1_score_by_turn[game_turn_number].append(hand_score(game_state, 1))
            profile.p0_hand_size_by_turn[game_turn_number].append(hand_size(game_state, 0))

    # Record outcome
    if game_state.is_terminal():
        winner = game_state._winner
        if winner == 0:
            profile.wins += 1
        elif winner == 1:
            profile.losses += 1
        else:
            profile.ties += 1
        profile.final_score_p0.append(hand_score(game_state, 0))
        profile.final_score_p1.append(hand_score(game_state, 1))
    else:
        profile.ties += 1  # max turns

    profile.total_turns.append(turn)


def _make_observation(game_state, action, acting_player) -> Optional[AgentObservation]:
    try:
        return AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=game_state.get_discard_top(),
            player_hand_sizes=[
                game_state.get_player_card_count(i) for i in range(len(game_state.players))
            ],
            stockpile_size=game_state.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=copy.deepcopy(game_state.snap_results_log),
            did_cambia_get_called=game_state.cambia_caller_id is not None,
            who_called_cambia=game_state.cambia_caller_id,
            is_game_over=game_state.is_terminal(),
            current_turn=game_state.get_turn_number(),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_profile(profile: BehaviorProfile):
    """Print formatted profile summary."""
    total_actions = sum(profile.action_counts.values())
    total_games = profile.wins + profile.losses + profile.ties
    wr = profile.wins / max(1, profile.wins + profile.losses) * 100

    print(f"\n{'='*70}")
    print(f"  BEHAVIORAL PROFILE: {profile.label}")
    print(f"{'='*70}")
    print(f"  Games: {total_games}  |  Wins: {profile.wins}  Losses: {profile.losses}  Ties: {profile.ties}  Errors: {profile.errors}")
    print(f"  Win Rate (excl ties): {wr:.1f}%")
    if profile.total_turns:
        print(f"  Avg game turns: {np.mean(profile.total_turns):.1f}  (median {np.median(profile.total_turns):.0f})")
    if profile.final_score_p0:
        print(f"  Avg final P0 score: {np.mean(profile.final_score_p0):.1f}  |  Avg final P1 score: {np.mean(profile.final_score_p1):.1f}")

    # Action frequencies
    print(f"\n  --- Action Frequencies (P0, {total_actions} total actions) ---")
    print(f"  {'Action':<25s} {'Count':>8s} {'Pct':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*7}")
    # Sort by frequency
    for cat, cnt in sorted(profile.action_counts.items(), key=lambda x: -x[1]):
        pct = cnt / max(1, total_actions) * 100
        print(f"  {cat:<25s} {cnt:>8d} {pct:>6.1f}%")

    # Ability utilization
    print(f"\n  --- Ability Utilization ---")
    if profile.ability_opportunities > 0:
        use_rate = profile.ability_used / profile.ability_opportunities * 100
        print(f"  Overall: {profile.ability_used}/{profile.ability_opportunities} = {use_rate:.1f}%")
        print(f"  {'Rank':<10s} {'Used':>6s} {'Opps':>6s} {'Rate':>7s}")
        print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*7}")
        for rank in (SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING):
            opps = profile.ability_opp_by_rank.get(rank, 0)
            used = profile.ability_use_by_rank.get(rank, 0)
            if opps > 0:
                rate = used / opps * 100
                print(f"  {rank:<10s} {used:>6d} {opps:>6d} {rate:>6.1f}%")
    else:
        print(f"  No ability opportunities recorded.")

    # Cambia timing
    print(f"\n  --- Cambia Timing ---")
    if profile.cambia_turns:
        arr = np.array(profile.cambia_turns)
        print(f"  Count: {len(arr)}  |  Mean turn: {arr.mean():.1f}  Median: {np.median(arr):.0f}")
        print(f"  Min: {arr.min()}  Max: {arr.max()}  Std: {arr.std():.1f}")
        # Distribution buckets
        bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 50]
        hist, _ = np.histogram(arr, bins=bins)
        print(f"  Distribution:")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"    Turn {bins[i]}-{bins[i+1]-1}: {hist[i]} ({hist[i]/len(arr)*100:.1f}%)")
    else:
        print(f"  No Cambia calls recorded.")

    # Replace slot preference
    print(f"\n  --- Replace Slot Preference ---")
    total_replaces = sum(profile.replace_slots.values())
    if total_replaces > 0:
        for slot in sorted(profile.replace_slots.keys()):
            cnt = profile.replace_slots[slot]
            pct = cnt / total_replaces * 100
            print(f"  Slot {slot}: {cnt:>6d} ({pct:.1f}%)")
    else:
        print(f"  No replace actions recorded.")

    # Per-turn score trajectory
    print(f"\n  --- Per-Turn Mean Hand Score ---")
    max_turn = min(20, max(profile.p0_score_by_turn.keys()) if profile.p0_score_by_turn else 0)
    if max_turn > 0:
        print(f"  {'Turn':>5s} {'P0 Score':>10s} {'P1 Score':>10s} {'Gap':>8s} {'P0 Hand':>8s} {'N games':>8s}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for t in range(0, max_turn + 1):
            if t in profile.p0_score_by_turn:
                p0_scores = profile.p0_score_by_turn[t]
                p1_scores = profile.p1_score_by_turn[t]
                p0_hand = profile.p0_hand_size_by_turn.get(t, [])
                n = len(p0_scores)
                p0_mean = np.mean(p0_scores)
                p1_mean = np.mean(p1_scores)
                gap = p0_mean - p1_mean
                hand_mean = np.mean(p0_hand) if p0_hand else float("nan")
                print(f"  {t:>5d} {p0_mean:>10.1f} {p1_mean:>10.1f} {gap:>+8.1f} {hand_mean:>8.1f} {n:>8d}")


def compare_profiles(p1: BehaviorProfile, p2: BehaviorProfile):
    """Print side-by-side comparison of two profiles."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {p1.label} vs {p2.label}")
    print(f"{'='*70}")

    # Action frequency comparison
    all_cats = sorted(set(list(p1.action_counts.keys()) + list(p2.action_counts.keys())))
    t1 = sum(p1.action_counts.values())
    t2 = sum(p2.action_counts.values())

    print(f"\n  --- Action Frequency Comparison ---")
    print(f"  {'Action':<25s} {p1.label:>12s} {p2.label:>12s} {'Delta':>8s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
    for cat in all_cats:
        c1 = p1.action_counts.get(cat, 0)
        c2 = p2.action_counts.get(cat, 0)
        pct1 = c1 / max(1, t1) * 100
        pct2 = c2 / max(1, t2) * 100
        delta = pct1 - pct2
        print(f"  {cat:<25s} {pct1:>11.1f}% {pct2:>11.1f}% {delta:>+7.1f}%")

    # Ability utilization comparison
    print(f"\n  --- Ability Utilization Comparison ---")
    r1 = p1.ability_used / max(1, p1.ability_opportunities) * 100
    r2 = p2.ability_used / max(1, p2.ability_opportunities) * 100
    print(f"  {p1.label}: {r1:.1f}% ({p1.ability_used}/{p1.ability_opportunities})")
    print(f"  {p2.label}: {r2:.1f}% ({p2.ability_used}/{p2.ability_opportunities})")

    # Cambia timing comparison
    print(f"\n  --- Cambia Timing Comparison ---")
    if p1.cambia_turns:
        print(f"  {p1.label}: mean turn {np.mean(p1.cambia_turns):.1f}, N={len(p1.cambia_turns)}")
    else:
        print(f"  {p1.label}: no Cambia calls")
    if p2.cambia_turns:
        print(f"  {p2.label}: mean turn {np.mean(p2.cambia_turns):.1f}, N={len(p2.cambia_turns)}")
    else:
        print(f"  {p2.label}: no Cambia calls")

    # Per-turn score gap comparison
    print(f"\n  --- Per-Turn P0 Score Gap (P0 - P1) ---")
    max_turn = min(15, max(
        max(p1.p0_score_by_turn.keys()) if p1.p0_score_by_turn else 0,
        max(p2.p0_score_by_turn.keys()) if p2.p0_score_by_turn else 0,
    ))
    if max_turn > 0:
        print(f"  {'Turn':>5s} {p1.label:>12s} {p2.label:>12s}")
        print(f"  {'-'*5} {'-'*12} {'-'*12}")
        for t in range(0, max_turn + 1):
            gap1 = "---"
            gap2 = "---"
            if t in p1.p0_score_by_turn and p1.p0_score_by_turn[t]:
                g = np.mean(p1.p0_score_by_turn[t]) - np.mean(p1.p1_score_by_turn[t])
                gap1 = f"{g:>+.1f}"
            if t in p2.p0_score_by_turn and p2.p0_score_by_turn[t]:
                g = np.mean(p2.p0_score_by_turn[t]) - np.mean(p2.p1_score_by_turn[t])
                gap2 = f"{g:>+.1f}"
            print(f"  {t:>5d} {gap1:>12s} {gap2:>12s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    NUM_GAMES = 2000
    CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "runs", "interleaved-resnet-adaptive", "config.yaml"
    )
    CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "runs",
        "interleaved-resnet-adaptive",
        "checkpoints",
        "deep_cfr_checkpoint_iter_450.pt",
    )

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config not found: {CONFIG_PATH}")
        sys.exit(1)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    # --- Profile 1: Trained agent (P0) vs imperfect_greedy (P1) ---
    print(f"Running {NUM_GAMES} games: deep_cfr vs imperfect_greedy ...")
    t0 = time.time()
    trained_profile = BehaviorProfile("deep_cfr_450")
    play_games("deep_cfr", "imperfect_greedy", CONFIG_PATH, CHECKPOINT_PATH, NUM_GAMES, trained_profile)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Profile 2: random_no_cambia (P0) vs imperfect_greedy (P1) ---
    print(f"Running {NUM_GAMES} games: random_no_cambia vs imperfect_greedy ...")
    t0 = time.time()
    random_profile = BehaviorProfile("random_no_cambia")
    play_games("random_no_cambia", "imperfect_greedy", CONFIG_PATH, None, NUM_GAMES, random_profile)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Profile 3: imperfect_greedy (P0) vs imperfect_greedy (P1) for reference ---
    print(f"Running {NUM_GAMES} games: imperfect_greedy vs imperfect_greedy ...")
    t0 = time.time()
    greedy_profile = BehaviorProfile("imp_greedy_mirror")
    play_games("imperfect_greedy", "imperfect_greedy", CONFIG_PATH, None, NUM_GAMES, greedy_profile)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Print profiles
    print_profile(trained_profile)
    print_profile(random_profile)
    print_profile(greedy_profile)

    # Comparison
    compare_profiles(trained_profile, random_profile)
    compare_profiles(trained_profile, greedy_profile)


if __name__ == "__main__":
    main()
