"""
tests/test_go_fuzz_oracle.py

Python fuzz oracle: plays N random games using the Python engine and records
a full trace file for validating the Go engine against the Python reference.

Each trace entry captures a single decision point:
  - seed              (int)  game index used to seed the RNG
  - turn_number       (int)  _turn_number at decision time
  - acting_player     (int)  player who must act (0 or 1)
  - decision_context  (int)  DecisionContext enum value (0-5)
  - legal_actions     (list[int]) sorted action indices
  - action_chosen_idx (int)  index of the action that was actually applied
  - is_terminal       (bool) whether the state is terminal after the action
  - utilities         (list[float])  [u0, u1], meaningful only when terminal
  - hand_sizes        (list[int])  [cards_p0, cards_p1]
  - discard_top_rank  (int | null)  Card.value of discard top, or null
  - stockpile_size    (int)  number of cards remaining in stockpile

Usage (standalone):
    python tests/test_go_fuzz_oracle.py --games 100 --output tests/fixtures/fuzz_traces.json

Usage (pytest):
    python -m pytest tests/test_go_fuzz_oracle.py -v

NOTE ON RNG ALIGNMENT:
  The Go engine uses a 64-bit xorshift RNG (xorshift64) for all randomness.
  To ensure Go and Python deal the same cards from the same seed, this oracle
  implements xorshift64 and uses it for the Fisher-Yates deck shuffle and
  starting-player selection.  All gameplay action choices use Python's stdlib
  random (seeded deterministically from the outer RNG) — this only affects
  which of the legal actions is chosen each step, not the card layout.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path and stub src.config if needed
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import types

_config_mod = sys.modules.get("src.config")
if _config_mod is None or not hasattr(_config_mod, "CambiaRulesConfig"):
    _config_stub = types.ModuleType("src.config")

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

    class _StubConfig:
        pass

    _config_stub.CambiaRulesConfig = _CambiaRulesConfig
    _config_stub.Config = _StubConfig
    _config_stub.CfrTrainingConfig = _StubConfig
    _config_stub.AgentParamsConfig = _StubConfig
    _config_stub.ApiConfig = _StubConfig
    _config_stub.SystemConfig = _StubConfig
    _config_stub.CfrPlusParamsConfig = _StubConfig
    _config_stub.PersistenceConfig = _StubConfig
    _config_stub.LoggingConfig = _StubConfig
    _config_stub.AgentsConfig = _StubConfig
    _config_stub.AnalysisConfig = _StubConfig
    sys.modules["src.config"] = _config_stub

# ---------------------------------------------------------------------------
# Imports (after sys.path / stub setup)
# ---------------------------------------------------------------------------
from src.constants import (  # noqa: E402
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionSnapOpponentMove,
    DecisionContext,
    NUM_PLAYERS,
)
from src.encoding import action_to_index, NUM_ACTIONS  # noqa: E402
from src.game.engine import CambiaGameState  # noqa: E402
from src.game.player_state import PlayerState  # noqa: E402
from src.card import create_standard_deck  # noqa: E402

# ---------------------------------------------------------------------------
# xorshift64 — mirrors Go engine's RNG exactly
# ---------------------------------------------------------------------------

# Python integers are arbitrary precision; we mask to 64-bit unsigned arithmetic.
_MASK64 = (1 << 64) - 1


class XorShift64:
    """
    64-bit xorshift pseudo-random number generator that exactly mirrors
    the Go engine's `nextRand` / `randN` methods in game.go.

    Algorithm (same constants as Go):
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
    """

    def __init__(self, seed: int) -> None:
        # Replicate Go's seed-zero correction.
        self.state: int = seed & _MASK64 or 1

    def next_rand(self) -> int:
        """Advance and return next 64-bit unsigned value."""
        x = self.state
        x ^= (x << 13) & _MASK64
        x ^= (x >> 7) & _MASK64
        x ^= (x << 17) & _MASK64
        self.state = x
        return x

    def rand_n(self, n: int) -> int:
        """Return a random integer in [0, n)."""
        return self.next_rand() % n

    def shuffle(self, lst: list) -> None:
        """In-place Fisher-Yates shuffle matching Go's Deal() loop."""
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = int(self.rand_n(i + 1))
            lst[i], lst[j] = lst[j], lst[i]


# ---------------------------------------------------------------------------
# Game setup using Go-compatible RNG
# ---------------------------------------------------------------------------

# House rules that match Go's DefaultHouseRules as used in the fuzz test
_CARDS_PER_PLAYER = 4
_NUM_PLAYERS = 2


def _setup_game_with_go_rng(seed: int) -> CambiaGameState:
    """
    Create a CambiaGameState whose deck layout and starting player match
    what Go's engine produces for the same seed.

    Go's Deal() (game.go):
      1. Fisher-Yates shuffle of 54-card deck using xorshift64.
      2. Deal CardsPerPlayer cards to each player (alternating: P0, P1, P0, P1, ...).
      3. Pick random starting player via randN(2).
      (Go also flips one card to discard; the fuzz test undoes this, so we skip it.)
    """
    rng = XorShift64(seed)

    # Build the deck in the same order Go does (suits 0-3, ranks 0-12, then jokers).
    # Go suits: Hearts=0, Diamonds=1, Clubs=2, Spades=3
    # Go ranks: Ace=0, 2..9=1..8, T=9, J=10, Q=11, K=12, Joker=13
    # Python suits: S, H, D, C  (ALL_SUITS = ['S','H','D','C'])
    # Python ranks: A,2,3,4,5,6,7,8,9,T,J,Q,K,R  (ALL_RANKS_STR)
    from src.constants import ALL_RANKS_STR, JOKER_RANK_STR

    # Build ordered deck matching Go's Stockpile initialization.
    # Go loop: suit in 0..3, rank in 0..12, then two jokers.
    go_suits = ["H", "D", "C", "S"]  # suits 0-3 in Go order
    go_non_joker_ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    from src.card import Card

    deck: List[Card] = []
    for suit in go_suits:
        for rank in go_non_joker_ranks:
            deck.append(Card(rank=rank, suit=suit))
    # Two jokers (no suit)
    deck.append(Card(rank=JOKER_RANK_STR))
    deck.append(Card(rank=JOKER_RANK_STR))

    assert len(deck) == 54, f"Expected 54-card deck, got {len(deck)}"

    # Fisher-Yates shuffle using Go's xorshift64.
    rng.shuffle(deck)

    # Deal cards alternating P0, P1 (CardsPerPlayer rounds).
    hands: List[List[Card]] = [[] for _ in range(_NUM_PLAYERS)]
    for _ in range(_CARDS_PER_PLAYER):
        for p in range(_NUM_PLAYERS):
            hands[p].append(deck.pop())  # pop() takes from end (== go's Stockpile[--StockLen])

    # Remaining deck is the stockpile (Python convention: top-of-deck is last element).
    stockpile = deck  # already in correct order

    # Starting player.
    starting_player = int(rng.rand_n(_NUM_PLAYERS))

    # Construct CambiaGameState without triggering auto-setup.
    # We pass players and stockpile directly; __post_init__ skips _setup_game
    # when players is non-empty.
    players = [
        PlayerState(hand=hands[p], initial_peek_indices=(0, 1))
        for p in range(_NUM_PLAYERS)
    ]

    state = CambiaGameState(
        players=players,
        stockpile=stockpile,
        discard_pile=[],
        current_player_index=starting_player,
    )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_decision_context(state: CambiaGameState) -> DecisionContext:
    """Infer the current DecisionContext from the game state (mirrors deep_worker logic)."""
    if state.snap_phase_active:
        return DecisionContext.SNAP_DECISION

    pending = state.pending_action
    if pending is not None:
        if isinstance(pending, ActionDiscard):
            return DecisionContext.POST_DRAW
        if isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            return DecisionContext.ABILITY_SELECT
        if isinstance(pending, ActionSnapOpponentMove):
            return DecisionContext.SNAP_MOVE
        # Fallback for any other pending type
        return DecisionContext.START_TURN

    return DecisionContext.START_TURN


def _discard_top_rank(state: CambiaGameState) -> Optional[int]:
    """Return the integer value of the top discard card, or None if empty."""
    top = state.get_discard_top()
    return top.value if top is not None else None


def _hand_sizes(state: CambiaGameState) -> List[int]:
    return [state.get_player_card_count(i) for i in range(state.num_players)]


# ---------------------------------------------------------------------------
# Core: play one game and collect traces
# ---------------------------------------------------------------------------

def _play_game(seed: int, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Play a single game seeded with `seed`.  Returns a list of trace entries,
    one per decision point encountered during the game.

    The deck layout and starting player are determined by Go-compatible xorshift64
    so that replaying the same action sequence in the Go engine produces the same
    card identities at every step.
    """
    state = _setup_game_with_go_rng(seed)

    traces: List[Dict[str, Any]] = []
    max_steps = 10_000  # Safety bound against infinite loops

    for _ in range(max_steps):
        if state.is_terminal():
            break

        # --- Record decision point BEFORE applying the action ---
        legal_set = state.get_legal_actions()
        if not legal_set:
            # No legal actions in non-terminal state — engine stalemate, stop
            break

        acting_player = state.get_acting_player()
        if acting_player == -1:
            break

        context = _get_decision_context(state)
        legal_indices = sorted(action_to_index(a) for a in legal_set)

        # Pick a random legal action (using outer rng for reproducibility across seeds)
        legal_list = sorted(list(legal_set), key=repr)
        chosen_action = rng.choice(legal_list)
        chosen_idx = action_to_index(chosen_action)

        # Apply the action
        state.apply_action(chosen_action)

        is_terminal = state.is_terminal()
        utilities: List[float]
        if is_terminal:
            utilities = [state.get_utility(i) for i in range(NUM_PLAYERS)]
        else:
            utilities = [0.0] * NUM_PLAYERS

        entry: Dict[str, Any] = {
            "seed": seed,
            "turn_number": state._turn_number,
            "acting_player": acting_player,
            "decision_context": context.value,
            "legal_actions": legal_indices,
            "action_chosen_idx": chosen_idx,
            "is_terminal": is_terminal,
            "utilities": utilities,
            "hand_sizes": _hand_sizes(state),
            "discard_top_rank": _discard_top_rank(state),
            "stockpile_size": state.get_stockpile_size(),
        }
        traces.append(entry)

    return traces


# ---------------------------------------------------------------------------
# Main: generate traces for N games
# ---------------------------------------------------------------------------

def generate_traces(num_games: int, output_path: str) -> List[Dict[str, Any]]:
    """
    Generate fuzz traces for `num_games` games and write them to `output_path`.

    Returns the full list of trace entries (all games combined).
    """
    all_traces: List[Dict[str, Any]] = []
    # Use a deterministic outer RNG so action choices are reproducible
    outer_rng = random.Random(42)

    for game_idx in range(num_games):
        game_traces = _play_game(seed=game_idx, rng=outer_rng)
        all_traces.extend(game_traces)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_traces, f, indent=2)

    print(
        f"Generated {len(all_traces)} trace entries from {num_games} games "
        f"-> {output_path}"
    )
    return all_traces


# ---------------------------------------------------------------------------
# Pytest test: validate oracle output format
# ---------------------------------------------------------------------------

def test_fuzz_oracle_output_format():
    """Verify the oracle produces valid, parseable trace data."""
    # Generate a small trace (5 games) in-memory
    outer_rng = random.Random(0)
    all_traces: List[Dict[str, Any]] = []
    num_test_games = 5

    for game_idx in range(num_test_games):
        traces = _play_game(seed=game_idx, rng=outer_rng)
        all_traces.extend(traces)

    # Basic sanity: we should have at least some entries
    assert len(all_traces) > 0, "Oracle produced no trace entries"

    # All games must terminate: last entry of each game should have is_terminal=True
    game_traces: Dict[int, List[Dict[str, Any]]] = {}
    for entry in all_traces:
        seed = entry["seed"]
        game_traces.setdefault(seed, []).append(entry)

    for seed, entries in game_traces.items():
        assert entries[-1]["is_terminal"], (
            f"Game {seed} did not terminate (last entry is_terminal=False)"
        )

    required_keys = {
        "seed",
        "turn_number",
        "acting_player",
        "decision_context",
        "legal_actions",
        "action_chosen_idx",
        "is_terminal",
        "utilities",
        "hand_sizes",
        "discard_top_rank",
        "stockpile_size",
    }

    for i, entry in enumerate(all_traces):
        # All required keys present
        missing = required_keys - entry.keys()
        assert not missing, f"Entry {i} missing keys: {missing}"

        # action indices in valid range [0, NUM_ACTIONS)
        for idx in entry["legal_actions"]:
            assert 0 <= idx < NUM_ACTIONS, (
                f"Entry {i}: legal action index {idx} out of range [0, {NUM_ACTIONS})"
            )
        assert 0 <= entry["action_chosen_idx"] < NUM_ACTIONS, (
            f"Entry {i}: action_chosen_idx {entry['action_chosen_idx']} out of range"
        )

        # chosen action must be in legal actions
        assert entry["action_chosen_idx"] in entry["legal_actions"], (
            f"Entry {i}: chosen action {entry['action_chosen_idx']} not in legal_actions"
        )

        # hand sizes are reasonable
        for size in entry["hand_sizes"]:
            assert 0 <= size <= 10, f"Entry {i}: unreasonable hand size {size}"

        # utilities: list of NUM_PLAYERS floats
        assert len(entry["utilities"]) == NUM_PLAYERS, (
            f"Entry {i}: utilities length {len(entry['utilities'])} != {NUM_PLAYERS}"
        )

        # decision_context in valid range
        valid_contexts = {dc.value for dc in DecisionContext}
        assert entry["decision_context"] in valid_contexts, (
            f"Entry {i}: invalid decision_context {entry['decision_context']}"
        )

        # acting_player is 0 or 1
        assert entry["acting_player"] in (0, 1), (
            f"Entry {i}: acting_player {entry['acting_player']} not in (0, 1)"
        )

        # stockpile_size is non-negative
        assert entry["stockpile_size"] >= 0, (
            f"Entry {i}: negative stockpile_size {entry['stockpile_size']}"
        )

        # discard_top_rank is int or None
        dtr = entry["discard_top_rank"]
        assert dtr is None or isinstance(dtr, int), (
            f"Entry {i}: discard_top_rank must be int or null, got {type(dtr)}"
        )

    # Verify round-trip through JSON
    serialized = json.dumps(all_traces)
    reloaded = json.loads(serialized)
    assert len(reloaded) == len(all_traces)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Python-engine fuzz traces for Go engine validation."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to simulate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/fuzz_traces.json",
        help="Output JSON file path (default: tests/fixtures/fuzz_traces.json)",
    )
    args = parser.parse_args()
    generate_traces(num_games=args.games, output_path=args.output)
