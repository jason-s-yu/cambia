"""src/play.py: Interactive human-vs-AI play for Cambia."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .constants import (
    ActionCallCambia,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
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
)
from .agents import action_codec
from .agents.game_view import tracked_opponent_seat
from .encoding import action_to_index
from .ffi.bridge import GoEngine

console = Console()


# ---------------------------------------------------------------------------
# Action formatting
# ---------------------------------------------------------------------------


def action_to_str(action: GameAction) -> str:
    """Human-readable description of a game action."""
    match action:
        case ActionDrawStockpile():
            return "Draw from stockpile"
        case ActionDrawDiscard():
            return "Draw from discard pile"
        case ActionCallCambia():
            return "Call Cambia!"
        case ActionDiscard(use_ability=True):
            return "Discard drawn card (use ability)"
        case ActionDiscard(use_ability=False):
            return "Discard drawn card (no ability)"
        case ActionReplace(target_hand_index=i):
            return f"Replace your slot {i} with drawn card"
        case ActionAbilityPeekOwnSelect(target_hand_index=i):
            return f"Peek your slot {i}"
        case ActionAbilityPeekOtherSelect(target_opponent_hand_index=i):
            return f"Peek opponent slot {i}"
        case ActionAbilityBlindSwapSelect(own_hand_index=o, opponent_hand_index=e):
            return f"Blind swap your slot {o} with opponent slot {e}"
        case ActionAbilityKingLookSelect(own_hand_index=o, opponent_hand_index=e):
            return f"Look at your slot {o} and opponent slot {e}"
        case ActionAbilityKingSwapDecision(perform_swap=True):
            return "Swap the looked-at cards"
        case ActionAbilityKingSwapDecision(perform_swap=False):
            return "Keep cards as they are"
        case ActionPassSnap():
            return "Pass (don't snap)"
        case ActionSnapOwn(own_card_hand_index=i):
            return f"Snap your slot {i} (match discard)"
        case ActionSnapOpponent(opponent_target_hand_index=i):
            return f"Snap opponent slot {i} (match discard)"
        case ActionSnapOpponentMove(
            own_card_to_move_hand_index=o, target_empty_slot_index=t
        ):
            return f"Move your slot {o} to opponent slot {t}"
        case _:
            return str(action)


_ACTION_ORDER = {
    "draw_stockpile": 0,
    "draw_discard": 1,
    "discard": 2,  # within discard: use_ability=True sorts before False
    "replace": 3,
    "call_cambia": 4,
    "peek_own": 5,
    "peek_other": 6,
    "blind_swap": 7,
    "king_look": 8,
    "king_swap": 9,
    "snap_own": 10,
    "snap_opp": 11,
    "snap_opp_move": 12,
    "pass_snap": 13,
}


def _action_sort_key(action: GameAction):
    """Sort actions in intuitive play order."""
    tag = getattr(action, "tag", "")
    order = _ACTION_ORDER.get(tag, 99)
    # For discard: ability first (True=1 sorts before False=0 when negated)
    if tag == "discard":
        return (order, not getattr(action, "use_ability", False))
    # For positional actions: sort by slot index
    for attr in (
        "target_hand_index",
        "own_hand_index",
        "own_card_hand_index",
        "target_opponent_hand_index",
        "opponent_hand_index",
        "opponent_target_hand_index",
    ):
        val = getattr(action, attr, None)
        if val is not None:
            return (order, val)
    return (order, 0)


def card_str(card) -> str:
    """Format a Card as a short string like '7H' or 'Joker'."""
    if card.rank == "R":
        return "Joker"
    return f"{card.rank}{card.suit}"


def card_value(card) -> int:
    """Get the point value of a card."""
    return card.value


# ---------------------------------------------------------------------------
# GoEngine legal-action decode / apply
# ---------------------------------------------------------------------------
#
# Mirrors _GoEvalGame in src/evaluate_agents.py: the 2-player action space is
# used directly at two seats; at more seats the sparse N-player space is
# projected onto the single opponent the GameView-ported agents track
# (game_view.tracked_opponent_seat), since both the AI seats and the human
# seat here are driven through the same BaseAgent.choose_action / manual
# selection over one legal_actions list. That list is not reusable from
# evaluate_agents.py (a private, engine-owning class there), so the decode is
# duplicated rather than imported.


def _legal_actions_for(engine: GoEngine, acting_player: int, num_players: int):
    """The acting seat's legal actions and the index each was decoded from."""
    if num_players == 2:
        mask = engine.legal_actions_mask()
        actions = action_codec.actions_from_mask(mask)
        index = {a: int(i) for a, i in zip(actions, np.flatnonzero(mask))}
        return actions, index

    pending = engine.get_pending()
    if pending.target_seat is not None and pending.target_seat != acting_player:
        target = int(pending.target_seat)
    else:
        target = tracked_opponent_seat(acting_player, num_players)
    rel = action_codec.relative_opponent_index(acting_player, target, num_players)

    preferred: List[GameAction] = []
    preferred_index: Dict[GameAction, int] = {}
    fallback: List[GameAction] = []
    fallback_index: Dict[GameAction, int] = {}

    mask = engine.nplayer_legal_actions_mask()
    for idx in np.flatnonzero(mask):
        entry = action_codec.nplayer_index_to_action(int(idx))
        action = entry.action
        if isinstance(action, ActionSnapOpponentMove):
            # The N-player space keeps only the own-card index; the target
            # slot lives on the pending record.
            action = ActionSnapOpponentMove(
                own_card_to_move_hand_index=action.own_card_to_move_hand_index,
                target_empty_slot_index=int(pending.target_slot or 0),
            )
        if entry.opp_idx is None or entry.opp_idx == rel:
            if action not in preferred_index:
                preferred.append(action)
                preferred_index[action] = int(idx)
        if action not in fallback_index:
            fallback.append(action)
            fallback_index[action] = int(idx)

    if preferred:
        return preferred, preferred_index
    return fallback, fallback_index


def _apply_action(
    engine: GoEngine,
    action: GameAction,
    index_map: Dict[GameAction, int],
    acting_player: int,
    num_players: int,
) -> None:
    """Apply a chosen action, re-encoding it if it was not in the decoded set.

    An action a seat builds without checking legality is left to the engine to
    reject, exactly as it did on the Python engine.
    """
    idx = index_map.get(action)
    if num_players == 2:
        if idx is None:
            idx = action_to_index(action)
        engine.apply_action(int(idx))
        return

    if idx is None:
        pending = engine.get_pending()
        if pending.target_seat is not None and pending.target_seat != acting_player:
            target = int(pending.target_seat)
        else:
            target = tracked_opponent_seat(acting_player, num_players)
        rel = action_codec.relative_opponent_index(acting_player, target, num_players)
        idx = action_codec.nplayer_index_for(action, rel)
    engine.apply_nplayer_action(int(idx))


# ---------------------------------------------------------------------------
# Player knowledge tracker
# ---------------------------------------------------------------------------


@dataclass
class PlayerKnowledge:
    """Tracks what a human player knows about cards on the table."""

    # Maps (player_id, slot_index) -> Card object the human knows about
    known_cards: Dict[tuple, Any] = field(default_factory=dict)

    def know(self, player_id: int, slot: int, card) -> None:
        self.known_cards[(player_id, slot)] = card

    def forget(self, player_id: int, slot: int) -> None:
        self.known_cards.pop((player_id, slot), None)

    def get(self, player_id: int, slot: int):
        return self.known_cards.get((player_id, slot))


# ---------------------------------------------------------------------------
# Seat config
# ---------------------------------------------------------------------------


@dataclass
class SeatConfig:
    """Configuration for a single seat at the table."""

    seat_id: int
    is_human: bool
    name: str
    agent_type: str = ""
    agent: Any = None  # BaseAgent instance for AI players
    knowledge: PlayerKnowledge = field(default_factory=PlayerKnowledge)


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def render_game_state(
    game: GoEngine,
    viewer_id: int,
    seats: List[SeatConfig],
    cambia_caller_id: Optional[int] = None,
) -> None:
    """Render the game state from a specific player's perspective."""
    viewer = seats[viewer_id]
    num_players = len(seats)

    # Header
    turn_info = f"Turn {game.turn_number()}"
    if cambia_caller_id is not None:
        caller_name = seats[cambia_caller_id].name
        turn_info += f"  |  Cambia called by {caller_name}"

    console.print()
    console.rule(f"[bold cyan]{turn_info}[/]")

    # Discard pile
    top = game.get_discard_top()
    if top is not None:
        console.print(
            f"  Discard pile top: [bold yellow]{card_str(top)}[/] (value {card_value(top)})"
        )
    else:
        console.print("  Discard pile: [dim]empty[/]")

    console.print(f"  Stockpile: {game.stock_len()} cards remaining")
    console.print()

    # Show each player's hand
    for pid in range(num_players):
        seat = seats[pid]
        hand = game.get_player_hand(pid)
        is_viewer = pid == viewer_id

        hand_display = []
        for slot_idx, card in enumerate(hand):
            known = viewer.knowledge.get(pid, slot_idx)
            if is_viewer and known is not None:
                hand_display.append(
                    f"[bold green]{card_str(known)}[/]({card_value(known)})"
                )
            elif not is_viewer and known is not None:
                hand_display.append(
                    f"[bold magenta]{card_str(known)}[/]({card_value(known)})"
                )
            else:
                hand_display.append("[dim][ ?? ][/]")

        label = f"[bold]>> {seat.name} (YOU)[/]" if is_viewer else f"   {seat.name}"
        if not hand:
            hand_str = "[dim]no cards[/]"
        else:
            hand_str = "  ".join(f"[{i}]{s}" for i, s in enumerate(hand_display))

        console.print(f"{label}:  {hand_str}")

    # Show drawn card if pending
    pending = game.get_pending()
    if pending.is_pending and viewer_id == game.acting_player():
        drawn = pending.drawn_card
        if drawn:
            console.print(
                f"\n  You drew: [bold yellow]{card_str(drawn)}[/] (value {card_value(drawn)})"
            )

    console.print()


def render_legal_actions(actions: List[GameAction]) -> None:
    """Show numbered menu of legal actions."""
    console.print("[bold]Choose an action:[/]")
    for i, action in enumerate(actions):
        console.print(f"  [cyan]{i}[/]) {action_to_str(action)}")
    console.print()


def prompt_action(actions: List[GameAction]) -> GameAction:
    """Prompt the human player to select an action."""
    while True:
        try:
            raw = console.input("[bold]> [/]").strip()
            if raw.lower() in ("q", "quit", "exit"):
                console.print("[dim]Game abandoned.[/]")
                sys.exit(0)
            idx = int(raw)
            if 0 <= idx < len(actions):
                return actions[idx]
            console.print(f"[red]Enter a number 0-{len(actions) - 1}[/]")
        except ValueError:
            console.print("[red]Enter a number or 'q' to quit[/]")


def render_ai_action(name: str, action: GameAction) -> None:
    """Display what an AI player did."""
    console.print(f"  [dim]{name}[/] -> {action_to_str(action)}")


def render_game_over(game: GoEngine, seats: List[SeatConfig], num_players: int) -> None:
    """Show final results."""
    console.print()
    console.rule("[bold red]Game Over[/]")

    table = Table(title="Final Hands")
    table.add_column("Player")
    table.add_column("Hand")
    table.add_column("Score", justify="right")

    scores = []
    for pid, seat in enumerate(seats):
        hand = game.get_player_hand(pid)
        hand_strs = [f"{card_str(c)}({card_value(c)})" for c in hand]
        score = sum(card_value(c) for c in hand)
        scores.append(score)
        table.add_row(
            seat.name, "  ".join(hand_strs) if hand_strs else "empty (0)", str(score)
        )

    console.print(table)

    # Winner: the sole argmax of the engine's utility vector, which already
    # encodes the whole scoring rule including the Cambia-caller tiebreak
    # (mirrors _GoEvalGame.winner() in evaluate_agents.py). Equal top
    # utilities are a tie.
    utils = game.get_nplayer_utility() if num_players > 2 else game.get_utility()
    utils = np.asarray(utils, dtype=np.float64)[:num_players]
    best = float(utils.max())
    leaders = np.flatnonzero(utils >= best - 1e-9)
    if len(leaders) == 1:
        console.print(f"\n  Winner: [bold green]{seats[int(leaders[0])].name}[/]!")
    else:
        winners = [seats[int(i)].name for i in leaders]
        console.print(f"\n  Tie between: {', '.join(winners)}")

    console.print()


# ---------------------------------------------------------------------------
# Knowledge update logic
# ---------------------------------------------------------------------------


def _shift_knowledge_after_removal(
    k: PlayerKnowledge, player_id: int, removed_slot: int
) -> None:
    """Shift knowledge entries down after a card is removed from a hand."""
    # Collect all entries for this player, sorted by slot
    entries = sorted(
        [(slot, card) for (pid, slot), card in k.known_cards.items() if pid == player_id],
        key=lambda x: x[0],
    )
    # Remove all entries for this player
    for slot, _ in entries:
        k.known_cards.pop((player_id, slot), None)
    # Re-add with shifted indices (skip the removed slot)
    for slot, card in entries:
        if slot < removed_slot:
            k.known_cards[(player_id, slot)] = card
        elif slot > removed_slot:
            k.known_cards[(player_id, slot - 1)] = card
        # slot == removed_slot: dropped (card was snapped away)


def update_knowledge(
    seat: SeatConfig,
    action: GameAction,
    acting_player: int,
    game: GoEngine,
    seats: List[SeatConfig],
) -> None:
    """Update a human player's knowledge based on an action that just occurred."""
    k = seat.knowledge
    viewer = seat.seat_id
    num_players = len(seats)

    match action:
        case ActionReplace(target_hand_index=slot):
            if acting_player == viewer:
                # We know what card we just placed (it was the drawn card)
                viewer_hand = game.get_player_hand(viewer)
                card = viewer_hand[slot] if slot < len(viewer_hand) else None
                if card:
                    k.know(viewer, slot, card)
            # If opponent replaced, we don't know their new card
            # but we saw the discarded card (goes to discard pile - visible)

        case ActionAbilityPeekOwnSelect(target_hand_index=slot):
            if acting_player == viewer:
                viewer_hand = game.get_player_hand(viewer)
                card = viewer_hand[slot] if slot < len(viewer_hand) else None
                if card:
                    k.know(viewer, slot, card)
                    console.print(
                        f"  [bold green]You peeked: your slot {slot} = {card_str(card)} ({card_value(card)})[/]"
                    )

        case ActionAbilityPeekOtherSelect(target_opponent_hand_index=slot):
            if acting_player == viewer:
                # Peeked at opponent's card. The action carries no target seat
                # (a 2-player-shaped index); at more seats this names the same
                # single opponent the GameView-ported agents track, which is a
                # pre-existing simplification carried forward from the
                # Python-engine version of this function, not new here.
                opp = tracked_opponent_seat(viewer, num_players)
                opp_hand = game.get_player_hand(opp)
                card = opp_hand[slot] if slot < len(opp_hand) else None
                if card:
                    k.know(opp, slot, card)
                    console.print(
                        f"  [bold magenta]You peeked: opponent slot {slot} = {card_str(card)} ({card_value(card)})[/]"
                    )

        case ActionSnapOwn(own_card_hand_index=snap_slot):
            # A card was removed from acting_player's hand at snap_slot.
            # All cards above that index shift down by 1.
            _shift_knowledge_after_removal(k, acting_player, snap_slot)
            # The snapped card went to discard: visible to all, no knowledge needed
            console.print(
                f"  [bold]Snap![/] {seats[acting_player].name} snapped slot {snap_slot}"
            )

        case ActionAbilityBlindSwapSelect(
            own_hand_index=own_slot, opponent_hand_index=opp_slot
        ):
            if acting_player == viewer:
                # After blind swap, we no longer know what's in our slot
                # (it's whatever opponent had, which we didn't see)
                opp = tracked_opponent_seat(viewer, num_players)
                k.forget(viewer, own_slot)
                k.forget(opp, opp_slot)

        case ActionAbilityKingLookSelect(
            own_hand_index=own_slot, opponent_hand_index=opp_slot
        ):
            if acting_player == viewer:
                opp = tracked_opponent_seat(viewer, num_players)
                viewer_hand = game.get_player_hand(viewer)
                opp_hand = game.get_player_hand(opp)
                own_card = viewer_hand[own_slot] if own_slot < len(viewer_hand) else None
                opp_card = opp_hand[opp_slot] if opp_slot < len(opp_hand) else None
                if own_card:
                    k.know(viewer, own_slot, own_card)
                    console.print(
                        f"  [bold green]You looked: your slot {own_slot} = {card_str(own_card)} ({card_value(own_card)})[/]"
                    )
                if opp_card:
                    k.know(opp, opp_slot, opp_card)
                    console.print(
                        f"  [bold magenta]You looked: opponent slot {opp_slot} = {card_str(opp_card)} ({card_value(opp_card)})[/]"
                    )

        case ActionAbilityKingSwapDecision(perform_swap=True):
            if acting_player == viewer:
                # Knowledge was set in KingLookSelect; the engine performs the
                # swap and the pre-swap knowledge entries already track the
                # cards (not the slots), so nothing further to update here.
                pass

        case _:
            pass


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


def play_game(
    seat_configs: List[SeatConfig],
    house_rules,
    num_players: Optional[int] = None,
) -> None:
    """Run an interactive game.

    ``num_players`` defaults to ``len(seat_configs)`` and is passed straight
    through to ``GoEngine``, which sizes its N-player utility buffer from it.
    ``CambiaRulesConfig`` carries no seat count of its own, so this argument
    is the sole authority on how many seats are dealt; the post-construction
    ``engine.num_players()`` check below is a defensive cross-check, mirroring
    the same check in ppo_env.CambiaEnv._deal and
    evaluate_agents._GoEvalGame, against a house_rules object that did carry
    one.
    """
    num_players = num_players or len(seat_configs)
    game = GoEngine(house_rules=house_rules, num_players=num_players)
    try:
        seats = game.num_players()
        if seats != num_players:
            raise RuntimeError(
                f"engine dealt {seats} seats, play_game was configured for "
                f"{num_players}."
            )

        # Set up seat IDs
        for i, seat in enumerate(seat_configs):
            seat.seat_id = i

        # Initialize AI agents that carry belief across the game (neural
        # wrappers backed by GoAgentState). The GameView-ported baselines
        # have no such hook: their per-game memory resets lazily on first
        # choose_action (see ImperfectMemoryMixin._needs_reinit).
        for seat in seat_configs:
            if not seat.is_human and seat.agent is not None:
                if hasattr(seat.agent, "initialize_state"):
                    seat.agent.initialize_state(game)

        # Initialize human knowledge with the initial peek: the first
        # initial_view_count slots, in deal order (the same convention
        # ImperfectMemoryMixin._init_memory uses for the Go engine).
        peek_count = game.get_house_rules().initial_view_count
        for seat in seat_configs:
            if seat.is_human:
                hand = game.get_player_hand(seat.seat_id)
                for slot in range(min(peek_count, len(hand))):
                    seat.knowledge.know(seat.seat_id, slot, hand[slot])

        # `turn` here counts individual actions, not engine turns: a single
        # engine turn can span several actions (draw, ability sub-selects,
        # snap responses), so the cap is house_rules.max_game_turns scaled up
        # rather than used directly -- the engine enforces its own turn cap
        # and reaches a real terminal state well within this budget; a tight
        # cap instead risks cutting the loop before is_terminal() is true and
        # scoring a mid-game state (mirrors the same scaling in
        # evaluate_agents.run_evaluation's engine_turn_cap).
        engine_turn_cap = (
            house_rules.max_game_turns
            if hasattr(house_rules, "max_game_turns") and house_rules.max_game_turns > 0
            else 500
        )
        max_turns = engine_turn_cap * 16
        turn = 0
        cambia_caller_id: Optional[int] = None

        # Show initial state for first human
        for seat in seat_configs:
            if seat.is_human:
                console.print(
                    f"\n[bold]Welcome, {seat.name}![/] You are Player {seat.seat_id}."
                )
                console.print(
                    f"Your initial peek shows you slots {list(range(min(peek_count, len(game.get_player_hand(seat.seat_id)))))}."
                )
                break

        while not game.is_terminal() and turn < max_turns:
            turn += 1
            acting_player = game.acting_player()
            if acting_player == -1:
                console.print("[red]Error: no acting player[/]")
                break

            legal_actions, index_map = _legal_actions_for(
                game, acting_player, num_players
            )
            if not legal_actions:
                if game.is_terminal():
                    break
                console.print("[red]Error: no legal actions in non-terminal state[/]")
                break

            seat = seat_configs[acting_player]
            action_list = sorted(legal_actions, key=_action_sort_key)

            if seat.is_human:
                render_game_state(game, acting_player, seat_configs, cambia_caller_id)
                render_legal_actions(action_list)
                chosen = prompt_action(action_list)
            else:
                chosen = seat.agent.choose_action(game, action_list)
                render_ai_action(seat.name, chosen)

            if isinstance(chosen, ActionCallCambia):
                cambia_caller_id = acting_player

            _apply_action(game, chosen, index_map, acting_player, num_players)

            # Update knowledge for all human players
            for s in seat_configs:
                if s.is_human:
                    update_knowledge(s, chosen, acting_player, game, seat_configs)

        render_game_over(game, seat_configs, num_players)
    finally:
        game.close()
