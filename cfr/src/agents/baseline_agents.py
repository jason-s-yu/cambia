"""Implements simple baseline agents for evaluation purposes."""

import random
import logging
from abc import ABC, abstractmethod
from typing import Set, Optional, List, Dict, Tuple

from ..game.engine import CambiaGameState
from ..constants import (
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionSnapOpponentMove,
    GameAction,
    CardObject,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionPassSnap,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    KING,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
)
from ..config import Config
from ..card import Card

logger = logging.getLogger(__name__)

# Average expected value of an unknown card in the deck (~6.5 for standard 54-card deck)
UNKNOWN_CARD_EXPECTED_VALUE = 6.5


class BaseAgent(ABC):
    """Abstract base class for Cambia agents."""

    player_id: int
    opponent_id: int
    config: Config

    def __init__(self, player_id: int, config: Config):
        self.player_id = player_id
        self.opponent_id = 1 - player_id  # Assuming 2 players
        self.config = config

    @abstractmethod
    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Selects an action based on the current game state and legal actions."""
        pass


class RandomAgent(BaseAgent):
    """An agent that chooses actions randomly from the legal set."""

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses a random action."""
        if not legal_actions:
            # This should ideally not happen if called correctly, game engine handles terminal/no-action states
            logger.error(
                "RandomAgent P%d received empty legal actions set in non-terminal state?",
                self.player_id,
            )
            # What action to return? Raise error? For now, let's try returning a dummy/invalid action
            # Or perhaps return a default safe action if possible, like PassSnap? Difficult to generalize.
            # Let's raise an error to highlight the issue upstream.
            raise ValueError(
                f"RandomAgent P{self.player_id} cannot choose from empty legal actions."
            )

        action_list = list(legal_actions)
        chosen_action = random.choice(action_list)
        logger.debug("RandomAgent P%d chose action: %s", self.player_id, chosen_action)
        return chosen_action


class GreedyAgent(BaseAgent):
    """
    A simple rule-based greedy agent.
    Assumes perfect information (direct access to game_state) for decision making.
    Uses configurable parameters for some decisions.
    """

    def __init__(self, player_id: int, config: Config):
        super().__init__(player_id, config)
        # Get greedy agent specific config
        self.cambia_threshold = config.agents.greedy_agent.cambia_call_threshold
        logger.info(
            "GreedyAgent P%d initialized (Cambia Threshold: %d)",
            self.player_id,
            self.cambia_threshold,
        )
        # Note: This agent currently uses memory_level 0 logic implicitly by accessing true state.

    def _get_hand_value(self, hand: List[CardObject]) -> int:
        """Calculates the point value of a hand (perfect info)."""
        value = 0
        for card in hand:
            if isinstance(card, Card):
                value += card.value
            else:
                value += 99  # Penalize non-card items heavily
        return value

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses an action based on greedy heuristics."""
        if not legal_actions:
            raise ValueError(
                f"GreedyAgent P{self.player_id} cannot choose from empty legal actions."
            )

        my_hand = game_state.get_player_hand(self.player_id)
        opp_hand = game_state.get_player_hand(
            self.opponent_id
        )  # Needs opponent's hand for some decisions

        # --- Rule Priorities ---

        # 1. Call Cambia if possible and hand value is low enough
        if ActionCallCambia() in legal_actions:
            current_value = self._get_hand_value(my_hand)
            if current_value <= self.cambia_threshold:
                logger.debug(
                    "GreedyAgent P%d calling Cambia (Value: %d <= Threshold: %d)",
                    self.player_id,
                    current_value,
                    self.cambia_threshold,
                )
                return ActionCallCambia()

        # 2. Handle Snapping: Always snap if possible (prefer own)
        snap_own_actions = {a for a in legal_actions if isinstance(a, ActionSnapOwn)}
        if snap_own_actions:
            chosen_action = min(
                snap_own_actions, key=lambda a: a.own_card_hand_index
            )  # Snap lowest index match
            logger.debug(
                "GreedyAgent P%d chose action (SnapOwn): %s",
                self.player_id,
                chosen_action,
            )
            return chosen_action
        snap_opp_actions = {a for a in legal_actions if isinstance(a, ActionSnapOpponent)}
        if snap_opp_actions:
            # Greedy needs perfect info to know *which* opponent card to snap.
            # Find the first valid snap opponent action based on true opponent hand.
            snap_card = game_state.snap_discarded_card
            if snap_card:
                target_rank = snap_card.rank
                for action in sorted(
                    list(snap_opp_actions), key=lambda a: a.opponent_target_hand_index
                ):
                    opp_idx_target = action.opponent_target_hand_index
                    if (
                        0 <= opp_idx_target < len(opp_hand)
                        and isinstance(opp_hand[opp_idx_target], Card)
                        and opp_hand[opp_idx_target].rank == target_rank
                    ):
                        logger.debug(
                            "GreedyAgent P%d chose action (SnapOpponent): %s",
                            self.player_id,
                            action,
                        )
                        return action
                logger.warning(
                    "GreedyAgent P%d found legal SnapOpponent actions but none matched opponent hand?",
                    self.player_id,
                )
            # Fall through if cannot confirm opponent card

        # 3. Handle Post-Draw Choice (Discard/Replace)
        if any(isinstance(a, (ActionDiscard, ActionReplace)) for a in legal_actions):
            drawn_card = game_state.pending_action_data.get("drawn_card")
            if not drawn_card or not isinstance(drawn_card, Card):
                logger.error(
                    "GreedyAgent P%d in PostDraw state but drawn_card invalid: %s",
                    self.player_id,
                    drawn_card,
                )
                # Fallback: Just discard without ability
                return (
                    ActionDiscard(use_ability=False)
                    if ActionDiscard(use_ability=False) in legal_actions
                    else list(legal_actions)[0]
                )

            best_replace_action: Optional[ActionReplace] = None
            max_value_reduction = -1  # Aim for lowest value hand

            # Evaluate potential replacements
            for i, current_card in enumerate(my_hand):
                if isinstance(current_card, Card):
                    value_if_replaced = self._get_hand_value(
                        my_hand[:i] + [drawn_card] + my_hand[i + 1 :]
                    )
                    current_hand_value = self._get_hand_value(my_hand)
                    reduction = current_hand_value - value_if_replaced
                    # Strictly better replacement based on known value
                    if (
                        drawn_card.value < current_card.value
                        and reduction > max_value_reduction
                    ):
                        max_value_reduction = reduction
                        best_replace_action = ActionReplace(target_hand_index=i)

            # Rule: Replace unknown if drawn card is low enough (at or below threshold)
            if best_replace_action is None and drawn_card.value <= self.cambia_threshold:
                # Find first 'unknown' card - greedy assumes perfect memory, so this rule isn't applicable directly
                # Modify: Replace highest value card if drawn card <= threshold
                highest_value = -float("inf")
                replace_idx = -1
                for i, card in enumerate(my_hand):
                    if isinstance(card, Card) and card.value > highest_value:
                        highest_value = card.value
                        replace_idx = i
                if (
                    replace_idx != -1 and drawn_card.value <= highest_value
                ):  # Only replace if drawn is <= highest
                    best_replace_action = ActionReplace(target_hand_index=replace_idx)
                    logger.debug(
                        "GreedyAgent P%d replacing highest value card %d (%d) with low drawn card %s (%d)",
                        self.player_id,
                        replace_idx,
                        highest_value,
                        drawn_card,
                        drawn_card.value,
                    )

            # If a good replacement exists, take it
            if best_replace_action and best_replace_action in legal_actions:
                logger.debug(
                    "GreedyAgent P%d chose action (Replace): %s",
                    self.player_id,
                    best_replace_action,
                )
                return best_replace_action

            # Otherwise, consider discarding with ability if useful
            can_discard_ability = ActionDiscard(use_ability=True) in legal_actions
            is_utility_card = drawn_card.rank in [
                SEVEN,
                EIGHT,
                NINE,
                TEN,
                KING,
            ]  # 7,8,9,T,K give knowledge/control
            if can_discard_ability and is_utility_card:
                logger.debug(
                    "GreedyAgent P%d chose action (Discard Utility): %s",
                    self.player_id,
                    ActionDiscard(use_ability=True),
                )
                return ActionDiscard(use_ability=True)

            # Default: Simple discard
            logger.debug(
                "GreedyAgent P%d chose action (Default Discard): %s",
                self.player_id,
                ActionDiscard(use_ability=False),
            )
            return ActionDiscard(use_ability=False)

        # 4. Handle Ability Choices
        if isinstance(next(iter(legal_actions), None), ActionAbilityPeekOwnSelect):  # 7/8
            # Peek first unknown card (not implemented as agent has perfect info)
            # Simple: Peek lowest index card
            action = ActionAbilityPeekOwnSelect(target_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (PeekOwn): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityPeekOtherSelect
        ):  # 9/T
            # Peek opponent lowest index
            action = ActionAbilityPeekOtherSelect(target_opponent_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (PeekOther): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityBlindSwapSelect
        ):  # J/Q
            # Rule: Ignore J/Q. How to implement "ignore"? Choose a default non-swap action if possible.
            # This state should only be reachable if discard+ability was forced.
            # Choose the lowest index swap (0,0) as a default if forced.
            action = ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0)
            if action in legal_actions:
                logger.debug(
                    "GreedyAgent P%d chose action (Forced BlindSwap): %s",
                    self.player_id,
                    action,
                )
                return action
            else:  # Should not happen if legal_actions contained only BlindSwap options
                logger.error(
                    "GreedyAgent P%d forced into BlindSwap, but (0,0) invalid?",
                    self.player_id,
                )
                return list(legal_actions)[0]  # Failsafe

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityKingLookSelect
        ):  # King Look
            # Rule: Prioritize own hand knowledge? Not applicable with perfect info.
            # Simple: Look at lowest indices
            action = ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0)
            logger.debug(
                "GreedyAgent P%d chose action (KingLook): %s", self.player_id, action
            )
            return action

        if isinstance(
            next(iter(legal_actions), None), ActionAbilityKingSwapDecision
        ):  # King Swap
            # Rule: Swap only if it reduces own hand value
            look_data = game_state.pending_action_data
            card1 = look_data.get("card1")  # Own card peeked
            card2 = look_data.get("card2")  # Opp card peeked
            if isinstance(card1, Card) and isinstance(card2, Card):
                if (
                    card2.value < card1.value
                ):  # Swap if opponent card is better (lower value)
                    action = ActionAbilityKingSwapDecision(perform_swap=True)
                    logger.debug(
                        "GreedyAgent P%d chose action (KingSwap=True): %s",
                        self.player_id,
                        action,
                    )
                    return action
            # Default: Don't swap
            action = ActionAbilityKingSwapDecision(perform_swap=False)
            logger.debug(
                "GreedyAgent P%d chose action (KingSwap=False): %s",
                self.player_id,
                action,
            )
            return action

        if isinstance(next(iter(legal_actions), None), ActionSnapOpponentMove):
            # Move lowest value card from own hand to opponent's empty slot
            best_card_idx = -1
            lowest_value = float("inf")
            for i, card in enumerate(my_hand):
                if isinstance(card, Card) and card.value < lowest_value:
                    lowest_value = card.value
                    best_card_idx = i

            if best_card_idx != -1:
                # Get target slot from the first legal action (it's the same for all)
                example_action = next(iter(legal_actions))
                target_slot = example_action.target_empty_slot_index
                chosen_action = ActionSnapOpponentMove(
                    own_card_to_move_hand_index=best_card_idx,
                    target_empty_slot_index=target_slot,
                )
                if chosen_action in legal_actions:
                    logger.debug(
                        "GreedyAgent P%d chose action (SnapMove): %s",
                        self.player_id,
                        chosen_action,
                    )
                    return chosen_action
                else:
                    logger.error(
                        "GreedyAgent P%d SnapMove calculation error?", self.player_id
                    )

            # Fallback if error or no cards
            logger.warning("GreedyAgent P%d fallback SnapMove.", self.player_id)
            return list(legal_actions)[0]

        # 5. Default/Fallback: Choose Draw Stockpile if available, else first legal action
        if ActionDrawStockpile() in legal_actions:
            action = ActionDrawStockpile()
            logger.debug(
                "GreedyAgent P%d chose action (Default DrawStock): %s",
                self.player_id,
                action,
            )
            return action

        # Should only reach here if DrawStockpile not legal (e.g., empty + no reshuffle)
        # Or if some state wasn't handled above.
        chosen_action = list(legal_actions)[0]  # Failsafe: choose first available
        logger.warning(
            "GreedyAgent P%d falling back to first legal action: %s",
            self.player_id,
            chosen_action,
        )
        return chosen_action


class ImperfectMemoryMixin:
    """
    Mixin that provides an imperfect information memory model.

    Tracks which cards the agent has seen and maintains estimates for unseen slots.
    Memory model: dict mapping hand_index -> (card_value, turn_seen).
    Unknown cards are estimated at UNKNOWN_CARD_EXPECTED_VALUE.
    """

    def _init_memory(self, game_state: CambiaGameState):
        """Initialize memory from initial peek (bottom 2 cards, indices 0 and 1)."""
        # own_memory[slot_index] = card_value or None if unknown
        self.own_memory: Dict[int, Optional[int]] = {}
        # opponent_memory[slot_index] = card_value or None if unknown
        self.opponent_memory: Dict[int, Optional[int]] = {}
        self._current_turn: int = 0

        my_hand = game_state.get_player_hand(self.player_id)
        num_cards = len(my_hand)
        # Initialize all slots as unknown
        for i in range(num_cards):
            self.own_memory[i] = None

        # Peek initial_view_count cards from the bottom (lowest indices per deal order)
        peek_count = game_state.house_rules.initial_view_count
        for i in range(min(peek_count, num_cards)):
            if isinstance(my_hand[i], Card):
                self.own_memory[i] = my_hand[i].value

        opp_hand = game_state.get_player_hand(self.opponent_id)
        for i in range(len(opp_hand)):
            self.opponent_memory[i] = None

    def _estimate_own_hand_value(self) -> float:
        """Estimate own hand value using known cards + expected value for unknowns."""
        total = 0.0
        for slot, val in self.own_memory.items():
            if val is not None:
                total += val
            else:
                total += UNKNOWN_CARD_EXPECTED_VALUE
        return total

    def _get_own_known_value(self, slot_index: int) -> Optional[int]:
        """Return the known value of own card at slot, or None if unknown."""
        return self.own_memory.get(slot_index)

    def _get_opp_known_value(self, slot_index: int) -> Optional[int]:
        """Return the known value of opponent's card at slot, or None if unknown."""
        return self.opponent_memory.get(slot_index)

    def _update_memory_peek_own(self, slot_index: int, game_state: CambiaGameState):
        """Update memory when we peek our own card at slot_index."""
        my_hand = game_state.get_player_hand(self.player_id)
        if 0 <= slot_index < len(my_hand) and isinstance(my_hand[slot_index], Card):
            self.own_memory[slot_index] = my_hand[slot_index].value

    def _update_memory_peek_opp(self, slot_index: int, game_state: CambiaGameState):
        """Update memory when we peek opponent's card at slot_index."""
        opp_hand = game_state.get_player_hand(self.opponent_id)
        if 0 <= slot_index < len(opp_hand) and isinstance(opp_hand[slot_index], Card):
            self.opponent_memory[slot_index] = opp_hand[slot_index].value

    def _update_memory_replace_own(self, slot_index: int, new_card: Card):
        """Update memory when we replace own card with new_card."""
        self.own_memory[slot_index] = new_card.value

    def _mark_own_unknown(self, slot_index: int):
        """Mark own card slot as unknown (e.g., after blind swap)."""
        self.own_memory[slot_index] = None

    def _mark_opp_unknown(self, slot_index: int):
        """Mark opponent card slot as unknown."""
        self.opponent_memory[slot_index] = None

    def _find_highest_known_own_slot(self) -> Optional[int]:
        """Return slot index of the highest known own card, or None if all unknown."""
        best_slot = None
        best_val = -float("inf")
        for slot, val in self.own_memory.items():
            if val is not None and val > best_val:
                best_val = val
                best_slot = slot
        return best_slot

    def _find_unknown_own_slot(self) -> Optional[int]:
        """Return the first unknown own card slot, or None if all known."""
        for slot, val in self.own_memory.items():
            if val is None:
                return slot
        return None

    def _own_card_matches_discard(
        self, slot_index: int, game_state: CambiaGameState
    ) -> bool:
        """Return True if we know own card at slot matches the discard top."""
        known_val = self.own_memory.get(slot_index)
        if known_val is None:
            return False
        discard_top = game_state.get_discard_top()
        if discard_top is None:
            return False
        return known_val == discard_top.value

    def _opp_card_matches_discard(
        self, slot_index: int, game_state: CambiaGameState
    ) -> bool:
        """Return True if we know opponent's card at slot matches the discard top."""
        known_val = self.opponent_memory.get(slot_index)
        if known_val is None:
            return False
        discard_top = game_state.get_discard_top()
        if discard_top is None:
            return False
        return known_val == discard_top.value

    def _handle_ability_phase_imperfect(
        self,
        game_state: CambiaGameState,
        legal_actions: Set[GameAction],
    ) -> Optional[GameAction]:
        """
        Handle ability phases for imperfect info agents.
        Prioritize peeking unknown cards to gain information.
        Returns chosen action or None if not in an ability phase.
        """
        sample = next(iter(legal_actions), None)

        if isinstance(sample, ActionAbilityPeekOwnSelect):
            # Peek first unknown own card
            unknown_slot = self._find_unknown_own_slot()
            if unknown_slot is not None:
                action = ActionAbilityPeekOwnSelect(target_hand_index=unknown_slot)
                if action in legal_actions:
                    return action
            # Fallback: any legal peek own
            return next(
                (a for a in legal_actions if isinstance(a, ActionAbilityPeekOwnSelect)),
                sample,
            )

        if isinstance(sample, ActionAbilityPeekOtherSelect):
            # Peek first unknown opponent card for snap setup
            unknown_opp = next(
                (
                    slot
                    for slot, val in self.opponent_memory.items()
                    if val is None
                ),
                None,
            )
            if unknown_opp is not None:
                action = ActionAbilityPeekOtherSelect(
                    target_opponent_hand_index=unknown_opp
                )
                if action in legal_actions:
                    return action
            return next(
                (
                    a
                    for a in legal_actions
                    if isinstance(a, ActionAbilityPeekOtherSelect)
                ),
                sample,
            )

        if isinstance(sample, ActionAbilityBlindSwapSelect):
            # Swap own highest-known with an unknown opponent slot
            own_high_slot = self._find_highest_known_own_slot()
            opp_unknown_slot = next(
                (slot for slot, val in self.opponent_memory.items() if val is None),
                None,
            )
            if own_high_slot is not None and opp_unknown_slot is not None:
                action = ActionAbilityBlindSwapSelect(
                    own_hand_index=own_high_slot,
                    opponent_hand_index=opp_unknown_slot,
                )
                if action in legal_actions:
                    # After blind swap, we lose knowledge of our card
                    self._mark_own_unknown(own_high_slot)
                    return action
            # Fallback: (0, 0)
            return next(
                (
                    a
                    for a in legal_actions
                    if isinstance(a, ActionAbilityBlindSwapSelect)
                ),
                sample,
            )

        if isinstance(sample, ActionAbilityKingLookSelect):
            # Look at own highest-known and opponent unknown
            own_high_slot = self._find_highest_known_own_slot() or 0
            opp_unknown = next(
                (slot for slot, val in self.opponent_memory.items() if val is None),
                0,
            )
            action = ActionAbilityKingLookSelect(
                own_hand_index=own_high_slot,
                opponent_hand_index=opp_unknown,
            )
            if action in legal_actions:
                return action
            return next(
                (
                    a
                    for a in legal_actions
                    if isinstance(a, ActionAbilityKingLookSelect)
                ),
                sample,
            )

        if isinstance(sample, ActionAbilityKingSwapDecision):
            # We've peeked via KingLook — use game state pending data if available
            look_data = game_state.pending_action_data
            card1 = look_data.get("card1")  # own card
            card2 = look_data.get("card2")  # opp card
            if isinstance(card1, Card) and isinstance(card2, Card):
                if card2.value < card1.value:
                    return ActionAbilityKingSwapDecision(perform_swap=True)
            return ActionAbilityKingSwapDecision(perform_swap=False)

        return None  # Not in an ability phase

    def _handle_snap_move_imperfect(
        self,
        game_state: CambiaGameState,
        legal_actions: Set[GameAction],
    ) -> Optional[GameAction]:
        """
        Handle SnapOpponentMove phase: give lowest-known own card.
        Returns chosen action or None if not applicable.
        """
        sample = next(iter(legal_actions), None)
        if not isinstance(sample, ActionSnapOpponentMove):
            return None

        # Give away lowest known own card; if all unknown, give slot 0
        best_slot = None
        lowest_val = float("inf")
        for slot, val in self.own_memory.items():
            effective_val = val if val is not None else UNKNOWN_CARD_EXPECTED_VALUE
            if effective_val < lowest_val:
                lowest_val = effective_val
                best_slot = slot

        if best_slot is None:
            best_slot = 0

        target_slot = sample.target_empty_slot_index
        action = ActionSnapOpponentMove(
            own_card_to_move_hand_index=best_slot,
            target_empty_slot_index=target_slot,
        )
        if action in legal_actions:
            # Update memory: the moved own card now goes to opponent
            self.opponent_memory[target_slot] = self.own_memory.get(best_slot)
            # Remove the slot from own memory (hand shrinks, so renumber remaining)
            if best_slot in self.own_memory:
                del self.own_memory[best_slot]
            return action

        return next(
            (a for a in legal_actions if isinstance(a, ActionSnapOpponentMove)), sample
        )


class ImperfectGreedyAgent(ImperfectMemoryMixin, BaseAgent):
    """
    Greedy agent using imperfect information only.

    Tracks own cards from initial peek + subsequent peeks/swaps.
    For unseen own cards: estimates value as UNKNOWN_CARD_EXPECTED_VALUE (~6.5).
    Cannot see opponent hand — uses expected value for opponent decisions.
    Snap: only snaps own cards it has seen and knows match discard.
    """

    def __init__(self, player_id: int, config: Config):
        super().__init__(player_id, config)
        self.cambia_threshold = config.agents.greedy_agent.cambia_call_threshold
        self._initialized = False
        logger.info(
            "ImperfectGreedyAgent P%d initialized (Cambia Threshold: %d)",
            self.player_id,
            self.cambia_threshold,
        )

    def _ensure_initialized(self, game_state: CambiaGameState):
        if not self._initialized:
            self._init_memory(game_state)
            self._initialized = True

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        if not legal_actions:
            raise ValueError(
                f"ImperfectGreedyAgent P{self.player_id} cannot choose from empty legal actions."
            )
        self._ensure_initialized(game_state)

        # 1. Ability phases
        ability_action = self._handle_ability_phase_imperfect(game_state, legal_actions)
        if ability_action is not None:
            return ability_action

        # 2. SnapMove phase
        snap_move_action = self._handle_snap_move_imperfect(game_state, legal_actions)
        if snap_move_action is not None:
            return snap_move_action

        # 3. Snap phase: only snap own cards we know match
        if game_state.snap_phase_active:
            snap_own_actions = {
                a for a in legal_actions if isinstance(a, ActionSnapOwn)
            }
            for action in snap_own_actions:
                if self._own_card_matches_discard(action.own_card_hand_index, game_state):
                    return action
            # Pass snap — don't snap opponent or unknown own cards
            pass_action = ActionPassSnap()
            if pass_action in legal_actions:
                return pass_action
            return next(iter(legal_actions))

        # 4. Call Cambia if estimated hand value is low (uncertainty-adjusted threshold)
        if ActionCallCambia() in legal_actions:
            estimated_value = self._estimate_own_hand_value()
            num_unknown = sum(
                1 for v in self.own_memory.values() if v is None
            )
            # Adjust threshold upward for unknowns: unknown cards estimated at 6.5,
            # so add 5.5 per unknown to trigger when known portion is below threshold.
            adjusted_threshold = self.cambia_threshold + num_unknown * 5.5
            # _turn_number counts rounds (not moves). With 2 players & 46 move cap,
            # max _turn_number ≈ 23. Fall back at round 18 (~turn 36).
            turn_fallback = game_state._turn_number >= 18
            if estimated_value <= adjusted_threshold or turn_fallback:
                return ActionCallCambia()

        # 5. Post-draw: discard or replace
        if any(isinstance(a, (ActionDiscard, ActionReplace)) for a in legal_actions):
            drawn_card = game_state.pending_action_data.get("drawn_card")
            if not drawn_card or not isinstance(drawn_card, Card):
                return (
                    ActionDiscard(use_ability=False)
                    if ActionDiscard(use_ability=False) in legal_actions
                    else list(legal_actions)[0]
                )

            # Find best replacement among known slots
            best_replace: Optional[ActionReplace] = None
            max_reduction = 0  # Only replace if it strictly improves
            for slot, known_val in self.own_memory.items():
                if known_val is not None and drawn_card.value < known_val:
                    reduction = known_val - drawn_card.value
                    if reduction > max_reduction:
                        max_reduction = reduction
                        best_replace = ActionReplace(target_hand_index=slot)

            # If drawn card is low, replace an unknown slot
            if best_replace is None and drawn_card.value <= self.cambia_threshold:
                unknown_slot = self._find_unknown_own_slot()
                if unknown_slot is not None:
                    best_replace = ActionReplace(target_hand_index=unknown_slot)

            if best_replace and best_replace in legal_actions:
                self._update_memory_replace_own(
                    best_replace.target_hand_index, drawn_card
                )
                return best_replace

            return (
                ActionDiscard(use_ability=False)
                if ActionDiscard(use_ability=False) in legal_actions
                else list(legal_actions)[0]
            )

        # 6. Draw stockpile
        if ActionDrawStockpile() in legal_actions:
            return ActionDrawStockpile()

        chosen = list(legal_actions)[0]
        logger.warning(
            "ImperfectGreedyAgent P%d fallback to: %s", self.player_id, chosen
        )
        return chosen


class MemoryHeuristicAgent(ImperfectMemoryMixin, BaseAgent):
    """
    Human-like player with imperfect information.

    - Always draws from stockpile (more information gain).
    - Swaps: replaces highest known card if drawn < known; if drawn <= 3, replace any unknown.
    - Abilities: peek own unknowns first, then peek opponent for snap setup,
      blind swap own highest-known with opponent unknown.
    - Snaps own when confident (seen card matches discard).
    - Calls Cambia when estimated hand total <= cambia_threshold.
    """

    def __init__(self, player_id: int, config: Config):
        super().__init__(player_id, config)
        self.cambia_threshold = config.agents.greedy_agent.cambia_call_threshold
        self._initialized = False
        logger.info(
            "MemoryHeuristicAgent P%d initialized (Cambia Threshold: %d)",
            self.player_id,
            self.cambia_threshold,
        )

    def _ensure_initialized(self, game_state: CambiaGameState):
        if not self._initialized:
            self._init_memory(game_state)
            self._initialized = True

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        if not legal_actions:
            raise ValueError(
                f"MemoryHeuristicAgent P{self.player_id} cannot choose from empty legal actions."
            )
        self._ensure_initialized(game_state)

        # 1. Ability phases — prioritize info gathering
        ability_action = self._handle_ability_phase_imperfect(game_state, legal_actions)
        if ability_action is not None:
            return ability_action

        # 2. SnapMove phase
        snap_move_action = self._handle_snap_move_imperfect(game_state, legal_actions)
        if snap_move_action is not None:
            return snap_move_action

        # 3. Snap phase: only snap own cards we know match
        if game_state.snap_phase_active:
            snap_own_actions = {
                a for a in legal_actions if isinstance(a, ActionSnapOwn)
            }
            for action in snap_own_actions:
                if self._own_card_matches_discard(action.own_card_hand_index, game_state):
                    return action
            pass_action = ActionPassSnap()
            if pass_action in legal_actions:
                return pass_action
            return next(iter(legal_actions))

        # 4. Call Cambia when estimated hand total <= threshold (uncertainty-adjusted)
        if ActionCallCambia() in legal_actions:
            estimated_value = self._estimate_own_hand_value()
            num_unknown = sum(
                1 for v in self.own_memory.values() if v is None
            )
            adjusted_threshold = self.cambia_threshold + num_unknown * 5.5
            turn_fallback = game_state._turn_number >= 18
            if estimated_value <= adjusted_threshold or turn_fallback:
                return ActionCallCambia()

        # 5. Post-draw: discard or replace
        if any(isinstance(a, (ActionDiscard, ActionReplace)) for a in legal_actions):
            drawn_card = game_state.pending_action_data.get("drawn_card")
            if not drawn_card or not isinstance(drawn_card, Card):
                return (
                    ActionDiscard(use_ability=False)
                    if ActionDiscard(use_ability=False) in legal_actions
                    else list(legal_actions)[0]
                )

            # Replace highest known if drawn card is lower
            high_slot = self._find_highest_known_own_slot()
            if high_slot is not None:
                high_val = self.own_memory[high_slot]
                if high_val is not None and drawn_card.value < high_val:
                    action = ActionReplace(target_hand_index=high_slot)
                    if action in legal_actions:
                        self._update_memory_replace_own(high_slot, drawn_card)
                        return action

            # If drawn card <= 3, replace any unknown slot
            if drawn_card.value <= 3:
                unknown_slot = self._find_unknown_own_slot()
                if unknown_slot is not None:
                    action = ActionReplace(target_hand_index=unknown_slot)
                    if action in legal_actions:
                        self._update_memory_replace_own(unknown_slot, drawn_card)
                        return action

            # Default: discard (use ability if card has one)
            is_ability_card = drawn_card.rank in [SEVEN, EIGHT, NINE, TEN, KING]
            if is_ability_card and ActionDiscard(use_ability=True) in legal_actions:
                return ActionDiscard(use_ability=True)
            return (
                ActionDiscard(use_ability=False)
                if ActionDiscard(use_ability=False) in legal_actions
                else list(legal_actions)[0]
            )

        # 6. Always draw from stockpile
        if ActionDrawStockpile() in legal_actions:
            return ActionDrawStockpile()

        chosen = list(legal_actions)[0]
        logger.warning(
            "MemoryHeuristicAgent P%d fallback to: %s", self.player_id, chosen
        )
        return chosen


class AggressiveSnapAgent(ImperfectMemoryMixin, BaseAgent):
    """
    High-risk card elimination agent.

    - Actively seeks snap opportunities using peek abilities.
    - Snaps both own AND opponent cards when confident of match.
    - For opponent snaps: gives away lowest-value own card.
    - Uses King look-and-swap to offload known high cards.
    - Calls Cambia aggressively when hand size <= 2 or total <= 4.
    """

    CAMBIA_HAND_SIZE_THRESHOLD = 2
    CAMBIA_VALUE_THRESHOLD = 4

    def __init__(self, player_id: int, config: Config):
        super().__init__(player_id, config)
        self._initialized = False
        logger.info("AggressiveSnapAgent P%d initialized", self.player_id)

    def _ensure_initialized(self, game_state: CambiaGameState):
        if not self._initialized:
            self._init_memory(game_state)
            self._initialized = True

    def _handle_ability_phase_aggressive(
        self,
        game_state: CambiaGameState,
        legal_actions: Set[GameAction],
    ) -> Optional[GameAction]:
        """
        Aggressive ability handling: peek opponent cards first (for snap setup),
        then peek own unknowns. King: swap if beneficial.
        """
        sample = next(iter(legal_actions), None)

        if isinstance(sample, ActionAbilityPeekOwnSelect):
            # Peek unknown own first
            unknown = self._find_unknown_own_slot()
            if unknown is not None:
                action = ActionAbilityPeekOwnSelect(target_hand_index=unknown)
                if action in legal_actions:
                    return action
            return next(
                (a for a in legal_actions if isinstance(a, ActionAbilityPeekOwnSelect)),
                sample,
            )

        if isinstance(sample, ActionAbilityPeekOtherSelect):
            # Peek unknown opponent card for snap setup
            unknown_opp = next(
                (slot for slot, val in self.opponent_memory.items() if val is None),
                None,
            )
            if unknown_opp is not None:
                action = ActionAbilityPeekOtherSelect(
                    target_opponent_hand_index=unknown_opp
                )
                if action in legal_actions:
                    return action
            return next(
                (
                    a
                    for a in legal_actions
                    if isinstance(a, ActionAbilityPeekOtherSelect)
                ),
                sample,
            )

        if isinstance(sample, ActionAbilityBlindSwapSelect):
            # Swap own highest-known with opponent unknown
            own_high_slot = self._find_highest_known_own_slot()
            opp_unknown_slot = next(
                (slot for slot, val in self.opponent_memory.items() if val is None),
                None,
            )
            if own_high_slot is not None and opp_unknown_slot is not None:
                action = ActionAbilityBlindSwapSelect(
                    own_hand_index=own_high_slot,
                    opponent_hand_index=opp_unknown_slot,
                )
                if action in legal_actions:
                    self._mark_own_unknown(own_high_slot)
                    return action
            return next(
                (
                    a
                    for a in legal_actions
                    if isinstance(a, ActionAbilityBlindSwapSelect)
                ),
                sample,
            )

        if isinstance(sample, ActionAbilityKingLookSelect):
            # Look at own highest-known and unknown opponent slot
            own_high_slot = self._find_highest_known_own_slot() or 0
            opp_unknown = next(
                (slot for slot, val in self.opponent_memory.items() if val is None),
                0,
            )
            action = ActionAbilityKingLookSelect(
                own_hand_index=own_high_slot,
                opponent_hand_index=opp_unknown,
            )
            if action in legal_actions:
                return action
            return next(
                (a for a in legal_actions if isinstance(a, ActionAbilityKingLookSelect)),
                sample,
            )

        if isinstance(sample, ActionAbilityKingSwapDecision):
            look_data = game_state.pending_action_data
            card1 = look_data.get("card1")  # own card
            card2 = look_data.get("card2")  # opp card
            if isinstance(card1, Card) and isinstance(card2, Card):
                if card2.value < card1.value:
                    return ActionAbilityKingSwapDecision(perform_swap=True)
            return ActionAbilityKingSwapDecision(perform_swap=False)

        return None

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        if not legal_actions:
            raise ValueError(
                f"AggressiveSnapAgent P{self.player_id} cannot choose from empty legal actions."
            )
        self._ensure_initialized(game_state)

        # 1. Aggressive ability phases
        ability_action = self._handle_ability_phase_aggressive(game_state, legal_actions)
        if ability_action is not None:
            return ability_action

        # 2. SnapMove phase: give lowest-value own card
        snap_move_action = self._handle_snap_move_imperfect(game_state, legal_actions)
        if snap_move_action is not None:
            return snap_move_action

        # 3. Snap phase: snap own known matches AND opponent known matches
        if game_state.snap_phase_active:
            # Try own snaps first
            snap_own_actions = {
                a for a in legal_actions if isinstance(a, ActionSnapOwn)
            }
            for action in snap_own_actions:
                if self._own_card_matches_discard(action.own_card_hand_index, game_state):
                    return action

            # Try opponent snaps
            snap_opp_actions = {
                a for a in legal_actions if isinstance(a, ActionSnapOpponent)
            }
            for action in snap_opp_actions:
                if self._opp_card_matches_discard(
                    action.opponent_target_hand_index, game_state
                ):
                    return action

            pass_action = ActionPassSnap()
            if pass_action in legal_actions:
                return pass_action
            return next(iter(legal_actions))

        # 4. Aggressive Cambia: call if hand small, total low, or turn fallback
        if ActionCallCambia() in legal_actions:
            hand_size = len(self.own_memory)
            estimated_value = self._estimate_own_hand_value()
            num_unknown = sum(
                1 for v in self.own_memory.values() if v is None
            )
            adjusted_threshold = self.CAMBIA_VALUE_THRESHOLD + num_unknown * 5.5
            turn_fallback = game_state._turn_number >= 18
            if (
                hand_size <= self.CAMBIA_HAND_SIZE_THRESHOLD
                or estimated_value <= adjusted_threshold
                or turn_fallback
            ):
                return ActionCallCambia()

        # 5. Post-draw: replace high cards, otherwise discard with ability (to peek opponent)
        if any(isinstance(a, (ActionDiscard, ActionReplace)) for a in legal_actions):
            drawn_card = game_state.pending_action_data.get("drawn_card")
            if not drawn_card or not isinstance(drawn_card, Card):
                return (
                    ActionDiscard(use_ability=False)
                    if ActionDiscard(use_ability=False) in legal_actions
                    else list(legal_actions)[0]
                )

            # Replace highest known if drawn is lower
            high_slot = self._find_highest_known_own_slot()
            if high_slot is not None:
                high_val = self.own_memory[high_slot]
                if high_val is not None and drawn_card.value < high_val:
                    action = ActionReplace(target_hand_index=high_slot)
                    if action in legal_actions:
                        self._update_memory_replace_own(high_slot, drawn_card)
                        return action

            # Prefer peek-other ability to gain opponent information
            is_peek_other = drawn_card.rank in [NINE, TEN]
            if is_peek_other and ActionDiscard(use_ability=True) in legal_actions:
                return ActionDiscard(use_ability=True)

            # Use any ability if card has one
            is_ability_card = drawn_card.rank in [SEVEN, EIGHT, NINE, TEN, KING]
            if is_ability_card and ActionDiscard(use_ability=True) in legal_actions:
                return ActionDiscard(use_ability=True)

            return (
                ActionDiscard(use_ability=False)
                if ActionDiscard(use_ability=False) in legal_actions
                else list(legal_actions)[0]
            )

        # 6. Draw from stockpile
        if ActionDrawStockpile() in legal_actions:
            return ActionDrawStockpile()

        chosen = list(legal_actions)[0]
        logger.warning(
            "AggressiveSnapAgent P%d fallback to: %s", self.player_id, chosen
        )
        return chosen
