"""
src/game/_snap_mixin.py

Implements the snap phase logic mixin for the Cambia game engine.
Handles snap action validation, execution, penalties, and state transitions.
"""

import logging
import copy
from typing import Set, Optional, Deque, TYPE_CHECKING

from .types import StateDelta
from .helpers import serialize_card
from ..card import Card
from ..constants import (
    GameAction,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
)
from src.cfr.exceptions import ActionApplicationError

# Use TYPE_CHECKING for CambiaGameState hint to avoid circular import
if TYPE_CHECKING:
    from .engine import CambiaGameState

logger = logging.getLogger(__name__)


class SnapLogicMixin:
    """Mixin handling the Snap phase logic for CambiaGameState."""

    # --- Snap Phase Legal Actions ---

    def _get_legal_snap_actions(
        self: "CambiaGameState", acting_player: int
    ) -> Set[GameAction]:
        """Calculates legal actions during the snap phase for the acting player."""
        legal_actions: Set[GameAction] = set()

        if not self.snap_phase_active:
            logger.error(
                "Snap Logic: _get_legal_snap_actions called when snap phase inactive."
            )
            return legal_actions
        if not (
            0 <= acting_player < len(self.players)
            and hasattr(self.players[acting_player], "hand")
        ):
            logger.error(
                "Snap Logic: Acting player P%d invalid for legal actions.", acting_player
            )
            return legal_actions

        snapper_hand = self.players[acting_player].hand
        if self.snap_discarded_card is None or not isinstance(
            self.snap_discarded_card, Card
        ):
            logger.error(
                "Snap Logic: Snap phase active but snap_discarded_card is invalid: %s.",
                self.snap_discarded_card,
            )
            return legal_actions

        # NOTE: Per RULES.md Sec.5, a snap targets ANY known card, and a rank
        # mismatch is a legal-but-penalized attempt ("If you snap incorrectly...
        # you receive a 2-card penalty"), not an illegal action. Legal-action
        # generation therefore does NOT filter by rank -- it mirrors the Go
        # engine's legalSnapDecision (engine/legal.go), offering SnapOwn /
        # SnapOpponent for every hand slot. _handle_snap_action below already
        # resolves the rank check and applies the penalty on a mismatch for
        # both SnapOwn and SnapOpponent.

        # Basic validation of snapper's hand
        if not all(isinstance(card, Card) for card in snapper_hand):
            logger.error(
                "Snap Logic: P%d hand contains non-Card objects: %s",
                acting_player,
                snapper_hand,
            )
            # Still allow PassSnap even with invalid hand? Yes.
            legal_actions.add(ActionPassSnap())
            return legal_actions

        # Always possible to pass
        legal_actions.add(ActionPassSnap())

        # Own snaps: every hand slot is a legal (possibly wrong, penalized) target.
        for i, card in enumerate(snapper_hand):
            if isinstance(card, Card):
                legal_actions.add(ActionSnapOwn(own_card_hand_index=i))

        # Opponent snaps if allowed: every opponent hand slot is a legal target,
        # gated only on the snapper having a card to move into the vacated slot.
        if self.house_rules.allowOpponentSnapping:
            opponent_idx = self.get_opponent_index(acting_player)
            if not (
                0 <= opponent_idx < len(self.players)
                and hasattr(self.players[opponent_idx], "hand")
            ):
                logger.warning(
                    "Snap Logic: Opponent P%d invalid, cannot check for SnapOpponent.",
                    opponent_idx,
                )
            else:
                opponent_hand = self.players[opponent_idx].hand
                if not all(isinstance(card, Card) for card in opponent_hand):
                    logger.error(
                        "Snap Logic: Opponent P%d hand contains non-Card objects: %s",
                        opponent_idx,
                        opponent_hand,
                    )
                else:
                    if len(snapper_hand) > 0:
                        for i, card in enumerate(opponent_hand):
                            if isinstance(card, Card):
                                legal_actions.add(
                                    ActionSnapOpponent(opponent_target_hand_index=i)
                                )
                    else:
                        logger.debug(
                            "P%d cannot SnapOpponent, has no cards to move.",
                            acting_player,
                        )

        return legal_actions

    # --- Snap Phase Action Processing ---

    def _handle_snap_action(
        self: "CambiaGameState",
        action: GameAction,
        acting_player: int,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> bool:
        """
        Processes an action during the snap phase. Modifies state via _add_change.
        Returns True if action processed, False on critical error (e.g., wrong player).
        """
        if not self.snap_phase_active:
            logger.error("Snap Logic: _handle_snap_action called when inactive.")
            return False
        if not (0 <= self.snap_current_snapper_idx < len(self.snap_potential_snappers)):
            logger.error(
                "Snap Logic: Invalid snap_current_snapper_idx %d (potential: %d)",
                self.snap_current_snapper_idx,
                len(self.snap_potential_snappers),
            )
            self._end_snap_phase(undo_stack, delta_list)
            return True

        expected_player = self.snap_potential_snappers[self.snap_current_snapper_idx]
        if acting_player != expected_player:
            logger.error(
                "Snap Logic: Action %s from P%d, expected P%d. Ignoring.",
                action,
                acting_player,
                expected_player,
            )
            return False

        if self.snap_discarded_card is None or not isinstance(
            self.snap_discarded_card, Card
        ):
            logger.error(
                "Snap Logic: Cannot process snap action, snap_discarded_card invalid: %s.",
                self.snap_discarded_card,
            )
            self._end_snap_phase(undo_stack, delta_list)
            return True

        logger.debug(
            "Snap Phase (T%d): P%d acting with %s",
            self._turn_number,
            acting_player,
            action,
        )

        # Race-ON (snapRace=true): record this snapper's commit without mutating any
        # hand, then resolve the N-way race once every snapper has committed. This
        # mirrors the Go engine's dispatcher intercept (ApplyAction -> recordSnapCommit)
        # and leaves the race-OFF sequential resolution below byte-identical.
        if self.house_rules.snapRace:
            return self._handle_snap_race_commit(
                action, acting_player, undo_stack, delta_list
            )

        target_rank = self.snap_discarded_card.rank
        snap_success = False
        snap_penalty = False
        attempted_card_str: Optional[str] = None
        action_type_str = type(action).__name__
        card_to_log: Optional[Card] = None

        # Ensure necessary methods are available
        if not hasattr(self, "_add_change") or not callable(self._add_change):
            logger.critical("Snap Logic: _add_change method not found on self!")
            return False
        if not hasattr(self, "_apply_penalty") or not callable(self._apply_penalty):
            logger.critical("Snap Logic: _apply_penalty method not found on self!")
            return False

        # Helper for logging snap results
        def log_snap_result(details_dict):
            original_log = list(self.snap_results_log)

            def change():
                self.snap_results_log.append(details_dict)

            def undo():
                # Check if the last item is indeed the one we added
                assert (
                    self.snap_results_log and self.snap_results_log[-1] == details_dict
                ), f"Undo snap log mismatch. Expected last item {details_dict}, log is {self.snap_results_log}"
                self.snap_results_log.pop()
                assert self.snap_results_log == original_log, "Undo snap log failed"

            self._add_change(
                change, undo, ("snap_log_append", details_dict), undo_stack, delta_list
            )

        # Process Specific Snap Actions
        try:
            if isinstance(action, ActionPassSnap):
                logger.debug("P%d passes snap.", acting_player)
                # Log result handled below

            elif isinstance(action, ActionSnapOwn):
                snap_idx = action.own_card_hand_index
                hand = self.players[acting_player].hand
                original_hand_state = list(hand)  # Capture for undo check
                original_discard_len = len(self.discard_pile)
                original_discard_top = self.get_discard_top()

                if not (0 <= snap_idx < len(hand)):
                    logger.warning(
                        "P%d invalid Snap Own index: %d (Hand size: %d). Penalty.",
                        acting_player,
                        snap_idx,
                        len(hand),
                    )
                    snap_penalty = True
                    attempted_card_str = f"Invalid Index {snap_idx}"
                else:
                    attempted_card = hand[snap_idx]
                    if not isinstance(attempted_card, Card):
                        logger.error(
                            "SnapOwn Error: P%d item at index %d is not Card: %s.",
                            acting_player,
                            snap_idx,
                            attempted_card,
                        )
                        snap_penalty = True
                        attempted_card_str = repr(attempted_card)
                    elif attempted_card.rank == target_rank:
                        card_to_remove = attempted_card
                        card_to_log = card_to_remove  # Log the card being snapped

                        def change_snap_own():
                            # Check index and card identity before popping
                            if (
                                0 <= snap_idx < len(self.players[acting_player].hand)
                                and self.players[acting_player].hand[snap_idx]
                                is card_to_remove
                            ):
                                removed = self.players[acting_player].hand.pop(snap_idx)
                                self.discard_pile.append(removed)
                            else:
                                logger.error(
                                    "SnapOwn change error: Index %d OOB or card mismatch (Hand size %d). Expected %s, Got %s",
                                    snap_idx,
                                    len(self.players[acting_player].hand),
                                    card_to_remove,
                                    (
                                        self.players[acting_player].hand[snap_idx]
                                        if 0
                                        <= snap_idx
                                        < len(self.players[acting_player].hand)
                                        else "OOB"
                                    ),
                                )
                                # Avoid state modification if precondition fails

                        def undo_snap_own():
                            # Assert preconditions
                            assert (
                                self.discard_pile
                                and self.discard_pile[-1] is card_to_remove
                            ), f"Undo SnapOwn: Discard top mismatch (Expected {card_to_remove}, Got {self.discard_pile[-1] if self.discard_pile else 'Empty'})"

                            popped = self.discard_pile.pop()

                            assert (
                                0 <= snap_idx <= len(self.players[acting_player].hand)
                            ), f"Undo SnapOwn: Insert index {snap_idx} invalid for hand size {len(self.players[acting_player].hand)}"
                            self.players[acting_player].hand.insert(snap_idx, popped)
                            # Assert postconditions
                            assert len(self.discard_pile) == original_discard_len
                            assert self.get_discard_top() is original_discard_top
                            assert (
                                self.players[acting_player].hand == original_hand_state
                            ), f"Undo SnapOwn: Hand state mismatch! Expected {original_hand_state}, Got {self.players[acting_player].hand}"

                        delta_snap_own = (
                            "snap_own_success",
                            acting_player,
                            snap_idx,
                            serialize_card(card_to_remove),
                        )
                        self._add_change(
                            change_snap_own,
                            undo_snap_own,
                            delta_snap_own,
                            undo_stack,
                            delta_list,
                        )
                        snap_success = True
                        logger.info(
                            "P%d snaps own %s (Rank %s) from index %d. New Hand size: %d",
                            acting_player,
                            card_to_remove,
                            target_rank,
                            snap_idx,
                            len(self.players[acting_player].hand),
                        )
                    else:  # Card rank doesn't match
                        logger.warning(
                            "P%d invalid Snap Own: %s (Target: %s, Attempted: %s). Penalty.",
                            acting_player,
                            action,
                            target_rank,
                            attempted_card,
                        )
                        snap_penalty = True
                        attempted_card_str = serialize_card(attempted_card)

            elif isinstance(action, ActionSnapOpponent):
                if not self.house_rules.allowOpponentSnapping:
                    logger.warning(
                        "Invalid Action: SnapOpponent attempted by P%d but disallowed.",
                        acting_player,
                    )
                    snap_penalty = True
                    attempted_card_str = "Disallowed Action"
                elif len(self.players[acting_player].hand) == 0:
                    logger.warning(
                        "Invalid Action: SnapOpponent attempted by P%d with 0 cards (cannot move). Penalty.",
                        acting_player,
                    )
                    snap_penalty = True
                    attempted_card_str = "No cards to move"
                else:
                    opp_idx = self.get_opponent_index(acting_player)
                    if not (
                        0 <= opp_idx < len(self.players)
                        and hasattr(self.players[opp_idx], "hand")
                    ):
                        logger.error("SnapOpponent Error: Opponent P%d invalid.", opp_idx)
                        snap_penalty = True
                        attempted_card_str = f"Invalid Opponent {opp_idx}"
                    else:
                        opp_hand = self.players[opp_idx].hand
                        original_opp_hand_state = list(opp_hand)  # Capture for undo check
                        target_opp_hand_idx = action.opponent_target_hand_index
                        if not (0 <= target_opp_hand_idx < len(opp_hand)):
                            logger.warning(
                                "P%d invalid Snap Opponent index: %d (Opp Hand size: %d). Penalty.",
                                acting_player,
                                target_opp_hand_idx,
                                len(opp_hand),
                            )
                            snap_penalty = True
                            attempted_card_str = f"Invalid Index {target_opp_hand_idx}"
                        else:
                            attempted_card = opp_hand[target_opp_hand_idx]
                            if not isinstance(attempted_card, Card):
                                logger.error(
                                    "SnapOpponent Error: P%d target P%d index %d holds non-Card: %s.",
                                    acting_player,
                                    opp_idx,
                                    target_opp_hand_idx,
                                    attempted_card,
                                )
                                snap_penalty = True
                                attempted_card_str = repr(attempted_card)
                            elif attempted_card.rank == target_rank:
                                card_to_remove = attempted_card
                                card_to_log = card_to_remove  # Log card being removed
                                original_discard_len_opp = len(self.discard_pile)
                                original_discard_top_opp = self.get_discard_top()

                                def change_snap_opp_remove():
                                    # Check index validity and card identity before popping
                                    if (
                                        0
                                        <= target_opp_hand_idx
                                        < len(self.players[opp_idx].hand)
                                        and self.players[opp_idx].hand[
                                            target_opp_hand_idx
                                        ]
                                        is card_to_remove
                                    ):
                                        removed = self.players[opp_idx].hand.pop(
                                            target_opp_hand_idx
                                        )
                                        # Per RULES.md Sec.5, a snapped card (own or
                                        # opponent's) goes onto the discard pile --
                                        # mirrors change_snap_own above and the Go
                                        # engine's snapOpponent (DiscardPile[DiscardLen]
                                        # = card; DiscardLen++). Previously omitted here:
                                        # the card vanished instead of joining discard,
                                        # silently desyncing discard-top/reshuffle
                                        # material from the Go engine.
                                        self.discard_pile.append(removed)
                                    else:
                                        logger.error(
                                            "SnapOpponent change error: Index %d OOB or card mismatch (Opp Hand size %d). Expected %s, Got %s",
                                            target_opp_hand_idx,
                                            len(self.players[opp_idx].hand),
                                            card_to_remove,
                                            (
                                                self.players[opp_idx].hand[
                                                    target_opp_hand_idx
                                                ]
                                                if 0
                                                <= target_opp_hand_idx
                                                < len(self.players[opp_idx].hand)
                                                else "OOB"
                                            ),
                                        )
                                        # Avoid state change on error

                                def undo_snap_opp_remove():
                                    # Assert preconditions (optional, but good practice)
                                    assert (
                                        self.discard_pile
                                        and self.discard_pile[-1] is card_to_remove
                                    ), f"Undo SnapOpponentRemove: Discard top mismatch (Expected {card_to_remove}, Got {self.discard_pile[-1] if self.discard_pile else 'Empty'})"

                                    self.discard_pile.pop()

                                    assert (
                                        0
                                        <= target_opp_hand_idx
                                        <= len(self.players[opp_idx].hand)
                                    ), f"Undo SnapOpponentRemove: Insert index {target_opp_hand_idx} invalid for opp hand size {len(self.players[opp_idx].hand)}"

                                    self.players[opp_idx].hand.insert(
                                        target_opp_hand_idx, card_to_remove
                                    )
                                    # Assert postconditions
                                    assert (
                                        self.players[opp_idx].hand
                                        == original_opp_hand_state
                                    ), f"Undo SnapOpponentRemove: Hand state mismatch! Expected {original_opp_hand_state}, Got {self.players[opp_idx].hand}"
                                    assert (
                                        len(self.discard_pile) == original_discard_len_opp
                                    )
                                    assert (
                                        self.get_discard_top() is original_discard_top_opp
                                    )

                                delta_snap_opp_remove = (
                                    "snap_opponent_remove",
                                    opp_idx,
                                    target_opp_hand_idx,
                                    serialize_card(card_to_remove),
                                )
                                self._add_change(
                                    change_snap_opp_remove,
                                    undo_snap_opp_remove,
                                    delta_snap_opp_remove,
                                    undo_stack,
                                    delta_list,
                                )

                                snap_success = True
                                logger.info(
                                    "P%d snaps opponent P%d's %s at index %d. Requires move.",
                                    acting_player,
                                    opp_idx,
                                    card_to_remove,
                                    target_opp_hand_idx,
                                )

                                # Set pending action for the MOVE step
                                original_pending = (
                                    self.pending_action,
                                    self.pending_action_player,
                                    copy.deepcopy(self.pending_action_data),
                                )
                                original_snap_active = (
                                    self.snap_phase_active
                                )  # Should be True
                                next_pending_action_type = ActionSnapOpponentMove(
                                    own_card_to_move_hand_index=-1,  # Placeholder
                                    target_empty_slot_index=target_opp_hand_idx,  # Set correct target
                                )
                                new_pending_data = {
                                    "target_empty_slot_index": target_opp_hand_idx
                                }

                                def change_pending_move():
                                    self.pending_action = next_pending_action_type
                                    self.pending_action_player = acting_player
                                    self.pending_action_data = new_pending_data
                                    self.snap_phase_active = (
                                        False  # Move happens outside snap phase
                                    )

                                def undo_pending_move():
                                    # Assert preconditions (optional)
                                    assert self.pending_action is next_pending_action_type
                                    assert self.pending_action_player == acting_player
                                    # Restore previous state
                                    (
                                        self.pending_action,
                                        self.pending_action_player,
                                        self.pending_action_data,
                                    ) = original_pending
                                    self.snap_phase_active = original_snap_active

                                prev_pending_type_name = (
                                    type(original_pending[0]).__name__
                                    if original_pending[0]
                                    else None
                                )
                                serialized_orig_data = {
                                    k: serialize_card(v) if isinstance(v, Card) else v
                                    for k, v in original_pending[2].items()
                                }
                                serialized_new_data = {
                                    "target_empty_slot_index": target_opp_hand_idx
                                }

                                delta_pending = (
                                    "set_pending_action",
                                    type(next_pending_action_type).__name__,
                                    acting_player,
                                    serialized_new_data,
                                    prev_pending_type_name,
                                    original_pending[1],
                                    serialized_orig_data,
                                )
                                delta_snap_active = (
                                    "set_attr",
                                    "snap_phase_active",
                                    False,
                                    original_snap_active,
                                )
                                self._add_change(
                                    change_pending_move,
                                    undo_pending_move,
                                    delta_pending,
                                    undo_stack,
                                    delta_list,
                                )
                                delta_list.append(
                                    delta_snap_active
                                )  # Log snap phase change separately

                                # SnapRace: explicitly clear remaining snap state so
                                # other snappers forfeit. snap_phase_active already False.
                                if self.house_rules.snapRace:
                                    orig_sr_potentials = list(
                                        self.snap_potential_snappers
                                    )
                                    orig_sr_card = self.snap_discarded_card
                                    orig_sr_idx = self.snap_current_snapper_idx

                                    def change_snap_race_opp_clear():
                                        self.snap_potential_snappers = []
                                        self.snap_discarded_card = None
                                        self.snap_current_snapper_idx = 0

                                    def undo_snap_race_opp_clear():
                                        self.snap_potential_snappers = orig_sr_potentials
                                        self.snap_discarded_card = orig_sr_card
                                        self.snap_current_snapper_idx = orig_sr_idx

                                    self._add_change(
                                        change_snap_race_opp_clear,
                                        undo_snap_race_opp_clear,
                                        (
                                            "snap_race_opp_clear",
                                            acting_player,
                                        ),
                                        undo_stack,
                                        delta_list,
                                    )

                            else:  # Card rank doesn't match
                                logger.warning(
                                    "P%d invalid Snap Opponent: %s (Target: %s, Attempted: %s). Penalty.",
                                    acting_player,
                                    action,
                                    target_rank,
                                    attempted_card,
                                )
                                snap_penalty = True
                                attempted_card_str = serialize_card(attempted_card)

            else:  # Invalid action type during snap phase
                logger.error(
                    "Invalid action type %s received during snap phase processing from P%d.",
                    type(action).__name__,
                    acting_player,
                )
                # Don't apply penalty for engine error, just log and advance snap turn

            # Log result (common logic for all snap attempts)
            log_details = {
                "snapper": acting_player,
                "action_type": action_type_str,
                "target_rank": target_rank,
                "success": snap_success,
                "penalty": snap_penalty,
            }
            if card_to_log:
                log_details["snapped_card"] = serialize_card(card_to_log)
            if attempted_card_str:
                log_details["attempted_card_str"] = attempted_card_str
            if isinstance(action, ActionSnapOwn):
                log_details["removed_own_index"] = (
                    action.own_card_hand_index if snap_success else None
                )
            if isinstance(action, ActionSnapOpponent):
                log_details["removed_opponent_index"] = (
                    action.opponent_target_hand_index if snap_success else None
                )
            log_snap_result(log_details)

            # Apply penalty if needed (after logging)
            if snap_penalty:
                penalty_deltas = self._apply_penalty(
                    acting_player, self.house_rules.penaltyDrawCount, undo_stack
                )
                delta_list.extend(penalty_deltas)

            # Return immediately if waiting for MOVE action
            if isinstance(action, ActionSnapOpponent) and snap_success:
                return True  # Move action is now pending

        except ActionApplicationError:
            # Re-raise action application errors
            raise
        except Exception as e_snap_handle:
            logger.exception(
                "Error handling snap action %s for P%d: %s",
                action,
                acting_player,
                e_snap_handle,
            )
            raise ActionApplicationError(
                f"Snap action handling failed for {action}"
            ) from e_snap_handle

        # --- Advance Snap Turn or End Phase ---
        # (Do not advance if ActionSnapOpponent succeeded and set a pending move)
        if not (isinstance(action, ActionSnapOpponent) and snap_success):
            # If snap succeeded and snap race is enabled, end the phase immediately
            # (remaining snappers forfeit their chance)
            if snap_success and self.house_rules.snapRace:
                self._end_snap_phase(undo_stack, delta_list)
            else:
                try:
                    original_snap_idx_local = self.snap_current_snapper_idx
                    next_snap_idx = original_snap_idx_local + 1

                    def change_snap_idx():
                        self.snap_current_snapper_idx = next_snap_idx

                    def undo_snap_idx():
                        # Assert precondition
                        assert self.snap_current_snapper_idx == next_snap_idx
                        self.snap_current_snapper_idx = original_snap_idx_local

                    self._add_change(
                        change_snap_idx,
                        undo_snap_idx,
                        (
                            "set_attr",
                            "snap_current_snapper_idx",
                            next_snap_idx,
                            original_snap_idx_local,
                        ),
                        undo_stack,
                        delta_list,
                    )

                    if next_snap_idx >= len(self.snap_potential_snappers):
                        self._end_snap_phase(undo_stack, delta_list)
                    # else: Next snapper's turn
                except ActionApplicationError:
                    raise
                except Exception as e_advance_snap:
                    # JUSTIFIED: Catch errors advancing snap turn to attempt cleanup
                    logger.exception(
                        "Error advancing snap turn index: %s", e_advance_snap
                    )
                    self._end_snap_phase(undo_stack, delta_list)  # Attempt cleanup

        return True  # Action was processed (passed, snapped, penalized, or errored out but handled)

    # --- Race-ON (snapRace) snap resolution ---
    # Mirrors the Go engine's snap_race.go: a true N-way race with a simultaneous
    # imperfect-info commit (a commit mutates no hand), one uniform-random winner
    # among the willing committers, and a penalty for every losing willing committer.

    def _handle_snap_race_commit(
        self: "CambiaGameState",
        action: GameAction,
        acting_player: int,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> bool:
        """Record one snapper's commit (no hand mutation) and, once the final
        snapper has committed, resolve the race. Mirrors Go recordSnapCommit +
        advanceSnapper."""
        idx = self.snap_current_snapper_idx
        n = len(self.snap_potential_snappers)

        # Record the commit into the parallel commit buffer.
        orig_commits = list(self.snap_commits)
        if idx == 0 or len(self.snap_commits) != n:
            base = [None] * n
        else:
            base = list(self.snap_commits)
        new_commits = list(base)
        if 0 <= idx < n:
            new_commits[idx] = action

        def change_commit():
            self.snap_commits = new_commits

        def undo_commit():
            self.snap_commits = orig_commits

        self._add_change(
            change_commit,
            undo_commit,
            ("snap_race_commit", acting_player, type(action).__name__),
            undo_stack,
            delta_list,
        )

        # Advance to the next committer (mirrors the race-OFF snapper advance).
        original_idx = idx
        next_idx = idx + 1

        def change_idx():
            self.snap_current_snapper_idx = next_idx

        def undo_idx():
            self.snap_current_snapper_idx = original_idx

        self._add_change(
            change_idx,
            undo_idx,
            ("set_attr", "snap_current_snapper_idx", next_idx, original_idx),
            undo_stack,
            delta_list,
        )

        if next_idx >= n:
            self._resolve_snap_race(undo_stack, delta_list)
        return True

    def _resolve_snap_race(
        self: "CambiaGameState", undo_stack: Deque, delta_list: StateDelta
    ) -> None:
        """Draw the uniform-random winner among willing committers, penalize the
        losers, and resolve the winner's snap. Mirrors Go resolveSnapRace."""
        n = len(self.snap_potential_snappers)
        willing = [
            i
            for i in range(n)
            if i < len(self.snap_commits)
            and self.snap_commits[i] is not None
            and not isinstance(self.snap_commits[i], ActionPassSnap)
        ]
        if not willing:
            # Everyone passed: no snap, no penalty.
            self._end_snap_phase(undo_stack, delta_list)
            return

        # Uniform-random winner among willing committers. randint(0, k-1) is
        # duck-compatible with both random.Random and GoXorShift64Rng, and on the
        # Go-synced RNG consumes exactly one draw == Go's randN(len(willing)).
        win_local = self._rng.randint(0, len(willing) - 1)
        win_idx = willing[win_local]

        # Penalize the losing willing committers first (matches Go's loser loop),
        # then resolve the winner. Penalty draws only append to a hand, so the
        # winner's committed indices stay valid.
        for li in willing:
            if li == win_idx:
                continue
            loser = self.snap_potential_snappers[li]
            penalty_deltas = self._apply_penalty(
                loser, self.house_rules.penaltyDrawCount, undo_stack
            )
            delta_list.extend(penalty_deltas)

        winner = self.snap_potential_snappers[win_idx]
        win_action = self.snap_commits[win_idx]
        if isinstance(win_action, ActionSnapOpponent):
            pending = self._resolve_winner_snap_opp(
                winner, win_action, undo_stack, delta_list
            )
        else:
            pending = self._resolve_winner_snap_own(
                winner, win_action, undo_stack, delta_list
            )

        if pending:
            # Winning opponent snap: the winner's move is pending and the snap phase
            # is already deactivated; do not end the phase (mirrors Go leaving a
            # PendingSnapMove that the winner's move then completes).
            return
        self._end_snap_phase(undo_stack, delta_list)

    def _log_snap_race_result(
        self: "CambiaGameState",
        snapper,
        action_type_str,
        target_rank,
        snap_success,
        snap_penalty,
        card_to_log,
        attempted_card_str,
        undo_stack,
        delta_list,
        removed_own_index=None,
        removed_opponent_index=None,
    ) -> None:
        """Append a snap result to snap_results_log (mirrors the race-OFF logger)."""
        log_details = {
            "snapper": snapper,
            "action_type": action_type_str,
            "target_rank": target_rank,
            "success": snap_success,
            "penalty": snap_penalty,
        }
        if card_to_log:
            log_details["snapped_card"] = serialize_card(card_to_log)
        if attempted_card_str:
            log_details["attempted_card_str"] = attempted_card_str
        if action_type_str == "ActionSnapOwn":
            log_details["removed_own_index"] = removed_own_index
        if action_type_str == "ActionSnapOpponent":
            log_details["removed_opponent_index"] = removed_opponent_index

        def change():
            self.snap_results_log.append(log_details)

        def undo():
            self.snap_results_log.pop()

        self._add_change(
            change, undo, ("snap_log_append", log_details), undo_stack, delta_list
        )

    def _resolve_winner_snap_own(
        self: "CambiaGameState", winner, action, undo_stack, delta_list
    ) -> bool:
        """Resolve a winning snap-own commit. Returns False (no pending move).
        Mirrors the race-OFF ActionSnapOwn resolution body."""
        target_rank = self.snap_discarded_card.rank
        snap_idx = action.own_card_hand_index
        hand = self.players[winner].hand
        original_hand_state = list(hand)
        snap_success = False
        snap_penalty = False
        card_to_log = None
        attempted_card_str = None

        if not (0 <= snap_idx < len(hand)):
            snap_penalty = True
            attempted_card_str = f"Invalid Index {snap_idx}"
        else:
            attempted_card = hand[snap_idx]
            if not isinstance(attempted_card, Card):
                snap_penalty = True
                attempted_card_str = repr(attempted_card)
            elif attempted_card.rank == target_rank:
                card_to_remove = attempted_card
                card_to_log = card_to_remove

                def change_snap_own():
                    if (
                        0 <= snap_idx < len(self.players[winner].hand)
                        and self.players[winner].hand[snap_idx] is card_to_remove
                    ):
                        removed = self.players[winner].hand.pop(snap_idx)
                        self.discard_pile.append(removed)

                def undo_snap_own():
                    assert self.discard_pile and self.discard_pile[-1] is card_to_remove
                    popped = self.discard_pile.pop()
                    self.players[winner].hand.insert(snap_idx, popped)
                    assert self.players[winner].hand == original_hand_state

                self._add_change(
                    change_snap_own,
                    undo_snap_own,
                    ("snap_own_success", winner, snap_idx, serialize_card(card_to_remove)),
                    undo_stack,
                    delta_list,
                )
                snap_success = True
            else:
                snap_penalty = True
                attempted_card_str = serialize_card(attempted_card)

        self._log_snap_race_result(
            winner,
            "ActionSnapOwn",
            target_rank,
            snap_success,
            snap_penalty,
            card_to_log,
            attempted_card_str,
            undo_stack,
            delta_list,
            removed_own_index=(snap_idx if snap_success else None),
        )
        if snap_penalty:
            penalty_deltas = self._apply_penalty(
                winner, self.house_rules.penaltyDrawCount, undo_stack
            )
            delta_list.extend(penalty_deltas)
        return False

    def _resolve_winner_snap_opp(
        self: "CambiaGameState", winner, action, undo_stack, delta_list
    ) -> bool:
        """Resolve a winning snap-opponent commit. Returns True if a pending move
        was set (success), else False. Mirrors the race-OFF ActionSnapOpponent body."""
        target_rank = self.snap_discarded_card.rank
        snap_success = False
        snap_penalty = False
        card_to_log = None
        attempted_card_str = None
        pending_set = False

        if not self.house_rules.allowOpponentSnapping:
            snap_penalty = True
            attempted_card_str = "Disallowed Action"
        elif len(self.players[winner].hand) == 0:
            snap_penalty = True
            attempted_card_str = "No cards to move"
        else:
            opp_idx = self.get_opponent_index(winner)
            if not (
                0 <= opp_idx < len(self.players)
                and hasattr(self.players[opp_idx], "hand")
            ):
                snap_penalty = True
                attempted_card_str = f"Invalid Opponent {opp_idx}"
            else:
                opp_hand = self.players[opp_idx].hand
                original_opp_hand_state = list(opp_hand)
                target_opp_hand_idx = action.opponent_target_hand_index
                if not (0 <= target_opp_hand_idx < len(opp_hand)):
                    snap_penalty = True
                    attempted_card_str = f"Invalid Index {target_opp_hand_idx}"
                else:
                    attempted_card = opp_hand[target_opp_hand_idx]
                    if not isinstance(attempted_card, Card):
                        snap_penalty = True
                        attempted_card_str = repr(attempted_card)
                    elif attempted_card.rank == target_rank:
                        card_to_remove = attempted_card
                        card_to_log = card_to_remove

                        def change_snap_opp_remove():
                            if (
                                0
                                <= target_opp_hand_idx
                                < len(self.players[opp_idx].hand)
                                and self.players[opp_idx].hand[target_opp_hand_idx]
                                is card_to_remove
                            ):
                                removed = self.players[opp_idx].hand.pop(
                                    target_opp_hand_idx
                                )
                                self.discard_pile.append(removed)

                        def undo_snap_opp_remove():
                            assert (
                                self.discard_pile
                                and self.discard_pile[-1] is card_to_remove
                            )
                            self.discard_pile.pop()
                            self.players[opp_idx].hand.insert(
                                target_opp_hand_idx, card_to_remove
                            )
                            assert (
                                self.players[opp_idx].hand == original_opp_hand_state
                            )

                        self._add_change(
                            change_snap_opp_remove,
                            undo_snap_opp_remove,
                            (
                                "snap_opponent_remove",
                                opp_idx,
                                target_opp_hand_idx,
                                serialize_card(card_to_remove),
                            ),
                            undo_stack,
                            delta_list,
                        )
                        snap_success = True

                        # Set the pending move, deactivate the snap phase, and clear
                        # snap state so no further committer resumes (the winner's
                        # move completes via the pending-action path).
                        original_pending = (
                            self.pending_action,
                            self.pending_action_player,
                            copy.deepcopy(self.pending_action_data),
                        )
                        original_snap_active = self.snap_phase_active
                        orig_potentials = list(self.snap_potential_snappers)
                        orig_card = self.snap_discarded_card
                        orig_idx = self.snap_current_snapper_idx
                        next_pending = ActionSnapOpponentMove(
                            own_card_to_move_hand_index=-1,
                            target_empty_slot_index=target_opp_hand_idx,
                        )
                        new_pending_data = {
                            "target_empty_slot_index": target_opp_hand_idx
                        }

                        def change_pending_move():
                            self.pending_action = next_pending
                            self.pending_action_player = winner
                            self.pending_action_data = new_pending_data
                            self.snap_phase_active = False
                            self.snap_potential_snappers = []
                            self.snap_discarded_card = None
                            self.snap_current_snapper_idx = 0

                        def undo_pending_move():
                            (
                                self.pending_action,
                                self.pending_action_player,
                                self.pending_action_data,
                            ) = original_pending
                            self.snap_phase_active = original_snap_active
                            self.snap_potential_snappers = orig_potentials
                            self.snap_discarded_card = orig_card
                            self.snap_current_snapper_idx = orig_idx

                        self._add_change(
                            change_pending_move,
                            undo_pending_move,
                            (
                                "set_pending_action",
                                type(next_pending).__name__,
                                winner,
                                new_pending_data,
                                type(original_pending[0]).__name__
                                if original_pending[0]
                                else None,
                                original_pending[1],
                                {},
                            ),
                            undo_stack,
                            delta_list,
                        )
                        pending_set = True
                    else:
                        snap_penalty = True
                        attempted_card_str = serialize_card(attempted_card)

        self._log_snap_race_result(
            winner,
            "ActionSnapOpponent",
            target_rank,
            snap_success,
            snap_penalty,
            card_to_log,
            attempted_card_str,
            undo_stack,
            delta_list,
            removed_opponent_index=(
                action.opponent_target_hand_index if snap_success else None
            ),
        )
        if snap_penalty:
            penalty_deltas = self._apply_penalty(
                winner, self.house_rules.penaltyDrawCount, undo_stack
            )
            delta_list.extend(penalty_deltas)
        return pending_set

    # --- Snap Phase Initiation and Termination ---

    def _initiate_snap_phase(
        self: "CambiaGameState",
        discarded_card: Card,
        undo_stack: Deque,
        delta_list: StateDelta,
    ) -> bool:
        """Checks if discard triggers snap phase, sets up state if so."""
        if not isinstance(discarded_card, Card):
            logger.error(
                "Cannot initiate snap phase: discarded_card is not a Card object (%s).",
                discarded_card,
            )
            return False

        potential_indices = []
        target_rank = discarded_card.rank
        # Player whose action led to this discard (need to get this from engine context if not passed)
        # Assuming self.current_player_index holds the player who just finished their *main* action (discard/replace)
        # Need to determine who discarded based on pending action context or last action?
        # Let's assume for now the discarder is the *opponent* of the current turn player if turn advanced,
        # or the current player if turn didn't advance (e.g., just finished discard).
        # THIS IS AMBIGUOUS - Assume self.current_player_index is the one whose turn *will be next*.
        discarder_player = (
            self.current_player_index - 1 + self.num_players
        ) % self.num_players
        logger.debug(
            "Initiating snap check. Card: %s, Discarder (Assumed Prev Player): P%d",
            discarded_card,
            discarder_player,
        )

        for p_idx in range(self.num_players):
            if p_idx == self.cambia_caller_id:
                continue  # Cambia caller cannot snap

            if not (
                0 <= p_idx < len(self.players) and hasattr(self.players[p_idx], "hand")
            ):
                logger.warning("Initiate Snap Check: P%d invalid. Skipping.", p_idx)
                continue

            hand = self.players[p_idx].hand
            if not all(isinstance(card, Card) for card in hand):
                logger.error(
                    "Initiate Snap Check: P%d hand invalid: %s. Skipping.", p_idx, hand
                )
                continue

            can_snap_own = any(card.rank == target_rank for card in hand)
            can_snap_opponent = False
            if (
                self.house_rules.allowOpponentSnapping and len(hand) > 0
            ):  # Must have card to move
                opp_snap_check_idx = self.get_opponent_index(p_idx)
                # Opponent cannot snap if they are the Cambia caller
                if opp_snap_check_idx != self.cambia_caller_id:
                    if 0 <= opp_snap_check_idx < len(self.players) and hasattr(
                        self.players[opp_snap_check_idx], "hand"
                    ):
                        opp_hand_snap_check = self.players[opp_snap_check_idx].hand
                        if not all(
                            isinstance(card, Card) for card in opp_hand_snap_check
                        ):
                            logger.error(
                                "Initiate Snap Check: Opponent P%d hand invalid. Skipping snap-opp for P%d.",
                                opp_snap_check_idx,
                                p_idx,
                            )
                        else:
                            can_snap_opponent = any(
                                card.rank == target_rank for card in opp_hand_snap_check
                            )
                    else:
                        logger.warning(
                            "Initiate Snap Check: Opponent P%d invalid for P%d checking SnapOpponent.",
                            opp_snap_check_idx,
                            p_idx,
                        )

            if can_snap_own or can_snap_opponent:
                potential_indices.append(p_idx)

        if not potential_indices:
            logger.debug(
                "No potential snappers found for discard of %s (Rank %s).",
                discarded_card,
                target_rank,
            )
            return False

        # Determine snap order: Start from player *after* the discarder.
        ordered_snappers = []
        # Check player after discarder first
        start_check_idx = (discarder_player + 1) % self.num_players
        if start_check_idx in potential_indices:
            ordered_snappers.append(start_check_idx)
        # Check discarder last (if eligible and not Cambia caller)
        if (
            discarder_player in potential_indices
            and discarder_player != self.cambia_caller_id
            and discarder_player != start_check_idx  # Avoid adding twice in 2p game
        ):
            ordered_snappers.append(discarder_player)

        if not ordered_snappers:
            logger.debug(
                "Potential snappers list empty after ordering/filtering. No snap phase."
            )
            return False

        # Start Snap Phase
        logger.info(
            "Snap phase started. Discard: %s. Snappers (ordered): %s.",
            discarded_card,
            ordered_snappers,
        )
        original_snap_phase = self.snap_phase_active
        original_snap_card = self.snap_discarded_card
        original_snap_potentials = list(self.snap_potential_snappers)
        original_snap_idx = self.snap_current_snapper_idx
        original_snap_log = list(self.snap_results_log)

        def change_snap_start():
            self.snap_phase_active = True
            self.snap_discarded_card = discarded_card
            self.snap_potential_snappers = ordered_snappers
            self.snap_current_snapper_idx = 0
            self.snap_results_log = []  # Clear log for new snap phase

        def undo_snap_start():
            # Assert preconditions
            assert self.snap_phase_active is True
            assert self.snap_discarded_card is discarded_card
            # Restore previous state
            self.snap_phase_active = original_snap_phase
            self.snap_discarded_card = original_snap_card
            self.snap_potential_snappers = original_snap_potentials
            self.snap_current_snapper_idx = original_snap_idx
            self.snap_results_log = original_snap_log
            logger.debug("Undo snap start.")

        delta_snap_start = (
            "start_snap_phase",
            serialize_card(discarded_card),
            ordered_snappers,
        )
        self._add_change(
            change_snap_start, undo_snap_start, delta_snap_start, undo_stack, delta_list
        )
        return True

    def _end_snap_phase(
        self: "CambiaGameState", undo_stack: Deque, delta_list: StateDelta
    ):
        """Cleans up snap phase state and advances the main game turn."""
        if not self.snap_phase_active:
            return

        logger.debug("Ending snap phase.")
        original_snap_phase = self.snap_phase_active
        original_snap_card = self.snap_discarded_card
        original_snap_potentials = list(self.snap_potential_snappers)
        original_snap_idx = self.snap_current_snapper_idx
        original_snap_log = list(
            self.snap_results_log
        )  # Capture log before potential clear

        def change_snap_end():
            self.snap_phase_active = False
            self.snap_discarded_card = None
            self.snap_potential_snappers = []
            self.snap_current_snapper_idx = 0
            self.snap_results_log = []  # Clear the log upon ending the phase

        def undo_snap_end():
            # Assert preconditions
            assert self.snap_phase_active is False
            # Restore previous state
            self.snap_phase_active = original_snap_phase
            self.snap_discarded_card = original_snap_card
            self.snap_potential_snappers = original_snap_potentials
            self.snap_current_snapper_idx = original_snap_idx
            self.snap_results_log = original_snap_log  # Restore the log
            logger.debug("Undo snap end.")

        delta_snap_end = ("end_snap_phase",)  # Log that the phase ended
        # Add a separate delta for clearing the log, if needed for fine-grained replay
        # For simplicity, the change_snap_end handles it.
        self._add_change(
            change_snap_end, undo_snap_end, delta_snap_end, undo_stack, delta_list
        )

        # Crucially, the turn advances *after* the snap phase concludes.
        if hasattr(self, "_advance_turn") and callable(self._advance_turn):
            logger.debug("Snap phase ended, advancing main game turn...")
            self._advance_turn(undo_stack, delta_list)
        else:
            logger.error(
                "Snap Logic: Cannot advance turn after ending snap phase - _advance_turn missing."
            )

    def _flush_snap_results_log(
        self: "CambiaGameState", undo_stack: Deque, delta_list: StateDelta
    ) -> None:
        """Clears snap_results_log once its contents have been exposed to the
        single observation that should see them.

        Mirrors the Go engine's tokenizer semantics (engine/agent/tokens.go
        Observe(): "if Snap.Active { emit the accumulated frames } else {
        silently reset }"). For a normal (non-SnapOpponent) last-snapper
        action, Go's Snap.Active goes false within the SAME ApplyAction call
        that appended the final entry, so no external reader (Go's own
        Observe(), called separately afterward) ever sees the accumulated
        block -- _end_snap_phase above already mirrors this (its clear runs
        before any external caller can read the log).

        A successful SnapOpponent is the one path where Go's Snap.Active
        stays true through the pending move (so the accumulated log IS
        externally visible for that one step), then goes false once the move
        resolves (silent reset, nothing further emitted). Python's
        change_pending_move (above) must set snap_phase_active=False at the
        SnapOpponent-success step itself -- apply_action's dispatch checks
        snap_phase_active before pending_action, so the follow-up
        SnapOpponentMove action would never reach _handle_pending_action
        otherwise -- which leaves snap_results_log populated with no later
        natural clear point. An external reader would then see the SAME
        entries again when the SnapOpponentMove resolves, double-counting
        the outcome relative to Go. This flush (called from the
        SnapOpponentMove completion handler in _ability_mixin.py) reproduces
        Go's post-move silent reset.
        """
        if not self.snap_results_log:
            return
        original_log = list(self.snap_results_log)

        def change_flush():
            self.snap_results_log = []

        def undo_flush():
            self.snap_results_log = original_log

        self._add_change(
            change_flush, undo_flush, ("flush_snap_results_log",), undo_stack, delta_list
        )

    def _resume_or_end_snap_phase_after_move(
        self: "CambiaGameState", undo_stack: Deque, delta_list: StateDelta
    ) -> None:
        """Mirrors Go's advanceSnapper(), called from snapOpponentMove() once
        a successful SnapOpponent's move completes: advances to the next
        snapper, resuming the (paused) snap decision phase if any remain, or
        genuinely concluding it (flush + advance the main turn) otherwise.

        Context: change_pending_move (in _handle_snap_action's SnapOpponent
        success branch, above) sets snap_phase_active=False immediately at
        the success step, because apply_action's dispatch checks
        snap_phase_active before pending_action -- the follow-up
        SnapOpponentMove action would never reach _handle_pending_action
        otherwise. Go has no such constraint: its Snap.Active stays true
        through the paused move, and advanceSnapper() (called after the move
        resolves) either advances to the next eligible snapper -- who still
        gets a real decision -- or ends the phase, exactly mirroring the
        normal (PassSnap/SnapOwn/failed-SnapOpponent) last-snapper path. This
        restores that resume-or-end behavior at the point Go's own state
        machine would reach it: right after the move completes.

        Not gated on snap_phase_active (already False here) or on
        house_rules.snapRace explicitly: a successful SnapRace snap already
        clears snap_potential_snappers to [] at the success step (see the
        SnapRace branch above), so next_idx >= len([]) is always true here,
        naturally routing to the "phase ends" branch without a separate
        check -- exactly matching Go's advanceSnapper(), which also performs
        no SnapRace-specific branching itself.
        """
        original_idx = self.snap_current_snapper_idx
        next_idx = original_idx + 1

        def change_advance():
            self.snap_current_snapper_idx = next_idx

        def undo_advance():
            self.snap_current_snapper_idx = original_idx

        self._add_change(
            change_advance,
            undo_advance,
            ("set_attr", "snap_current_snapper_idx", next_idx, original_idx),
            undo_stack,
            delta_list,
        )

        if next_idx >= len(self.snap_potential_snappers):
            # No snapper remains: the phase is genuinely over. Flush the
            # results log now (matching Go's Snap.Active -> false silent
            # token reset). Do NOT advance the turn here: the caller
            # (_ability_mixin.py's SnapOpponentMove handler returns None with
            # pending_action already cleared) falls into engine.py's generic
            # "handler returned None but cleared state" path, which already
            # sets turn_should_advance_after_action=True and calls
            # _advance_turn() itself once snap_phase_active is confirmed
            # false -- calling it again here would double-advance the turn.
            self._flush_snap_results_log(undo_stack, delta_list)
        else:
            # A snapper remains: resume the paused decision phase for them.
            # Do NOT flush snap_results_log here -- Go's tokenizer keeps
            # re-emitting the accumulated block while Snap.Active stays true,
            # so both this move-step's own observation and the resumed
            # snapper's subsequent one must still see it.
            original_active = self.snap_phase_active

            def change_resume():
                self.snap_phase_active = True

            def undo_resume():
                self.snap_phase_active = original_active

            self._add_change(
                change_resume,
                undo_resume,
                ("set_attr", "snap_phase_active", True, original_active),
                undo_stack,
                delta_list,
            )

    def _get_snap_target_rank_str(self: "CambiaGameState") -> str:
        """Helper for __str__."""
        if self.snap_phase_active and self.snap_discarded_card:
            return str(getattr(self.snap_discarded_card, "rank", "Invalid"))
        return "N/A"
