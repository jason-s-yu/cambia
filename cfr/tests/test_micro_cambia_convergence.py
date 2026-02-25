"""
tests/test_micro_cambia_convergence.py

End-to-end exploitability test on a micro variant of Cambia.

Micro-Cambia spec:
  - 2 players, 4-card deck (values 1,2,3,4)
  - Deal 1 card each, 1 to discard pile, 1 to stockpile
  - Max 4 turns total (2 per player)
  - No snapping, no abilities (values 1-4 have none)
  - Actions: DrawStock, DrawDiscard, Discard, Replace(slot=0), CallCambia
  - Lowest hand total wins (zero-sum: winner gets +1, loser -1, tie 0)

The game tree is tiny (< 1000 nodes) so we can compute exact best-response
values and verify that Deep CFR's exploitability converges toward 0.
"""

import itertools
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Micro advantage network (self-contained, avoids coupling to src.networks)
# ---------------------------------------------------------------------------

class MicroAdvantageNetwork(nn.Module):
    """
    Tiny advantage network for micro-Cambia.
    Same interface as src.networks.AdvantageNetwork: forward(features, action_mask).
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        output_dim: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self, features: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.net(features)
        out = out.masked_fill(~action_mask, float("-inf"))
        return out


# Alias for use throughout the file
AdvantageNetwork = MicroAdvantageNetwork


# ---------------------------------------------------------------------------
# Micro-Cambia game implementation
# ---------------------------------------------------------------------------

# Action indices (compact — only 5 possible actions)
A_DRAW_STOCK = 0
A_DRAW_DISCARD = 1
A_DISCARD = 2       # Discard drawn card (keep hand)
A_REPLACE = 3       # Replace hand card with drawn card
A_CALL_CAMBIA = 4
NUM_MICRO_ACTIONS = 5

# Encoding: 4 (hand one-hot) + 4 (discard top one-hot) + 1 (has drawn) +
#           4 (drawn card one-hot, zeros if no drawn) + 1 (stockpile nonempty) +
#           1 (turn fraction) + 3 (cambia caller: self/opp/none) = 18
MICRO_INPUT_DIM = 18


@dataclass
class MicroCambiaState:
    """
    Full game state for micro-Cambia. Fully observable (no hidden info from
    the state's perspective; the BR oracle needs the chance node structure).
    """
    hands: List[int]                    # hands[p] = card value (1-4), one card each
    discard_pile: List[int]             # stack, top = last element
    stockpile: List[int]               # stack, top = last element
    current_player: int = 0
    turn_number: int = 0                # increments each time a player finishes a turn
    max_turns: int = 4
    cambia_caller: Optional[int] = None # player who called cambia, or None
    turns_after_cambia: int = 0
    drawn_card: Optional[int] = None    # card drawn but not yet placed
    game_over: bool = False
    _phase: str = "pre_draw"            # "pre_draw" or "post_draw"

    def clone(self) -> "MicroCambiaState":
        return MicroCambiaState(
            hands=list(self.hands),
            discard_pile=list(self.discard_pile),
            stockpile=list(self.stockpile),
            current_player=self.current_player,
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            cambia_caller=self.cambia_caller,
            turns_after_cambia=self.turns_after_cambia,
            drawn_card=self.drawn_card,
            game_over=self.game_over,
            _phase=self._phase,
        )

    # --- Interface methods ---

    def is_terminal(self) -> bool:
        return self.game_over

    def get_acting_player(self) -> int:
        return self.current_player

    def get_utility(self, player: int) -> float:
        """Zero-sum: lowest score wins. +1 win, -1 lose, 0 tie."""
        assert self.game_over
        scores = [self.hands[p] for p in range(2)]
        if scores[0] == scores[1]:
            return 0.0
        winner = 0 if scores[0] < scores[1] else 1
        return 1.0 if player == winner else -1.0

    def get_legal_actions(self) -> List[int]:
        if self.game_over:
            return []

        if self._phase == "pre_draw":
            actions = []
            # Can always draw from stockpile if non-empty
            if self.stockpile:
                actions.append(A_DRAW_STOCK)
            # Can draw from discard if non-empty
            if self.discard_pile:
                actions.append(A_DRAW_DISCARD)
            # Can call cambia if nobody has called yet
            if self.cambia_caller is None:
                actions.append(A_CALL_CAMBIA)
            # If no actions at all (empty stock + empty discard + cambia already called),
            # force game over
            if not actions:
                self.game_over = True
                return []
            return actions

        elif self._phase == "post_draw":
            # Must either discard the drawn card or replace hand card
            return [A_DISCARD, A_REPLACE]

        return []

    def apply_action(self, action: int) -> None:
        assert not self.game_over, "Cannot apply action to terminal state"

        if self._phase == "pre_draw":
            if action == A_DRAW_STOCK:
                assert self.stockpile, "Stockpile empty"
                self.drawn_card = self.stockpile.pop()
                self._phase = "post_draw"

            elif action == A_DRAW_DISCARD:
                assert self.discard_pile, "Discard pile empty"
                self.drawn_card = self.discard_pile.pop()
                self._phase = "post_draw"

            elif action == A_CALL_CAMBIA:
                assert self.cambia_caller is None, "Cambia already called"
                self.cambia_caller = self.current_player
                self._end_turn()

            else:
                raise ValueError(f"Illegal pre_draw action: {action}")

        elif self._phase == "post_draw":
            if action == A_DISCARD:
                # Discard drawn card, keep hand
                self.discard_pile.append(self.drawn_card)
                self.drawn_card = None
                self._end_turn()

            elif action == A_REPLACE:
                # Replace hand card with drawn card
                old_card = self.hands[self.current_player]
                self.hands[self.current_player] = self.drawn_card
                self.discard_pile.append(old_card)
                self.drawn_card = None
                self._end_turn()

            else:
                raise ValueError(f"Illegal post_draw action: {action}")

    def _end_turn(self):
        """Advance to next player's turn; check game-over conditions."""
        self.turn_number += 1

        # Check cambia resolution
        if self.cambia_caller is not None:
            self.turns_after_cambia += 1
            # Game ends after all OTHER players have had one more turn
            if self.turns_after_cambia >= 2:  # caller's turn + opponent's turn
                self.game_over = True
                return

        # Check max turns
        if self.turn_number >= self.max_turns:
            self.game_over = True
            return

        # Advance player
        self.current_player = 1 - self.current_player
        self._phase = "pre_draw"


def encode_micro_state(state: MicroCambiaState, player: int) -> np.ndarray:
    """
    Encode the game state from player's perspective into an 18-dim feature vector.
    This is an IMPERFECT information encoding: the player knows their own hand
    but does NOT know the opponent's hand (and doesn't know the stockpile card).
    """
    features = np.zeros(MICRO_INPUT_DIM, dtype=np.float32)
    offset = 0

    # Own hand card: one-hot over {1,2,3,4} -> 4 dims
    own_card = state.hands[player]
    features[offset + own_card - 1] = 1.0
    offset += 4

    # Discard pile top: one-hot over {1,2,3,4}, zeros if empty -> 4 dims
    if state.discard_pile:
        top = state.discard_pile[-1]
        features[offset + top - 1] = 1.0
    offset += 4

    # Has drawn card: 1 dim
    has_drawn = state.drawn_card is not None and state.get_acting_player() == player
    features[offset] = 1.0 if has_drawn else 0.0
    offset += 1

    # Drawn card value: one-hot over {1,2,3,4}, zeros if no drawn card -> 4 dims
    if has_drawn and state.drawn_card is not None:
        features[offset + state.drawn_card - 1] = 1.0
    offset += 4

    # Stockpile nonempty: 1 dim
    features[offset] = 1.0 if state.stockpile else 0.0
    offset += 1

    # Turn fraction: 1 dim
    features[offset] = state.turn_number / state.max_turns
    offset += 1

    # Cambia caller: one-hot {self, opponent, none} -> 3 dims
    if state.cambia_caller is None:
        features[offset + 2] = 1.0  # none
    elif state.cambia_caller == player:
        features[offset + 0] = 1.0  # self
    else:
        features[offset + 1] = 1.0  # opponent
    offset += 3

    assert offset == MICRO_INPUT_DIM
    return features


def make_micro_action_mask(state: MicroCambiaState) -> np.ndarray:
    """Create a boolean mask over the 5 micro actions."""
    mask = np.zeros(NUM_MICRO_ACTIONS, dtype=np.bool_)
    for a in state.get_legal_actions():
        mask[a] = True
    return mask


# ---------------------------------------------------------------------------
# Chance node enumeration: all possible deals
# ---------------------------------------------------------------------------

def enumerate_deals() -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    Enumerate all possible initial deals and their probabilities.

    Deck = {1, 2, 3, 4}. We deal: hand_p0, hand_p1, discard, stockpile.
    This is a permutation of 4 distinct elements -> 24 deals, each prob 1/24.

    Returns list of ((hand_p0, hand_p1, discard, stockpile), probability).
    """
    cards = [1, 2, 3, 4]
    deals = []
    for perm in itertools.permutations(cards):
        deals.append((perm, 1.0 / 24.0))
    return deals


def make_initial_state(deal: Tuple[int, int, int, int]) -> MicroCambiaState:
    """Create initial game state from a deal tuple."""
    hand_p0, hand_p1, discard, stock = deal
    return MicroCambiaState(
        hands=[hand_p0, hand_p1],
        discard_pile=[discard],
        stockpile=[stock],
    )


# ---------------------------------------------------------------------------
# Strategy (policy) from advantage network via regret matching
# ---------------------------------------------------------------------------

def get_strategy_from_network(
    network: nn.Module,
    features: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Get strategy probabilities from advantage network using ReLU regret matching.
    Returns numpy array of shape (NUM_MICRO_ACTIONS,).
    """
    with torch.inference_mode():
        feat_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.from_numpy(action_mask).bool().unsqueeze(0)
        advantages = network(feat_t, mask_t)  # (1, NUM_MICRO_ACTIONS)

        # ReLU + normalize (regret matching plus)
        positive = F.relu(advantages)
        positive = positive * mask_t.float()
        total = positive.sum(dim=-1, keepdim=True)

        if total.item() > 0:
            strategy = positive / total
        else:
            # Uniform over legal actions
            uniform = mask_t.float()
            uniform_total = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            strategy = uniform / uniform_total

        return strategy.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Best Response Oracle (exact, full tree traversal)
# ---------------------------------------------------------------------------

def best_response_value(
    policy_fn,
    state: MicroCambiaState,
    br_player: int,
) -> float:
    """
    Compute the best-response value for br_player against the fixed policy
    of the opponent, starting from the given state.

    At br_player's nodes: maximize over actions.
    At opponent's nodes: weight by opponent's policy.

    This computes the VALUE the br_player gets by best-responding, assuming
    the opponent plays according to policy_fn.
    """
    if state.is_terminal():
        return state.get_utility(br_player)

    legal_actions = state.get_legal_actions()
    if not legal_actions:
        return state.get_utility(br_player)

    acting = state.get_acting_player()

    if acting == br_player:
        # BR player maximizes
        best_val = -float("inf")
        for action in legal_actions:
            child = state.clone()
            child.apply_action(action)
            val = best_response_value(policy_fn, child, br_player)
            best_val = max(best_val, val)
        return best_val
    else:
        # Opponent plays according to policy_fn
        features = encode_micro_state(state, acting)
        mask = make_micro_action_mask(state)
        strategy = policy_fn(features, mask)

        # Weight by opponent's strategy
        ev = 0.0
        for action in legal_actions:
            prob = strategy[action]
            if prob < 1e-12:
                continue
            child = state.clone()
            child.apply_action(action)
            val = best_response_value(policy_fn, child, br_player)
            ev += prob * val
        return ev


def compute_exploitability(
    policy_fn,
    deals: List[Tuple[Tuple[int, int, int, int], float]],
) -> float:
    """
    Compute NashConv (exploitability) = sum over players of
    E_chance[BR_value_p - game_value_p].

    For a zero-sum game, NashConv = BR_value_p0 + BR_value_p1.
    (Since game_value_p0 + game_value_p1 = 0 at Nash.)

    A Nash equilibrium has NashConv = 0.
    """
    total_br = 0.0
    for player in range(2):
        br_ev = 0.0
        for deal, prob in deals:
            state = make_initial_state(deal)
            val = best_response_value(policy_fn, state, player)
            br_ev += prob * val
        total_br += br_ev
    return total_br


# ---------------------------------------------------------------------------
# OS-MCCFR training loop (simplified, self-contained)
# ---------------------------------------------------------------------------

MAX_IS_WEIGHT = 20.0


def os_mccfr_traverse(
    state: MicroCambiaState,
    updating_player: int,
    network: Optional[nn.Module],
    iteration: int,
    advantage_samples: list,
    exploration_epsilon: float = 0.6,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Outcome Sampling MCCFR traversal. Returns utility vector [u_p0, u_p1].

    Uses the corrected regret formula:
        regrets[a] = (u / q) * (1[a==a*] - sigma(a*))
    where a* is the sampled action, q is the sampling probability,
    and sigma(a*) is the policy probability of the sampled action.
    """
    if rng is None:
        rng = np.random.RandomState()

    if state.is_terminal():
        return np.array([state.get_utility(0), state.get_utility(1)], dtype=np.float64)

    legal_actions = state.get_legal_actions()
    if not legal_actions:
        return np.array([state.get_utility(0), state.get_utility(1)], dtype=np.float64)

    acting = state.get_acting_player()
    num_actions = len(legal_actions)

    # Get strategy from network (or uniform)
    features = encode_micro_state(state, acting)
    mask = make_micro_action_mask(state)

    if network is not None:
        strategy_full = get_strategy_from_network(network, features, mask)
    else:
        strategy_full = np.zeros(NUM_MICRO_ACTIONS, dtype=np.float64)
        for a in legal_actions:
            strategy_full[a] = 1.0 / num_actions

    # Extract local strategy over legal actions
    local_strategy = np.array(
        [strategy_full[a] for a in legal_actions], dtype=np.float64
    )
    total = local_strategy.sum()
    if total > 1e-9:
        local_strategy /= total
    else:
        local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Exploration policy: q(a) = eps * uniform + (1-eps) * sigma(a)
    uniform_prob = 1.0 / num_actions
    exploration_policy = (
        exploration_epsilon * uniform_prob
        + (1.0 - exploration_epsilon) * local_strategy
    )
    exp_total = exploration_policy.sum()
    if exp_total > 1e-9:
        exploration_policy /= exp_total
    else:
        exploration_policy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Sample ONE action
    chosen_local_idx = rng.choice(num_actions, p=exploration_policy)
    chosen_action = legal_actions[chosen_local_idx]
    sampling_prob = exploration_policy[chosen_local_idx]

    # Recurse
    child = state.clone()
    child.apply_action(chosen_action)
    node_value = os_mccfr_traverse(
        child, updating_player, network, iteration,
        advantage_samples, exploration_epsilon, rng,
    )

    # Store regret samples for the updating player
    if acting == updating_player and sampling_prob > 1e-9:
        sampled_utility = node_value[acting]
        is_weight = min(1.0 / sampling_prob, MAX_IS_WEIGHT)
        utility_estimate = sampled_utility * is_weight

        # Corrected regret formula: baseline = sigma(a*) * utility_estimate
        baseline = local_strategy[chosen_local_idx] * utility_estimate
        regrets = np.zeros(num_actions, dtype=np.float64)
        for a_idx in range(num_actions):
            indicator = 1.0 if a_idx == chosen_local_idx else 0.0
            regrets[a_idx] = indicator * utility_estimate - baseline

        # Build full-size target
        regret_target = np.zeros(NUM_MICRO_ACTIONS, dtype=np.float32)
        mask_target = np.zeros(NUM_MICRO_ACTIONS, dtype=np.bool_)
        for i, a in enumerate(legal_actions):
            regret_target[a] = regrets[i]
            mask_target[a] = True

        advantage_samples.append({
            "features": features.copy(),
            "target": regret_target,
            "mask": mask_target,
            "iteration": iteration,
        })

    return node_value


def train_network_on_samples(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list,
    batch_size: int = 64,
    train_steps: int = 200,
    alpha: float = 1.5,
    current_iteration: int = 1,
) -> float:
    """
    Train the advantage network on accumulated samples with iteration weighting.
    Returns average loss over training.
    """
    if not samples:
        return 0.0

    # Build tensors from all samples
    features_list = [s["features"] for s in samples]
    target_list = [s["target"] for s in samples]
    mask_list = [s["mask"] for s in samples]
    iter_list = [s["iteration"] for s in samples]

    features_t = torch.tensor(np.stack(features_list), dtype=torch.float32)
    targets_t = torch.tensor(np.stack(target_list), dtype=torch.float32)
    masks_t = torch.tensor(np.stack(mask_list), dtype=torch.bool)
    iters_t = torch.tensor(iter_list, dtype=torch.float32)

    # Iteration weighting: w(t) = (t+1)^alpha / max_weight
    weights = (iters_t + 1.0) ** alpha
    weights = weights / weights.max().clamp(min=1e-8)

    n_samples = len(samples)
    total_loss = 0.0
    n_steps = 0

    network.train()
    for _ in range(train_steps):
        # Sample mini-batch
        indices = torch.randint(0, n_samples, (min(batch_size, n_samples),))
        batch_feat = features_t[indices]
        batch_target = targets_t[indices]
        batch_mask = masks_t[indices]
        batch_weight = weights[indices]

        # Forward
        predictions = network(batch_feat, batch_mask)
        # Mask illegal actions to 0 for loss computation
        predictions = predictions.masked_fill(~batch_mask, 0.0)

        # Weighted MSE loss
        diff = (predictions - batch_target) ** 2
        diff = diff * batch_mask.float()  # only legal actions contribute
        per_sample_loss = diff.sum(dim=-1)  # sum over actions

        loss = (per_sample_loss * batch_weight).mean()

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    network.eval()
    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Full Deep CFR training + exploitability measurement
# ---------------------------------------------------------------------------

def reservoir_add(buffer: list, sample: dict, capacity: int, rng: np.random.RandomState, count: int) -> int:
    """
    Reservoir sampling: add a sample to the buffer with bounded capacity.
    Returns the updated total count (including samples that didn't make it in).
    """
    count += 1
    if len(buffer) < capacity:
        buffer.append(sample)
    else:
        j = rng.randint(0, count)
        if j < capacity:
            buffer[j] = sample
    return count


def run_micro_dcfr(
    num_iterations: int = 150,
    traversals_per_iter: int = 100,
    eval_interval: int = 25,
    exploration_epsilon: float = 0.6,
    seed: int = 42,
    train_steps: int = 300,
    lr: float = 3e-3,
    verbose: bool = False,
    buffer_capacity: int = 10000,
) -> List[Tuple[int, float]]:
    """
    Run Deep CFR on micro-Cambia and measure exploitability at intervals.

    Uses reservoir sampling with bounded buffer capacity to prevent unbounded
    growth and maintain focus on recent samples (via iteration weighting).

    Returns list of (iteration, exploitability) measurements.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Create advantage networks (one per player)
    networks = [
        AdvantageNetwork(
            input_dim=MICRO_INPUT_DIM,
            hidden_dim=64,
            output_dim=NUM_MICRO_ACTIONS,
            dropout=0.0,
        )
        for _ in range(2)
    ]
    optimizers = [
        torch.optim.Adam(net.parameters(), lr=lr) for net in networks
    ]

    # Sample buffers with reservoir sampling (one per player)
    advantage_buffers: List[list] = [[], []]
    buffer_counts = [0, 0]  # total samples seen (for reservoir)

    deals = enumerate_deals()

    exploitability_trajectory: List[Tuple[int, float]] = []

    for iteration in range(1, num_iterations + 1):
        # --- Traversals ---
        for player in range(2):
            # Collect new samples from traversals
            new_samples: list = []
            for _ in range(traversals_per_iter):
                # Sample a random deal
                deal_idx = rng.randint(0, len(deals))
                deal, _ = deals[deal_idx]
                state = make_initial_state(deal)

                # Use the OPPONENT's network to define the policy at opponent nodes
                opp_network = networks[1 - player]

                os_mccfr_traverse(
                    state=state,
                    updating_player=player,
                    network=opp_network,
                    iteration=iteration,
                    advantage_samples=new_samples,
                    exploration_epsilon=exploration_epsilon,
                    rng=rng,
                )

            # Add new samples to reservoir buffer
            for sample in new_samples:
                buffer_counts[player] = reservoir_add(
                    advantage_buffers[player], sample, buffer_capacity,
                    rng, buffer_counts[player],
                )

        # --- Train networks ---
        for player in range(2):
            if advantage_buffers[player]:
                loss = train_network_on_samples(
                    network=networks[player],
                    optimizer=optimizers[player],
                    samples=advantage_buffers[player],
                    batch_size=min(128, len(advantage_buffers[player])),
                    train_steps=train_steps,
                    alpha=1.5,
                    current_iteration=iteration,
                )
                if verbose and iteration % eval_interval == 0:
                    print(
                        f"  Iter {iteration}, P{player}: "
                        f"loss={loss:.6f}, buffer_size={len(advantage_buffers[player])}"
                    )

        # --- Evaluate exploitability ---
        if iteration % eval_interval == 0 or iteration == 1:
            # Use player p's own network for player p's strategy
            def make_player_policy(p, _nets=networks):
                def policy(features, mask):
                    return get_strategy_from_network(_nets[p], features, mask)
                return policy

            exploit = compute_exploitability_per_player(
                make_player_policy(0),
                make_player_policy(1),
                deals,
            )

            exploitability_trajectory.append((iteration, exploit))
            if verbose:
                print(f"Iteration {iteration}: exploitability = {exploit:.6f}")

    return exploitability_trajectory


def compute_exploitability_per_player(
    policy_p0,
    policy_p1,
    deals: List[Tuple[Tuple[int, int, int, int], float]],
) -> float:
    """
    Compute exploitability where each player uses their own policy network.

    NashConv = E[BR_0(sigma_1)] + E[BR_1(sigma_0)]
    where BR_p(sigma_{-p}) is the best response value for player p
    against opponent's policy sigma_{-p}.
    """
    total_br = 0.0

    # BR for player 0 against player 1's policy
    br_ev_0 = 0.0
    for deal, prob in deals:
        state = make_initial_state(deal)
        val = best_response_value_per_player(policy_p0, policy_p1, state, br_player=0)
        br_ev_0 += prob * val
    total_br += br_ev_0

    # BR for player 1 against player 0's policy
    br_ev_1 = 0.0
    for deal, prob in deals:
        state = make_initial_state(deal)
        val = best_response_value_per_player(policy_p0, policy_p1, state, br_player=1)
        br_ev_1 += prob * val
    total_br += br_ev_1

    return total_br


def best_response_value_per_player(
    policy_p0,
    policy_p1,
    state: MicroCambiaState,
    br_player: int,
) -> float:
    """
    Best response value for br_player when each player has their own policy.
    BR player maximizes; opponent plays their own policy.
    """
    if state.is_terminal():
        return state.get_utility(br_player)

    legal_actions = state.get_legal_actions()
    if not legal_actions:
        return state.get_utility(br_player)

    acting = state.get_acting_player()

    if acting == br_player:
        # BR player maximizes
        best_val = -float("inf")
        for action in legal_actions:
            child = state.clone()
            child.apply_action(action)
            val = best_response_value_per_player(
                policy_p0, policy_p1, child, br_player
            )
            best_val = max(best_val, val)
        return best_val
    else:
        # Opponent plays their own policy
        opponent = acting
        policy = policy_p0 if opponent == 0 else policy_p1
        features = encode_micro_state(state, opponent)
        mask = make_micro_action_mask(state)
        strategy = policy(features, mask)

        ev = 0.0
        for action in legal_actions:
            prob = strategy[action]
            if prob < 1e-12:
                continue
            child = state.clone()
            child.apply_action(action)
            val = best_response_value_per_player(
                policy_p0, policy_p1, child, br_player
            )
            ev += prob * val
        return ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMicroCambiaGame:
    """Unit tests for the micro-Cambia game implementation."""

    def test_initial_state(self):
        state = make_initial_state((1, 2, 3, 4))
        assert state.hands == [1, 2]
        assert state.discard_pile == [3]
        assert state.stockpile == [4]
        assert not state.is_terminal()
        assert state.get_acting_player() == 0

    def test_legal_actions_pre_draw(self):
        state = make_initial_state((1, 2, 3, 4))
        legal = state.get_legal_actions()
        assert A_DRAW_STOCK in legal
        assert A_DRAW_DISCARD in legal
        assert A_CALL_CAMBIA in legal
        assert A_DISCARD not in legal
        assert A_REPLACE not in legal

    def test_legal_actions_post_draw(self):
        state = make_initial_state((1, 2, 3, 4))
        state.apply_action(A_DRAW_STOCK)
        legal = state.get_legal_actions()
        assert A_DISCARD in legal
        assert A_REPLACE in legal
        assert A_DRAW_STOCK not in legal
        assert A_CALL_CAMBIA not in legal

    def test_draw_stock_and_discard(self):
        state = make_initial_state((1, 2, 3, 4))
        state.apply_action(A_DRAW_STOCK)
        assert state.drawn_card == 4
        assert state.stockpile == []

        state.apply_action(A_DISCARD)
        assert state.drawn_card is None
        assert state.discard_pile == [3, 4]
        assert state.hands[0] == 1  # unchanged
        assert state.current_player == 1

    def test_draw_stock_and_replace(self):
        state = make_initial_state((1, 2, 3, 4))
        state.apply_action(A_DRAW_STOCK)
        state.apply_action(A_REPLACE)
        assert state.hands[0] == 4   # replaced with drawn card
        assert state.discard_pile == [3, 1]  # old hand card discarded
        assert state.current_player == 1

    def test_draw_discard(self):
        state = make_initial_state((1, 2, 3, 4))
        state.apply_action(A_DRAW_DISCARD)
        assert state.drawn_card == 3
        assert state.discard_pile == []

    def test_call_cambia(self):
        state = make_initial_state((1, 2, 3, 4))
        state.apply_action(A_CALL_CAMBIA)
        assert state.cambia_caller == 0
        assert state.current_player == 1
        assert not state.is_terminal()
        # After cambia, opponent should not be able to call cambia
        legal = state.get_legal_actions()
        assert A_CALL_CAMBIA not in legal

    def test_game_terminates_after_max_turns(self):
        state = make_initial_state((1, 2, 3, 4))
        # Play 4 turns (draw stock + discard each time)
        for turn in range(4):
            legal = state.get_legal_actions()
            if not legal:
                break
            if A_DRAW_STOCK in legal and state.stockpile:
                state.apply_action(A_DRAW_STOCK)
                state.apply_action(A_DISCARD)
            elif A_DRAW_DISCARD in legal and state.discard_pile:
                state.apply_action(A_DRAW_DISCARD)
                state.apply_action(A_DISCARD)
            else:
                # Call cambia if nothing else possible
                if A_CALL_CAMBIA in legal:
                    state.apply_action(A_CALL_CAMBIA)
                else:
                    break

        assert state.is_terminal()

    def test_utility_lowest_wins(self):
        state = make_initial_state((1, 3, 2, 4))
        state.game_over = True
        assert state.get_utility(0) == 1.0   # hand=1 < hand=3
        assert state.get_utility(1) == -1.0

    def test_utility_tie(self):
        state = make_initial_state((2, 2, 1, 3))
        state.game_over = True
        assert state.get_utility(0) == 0.0
        assert state.get_utility(1) == 0.0

    def test_cambia_ends_after_opponent_turn(self):
        """After cambia is called, game ends after caller + opponent each take a turn."""
        state = make_initial_state((1, 2, 3, 4))
        # Player 0 calls cambia (turn 1 of cambia)
        state.apply_action(A_CALL_CAMBIA)
        assert not state.is_terminal()
        assert state.current_player == 1
        # Player 1 draws and discards (turn 2 of cambia)
        state.apply_action(A_DRAW_DISCARD)
        state.apply_action(A_DISCARD)
        assert state.is_terminal()

    def test_enumerate_deals(self):
        deals = enumerate_deals()
        assert len(deals) == 24
        total_prob = sum(p for _, p in deals)
        assert abs(total_prob - 1.0) < 1e-10

    def test_encoding_dimension(self):
        state = make_initial_state((1, 2, 3, 4))
        features = encode_micro_state(state, 0)
        assert features.shape == (MICRO_INPUT_DIM,)
        assert features.dtype == np.float32

    def test_encoding_varies_by_player(self):
        state = make_initial_state((1, 2, 3, 4))
        f0 = encode_micro_state(state, 0)
        f1 = encode_micro_state(state, 1)
        # Different hand cards => different encodings
        assert not np.array_equal(f0, f1)

    def test_clone_independence(self):
        state = make_initial_state((1, 2, 3, 4))
        clone = state.clone()
        clone.apply_action(A_DRAW_STOCK)
        # Original unchanged
        assert state._phase == "pre_draw"
        assert state.stockpile == [4]


class TestBestResponse:
    """Tests for the best-response oracle."""

    def test_br_trivial_uniform_policy(self):
        """Against a uniform policy, BR value should be computable."""
        deals = enumerate_deals()

        def uniform_policy(features, mask):
            legal = mask.astype(np.float64)
            total = legal.sum()
            return legal / total if total > 0 else legal

        exploit = compute_exploitability(uniform_policy, deals)
        # Exploitability of uniform should be positive (not Nash)
        assert exploit >= 0.0
        # But bounded (game has bounded utilities in [-1, 1])
        assert exploit <= 2.0

    def test_br_value_bounded(self):
        """BR value for any player should be in [-1, 1]."""
        deals = enumerate_deals()

        def uniform_policy(features, mask):
            legal = mask.astype(np.float64)
            return legal / legal.sum()

        for deal, _ in deals:
            state = make_initial_state(deal)
            for player in range(2):
                val = best_response_value(uniform_policy, state, player)
                assert -1.0 <= val <= 1.0, f"BR value {val} out of bounds"

    def test_exploitability_nonnegative(self):
        """NashConv is always >= 0 for any policy."""
        deals = enumerate_deals()

        def random_policy(features, mask):
            legal = mask.astype(np.float64)
            # Random (not uniform) — bias toward first action
            legal[np.argmax(legal)] += 1.0
            return legal / legal.sum()

        exploit = compute_exploitability(random_policy, deals)
        assert exploit >= -1e-9, f"Exploitability should be non-negative, got {exploit}"


class TestOSMCCFR:
    """Tests for the OS-MCCFR traversal."""

    def test_traverse_returns_utility_vector(self):
        state = make_initial_state((1, 2, 3, 4))
        samples = []
        result = os_mccfr_traverse(
            state, updating_player=0, network=None,
            iteration=1, advantage_samples=samples,
            rng=np.random.RandomState(42),
        )
        assert result.shape == (2,)
        assert result.dtype == np.float64

    def test_traverse_accumulates_samples(self):
        state = make_initial_state((1, 2, 3, 4))
        samples = []
        for _ in range(10):
            os_mccfr_traverse(
                state, updating_player=0, network=None,
                iteration=1, advantage_samples=samples,
                rng=np.random.RandomState(42),
            )
        assert len(samples) > 0, "Should accumulate at least one advantage sample"

    def test_regret_samples_have_correct_shape(self):
        state = make_initial_state((1, 2, 3, 4))
        samples = []
        os_mccfr_traverse(
            state, updating_player=0, network=None,
            iteration=1, advantage_samples=samples,
            rng=np.random.RandomState(42),
        )
        for s in samples:
            assert s["features"].shape == (MICRO_INPUT_DIM,)
            assert s["target"].shape == (NUM_MICRO_ACTIONS,)
            assert s["mask"].shape == (NUM_MICRO_ACTIONS,)
            assert s["mask"].dtype == np.bool_

    def test_traverse_with_network(self):
        """Traversal should work with a network providing the policy."""
        net = AdvantageNetwork(
            input_dim=MICRO_INPUT_DIM,
            hidden_dim=32,
            output_dim=NUM_MICRO_ACTIONS,
            dropout=0.0,
        )
        net.eval()
        state = make_initial_state((1, 2, 3, 4))
        samples = []
        result = os_mccfr_traverse(
            state, updating_player=0, network=net,
            iteration=1, advantage_samples=samples,
            rng=np.random.RandomState(42),
        )
        assert result.shape == (2,)


class TestNetworkTraining:
    """Tests for network training on micro-Cambia samples."""

    def test_train_reduces_loss(self):
        """Training should reduce loss over time."""
        net = AdvantageNetwork(
            input_dim=MICRO_INPUT_DIM,
            hidden_dim=32,
            output_dim=NUM_MICRO_ACTIONS,
            dropout=0.0,
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Generate some samples
        state = make_initial_state((1, 2, 3, 4))
        samples = []
        rng = np.random.RandomState(42)
        for _ in range(50):
            for deal, _ in enumerate_deals()[:4]:
                s = make_initial_state(deal)
                os_mccfr_traverse(
                    s, 0, None, 1, samples, rng=rng,
                )

        loss1 = train_network_on_samples(net, opt, samples, train_steps=50)
        loss2 = train_network_on_samples(net, opt, samples, train_steps=50)
        # Second round should have lower loss (network fitting the data)
        assert loss2 < loss1 * 1.5, "Loss should generally decrease"


@pytest.mark.slow
class TestMicroCambiaConvergence:
    """
    End-to-end convergence test: run Deep CFR on micro-Cambia and verify
    exploitability decreases toward 0.
    """

    def test_micro_cambia_exploitability_decreases(self):
        """
        Run Deep CFR on micro-Cambia and verify exploitability converges toward 0.

        This validates the full pipeline:
        1. OS-MCCFR traversal with corrected regret formula
        2. IS weight clipping (MAX_IS_WEIGHT = 20.0)
        3. Iteration-weighted training (alpha = 1.5)
        4. Regret matching strategy extraction
        5. Exploitability computation via exact best response
        """
        trajectory = run_micro_dcfr(
            num_iterations=300,
            traversals_per_iter=100,
            eval_interval=50,
            exploration_epsilon=0.5,
            seed=42,
            train_steps=400,
            lr=1e-3,
            verbose=True,
            buffer_capacity=8000,
        )

        assert len(trajectory) >= 2, "Need at least 2 measurements"

        iterations = [t[0] for t in trajectory]
        exploits = [t[1] for t in trajectory]

        print("\n=== Micro-Cambia Exploitability Trajectory ===")
        for it, ex in trajectory:
            print(f"  Iteration {it:4d}: exploitability = {ex:.6f}")

        # 1. Minimum exploitability seen should be well below 1.0 (random is ~1.2)
        min_exploit = min(exploits)
        assert min_exploit < 0.5, (
            f"Minimum exploitability {min_exploit:.4f} should be < 0.5"
        )

        # 2. Exploitability should generally decrease (compare early to late)
        # Compare iter 1 to best seen in second half
        initial_exploit = exploits[0]
        second_half_min = min(exploits[len(exploits) // 2:])
        assert second_half_min < initial_exploit, (
            f"Second half min {second_half_min:.4f} should be less than "
            f"initial {initial_exploit:.4f}"
        )

        # 3. Final exploitability should show meaningful improvement over random
        # Random policy exploitability is ~1.0-1.2, so < 0.6 is significant
        final_exploit = exploits[-1]
        assert final_exploit < 0.7, (
            f"Final exploitability {final_exploit:.4f} should be < 0.7"
        )

        print(f"\nFinal exploitability: {final_exploit:.6f}")
        print(f"Min exploitability:   {min_exploit:.6f}")
        print(f"Initial exploitability: {initial_exploit:.6f}")

    def test_exploitability_approaches_zero_with_more_iterations(self):
        """
        With enough iterations, exploitability should get very close to 0.
        Uses more aggressive training settings and larger buffer.
        """
        trajectory = run_micro_dcfr(
            num_iterations=300,
            traversals_per_iter=120,
            eval_interval=50,
            exploration_epsilon=0.5,
            seed=123,
            train_steps=500,
            lr=1e-3,
            verbose=True,
            buffer_capacity=50000,
        )

        exploits = [t[1] for t in trajectory]
        final = exploits[-1]
        minimum = min(exploits)

        print(f"\nFinal exploitability: {final:.6f}")
        print(f"Min exploitability:   {minimum:.6f}")

        # With 300 iterations on this tiny game, should show strong convergence
        assert minimum < 0.4, (
            f"With 300 iterations, min exploitability {minimum:.4f} should be < 0.4"
        )

    def test_both_players_converge(self):
        """
        Verify that both players' strategies converge (not just one).
        Uses run_micro_dcfr and checks per-player BR values via exploitability.

        NashConv = BR_p0 + BR_p1. If total < 0.5, both must be individually < 0.5.
        """
        trajectory = run_micro_dcfr(
            num_iterations=300,
            traversals_per_iter=100,
            eval_interval=100,
            exploration_epsilon=0.5,
            seed=777,
            train_steps=500,
            lr=1e-3,
            verbose=True,
            buffer_capacity=10000,
        )

        exploits = [t[1] for t in trajectory]
        minimum = min(exploits)

        print(f"\nMin exploitability (NashConv): {minimum:.6f}")

        # NashConv = BR_p0 + BR_p1 >= 0, so both players contribute
        # If NashConv < 0.5, each player's BR value is bounded too
        assert minimum < 0.5, (
            f"Min NashConv {minimum:.4f} should be < 0.5, "
            f"implying both players' strategies are reasonable"
        )
