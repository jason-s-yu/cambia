"""
src/networks.py

PyTorch neural network modules for Deep CFR.

AdvantageNetwork: Predicts per-action advantage/regret values.
StrategyNetwork: Predicts per-action strategy probabilities.

Architecture (shared):
  Input(222) -> Linear(256) -> ReLU -> Dropout(0.1)
             -> Linear(256) -> ReLU -> Dropout(0.1)
             -> Linear(128) -> ReLU -> Linear(146)

Total parameters: ~175K (174,610 per network).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import INPUT_DIM, NUM_ACTIONS
from .cfr.exceptions import InvalidNetworkInputError
from .pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from .constants import (
    EP_PBS_INPUT_DIM,
    EP_PBS_MAX_SLOTS,
    EP_PBS_TAG_DIM,
    EP_PBS_BUCKET_DIM,
    EP_PBS_SLOT_DIM,
    EP_PBS_SLOT_REPR_DIM,
    EP_PBS_POS_EMBED_DIM,
    EP_PBS_PUBLIC_DIM_V2,
)


class AdvantageNetwork(nn.Module):
    """
    Predicts per-action advantage (regret) values for a given information set.

    Forward pass applies action masking: illegal actions are set to -inf.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._validate_inputs = validate_inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
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
        """
        Args:
            features: (batch, input_dim) float tensor of encoded infoset features.
            action_mask: (batch, output_dim) bool tensor, True for legal actions.

        Returns:
            (batch, output_dim) float tensor of advantage values, -inf for illegal actions.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        # Validate input shape
        if features.dim() != 2 or features.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features shape: expected (batch, {self._input_dim}), got {tuple(features.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), got {tuple(action_mask.shape)}"
            )
        if features.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: features has {features.shape[0]}, action_mask has {action_mask.shape[0]}"
            )

        # Validate NaN
        if self._validate_inputs and torch.isnan(features).any():
            raise InvalidNetworkInputError("Features tensor contains NaN values")

        out = self.net(features)
        out = out.masked_fill(~action_mask, float("-inf"))
        return out


class _ResBlock(nn.Module):
    """Residual block: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm + skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class ResidualAdvantageNetwork(nn.Module):
    """
    Advantage network with residual connections for improved gradient flow.

    Architecture: input_proj → [ResBlock × N] → output_head
    ResBlock: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm + skip
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._validate_inputs = validate_inputs

        # Input projection: input_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.res_blocks.append(
                _ResBlock(hidden_dim, dropout)
            )

        # Output head: hidden_dim → output_dim
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self, features: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        if features.dim() != 2 or features.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features shape: expected (batch, {self._input_dim}), got {tuple(features.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), got {tuple(action_mask.shape)}"
            )
        if features.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: features has {features.shape[0]}, action_mask has {action_mask.shape[0]}"
            )
        if self._validate_inputs and torch.isnan(features).any():
            raise InvalidNetworkInputError("Features tensor contains NaN values")

        x = self.input_proj(features)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_head(x)
        out = out.masked_fill(~action_mask, float("-inf"))
        return out


class SlotFiLMAdvantageNetwork(nn.Module):
    """
    Slot-structured advantage network with Residual FiLM conditioning.
    Processes EP-PBS interleaved encoding: [public(42)][12×slot(13)][pad(2)] = 200 dims.
    """

    def __init__(
        self,
        input_dim: int = EP_PBS_INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
        validate_inputs: bool = True,
        num_hidden_layers: int = 3,
        num_slots: int = EP_PBS_MAX_SLOTS,
        public_dim: int = EP_PBS_PUBLIC_DIM_V2,
        slot_dim: int = EP_PBS_SLOT_DIM,
        tag_dim: int = EP_PBS_TAG_DIM,
        id_dim: int = EP_PBS_BUCKET_DIM,
        slot_repr_dim: int = EP_PBS_SLOT_REPR_DIM,
        pos_embed_dim: int = EP_PBS_POS_EMBED_DIM,
        embed_dim: int = 32,
        use_film: bool = True,
        use_pos_embed: bool = True,
        num_players: int = 2,
    ):
        super().__init__()
        self.validate_inputs = validate_inputs
        self.num_slots = num_slots
        self.public_dim = public_dim
        self.slot_dim = slot_dim
        self.tag_dim = tag_dim
        self.id_dim = id_dim
        self.slot_repr_dim = slot_repr_dim
        self.pos_embed_dim = pos_embed_dim
        self.embed_dim = embed_dim
        self.use_film = use_film
        self.use_pos_embed = use_pos_embed
        self.num_players = num_players
        self.slots_per_player = num_slots // num_players

        # Stage 1 — Slot Encoder (shared across all slots)
        self.tag_embed = nn.Linear(tag_dim, embed_dim)
        self.id_embed = nn.Linear(id_dim, embed_dim)

        if use_film:
            self.film_gamma = nn.Linear(embed_dim, embed_dim)
            self.film_beta = nn.Linear(embed_dim, embed_dim)
        else:
            self.tag_gate = nn.Linear(embed_dim, embed_dim)

        if use_pos_embed:
            self.slot_pos_embed = nn.Embedding(self.slots_per_player, 4)
            self.owner_embed = nn.Embedding(num_players, 4)
            self.slot_proj = nn.Linear(embed_dim + pos_embed_dim, slot_repr_dim)
        else:
            self.slot_proj = nn.Linear(embed_dim, slot_repr_dim)

        self.slot_norm = nn.LayerNorm(slot_repr_dim)

        # Stage 2 — Aggregation
        self.global_proj = nn.Linear(public_dim, slot_repr_dim)
        aggregated_dim = slot_repr_dim * (num_players + 1)
        self.input_proj = nn.Linear(aggregated_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Stage 3 — Trunk
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden_dim, dropout) for _ in range(num_hidden_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "film_gamma" in name or "film_beta" in name:
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif "tag_embed" in name or "id_embed" in name:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]

        if self.validate_inputs:
            if torch.isnan(features).any():
                raise ValueError("NaN in input features")

        # Extract public and slot portions
        public = features[:, : self.public_dim]
        slot_start = self.public_dim
        slot_end = slot_start + self.num_slots * self.slot_dim
        slot_data = features[:, slot_start:slot_end].view(
            batch, self.num_slots, self.slot_dim
        )

        # Split tag and identity
        tag_input = slot_data[:, :, : self.tag_dim]
        id_input = slot_data[:, :, self.tag_dim :]

        # Shared encoder
        tag_h = F.relu(self.tag_embed(tag_input))
        id_h = F.relu(self.id_embed(id_input))

        if self.use_film:
            gamma_delta = self.film_gamma(tag_h)
            beta = self.film_beta(tag_h)
            gated = (1.0 + gamma_delta) * id_h + beta
        else:
            gate = torch.sigmoid(self.tag_gate(tag_h))
            gated = gate * id_h

        if self.use_pos_embed:
            slot_indices = torch.arange(self.num_slots, device=features.device)
            slot_pos_idx = slot_indices % self.slots_per_player
            owner_idx = slot_indices // self.slots_per_player
            pos = torch.cat(
                [self.slot_pos_embed(slot_pos_idx), self.owner_embed(owner_idx)], dim=-1
            )
            pos = pos.unsqueeze(0).expand(batch, -1, -1)
            slot_input = torch.cat([gated, pos], dim=-1)
        else:
            slot_input = gated

        slot_repr = F.relu(self.slot_norm(self.slot_proj(slot_input)))

        # Aggregate by ownership group
        pools = []
        for p in range(self.num_players):
            start = p * self.slots_per_player
            end = start + self.slots_per_player
            pools.append(slot_repr[:, start:end].mean(dim=1))

        global_feat = F.relu(self.global_proj(public))
        x = torch.cat(pools + [global_feat], dim=1)

        # Trunk
        x = F.relu(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)

        # Output
        out = self.output_head(x)
        out = out.masked_fill(~action_mask.bool(), float("-inf"))
        return out


def build_advantage_network(
    input_dim: int = INPUT_DIM,
    hidden_dim: int = 256,
    output_dim: int = NUM_ACTIONS,
    dropout: float = 0.1,
    validate_inputs: bool = True,
    num_hidden_layers: int = 2,
    use_residual: bool = False,
    network_type: str = "residual",
    **kwargs,  # forward extra args to SlotFiLM (use_film, use_pos_embed, num_players, etc.)
) -> nn.Module:
    """Factory: dispatches to AdvantageNetwork, ResidualAdvantageNetwork, or SlotFiLMAdvantageNetwork.

    network_type values:
      "mlp"           -> AdvantageNetwork
      "residual"      -> ResidualAdvantageNetwork if use_residual=True, else AdvantageNetwork
      "slot_film"     -> SlotFiLMAdvantageNetwork(use_film=True)
      "slot_multiply" -> SlotFiLMAdvantageNetwork(use_film=False)
    """
    if network_type == "mlp":
        return AdvantageNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            validate_inputs=validate_inputs,
        )
    elif network_type == "residual":
        if use_residual:
            return ResidualAdvantageNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                output_dim=output_dim,
                dropout=dropout,
                validate_inputs=validate_inputs,
            )
        return AdvantageNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            validate_inputs=validate_inputs,
        )
    elif network_type in ("slot_film", "slot_multiply"):
        use_film = network_type == "slot_film"
        return SlotFiLMAdvantageNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            validate_inputs=validate_inputs,
            num_hidden_layers=num_hidden_layers,
            use_film=use_film,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown network_type '{network_type}'. "
            "Valid values: 'mlp', 'residual', 'slot_film', 'slot_multiply'."
        )


class StrategyNetwork(nn.Module):
    """
    Predicts per-action strategy probabilities for a given information set.

    Forward pass applies action masking and softmax: illegal actions get 0 probability.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = NUM_ACTIONS,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._validate_inputs = validate_inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
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
        """
        Args:
            features: (batch, input_dim) float tensor of encoded infoset features.
            action_mask: (batch, output_dim) bool tensor, True for legal actions.

        Returns:
            (batch, output_dim) float tensor of strategy probabilities summing to 1
            per row. Illegal actions have probability 0.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        # Validate input shape
        if features.dim() != 2 or features.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features shape: expected (batch, {self._input_dim}), got {tuple(features.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), got {tuple(action_mask.shape)}"
            )
        if features.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: features has {features.shape[0]}, action_mask has {action_mask.shape[0]}"
            )

        # Validate NaN
        if self._validate_inputs and torch.isnan(features).any():
            raise InvalidNetworkInputError("Features tensor contains NaN values")

        out = self.net(features)
        # Mask illegal actions to -inf before softmax
        out = out.masked_fill(~action_mask, float("-inf"))
        # Softmax over action dimension
        probs = F.softmax(out, dim=-1)
        # NaN guard: if all actions are masked (shouldn't happen), replace with 0
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs


class HistoryValueNetwork(nn.Module):
    """Predicts scalar utility V(h) for traversing player at game history h.

    Input: concatenation of both players' infoset encodings (444-dim = 2 * INPUT_DIM).
    Output: scalar utility estimate (batch, 1).

    Architecture: 444 -> 512 -> 512 -> 256 -> 1
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM * 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._validate_inputs = validate_inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, features_both: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_both: (batch, input_dim) float tensor of concatenated both-player
                infoset encodings. input_dim is typically 444 (2 * INPUT_DIM).

        Returns:
            (batch, 1) float tensor of scalar utility estimates.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        if features_both.dim() != 2 or features_both.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid features_both shape: expected (batch, {self._input_dim}), "
                f"got {tuple(features_both.shape)}"
            )

        if self._validate_inputs and torch.isnan(features_both).any():
            raise InvalidNetworkInputError("features_both tensor contains NaN values")

        return self.net(features_both)


class PBSValueNetwork(nn.Module):
    """Predicts counterfactual values for each player's hand types given a PBS encoding.

    Input: PBS encoding (batch, input_dim).
    Output: counterfactual value vector (batch, output_dim).

    Architecture: input_dim -> hidden_dim -> hidden_dim -> 512 -> output_dim
    with GeLU activations, LayerNorm, and Dropout after each of the first two layers.
    ~4.3M parameters with default dims (input_dim=956, hidden_dim=1024, output_dim=936).
    """

    def __init__(
        self,
        input_dim: int = 956,
        hidden_dim: int = 1024,
        output_dim: int = 936,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._validate_inputs = validate_inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, pbs_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pbs_encoding: (batch, input_dim) float tensor of PBS encoding.

        Returns:
            (batch, output_dim) float tensor of counterfactual value estimates.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        if pbs_encoding.dim() != 2 or pbs_encoding.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid pbs_encoding shape: expected (batch, {self._input_dim}), "
                f"got {tuple(pbs_encoding.shape)}"
            )

        if self._validate_inputs and torch.isnan(pbs_encoding).any():
            raise InvalidNetworkInputError("pbs_encoding tensor contains NaN values")

        return self.net(pbs_encoding)


class PBSPolicyNetwork(nn.Module):
    """Predicts action probabilities given a PBS encoding, with action masking.

    Input: PBS encoding (batch, input_dim) + action mask (batch, output_dim).
    Output: masked softmax probabilities (batch, output_dim).

    Architecture: input_dim -> hidden_dim -> 256 -> output_dim
    with GeLU activations and Dropout after the first layer.
    ~0.7M parameters with default dims (input_dim=956, hidden_dim=512, output_dim=146).
    """

    def __init__(
        self,
        input_dim: int = 956,
        hidden_dim: int = 512,
        output_dim: int = 146,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._validate_inputs = validate_inputs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self, pbs_encoding: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pbs_encoding: (batch, input_dim) float tensor of PBS encoding.
            action_mask: (batch, output_dim) bool tensor, True for legal actions.

        Returns:
            (batch, output_dim) float tensor of action probabilities summing to 1
            per row. Illegal actions have probability 0.

        Raises:
            InvalidNetworkInputError: If input shape is incorrect or contains NaN values.
        """
        if pbs_encoding.dim() != 2 or pbs_encoding.shape[1] != self._input_dim:
            raise InvalidNetworkInputError(
                f"Invalid pbs_encoding shape: expected (batch, {self._input_dim}), "
                f"got {tuple(pbs_encoding.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._output_dim:
            raise InvalidNetworkInputError(
                f"Invalid action_mask shape: expected (batch, {self._output_dim}), "
                f"got {tuple(action_mask.shape)}"
            )
        if pbs_encoding.shape[0] != action_mask.shape[0]:
            raise InvalidNetworkInputError(
                f"Batch size mismatch: pbs_encoding has {pbs_encoding.shape[0]}, "
                f"action_mask has {action_mask.shape[0]}"
            )

        if self._validate_inputs and torch.isnan(pbs_encoding).any():
            raise InvalidNetworkInputError("pbs_encoding tensor contains NaN values")

        out = self.net(pbs_encoding)
        # Mask illegal actions to -inf before softmax
        out = out.masked_fill(~action_mask, float("-inf"))
        # Softmax over action dimension
        probs = F.softmax(out, dim=-1)
        # NaN guard: if all actions are masked (shouldn't happen), replace with 0
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs


class CVPN(nn.Module):
    """Counterfactual Value-and-Policy Network for GT-CFR.

    Shared ResNet trunk with dual heads:
    - Value head: predicts per-hand-type CFVs for both players (936-dim)
    - Policy head: predicts action logits with masking (146-dim)

    Architecture (~400K params with defaults):
      input_proj(956 → hidden_dim) → [ResBlock × num_blocks] → {
        value_head: Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(→ 936)
        policy_head: Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(→ 146)
      }
    """

    def __init__(
        self,
        input_dim: int = 956,  # PBS_INPUT_DIM
        hidden_dim: int = 512,
        num_blocks: int = 4,
        value_dim: int = 936,  # 2 * NUM_HAND_TYPES
        policy_dim: int = 146,  # NUM_ACTIONS
        dropout: float = 0.1,
        validate_inputs: bool = True,
        detach_policy_grad: bool = False,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._value_dim = value_dim
        self._policy_dim = policy_dim
        self._validate_inputs = validate_inputs
        self._detach_policy_grad = detach_policy_grad

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Shared trunk
        self.trunk = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])

        # Value head
        half = hidden_dim // 2
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Linear(half, value_dim),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Linear(half, policy_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self, pbs_encoding: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pbs_encoding: (B, input_dim) PBS encoding.
            action_mask: (B, policy_dim) bool, True for legal actions.

        Returns:
            (values, policy_logits):
              values: (B, value_dim) counterfactual value estimates
              policy_logits: (B, policy_dim) masked action logits (-inf for illegal)
        """
        if self._validate_inputs:
            if pbs_encoding.dim() != 2 or pbs_encoding.shape[1] != self._input_dim:
                raise InvalidNetworkInputError(
                    f"Invalid pbs_encoding shape: expected (batch, {self._input_dim}), "
                    f"got {tuple(pbs_encoding.shape)}"
                )
            if action_mask.dim() != 2 or action_mask.shape[1] != self._policy_dim:
                raise InvalidNetworkInputError(
                    f"Invalid action_mask shape: expected (batch, {self._policy_dim}), "
                    f"got {tuple(action_mask.shape)}"
                )
            if pbs_encoding.shape[0] != action_mask.shape[0]:
                raise InvalidNetworkInputError(
                    f"Batch size mismatch: pbs_encoding has {pbs_encoding.shape[0]}, "
                    f"action_mask has {action_mask.shape[0]}"
                )
            if torch.isnan(pbs_encoding).any():
                raise InvalidNetworkInputError("pbs_encoding tensor contains NaN values")

        x = self.input_proj(pbs_encoding)
        x = self.trunk(x)
        values = self.value_head(x)
        policy_logits = self.policy_head(x.detach() if self._detach_policy_grad else x)
        policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))
        return values, policy_logits

    def policy_probs(
        self, pbs_encoding: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: returns softmax policy probabilities (B, policy_dim)."""
        _, logits = self.forward(pbs_encoding, action_mask)
        probs = F.softmax(logits, dim=-1)
        return torch.nan_to_num(probs, nan=0.0)


def build_cvpn(
    input_dim: int = 956,
    hidden_dim: int = 512,
    num_blocks: int = 4,
    value_dim: int = 936,
    policy_dim: int = 146,
    dropout: float = 0.1,
    validate_inputs: bool = True,
    detach_policy_grad: bool = False,
) -> "CVPN":
    """Factory for CVPN with default PBS dimensions."""
    return CVPN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        value_dim=value_dim,
        policy_dim=policy_dim,
        dropout=dropout,
        validate_inputs=validate_inputs,
        detach_policy_grad=detach_policy_grad,
    )


def warm_start_cvpn_from_rebel(
    cvpn: "CVPN",
    policy_state_dict: dict,
    value_state_dict: Optional[dict] = None,
) -> list:
    """Load compatible weights from Phase 1 PBSPolicyNetwork/PBSValueNetwork into CVPN.

    Copies weights layer-by-layer where shapes match. Returns list of skipped keys.
    Policy head gets priority (most useful for PUCT guidance early in training).
    """
    cvpn_state = cvpn.state_dict()
    skipped: list[str] = []

    source_dicts = []
    if policy_state_dict:
        source_dicts.append(policy_state_dict)
    if value_state_dict:
        source_dicts.append(value_state_dict)

    for src_dict in source_dicts:
        for src_key, src_tensor in src_dict.items():
            if src_key in cvpn_state and cvpn_state[src_key].shape == src_tensor.shape:
                cvpn_state[src_key] = src_tensor
            else:
                skipped.append(src_key)

    cvpn.load_state_dict(cvpn_state)
    return skipped


def get_strategy_from_advantages(
    advantages: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute a strategy from advantage predictions using ReLU + normalize (regret matching).

    This is used during traversal to convert the AdvantageNetwork output into
    a probability distribution, matching the RM+ approach from tabular CFR.

    Args:
        advantages: (batch, 146) or (146,) raw advantage values.
        action_mask: Same shape as advantages, bool tensor of legal actions.

    Returns:
        Probability distribution over actions, same shape as input.
        Illegal actions have probability 0.
        Falls back to uniform over legal actions if all advantages <= 0.
    """
    # ReLU: only positive advantages get probability mass
    positive = F.relu(advantages)
    # Mask illegal actions
    positive = positive * action_mask.float()
    # Normalize
    total = positive.sum(dim=-1, keepdim=True)
    # Check if any positive advantages exist
    has_positive = total > 0
    if has_positive.all():
        return positive / total

    # Fallback: uniform over legal actions where total == 0
    uniform = action_mask.float()
    uniform_total = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = uniform / uniform_total

    # Use normalized positive where available, uniform otherwise
    result = torch.where(has_positive, positive / total.clamp(min=1e-10), uniform)
    return result
