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


def build_advantage_network(
    input_dim: int = INPUT_DIM,
    hidden_dim: int = 256,
    output_dim: int = NUM_ACTIONS,
    dropout: float = 0.1,
    validate_inputs: bool = True,
    num_hidden_layers: int = 2,
    use_residual: bool = False,
) -> nn.Module:
    """Factory: returns ResidualAdvantageNetwork when use_residual=True, else AdvantageNetwork."""
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
        import warnings

        warnings.warn(
            "PBSValueNetwork is DEPRECATED and will be removed. "
            "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
            "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        import warnings

        warnings.warn(
            "PBSPolicyNetwork is DEPRECATED and will be removed. "
            "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
            "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
            DeprecationWarning,
            stacklevel=2,
        )
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
