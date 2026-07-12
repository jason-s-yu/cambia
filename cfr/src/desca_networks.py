"""DESCA networks for v3.1 Phase 1 (Dense ESCHER + Semantic Action Abstraction).

Three networks share a common trunk architecture but maintain independent weights:
    Linear(input_dim, hidden_dim) -> LayerNorm -> SiLU -> 3 x ResBlock(hidden_dim)

Networks:
- RegretNetwork: 257-dim infoset features -> 32-dim per-action regret vector.
- AvgStrategyNetwork: 257-dim features + action mask -> 32-dim masked-softmax probs.
- HistoryValueNetwork: 257-dim features + optional 120-dim hidden-card one-hot
  (omniscient training) or zero-pad (fair eval) -> scalar value.

Reuses ``_ResBlock`` from ``src.networks`` for residual blocks. Trunk replaces
the inner ReLU activations of ``_ResBlock`` only at the top-level boundary; per
contract the trunk-level activation between input projection and first block is
SiLU. Shared trunk has independent parameters per network in Phase 1; weight
distillation may follow in later phases.

Omniscient gradient leakage prevention: ``HistoryValueNetwork`` is constructed
as a separate ``nn.Module`` with its own parameters. Backward through the value
output cannot reach ``RegretNetwork`` or ``AvgStrategyNetwork`` parameters
unless callers explicitly merge optimizers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EP_PBS_V2_INPUT_DIM
from .action_abstraction import NUM_ABSTRACT_ACTIONS_2P
from .networks import _ResBlock

DEFAULT_HIDDEN_DIM: int = 512
DEFAULT_NUM_BLOCKS: int = 3
DEFAULT_OMNISCIENT_DIM_2P: int = 120  # 2 players x 6 max hand slots x 10 buckets


def _build_trunk(
    input_dim: int,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the shared trunk: input projection then N residual blocks.

    Layout: ``Linear(input_dim, hidden_dim) -> LayerNorm -> SiLU -> N x _ResBlock``.
    """

    class _Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.input_act = nn.SiLU()
            self.blocks = nn.ModuleList(
                [_ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.input_act(x)
            for block in self.blocks:
                x = block(x)
            return x

    return _Trunk()


def _init_module(module: nn.Module) -> None:
    """Kaiming-init linear layers; standard init for LayerNorm and Embedding."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class RegretNetwork(nn.Module):
    """Per-iteration regret estimator over the abstract action space.

    Forward returns raw regret values; legal-action filtering is the caller's
    responsibility (regret-matching downstream applies ReLU + normalize and
    handles masking via ``get_strategy_from_advantages`` or equivalent).
    """

    def __init__(
        self,
        input_dim: int = EP_PBS_V2_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_actions: int = NUM_ABSTRACT_ACTIONS_2P,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._num_actions = int(num_actions)
        self.trunk = _build_trunk(input_dim, hidden_dim, num_blocks, dropout)
        self.head = nn.Linear(hidden_dim, num_actions)
        _init_module(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self._input_dim:
            raise ValueError(
                f"RegretNetwork expected (batch, {self._input_dim}), got {tuple(x.shape)}"
            )
        return self.head(self.trunk(x))


class AvgStrategyNetwork(nn.Module):
    """Average-strategy estimator with masked-softmax output.

    Illegal abstract actions receive zero probability. Probabilities sum to 1
    over the legal-action subset. If a row has no legal actions (degenerate
    edge case), the row is replaced with zeros via ``nan_to_num``.
    """

    def __init__(
        self,
        input_dim: int = EP_PBS_V2_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_actions: int = NUM_ABSTRACT_ACTIONS_2P,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._num_actions = int(num_actions)
        self.trunk = _build_trunk(input_dim, hidden_dim, num_blocks, dropout)
        self.head = nn.Linear(hidden_dim, num_actions)
        _init_module(self)

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self._input_dim:
            raise ValueError(
                f"AvgStrategyNetwork expected (batch, {self._input_dim}), got {tuple(x.shape)}"
            )
        if action_mask.dim() != 2 or action_mask.shape[1] != self._num_actions:
            raise ValueError(
                f"action_mask expected (batch, {self._num_actions}), got {tuple(action_mask.shape)}"
            )
        if action_mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]}, mask has {action_mask.shape[0]}"
            )
        logits = self.head(self.trunk(x))
        mask_bool = action_mask.bool()
        logits = logits.masked_fill(~mask_bool, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.nan_to_num(probs, nan=0.0)


class HistoryValueNetwork(nn.Module):
    """Scalar history-value estimator with optional perfect-info conditioning.

    Trunk input is the concatenation ``[features (input_dim), hidden_cards (omniscient_dim)]``.
    When ``hidden_cards`` is ``None`` (fair / deployment mode), the omniscient
    channel is replaced with zeros so the same trunk and head are used. This
    matches the contract: training pass uses omniscient features, eval pass
    uses zeros, both share weights.
    """

    def __init__(
        self,
        input_dim: int = EP_PBS_V2_INPUT_DIM,
        omniscient_dim: int = DEFAULT_OMNISCIENT_DIM_2P,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        value_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._omniscient_dim = int(omniscient_dim)
        self._value_dim = int(value_dim)
        trunk_input = self._input_dim + self._omniscient_dim
        self.trunk = _build_trunk(trunk_input, hidden_dim, num_blocks, dropout)
        self.head = nn.Linear(hidden_dim, value_dim)
        _init_module(self)

    def forward(
        self, x: torch.Tensor, hidden_cards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self._input_dim:
            raise ValueError(
                f"HistoryValueNetwork expected (batch, {self._input_dim}), got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        if hidden_cards is None:
            omni = torch.zeros(
                batch, self._omniscient_dim, dtype=x.dtype, device=x.device
            )
        else:
            if hidden_cards.dim() != 2 or hidden_cards.shape[1] != self._omniscient_dim:
                raise ValueError(
                    f"hidden_cards expected (batch, {self._omniscient_dim}), "
                    f"got {tuple(hidden_cards.shape)}"
                )
            if hidden_cards.shape[0] != batch:
                raise ValueError(
                    f"Batch size mismatch: x has {batch}, hidden_cards has "
                    f"{hidden_cards.shape[0]}"
                )
            omni = hidden_cards.to(dtype=x.dtype, device=x.device)
        merged = torch.cat([x, omni], dim=-1)
        return self.head(self.trunk(merged))


__all__ = [
    "RegretNetwork",
    "AvgStrategyNetwork",
    "HistoryValueNetwork",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_OMNISCIENT_DIM_2P",
]
