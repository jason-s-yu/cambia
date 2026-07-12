"""src/cfr/prtcfr_net.py

PRT-CFR (Perfect-Recall Trajectory CFR) sequence network and the single-sourced
token helper.

PRT-CFR conditions regret and policy on the EXACT observation-action token
sequence (perfect recall by construction; v0.4 design decision 1). The net is a
GRU sequence encoder over the tokenizer's vocabulary (src.sequence_encoding,
vocab 325) followed by a small MLP head producing one raw advantage per
146-action slot. No action abstraction (decision 3). No strategy net: the
average strategy is realized exactly via SD-CFR snapshot sampling (decision 4),
so this single regret-net architecture is the only learned object.

The token helper ``tiny_node_to_tokens`` is the critical parity seam: both the
PRT-CFR worker (training) and the X2 scorer (eval) call it to tokenize the same
tiny_solver tree nodes. If training and eval tokens diverged for one infoset the
net would be queried with mismatched input and NashConv would be meaningless
(the RC-B train/eval representation-mismatch bug class). The seam is single-
sourced: ``tiny_node_to_tokens`` returns the token array the tree builder already
stored on the node (``Decision.seq_tokens``, produced by the one tokenizer in
``tools/tiny_solver.py`` via ``src.sequence_encoding.encode_observation_sequence``),
so there is exactly one token array per node and parity is true by construction.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import NUM_ACTIONS
from ..sequence_encoding import PAD_ID, SEQ_CAP, VOCAB_SIZE

# Architecture defaults (pinned by the Phase 1 Sprint 1 interface contract).
GRU_VOCAB_SIZE: int = VOCAB_SIZE  # 325
GRU_EMBED_DIM: int = 64
GRU_HIDDEN_DIM: int = 256
GRU_NUM_LAYERS: int = 2
GRU_DROPOUT: float = 0.1
HEAD_HIDDEN_DIM: int = 256


# ---------------------------------------------------------------------------
# Token helper (single-sourced parity seam)
# ---------------------------------------------------------------------------


def pad_tokens(tokens: List[int], seq_cap: int = SEQ_CAP) -> np.ndarray:
    """Right-pad (or truncate-keep-most-recent) a token list to a fixed width.

    Returns an int32 array of length ``seq_cap``. Padding uses ``PAD_ID`` (0).
    Tiny-game sequences run ~5-80 tokens, well under the default cap, so this is
    pure right-padding there; the keep-most-recent truncation mirrors
    ``encode_observation_sequence`` for the (unreached, on the tiny game) overflow
    case so the most recent frames survive.
    """
    arr = np.full(seq_cap, PAD_ID, dtype=np.int32)
    if not tokens:
        return arr
    if len(tokens) > seq_cap:
        tokens = tokens[-seq_cap:]
    arr[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
    return arr


def tiny_node_to_tokens(node, seq_cap: int = SEQ_CAP) -> List[int]:
    """Tokenize a tiny_solver Decision node's perfect-recall history.

    SINGLE SOURCE for both training and eval token inputs. The tree builder
    (``tools/tiny_solver.py`` with ``tokenize=True``) stores the acting player's
    perfect-recall observation-action token stream on ``node.seq_tokens`` via the
    one tokenizer (``src.sequence_encoding.encode_observation_sequence``). This
    helper returns that exact stream (unpadded list), so the worker and the X2
    scorer feed byte-identical tokens for any given infoset.

    Raises if the tree was not built with ``tokenize=True`` (``seq_tokens`` is
    None): the caller must build the tree with tokenization enabled.
    """
    toks = getattr(node, "seq_tokens", None)
    if toks is None:
        raise ValueError(
            "tiny_node_to_tokens: node has no seq_tokens. Build the tree with "
            "build_tree(..., tokenize=True) so the single-sourced token stream is "
            "populated on every Decision node."
        )
    if len(toks) > seq_cap:
        return list(toks[-seq_cap:])
    return list(toks)


def tiny_node_to_token_array(node, seq_cap: int = SEQ_CAP) -> np.ndarray:
    """``tiny_node_to_tokens`` padded to a fixed-width int32 array (reservoir feature)."""
    return pad_tokens(tiny_node_to_tokens(node, seq_cap=seq_cap), seq_cap=seq_cap)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class PRTCFRNet(nn.Module):
    """GRU sequence encoder + 146-action advantage head for PRT-CFR.

    encoder: Embedding(vocab, embed) -> GRU(embed, hidden, num_layers,
             batch_first, dropout) -> top-layer final hidden -> LayerNorm(hidden)
    head:    Linear(hidden, head_hidden) -> ReLU -> Linear(head_hidden, NUM_ACTIONS)

    Kaiming-normal init on linear weights, zero bias; LayerNorm ones/zeros
    (matching src/networks.py). The encoder and head expose separate
    state_dicts so SD-CFR snapshots store ``{encoder_state_dict, head_state_dict,
    iteration}`` (the pinned checkpoint format).
    """

    def __init__(
        self,
        vocab_size: int = GRU_VOCAB_SIZE,
        embed_dim: int = GRU_EMBED_DIM,
        hidden_dim: int = GRU_HIDDEN_DIM,
        num_layers: int = GRU_NUM_LAYERS,
        head_hidden_dim: int = HEAD_HIDDEN_DIM,
        num_actions: int = NUM_ACTIONS,
        dropout: float = GRU_DROPOUT,
        pad_id: int = PAD_ID,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head_hidden_dim = head_hidden_dim
        self.num_actions = num_actions
        self.pad_id = pad_id

        # PyTorch warns (and ignores dropout) when num_layers == 1; guard so a
        # 1-layer override stays clean.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.ModuleDict(
            {
                "embed": nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id),
                "gru": nn.GRU(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=gru_dropout,
                ),
                "norm": nn.LayerNorm(hidden_dim),
            }
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_actions),
        )

        self._init_weights()

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(self.device)

    def _init_weights(self):
        # Linear layers: Kaiming-normal (relu), zero bias. Embedding: normal, with
        # the pad row zeroed. GRU: PyTorch's default uniform init (standard for
        # recurrent stacks). LayerNorm: ones/zeros.
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.encoder["embed"].weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.encoder["embed"].weight[self.pad_id].zero_()
        norm = self.encoder["norm"]
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)

    def _embed_pack_gru(
        self, tokens: torch.Tensor, h0: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Shared embed + pack + GRU-forward step: (B, L) tokens plus an optional
        initial hidden state ``h0`` (num_layers, B, hidden; zero-init when None)
        to the new raw (num_layers, B, hidden) hidden state (pre-LayerNorm).

        The GRU reads only the real (non-PAD) prefix of each row via
        pack_padded_sequence, so the top-layer final hidden state is taken AT THE
        LAST REAL TOKEN, not after the right-pad tail. Running the GRU over the long
        zero-embedding PAD tail (seq_cap 256 vs ~5-80 real tokens) drifts the hidden
        state toward a PAD-driven fixed point and washes out the real-prefix signal,
        collapsing infoset discriminability; packing removes that failure.

        Passing ``h0`` from a prior call over ``tokens[0:k]`` and calling this again
        with ``tokens[k:n]`` is the SAME recurrence as one call over the full
        ``tokens[0:n]`` (h0=None) -- the incremental-encode identity ``step_hidden``
        exposes publicly (see cambia-249, prtcfr_infer.PRTCFRInferenceService for the
        production analogue).
        """
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        emb = self.encoder["embed"](tokens)  # (B, L, embed)
        lengths = (tokens != self.pad_id).sum(dim=1).clamp(min=1)  # real prefix length
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        if h0 is None:
            _out, h_n = self.encoder["gru"](packed)  # h_n: (num_layers, B, hidden)
        else:
            _out, h_n = self.encoder["gru"](packed, h0)
        return h_n

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a (B, L) int token batch to a (B, hidden) sequence embedding.

        Top-layer final hidden state (at each row's last real token) through
        LayerNorm. See ``_embed_pack_gru`` for the shared embed+pack+GRU step.
        """
        h_n = self._embed_pack_gru(tokens, h0=None)
        top = h_n[-1]  # (B, hidden) top-layer hidden at each row's last real token
        return self.encoder["norm"](top)

    def step_hidden(
        self, tokens: torch.Tensor, h0: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Advance a raw (num_layers, B, hidden) hidden state by ``tokens`` (B, L),
        WITHOUT LayerNorm (a read-time op, see ``advantages_from_hidden``).

        ``h0=None`` starts from a zero hidden state (equivalent to ``register``
        in prtcfr_infer.PRTCFRInferenceService); a non-None ``h0`` continues an
        existing carry (equivalent to ``step``/``advance``). This is the public
        incremental-encode primitive: callers that cache the returned hidden
        state and feed only newly-appended tokens on each call avoid re-running
        the GRU over the whole prefix at every query (cambia-249).
        """
        return self._embed_pack_gru(tokens, h0)

    def advantages_from_hidden(self, top_hidden: torch.Tensor) -> torch.Tensor:
        """Read-time advantage logits (B, 146) from an already-computed top-layer
        raw hidden state (B, hidden): LayerNorm then head. LayerNorm is a
        read-time op on the carried hidden (never cached), matching ``encode``'s
        own LayerNorm placement.
        """
        return self.head(self.encoder["norm"](top_hidden))

    def strategy_from_hidden(
        self, top_hidden: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Regret-matched strategy (B, 146) from an already-computed top-layer
        raw hidden state. Mirrors ``strategy_from_tokens`` but skips re-deriving
        the hidden state from a full token batch."""
        adv = self.advantages_from_hidden(top_hidden)
        return _regret_match(adv, mask)

    def raw_advantages(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Raw advantage logits (B, 146) for a token batch (B, L).

        ``mask`` is accepted for signature symmetry with the deep-CFR nets but is
        NOT applied here: masking happens in ``strategy_from_tokens`` (regret
        matching) and in the trainer's loss (summed over legal actions only). The
        raw head output over all 146 slots is returned unchanged.
        """
        return self.head(self.encode(tokens))

    def forward(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.raw_advantages(tokens, mask)

    def strategy_from_tokens(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Regret-matched strategy (B, 146) from a token batch.

        ReLU(advantages) * mask, normalized; uniform over legal actions where the
        masked-ReLU sum is 0. Mirrors src.networks.get_strategy_from_advantages so
        the worker's traversal policy matches the deep-CFR regret-matching exactly.
        """
        adv = self.raw_advantages(tokens, mask)
        return _regret_match(adv, mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def head_state_dict(self):
        return self.head.state_dict()

    def load_encoder_head(self, encoder_state_dict, head_state_dict):
        self.encoder.load_state_dict(encoder_state_dict)
        self.head.load_state_dict(head_state_dict)


def _regret_match(advantages: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Regret matching: ReLU(adv) * mask normalized, uniform-over-legal fallback.

    Bit-for-bit the logic of src.networks.get_strategy_from_advantages, restated
    here so prtcfr_net has no import dependency on the deep-CFR network module.
    """
    positive = F.relu(advantages) * action_mask.float()
    total = positive.sum(dim=-1, keepdim=True)
    has_positive = total > 0
    if has_positive.all():
        return positive / total
    uniform = action_mask.float()
    uniform_total = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = uniform / uniform_total
    return torch.where(has_positive, positive / total.clamp(min=1e-10), uniform)


def build_prtcfr_net(config=None, device: Optional[str] = None) -> PRTCFRNet:
    """Construct a PRTCFRNet from a PRTCFRConfig (or defaults).

    ``config`` may be a PRTCFRConfig (or any object exposing the gru_*/head_*
    fields); missing attributes fall back to the module defaults. ``device``
    overrides config.device when given.
    """

    def g(name, default):
        return getattr(config, name, default) if config is not None else default

    dev = device if device is not None else g("device", None)
    return PRTCFRNet(
        vocab_size=g("gru_vocab_size", GRU_VOCAB_SIZE),
        embed_dim=g("gru_embed_dim", GRU_EMBED_DIM),
        hidden_dim=g("gru_hidden_dim", GRU_HIDDEN_DIM),
        num_layers=g("gru_num_layers", GRU_NUM_LAYERS),
        head_hidden_dim=g("head_hidden_dim", HEAD_HIDDEN_DIM),
        dropout=g("gru_dropout", GRU_DROPOUT),
        device=dev,
    )
