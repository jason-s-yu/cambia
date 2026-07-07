"""src/cfr/prtcfr_critic.py

PRT-CFR V_phi critic, OUTSIDE the regret path (p2-redesign.md sec 2.4;
v0.4 Phase 2 Sprint 1 task S1W6).

    "A value net V_phi(omniscient features + sequence) is trained on the
    same data stream with targets = the node's pooled rollout mean ... 90/10
    train/held-out split, held-out MSE logged every iteration against the
    predict-the-buffer-mean baseline (the f5-01 train-loss illusion becomes
    impossible to repeat: the logged number is held-out by construction) ...
    The critic has no influence on regret targets; if it underfits, evals get
    noisier and search stays gated, and nothing else degrades."

Roles (deferred, not implemented here): (a) AIVAT control variate for
evaluation variance (Phase 3), (b) leaf evaluator for the deferred search
phase (Phase 4, gated on held-out MSE beating the constant predictor). This
module only produces the net, the sample buffer, and the fit/held-out-MSE
instrument the X4 battery reads (contract.md Scope: "V_phi critic outside the
regret path").

Three pieces:

  1. ``PRTCFRCriticNet`` -- a small GRU sequence encoder (INDEPENDENT
     parameters from ``prtcfr_net.PRTCFRNet``, the regret net: a fresh
     ``nn.Module`` with its own embedding/GRU/head, never constructed from or
     sharing a submodule with the regret net) fused with the training-only
     omniscient feature vector, predicting the traverser's expected terminal
     utility at h.

  2. Target plumbing: ``omniscient_features_from_driver`` resolves the
     "omniscient-features-input" a ``prtcfr_worker.GameDriver`` supplies at a
     value_sink call into the actual feature vector -- ALWAYS by calling
     ``omniscient.compute_omniscient_features`` (the training-only boundary
     function; this module never open-codes the one-hot extraction itself),
     and ``CriticReservoirSink`` is a ready-made ``value_sink`` callable that
     adapts ``PRTCFRProductionWorker.traverse``'s optional hook
     (``tokens_h, driver, pooled_rollout_mean, iteration``) into
     ``CriticReservoir`` samples.

  3. ``CriticReservoir`` (deterministic 90/10-style held-out split, decided at
     insertion time) + ``CriticTrainer.fit`` (buffer -> SGD steps -> held-out
     MSE vs the predict-the-train-mean constant-predictor baseline, always
     paired -- the f5-01 train-loss illusion is impossible to repeat because
     the logged metric IS the held-out split by construction).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import MAX_HAND
from ..sequence_encoding import PAD_ID, SEQ_CAP, VOCAB_SIZE
from .omniscient import compute_omniscient_features, omniscient_dim
from .prtcfr_net import pad_tokens

# ---------------------------------------------------------------------------
# Omniscient-features-input resolution (training-only boundary).
#
# `compute_omniscient_features(engine)` is duck-typed: it only ever calls
# `engine._get_all_cards_unsafe()` (see omniscient.py). The real GoEngine
# (FFI) satisfies that directly. The current production sampler
# (prtcfr_worker.PythonEngineGameDriver) wraps a pure-Python
# `CambiaGameState` instead, which has no such method -- so a small adapter
# is built here reading ground-truth hands directly off the Python engine and
# re-packing them into the identical byte-format GoEngine._get_all_cards_unsafe
# documents (packed uint8 CardBucket per slot, 0xFF sentinel for
# empty/unknown, MAX_HAND slots per player). This mirrors the established
# codebase pattern for the same problem (cli.py's `_GoEngineAdapter`/
# `_Engine._omniscient_features` and desca_worker.py's `_encode_omniscient`
# dispatch), but -- per this task's explicit rule -- always finishes by
# calling `compute_omniscient_features`, never returning a bespoke one-hot
# vector directly. That keeps exactly one function in the codebase that
# produces omniscient features.
# ---------------------------------------------------------------------------

_EMPTY_SENTINEL = 0xFF


class _PythonEngineOmniscientAdapter:
    """Duck-typed `_get_all_cards_unsafe()` source over a pure-Python
    `CambiaGameState`-like ``game`` object (``game.players[p].hand``).

    Never used directly for feature computation -- always passed through
    `compute_omniscient_features`, matching every other backend's path.
    """

    __slots__ = ("_game", "_num_players")

    def __init__(self, game: Any, num_players: int) -> None:
        self._game = game
        self._num_players = num_players

    def _get_all_cards_unsafe(self) -> np.ndarray:
        from ..abstraction import get_card_bucket

        out = np.full(self._num_players * MAX_HAND, _EMPTY_SENTINEL, dtype=np.uint8)
        players = self._game.players
        for p in range(min(self._num_players, len(players))):
            hand = players[p].hand
            for s in range(min(len(hand), MAX_HAND)):
                card = hand[s]
                if card is None:
                    continue
                bucket = int(get_card_bucket(card).value)
                out[p * MAX_HAND + s] = bucket if bucket < 9 else _EMPTY_SENTINEL
        return out


def omniscient_features_from_driver(driver: Any, num_players: int = 2) -> np.ndarray:
    """Resolve the omniscient feature vector for ANY GameDriver-conforming
    ``driver``, ALWAYS via ``compute_omniscient_features`` (never a bespoke
    extraction path in this function).

    Prefers ``driver``'s own ``_get_all_cards_unsafe`` when present (a
    GoEngine, or any future Go-FFI-backed driver duck-typed the same way --
    the eventual S1W2-integrated production driver lands here for free with
    zero change to this function). Falls back to wrapping
    ``driver.game`` (the current ``PythonEngineGameDriver`` stub) with
    ``_PythonEngineOmniscientAdapter``.

    Raises ``TypeError`` if neither is available -- an explicit failure
    rather than a silent zero-vector, so a misconfigured sink is caught at
    the first sample rather than producing a critic trained on garbage.
    """
    source: Any = driver
    if not hasattr(source, "_get_all_cards_unsafe"):
        game = getattr(driver, "game", None)
        if game is None:
            raise TypeError(
                f"omniscient_features_from_driver: driver {type(driver)!r} exposes "
                "neither `_get_all_cards_unsafe` nor `.game` -- cannot resolve an "
                "omniscient-features-input source."
            )
        source = _PythonEngineOmniscientAdapter(game, num_players)
    return compute_omniscient_features(source)


# ---------------------------------------------------------------------------
# Critic sample + value_sink adapter
# ---------------------------------------------------------------------------


@dataclass
class CriticSample:
    """One V_phi training example: the traverser's full-recall token prefix
    (padded to a fixed width), the training-only omniscient feature vector,
    and the pooled-rollout-mean target (p2 sec 2.4)."""

    tokens: np.ndarray  # int32, (seq_cap,)
    omniscient: np.ndarray  # float32, (omniscient_dim,)
    target: float  # pooled rollout mean (traverser's expected terminal utility)
    iteration: int


class CriticReservoirSink:
    """Ready-made ``value_sink`` for ``PRTCFRProductionWorker.traverse``.

    Adapts the worker's synchronous ``(tokens_h, driver, pooled_mean,
    iteration)`` call into a ``CriticReservoir.add``. Resolves omniscient
    features SYNCHRONOUSLY during the call (per the worker's documented
    contract: ``driver`` is a live reference valid only for the call's
    duration), so nothing here retains a reference to ``driver`` itself --
    only the materialized ``CriticSample`` is stored.
    """

    def __init__(
        self,
        reservoir: "CriticReservoir",
        num_players: int = 2,
        seq_cap: int = SEQ_CAP,
    ):
        self.reservoir = reservoir
        self.num_players = num_players
        self.seq_cap = seq_cap

    def __call__(self, tokens_h, driver: Any, pooled_mean: float, iteration: int) -> None:
        omni = omniscient_features_from_driver(driver, num_players=self.num_players)
        self.reservoir.add(
            CriticSample(
                tokens=pad_tokens(tokens_h, seq_cap=self.seq_cap),
                omniscient=omni.astype(np.float32, copy=False),
                target=float(pooled_mean),
                iteration=int(iteration),
            )
        )


# ---------------------------------------------------------------------------
# Reservoir: fixed-capacity, deterministic held-out split
# ---------------------------------------------------------------------------


@dataclass
class CriticBatch:
    tokens: np.ndarray  # (N, seq_cap) int32
    omniscient: np.ndarray  # (N, omni_dim) float32
    targets: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return int(self.targets.shape[0])


class _ArrayReservoir:
    """Vitter Algorithm R fixed-capacity uniform sample.

    Restates ``src/reservoir.py``'s ``ReservoirBuffer.add`` logic (same
    probability-of-replacement rule) over this module's own columnar schema
    (tokens + omniscient + scalar target), since the regret reservoir's
    single-feature-array / 146-wide-target layout does not fit a
    (tokens, omniscient, scalar) sample.
    """

    def __init__(self, capacity: int, token_dim: int, omni_dim: int, seed: int):
        self.capacity = max(1, int(capacity))
        self._tokens = np.zeros((self.capacity, token_dim), dtype=np.int32)
        self._omni = np.zeros((self.capacity, omni_dim), dtype=np.float32)
        self._targets = np.zeros(self.capacity, dtype=np.float32)
        self._iterations = np.zeros(self.capacity, dtype=np.int64)
        self._size = 0
        self.seen_count = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, sample: CriticSample) -> None:
        self.seen_count += 1
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            idx = self._rng.randint(0, self.seen_count - 1)
            if idx >= self.capacity:
                return
        self._tokens[idx] = sample.tokens
        self._omni[idx] = sample.omniscient
        self._targets[idx] = sample.target
        self._iterations[idx] = sample.iteration

    def sample_batch(
        self, batch_size: int, rng: Optional[random.Random] = None
    ) -> CriticBatch:
        picker = rng if rng is not None else random
        actual = min(batch_size, self._size)
        if actual == 0:
            return CriticBatch(
                tokens=np.empty((0, self._tokens.shape[1]), dtype=np.int32),
                omniscient=np.empty((0, self._omni.shape[1]), dtype=np.float32),
                targets=np.empty(0, dtype=np.float32),
            )
        if actual < self._size:
            idx = np.asarray(picker.sample(range(self._size), actual), dtype=np.int64)
        else:
            idx = np.arange(self._size, dtype=np.int64)
        return CriticBatch(
            tokens=self._tokens[idx].copy(),
            omniscient=self._omni[idx].copy(),
            targets=self._targets[idx].copy(),
        )

    def all_batch(self) -> CriticBatch:
        return CriticBatch(
            tokens=self._tokens[: self._size].copy(),
            omniscient=self._omni[: self._size].copy(),
            targets=self._targets[: self._size].copy(),
        )


class CriticReservoir:
    """Fixed-capacity V_phi sample buffer with a DETERMINISTIC held-out split
    assigned at insertion time.

    p2-redesign.md sec 2.4: "90/10 train/held-out split, held-out MSE logged
    every iteration against the predict-the-buffer-mean baseline". Held-out
    membership is decided by a seeded RNG stream at ``add()`` time, BEFORE
    either sub-reservoir's own Vitter-R eviction runs -- so the split is
    reproducible given the same insertion sequence and seed, independent of
    which specific samples a sub-reservoir later evicts.

    The constant-predictor baseline is the RUNNING MEAN of every sample ever
    routed to the train split (an O(1) sum/count, not the mean of the
    currently-retained train-reservoir contents): the "predict-the-buffer-
    mean baseline" from p2 sec 2.4/3, held out by construction since it never
    reads a held-out target.
    """

    def __init__(
        self,
        capacity: int = 200_000,
        held_out_fraction: float = 0.1,
        seq_cap: int = SEQ_CAP,
        num_players: int = 2,
        seed: int = 0,
    ):
        if not 0.0 < held_out_fraction < 1.0:
            raise ValueError("held_out_fraction must be in (0, 1)")
        self.seq_cap = seq_cap
        self.num_players = num_players
        self.omni_dim = omniscient_dim(num_players)
        self.held_out_fraction = held_out_fraction

        train_capacity = max(1, int(round(capacity * (1.0 - held_out_fraction))))
        holdout_capacity = max(1, int(capacity) - train_capacity)
        self._train = _ArrayReservoir(train_capacity, seq_cap, self.omni_dim, seed=seed)
        self._holdout = _ArrayReservoir(
            holdout_capacity, seq_cap, self.omni_dim, seed=seed + 1
        )
        # Separate RNG stream purely for the train/held-out routing decision
        # -- deterministic given `seed`, independent of either sub-reservoir's
        # own eviction randomness.
        self._split_rng = random.Random(seed + 2)
        self._train_target_sum = 0.0
        self._train_target_count = 0

    def add(self, sample: CriticSample) -> bool:
        """Route ``sample`` to train or held-out. Returns True iff held-out
        (exposed for split-determinism testing)."""
        if sample.tokens.shape[0] != self.seq_cap:
            raise ValueError(
                f"CriticReservoir.add: tokens width {sample.tokens.shape[0]} != "
                f"seq_cap {self.seq_cap}"
            )
        if sample.omniscient.shape[0] != self.omni_dim:
            raise ValueError(
                f"CriticReservoir.add: omniscient dim {sample.omniscient.shape[0]} != "
                f"{self.omni_dim} (num_players={self.num_players})"
            )
        is_held_out = self._split_rng.random() < self.held_out_fraction
        if is_held_out:
            self._holdout.add(sample)
        else:
            self._train.add(sample)
            self._train_target_sum += sample.target
            self._train_target_count += 1
        return is_held_out

    def __len__(self) -> int:
        return len(self._train) + len(self._holdout)

    @property
    def train_len(self) -> int:
        return len(self._train)

    @property
    def held_out_len(self) -> int:
        return len(self._holdout)

    def train_target_mean(self) -> float:
        """The constant-predictor baseline: running mean of every target ever
        routed to the train split (see class docstring)."""
        if self._train_target_count == 0:
            return 0.0
        return self._train_target_sum / self._train_target_count

    def sample_train_batch(
        self, batch_size: int, rng: Optional[random.Random] = None
    ) -> CriticBatch:
        return self._train.sample_batch(batch_size, rng=rng)

    def held_out_batch(self) -> CriticBatch:
        return self._holdout.all_batch()


# ---------------------------------------------------------------------------
# Network: V_phi(omniscient features + sequence), independent of the regret
# net by construction (a fresh nn.Module, no shared submodules).
# ---------------------------------------------------------------------------

# "Own small GRU" (p2 sec 2.4): deliberately smaller than PRTCFRNet's
# regret-net encoder (embed 64 / hidden 256) -- the critic is a lighter,
# secondary object outside the regret path.
CRITIC_GRU_EMBED_DIM: int = 32
CRITIC_GRU_HIDDEN_DIM: int = 128
CRITIC_GRU_NUM_LAYERS: int = 1
CRITIC_GRU_DROPOUT: float = 0.0
CRITIC_HEAD_HIDDEN_DIM: int = 128


class PRTCFRCriticNet(nn.Module):
    """V_phi(h): a value net over the traverser's full-recall token sequence
    (own small GRU encoder) fused with the training-only omniscient feature
    vector (p2-redesign.md sec 2.4).

    INDEPENDENT parameters from ``prtcfr_net.PRTCFRNet`` (the regret net): a
    fresh ``nn.Module`` instance with its own embedding/GRU/head, never
    constructed from or sharing a submodule with the regret net -- so
    gradients through the critic loss can never reach the regret net
    (decision 2: no critic in the regret path).

    Output: a single scalar per row, tanh-squashed to (-1, 1). Cambia 2P
    per-game utilities are exactly {-1, 0, +1}; a pooled multi-rollout mean is
    a convex combination of terminal utilities and stays inside that range.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = CRITIC_GRU_EMBED_DIM,
        hidden_dim: int = CRITIC_GRU_HIDDEN_DIM,
        num_layers: int = CRITIC_GRU_NUM_LAYERS,
        omniscient_input_dim: int = 120,
        head_hidden_dim: int = CRITIC_HEAD_HIDDEN_DIM,
        dropout: float = CRITIC_GRU_DROPOUT,
        pad_id: int = PAD_ID,
        device: Optional[str] = "cuda",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omniscient_input_dim = omniscient_input_dim
        self.head_hidden_dim = head_hidden_dim
        self.pad_id = pad_id

        # PyTorch warns (and ignores dropout) when num_layers == 1.
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
            nn.Linear(hidden_dim + omniscient_input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

        self._init_weights()

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(self.device)

    def _init_weights(self) -> None:
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

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a (B, L) int token batch to a (B, hidden) sequence
        embedding, reading only the real (non-PAD) prefix of each row via
        pack_padded_sequence (mirrors ``prtcfr_net.PRTCFRNet.encode``)."""
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        emb = self.encoder["embed"](tokens)  # (B, L, embed)
        lengths = (tokens != self.pad_id).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        _out, h_n = self.encoder["gru"](packed)
        top = h_n[-1]
        return self.encoder["norm"](top)

    def forward(self, tokens: torch.Tensor, omniscient: torch.Tensor) -> torch.Tensor:
        """Returns (B,) predicted values in (-1, 1)."""
        h = self.encode(tokens)
        fused = torch.cat([h, omniscient.to(dtype=h.dtype, device=h.device)], dim=-1)
        return torch.tanh(self.head(fused).squeeze(-1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_prtcfr_critic_net(
    config: Any = None, num_players: int = 2, device: Optional[str] = None
) -> PRTCFRCriticNet:
    """Construct a PRTCFRCriticNet from a config-like object (or defaults).

    ``config`` may expose ``critic_gru_*``/``critic_head_hidden_dim`` fields
    (getattr-config pattern: S1W5 wires real PRTCFRConfig fields later, out of
    this task's scope -- this factory works with plain constructor args or a
    duck-typed config today without needing config.py touched).
    """

    def g(name, default):
        return getattr(config, name, default) if config is not None else default

    dev = device if device is not None else g("critic_device", g("device", "cuda"))
    return PRTCFRCriticNet(
        vocab_size=g("critic_vocab_size", VOCAB_SIZE),
        embed_dim=g("critic_gru_embed_dim", CRITIC_GRU_EMBED_DIM),
        hidden_dim=g("critic_gru_hidden_dim", CRITIC_GRU_HIDDEN_DIM),
        num_layers=g("critic_gru_num_layers", CRITIC_GRU_NUM_LAYERS),
        omniscient_input_dim=g("critic_omniscient_dim", omniscient_dim(num_players)),
        head_hidden_dim=g("critic_head_hidden_dim", CRITIC_HEAD_HIDDEN_DIM),
        dropout=g("critic_gru_dropout", CRITIC_GRU_DROPOUT),
        device=dev,
    )


# ---------------------------------------------------------------------------
# Fit loop + held-out MSE vs constant-predictor baseline
# ---------------------------------------------------------------------------


@dataclass
class CriticFitMetrics:
    """Always-paired instrument (p2 sec 2.4/3): held-out MSE is meaningless
    without the constant-predictor baseline computed from disjoint (train-
    only) data next to it. ``ratio`` < 1 means the critic beats a constant
    prediction on held-out data; the f5-01 illusion (a logged number that was
    actually train loss) cannot recur because ``held_out_mse`` is computed
    exclusively from the held-out split by construction."""

    held_out_mse: float
    constant_baseline_mse: float
    ratio: float
    n_train_seen: int
    n_train_batch_steps: int
    n_held_out: int
    final_train_loss: float = field(default=float("nan"))


def _fit_metrics_from_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    constant_baseline: float,
    n_train_seen: int,
    n_train_batch_steps: int,
    final_train_loss: float,
) -> CriticFitMetrics:
    """Pure MSE/ratio math, factored out of ``CriticTrainer.evaluate_held_out``
    so it is independently testable without a net or a reservoir."""
    n = int(targets.shape[0])
    if n == 0:
        return CriticFitMetrics(
            held_out_mse=float("nan"),
            constant_baseline_mse=float("nan"),
            ratio=float("nan"),
            n_train_seen=n_train_seen,
            n_train_batch_steps=n_train_batch_steps,
            n_held_out=0,
            final_train_loss=final_train_loss,
        )
    held_out_mse = float(np.mean((predictions - targets) ** 2))
    constant_baseline_mse = float(np.mean((targets - constant_baseline) ** 2))
    if constant_baseline_mse <= 1e-12:
        ratio = 0.0 if held_out_mse <= 1e-12 else float("inf")
    else:
        ratio = held_out_mse / constant_baseline_mse
    return CriticFitMetrics(
        held_out_mse=held_out_mse,
        constant_baseline_mse=constant_baseline_mse,
        ratio=ratio,
        n_train_seen=n_train_seen,
        n_train_batch_steps=n_train_batch_steps,
        n_held_out=n,
        final_train_loss=final_train_loss,
    )


class CriticTrainer:
    """Fits a ``PRTCFRCriticNet`` on a ``CriticReservoir``.

    p2-redesign.md sec 2.4: held-out MSE logged every iteration against the
    predict-the-buffer-mean constant-predictor baseline -- the f5-01
    train-loss illusion (a logged number that was actually TRAIN loss;
    ``desca_trainer.py``'s V_omni logged 0.15-0.18 vs the true held-out
    0.754) is impossible to repeat here: ``fit()`` always returns a metric
    computed on the held-out split, always paired with the constant baseline
    computed from disjoint (train-only) data.
    """

    def __init__(self, net: PRTCFRCriticNet, lr: float = 1e-3, weight_decay: float = 0.0):
        self.net = net
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _forward_batch(self, batch: CriticBatch) -> torch.Tensor:
        tokens = torch.as_tensor(batch.tokens, dtype=torch.long, device=self.net.device)
        omni = torch.as_tensor(
            batch.omniscient, dtype=torch.float32, device=self.net.device
        )
        return self.net(tokens, omni)

    def fit(
        self,
        reservoir: CriticReservoir,
        steps: int,
        batch_size: int,
        rng: Optional[random.Random] = None,
    ) -> CriticFitMetrics:
        """Runs up to ``steps`` Adam/MSE SGD steps over train-split batches
        (stops early if the train split is empty), then returns held-out MSE
        vs the constant-predictor baseline (always computed, even if 0 steps
        ran)."""
        self.net.train()
        final_loss = float("nan")
        steps_run = 0
        for _ in range(steps):
            batch = reservoir.sample_train_batch(batch_size, rng=rng)
            if len(batch) == 0:
                break
            pred = self._forward_batch(batch)
            targets = torch.as_tensor(
                batch.targets, dtype=pred.dtype, device=self.net.device
            )
            loss = F.mse_loss(pred, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.item())
            steps_run += 1
        return self.evaluate_held_out(
            reservoir, n_train_batch_steps=steps_run, final_train_loss=final_loss
        )

    def evaluate_held_out(
        self,
        reservoir: CriticReservoir,
        n_train_batch_steps: int = 0,
        final_train_loss: float = float("nan"),
    ) -> CriticFitMetrics:
        """Held-out MSE vs the constant-predictor baseline (train-buffer
        mean). Held out by construction: ``reservoir.held_out_batch()`` never
        overlaps ``reservoir.sample_train_batch``'s source data (see
        ``CriticReservoir``'s deterministic split)."""
        self.net.eval()
        batch = reservoir.held_out_batch()
        constant = reservoir.train_target_mean()
        if len(batch) == 0:
            predictions = np.empty(0, dtype=np.float32)
        else:
            with torch.no_grad():
                predictions = self._forward_batch(batch).detach().cpu().numpy()
        return _fit_metrics_from_predictions(
            predictions,
            batch.targets,
            constant_baseline=constant,
            n_train_seen=reservoir.train_len,
            n_train_batch_steps=n_train_batch_steps,
            final_train_loss=final_train_loss,
        )


__all__ = [
    "CriticBatch",
    "CriticFitMetrics",
    "CriticReservoir",
    "CriticReservoirSink",
    "CriticSample",
    "CriticTrainer",
    "PRTCFRCriticNet",
    "build_prtcfr_critic_net",
    "omniscient_features_from_driver",
]
