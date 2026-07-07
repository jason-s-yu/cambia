"""tests/test_prtcfr_critic.py

Scoped tests for the PRT-CFR V_phi critic (src/cfr/prtcfr_critic.py),
S1W6 -- outside the regret path (p2-redesign.md sec 2.4).

Coverage:
  - PRTCFRCriticNet forward shape/range, and full parameter/gradient
    isolation from the regret net (prtcfr_net.PRTCFRNet).
  - PRTCFRProductionWorker's value_sink hook emits the EXACT pooled rollout
    mean (baseline) already used for that traverser decision's regret
    targets, driven through a tiny fully-deterministic scripted GameDriver
    (zero MC variance, so the expected value is exact, not statistical).
  - omniscient_features_from_driver resolves through compute_omniscient_features
    for a Python-engine-shaped driver.
  - CriticReservoir's held-out split is deterministic given a fixed seed and
    insertion order.
  - Constant-predictor-baseline MSE/ratio math (pure function, edge cases).
  - Smoke fit: held-out MSE drops below the constant-predictor baseline on a
    synthetic signal the omniscient input alone determines.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card import Card  # noqa: E402
from src.constants import ActionCallCambia, ActionDrawStockpile  # noqa: E402
from src.encoding import NUM_ACTIONS, action_to_index  # noqa: E402
from src.reservoir import ReservoirBuffer  # noqa: E402
from src.cfr.omniscient import omniscient_dim  # noqa: E402
from src.cfr.prtcfr_net import PRTCFRNet, pad_tokens  # noqa: E402
from src.cfr.prtcfr_worker import PRTCFRProductionWorker  # noqa: E402
from src.cfr.prtcfr_critic import (  # noqa: E402
    CriticReservoir,
    CriticSample,
    CriticTrainer,
    PRTCFRCriticNet,
    _fit_metrics_from_predictions,
    omniscient_features_from_driver,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Net shapes / gradient isolation from the regret net
# ---------------------------------------------------------------------------


def _tiny_critic_net(seq_cap: int = 8, omni_dim: int = 120) -> PRTCFRCriticNet:
    return PRTCFRCriticNet(
        vocab_size=32,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        omniscient_input_dim=omni_dim,
        head_hidden_dim=16,
        device=_DEVICE,
    )


def test_critic_net_forward_shape_and_range():
    net = _tiny_critic_net(seq_cap=8, omni_dim=12)
    batch = 5
    tokens = torch.randint(0, 32, (batch, 8), device=_DEVICE)
    omni = torch.rand(batch, 12, device=_DEVICE)
    out = net(tokens, omni)
    assert out.shape == (batch,)
    assert torch.all(out > -1.0) and torch.all(out < 1.0)


def test_critic_net_independent_of_regret_net():
    """No shared parameter tensors, and backpropagating a critic-only loss
    must never populate .grad on any regret-net parameter (decision 2: no
    critic in the regret path)."""
    regret_net = PRTCFRNet(
        vocab_size=32,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        head_hidden_dim=16,
        device=_DEVICE,
    )
    critic_net = _tiny_critic_net(seq_cap=8, omni_dim=12)

    regret_param_ids = {id(p) for p in regret_net.parameters()}
    critic_param_ids = {id(p) for p in critic_net.parameters()}
    assert regret_param_ids.isdisjoint(critic_param_ids)
    assert len(regret_param_ids) > 0 and len(critic_param_ids) > 0

    for p in regret_net.parameters():
        assert p.grad is None

    tokens = torch.randint(0, 32, (4, 8), device=_DEVICE)
    omni = torch.rand(4, 12, device=_DEVICE)
    target = torch.zeros(4, device=_DEVICE)
    loss = torch.nn.functional.mse_loss(critic_net(tokens, omni), target)
    loss.backward()

    for p in critic_net.parameters():
        # Every critic param that participates in the forward graph gets a
        # gradient; embedding rows for unseen tokens may stay zero but the
        # tensor itself is populated (not None) once backward has run over
        # the whole module graph -- head/gru/embed are all fully connected.
        assert p.grad is not None
    for p in regret_net.parameters():
        assert p.grad is None, "critic backward must never reach the regret net"


# ---------------------------------------------------------------------------
# value_sink pooled-mean correctness (tiny deterministic scripted GameDriver)
# ---------------------------------------------------------------------------


class _ScriptedDriver:
    """A single traverser decision with exactly 2 legal actions, each
    resolving IMMEDIATELY to a terminal state: ActionDrawStockpile ->
    traverser utility +1.0, ActionCallCambia -> traverser utility -1.0.

    Zero rollout variance by construction (a "rollout" from either child is
    already terminal, so `_rollout_to_terminal`'s while-loop short-circuits
    on the first check and returns the fixed utility) -- the pooled mean the
    worker computes is EXACT for any m_rollouts, not a statistical estimate,
    so the sink correctness check below uses a zero-tolerance comparison
    against a hand-computed sigma-weighted value.
    """

    def __init__(self, terminal: bool = False, util: float = 0.0):
        self._terminal = terminal
        self._util = util

    def current_player(self) -> int:
        return 0

    def is_terminal(self) -> bool:
        return self._terminal

    def utility(self, player: int) -> float:
        return self._util if player == 0 else -self._util

    def legal_actions(self):
        return [] if self._terminal else [ActionDrawStockpile(), ActionCallCambia()]

    def apply(self, action) -> bool:
        if isinstance(action, ActionDrawStockpile):
            self._terminal, self._util = True, 1.0
            return True
        if isinstance(action, ActionCallCambia):
            self._terminal, self._util = True, -1.0
            return True
        return False

    def tokens(self, player: int):
        return [5, 6, 7]

    def clone(self) -> "_ScriptedDriver":
        return _ScriptedDriver(self._terminal, self._util)


_IDX_DRAW = action_to_index(ActionDrawStockpile())
_IDX_CAMBIA = action_to_index(ActionCallCambia())


def _fixed_sigma(tokens, legal_mask):
    """p(ActionDrawStockpile) = 0.75, p(ActionCallCambia) = 0.25."""
    probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
    probs[_IDX_DRAW] = 0.75
    probs[_IDX_CAMBIA] = 0.25
    return probs


def test_value_sink_emits_exact_pooled_mean():
    sink_calls = []

    def sink(tokens_h, driver, pooled_mean, iteration):
        # Synchronous contract: materialize everything needed NOW (mirrors
        # CriticReservoirSink's own behavior -- never retain `driver`).
        sink_calls.append(
            (list(tokens_h), driver.is_terminal(), float(pooled_mean), int(iteration))
        )

    worker = PRTCFRProductionWorker(
        sigma=_fixed_sigma,
        m_rollouts=3,
        seq_cap=16,
        seed=0,
        value_sink=sink,
    )
    buf = ReservoirBuffer(capacity=100, input_dim=16, has_mask=True)
    driver = _ScriptedDriver()

    n_added = worker.traverse(driver, traverser=0, iteration=7, buf=buf)

    assert n_added == 1
    assert len(sink_calls) == 1
    tokens_h, was_terminal_at_call, pooled_mean, iteration = sink_calls[0]

    expected_baseline = 0.75 * 1.0 + 0.25 * (-1.0)  # = 0.5
    assert pooled_mean == pytest.approx(expected_baseline, abs=1e-12)
    assert tokens_h == [5, 6, 7]
    assert iteration == 7
    # The sink is called BEFORE the trajectory advances past this decision.
    assert was_terminal_at_call is False
    # The trajectory itself proceeds to a real terminal afterward.
    assert driver.is_terminal() is True


def test_value_sink_default_none_is_zero_behavior_change():
    """Omitting value_sink must not alter sampler output at all."""
    buf_with_none = ReservoirBuffer(capacity=100, input_dim=16, has_mask=True)
    worker = PRTCFRProductionWorker(
        sigma=_fixed_sigma,
        m_rollouts=3,
        seq_cap=16,
        seed=0,
        value_sink=None,
    )
    n = worker.traverse(_ScriptedDriver(), traverser=0, iteration=1, buf=buf_with_none)
    assert n == 1
    assert len(buf_with_none) == 1


# ---------------------------------------------------------------------------
# omniscient_features_from_driver
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, hand):
        self.hand = hand


class _FakeGame:
    def __init__(self, hands):
        self.players = [_FakePlayer(h) for h in hands]


class _FakeDriverWithGame:
    def __init__(self, game):
        self.game = game


def test_omniscient_features_from_driver_python_engine_adapter():
    # Player 0 slot 0 = black King (HIGH_KING bucket 8); player 1 has an
    # empty hand (all sentinel slots).
    game = _FakeGame(hands=[[Card(rank="K", suit="S")], []])
    driver = _FakeDriverWithGame(game)

    feats = omniscient_features_from_driver(driver, num_players=2)
    assert feats.shape == (omniscient_dim(2),)
    assert feats.dtype == np.float32

    reshaped = feats.reshape(-1, 10)
    # Exactly one one-hot bit set per slot.
    np.testing.assert_array_equal(reshaped.sum(axis=1), np.ones(reshaped.shape[0]))
    # Player 0 slot 0 -> bucket index 8 (HIGH_KING).
    assert reshaped[0, 8] == 1.0
    # Player 0 slots 1-5 (empty) and all of player 1's slots -> sentinel bit 9.
    for slot in range(1, 6):
        assert reshaped[slot, 9] == 1.0
    for slot in range(6, 12):
        assert reshaped[slot, 9] == 1.0


def test_omniscient_features_from_driver_raises_without_source():
    class _Empty:
        pass

    with pytest.raises(TypeError):
        omniscient_features_from_driver(_Empty(), num_players=2)


def test_omniscient_features_from_driver_prefers_native_get_all_cards():
    """A driver that already exposes `_get_all_cards_unsafe` (e.g. a future
    Go-FFI-backed driver) is used directly, never wrapped."""

    class _NativeSource:
        def _get_all_cards_unsafe(self):
            out = np.full(2 * 6, 0xFF, dtype=np.uint8)
            out[0] = 2  # player 0 slot 0 -> ACE bucket
            return out

    feats = omniscient_features_from_driver(_NativeSource(), num_players=2)
    reshaped = feats.reshape(-1, 10)
    assert reshaped[0, 2] == 1.0


# ---------------------------------------------------------------------------
# CriticReservoir: deterministic held-out split
# ---------------------------------------------------------------------------


def _make_sample(i: int, seq_cap: int, omni_dim: int) -> CriticSample:
    tokens = pad_tokens([1 + (i % 5), 2 + (i % 3)], seq_cap=seq_cap)
    omni = np.zeros(omni_dim, dtype=np.float32)
    omni[i % omni_dim] = 1.0
    return CriticSample(tokens=tokens, omniscient=omni, target=float(i), iteration=1)


def test_held_out_split_is_deterministic_given_fixed_seed():
    seq_cap, omni_dim = 4, omniscient_dim(2)

    def run():
        res = CriticReservoir(
            capacity=50,
            held_out_fraction=0.3,
            seq_cap=seq_cap,
            num_players=2,
            seed=42,
        )
        decisions = []
        for i in range(200):
            decisions.append(res.add(_make_sample(i, seq_cap, omni_dim)))
        return decisions, res.held_out_batch().targets

    decisions_a, held_targets_a = run()
    decisions_b, held_targets_b = run()

    assert decisions_a == decisions_b
    assert any(decisions_a), "expected at least one held-out routing over 200 inserts"
    np.testing.assert_array_equal(held_targets_a, held_targets_b)


def test_held_out_split_differs_with_different_seed():
    seq_cap, omni_dim = 4, omniscient_dim(2)

    def run(seed):
        res = CriticReservoir(
            capacity=50,
            held_out_fraction=0.3,
            seq_cap=seq_cap,
            num_players=2,
            seed=seed,
        )
        return [res.add(_make_sample(i, seq_cap, omni_dim)) for i in range(200)]

    decisions_seed0 = run(0)
    decisions_seed1 = run(1)
    assert decisions_seed0 != decisions_seed1


def test_reservoir_add_validates_shapes():
    res = CriticReservoir(
        capacity=10, held_out_fraction=0.5, seq_cap=4, num_players=2, seed=0
    )
    bad_tokens = CriticSample(
        tokens=np.zeros(3, dtype=np.int32),  # wrong width
        omniscient=np.zeros(omniscient_dim(2), dtype=np.float32),
        target=0.0,
        iteration=0,
    )
    with pytest.raises(ValueError):
        res.add(bad_tokens)

    bad_omni = CriticSample(
        tokens=np.zeros(4, dtype=np.int32),
        omniscient=np.zeros(5, dtype=np.float32),  # wrong dim
        target=0.0,
        iteration=0,
    )
    with pytest.raises(ValueError):
        res.add(bad_omni)


# ---------------------------------------------------------------------------
# Constant-predictor-baseline MSE/ratio math
# ---------------------------------------------------------------------------


def test_fit_metrics_constant_baseline_math():
    predictions = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    targets = np.array([1.0, -1.0, 0.5], dtype=np.float32)  # perfect predictions
    constant = 0.2

    metrics = _fit_metrics_from_predictions(
        predictions,
        targets,
        constant_baseline=constant,
        n_train_seen=5,
        n_train_batch_steps=10,
        final_train_loss=0.0,
    )

    expected_constant_mse = float(np.mean((targets - constant) ** 2))
    assert metrics.held_out_mse == pytest.approx(0.0, abs=1e-9)
    assert metrics.constant_baseline_mse == pytest.approx(expected_constant_mse)
    assert metrics.ratio == pytest.approx(0.0, abs=1e-9)
    assert metrics.n_held_out == 3
    assert metrics.n_train_seen == 5
    assert metrics.n_train_batch_steps == 10


def test_fit_metrics_worse_than_constant_gives_ratio_above_one():
    targets = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    predictions = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)  # maximally wrong
    constant = 0.0  # mean of targets

    metrics = _fit_metrics_from_predictions(
        predictions,
        targets,
        constant_baseline=constant,
        n_train_seen=4,
        n_train_batch_steps=1,
        final_train_loss=1.0,
    )
    assert metrics.ratio > 1.0
    assert metrics.held_out_mse == pytest.approx(4.0)
    assert metrics.constant_baseline_mse == pytest.approx(1.0)


def test_fit_metrics_empty_held_out_is_nan():
    metrics = _fit_metrics_from_predictions(
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        constant_baseline=0.0,
        n_train_seen=10,
        n_train_batch_steps=5,
        final_train_loss=0.1,
    )
    assert np.isnan(metrics.held_out_mse)
    assert np.isnan(metrics.constant_baseline_mse)
    assert np.isnan(metrics.ratio)
    assert metrics.n_held_out == 0


def test_fit_metrics_degenerate_zero_constant_mse():
    # Every held-out target exactly equals the constant baseline.
    targets = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    constant = 0.3
    perfect = _fit_metrics_from_predictions(
        targets.copy(),
        targets,
        constant_baseline=constant,
        n_train_seen=3,
        n_train_batch_steps=1,
        final_train_loss=0.0,
    )
    assert perfect.constant_baseline_mse == pytest.approx(0.0, abs=1e-12)
    assert perfect.ratio == pytest.approx(0.0, abs=1e-9)

    imperfect = _fit_metrics_from_predictions(
        targets + 1.0,
        targets,
        constant_baseline=constant,
        n_train_seen=3,
        n_train_batch_steps=1,
        final_train_loss=0.0,
    )
    assert imperfect.constant_baseline_mse == pytest.approx(0.0, abs=1e-12)
    assert imperfect.ratio == float("inf")


# ---------------------------------------------------------------------------
# Smoke fit: held-out MSE beats the constant baseline on a learnable signal
# ---------------------------------------------------------------------------


def test_smoke_fit_beats_constant_baseline_on_learnable_signal():
    """Synthetic target is a deterministic function of ONE omniscient slot
    (player 0, hand slot 0 bucket == HIGH_KING(8) -> +1.0, else -1.0);
    everything else (tokens, remaining 11 slots) is noise. A net conditioned
    on the omniscient input should learn this quickly; a constant predictor
    cannot (targets are balanced +-1, so its MSE floor is ~1.0)."""
    rng = np.random.RandomState(0)
    seq_cap, num_players = 8, 2
    omni_dim = omniscient_dim(num_players)

    reservoir = CriticReservoir(
        capacity=4000,
        held_out_fraction=0.2,
        seq_cap=seq_cap,
        num_players=num_players,
        seed=123,
    )

    n_samples = 1500
    for i in range(n_samples):
        tokens = pad_tokens(
            list(rng.randint(1, 31, size=rng.randint(1, seq_cap))), seq_cap=seq_cap
        )
        omni = np.zeros((num_players * 6, 10), dtype=np.float32)
        for slot in range(num_players * 6):
            bucket = rng.randint(0, 10)
            omni[slot, bucket] = 1.0
        is_high_king = rng.rand() < 0.5
        omni[0, :] = 0.0
        omni[0, 8 if is_high_king else rng.choice([b for b in range(9) if b != 8])] = 1.0
        target = 1.0 if is_high_king else -1.0

        reservoir.add(
            CriticSample(
                tokens=tokens,
                omniscient=omni.reshape(-1).astype(np.float32),
                target=target,
                iteration=1,
            )
        )

    assert reservoir.train_len > 0 and reservoir.held_out_len > 0

    net = PRTCFRCriticNet(
        vocab_size=32,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        omniscient_input_dim=omni_dim,
        head_hidden_dim=32,
        device="cpu",
    )
    trainer = CriticTrainer(net, lr=2e-3)
    fit_rng = random.Random(7)
    metrics = trainer.fit(reservoir, steps=300, batch_size=64, rng=fit_rng)

    assert metrics.n_held_out > 0
    assert metrics.held_out_mse < metrics.constant_baseline_mse
    assert metrics.ratio < 0.5, (
        f"expected the critic to clearly beat the constant baseline on a "
        f"directly-omniscient-determined signal; got ratio={metrics.ratio} "
        f"(held_out_mse={metrics.held_out_mse}, "
        f"constant_baseline_mse={metrics.constant_baseline_mse})"
    )
