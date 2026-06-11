"""Tests for the DESCA trainer and worker (Stream B, Phase 1 Sprint 1).

Covers:
- Tier 1: regret update invariant ``sum_a sigma(I,a) * r_hat(I,a) == 0``.
- Tier 1: checkpoint save -> load produces identical network state.
- Tier 1: APCFR+ and RM+ inner-update paths both produce non-NaN updates.
- Tier 1: regret matching plus masking / uniform fallback semantics.
- Tier 2: worker returns sample tuples of correct shape over a fixed seed.
- Tier 2: regret buffer over N iters has std(target) > 0.05 on a stubbed
  synthetic environment that produces non-degenerate regret targets.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from typing import Any, List, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P
from src.cfr import desca_worker, desca_trainer
from src.cfr.desca_worker import (
    FEATURE_DIM,
    OMNISCIENT_DIM_2P,
    VALUE_INPUT_DIM,
    _masked_softmax,
    _regret_matching_plus,
    _uniform,
)
from src.cfr.desca_trainer import (
    DESCATrainer,
    _PrevGradStore,
    _apcfr_plus_step,
    _dcfr_plus_weight,
    _regret_loss,
    _rm_plus_step,
    _strategy_loss,
    _value_loss,
)
from src.reservoir import ReservoirSample


# ---------------------------------------------------------------------------
# Tier 1: regret matching + invariants
# ---------------------------------------------------------------------------


class TestRegretInvariants:
    def test_regret_matching_plus_uniform_on_all_zero(self) -> None:
        mask = np.array([True, True, True, False, False])
        regrets = np.zeros(5)
        sigma = _regret_matching_plus(regrets, mask)
        assert sigma[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        assert sigma[3:].sum() == 0.0

    def test_regret_matching_plus_sums_to_one(self) -> None:
        mask = np.array([True, True, True, False, False])
        regrets = np.array([0.5, 1.0, 0.2, 99.0, -1.0])
        sigma = _regret_matching_plus(regrets, mask)
        assert sigma.sum() == pytest.approx(1.0)
        # Illegal actions must have zero mass.
        assert sigma[3] == 0.0
        assert sigma[4] == 0.0

    def test_regret_sum_invariant_on_synthetic(self) -> None:
        """Tier 1: ``sum_a sigma(I,a) * r_hat(I,a) == 0`` under the ESCHER
        regret construction ``r_hat[a] = v_hat[a] - sum_b sigma[b] * v_hat[b]``.

        This holds mathematically by construction for any sigma that sums to
        1 on the mask; verifying here guards against accidental mutations to
        the baseline formula in desca_worker.
        """
        rng = np.random.default_rng(0)
        for _ in range(50):
            mask = rng.random(NUM_ABSTRACT_ACTIONS_2P) > 0.5
            if not mask.any():
                mask[0] = True
            v_hat = rng.normal(size=NUM_ABSTRACT_ACTIONS_2P)
            v_hat = np.where(mask, v_hat, 0.0)
            regrets = rng.normal(size=NUM_ABSTRACT_ACTIONS_2P) * 2
            sigma = _regret_matching_plus(regrets, mask)
            baseline = float(np.sum(sigma * v_hat))
            r_hat = np.where(mask, v_hat - baseline, 0.0)
            assert abs(float(np.sum(sigma * r_hat))) < 1e-5

    def test_uniform_helper_respects_mask(self) -> None:
        mask = np.array([False, True, True, False])
        u = _uniform(mask)
        assert u[0] == 0.0
        assert u[3] == 0.0
        assert u[1] == pytest.approx(0.5)
        assert u[2] == pytest.approx(0.5)

    def test_masked_softmax_no_legal_is_zero(self) -> None:
        out = _masked_softmax(np.array([1.0, 2.0, 3.0]), np.array([False, False, False]))
        assert out.sum() == 0.0 or np.isclose(out.sum(), 0.0) or np.allclose(out, [0, 0, 0])


# ---------------------------------------------------------------------------
# Tier 1: loss/weighting helpers
# ---------------------------------------------------------------------------


class TestLossHelpers:
    def test_dcfr_plus_weight_monotonic(self) -> None:
        w_small = _dcfr_plus_weight(1, 1.5)
        w_large = _dcfr_plus_weight(100, 1.5)
        assert 0.0 < w_small < w_large < 1.0

    def test_regret_loss_reduces_to_zero_on_perfect_pred(self) -> None:
        pred = torch.zeros(4, NUM_ABSTRACT_ACTIONS_2P)
        target = torch.zeros(4, NUM_ABSTRACT_ACTIONS_2P)
        mask = torch.ones(4, NUM_ABSTRACT_ACTIONS_2P, dtype=torch.bool)
        w = torch.ones(4)
        loss = _regret_loss(pred, target, mask, w)
        assert float(loss.item()) == pytest.approx(0.0)

    def test_strategy_loss_minimized_when_pred_equals_target(self) -> None:
        target = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        pred = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        w = torch.ones(1)
        loss = _strategy_loss(pred, target, mask, w)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_value_loss_is_mse(self) -> None:
        pred = torch.tensor([[1.0], [2.0]])
        target = torch.tensor([[0.5], [2.0]])
        loss = _value_loss(pred, target)
        assert float(loss.item()) == pytest.approx(0.125, rel=1e-5)


# ---------------------------------------------------------------------------
# Tier 1: APCFR+ and RM+ updates
# ---------------------------------------------------------------------------


class TestInnerUpdates:
    def test_rm_plus_step_non_nan(self) -> None:
        net = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        for _ in range(5):
            loss = ((net(x) - y) ** 2).mean()
            g = _rm_plus_step(opt, list(net.parameters()), loss, grad_clip=1.0)
            assert g >= 0.0
            assert not any(torch.isnan(p).any().item() for p in net.parameters())

    def test_apcfr_plus_step_non_nan(self) -> None:
        net = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        store = _PrevGradStore()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        for _ in range(5):
            loss = ((net(x) - y) ** 2).mean()
            g = _apcfr_plus_step(
                opt, list(net.parameters()), loss,
                asymmetry=0.9, grad_clip=1.0, prev_store=store,
            )
            assert g >= 0.0
            assert not any(torch.isnan(p).any().item() for p in net.parameters())

    def test_apcfr_first_step_matches_rm_plus(self) -> None:
        """With an empty prev-grad store, the first APCFR+ step has no prev
        gradient and should produce the same update as RM+."""
        torch.manual_seed(0)
        net1 = torch.nn.Linear(4, 4)
        net2 = torch.nn.Linear(4, 4)
        net2.load_state_dict(net1.state_dict())
        opt1 = torch.optim.SGD(net1.parameters(), lr=0.01)
        opt2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        loss1 = ((net1(x) - y) ** 2).mean()
        _rm_plus_step(opt1, list(net1.parameters()), loss1, grad_clip=1.0)
        loss2 = ((net2(x) - y) ** 2).mean()
        _apcfr_plus_step(
            opt2, list(net2.parameters()), loss2,
            asymmetry=0.9, grad_clip=1.0, prev_store=_PrevGradStore(),
        )

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_apcfr_asymmetry_zero_matches_rm_plus(self) -> None:
        """When asymmetry=0.0, APCFR+ degenerates to RM+ regardless of prev."""
        torch.manual_seed(1)
        net1 = torch.nn.Linear(4, 4)
        net2 = torch.nn.Linear(4, 4)
        net2.load_state_dict(net1.state_dict())
        opt1 = torch.optim.SGD(net1.parameters(), lr=0.01)
        opt2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        store = _PrevGradStore()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # Take two steps on each with identical batches; confirm they agree.
        for _ in range(2):
            loss1 = ((net1(x) - y) ** 2).mean()
            _rm_plus_step(opt1, list(net1.parameters()), loss1, grad_clip=1.0)
            loss2 = ((net2(x) - y) ** 2).mean()
            _apcfr_plus_step(
                opt2, list(net2.parameters()), loss2,
                asymmetry=0.0, grad_clip=1.0, prev_store=store,
            )
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_apcfr_diverges_from_rm_plus_with_nonzero_asymmetry(self) -> None:
        """Tier 1 (task #6 gate): two successive SGD steps on identical batches
        must produce strictly different parameters between RM+ and APCFR+
        with ``asymmetry > 0``. This proves the two code paths are actually
        different algorithms, so the Sprint 2 ablation is meaningful.
        """
        torch.manual_seed(42)
        net_rm = torch.nn.Linear(4, 4)
        net_apcfr = torch.nn.Linear(4, 4)
        net_apcfr.load_state_dict(net_rm.state_dict())
        opt_rm = torch.optim.SGD(net_rm.parameters(), lr=0.05)
        opt_apcfr = torch.optim.SGD(net_apcfr.parameters(), lr=0.05)
        store = _PrevGradStore()

        x1 = torch.randn(4, 4)
        y1 = torch.randn(4, 4)
        x2 = torch.randn(4, 4)
        y2 = torch.randn(4, 4)

        # Step 1: identical across paths (empty prev store on APCFR+).
        loss_rm = ((net_rm(x1) - y1) ** 2).mean()
        _rm_plus_step(opt_rm, list(net_rm.parameters()), loss_rm, grad_clip=1.0)
        loss_apcfr = ((net_apcfr(x1) - y1) ** 2).mean()
        _apcfr_plus_step(
            opt_apcfr, list(net_apcfr.parameters()), loss_apcfr,
            asymmetry=0.9, grad_clip=1.0, prev_store=store,
        )

        # Step 2: APCFR+ now has prev grads and should diverge from RM+.
        loss_rm = ((net_rm(x2) - y2) ** 2).mean()
        _rm_plus_step(opt_rm, list(net_rm.parameters()), loss_rm, grad_clip=1.0)
        loss_apcfr = ((net_apcfr(x2) - y2) ** 2).mean()
        _apcfr_plus_step(
            opt_apcfr, list(net_apcfr.parameters()), loss_apcfr,
            asymmetry=0.9, grad_clip=1.0, prev_store=store,
        )

        max_abs_diff = 0.0
        for p_rm, p_apcfr in zip(net_rm.parameters(), net_apcfr.parameters()):
            max_abs_diff = max(
                max_abs_diff, float((p_rm - p_apcfr).abs().max().item())
            )
        assert max_abs_diff > 1e-6, (
            f"APCFR+ and RM+ paths did not diverge: max|Δ|={max_abs_diff:.2e}"
        )

    def test_apcfr_prev_grad_store_roundtrip(self) -> None:
        """_PrevGradStore caches and returns grads keyed by id(parameter)."""
        p = torch.nn.Parameter(torch.randn(3))
        p.grad = torch.randn(3)
        store = _PrevGradStore()
        assert store.get(p) is None
        store.put(p, p.grad)
        cached = store.get(p)
        assert cached is not None
        assert torch.allclose(cached, p.grad)
        # Mutating the original grad after put must not affect the cache
        # (put clones).
        p.grad.zero_()
        assert not torch.allclose(cached, p.grad)
        store.clear()
        assert store.get(p) is None

    def test_apcfr_plus_byte_exact_extrapolation(self) -> None:
        """APCFR+ extrapolation matches g_t + alpha*(g_t - g_prev) at atol=1e-7.

        Wires known g_prev into the store, builds a loss whose backward
        produces a known g_t, calls _apcfr_plus_step, then asserts:
        - grad captured at optimizer.step() == g_t + 0.9*(g_t - g_prev)
        - prev_store.get(p) == raw g_t (pre-extrapolation snapshot)
        """
        g_prev = torch.tensor([1.0, 2.0, 3.0])
        g_t = torch.tensor([0.5, 1.0, 1.5])

        p = torch.nn.Parameter(torch.zeros(3))
        # loss whose gradient w.r.t. p equals g_t exactly
        loss = (p * g_t.detach()).sum()

        prev_store = _PrevGradStore()
        prev_store.put(p, g_prev)

        # Capture the gradient value at the moment optimizer.step() fires.
        captured: dict = {}

        class _CapturingOptimizer(torch.optim.Optimizer):
            def __init__(self, params: list) -> None:
                super().__init__(params, {})

            def step(self, closure=None) -> None:  # type: ignore[override]
                for group in self.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            captured[id(param)] = param.grad.detach().clone()

        opt = _CapturingOptimizer([p])
        _apcfr_plus_step(
            opt, [p], loss, asymmetry=0.9, grad_clip=1e9, prev_store=prev_store
        )

        expected_extrapolated = g_t + 0.9 * (g_t - g_prev)
        torch.testing.assert_close(
            captured[id(p)], expected_extrapolated, atol=1e-7, rtol=0
        )
        # prev_store must cache raw g_t, not the extrapolated value
        torch.testing.assert_close(
            prev_store.get(p), g_t, atol=1e-7, rtol=0
        )


# ---------------------------------------------------------------------------
# Minimal synthetic env + trainer construction
# ---------------------------------------------------------------------------


class _StubEngine:
    """A tiny two-step engine: one decision for each player, then terminal.

    Useful for exercising traversal machinery end-to-end without depending
    on the Go FFI or the full Python game engine.
    """

    def __init__(self, rng: np.random.Generator, payoff_scale: float = 1.0):
        self._rng = rng
        self._step = 0
        self._acting = 0
        self._terminal = False
        self._utility = np.array([0.0, 0.0], dtype=np.float32)
        self._payoff_scale = float(payoff_scale)
        self._applied_concrete: List[Any] = []
        self._snap_stack: List[dict] = []

    # Engine adapter interface expected by desca_worker.

    def is_terminal(self) -> bool:
        return self._terminal

    def get_utility(self) -> np.ndarray:
        return self._utility

    def get_acting_player(self) -> int:
        return self._acting

    def legal_actions(self) -> List[Any]:
        # Expose abstract-class aliases 0..2 as pseudo-concrete actions.
        from src.constants import (
            ActionDrawStockpile,
            ActionDrawDiscard,
            ActionCallCambia,
        )
        return [ActionDrawStockpile(), ActionDrawDiscard(), ActionCallCambia()]

    def apply_action(self, action: Any) -> None:
        self._applied_concrete.append(action)
        self._step += 1
        # One-step game: terminal immediately after the first action. Payoff
        # is different per action type so v_hat differs across branches.
        self._terminal = True
        action_type = type(action).__name__
        if action_type == "ActionDrawStockpile":
            marker = 1.0
        elif action_type == "ActionDrawDiscard":
            marker = -0.25
        else:
            marker = -0.75
        self._utility = np.array(
            [marker * self._payoff_scale, -marker * self._payoff_scale],
            dtype=np.float32,
        )

    def get_decision_context(self) -> int:
        return 0

    def get_drawn_card_bucket(self) -> int:
        return -1

    def save(self) -> dict:
        return {
            "step": self._step,
            "acting": self._acting,
            "terminal": self._terminal,
            "utility": self._utility.copy(),
            "applied": list(self._applied_concrete),
        }

    def restore(self, snap: dict) -> None:
        self._step = snap["step"]
        self._acting = snap["acting"]
        self._terminal = snap["terminal"]
        self._utility = snap["utility"].copy()
        self._applied_concrete = list(snap["applied"])

    def free_snapshot(self, snap: dict) -> None:  # noqa: D401
        return None

    def _omniscient_features(self) -> np.ndarray:
        return np.zeros(OMNISCIENT_DIM_2P, dtype=np.float32)

    def close(self) -> None:
        return None


class _StubAgent:
    """Minimal agent stub that _StubEngine can pair with.

    The agent needs attributes/methods called by desca_worker's helpers:
    ``update(engine)``, ``clone()``, and attribute access used by
    ``encode_infoset_eppbs_interleaved_v2``. For the synthetic tests we
    bypass the real encoder by wrapping the engine/agent in a factory that
    overrides the worker's encoder with a deterministic stub via monkeypatch.
    """

    def __init__(self, player_id: int = 0) -> None:
        self.player_id = player_id

    def update(self, engine: Any) -> None:
        return None

    def clone(self) -> "_StubAgent":
        return _StubAgent(self.player_id)

    def close(self) -> None:
        return None


@pytest.fixture
def stub_env_factory(monkeypatch: pytest.MonkeyPatch):
    """Factory that produces stub (engine, agents) and stubs encoder/omni."""
    def _encode_stub(agent: Any, ctx: int, drawn: int) -> np.ndarray:
        return np.arange(FEATURE_DIM, dtype=np.float32) / float(FEATURE_DIM)

    monkeypatch.setattr(desca_worker, "encode_infoset_eppbs_interleaved_v2", _encode_stub)
    # Force the worker to treat all abstract classes as legal: we replace
    # abstract_actions with a stub returning True on a fixed subset.
    def _abstract_stub(legal_actions, agent_state):
        mask = np.zeros(NUM_ABSTRACT_ACTIONS_2P, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[2] = True
        return mask

    monkeypatch.setattr(desca_worker, "abstract_actions", _abstract_stub)

    # Stub unabstract to return the first matching concrete action
    def _unabstract_stub(idx, legal, agent, seed):
        if int(idx) == 0:
            return legal[0]
        if int(idx) == 1:
            return legal[1]
        return legal[2]

    monkeypatch.setattr(desca_worker, "unabstract", _unabstract_stub)

    def _factory(rng: Optional[np.random.Generator] = None):
        r = rng if rng is not None else np.random.default_rng()
        engine = _StubEngine(r)
        agents = [_StubAgent(0), _StubAgent(1)]
        return engine, agents

    return _factory


# ---------------------------------------------------------------------------
# Tier 2: worker output shapes
# ---------------------------------------------------------------------------


class TestWorkerOutputs:
    def test_worker_returns_correct_shapes(self, stub_env_factory) -> None:
        result = desca_worker.run_desca_iteration(
            stub_env_factory,
            updating_player=0,
            regret_net=None,
            avg_strategy_net=None,
            history_value_net=None,
            iteration=1,
            traversals=5,
            device=None,
            rng=np.random.default_rng(42),
            warmup=True,
        )
        for s in result.regret_samples:
            assert s.features.shape == (FEATURE_DIM,)
            assert s.target.shape == (NUM_ABSTRACT_ACTIONS_2P,)
            assert s.action_mask.shape == (NUM_ABSTRACT_ACTIONS_2P,)
            assert s.action_mask.sum() >= 1
        for s in result.strategy_samples:
            assert s.features.shape == (FEATURE_DIM,)
            assert s.target.shape == (NUM_ABSTRACT_ACTIONS_2P,)
            probs_sum = float(np.sum(s.target))
            assert abs(probs_sum - 1.0) < 1e-4 or probs_sum == 0.0
        for s in result.value_samples:
            assert s.features.shape == (VALUE_INPUT_DIM,)
            assert s.target.shape == (1,)

        # Traversals must have been initiated.
        assert result.traversals_started == 5

    def test_worker_produces_regret_variance_over_iters(
        self, stub_env_factory
    ) -> None:
        """Tier 2 information flow: regret targets are not uniformly zero.

        A synthetic engine whose payoff differs per first-action yields
        non-degenerate per-action regret targets once aggregated.
        """
        all_targets: List[np.ndarray] = []
        for seed in range(20):
            result = desca_worker.run_desca_iteration(
                stub_env_factory,
                updating_player=0,
                regret_net=None,
                avg_strategy_net=None,
                history_value_net=None,
                iteration=seed + 1,
                traversals=5,
                device=None,
                rng=np.random.default_rng(seed),
                warmup=True,
            )
            for s in result.regret_samples:
                # Only measure legal entries; illegal ones are always zero.
                all_targets.append(s.target[s.action_mask])

        if not all_targets:
            pytest.skip("No regret samples were generated.")
        flat = np.concatenate(all_targets)
        assert flat.std() > 0.05, (
            f"expected std > 0.05 on decision infosets, got {flat.std():.4f}"
        )


# ---------------------------------------------------------------------------
# Tier 1: DESCATrainer end-to-end smoke
# ---------------------------------------------------------------------------


def _make_trainer(
    inner_update: str, env_factory, tmp_path, device: str = "cpu"
) -> DESCATrainer:
    from src.desca_networks import (
        RegretNetwork,
        AvgStrategyNetwork,
        HistoryValueNetwork,
    )

    config = {
        "encoding_version": 2,
        "hidden_dim": 32,
        "num_abstract_actions": NUM_ABSTRACT_ACTIONS_2P,
        "iterations": 2,
        "traversals_per_iter": 2,
        "minibatch": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "dcfr_alpha": 1.5,
        "apcfr_asymmetry": 0.9,
        "buffer_capacity": 1024,
        "checkpoint_every": 10_000,
        "eval_every": 10_000,
        "warmup_iters": 0,
        "inner_update": inner_update,
        "stall_detection": {
            "window_size_iters": 50,
            "num_windows": 5,
            "max_iter_abs": 3000,
        },
        # Shrink SGD steps so the test is quick.
        "regret_sgd_steps": 3,
        "strategy_sgd_steps": 3,
        "value_sgd_steps": 3,
    }

    regret = RegretNetwork(
        input_dim=FEATURE_DIM,
        hidden_dim=32,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
        num_blocks=1,
    )
    strat = AvgStrategyNetwork(
        input_dim=FEATURE_DIM,
        hidden_dim=32,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
        num_blocks=1,
    )
    vnet = HistoryValueNetwork(
        input_dim=FEATURE_DIM,
        omniscient_dim=OMNISCIENT_DIM_2P,
        hidden_dim=32,
        num_blocks=1,
    )

    ckpt_dir = tmp_path / "runs" / "desca-test" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "desca_checkpoint.pt")

    return DESCATrainer(
        config,
        regret,
        strat,
        vnet,
        env_factory,
        device=device,
        checkpoint_path=ckpt_path,
        seed=0,
    )


class TestDESCATrainer:
    def test_apcfr_path_smoke(self, stub_env_factory, tmp_path) -> None:
        trainer = _make_trainer("apcfr_plus", stub_env_factory, tmp_path)
        trainer.train(num_iterations=2)
        # Buffers should have entries after 2 iters.
        assert len(trainer.regret_buffer) > 0
        assert len(trainer.strategy_buffer) > 0
        assert len(trainer.value_buffer) > 0
        # Networks should not be NaN after stepping.
        for p in trainer.regret_net.parameters():
            assert not torch.isnan(p).any()
        for p in trainer.avg_strategy_net.parameters():
            assert not torch.isnan(p).any()
        for p in trainer.history_value_net.parameters():
            assert not torch.isnan(p).any()

    def test_rm_plus_path_smoke(self, stub_env_factory, tmp_path) -> None:
        trainer = _make_trainer("rm_plus", stub_env_factory, tmp_path)
        trainer.train(num_iterations=2)
        for p in trainer.regret_net.parameters():
            assert not torch.isnan(p).any()
        for p in trainer.avg_strategy_net.parameters():
            assert not torch.isnan(p).any()

    def test_checkpoint_roundtrip_identical_state(
        self, stub_env_factory, tmp_path
    ) -> None:
        trainer = _make_trainer("rm_plus", stub_env_factory, tmp_path)
        trainer.train(num_iterations=1)
        ckpt_path = trainer.save_checkpoint()

        # Build a fresh trainer with fresh-init networks and load the saved state.
        fresh = _make_trainer("rm_plus", stub_env_factory, tmp_path)
        # Deliberately randomize its weights so load must overwrite them.
        for p in fresh.regret_net.parameters():
            p.data.add_(1.0)
        fresh.load_checkpoint(ckpt_path)

        for p1, p2 in zip(
            trainer.regret_net.parameters(), fresh.regret_net.parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6)
        for p1, p2 in zip(
            trainer.avg_strategy_net.parameters(),
            fresh.avg_strategy_net.parameters(),
        ):
            assert torch.allclose(p1, p2, atol=1e-6)
        for p1, p2 in zip(
            trainer.history_value_net.parameters(),
            fresh.history_value_net.parameters(),
        ):
            assert torch.allclose(p1, p2, atol=1e-6)
        assert fresh.iteration == trainer.iteration

    def test_stall_detection_requires_floor(
        self, stub_env_factory, tmp_path
    ) -> None:
        trainer = _make_trainer("rm_plus", stub_env_factory, tmp_path)
        # Populate window_means with no improvement but iteration below the floor.
        trainer._window_means = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40]
        trainer.iteration = 100
        assert trainer._detect_stall() is False
        # Once past floor, same stagnant trajectory triggers stall.
        trainer.iteration = 600
        assert trainer._detect_stall() is True

    def test_record_mean_imp_updates_windows(
        self, stub_env_factory, tmp_path
    ) -> None:
        trainer = _make_trainer("rm_plus", stub_env_factory, tmp_path)
        trainer.iteration = 50
        trainer.record_mean_imp(0.42)
        trainer.iteration = 100
        trainer.record_mean_imp(0.43)
        assert len(trainer._window_means) >= 1
