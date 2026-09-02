"""
Engine failures inside a traversal produce no fabricated samples (cambia-722).

Every Go traversal used to skip a failed engine call and carry on with pre-zeroed
values: the sample still went to the reservoir, carrying a utility of 0 that sits
inside the legal [-1, +1] range and is therefore indistinguishable downstream from a
genuine draw. ESCHER additionally stored a 0.0 value-net target and a regret of
exactly 0 for any action whose counterfactual could not be evaluated.

The failure is injected through a stand-in engine that raises on a chosen
apply_action call, since the Go engine has no supported way to fail one action.
The tree is one traverser node with two terminal children, which is the smallest
shape that separates "this action failed" from "this node failed".
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest
import torch

from src.cfr.deep_worker import (
    _deep_traverse_go,
    _deep_traverse_os_go,
    _deep_traverse_os_go_nplayer,
    _escher_traverse_go,
)
from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS, NUM_PLAYERS
from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.utils import WorkerStats

# Root node, player 0, two actions, both leading straight to a terminal.
TERMINALS: Dict[int, Tuple[float, float]] = {0: (1.0, -1.0), 1: (-1.0, 1.0)}


class _FaultyEngine:
    """
    Stand-in for GoEngine that raises on chosen apply_action calls.

    ``fail_apply_calls`` holds 1-based indices into the sequence of apply_action
    calls this engine receives, which is deterministic regardless of which action a
    sampler happens to draw: a traversal's own action is applied first, and ESCHER's
    counterfactual sweep applies the remaining actions afterwards.
    """

    def __init__(self, fail_apply_calls: Optional[Set[int]] = None) -> None:
        self.fail_apply_calls = fail_apply_calls or set()
        self.terminal_utility: Optional[Tuple[float, float]] = None
        self.apply_calls = 0
        self._snaps: Dict[int, Optional[Tuple[float, float]]] = {}
        self._next_snap = 1
        self.live_snapshots = 0

    def is_terminal(self) -> bool:
        return self.terminal_utility is not None

    def get_utility(self) -> np.ndarray:
        return np.array(self.terminal_utility, dtype=np.float64)

    get_nplayer_utility = get_utility

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[0] = 1
        mask[1] = 1
        return mask

    def nplayer_legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(N_PLAYER_NUM_ACTIONS, dtype=np.int8)
        mask[0] = 1
        mask[1] = 1
        return mask

    def decision_ctx(self) -> int:
        return 0

    def acting_player(self) -> int:
        return 0

    def get_drawn_card_bucket(self) -> int:
        return -1

    def save(self) -> int:
        token = self._next_snap
        self._next_snap += 1
        self._snaps[token] = self.terminal_utility
        self.live_snapshots += 1
        return token

    def restore(self, token: int) -> None:
        self.terminal_utility = self._snaps[token]

    def free_snapshot(self, token: int) -> None:
        if token in self._snaps:
            del self._snaps[token]
            self.live_snapshots -= 1

    def apply_action(self, action: int) -> None:
        self.apply_calls += 1
        if self.apply_calls in self.fail_apply_calls:
            raise RuntimeError(
                f"injected engine failure on apply call {self.apply_calls}"
            )
        self.terminal_utility = TERMINALS[int(action)]

    apply_nplayer_action = apply_action

    def update_both(self, _a0, _a1) -> None:
        pass


class _StubAgent:
    def __init__(self) -> None:
        self.closed = False

    def encode(self, _ctx: int, drawn_bucket: int = -1) -> np.ndarray:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    def encode_nplayer(self, _ctx: int, drawn_bucket: int = -1) -> np.ndarray:
        return np.zeros(N_PLAYER_INPUT_DIM, dtype=np.float32)

    def update_nplayer(self, _engine) -> None:
        pass

    def clone(self) -> "_StubAgent":
        return _StubAgent()

    def close(self) -> None:
        self.closed = True


class _StubValueNet(torch.nn.Module):
    """Constant value head: enough for the counterfactual sweep to produce a number."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full((features.shape[0], 1), 0.25)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        system=SimpleNamespace(recursion_limit=50),
        deep_cfr=SimpleNamespace(
            traversal_depth_limit=0,
            encoding_mode="legacy",
            network_type="residual",
            encoding_layout="auto",
        ),
    )


def _common_kwargs(engine, agents, advantage, strategy, stats):
    return dict(
        engine=engine,
        agent_states=agents,
        updating_player=0,
        network=None,
        iteration=0,
        config=_config(),
        advantage_samples=advantage,
        strategy_samples=strategy,
        depth=0,
        worker_stats=stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
    )


def test_es_discards_the_node_when_one_enumerated_action_fails():
    """
    External sampling needs every action's value for the baseline v(I), so one
    failed action contaminates every regret at the node, not just its own entry.
    """
    engine = _FaultyEngine(fail_apply_calls={2})
    agents = [_StubAgent(), _StubAgent()]
    advantage: List = []
    strategy: List = []
    stats = WorkerStats()

    result = _deep_traverse_go(
        **_common_kwargs(engine, agents, advantage, strategy, stats)
    )

    assert result is None, "a node with an unusable action must report the failure"
    assert advantage == [], "no advantage sample may carry the fabricated zero"
    assert stats.error_count >= 1
    assert engine.live_snapshots == 0, "the failing path leaked an engine snapshot"


def test_es_still_emits_a_clean_node():
    """The guard fires on failure only: an error-free node behaves as before."""
    engine = _FaultyEngine()
    agents = [_StubAgent(), _StubAgent()]
    advantage: List = []
    strategy: List = []
    stats = WorkerStats()

    result = _deep_traverse_go(
        **_common_kwargs(engine, agents, advantage, strategy, stats)
    )

    assert result is not None and len(result) == NUM_PLAYERS
    assert len(advantage) == 1
    assert stats.error_count == 0
    # Uniform strategy over (+1, -1) and (-1, +1) gives v(I) = 0 and regrets +/-1.
    target = advantage[0].target
    assert np.isclose(target[0], 1.0) and np.isclose(target[1], -1.0)


def test_os_emits_nothing_when_the_sampled_action_fails():
    engine = _FaultyEngine(fail_apply_calls={1})
    agents = [_StubAgent(), _StubAgent()]
    advantage: List = []
    strategy: List = []
    stats = WorkerStats()

    utility, _tail = _deep_traverse_os_go(
        exploration_epsilon=0.6,
        **_common_kwargs(engine, agents, advantage, strategy, stats),
    )

    assert utility is None
    assert advantage == [], "the OS advantage target rests entirely on the sampled path"
    assert stats.error_count >= 1
    assert engine.live_snapshots == 0


def test_nplayer_os_emits_nothing_when_the_sampled_action_fails():
    engine = _FaultyEngine(fail_apply_calls={1})
    agents = [_StubAgent(), _StubAgent()]
    advantage: List = []
    strategy: List = []
    stats = WorkerStats()

    utility, _tail = _deep_traverse_os_go_nplayer(
        exploration_epsilon=0.6,
        num_players=2,
        **_common_kwargs(engine, agents, advantage, strategy, stats),
    )

    assert utility is None
    assert advantage == []
    assert stats.error_count >= 1
    assert engine.live_snapshots == 0


def _run_escher(engine, agents, regret, value, policy, stats, batch: bool):
    return _escher_traverse_go(
        engine=engine,
        agent_states=agents,
        updating_player=0,
        regret_net=None,
        value_net=_StubValueNet(),
        iteration=0,
        config=_config(),
        regret_samples=regret,
        value_samples=value,
        policy_samples=policy,
        depth=0,
        worker_stats=stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        value_net_device=torch.device("cpu"),
        batch_counterfactuals=batch,
    )


@pytest.mark.parametrize("batch", [True, False])
def test_escher_drops_the_value_target_and_masks_the_sampled_action(batch: bool):
    """
    A failed subtree used to become a value-net target of exactly 0.0 and a sampled
    regret of `0 - v_hat`. Now the value sample is not stored at all and the sampled
    action is masked out of the regret target, while the counterfactual action, which
    was evaluated successfully, survives.
    """
    engine = _FaultyEngine(fail_apply_calls={1})
    agents = [_StubAgent(), _StubAgent()]
    regret: List = []
    value: List = []
    policy: List = []
    stats = WorkerStats()

    result = _run_escher(engine, agents, regret, value, policy, stats, batch)

    assert result is None
    assert value == [], "no fabricated 0.0 value-net target may be stored"
    assert len(regret) == 1
    mask = regret[0].action_mask
    sampled = int(np.argmax(regret[0].target != 0.0)) if regret[0].target.any() else None
    # Exactly one of the two legal actions survives: the one the CF sweep evaluated.
    assert mask[:2].sum() == 1, f"expected one surviving action, mask={mask[:2]}"
    if sampled is not None:
        assert mask[sampled], "the surviving entry must be the one that was computed"
    assert stats.error_count >= 1
    assert engine.live_snapshots == 0


@pytest.mark.parametrize("batch", [True, False])
def test_escher_masks_an_action_whose_counterfactual_failed(batch: bool):
    """
    The sampled action succeeds, the counterfactual sweep fails on the other action.
    That action used to be written as `v_hat - v_hat`, a regret of exactly 0 with its
    mask bit still set; it is now masked out instead.
    """
    engine = _FaultyEngine(fail_apply_calls={2})
    agents = [_StubAgent(), _StubAgent()]
    regret: List = []
    value: List = []
    policy: List = []
    stats = WorkerStats()

    result = _run_escher(engine, agents, regret, value, policy, stats, batch)

    assert result is not None and len(result) == NUM_PLAYERS
    assert len(value) == 1, "the sampled subtree succeeded, so its value target is real"
    assert len(regret) == 1
    mask = regret[0].action_mask
    assert mask[:2].sum() == 1, f"expected the failed CF action masked, mask={mask[:2]}"
    # The surviving entry is the sampled action, whose regret is a real utility.
    surviving = int(np.flatnonzero(mask[:2])[0])
    assert regret[0].target[surviving] != 0.0


@pytest.mark.parametrize("batch", [True, False])
def test_escher_clean_node_keeps_both_actions(batch: bool):
    engine = _FaultyEngine()
    agents = [_StubAgent(), _StubAgent()]
    regret: List = []
    value: List = []
    policy: List = []
    stats = WorkerStats()

    result = _run_escher(engine, agents, regret, value, policy, stats, batch)

    assert result is not None
    assert len(value) == 1
    assert len(regret) == 1
    assert regret[0].action_mask[:2].sum() == 2
    assert stats.error_count == 0


def test_worker_stats_counts_the_injected_failure():
    """
    error_count is what the trainer's per-step ceiling reads, so a discarded sample
    has to leave a countable trace rather than only a log line.
    """
    engine = _FaultyEngine(fail_apply_calls={1})
    agents = [_StubAgent(), _StubAgent()]
    stats = WorkerStats()

    _deep_traverse_go(**_common_kwargs(engine, agents, [], [], stats))

    assert stats.error_count >= 1
