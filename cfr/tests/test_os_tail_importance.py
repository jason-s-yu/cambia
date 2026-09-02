"""
Unbiasedness of the outcome-sampling advantage target on a tiny tree (cambia-715).

The Go traversals sample the traverser from an epsilon-mixed policy
q = epsilon * uniform + (1 - epsilon) * sigma at every node of the sampled path,
so the utility that comes back from the recursion is the value of the epsilon-mixed
continuation. Correcting it needs the tail importance ratio pi^sigma(h, z) / q(h -> z)
(Lanctot et al. 2009); without it the root regret target converges to the wrong
vector, and on the tree below it converges to a vector with the opposite sign.

The tree is driven through a stand-in engine rather than the Go engine: Cambia has
no subtree whose exact counterfactual value is known in closed form, and the point
of the test is to compare a sampled mean against an exactly computed one.

Tree (traverser = player 0, actions 0 and 1 at every node):

    node 0  player 0   a0 -> node 1            a1 -> terminal (+0.2, -0.2)
    node 1  player 1   a0 -> node 2            a1 -> terminal (-0.5, +0.5)
    node 2  player 0   a0 -> terminal (+1,-1)  a1 -> terminal (-1,+1)
"""

from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest

from src.cfr import deep_worker
from src.cfr.deep_worker import (
    MAX_IS_WEIGHT,
    _deep_traverse_os_go,
    _deep_traverse_os_go_nplayer,
)
from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS
from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.utils import WorkerStats

EPSILON = 0.6

# node id -> (acting player, {action: ("node", child_id) | ("term", utility pair)})
TREE: Dict[int, Tuple[int, Dict[int, Tuple[str, object]]]] = {
    0: (0, {0: ("node", 1), 1: ("term", (0.2, -0.2))}),
    1: (1, {0: ("node", 2), 1: ("term", (-0.5, 0.5))}),
    2: (0, {0: ("term", (1.0, -1.0)), 1: ("term", (-1.0, 1.0))}),
}

# sigma at each node, over actions (0, 1). Node 2 is skewed hard so that sigma's
# continuation value and the epsilon-mixed one are far apart.
SIGMA: Dict[int, List[float]] = {0: [0.6, 0.4], 1: [0.7, 0.3], 2: [0.9, 0.1]}


def _sampling_policy() -> Dict[int, List[float]]:
    """q at each node: epsilon-mixed at traverser nodes, sigma elsewhere."""
    out: Dict[int, List[float]] = {}
    for node, (player, _edges) in TREE.items():
        sigma = SIGMA[node]
        if player == 0:
            out[node] = [EPSILON * 0.5 + (1.0 - EPSILON) * p for p in sigma]
        else:
            out[node] = list(sigma)
    return out


Q = _sampling_policy()


def _value_under(node: int, policy: Dict[int, List[float]]) -> np.ndarray:
    """Expected utility vector from `node` when every seat plays `policy`."""
    _player, edges = TREE[node]
    total = np.zeros(2, dtype=np.float64)
    for action, (kind, payload) in edges.items():
        prob = policy[node][action]
        if kind == "term":
            total += prob * np.array(payload, dtype=np.float64)
        else:
            total += prob * _value_under(int(payload), policy)
    return total


def _root_regrets(policy_below_root: Dict[int, List[float]]) -> np.ndarray:
    """
    Traverser regret vector at the root: r(a) = v(root . a) - sum_a sigma(a) v(root . a).

    `policy_below_root` is the policy the continuation values are taken under. With
    sigma it is the exact counterfactual regret the corrected estimator targets; with
    q it is what the uncorrected estimator converges to, since its recursion returns
    the value of the sampled (epsilon-mixed) continuation.
    """
    action_values = []
    for action in (0, 1):
        kind, payload = TREE[0][1][action]
        if kind == "term":
            action_values.append(float(payload[0]))
        else:
            action_values.append(float(_value_under(int(payload), policy_below_root)[0]))
    baseline = sum(SIGMA[0][a] * action_values[a] for a in (0, 1))
    return np.array([v - baseline for v in action_values], dtype=np.float64)


EXACT_REGRETS = _root_regrets(SIGMA)
UNCORRECTED_REGRETS = _root_regrets(Q)


class _TinyEngine:
    """Stand-in for GoEngine over TREE, covering both the 2P and N-player surfaces."""

    def __init__(self) -> None:
        self.node = 0
        self.utility = None
        self._snaps: Dict[int, Tuple[object, object]] = {}
        self._next_snap = 1
        self.live_snapshots = 0

    def is_terminal(self) -> bool:
        return self.node is None

    def get_utility(self) -> np.ndarray:
        return np.array(self.utility, dtype=np.float64)

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
        return TREE[self.node][0]

    def get_drawn_card_bucket(self) -> int:
        return -1

    def save(self) -> int:
        token = self._next_snap
        self._next_snap += 1
        self._snaps[token] = (self.node, self.utility)
        self.live_snapshots += 1
        return token

    def restore(self, token: int) -> None:
        self.node, self.utility = self._snaps[token]

    def free_snapshot(self, token: int) -> None:
        if self._snaps.pop(token, None) is not None:
            self.live_snapshots -= 1

    def apply_action(self, action: int) -> None:
        kind, payload = TREE[self.node][1][int(action)]
        if kind == "node":
            self.node, self.utility = int(payload), None
        else:
            self.node, self.utility = None, payload

    apply_nplayer_action = apply_action

    def update_both(self, _a0, _a1) -> None:
        pass


class _TinyAgent:
    """Stand-in for GoAgentState: features carry the current node id in slot 0."""

    def __init__(self, engine: _TinyEngine) -> None:
        self.engine = engine
        self.closed = False

    def _encode(self, dim: int) -> np.ndarray:
        features = np.zeros(dim, dtype=np.float32)
        features[0] = -1.0 if self.engine.node is None else float(self.engine.node)
        return features

    def encode(self, _ctx: int, drawn_bucket: int = -1) -> np.ndarray:
        return self._encode(INPUT_DIM)

    def encode_nplayer(self, _ctx: int, drawn_bucket: int = -1) -> np.ndarray:
        return self._encode(N_PLAYER_INPUT_DIM)

    def update_nplayer(self, _engine) -> None:
        pass

    def clone(self) -> "_TinyAgent":
        return _TinyAgent(self.engine)

    def close(self) -> None:
        self.closed = True


def _tiny_config() -> SimpleNamespace:
    config = SimpleNamespace()
    config.system = SimpleNamespace(recursion_limit=50)
    config.deep_cfr = SimpleNamespace(
        traversal_depth_limit=0,
        encoding_mode="legacy",
        network_type="residual",
        encoding_layout="auto",
    )
    return config


@pytest.fixture
def tiny_strategy(monkeypatch):
    """Route the network strategy lookup to SIGMA, keyed by the node id in slot 0."""

    def _strategy(_network, features, _action_mask, _feat_buf=None, _mask_buf=None):
        node = int(round(float(features[0])))
        width = NUM_ACTIONS if len(features) == INPUT_DIM else N_PLAYER_NUM_ACTIONS
        strategy = np.zeros(width, dtype=np.float64)
        strategy[0], strategy[1] = SIGMA[node]
        return strategy

    monkeypatch.setattr(deep_worker, "_get_strategy_from_network", _strategy)


@pytest.fixture
def weight_recorder(monkeypatch):
    """Record every raw importance weight so a test can prove the clip never bound."""
    seen: List[float] = []
    original = deep_worker._clipped_is_weight

    def _recording(tail_ratio: float, sampling_prob: float) -> float:
        seen.append(tail_ratio / sampling_prob)
        return original(tail_ratio, sampling_prob)

    monkeypatch.setattr(deep_worker, "_clipped_is_weight", _recording)
    return seen


def _run_traversals(num_traversals: int, nplayer: bool = False) -> np.ndarray:
    """
    Mean root regret target over `num_traversals` outcome-sampling traversals.

    The root sample is the one whose features carry node id 0.
    """
    width = N_PLAYER_NUM_ACTIONS if nplayer else NUM_ACTIONS
    accumulator = np.zeros(2, dtype=np.float64)
    root_samples = 0

    for _ in range(num_traversals):
        engine = _TinyEngine()
        agents = [_TinyAgent(engine), _TinyAgent(engine)]
        advantage_samples: List = []
        strategy_samples: List = []
        kwargs = dict(
            engine=engine,
            agent_states=agents,
            updating_player=0,
            network=object(),
            iteration=0,
            config=_tiny_config(),
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            depth=0,
            worker_stats=WorkerStats(),
            progress_queue=None,
            worker_id=0,
            min_depth_after_bottom_out_tracker=[float("inf")],
            has_bottomed_out_tracker=[False],
            simulation_nodes=[],
            exploration_epsilon=EPSILON,
        )
        if nplayer:
            _utility, _tail = _deep_traverse_os_go_nplayer(num_players=2, **kwargs)
        else:
            _utility, _tail = _deep_traverse_os_go(**kwargs)

        assert engine.live_snapshots == 0, "traversal leaked an engine snapshot"

        for sample in advantage_samples:
            if int(round(float(sample.features[0]))) == 0:
                assert len(sample.target) == width
                accumulator += sample.target[:2].astype(np.float64)
                root_samples += 1

    assert root_samples == num_traversals, (
        f"expected one root advantage sample per traversal, got {root_samples} "
        f"over {num_traversals} traversals"
    )
    return accumulator / num_traversals


def test_tree_constants_are_what_the_docstring_claims():
    """Guard the fixture: the two target vectors have to differ, and differ in sign."""
    assert np.allclose(EXACT_REGRETS, [0.084, -0.126])
    assert np.allclose(UNCORRECTED_REGRETS, [-0.0504, 0.0756])
    # The uncorrected estimator does not merely shrink the regrets, it reverses
    # which root action looks better.
    assert EXACT_REGRETS[0] > 0 > EXACT_REGRETS[1]
    assert UNCORRECTED_REGRETS[0] < 0 < UNCORRECTED_REGRETS[1]


def test_clip_never_binds_on_this_tree(tiny_strategy, weight_recorder):
    """
    The unbiasedness assertions below are only meaningful where MAX_IS_WEIGHT is
    inactive, since the clip is a deliberately biased truncation.
    """
    np.random.seed(4242)
    _run_traversals(500)
    assert weight_recorder, "no importance weight was computed"
    assert max(weight_recorder) < MAX_IS_WEIGHT, (
        f"max raw importance weight {max(weight_recorder):.3f} reached the "
        f"{MAX_IS_WEIGHT} clip; the tree no longer isolates the tail correction"
    )


def test_corrected_estimator_is_unbiased_at_the_root(tiny_strategy):
    """The corrected root regret target averages to the exact counterfactual regret."""
    np.random.seed(20250902)
    mean = _run_traversals(40000)

    assert np.allclose(mean, EXACT_REGRETS, atol=0.02), (
        f"corrected mean {mean} is not within 0.02 of the exact regrets "
        f"{EXACT_REGRETS}"
    )
    # And it is nowhere near what the uncorrected estimator targets.
    assert not np.allclose(mean, UNCORRECTED_REGRETS, atol=0.05)


def test_uncorrected_estimator_is_biased_at_the_root(tiny_strategy, monkeypatch):
    """
    Dropping the tail ratio reproduces the pre-fix estimator, which converges to the
    value of the epsilon-mixed continuation instead of sigma's.
    """

    def _local_weight_only(_tail_ratio: float, sampling_prob: float) -> float:
        return min(1.0 / sampling_prob, MAX_IS_WEIGHT)

    monkeypatch.setattr(deep_worker, "_clipped_is_weight", _local_weight_only)

    np.random.seed(20250902)
    mean = _run_traversals(40000)

    assert np.allclose(mean, UNCORRECTED_REGRETS, atol=0.02), (
        f"uncorrected mean {mean} did not converge to the epsilon-mixed target "
        f"{UNCORRECTED_REGRETS}"
    )
    assert not np.allclose(mean, EXACT_REGRETS, atol=0.05), (
        "uncorrected mean landed on the exact regrets; the test no longer "
        "separates the two estimators"
    )
    # The gap the fix closes, on this tree: the regret ordering of the two root
    # actions is inverted.
    assert mean[0] < 0 < mean[1]


def test_nplayer_variant_carries_the_same_correction(tiny_strategy):
    """The N-player traversal estimates the same exact regrets at 2 seats."""
    np.random.seed(31337)
    mean = _run_traversals(40000, nplayer=True)

    assert np.allclose(mean, EXACT_REGRETS, atol=0.02), (
        f"N-player corrected mean {mean} is not within 0.02 of the exact regrets "
        f"{EXACT_REGRETS}"
    )


def test_root_tail_ratio_matches_the_sampled_path(tiny_strategy):
    """
    The ratio a node hands its parent is sigma(a|h) / q(a|h) times the child's, and
    opponent nodes contribute exactly 1. The tree has four root-to-terminal paths, so
    the root's ratio has to be one of four values, and its mean over many traversals
    has to be 1 (the defining property of an importance ratio).
    """
    root0 = SIGMA[0][0] / Q[0][0]
    root1 = SIGMA[0][1] / Q[0][1]
    expected = {
        root1,  # root a1 -> terminal
        root0,  # root a0 -> opponent a1 -> terminal
        root0 * SIGMA[2][0] / Q[2][0],  # ... -> opponent a0 -> traverser a0
        root0 * SIGMA[2][1] / Q[2][1],  # ... -> opponent a0 -> traverser a1
    }

    np.random.seed(7)
    ratios = []
    for _ in range(20000):
        engine = _TinyEngine()
        agents = [_TinyAgent(engine), _TinyAgent(engine)]
        _utility, tail = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
            updating_player=0,
            network=object(),
            iteration=0,
            config=_tiny_config(),
            advantage_samples=[],
            strategy_samples=[],
            depth=0,
            worker_stats=WorkerStats(),
            progress_queue=None,
            worker_id=0,
            min_depth_after_bottom_out_tracker=[float("inf")],
            has_bottomed_out_tracker=[False],
            simulation_nodes=[],
            exploration_epsilon=EPSILON,
        )
        assert any(
            abs(tail - value) < 1e-12 for value in expected
        ), f"tail ratio {tail} matches none of the four path ratios {sorted(expected)}"
        ratios.append(tail)

    assert abs(float(np.mean(ratios)) - 1.0) < 0.02, (
        f"mean tail ratio {np.mean(ratios):.4f} is not 1; the ratio is not a "
        "proper importance weight"
    )
