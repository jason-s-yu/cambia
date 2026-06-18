"""Scoped tests for the PRT-CFR external-sampling worker and its MC estimator.

The core correctness check is MC-target UNBIASEDNESS: on the tiny tree the
m-rollout Monte-Carlo child-value mean must converge to the EXACT child values
(computed by enumeration under the same fixed policy) as m grows. This is the
property v0.4 design decision 2 rests on (E[q_hat] = q for any m >= 1; bias is
traded for zero-mean noise that regression averages out).
"""

import numpy as np
import pytest

from src.config import load_config
from src.cfr.prtcfr_worker import (
    PRTCFRWorker,
    exact_child_values,
    first_traverser_decision,
    uniform_policy,
    _child_value_mc,
)
from src.encoding import NUM_ACTIONS
from src.reservoir import ReservoirBuffer
from src.sequence_encoding import SEQ_CAP
from tools.tiny_solver import build_tree


CONFIG_2CARD = "config/tiny_2card_plateau.yaml"


@pytest.fixture(scope="module")
def tiny_tree():
    cfg = load_config(CONFIG_2CARD)
    root, isets, nnodes, aborted = build_tree(
        cfg, n_deals=5, seed0=0, max_nodes_per_deal=2_000_000,
        enumerate_draws=True, perfect_recall=True, tokenize=True, seq_cap=256,
    )
    assert aborted == 0
    return root


def test_mc_target_unbiasedness_converges_to_exact(tiny_tree):
    """THE core check. Pick a traverser decision node with >= 2 legal actions;
    under the uniform fixed policy, the averaged m-rollout MC child-value estimate
    must converge to the exact enumerated child values as m grows."""
    traverser = 0
    node = first_traverser_decision(tiny_tree, traverser)
    assert node is not None and len(node.actions) >= 2

    exact = exact_child_values(node, traverser, uniform_policy)

    # For each m, average many independent CRN blocks; the per-block estimate is
    # unbiased, so the average converges to the exact value and the error shrinks
    # roughly as 1/sqrt(m * reps).
    reps = 300
    errs = {}
    for m in (1, 8, 64):
        est = np.zeros(len(node.actions))
        for r in range(reps):
            seed_base = (r + 1) * 9973
            for i, child in enumerate(node.children):
                est[i] += _child_value_mc(child, traverser, uniform_policy, m, seed_base)
        est /= reps
        errs[m] = float(np.max(np.abs(est - exact)))

    # Convergence: larger m -> smaller error, and m=64 estimate is tight.
    assert errs[64] < errs[1], f"MC error did not shrink with m: {errs}"
    assert errs[64] < 0.02, f"m=64 estimate not within 0.02 of exact: {errs}"


def test_mc_estimate_within_one_action_is_constant(tiny_tree):
    """A traverser node where some child leads immediately to a terminal: the MC
    estimate of that child equals the exact terminal value with zero variance
    (no stochastic descent), so any m and any seed give the exact value."""
    traverser = 0
    # Find a traverser decision whose first child is a Terminal.
    def find(node):
        if node.kind == "T":
            return None
        if node.kind == "C":
            for c in node.children:
                r = find(c)
                if r is not None:
                    return r
            return None
        if node.player == traverser:
            for ch in node.children:
                if ch.kind == "T":
                    return (node, ch)
        for c in node.children:
            r = find(c)
            if r is not None:
                return r
        return None

    found = find(tiny_tree)
    if found is None:
        pytest.skip("no traverser node with an immediate terminal child in this tree")
    _node, term_child = found
    v = _child_value_mc(term_child, traverser, uniform_policy, m=3, seed_base=123)
    assert v == pytest.approx(term_child.util[traverser], abs=1e-9)


def test_worker_traversal_appends_valid_regret_samples(tiny_tree):
    """One external-sampling traversal appends well-formed regret samples: token
    features (seq_cap int width), 146-dim target, 146-dim mask with the legal-set
    True, and the recorded iteration."""
    buf = ReservoirBuffer(capacity=10000, input_dim=SEQ_CAP, target_dim=NUM_ACTIONS, has_mask=True)
    worker = PRTCFRWorker(tiny_tree, uniform_policy, m_rollouts=2, seq_cap=SEQ_CAP, seed=7)
    added = worker.traverse(traverser=0, iteration=5, buf=buf)
    assert added >= 1
    assert len(buf) == added
    sample = buf.buffer[0]
    assert sample.features.shape == (SEQ_CAP,)
    assert sample.target.shape == (NUM_ACTIONS,)
    assert sample.action_mask.shape == (NUM_ACTIONS,)
    assert sample.iteration == 5
    # at least one legal action flagged, and target is zero on illegal slots
    assert sample.action_mask.sum() >= 1
    assert np.all(sample.target[~sample.action_mask] == 0.0)


def test_regret_baseline_zero_sum_under_policy(tiny_tree):
    """At a traverser node, sum_a sigma(a) * r_hat(a) = sum_a sigma(a)(q(a)-b) = 0
    by construction (b is the sigma-weighted mean). Verify with EXACT child values
    so the property is exact, not noisy."""
    traverser = 0
    node = first_traverser_decision(tiny_tree, traverser)
    sigma = uniform_policy(node)
    q = exact_child_values(node, traverser, uniform_policy)
    baseline = float(np.dot(sigma, q))
    regret = q - baseline
    assert float(np.dot(sigma, regret)) == pytest.approx(0.0, abs=1e-9)
