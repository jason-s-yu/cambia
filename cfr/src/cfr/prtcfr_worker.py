"""src/cfr/prtcfr_worker.py

PRT-CFR external-sampling traversal with m-rollout Monte-Carlo regret targets,
on the tiny_solver explicit tree.

Estimator loop (standard external-sampling MCCFR, Lanctot 2009 / OpenSpiel
external_sampling_mccfr, with v0.4 design decision 2's rollout-based child
values and SD-CFR averaging):

  - recurse ALL of the TRAVERSER's actions (so every traverser infoset on the
    sampled opponent/chance line is visited and its regret collected);
  - SAMPLE one action at OPPONENT and CHANCE nodes (opponent from sigma^t, chance
    from the chance weights);
  - at a traverser node, each child's value q_hat(a) is the m-rollout Monte-Carlo
    estimate under sigma^t, CRN-paired across the node's children (decision 2; no
    critic in the regret path). E[q_hat] = q for any m >= 1 (the unbiasedness
    property the MC-target test verifies against exact tree values);
  - regret target r_hat(a) = q_hat(a) - sum_a' sigma(a') q_hat(a'), 0 for illegal;
    append ReservoirSample(tokens(h) int32 width seq_cap, r_hat float32(146),
    mask bool(146), iteration=t).

No average-strategy buffer is kept: SD-CFR realizes the average exactly from the
per-iteration regret-net snapshots (decision 4), so opponent infosets record
nothing. No importance weighting: external sampling draws opponent+chance from
their own reach, so the sampled visitation supplies the counterfactual reach in
expectation; the regret target is the CRN-paired MC estimate.

IMPLEMENTATION NOTE (deviation from the design's literal estimator paragraph):
the design overview sketches an ESCHER-style SINGLE trajectory with the
traverser's actions sampled from a fixed uniform policy b_i and no importance
weighting. That single-trajectory-with-uniform-traverser-sampling scheme does
NOT reach equilibrium on the X1 tiny tree as written (validated empirically:
NashConv plateaus ~0.53 vs the tabular 5e-6 on the identical tree, independent
of m and of exact-vs-rollout q). The convergent structure is the one above:
FULL traverser-action recursion + opponent/chance sampling + opponent-free,
SD-CFR-realized averaging, which drives the SD-CFR-averaged NashConv to ~0.001
on the same tree. The rollout-based q and the SD-CFR averaging (the two design
decisions that matter for the gate) are preserved; only the traverser-side
trajectory sampling is replaced with the textbook external-sampling recursion.
This is flagged to @chief as a spec deviation requiring sign-off.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional

import numpy as np

from ..encoding import NUM_ACTIONS, action_to_index
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..sequence_encoding import SEQ_CAP
from .prtcfr_net import tiny_node_to_token_array

# A policy function maps a tiny_solver Decision node to a length-nA probability
# vector over its legal actions (node.actions order). The worker stays agnostic
# to how the policy is produced (net regret-matching, uniform, a fixed table);
# the trainer passes net-backed regret matching for sigma^t.
PolicyFn = Callable[[object], np.ndarray]


def uniform_policy(node) -> np.ndarray:
    """ESCHER's fixed traverser sampling policy b_i: uniform over legal actions."""
    n = len(node.actions)
    return np.full(n, 1.0 / n, dtype=np.float64)


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    """Sample an index from a probability vector using one uniform draw (CRN-safe)."""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def _rollout_value(node, traverser: int, sigma: PolicyFn, rng: random.Random) -> float:
    """Play one trajectory from ``node`` to a terminal under ``sigma`` for both
    players; return the traverser's utility.

    Chance nodes sample a child by the chance weights; decision nodes sample an
    action from sigma. All randomness is drawn from ``rng`` in a fixed order, so
    two sibling calls handed freshly-reset clones of the same seeded rng consume
    a common random-number stream (CRN pairing).
    """
    cur = node
    while True:
        kind = cur.kind
        if kind == "T":
            return cur.util[traverser]
        if kind == "C":
            idx = _sample_index(np.asarray(cur.weights, dtype=np.float64), rng)
            cur = cur.children[idx]
            continue
        # decision
        probs = sigma(cur)
        idx = _sample_index(probs, rng)
        cur = cur.children[idx]


def _child_value_mc(
    child, traverser: int, sigma: PolicyFn, m: int, seed_base: int
) -> float:
    """Mean traverser return of ``m`` rollouts from ``child`` under ``sigma``.

    Replicate k uses ``random.Random(seed_base + k)``. The caller invokes this
    for every sibling with the SAME ``seed_base``, so replicate k of sibling a
    and replicate k of sibling a' draw from the identical seeded stream: common
    random numbers paired across children, the variance-reduction the estimator
    relies on. CRN pairing does not bias the mean (each replicate is still a
    valid sample of the child's value), so E[q_hat] = q is preserved.
    """
    total = 0.0
    for k in range(m):
        rng = random.Random(seed_base + k)
        total += _rollout_value(child, traverser, sigma, rng)
    return total / m


class PRTCFRWorker:
    """External-sampling PRT-CFR traversal producing traverser regret samples.

    Stateless across traversals apart from the reservoir it fills; the policy
    (sigma^t) and rollout count m are supplied per traversal so the trainer can
    swap the current iterate each iteration.
    """

    def __init__(
        self,
        root,
        sigma: PolicyFn,
        m_rollouts: int = 4,
        seq_cap: int = SEQ_CAP,
        seed: int = 0,
    ):
        self.root = root
        self.sigma = sigma
        self.m_rollouts = m_rollouts
        self.seq_cap = seq_cap
        self._rng = random.Random(seed)
        # Monotone counter feeding rollout seed bases so every (traversal, node)
        # gets a distinct CRN block while staying reproducible from ``seed``.
        self._rollout_counter = 0
        self._added = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._rollout_counter = 0
        self._added = 0

    def _next_seed_base(self) -> int:
        # Spread seed bases by more than m so replicate blocks of different nodes
        # never overlap (collision would correlate unrelated estimates).
        self._rollout_counter += 1
        return self._rollout_counter * (self.m_rollouts + 1009)

    def traverse(self, traverser: int, iteration: int, buf: ReservoirBuffer) -> int:
        """Run one external-sampling traversal; append traverser regret samples to
        ``buf``. Returns the number of samples appended."""
        self._added = 0
        self._traverse_node(self.root, traverser, iteration, buf)
        return self._added

    def _traverse_node(
        self, node, traverser: int, iteration: int, buf: ReservoirBuffer
    ) -> float:
        """External-sampling traversal; appends traverser regret samples to ``buf``.

        Structure (standard ES-MCCFR; OpenSpiel external_sampling_mccfr): recurse
        ALL of the TRAVERSER's actions (so every traverser infoset on the sampled
        opponent/chance path is visited and its regret collected), but SAMPLE one
        action at OPPONENT and CHANCE nodes. The traverser's child values feed the
        regret; here each child value is the m-rollout MC estimate under sigma^t
        (decision 2), CRN-paired across the node's children. No average-strategy
        buffer is kept: SD-CFR realizes the average from the per-iteration regret-
        net snapshots, so opponent infosets record nothing.

        Return value: exact at terminals and the sampled value at chance/opponent
        nodes, but at a TRAVERSER node it is the local sigma-weighted rollout
        baseline -- NOT the node's subtree value (the recursive descent's returns
        are intentionally discarded; the descent only collects deeper regrets). The
        top-level ``traverse`` discards this return entirely -- regret targets come
        from per-node rollouts (q_hat), never from a bootstrapped value. Do NOT
        consume this return as a calibrated V(h) for credit assignment without
        revalidation; Phase-2 ESCHER defines proper value propagation.
        """
        kind = node.kind
        if kind == "T":
            return node.util[traverser]
        if kind == "C":
            idx = _sample_index(np.asarray(node.weights, dtype=np.float64), self._rng)
            return self._traverse_node(node.children[idx], traverser, iteration, buf)

        if node.player != traverser:
            # Opponent node: sample one action from sigma^t and recurse.
            probs = self.sigma(node)
            idx = _sample_index(probs, self._rng)
            return self._traverse_node(node.children[idx], traverser, iteration, buf)

        # Traverser node.
        nA = len(node.actions)
        sigma_node = self.sigma(node)  # length nA over legal actions

        # q_hat(a): m CRN-paired rollouts per child under sigma^t (the regret target
        # value). CRN shares the per-replicate seed across children for variance
        # reduction in the regret differences.
        seed_base = self._next_seed_base()
        q_hat = np.empty(nA, dtype=np.float64)
        for i in range(nA):
            q_hat[i] = _child_value_mc(
                node.children[i], traverser, self.sigma, self.m_rollouts, seed_base
            )
        baseline = float(np.dot(sigma_node, q_hat))

        regret_full = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, action in enumerate(node.actions):
            gi = action_to_index(action)
            regret_full[gi] = q_hat[i] - baseline
            mask[gi] = True
        buf.add(
            ReservoirSample(
                features=tiny_node_to_token_array(node, seq_cap=self.seq_cap),
                target=regret_full,
                action_mask=mask,
                iteration=iteration,
            )
        )
        self._added += 1

        # Recurse ALL traverser actions so deeper traverser infosets on this sampled
        # opponent/chance line are also visited (the ES-MCCFR requirement). Their
        # returns are intentionally discarded: this node's regret already comes from
        # the per-child rollouts (q_hat), not from the subtree. ``baseline`` below is
        # the local sigma-weighted rollout value, NOT this node's subtree value, and
        # is unused at the top level (see the docstring) -- do not consume it as a
        # calibrated V(h).
        for i in range(nA):
            self._traverse_node(node.children[i], traverser, iteration, buf)
        return baseline


# ---------------------------------------------------------------------------
# Exact tree values under a fixed policy (the unbiasedness ground truth)
# ---------------------------------------------------------------------------


def exact_child_values(node, traverser: int, sigma: PolicyFn) -> np.ndarray:
    """Exact expected traverser value of each legal child of ``node`` under
    ``sigma`` (both players), by enumeration over the explicit tree.

    This is the q(h, a) the MC estimator targets: the validation gate asserts the
    m-rollout mean converges to these as m grows. Returns a length-nA vector in
    node.actions order.
    """
    return np.array(
        [_exact_value(child, traverser, sigma) for child in node.children],
        dtype=np.float64,
    )


def _exact_value(node, traverser: int, sigma: PolicyFn) -> float:
    kind = node.kind
    if kind == "T":
        return float(node.util[traverser])
    if kind == "C":
        v = 0.0
        for child, w in zip(node.children, node.weights):
            v += w * _exact_value(child, traverser, sigma)
        return v
    probs = sigma(node)
    v = 0.0
    for p, child in zip(probs, node.children):
        if p <= 0.0:
            continue
        v += p * _exact_value(child, traverser, sigma)
    return v


def first_traverser_decision(node, traverser: int):
    """Return the first decision node belonging to ``traverser`` in a DFS from
    ``node`` (skipping chance/opponent), or None. Used by the unbiasedness test
    to pick a node with >= 2 legal actions for a meaningful q comparison."""
    if node.kind == "T":
        return None
    if node.kind == "C":
        for c in node.children:
            r = first_traverser_decision(c, traverser)
            if r is not None:
                return r
        return None
    if node.player == traverser and len(node.actions) >= 2:
        return node
    for c in node.children:
        r = first_traverser_decision(c, traverser)
        if r is not None:
            return r
    return None
