"""Brute-force check of the realization-equivalence claim (cambia-737).

``scripts/gate_vs_served_gap.py`` measures the served SD-CFR policy's
exploitability by scoring ONE reconstructed behavioral strategy, on the ground
that a per-episode snapshot mixture is realization equivalent to the own-reach-
weighted average of those snapshots. The sibling test file pins that at the
level of realization plans. This one pins the consequence that actually carries
the report: that the NashConv of the reconstruction equals the NashConv of the
sampled mixture, computed from the definition.

The definition side uses no reach weighting and no equivalence argument at all:

    on-policy value  = sum_{t,u} p_t p_u u(sigma_t, sigma_u)
    best-response    = max over player i's PURE strategies of
                       sum_t p_t u_i(pure_i, sigma_t)
    NashConv         = sum_i (BR_i - on-policy_i)

which is exactly what the wrapper's two seats do when each samples its own
snapshot per episode and plays it for the whole game. It is compared against
``tools.tiny_solver.exploitability`` on the reconstructed policy, the same call
the script's score stage makes.

The tree is a hand-built 2-player zero-sum game with a chance root and two own
decisions per player on some lines, using tiny_solver's own node classes.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np
import pytest

_CFR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CFR_ROOT not in sys.path:
    sys.path.insert(0, _CFR_ROOT)

from tools.tiny_solver import Chance, Decision, Terminal, exploitability  # noqa: E402

_SCRIPT = os.path.join(_CFR_ROOT, "scripts", "gate_vs_served_gap.py")
_spec = importlib.util.spec_from_file_location("gate_vs_served_gap_eq", _SCRIPT)
gvs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gvs)


class _Act:
    """Minimal stand-in for a GameAction: only repr-order and identity matter."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _leaf(u0):
    return Terminal((u0, -u0))


def _build_tree(rng):
    """Chance root -> P0 decision -> P1 decision -> P0 decision -> terminal.

    Two chance branches give P0 two distinct first infosets; P1 sees only P0's
    action, so its infoset merges the two chance branches (imperfect information)
    while both players keep perfect recall of their own actions.
    """
    a0, a1 = _Act("a0"), _Act("a1")
    b0, b1 = _Act("b0"), _Act("b1")
    c0, c1 = _Act("c0"), _Act("c1")

    root = Chance()
    for deal, weight in ((0, 0.4), (1, 0.6)):
        p0 = Decision(0, None, ("P0", "deal", deal), [a0, a1])
        for ai, _a in enumerate((a0, a1)):
            # P1's infoset does NOT see the deal: same key for both branches.
            p1 = Decision(1, None, ("P1", "saw", ai), [b0, b1])
            for bi, _b in enumerate((b0, b1)):
                # P0's second decision remembers its own first action and the
                # deal (perfect recall) plus the public b.
                p0b = Decision(0, None, ("P0", "deal", deal, ai, bi), [c0, c1])
                for _ci in range(2):
                    p0b.children.append(_leaf(float(rng.uniform(-1.0, 1.0))))
                p1.children.append(p0b)
            p0.children.append(p1)
        root.children.append(p0)
        root.weights.append(weight)
    return root


def _decision_nodes(root):
    out, stack = [], [root]
    while stack:
        nd = stack.pop()
        if nd.kind == "T":
            continue
        if nd.kind == "C":
            stack.extend(nd.children)
            continue
        out.append(nd)
        stack.extend(nd.children)
    return out


def _infosets(root):
    """(ordered pkeys, index map, action counts, owner) for the built tree."""
    order, index, counts, owner = [], {}, [], []
    for nd in _decision_nodes(root):
        if nd.pkey in index:
            continue
        index[nd.pkey] = len(order)
        order.append(nd.pkey)
        counts.append(len(nd.actions))
        owner.append(nd.player)
    return order, index, np.asarray(counts, dtype=np.int64), np.asarray(owner)


def _value(node, policies, who):
    """Expected utility to ``who`` with each player playing policies[player]."""
    if node.kind == "T":
        return node.util[who]
    if node.kind == "C":
        return sum(
            w * _value(c, policies, who) for c, w in zip(node.children, node.weights)
        )
    dist = policies[node.player][node.pkey]
    return sum(
        p * _value(child, policies, who)
        for p, child in zip(dist, node.children)
        if p > 0.0
    )


def _pure_strategies(order, owner, counts, player):
    """Every deterministic assignment of one action per infoset of ``player``."""
    mine = [i for i in range(len(order)) if owner[i] == player]
    for combo in itertools.product(*[range(int(counts[i])) for i in mine]):
        pol = {}
        for i, choice in zip(mine, combo):
            vec = np.zeros(int(counts[i]))
            vec[choice] = 1.0
            pol[order[i]] = vec
        yield pol


def _mixture_nashconv(order, owner, counts, root, snapshots, weights):
    """NashConv of the sampled mixture, straight from the definition."""
    p = np.asarray(weights, dtype=np.float64)
    p = p / p.sum()
    on_policy = [0.0, 0.0]
    for (pi_t, sig_t), (pi_u, sig_u) in itertools.product(zip(p, snapshots), repeat=2):
        for who in (0, 1):
            on_policy[who] += pi_t * pi_u * _value(root, {0: sig_t, 1: sig_u}, who)
    br = [0.0, 0.0]
    for player in (0, 1):
        best = -float("inf")
        for pure in _pure_strategies(order, owner, counts, player):
            val = 0.0
            for pi_t, sig_t in zip(p, snapshots):
                pols = {player: pure, 1 - player: sig_t}
                val += pi_t * _value(root, pols, player)
            best = max(best, val)
        br[player] = best
    return (br[0] - on_policy[0]) + (br[1] - on_policy[1])


def _random_policy(order, counts, rng):
    pol = {}
    for key, nA in zip(order, counts):
        vec = rng.random(int(nA)) + 0.05
        pol[key] = vec / vec.sum()
    return pol


def _flatten(pol, order, counts):
    return np.concatenate([pol[k] for k in order])


def _structure(root, order, index, counts):
    """Own-reach parent structure for the hand-built tree, script-side format."""
    n = len(order)
    parent_iset = np.full(n, -1, dtype=np.int64)
    parent_slot = np.full(n, -1, dtype=np.int64)
    legal_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=legal_off[1:])
    stack = [(root, (None, None))]
    while stack:
        node, last = stack.pop()
        if node.kind == "T":
            continue
        if node.kind == "C":
            for child in node.children:
                stack.append((child, last))
            continue
        i = index[node.pkey]
        cand = last[node.player]
        parent_iset[i] = -1 if cand is None else cand[0]
        parent_slot[i] = -1 if cand is None else cand[1]
        for k, child in enumerate(node.children):
            nxt = list(last)
            nxt[node.player] = (i, k)
            stack.append((child, (nxt[0], nxt[1])))
    depth = np.zeros(n, dtype=np.int64)
    for _ in range(n):
        for i in range(n):
            if parent_iset[i] >= 0:
                depth[i] = depth[parent_iset[i]] + 1
    order_idx = np.argsort(depth, kind="stable")
    depth_start = np.zeros(int(depth.max()) + 2, dtype=np.int64)
    np.cumsum(np.bincount(depth), out=depth_start[1:])
    return legal_off, parent_iset, parent_slot, order_idx, depth_start


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reconstructed_policy_has_the_mixtures_exact_nashconv(seed):
    rng = np.random.default_rng(seed)
    root = _build_tree(rng)
    order, index, counts, owner = _infosets(root)
    legal_off, parent_iset, parent_slot, order_idx, depth_start = _structure(
        root, order, index, counts
    )

    iters = [3, 11, 40]
    snapshots = [_random_policy(order, counts, rng) for _ in iters]

    acc_gate = np.zeros(int(legal_off[-1]))
    acc_served = np.zeros(int(legal_off[-1]))
    reach_weight = np.zeros(len(order))
    wsum = 0.0
    for it, snap in zip(iters, snapshots):
        wsum += gvs.accumulate_snapshot(
            acc_gate, acc_served, reach_weight, _flatten(snap, order, counts),
            float(it), legal_off, parent_iset, parent_slot, order_idx, depth_start,
            counts,
        )
    served_flat, unreached = gvs.finalize_served(acc_served, reach_weight, legal_off)
    assert not unreached.any()
    served = {
        key: served_flat[legal_off[i] : legal_off[i + 1]] for i, key in enumerate(order)
    }

    got, _components = exploitability(root, served)
    want = _mixture_nashconv(order, owner, counts, root, snapshots, iters)
    assert got == pytest.approx(want, abs=1e-10)


@pytest.mark.parametrize("seed", [7, 8])
def test_gate_average_does_not_have_the_mixtures_nashconv(seed):
    """The reach-unweighted gate object scores a genuinely different number."""
    rng = np.random.default_rng(seed)
    root = _build_tree(rng)
    order, index, counts, owner = _infosets(root)
    legal_off, parent_iset, parent_slot, order_idx, depth_start = _structure(
        root, order, index, counts
    )
    iters = [3, 11, 40]
    snapshots = [_random_policy(order, counts, rng) for _ in iters]

    acc_gate = np.zeros(int(legal_off[-1]))
    acc_served = np.zeros(int(legal_off[-1]))
    reach_weight = np.zeros(len(order))
    wsum = 0.0
    for it, snap in zip(iters, snapshots):
        wsum += gvs.accumulate_snapshot(
            acc_gate, acc_served, reach_weight, _flatten(snap, order, counts),
            float(it), legal_off, parent_iset, parent_slot, order_idx, depth_start,
            counts,
        )
    gate_flat = gvs.finalize_gate(acc_gate, wsum, legal_off)
    gate = {
        key: gate_flat[legal_off[i] : legal_off[i + 1]] for i, key in enumerate(order)
    }
    gate_nc, _c = exploitability(root, gate)
    mix_nc = _mixture_nashconv(order, owner, counts, root, snapshots, iters)
    assert abs(gate_nc - mix_nc) > 1e-6
