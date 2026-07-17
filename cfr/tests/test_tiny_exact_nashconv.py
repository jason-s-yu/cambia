"""Exact-rational NashConv certifier tests (cambia-530, amendment A2).

Certifies tools/tiny_exact.py: exact-rational recomputation of the X2 gate's
NashConv/exploitability on the tiny {A,6} tree, used to rule X2R5 replication
cells against bar_respec = 0.057 (cambia-521).

Three layers:
  1. Hand-checkable fixtures -- tiny explicit trees whose exact NashConv and its
     (br0, br1, onp0, onp1) breakdown are computed by hand and asserted as exact
     Fractions, including a non-dyadic chance node (mass 1/3, 2/3) that only the
     exact path represents without rounding.
  2. Differential bound -- |float64 - exact| on the fixtures and on the real
     uniform policy, asserted comfortably below the gate decision scale.
  3. Determinism -- the exact path returns bit-identical Fractions on repeat.

Real-tree tests build the 230k-node {A,6} tree once (session fixture) and
recompute U (the uniform-policy game-value anchor: bar_respec = 0.0340 * U) and,
if an X2 snapshot is readable in the main repo, one real neural snapshot.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tools.tiny_solver import Terminal, Chance, Decision, exploitability
from tools import tiny_exact
from fractions import Fraction

# ---------------------------------------------------------------------------
# Fixture-tree constructors (lightweight explicit nodes; no engine needed).
# ---------------------------------------------------------------------------


def mk_terminal(u0, u1):
    return Terminal((float(u0), float(u1)))


def mk_chance(children, fracs):
    """Chance node with exact wfrac (list of Fraction) and parallel float weights."""
    ch = Chance()
    ch.children = list(children)
    ch.wfrac = [Fraction(f) for f in fracs]
    ch.weights = [float(f) for f in ch.wfrac]
    return ch


def mk_decision(player, pkey, children):
    """Decision node; actions is a dummy length-nA list (only len is read)."""
    nA = len(children)
    nd = Decision(player, pkey, pkey, list(range(nA)))
    nd.children = list(children)
    return nd


def _pol(**kw):
    """Policy dict keyed by bare pkey -> float64 vector."""
    return {k: np.asarray(v, dtype=np.float64) for k, v in kw.items()}


# ---------------------------------------------------------------------------
# Layer 1: hand-checkable fixtures.
# ---------------------------------------------------------------------------


def _wrap_root(decision_or_node):
    """Fixtures need a chance root (require_exact_tree). Wrap a single subtree as
    a degenerate 1-child chance root with mass 1 (identity for values)."""
    return mk_chance([decision_or_node], [Fraction(1)])


def test_fixture1_single_decision():
    """P0 picks a/b -> terminals (1,-1)/(-1,1); policy [3/4,1/4]. P1 never acts.

    onp0 = 3/4*1 + 1/4*(-1) = 1/2 ; br0 = max(1,-1) = 1 ; P1 side contributes 0.
    NashConv = (1 - 1/2) = 1/2.
    """
    root = _wrap_root(mk_decision(0, "I0", [mk_terminal(1, -1), mk_terminal(-1, 1)]))
    policy = _pol(I0=[0.75, 0.25])
    nc, (br0, br1, onp0, onp1) = tiny_exact.exploitability_exact(root, policy)
    assert onp0 == Fraction(1, 2)
    assert onp1 == Fraction(-1, 2)
    assert br0 == Fraction(1, 1)
    assert br1 == onp1
    assert nc == Fraction(1, 2)


def test_fixture2_nondyadic_chance_shared_infoset():
    """Chance root mass (1/3, 2/3) over two branches sharing P0 infoset I.

    Branch A: a->(1,-1) b->(-1,1) ; Branch B: a->(-1,1) b->(1,-1). Policy I=[3/4,1/4].
    onp0 = 1/3*(1/2) + 2/3*(-1/2) = -1/6.
    BR commits ONE action for I: cf value a = 1/3*1 + 2/3*(-1) = -1/3 ;
                                 b = 1/3*(-1) + 2/3*1 = +1/3 -> picks b.
    br0 = 1/3*(-1) + 2/3*(1) = 1/3.  NashConv = 1/3 - (-1/6) = 1/2.
    The 1/3 chance mass forces a denominator divisible by 3: only the exact path
    (wfrac) represents it without rounding.
    """
    branchA = mk_decision(0, "I", [mk_terminal(1, -1), mk_terminal(-1, 1)])
    branchB = mk_decision(0, "I", [mk_terminal(-1, 1), mk_terminal(1, -1)])
    root = mk_chance([branchA, branchB], [Fraction(1, 3), Fraction(2, 3)])
    policy = _pol(I=[0.75, 0.25])
    nc, (br0, br1, onp0, onp1) = tiny_exact.exploitability_exact(root, policy)
    assert onp0 == Fraction(-1, 6)
    assert br0 == Fraction(1, 3)
    assert onp1 == Fraction(1, 6)  # zero-sum
    assert br1 == onp1  # P1 has no decision
    assert nc == Fraction(1, 2)
    # non-dyadic chance signature: denominator carries the factor 3.
    assert onp0.denominator % 3 == 0


def test_fixture3_both_players_matching_pennies():
    """2x2 simultaneous game (P1 shares infoset P1 across P0's two moves).

    u0: a0a1=+1 a0b1=-1 b0a1=-1 b0b1=+1 ; policies [3/4,1/4] each.
    onp0 = 1/4, onp1 = -1/4 ; br0 = 1/2 (pick a0), br1 = 1/2 (pick b1).
    NashConv = (1/2-1/4) + (1/2-(-1/4)) = 1/4 + 3/4 = 1.
    """
    # P1 nodes under a0 and b0 share pkey "P1" (imperfect info).
    p1_under_a0 = mk_decision(1, "P1", [mk_terminal(1, -1), mk_terminal(-1, 1)])
    p1_under_b0 = mk_decision(1, "P1", [mk_terminal(-1, 1), mk_terminal(1, -1)])
    root = _wrap_root(mk_decision(0, "P0", [p1_under_a0, p1_under_b0]))
    policy = _pol(P0=[0.75, 0.25], P1=[0.75, 0.25])
    nc, (br0, br1, onp0, onp1) = tiny_exact.exploitability_exact(root, policy)
    assert onp0 == Fraction(1, 4)
    assert onp1 == Fraction(-1, 4)
    assert br0 == Fraction(1, 2)
    assert br1 == Fraction(1, 2)
    assert nc == Fraction(1, 1)


def test_require_exact_tree_guard():
    """A tree without exact wfrac must hard-error, never silently use floats."""
    ch = Chance()
    ch.children = [mk_terminal(1, -1)]
    ch.weights = [1.0]  # wfrac stays None
    with pytest.raises(ValueError, match="exact_weights"):
        tiny_exact.exploitability_exact(ch, {})


def test_certify_bundle_and_margin():
    """certify() reports exact nashconv, exact signed margin vs 0.057, verdict."""
    root = _wrap_root(mk_decision(0, "I0", [mk_terminal(1, -1), mk_terminal(-1, 1)]))
    policy = _pol(I0=[0.75, 0.25])
    out = tiny_exact.certify(root, policy)
    assert out["nashconv"] == Fraction(1, 2)
    assert out["bar"] == Fraction(57, 1000)
    assert out["margin"] == Fraction(1, 2) - Fraction(57, 1000)
    assert out["passed"] is False  # 1/2 >> 0.057
    assert math.isclose(out["nashconv_float"], 0.5, rel_tol=0, abs_tol=1e-15)


# ---------------------------------------------------------------------------
# Layer 2: differential bound (float64 vs exact) on fixtures.
# ---------------------------------------------------------------------------

# The gate decides pass/fail by a strict inequality vs bar_respec=0.057; a cell
# meaningfully near the bar sits O(1e-3) away. float64 round-off must be far
# below that to never flip a verdict. We assert the differential is below this
# conservative bound on every fixture.
GATE_DECISION_SCALE = 1e-3
DIFF_BOUND = 1e-9  # >> observed (~1e-16), << GATE_DECISION_SCALE


def _fixture_trees():
    f1 = (
        _wrap_root(mk_decision(0, "I0", [mk_terminal(1, -1), mk_terminal(-1, 1)])),
        _pol(I0=[0.75, 0.25]),
    )
    f2 = (
        mk_chance(
            [
                mk_decision(0, "I", [mk_terminal(1, -1), mk_terminal(-1, 1)]),
                mk_decision(0, "I", [mk_terminal(-1, 1), mk_terminal(1, -1)]),
            ],
            [Fraction(1, 3), Fraction(2, 3)],
        ),
        _pol(I=[0.75, 0.25]),
    )
    # non-dyadic policy (0.1 is not exactly representable) over non-dyadic chance:
    # the float path rounds both; the exact path does not. Differential must
    # still be machine-epsilon-small.
    f3 = (
        mk_chance(
            [
                mk_decision(0, "J", [mk_terminal(1, -1), mk_terminal(-1, 1)]),
                mk_decision(0, "J", [mk_terminal(-1, 1), mk_terminal(1, -1)]),
            ],
            [Fraction(1, 3), Fraction(2, 3)],
        ),
        _pol(J=[0.1, 0.9]),
    )
    return [f1, f2, f3]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_differential_bound_fixtures(idx):
    root, policy = _fixture_trees()[idx]
    nc_f, _ = exploitability(root, policy)
    nc_e, _ = tiny_exact.exploitability_exact(root, policy)
    diff = abs(float(nc_f) - float(nc_e))
    assert diff < DIFF_BOUND
    assert diff < GATE_DECISION_SCALE  # comfortably below the verdict scale


def test_determinism_fixtures():
    """Exact path is bit-identical on repeat (same Fraction, not just close)."""
    for root, policy in _fixture_trees():
        a, _ = tiny_exact.exploitability_exact(root, policy)
        b, _ = tiny_exact.exploitability_exact(root, policy)
        assert a == b
        assert a.numerator == b.numerator and a.denominator == b.denominator


def test_lossless_float_lift():
    """Fraction(float(x)) reconstructs the exact float value for f32 and f64."""
    for x in (0.1, 0.2, 1.0 / 3.0, np.float32(0.1), np.float64(0.7)):
        fx = float(x)
        fr = Fraction(fx)
        assert float(fr) == fx  # exact round-trip
        assert fr.denominator & (fr.denominator - 1) == 0  # dyadic (power of two)


# ---------------------------------------------------------------------------
# Layer 3: real 230k-node {A,6} tree (session-scoped build).
# ---------------------------------------------------------------------------

_MAIN_SNAPSHOT_CANDIDATES = [
    "/home/jasonyu/dev/cambia/cfr/runs/v0.4-x2-ext-1000-xpu/snapshots/"
    "prtcfr_snapshot_iter_595.pt",
    "/home/jasonyu/dev/cambia/cfr/runs/v0.4-x2-fresh-1000-xpu/snapshots/"
    "prtcfr_snapshot_iter_595.pt",
]


def _find_snapshot():
    for p in _MAIN_SNAPSHOT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="module")
def exact_tree():
    from src.cfr import prtcfr_eval

    root, isets, n, ab = prtcfr_eval.build_tiny_tree(exact_weights=True)
    assert not ab and n == 230206 and len(isets) == 69636
    assert root.wfrac is not None and root.wfrac[0] == Fraction(1, 5)
    return root


def _uniform_policy(root):
    from src.cfr import prtcfr_eval

    nodes = prtcfr_eval.enumerate_infosets(root)
    return {
        nd.pkey: np.ones(len(nd.actions), dtype=np.float64) / len(nd.actions)
        for nd in nodes
    }


def test_real_uniform_U_exact(exact_tree):
    """Recompute U (uniform-policy exact NashConv, the bar anchor) exactly.

    bar_respec = round_2sig(0.0340 * U) must equal 0.057; assert the derivation.
    """
    policy = _uniform_policy(exact_tree)
    nc_f, _ = exploitability(exact_tree, policy)
    nc_e, _ = tiny_exact.exploitability_exact(exact_tree, policy)
    # exact vs float agree to machine epsilon.
    assert abs(float(nc_f) - float(nc_e)) < 1e-12
    U = nc_e
    assert math.isclose(float(U), 1.6727709190672155, rel_tol=0, abs_tol=1e-9)
    # bar derivation: 0.0340 * U rounded to 2 significant figures == 0.057.
    bar = 0.0340 * float(U)
    d = 2 - int(math.floor(math.log10(abs(bar)))) - 1
    assert round(bar, d) == 0.057
    # exact bar-derivation as a rational: 34/1000 * U, no rounding until the sig-fig step.
    assert isinstance(Fraction(34, 1000) * U, Fraction)


@pytest.mark.skipif(_find_snapshot() is None, reason="no X2 snapshot on disk")
def test_real_neural_snapshot_exact(exact_tree):
    """Exact vs float64 on one real trained snapshot; assert the differential."""
    from src.cfr import prtcfr_eval

    snap = _find_snapshot()
    it = 595
    net = prtcfr_eval._load_net(snap, device="cpu")
    policy = prtcfr_eval.materialize_policy_incremental(
        exact_tree, [(it, net)], weighting="linear"
    )
    nc_f, _ = exploitability(exact_tree, policy)
    nc_e, comp_e = tiny_exact.exploitability_exact(exact_tree, policy)
    diff = abs(float(nc_f) - float(nc_e))
    assert diff < DIFF_BOUND
    # exact verdict is well-defined and matches the float sign vs the bar.
    passed_exact = bool(nc_e < Fraction(57, 1000))
    passed_float = bool(float(nc_f) < 0.057)
    assert passed_exact == passed_float  # no verdict flip at this distance from bar
