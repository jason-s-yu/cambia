"""Exact-rational NashConv certifier for the X2 gate (cambia-530, amendment A2).

The X2 gate scores a trained PRT-CFR policy's NashConv/exploitability on the
exact tiny {A,6} Cambia tree in float64 (``tools.tiny_solver.exploitability``)
and rules replication cells against ``bar_respec = 0.057``. Because that verdict
is a strict inequality against a fixed bar, float64 round-off could in principle
flip pass/fail near the bar. This module recomputes the SAME quantity with
Python ``fractions.Fraction`` end-to-end so the gate can be certified: no
rounding, no ``limit_denominator``, no epsilon comparison anywhere on the exact
path.

Exactness of every input:
  - Tree utilities are engine payoffs in {-1, 0, +1} (float, but integer-valued);
    ``Fraction(u)`` is exact.
  - Chance mass is the true rational, read from ``Chance.wfrac`` (1/K deal, cnt/
    total draw), NOT the rounded float in ``Chance.weights``. The tree MUST be
    built with ``build_tree(..., exact_weights=True)`` / ``build_tiny_tree(...,
    exact_weights=True)``; ``require_exact_tree`` enforces this.
  - Policy entries are float64 probabilities; ``Fraction(float(x))`` is the exact
    dyadic value of the float, a lossless lift. The policy is the artifact under
    certification, so it is taken as-is (no renormalization), exactly matching
    what the float scorer multiplies.

Semantics mirror ``tools.tiny_solver`` node-for-node so the exact number is the
infinite-precision value of the very arithmetic float64 approximates:
  - on-policy value: both players play ``policy``;
  - best response: exact infoset best response by policy iteration over the
    single-player MDP induced by fixing the opponent, argmax by exact Fraction
    comparison (first-max tie-break, matching ``numpy.argmax``);
  - NashConv = (br0 - onp0) + (br1 - onp1).

Denominator growth: float64 policy entries carry ~52-bit denominators, so on the
real neural tree the exact reach fractions accumulate very large denominators
(the exact pass is far slower than float64 and can be heavy). The uniform policy
(1/nA, tiny denominators) is cheap and yields U exactly. Callers should MEASURE
wall time rather than assume the neural exact pass is free.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, Tuple

# X2 respec bar as an exact rational (bar_respec = 0.057, cambia-517).
BAR_RESPEC = Fraction(57, 1000)


def require_exact_tree(root) -> None:
    """Fail loudly unless the tree carries exact-rational chance weights.

    A silent fall-through to float weights would defeat the whole certification,
    so this is a hard precondition, not a warning.
    """
    if getattr(root, "kind", None) != "C" or getattr(root, "wfrac", None) is None:
        raise ValueError(
            "tiny_exact requires a tree built with exact_weights=True "
            "(root.wfrac is None). Call build_tiny_tree(exact_weights=True) or "
            "build_tree(..., exact_weights=True)."
        )


def _chance_wfrac(node) -> list:
    """Exact weights for a chance node; hard error if missing (never float)."""
    wf = node.wfrac
    if wf is None:
        raise ValueError(
            "chance node without exact wfrac; build with exact_weights=True"
        )
    return wf


def _lift_dist(pol: Dict[Any, Any], node) -> list:
    """Exact policy distribution for a decision node, mirroring tiny_solver._lookup.

    Tries the compound key (pkey, nA) then the bare pkey (production keying);
    returns a length-nA list of Fractions lifted losslessly from the stored
    float64 vector, or exact uniform 1/nA if missing or length-mismatched.
    ``Fraction(float(x))`` is exact for float32/float64 alike (float() upcasts
    losslessly), so this never rounds.
    """
    nA = len(node.actions)
    d = pol.get((node.pkey, nA))
    if d is None or len(d) != nA:
        d = pol.get(node.pkey)
    if d is not None and len(d) == nA:
        return [Fraction(float(x)) for x in d]
    u = Fraction(1, nA)
    return [u] * nA


def _policy_value(node, policy_by_player: Dict[int, Any], who: int) -> Fraction:
    """Exact value for player ``who`` when both players play their given policies."""
    kind = node.kind
    if kind == "T":
        return Fraction(node.util[who])
    if kind == "C":
        wf = _chance_wfrac(node)
        v = Fraction(0)
        for c, w in zip(node.children, wf):
            v += w * _policy_value(c, policy_by_player, who)
        return v
    dist = _lift_dist(policy_by_player[node.player], node)
    v = Fraction(0)
    for i, p in enumerate(dist):
        if p <= 0:
            continue
        v += p * _policy_value(node.children[i], policy_by_player, who)
    return v


def _br_eval_rec(node, br_player, policy, br_actions, cfav, cfreach) -> Fraction:
    """Value to ``br_player`` under the fixed BR action map, accumulating exact
    per-infoset counterfactual action values into ``cfav`` (mirrors
    tiny_solver._br_eval_rec with Fraction arithmetic)."""
    kind = node.kind
    if kind == "T":
        return Fraction(node.util[br_player])
    if kind == "C":
        wf = _chance_wfrac(node)
        v = Fraction(0)
        for c, w in zip(node.children, wf):
            v += w * _br_eval_rec(
                c, br_player, policy, br_actions, cfav, cfreach * w
            )
        return v
    if node.player == br_player:
        nA = len(node.actions)
        bkey = (node.pkey, nA)
        vals = [
            _br_eval_rec(
                node.children[i], br_player, policy, br_actions, cfav, cfreach
            )
            for i in range(nA)
        ]
        acc = cfav.get(bkey)
        if acc is None:
            acc = [Fraction(0)] * nA
            cfav[bkey] = acc
        cfav[bkey] = [a + cfreach * v for a, v in zip(acc, vals)]
        chosen = br_actions.get(bkey, 0)
        if chosen >= nA:
            chosen = 0
        return vals[chosen]
    # opponent node: weight children by opponent policy (folded into cfreach)
    dist = _lift_dist(policy, node)
    v = Fraction(0)
    for i, p in enumerate(dist):
        if p <= 0:
            continue
        v += p * _br_eval_rec(
            node.children[i], br_player, policy, br_actions, cfav, cfreach * p
        )
    return v


def _argmax_first(acc) -> int:
    """Index of the first maximal exact value, matching numpy.argmax tie-break."""
    best_i = 0
    best_v = acc[0]
    for i in range(1, len(acc)):
        if acc[i] > best_v:
            best_v = acc[i]
            best_i = i
    return best_i


def _br_value(node, br_player, policy, max_sweeps: int = 64) -> Fraction:
    """Exact infoset best-response value for ``br_player`` vs ``policy``.

    Policy iteration on the single-player MDP induced by fixing the opponent;
    on a finite tree this converges to the exact infoset best response. Argmax
    over accumulated exact counterfactual values, first-max tie-break."""
    br_actions: Dict[Any, int] = {}
    last_val = None
    for _ in range(max_sweeps):
        cfav: Dict[Any, list] = {}
        val = _br_eval_rec(node, br_player, policy, br_actions, cfav, Fraction(1))
        changed = False
        for iset, acc in cfav.items():
            best = _argmax_first(acc)
            if br_actions.get(iset) != best:
                br_actions[iset] = best
                changed = True
        if not changed and last_val is not None:
            return val
        last_val = val
    cfav = {}
    return _br_eval_rec(node, br_player, policy, br_actions, cfav, Fraction(1))


def exploitability_exact(root, policy) -> Tuple[Fraction, Tuple[Fraction, ...]]:
    """Exact NashConv (sum over players of BR value - on-policy value).

    Zero-sum two-player => this is the standard exploitability/NashConv in
    utility units. Returns (nashconv, (br0, br1, onp0, onp1)), all Fraction.
    """
    require_exact_tree(root)
    pol_by_player = {0: policy, 1: policy}
    onp0 = _policy_value(root, pol_by_player, 0)
    onp1 = _policy_value(root, pol_by_player, 1)
    br0 = _br_value(root, 0, policy)
    br1 = _br_value(root, 1, policy)
    nc = (br0 - onp0) + (br1 - onp1)
    return nc, (br0, br1, onp0, onp1)


def signed_margin(nashconv: Fraction, bar: Fraction = BAR_RESPEC) -> Fraction:
    """Exact signed margin nashconv - bar. Negative => passes (nashconv < bar)."""
    return nashconv - bar


def certify(root, policy, bar: Fraction = BAR_RESPEC) -> Dict[str, Any]:
    """Exact NashConv verdict bundle for a policy on an exact-weight tiny tree.

    Returns exact Fractions plus float projections for logging:
        {"nashconv": Fraction, "nashconv_float": float,
         "components": (br0, br1, onp0, onp1) Fractions,
         "margin": Fraction (nashconv - bar), "margin_float": float,
         "bar": Fraction, "passed": bool (nashconv < bar, exact)}
    """
    nc, comp = exploitability_exact(root, policy)
    margin = signed_margin(nc, bar)
    return {
        "nashconv": nc,
        "nashconv_float": float(nc),
        "components": comp,
        "components_float": tuple(float(x) for x in comp),
        "margin": margin,
        "margin_float": float(margin),
        "bar": bar,
        "passed": bool(nc < bar),
    }
