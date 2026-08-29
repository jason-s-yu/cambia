"""Tests for scripts/gate_vs_served_gap.py (cambia-737).

The load-bearing claim of that script is that the reach-weighted average it
builds from the run's snapshots IS the policy the SD-CFR wrapper plays: the
wrapper samples one snapshot per EPISODE proportional to ``w_t = t`` and plays
the whole game with it, and under perfect recall that mixture over behavioral
strategies is realization equivalent to the own-reach-weighted average.

These tests pin that on a hand-built two-snapshot case where both objects can be
written down in closed form:

  - the realization plan (probability of each own action sequence) of the
    reconstructed behavioral strategy must equal the mixture's own realization
    plan, sequence by sequence -- the definition of realization equivalence, and
    the property that makes best-response values and exploitability agree;
  - the gate's reach-unweighted average must NOT match it, so the test would
    fail if the script silently computed the gate object twice;
  - the accumulator path (weights folded snapshot by snapshot, exactly as the
    sharded run does) must reproduce the closed form.

No torch, no game engine, no run directory: the reconstruction math is pure
numpy and is imported directly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "gate_vs_served_gap.py",
)
_spec = importlib.util.spec_from_file_location("gate_vs_served_gap", _SCRIPT)
gvs = importlib.util.module_from_spec(_spec)
sys.modules["gate_vs_served_gap"] = gvs
_spec.loader.exec_module(gvs)


# ---------------------------------------------------------------------------
# A three-infoset own-decision chain for one player:
#
#   I0 (root, 2 actions)  --a0--> I1 (2 actions)
#                         --a1--> I2 (3 actions)
#
# Every leaf below I1/I2 is terminal for this player's own sequence, so the
# player's realization plan is fully described by the probabilities of the
# sequences (a), (a, b).
# ---------------------------------------------------------------------------

COUNTS = np.array([2, 2, 3], dtype=np.int64)
LEGAL_OFF = np.array([0, 2, 4, 7], dtype=np.int64)
PARENT_ISET = np.array([-1, 0, 0], dtype=np.int64)
PARENT_SLOT = np.array([-1, 0, 1], dtype=np.int64)
DEPTH_ORDER = np.array([0, 1, 2], dtype=np.int64)
DEPTH_START = np.array([0, 1, 3], dtype=np.int64)

# Two snapshots with deliberately different behavior at every infoset.
SNAP_A = np.array([0.8, 0.2, 0.5, 0.5, 0.1, 0.6, 0.3], dtype=np.float64)
SNAP_B = np.array([0.1, 0.9, 0.25, 0.75, 0.7, 0.2, 0.1], dtype=np.float64)
ITERS = (3, 7)  # linear SD-CFR weights w_t = t


def _slice(flat, i):
    return flat[LEGAL_OFF[i] : LEGAL_OFF[i + 1]]


def _sequences(flat):
    """Realization plan of one behavioral strategy: {sequence: probability}."""
    out = {}
    root = _slice(flat, 0)
    for a in range(2):
        out[(a,)] = root[a]
        child = 1 if a == 0 else 2
        for b in range(int(COUNTS[child])):
            out[(a, b)] = root[a] * _slice(flat, child)[b]
    return out


def _mixture_sequences():
    """Realization plan of the SAMPLED mixture: E_t[realization plan of sigma_t]."""
    w = np.array(ITERS, dtype=np.float64)
    p = w / w.sum()
    plans = [_sequences(SNAP_A), _sequences(SNAP_B)]
    keys = plans[0].keys()
    return {k: sum(pi * plan[k] for pi, plan in zip(p, plans)) for k in keys}


def _accumulate(snapshots):
    acc_gate = np.zeros(int(LEGAL_OFF[-1]), dtype=np.float64)
    acc_served = np.zeros(int(LEGAL_OFF[-1]), dtype=np.float64)
    reach_weight = np.zeros(COUNTS.shape[0], dtype=np.float64)
    wsum = 0.0
    for it, strat in snapshots:
        wsum += gvs.accumulate_snapshot(
            acc_gate, acc_served, reach_weight, strat, float(it), LEGAL_OFF,
            PARENT_ISET, PARENT_SLOT, DEPTH_ORDER, DEPTH_START, COUNTS,
        )
    return acc_gate, acc_served, reach_weight, wsum


def test_own_reaches_are_the_ancestor_action_products():
    pi = gvs.own_reaches(
        SNAP_A, LEGAL_OFF, PARENT_ISET, PARENT_SLOT, DEPTH_ORDER, DEPTH_START
    )
    assert pi[0] == pytest.approx(1.0)
    assert pi[1] == pytest.approx(0.8)  # sigma_A(I0)[a0]
    assert pi[2] == pytest.approx(0.2)  # sigma_A(I0)[a1]


def test_served_mixture_is_realization_equivalent_to_episode_sampling():
    _g, acc_served, reach_weight, _w = _accumulate(
        [(ITERS[0], SNAP_A), (ITERS[1], SNAP_B)]
    )
    served, unreached = gvs.finalize_served(acc_served, reach_weight, LEGAL_OFF)
    assert not unreached.any()

    got = _sequences(served)
    want = _mixture_sequences()
    for key, value in want.items():
        assert got[key] == pytest.approx(value, abs=1e-12), key


def test_gate_average_is_a_different_object_and_is_not_realization_equivalent():
    acc_gate, _s, _r, wsum = _accumulate([(ITERS[0], SNAP_A), (ITERS[1], SNAP_B)])
    gate = gvs.finalize_gate(acc_gate, wsum, LEGAL_OFF)

    # The gate is the plain weighted mean of the per-infoset distributions.
    w = np.array(ITERS, dtype=np.float64)
    expected = (w[0] * SNAP_A + w[1] * SNAP_B) / w.sum()
    assert gate == pytest.approx(expected, abs=1e-12)

    # It is NOT the served policy: the two agree at the root (own reach 1 there)
    # and diverge below it, so their realization plans differ.
    _g2, acc_served, reach_weight, _w2 = _accumulate(
        [(ITERS[0], SNAP_A), (ITERS[1], SNAP_B)]
    )
    served, _u = gvs.finalize_served(acc_served, reach_weight, LEGAL_OFF)
    assert _slice(gate, 0) == pytest.approx(_slice(served, 0), abs=1e-12)
    assert not np.allclose(_slice(gate, 1), _slice(served, 1))

    gate_plan = _sequences(gate)
    mix_plan = _mixture_sequences()
    assert not np.isclose(gate_plan[(0, 0)], mix_plan[(0, 0)], atol=1e-9)


def test_single_snapshot_mixture_is_that_snapshot():
    for it in (1, 5, 900):
        _g, acc_served, reach_weight, _w = _accumulate([(it, SNAP_A)])
        served, unreached = gvs.finalize_served(acc_served, reach_weight, LEGAL_OFF)
        assert not unreached.any()
        assert served == pytest.approx(SNAP_A, abs=1e-12)


def test_unreachable_infoset_falls_back_to_uniform_and_is_flagged():
    # Both snapshots put zero probability on the root action leading to I2, so
    # I2's realization weight is zero under the whole mixture.
    a = np.array([1.0, 0.0, 0.5, 0.5, 0.1, 0.6, 0.3], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.25, 0.75, 0.7, 0.2, 0.1], dtype=np.float64)
    _g, acc_served, reach_weight, _w = _accumulate([(3, a), (7, b)])
    served, unreached = gvs.finalize_served(acc_served, reach_weight, LEGAL_OFF)
    assert unreached.tolist() == [False, False, True]
    assert _slice(served, 2) == pytest.approx(np.full(3, 1.0 / 3.0))
    # The reached part still matches the closed-form mixture.
    assert _slice(served, 1) == pytest.approx((3 * a[2:4] + 7 * b[2:4]) / 10.0)


def test_accumulator_is_order_and_shard_invariant():
    whole = _accumulate([(2, SNAP_A), (5, SNAP_B), (9, SNAP_A)])
    part1 = _accumulate([(2, SNAP_A)])
    part2 = _accumulate([(5, SNAP_B), (9, SNAP_A)])
    for k in range(3):
        assert whole[k] == pytest.approx(part1[k] + part2[k], abs=1e-12)
    assert whole[3] == pytest.approx(part1[3] + part2[3])


def test_shard_plan_lands_on_every_checkpoint():
    plan = gvs._shard_plan([350, 700, 1000], 1000, 25)
    assert plan[0][0] == 1
    assert plan[-1][1] == 1000
    boundaries = {hi for _lo, hi in plan}
    assert {350, 700, 1000} <= boundaries
    covered = []
    for lo, hi in plan:
        covered.extend(range(lo, hi + 1))
    assert covered == list(range(1, 1001))


def test_atomic_savez_leaves_no_partial_and_round_trips(tmp_path):
    out = tmp_path / "shard_00001_00025.npz"
    gvs._atomic_savez(out, acc=np.arange(5.0), wsum=np.float64(3.0))
    assert out.is_file()
    assert list(tmp_path.glob("*.partial")) == []
    z = np.load(out, allow_pickle=False)
    assert z["acc"] == pytest.approx(np.arange(5.0))
    assert float(z["wsum"]) == 3.0


def test_completion_needs_the_sidecar_not_just_the_payload(tmp_path):
    payload = tmp_path / "shard_00001_00025.npz"
    payload.write_bytes(b"truncated")
    # A half-written shard whose sidecar never landed must not read as done,
    # or a relaunch would fold garbage into a checkpoint's accumulator.
    assert not gvs._is_complete(payload, ".info.json")
    (tmp_path / "shard_00001_00025.npz.info.json").write_text("{}")
    assert gvs._is_complete(payload, ".info.json")


def test_sweep_partials_removes_only_temp_files(tmp_path):
    (tmp_path / "shard_00001_00025.npz").write_bytes(b"x")
    (tmp_path / "shard_00026_00050.npz.partial").write_bytes(b"x")
    (tmp_path / "tree.npz.meta.json.partial").write_text("{")
    assert gvs._sweep_partials(tmp_path) == 2
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["shard_00001_00025.npz"]
    assert gvs._sweep_partials(tmp_path / "nonexistent") == 0


def test_shard_plan_stops_at_the_last_checkpoint():
    # c1's shape: 747 snapshots on disk, but 1000 is unreachable, so nothing
    # past 700 is ever folded into an accumulator and nothing past 700 is
    # encoded.
    plan = gvs._shard_plan([350, 700], 747, 25)
    assert plan[-1][1] == 700
    boundaries = {hi for _lo, hi in plan}
    assert {350, 700} <= boundaries
    covered = []
    for lo, hi in plan:
        covered.extend(range(lo, hi + 1))
    assert covered == list(range(1, 701))
