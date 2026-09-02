"""Tier 1 tests for encoding v2 (Phase 0 DESCA foundation).

Covers:
- Dispatch on ``encoding_version``: v1 -> 224-dim, v2 -> 257-dim.
- Card-counting posterior: sums to 1.0, is non-constant across sampled states.
- Action history window: oldest-first ordering, 4-dim encoding per slot.
- Information-flow: specific dim values change when a slot is peeked as Ace (age 0)
  vs unknown (age N/A), at >= 10 distinct dim indices.
- Cross-path parity with Go v2 encoder: deferred hook - populated once Stream A's A2
  golden file is available. The hook is present here and will be activated by removing
  the skip guard when the Go fixture exists at tests/fixtures/encoding_v2_go.npz.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

from src.constants import (
    EP_PBS_INPUT_DIM,
    EP_PBS_V2_INPUT_DIM,
    EpistemicTag,
    V2_ACTION_CATEGORY_ABILITY_SNAP,
    V2_ACTION_CATEGORY_DIM,
    V2_ACTION_CATEGORY_DISCARD,
    V2_ACTION_CATEGORY_DRAW,
    V2_ACTION_HISTORY_DIM,
    V2_ACTION_HISTORY_PER_PLAYER,
    V2_ACTION_HISTORY_SLOTS,
    V2_ACTION_SLOT_FEATURE_DIM,
    V2_CARD_COUNT_DIM,
)
from src.encoding import (
    compute_action_history_window,
    compute_card_counting_posterior,
    encode_infoset_eppbs_interleaved,
    encode_infoset_eppbs_interleaved_v2,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_state(
    *,
    own_buckets=None,
    opp_pub_buckets=None,
    discard_counts=None,
    total_discards=None,
    action_history=None,
    current_turn: int = 0,
    max_game_turns: int = 300,
    slot_last_seen=None,
    player_id: int = 0,
    opponent_id: int = 1,
    own_hand_size: int = 4,
    opp_hand_size: int = 4,
):
    """Build a minimal object exposing the attributes the encoders read.

    Using a plain ``types.SimpleNamespace``-style object keeps the test decoupled
    from the real ``AgentState`` constructor and its config dependencies.
    """

    class _Stub:
        pass

    s = _Stub()
    s.player_id = int(player_id)
    s.opponent_id = int(opponent_id)
    s.slot_tags = [EpistemicTag.UNK] * 12
    s.slot_buckets = [0] * 12
    s.own_hand = {i: None for i in range(own_hand_size)}
    s.opponent_card_count = int(opp_hand_size)
    s.known_discard_top_bucket = type("B", (), {"value": 0})()
    s.stockpile_estimate = type("S", (), {"value": 0})()
    s.game_phase = type("P", (), {"value": 0})()
    s.cambia_caller = None
    s._current_game_turn = int(current_turn)
    s.max_game_turns = int(max_game_turns)
    s.slot_last_seen_turn = (
        list(slot_last_seen) if slot_last_seen is not None else [-1] * 12
    )
    s.discard_bucket_counts = (
        list(discard_counts) if discard_counts is not None else [0] * 9
    )
    s.total_discards_seen = int(
        total_discards if total_discards is not None else sum(s.discard_bucket_counts)
    )
    s.action_history = (
        action_history
        if action_history is not None
        else {
            int(player_id): [None, None, None],
            int(opponent_id): [None, None, None],
        }
    )

    if own_buckets is not None:
        for slot, (tag, bucket) in own_buckets.items():
            s.slot_tags[slot] = tag
            s.slot_buckets[slot] = bucket
    if opp_pub_buckets is not None:
        for slot, bucket in opp_pub_buckets.items():
            s.slot_tags[6 + slot] = EpistemicTag.PUB
            s.slot_buckets[6 + slot] = bucket
    return s


# ---------------------------------------------------------------------------
# Dispatch on encoding_version
# ---------------------------------------------------------------------------


def test_v1_dispatch_returns_224_dim():
    out = encode_infoset_eppbs_interleaved(
        slot_tags=[EpistemicTag.UNK] * 12,
        slot_buckets=[0] * 12,
        discard_top_bucket=0,
        stock_estimate=0,
        game_phase=0,
        decision_context=0,
        cambia_state=2,
    )
    assert out.shape == (EP_PBS_INPUT_DIM,)
    assert out.dtype == np.float32


def test_v2_dispatch_returns_257_dim_with_zero_extras():
    out = encode_infoset_eppbs_interleaved(
        slot_tags=[EpistemicTag.UNK] * 12,
        slot_buckets=[0] * 12,
        discard_top_bucket=0,
        stock_estimate=0,
        game_phase=0,
        decision_context=0,
        cambia_state=2,
        encoding_version=2,
    )
    assert out.shape == (EP_PBS_V2_INPUT_DIM,)
    # Posterior + action history default to zeros when omitted.
    assert np.all(out[EP_PBS_INPUT_DIM:] == 0.0)


def test_v2_base_prefix_matches_v1():
    """[0:224] of v2 output is bit-identical to v1 output on the same inputs."""
    kwargs = dict(
        slot_tags=[
            EpistemicTag.PRIV_OWN,
            EpistemicTag.UNK,
            EpistemicTag.PUB,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
            EpistemicTag.PRIV_OPP,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
            EpistemicTag.UNK,
        ],
        slot_buckets=[2, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        discard_top_bucket=3,
        stock_estimate=1,
        game_phase=2,
        decision_context=1,
        cambia_state=2,
        drawn_card_bucket=5,
        own_hand_size=4,
        opp_hand_size=4,
        own_obs_ages=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        opp_obs_ages=[0.0] * 6,
        dead_card_histogram=[0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
        turn_progress=0.5,
    )
    v1 = encode_infoset_eppbs_interleaved(**kwargs)
    posterior = np.array([1.0 / 9] * 9, dtype=np.float32)
    history = np.arange(V2_ACTION_HISTORY_DIM, dtype=np.float32) / 100.0
    v2 = encode_infoset_eppbs_interleaved(
        **kwargs,
        encoding_version=2,
        card_counting_posterior=posterior,
        action_history_window=history,
    )
    assert np.allclose(v1, v2[:EP_PBS_INPUT_DIM])
    assert np.allclose(
        v2[EP_PBS_INPUT_DIM : EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM], posterior
    )
    assert np.allclose(v2[EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM :], history)


def test_v2_rejects_wrong_shape_posterior():
    with pytest.raises(Exception):
        encode_infoset_eppbs_interleaved(
            slot_tags=[EpistemicTag.UNK] * 12,
            slot_buckets=[0] * 12,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
            encoding_version=2,
            card_counting_posterior=np.zeros(5, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Card-counting posterior
# ---------------------------------------------------------------------------


def test_posterior_sums_to_one_empty_observation():
    st = _make_agent_state()
    post = compute_card_counting_posterior(st)
    assert post.shape == (V2_CARD_COUNT_DIM,)
    assert math.isclose(float(post.sum()), 1.0, abs_tol=1e-6)


def test_posterior_sums_to_one_with_observations():
    # Own hand reveals an Ace (bucket 2) at slot 0; discard pile shows a HighKing.
    st = _make_agent_state(
        own_buckets={0: (EpistemicTag.PRIV_OWN, 2)},
        discard_counts=[0, 0, 0, 0, 0, 0, 0, 0, 1],
        total_discards=1,
    )
    post = compute_card_counting_posterior(st)
    assert math.isclose(float(post.sum()), 1.0, abs_tol=1e-6)


def test_posterior_non_constant_across_states():
    rng = random.Random(1234)
    samples = []
    for _ in range(20):
        own = {
            0: (EpistemicTag.PRIV_OWN, rng.randint(0, 8)),
            1: (EpistemicTag.PRIV_OWN, rng.randint(0, 8)),
        }
        disc = [rng.randint(0, 2) for _ in range(9)]
        st = _make_agent_state(
            own_buckets=own, discard_counts=disc, total_discards=sum(disc)
        )
        samples.append(compute_card_counting_posterior(st))
    arr = np.stack(samples, axis=0)
    # Each column should vary across the 20 samples; at least 5 columns non-constant.
    col_stds = arr.std(axis=0)
    assert (col_stds > 1e-6).sum() >= 5


def test_posterior_reflects_observation_direction():
    """Observing more Aces reduces the Ace mass in the posterior."""
    baseline = compute_card_counting_posterior(_make_agent_state())
    observed = compute_card_counting_posterior(
        _make_agent_state(
            own_buckets={
                0: (EpistemicTag.PRIV_OWN, 2),
                1: (EpistemicTag.PRIV_OWN, 2),
            },
            discard_counts=[0, 0, 1, 0, 0, 0, 0, 0, 0],
            total_discards=1,
        )
    )
    # Ace bucket = 2
    assert observed[2] < baseline[2]


# ---------------------------------------------------------------------------
# Action history window
# ---------------------------------------------------------------------------


def test_action_history_window_shape_and_default():
    st = _make_agent_state()
    out = compute_action_history_window(st)
    assert out.shape == (V2_ACTION_HISTORY_DIM,)
    assert np.all(out == 0.0)


def test_action_history_window_layout_oldest_first():
    # Build a ring with (oldest, mid, newest) categories for each player.
    own_ring = [
        (V2_ACTION_CATEGORY_DRAW, 0.0),
        (V2_ACTION_CATEGORY_DISCARD, 0.4),
        (V2_ACTION_CATEGORY_ABILITY_SNAP, 1.0),
    ]
    opp_ring = [None, (V2_ACTION_CATEGORY_DISCARD, 0.2), (V2_ACTION_CATEGORY_DRAW, 0.0)]
    st = _make_agent_state(
        player_id=0,
        opponent_id=1,
        action_history={0: own_ring, 1: opp_ring},
    )
    out = compute_action_history_window(st)
    # Own oldest slot: category DRAW = 0 -> out[0] = 1.0, scalar 0.0 at out[3].
    assert out[0] == 1.0
    assert out[3] == 0.0
    # Own mid slot: DISCARD=1 at offset 4; scalar 0.4 at offset 7.
    assert out[4 + 1] == 1.0
    assert math.isclose(float(out[7]), 0.4, abs_tol=1e-6)
    # Own newest: ABILITY_SNAP=2 at offset 8; scalar 1.0 at offset 11.
    assert out[8 + 2] == 1.0
    assert math.isclose(float(out[11]), 1.0, abs_tol=1e-6)
    # Opponent oldest slot empty: all zeros at offsets [12:16].
    assert np.all(
        out[V2_ACTION_HISTORY_PER_PLAYER : V2_ACTION_HISTORY_PER_PLAYER + 4] == 0.0
    )
    # Opponent mid slot: DISCARD=1, scalar 0.2 at offsets [16:20].
    assert out[V2_ACTION_HISTORY_PER_PLAYER + 4 + 1] == 1.0
    assert math.isclose(
        float(out[V2_ACTION_HISTORY_PER_PLAYER + 4 + 3]), 0.2, abs_tol=1e-6
    )


def test_action_history_category_one_hot_exclusivity():
    """In a filled slot, exactly one of the 3 category dims is set."""
    st = _make_agent_state(
        player_id=0,
        opponent_id=1,
        action_history={
            0: [
                (V2_ACTION_CATEGORY_DRAW, 0.0),
                (V2_ACTION_CATEGORY_DISCARD, 0.0),
                (V2_ACTION_CATEGORY_ABILITY_SNAP, 0.0),
            ],
            1: [None, None, None],
        },
    )
    out = compute_action_history_window(st)
    for slot_idx in range(V2_ACTION_HISTORY_SLOTS):
        base = slot_idx * V2_ACTION_SLOT_FEATURE_DIM
        one_hot = out[base : base + V2_ACTION_CATEGORY_DIM]
        assert int(one_hot.sum()) == 1


# ---------------------------------------------------------------------------
# Information-flow tests (contract gate)
# ---------------------------------------------------------------------------


class _DecisionContext:
    def __init__(self, val: int):
        self.value = val


def test_info_flow_peeked_ace_vs_unknown_differs_on_10_dims():
    """Peeking slot 0 as Ace (age 0) vs leaving it unknown changes >= 10 dims."""
    # Base: unknown state.
    base = _make_agent_state(current_turn=5, max_game_turns=300)
    # Peeked: slot 0 is PRIV_OWN with bucket 2 (Ace), last_seen_turn = 5 (age 0).
    base.slot_tags = list(base.slot_tags)
    base.slot_buckets = list(base.slot_buckets)
    peeked_last_seen = list(base.slot_last_seen_turn)
    peeked_last_seen[0] = 5
    peeked = _make_agent_state(
        current_turn=5,
        max_game_turns=300,
        own_buckets={0: (EpistemicTag.PRIV_OWN, 2)},
        slot_last_seen=peeked_last_seen,
    )
    ctx = _DecisionContext(0)
    enc_base = encode_infoset_eppbs_interleaved_v2(base, ctx, drawn_card_bucket=-1)
    enc_peeked = encode_infoset_eppbs_interleaved_v2(peeked, ctx, drawn_card_bucket=-1)
    assert enc_base.shape == (EP_PBS_V2_INPUT_DIM,)
    diff_dims = int(np.sum(np.abs(enc_base - enc_peeked) > 1e-7))
    assert (
        diff_dims >= 10
    ), f"Expected >= 10 differing dims between peeked-Ace and unknown states; got {diff_dims}"


# ---------------------------------------------------------------------------
# Cross-path parity with Go: live FFI-driven test (Phase 0 decision gate)
# ---------------------------------------------------------------------------
#
# This runs matched Go+Python games through identical action sequences and
# compares v2 encoder output at every decision point. The Python AgentState
# tracks parity fields (B1) that match the Go AgentState's semantics, and
# encode_infoset_eppbs_interleaved_v2 pulls from those fields. Any divergence
# in the first 224 dims reflects a v1-layer mismatch (pre-existing, not v2).
# Divergence in [224:257] reflects a v2-specific mismatch.

try:
    from src.ffi.bridge import GoAgentState, GoEngine

    _HAS_GO = True
except Exception:
    _HAS_GO = False

_skipgo = pytest.mark.skipif(not _HAS_GO, reason="libcambia.so not available")


def _cross_path_parity_seeds():
    """Return a list of seeds such that combined decision points total >= 100.

    A single Cambia game produces roughly 15-30 decision points before terminal,
    so 10 seeds comfortably clears the 100-state target per the Phase 0 gate.

    The pool is sized larger than strictly required so pre-snap decision points
    (where the action-history window dims [233:257] can be compared against Go)
    accumulate across seeds even when most games enter snap phases early. The
    harness narrows to [0:233] after the first snap on a seed due to the
    PassSnap acting_player asymmetry documented in the test body.
    """
    return [
        42,
        137,
        313,
        1729,
        2718,
        3141,
        4096,
        5551,
        7777,
        12345,
        54321,
        99991,
        100003,
        131313,
        242424,
        333667,
        414213,
        500000,
        600613,
        714285,
        808080,
        919191,
        10000019,
        31337,
        65537,
        104729,
        271828,
        998244353,
        1000003,
        2718281,
    ]


@_skipgo
def test_python_v2_matches_go_v2_live_ffi_100_states():
    """Python v2 encoding matches Go v2 encoding within 1e-5 on >= 100 matched states.

    Drives matched Go+Python games through the same legal-action sequence and
    compares the 257-dim v2 output at each decision point. The test runs until
    at least 100 encoding comparisons have been made across the seed set.
    """
    # Local imports to avoid polluting module load when libcambia is absent.
    from src.agent_state import AgentObservation  # noqa: F401 (used via helpers)
    from src.constants import ActionPassSnap, NUM_PLAYERS
    from src.encoding import (
        NUM_ACTIONS,
        action_to_index,
        encode_action_mask,
    )

    # Import lockstep helpers from the existing cross-engine harness.
    try:
        from tests.test_cross_engine_samples import (
            _setup_python_game_matching_go,
            _is_snap_only,
            _PASS_SNAP_IDX,
            _TEST_RULES,
        )
    except ImportError:
        from test_cross_engine_samples import (  # type: ignore
            _setup_python_game_matching_go,
            _is_snap_only,
            _PASS_SNAP_IDX,
            _TEST_RULES,
        )
    try:
        from tests.test_cross_validation import (
            _build_py_agents,
            _create_py_observation,
            _dc_int_to_enum,
            _make_config,
        )
    except ImportError:
        from test_cross_validation import (  # type: ignore
            _build_py_agents,
            _create_py_observation,
            _dc_int_to_enum,
            _make_config,
        )

    _SNAP_ACTION_MIN = 97
    snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))

    config = _make_config()
    total_comparisons = 0
    full_257_comparisons = 0  # comparisons that exercised dims [233:257]
    first_divergence = None
    max_abs_diff_overall = 0.0

    for seed in _cross_path_parity_seeds():
        if total_comparisons >= 100:
            break

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)
        go_agents = [GoAgentState(go_engine, i) for i in range(NUM_PLAYERS)]
        py_agents = _build_py_agents(py_state, config)

        try:
            snap_passes = 0
            # ring_drift tracks asymmetric snap-pass pushes: if one side applies
            # ActionPassSnap while the other does not, the two action-history rings
            # receive different sequences, so comparing [233:257] would report a
            # false divergence. When drift > 0, compare only [0:233].
            ring_drift = 0
            for step in range(200):
                if go_engine.is_terminal() or py_state.is_terminal():
                    break

                go_mask = go_engine.legal_actions_mask()
                go_actions = set(np.where(go_mask > 0)[0].tolist())
                py_legal = py_state.get_legal_actions()
                py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
                py_actions = set(np.where(py_mask > 0)[0].tolist())

                # Drain snap-only states identically on both sides.
                # Go stamps LastAction.ActingPlayer = current acting player when
                # PassSnap is applied (see engine/snap.go). Capture that player id
                # before apply so Python's _create_py_observation can pass the same
                # actor to its ring-buffer update, keeping [233:257] in sync across
                # paths rather than narrowing the parity compare after the first snap.
                go_snap_only = _is_snap_only(go_actions)
                py_snap_phase = py_state.snap_phase_active
                if go_snap_only:
                    snapper = go_engine.acting_player()
                    go_engine.apply_action(_PASS_SNAP_IDX)
                    go_engine.update_both(go_agents[0], go_agents[1])
                    snap_passes += 1
                    if py_snap_phase:
                        py_state.apply_action(ActionPassSnap())
                        obs = _create_py_observation(py_state, ActionPassSnap(), snapper)
                        for pa in py_agents:
                            try:
                                pa.update(obs)
                            except Exception:
                                pass
                    else:
                        ring_drift += 1  # Go pushed PassSnap, Python did not.
                    continue
                if py_snap_phase:
                    snapper = go_engine.acting_player()
                    py_state.apply_action(ActionPassSnap())
                    obs = _create_py_observation(py_state, ActionPassSnap(), snapper)
                    for pa in py_agents:
                        try:
                            pa.update(obs)
                        except Exception:
                            pass
                    snap_passes += 1
                    ring_drift += 1  # Python pushed PassSnap, Go did not.
                    continue

                if snap_passes > 0:
                    go_non_snap = go_actions - snap_indices
                    py_non_snap = py_actions - snap_indices
                    if go_non_snap != py_non_snap:
                        break

                actor = go_engine.acting_player()
                ctx_int = go_engine.decision_ctx()
                drawn_int = go_engine.get_drawn_card_bucket()

                go_v2 = go_agents[actor].encode_eppbs_interleaved_v2(ctx_int, drawn_int)
                py_ctx = _dc_int_to_enum(ctx_int)
                py_v2 = encode_infoset_eppbs_interleaved_v2(
                    py_agents[actor],
                    py_ctx,
                    drawn_card_bucket=int(drawn_int),
                )

                assert go_v2.shape == (EP_PBS_V2_INPUT_DIM,)
                assert py_v2.shape == (EP_PBS_V2_INPUT_DIM,)
                assert np.all(np.isfinite(go_v2))
                assert np.all(np.isfinite(py_v2))

                # The PassSnap harness above passes Go's acting_player as the Python
                # actor, so both rings receive aligned category-2 pushes and the full
                # 257-dim compare can stay active across snap phases. ring_drift is
                # only incremented when one side applies PassSnap while the other
                # does not (happens if Python detects no snap_phase_active when Go
                # enters snap-only). In that rare case, narrow the compare to [0:233]
                # to avoid false divergence reports. The action-history layout is
                # separately verified by unit tests on each side
                # (test_action_history_window_layout_oldest_first and Go's
                # encoding_v2_test.go ring-buffer tests).
                if ring_drift > 0:
                    compare_slice = slice(0, EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM)
                else:
                    compare_slice = slice(0, EP_PBS_V2_INPUT_DIM)
                    full_257_comparisons += 1
                go_cmp = go_v2[compare_slice]
                py_cmp = py_v2[compare_slice]
                abs_diff = np.abs(go_cmp - py_cmp)
                max_abs_diff_overall = max(max_abs_diff_overall, float(abs_diff.max()))

                if not np.allclose(go_cmp, py_cmp, atol=1e-5):
                    diff_idx = np.where(abs_diff > 1e-5)[0]
                    first_divergence = (
                        f"seed={seed} step={step} actor=P{actor} "
                        f"ctx={ctx_int} drawn={drawn_int} ring_drift={ring_drift}\n"
                        f"  compared slice: {compare_slice}\n"
                        f"  divergent dims: {diff_idx.tolist()[:20]} "
                        f"(total {len(diff_idx)})\n"
                        f"  go values: {go_cmp[diff_idx].tolist()[:10]}\n"
                        f"  py values: {py_cmp[diff_idx].tolist()[:10]}\n"
                        f"  max abs diff: {abs_diff.max():.6g}\n"
                        f"  total comparisons before divergence: {total_comparisons}"
                    )
                    break

                total_comparisons += 1
                if total_comparisons >= 100:
                    break

                # Advance: pick lowest common non-snap legal action.
                go_non_snap = go_actions - snap_indices
                py_non_snap = py_actions - snap_indices
                common = sorted(go_non_snap & py_non_snap)
                if not common:
                    break
                action_idx = common[0]

                py_action = None
                for a in py_legal:
                    try:
                        if action_to_index(a) == action_idx:
                            py_action = a
                            break
                    except Exception:
                        pass
                if py_action is None:
                    break

                go_engine.apply_action(action_idx)
                go_engine.update_both(go_agents[0], go_agents[1])
                py_state.apply_action(py_action)
                obs = _create_py_observation(py_state, py_action, actor)
                for pa in py_agents:
                    try:
                        pa.update(obs)
                    except Exception:
                        pass

            if first_divergence is not None:
                break
        finally:
            for a in go_agents:
                a.close()
            go_engine.close()

    if first_divergence is not None:
        pytest.fail(
            f"V2 cross-path parity failure (max abs diff so far = "
            f"{max_abs_diff_overall:.6g}):\n{first_divergence}"
        )

    assert total_comparisons >= 100, (
        f"Only compared {total_comparisons} states; expected >= 100. "
        f"max abs diff observed: {max_abs_diff_overall:.6g}"
    )
    # Guard against the parity test accidentally collapsing to [0:233] coverage
    # only. The action-history dims [233:257] are under-specified by unit tests
    # alone; the cross-path gate needs a non-trivial count of full-257 compares
    # across the seed set.
    assert full_257_comparisons >= 50, (
        f"Only {full_257_comparisons}/{total_comparisons} comparisons exercised "
        f"dims [233:257] (full 257-dim parity). The rest were narrowed to "
        f"[0:233] due to snap-pass harness asymmetry. Gate criterion #3 requires "
        f"substantive coverage of the action-history window across seeds."
    )


# ---------------------------------------------------------------------------
# Wrapper-vs-trainer encoder parity (S1W2), migrated to the Go backend by
# cambia-1522: the eval wrapper must encode through the encoder the trainer
# feeds, sourcing the decision context and drawn-card bucket the way its own
# choose_action does.
#
# This started as a pure-Python smoke check: it attached a Python reference
# AgentState to the wrapper and compared dims [224:257] against
# encode_infoset_eppbs_interleaved_v2, because the pre-fix wrappers hand-rolled
# a low-level call that omitted the posterior + action-history kwargs and left
# that block zero (RC-B). cambia-1422 retired the Python rules engine and
# cambia-1426 moved the eval battery onto libcambia, so the wrappers encode
# through GoAgentState only and the Python attach raised AttributeError.
#
# The check is kept rather than deleted, because the risk it guards did not
# retire with the Python path: a wrapper can still hand its encoder the wrong
# decision context or the wrong drawn-card bucket, and on Go it can still name
# an encoder entry the belief does not expose. What did retire is the zeroed
# block: the Go encoder fills all 257 dims or fails. So the comparison now runs
# over all 257 dims (a superset of the [224:257] block in the test name), which
# is what makes a ctx or bucket drift visible instead of cancelling out.
# ---------------------------------------------------------------------------


#: Whether each wrapper hands its encoder the engine's drawn-card bucket.
#:
#: DESCAAgentWrapper.choose_action passes it, matching desca_worker._encode_state.
#: PPOAgentWrapper._encode_obs pins it to -1. That -1 no longer matches
#: ppo_env._get_obs, which has passed the real bucket since cambia-1376; the
#: divergence is reported against the PPO anchor, not settled here, so the
#: reference follows each wrapper's declared policy and this test pins the
#: encoder entry and the context sourcing without ruling on the bucket.
_WRAPPER_PASSES_DRAWN_BUCKET = {"desca": True, "ppo": False}


def _smoke_parity_seeds():
    """Small seed set; a handful of games clears the decision-point target."""
    return [42, 137, 313, 1729, 2718, 3141, 4096, 5551]


def _make_test_wrapper(kind, player_id):
    """Build a wrapper exposing its v2 encode without loading a checkpoint.

    ``__init__`` is bypassed (it imports torch and loads a checkpoint); only the
    attributes the belief lifecycle and the encode path read are bound, so the
    wrapper's own attach_belief runs and builds a real GoAgentState.

    ``config`` is a namespace rather than a Config: attach_belief reads only
    ``agent_params.memory_level`` and ``agent_params.time_decay_turns``, and the
    tests/conftest.py Config stub carries neither.
    """
    from types import SimpleNamespace

    from src.evaluate_agents import DESCAAgentWrapper, PPOAgentWrapper

    if kind == "desca":
        w = object.__new__(DESCAAgentWrapper)
        w.agent_state = None
    else:
        w = object.__new__(PPOAgentWrapper)
        w.agent_state = None
        w._encoding_version = 2
        w._obs_dim = EP_PBS_V2_INPUT_DIM
    w.player_id = int(player_id)
    w.opponent_id = 1 - int(player_id)
    w.config = SimpleNamespace(
        agent_params=SimpleNamespace(memory_level=0, time_decay_turns=0)
    )
    w._num_players = 2
    return w


def _wrapper_belief(wrapper, kind):
    """The GoAgentState the wrapper attached.

    One name for both: PPOAgentWrapper published its belief privately until
    cambia-2022, which is what made the N-player transition step raise for a PPO
    seat above two seats.
    """
    return wrapper.agent_state


def _wrapper_v2_encode(wrapper, kind, engine):
    """Drive the wrapper's v2 encode the way its own choose_action drives it."""
    from src.evaluate_agents import _decision_context, _drawn_card_bucket

    if kind == "desca":
        # DESCAAgentWrapper.choose_action sources both from the engine.
        return wrapper._encode_v2(
            wrapper._get_decision_context(engine), _drawn_card_bucket(engine)
        )
    # PPOAgentWrapper.choose_action sources the context only.
    return wrapper._encode_obs(_decision_context(engine))


@_skipgo
@pytest.mark.parametrize("wrapper_kind", ["desca", "ppo"])
def test_v2_eval_wrapper_matches_trainer_encoder_on_dims_224_257(wrapper_kind):
    """Eval wrapper v2 encode must match the trainer encoder, all 257 dims.

    Plays seeded Go games, encoding at every non-snap decision point through
    both the wrapper path and ``encode_infoset_eppbs_interleaved_v2``, the entry
    desca_worker._encode_state calls and the one ppo_env._get_obs short-circuits
    to for a Go-backed belief. Asserts exact equality over the full vector, so
    the [224:257] posterior + action-history block named in this test is covered
    along with the public block that carries the decision context and the
    drawn-card one-hot.
    """
    from src.constants import NUM_PLAYERS
    from src.encoding import NUM_ACTIONS

    try:
        from tests.test_cross_engine_samples import (
            _is_snap_only,
            _PASS_SNAP_IDX,
            _TEST_RULES,
        )
    except ImportError:
        from test_cross_engine_samples import (  # type: ignore
            _is_snap_only,
            _PASS_SNAP_IDX,
            _TEST_RULES,
        )

    pass_bucket = _WRAPPER_PASSES_DRAWN_BUCKET[wrapper_kind]

    _SNAP_ACTION_MIN = 97
    snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))

    total_comparisons = 0
    full_block_comparisons = 0  # comparisons where [224:257] was non-zero
    bucket_comparisons = 0  # comparisons at a live drawn-card bucket
    first_divergence = None

    for seed in _smoke_parity_seeds():
        if total_comparisons >= 60:
            break

        engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        wrappers = [_make_test_wrapper(wrapper_kind, i) for i in range(NUM_PLAYERS)]
        try:
            beliefs = [w.attach_belief(engine, NUM_PLAYERS) for w in wrappers]
            for _step in range(200):
                if engine.is_terminal():
                    break

                mask = engine.legal_actions_mask()
                actions = set(np.where(mask > 0)[0].tolist())

                # Drain snap-only states: PassSnap is the only move, and no
                # wrapper encodes there (choose_action is not consulted).
                if _is_snap_only(actions):
                    engine.apply_action(_PASS_SNAP_IDX)
                    engine.update_both(beliefs[0], beliefs[1])
                    continue

                actor = engine.acting_player()
                engine_bucket = int(engine.get_drawn_card_bucket())

                # Reference: the trainer's encoder entry, with the context and
                # bucket read straight off the engine.
                ref = encode_infoset_eppbs_interleaved_v2(
                    beliefs[actor],
                    int(engine.decision_ctx()),
                    drawn_card_bucket=(engine_bucket if pass_bucket else -1),
                )
                got = _wrapper_v2_encode(wrappers[actor], wrapper_kind, engine)

                assert ref.shape == (EP_PBS_V2_INPUT_DIM,)
                assert got.shape == (EP_PBS_V2_INPUT_DIM,)

                if not np.array_equal(ref, got):
                    diff_idx = np.where(np.abs(ref - got) > 0)[0]
                    first_divergence = (
                        f"seed={seed} step={_step} actor=P{actor} "
                        f"ctx={int(engine.decision_ctx())} bucket={engine_bucket} "
                        f"kind={wrapper_kind}\n"
                        f"  divergent dims: {diff_idx.tolist()[:24]}\n"
                        f"  ref at divergence: {ref[diff_idx].tolist()[:24]}\n"
                        f"  got at divergence: {got[diff_idx].tolist()[:24]}"
                    )
                    break

                # The posterior block [224:233] always sums to 1.0 on a live
                # state, so [224:257] is never all-zero. Tracking it proves the
                # block is exercised rather than compared as two zero vectors.
                if np.any(ref[EP_PBS_INPUT_DIM:EP_PBS_V2_INPUT_DIM] != 0.0):
                    full_block_comparisons += 1
                if engine_bucket >= 0:
                    bucket_comparisons += 1
                total_comparisons += 1
                if total_comparisons >= 60:
                    break

                # Advance on the lowest-index non-snap legal action.
                non_snap = sorted(actions - snap_indices)
                if not non_snap:
                    break
                engine.apply_action(non_snap[0])
                engine.update_both(beliefs[0], beliefs[1])
        finally:
            for w in wrappers:
                w.release_belief()
            engine.close()

        if first_divergence is not None:
            break

    assert first_divergence is None, (
        f"{wrapper_kind} eval wrapper diverged from the trainer encoder:\n"
        f"{first_divergence}"
    )
    assert total_comparisons >= 20, (
        f"Only {total_comparisons} wrapper/trainer encode comparisons ran; the "
        f"seed set should clear 20 non-snap decision points."
    )
    assert full_block_comparisons == total_comparisons, (
        f"{full_block_comparisons}/{total_comparisons} comparisons had a "
        f"non-zero [224:257] block; a zeroed block means the v2 extras were "
        f"never exercised and the parity claim is vacuous."
    )
    assert bucket_comparisons >= 1, (
        "No comparison ran at a live drawn-card bucket, so the bucket half of "
        "the wrapper's encoder contract went unexercised."
    )


# ---------------------------------------------------------------------------
# The encoder contract itself (cambia-1522 AC1): every eval encode path resolves
# its belief through _require_go_belief, so a belief that does not expose the
# named Go encoder is refused by name at the wrapper boundary instead of raising
# AttributeError from inside the encoder call.
# ---------------------------------------------------------------------------


def test_require_go_belief_refuses_a_belief_without_the_encoder():
    """A non-Go belief is rejected by name, naming the missing encoder."""
    from src.cfr.exceptions import AgentStateError
    from src.evaluate_agents import DESCAAgentWrapper, _require_go_belief

    class _NoEncoder:
        pass

    owner = object.__new__(DESCAAgentWrapper)
    owner.player_id = 0

    with pytest.raises(AgentStateError) as unattached:
        _require_go_belief(owner, None, "encode_eppbs_interleaved_v2")
    assert "belief not attached" in str(unattached.value)

    with pytest.raises(AgentStateError) as wrong_backend:
        _require_go_belief(owner, _NoEncoder(), "encode_eppbs_interleaved_v2")
    message = str(wrong_backend.value)
    assert "encode_eppbs_interleaved_v2" in message
    assert "_NoEncoder" in message
    assert "GoAgentState" in message


@_skipgo
def test_require_go_belief_accepts_a_go_backed_belief():
    """A GoAgentState passes the contract and is returned unchanged."""
    from src.evaluate_agents import DESCAAgentWrapper, _require_go_belief

    try:
        from tests.test_cross_engine_samples import _TEST_RULES
    except ImportError:
        from test_cross_engine_samples import _TEST_RULES  # type: ignore

    owner = object.__new__(DESCAAgentWrapper)
    owner.player_id = 0

    engine = GoEngine(seed=42, house_rules=_TEST_RULES)
    belief = GoAgentState(engine, 0)
    try:
        for encoder in (
            "encode",
            "encode_eppbs",
            "encode_eppbs_dealiased",
            "encode_eppbs_interleaved",
            "encode_eppbs_interleaved_v2",
            "encode_nplayer",
        ):
            assert _require_go_belief(owner, belief, encoder) is belief
    finally:
        belief.close()
        engine.close()
