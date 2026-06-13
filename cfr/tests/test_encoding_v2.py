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
    s.slot_last_seen_turn = list(slot_last_seen) if slot_last_seen is not None else [-1] * 12
    s.discard_bucket_counts = (
        list(discard_counts) if discard_counts is not None else [0] * 9
    )
    s.total_discards_seen = int(
        total_discards
        if total_discards is not None
        else sum(s.discard_bucket_counts)
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
            EpistemicTag.PRIV_OWN, EpistemicTag.UNK, EpistemicTag.PUB, EpistemicTag.UNK,
            EpistemicTag.UNK, EpistemicTag.UNK,
            EpistemicTag.PRIV_OPP, EpistemicTag.UNK, EpistemicTag.UNK, EpistemicTag.UNK,
            EpistemicTag.UNK, EpistemicTag.UNK,
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
    assert np.allclose(v2[EP_PBS_INPUT_DIM:EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM], posterior)
    assert np.allclose(v2[EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM:], history)


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
        st = _make_agent_state(own_buckets=own, discard_counts=disc,
                               total_discards=sum(disc))
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
    assert np.all(out[V2_ACTION_HISTORY_PER_PLAYER:V2_ACTION_HISTORY_PER_PLAYER + 4] == 0.0)
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
        one_hot = out[base:base + V2_ACTION_CATEGORY_DIM]
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
    assert diff_dims >= 10, (
        f"Expected >= 10 differing dims between peeked-Ace and unknown states; got {diff_dims}"
    )


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
        42, 137, 313, 1729, 2718, 3141, 4096, 5551, 7777, 12345,
        54321, 99991, 100003, 131313, 242424, 333667, 414213, 500000,
        600613, 714285, 808080, 919191, 10000019, 31337, 65537,
        104729, 271828, 998244353, 1000003, 2718281,
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
                    py_agents[actor], py_ctx, drawn_card_bucket=int(drawn_int),
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
# Wrapper-vs-trainer encoder parity (S1W2): the eval wrapper must encode through
# the SAME high-level encoder the trainer uses. Pre-fix, DESCAAgentWrapper and
# the v2 PPO wrapper hand-rolled a low-level call that omitted the posterior +
# action-history kwargs, zeroing dims [224:257] relative to training (RC-B).
#
# This is a pure-Python smoke check (no Go FFI): it replays a few seeded games
# through Python AgentState.update() the way the eval harness does, then asserts
# the wrapper encode path is byte-equal to encode_infoset_eppbs_interleaved_v2
# on the v2-specific dims [224:257]. The hardened 100+-state Go cross-path test
# above is the separate gate; this just guards the wrapper delegation.
# ---------------------------------------------------------------------------


def _smoke_parity_seeds():
    """Small seed set; a handful of games clears the decision-point target."""
    return [42, 137, 313, 1729, 2718, 3141, 4096, 5551]


def _wrapper_v2_encode(wrapper, py_state, agent_state, ctx):
    """Drive a wrapper's bound v2 encode method against a Python agent_state.

    Sets the AgentState on the wrapper (both the public ``agent_state`` and the
    private ``_agent_state`` PPO uses) and calls its encode entry. Returns the
    257-dim vector the wrapper would feed its network.
    """
    wrapper.agent_state = agent_state
    wrapper._agent_state = agent_state
    return wrapper._encode_v2_for_test(py_state, ctx)


@pytest.mark.parametrize("wrapper_kind", ["desca", "ppo"])
def test_v2_eval_wrapper_matches_trainer_encoder_on_dims_224_257(wrapper_kind):
    """Eval wrapper v2 encode must match the trainer encoder on dims [224:257].

    Replays seeded Python games, encoding at every decision point through both
    the wrapper path and ``encode_infoset_eppbs_interleaved_v2``. Asserts exact
    byte-equality on the v2 posterior + action-history block. Pre-fix this fails
    because the wrapper omitted those kwargs and the block stayed all-zero.
    """
    from src.constants import ActionPassSnap, CardBucket, DecisionContext, NUM_PLAYERS
    from src.encoding import action_to_index, encode_action_mask

    try:
        from tests.test_cross_engine_samples import _setup_python_game_matching_go
        from tests.test_cross_validation import (
            _build_py_agents,
            _create_py_observation,
            _make_config,
        )
    except ImportError:
        from test_cross_engine_samples import _setup_python_game_matching_go  # type: ignore
        from test_cross_validation import (  # type: ignore
            _build_py_agents,
            _create_py_observation,
            _make_config,
        )

    wrapper = _make_test_wrapper(wrapper_kind)
    config = _make_config()

    _SNAP_ACTION_MIN = 97
    NUM_ACTIONS = encode_action_mask([]).shape[0]
    snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))

    total_comparisons = 0
    full_block_comparisons = 0  # comparisons where [224:257] was non-zero
    first_divergence = None

    for seed in _smoke_parity_seeds():
        if total_comparisons >= 60:
            break
        py_state = _setup_python_game_matching_go(seed)
        py_agents = _build_py_agents(py_state, config)

        for step in range(200):
            if py_state.is_terminal():
                break

            py_legal = py_state.get_legal_actions()
            py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
            py_actions = set(np.where(py_mask > 0)[0].tolist())

            # Drain snap-only phases uniformly (no encoding comparison there;
            # the acting player is mid-snap and PassSnap is the only move).
            if py_state.snap_phase_active:
                actor = py_state.get_acting_player()
                py_state.apply_action(ActionPassSnap())
                obs = _create_py_observation(py_state, ActionPassSnap(), actor)
                for pa in py_agents:
                    try:
                        pa.update(obs)
                    except Exception:
                        pass
                continue

            actor = py_state.get_acting_player()
            ctx = _py_decision_context(py_state)

            # Reference: canonical trainer encoder, with the drawn bucket sourced
            # the same way the wrapper sources it.
            drawn_bucket = _py_drawn_card_bucket(py_state)
            ref = encode_infoset_eppbs_interleaved_v2(
                py_agents[actor], ctx, drawn_card_bucket=drawn_bucket
            )
            got = _wrapper_v2_encode(wrapper, py_state, py_agents[actor], ctx)

            assert ref.shape == (EP_PBS_V2_INPUT_DIM,)
            assert got.shape == (EP_PBS_V2_INPUT_DIM,)

            block_ref = ref[EP_PBS_INPUT_DIM:EP_PBS_V2_INPUT_DIM]
            block_got = got[EP_PBS_INPUT_DIM:EP_PBS_V2_INPUT_DIM]
            if not np.array_equal(block_ref, block_got):
                diff_idx = (
                    np.where(np.abs(block_ref - block_got) > 0)[0] + EP_PBS_INPUT_DIM
                )
                first_divergence = (
                    f"seed={seed} step={step} actor=P{actor} ctx={int(ctx.value)} "
                    f"kind={wrapper_kind}\n"
                    f"  divergent dims (abs): {diff_idx.tolist()[:24]}\n"
                    f"  ref[224:257]={block_ref.tolist()}\n"
                    f"  got[224:257]={block_got.tolist()}"
                )
                break

            # The posterior block [224:233] always sums to 1.0, so [224:257] is
            # never all-zero on a live state. Track that to prove the block is
            # actually exercised (a wrapper that zeroed it would diverge above,
            # but this also guards a degenerate ref).
            if np.any(block_ref != 0.0):
                full_block_comparisons += 1
            total_comparisons += 1
            if total_comparisons >= 60:
                break

            # Advance on the lowest-index non-snap legal action.
            non_snap = sorted(py_actions - snap_indices)
            if not non_snap:
                break
            action_idx = non_snap[0]
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
            py_state.apply_action(py_action)
            obs = _create_py_observation(py_state, py_action, actor)
            for pa in py_agents:
                try:
                    pa.update(obs)
                except Exception:
                    pass

        if first_divergence is not None:
            break

    if first_divergence is not None:
        pytest.fail(
            f"Wrapper-vs-trainer v2 encoder parity failure on [224:257]:\n"
            f"{first_divergence}"
        )

    assert total_comparisons >= 20, (
        f"Only compared {total_comparisons} states; expected >= 20 for a "
        f"meaningful smoke check."
    )
    assert full_block_comparisons == total_comparisons, (
        f"{total_comparisons - full_block_comparisons} comparisons had an "
        f"all-zero [224:257] reference block; the posterior must always sum to "
        f"1.0 on a live state, so this indicates a broken reference encoder."
    )


def _make_test_wrapper(kind):
    """Build a wrapper instance exposing ``_encode_v2_for_test`` without loading
    a checkpoint or torch model. We bypass ``__init__`` and bind only what the
    encode method reads.
    """
    from src.constants import DecisionContext  # noqa: F401 (sanity import)
    from src.evaluate_agents import DESCAAgentWrapper, PPOAgentWrapper

    if kind == "desca":
        w = object.__new__(DESCAAgentWrapper)
        w.player_id = 0
        w.opponent_id = 1

        def _encode(py_state, ctx, _w=w):
            return _w._encode_v2(ctx, drawn_card_bucket=_py_drawn_card_bucket(py_state))

        w._encode_v2_for_test = _encode
        return w

    # PPO: the v2 branch lives inside choose_action. We exercise the same
    # canonical-encoder delegation the fixed wrapper uses for a 257-dim model.
    w = object.__new__(PPOAgentWrapper)
    w.player_id = 0
    w.opponent_id = 1
    w._encoding_version = 2
    w._obs_dim = EP_PBS_V2_INPUT_DIM

    def _encode_ppo(py_state, ctx, _w=w):
        return _w._encode_obs(ctx, drawn_card_bucket=_py_drawn_card_bucket(py_state))

    w._encode_v2_for_test = _encode_ppo
    return w


def _py_decision_context(py_state):
    """Mirror the wrappers' decision-context derivation for a Python game."""
    from src.constants import (
        ActionAbilityBlindSwapSelect,
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionAbilityPeekOtherSelect,
        ActionAbilityPeekOwnSelect,
        ActionDiscard,
        ActionSnapOpponentMove,
        DecisionContext,
    )

    if py_state.snap_phase_active:
        return DecisionContext.SNAP_DECISION
    pending = py_state.pending_action
    if pending is not None:
        if isinstance(pending, ActionDiscard):
            return DecisionContext.POST_DRAW
        if isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            return DecisionContext.ABILITY_SELECT
        if isinstance(pending, ActionSnapOpponentMove):
            return DecisionContext.SNAP_MOVE
    return DecisionContext.START_TURN


def _py_drawn_card_bucket(py_state):
    """The acting player's own drawn-card bucket, or -1 if none is pending.

    Mirrors Go ``cambia_game_get_drawn_card_bucket``: a bucket only exists while
    a discard decision is pending (POST_DRAW). The drawn card is the acting
    player's own private info, legitimately known to them at decision time.
    """
    from src.abstraction import get_card_bucket
    from src.constants import ActionDiscard, CardBucket

    if py_state.snap_phase_active:
        return -1
    if not isinstance(py_state.pending_action, ActionDiscard):
        return -1
    drawn = (py_state.pending_action_data or {}).get("drawn_card")
    if drawn is None:
        return -1
    bucket = get_card_bucket(drawn)
    return int(bucket.value) if bucket != CardBucket.UNKNOWN else -1
