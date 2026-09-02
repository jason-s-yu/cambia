"""CI gate: eval wrapper v2 encode path byte-equals trainer encoder across 100+ live states.

Hardened parity check for AC1 / finding f4-05 (Phase 0 measurement repair S1W7).

Background: The S1W2 encoder unification routed every v2 eval wrapper through
``encode_infoset_eppbs_interleaved_v2`` (the canonical trainer encoder), fixing
RC-B (57+ dims zeroed at eval time). This file provides the always-run CI gate
that validates the unification holds across a large, diverse sample of full-game
decision points.

What this test does:
- Replays complete seeded GO games (cambia-1426: the eval loop plays there, and
  the wrappers encode through the Go encoders).
- At every decision point (excluding snap-only drain steps), encodes the current
  GoAgentState belief through both the wrapper's encode path and the canonical
  encoder ``GoAgentState.encode_eppbs_interleaved_v2`` directly.
- Asserts exact byte-equality (``np.array_equal``) across the full 257-dim
  vector on every comparison.
- Runs until >= 100 distinct decision points have been compared across the seed
  set (typically reached in 5-7 seeds; 30 seeds are supplied for headroom).
- Covers both DESCA and PPO-v2 wrappers.

Design decisions:
- Needs libcambia.so, since the belief and the encoder both live behind the FFI.
- No ``pytest.skip`` / ``pytest.mark.xfail``; always run.
- Exact equality (not ``np.allclose``): the wrapper delegates to the identical
  function with the identical arguments, so floating-point output must be
  identical, not merely close.
- Parametrized over wrappers so each wrapper is an independent CI check.
- If a divergence is found, the test fails with a structured message naming
  the seed, step, actor, and divergent dim indices. This indicates the
  unification missed a wrapper or dim -- do not patch here; route back to
  encoder work.

Scope: tests the wrapper encode method only -- that it reaches the canonical
encoder with its own trainer's arguments and adds no re-derivation of its own.
Cross-engine (Python vs Go) parity at the 257-dim level is owned by
test_encoding_v2.py (test_python_v2_matches_go_v2_live_ffi_100_states), so this
file deliberately does not re-check it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

# Ensure cfr/src is importable.
_CFR_ROOT = Path(__file__).parent.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

from src.constants import (
    EP_PBS_INPUT_DIM,
    EP_PBS_V2_INPUT_DIM,
    ActionPassSnap,
    DecisionContext,
    ActionDiscard,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionSnapOpponentMove,
    NUM_PLAYERS,
)
from src.encoding import (
    encode_infoset_eppbs_interleaved_v2,
    encode_action_mask,
    action_to_index,
)

# ---------------------------------------------------------------------------
# Seed set: 30 seeds, each producing 15-30+ decision points -> well above 100.
# ---------------------------------------------------------------------------

_SEEDS = [
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

_SNAP_ACTION_MIN = 97

# Import helpers from existing infrastructure. These are stable; used by
# test_cross_engine_samples.py and test_cross_validation.py already.
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


# ---------------------------------------------------------------------------
# Decision-context + drawn-bucket helpers (mirrors the wrappers' own logic)
# ---------------------------------------------------------------------------


def _decision_context(py_state) -> DecisionContext:
    """Derive the DecisionContext enum for the current state, exactly as wrappers do."""
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


def _drawn_card_bucket(py_state) -> int:
    """The acting player's drawn-card bucket, -1 if none pending (POST_DRAW only)."""
    from src.abstraction import get_card_bucket
    from src.constants import CardBucket

    if py_state.snap_phase_active:
        return -1
    if not isinstance(py_state.pending_action, ActionDiscard):
        return -1
    drawn = (py_state.pending_action_data or {}).get("drawn_card")
    if drawn is None:
        return -1
    bucket = get_card_bucket(drawn)
    return int(bucket.value) if bucket != CardBucket.UNKNOWN else -1


# ---------------------------------------------------------------------------
# Wrapper stubs: build a bare wrapper that exposes the wrapper encode method
# without loading a checkpoint or torch model.
# ---------------------------------------------------------------------------


def _make_desca_wrapper():
    """DESCAAgentWrapper stub: bypasses __init__, wires only what _encode_v2 reads."""
    from src.evaluate_agents import DESCAAgentWrapper

    w = object.__new__(DESCAAgentWrapper)
    w.player_id = 0
    w.opponent_id = 1
    w._num_players = 2
    return w


def _make_ppo_v2_wrapper():
    """PPOAgentWrapper stub configured for v2 (257-dim), bypasses __init__."""
    from src.evaluate_agents import PPOAgentWrapper

    w = object.__new__(PPOAgentWrapper)
    w.player_id = 0
    w.opponent_id = 1
    w._num_players = 2
    w._encoding_version = 2
    w._obs_dim = EP_PBS_V2_INPUT_DIM
    return w


def _wrapper_encode(wrapper, agent_state, ctx: DecisionContext, drawn: int) -> np.ndarray:
    """Call the wrapper's v2 encode path against ``agent_state``."""
    from src.evaluate_agents import DESCAAgentWrapper, PPOAgentWrapper

    if isinstance(wrapper, DESCAAgentWrapper):
        wrapper.agent_state = agent_state
        return wrapper._encode_v2(ctx, drawn_card_bucket=drawn)
    if isinstance(wrapper, PPOAgentWrapper):
        wrapper.agent_state = agent_state
        return wrapper._encode_obs(ctx, drawn_card_bucket=drawn)
    raise TypeError(f"Unrecognised wrapper type: {type(wrapper)}")


# ---------------------------------------------------------------------------
# Core parity loop: replays seeded Go games and compares encode outputs per state.
# ---------------------------------------------------------------------------


def _run_parity_check(wrapper, min_states: int = 100):
    """Replay seeded Go games, compare wrapper encode vs the canonical encoder.

    Returns:
        (total_compared, first_divergence_message | None)
    """
    from src.agents import action_codec
    from src.evaluate_agents import PPOAgentWrapper, _decision_context, _drawn_card_bucket
    from src.ffi.bridge import GoAgentState, GoEngine, apply_games_batch

    config = _make_config()
    snap_indices = set(range(_SNAP_ACTION_MIN, 146))
    total = 0
    first_divergence: Optional[str] = None

    for seed in _SEEDS:
        if total >= min_states:
            break

        engine = GoEngine(seed=seed, house_rules=config.cambia_rules)
        agents = [GoAgentState(engine, pid) for pid in range(2)]
        try:
            for step in range(300):
                if engine.is_terminal():
                    break

                legal_idx = set(
                    int(i) for i in np.flatnonzero(engine.legal_actions_mask())
                )
                if not legal_idx:
                    break

                actor = engine.acting_player()
                snap_only = not (legal_idx - snap_indices)

                if not snap_only:
                    ctx = _decision_context(engine)
                    drawn = _drawn_card_bucket(engine)
                    agent_state = agents[actor]

                    # Reference: the canonical v2 encoder fed exactly what THIS
                    # wrapper's trainer feeds. DESCA's trainer passes the
                    # drawn-card bucket; PPO's (ppo_env._get_obs) does not
                    # (default -1). Per-agent train/eval parity means the
                    # reference must match each wrapper's own training
                    # convention, not a fixed bucket.
                    ref_bucket = -1 if isinstance(wrapper, PPOAgentWrapper) else drawn
                    ctx_val = ctx.value if hasattr(ctx, "value") else int(ctx)
                    ref = agent_state.encode_eppbs_interleaved_v2(ctx_val, ref_bucket)
                    got = _wrapper_encode(wrapper, agent_state, ctx, drawn)

                    assert ref.shape == (
                        EP_PBS_V2_INPUT_DIM,
                    ), f"Reference encoder returned shape {ref.shape}; expected (257,)"
                    assert got.shape == (
                        EP_PBS_V2_INPUT_DIM,
                    ), f"Wrapper returned shape {got.shape}; expected (257,)"

                    if not np.array_equal(ref, got):
                        diff_mask = ref != got
                        diff_indices = np.where(diff_mask)[0].tolist()
                        first_divergence = (
                            f"seed={seed} step={step} actor=P{actor} "
                            f"ctx={ctx.name} drawn_bucket={drawn}\n"
                            f"  divergent dim count: {len(diff_indices)}\n"
                            f"  divergent dims (first 24): {diff_indices[:24]}\n"
                            f"  ref  at divergent dims: {ref[diff_indices[:10]].tolist()}\n"
                            f"  got  at divergent dims: {got[diff_indices[:10]].tolist()}\n"
                            f"  v1 base [0:224] diverged: "
                            f"{bool(np.any(diff_mask[:EP_PBS_INPUT_DIM]))}\n"
                            f"  posterior [224:233] diverged: "
                            f"{bool(np.any(diff_mask[EP_PBS_INPUT_DIM:EP_PBS_INPUT_DIM + 9]))}\n"
                            f"  action-history [233:257] diverged: "
                            f"{bool(np.any(diff_mask[EP_PBS_INPUT_DIM + 9:]))}\n"
                            f"  total compared before divergence: {total}"
                        )
                        break

                    total += 1
                    if total >= min_states:
                        break

                # Advance: lowest-index legal action, preferring non-snap so the
                # walk keeps reaching fresh decision contexts.
                non_snap = sorted(legal_idx - snap_indices)
                action_idx = non_snap[0] if non_snap else sorted(legal_idx)[0]
                # Applied through the batch path so BOTH agents' beliefs advance,
                # which is what the eval loop does and what makes the belief under
                # comparison the one an evaluated agent actually holds.
                apply_games_batch(
                    [engine.handle],
                    [agents[0].handle],
                    [agents[1].handle],
                    [int(action_idx)],
                )
        finally:
            for a in agents:
                a.close()
            engine.close()

        if first_divergence is not None:
            break

    return total, first_divergence


# ---------------------------------------------------------------------------
# Tests: one per wrapper kind. Parametrized for independent CI reporting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_factory,wrapper_label",
    [
        (_make_desca_wrapper, "DESCAAgentWrapper"),
        (_make_ppo_v2_wrapper, "PPOAgentWrapper-v2"),
    ],
)
def test_v2_wrapper_byte_equal_to_trainer_encoder_257dim_100_states(
    wrapper_factory, wrapper_label
):
    """Wrapper v2 encode path must be byte-equal to encoder on full 257 dims, 100+ states.

    This is the CI gate for finding f4-05 (Phase 0 measurement repair S1W7).
    Replays 100+ seeded full-game decision points, asserting exact byte-equality
    (np.array_equal, not approximate) between the wrapper's encode path and the
    canonical encoder GoAgentState.encode_eppbs_interleaved_v2 on all 257 dims.

    Any divergence is reported with the first failing seed, step, actor, and the
    divergent dim indices. A divergence means the unification missed a dim or
    wrapper path and requires source-level attention -- do not patch here.
    """
    wrapper = wrapper_factory()
    total, divergence = _run_parity_check(wrapper, min_states=100)

    if divergence is not None:
        pytest.fail(
            f"[{wrapper_label}] Full 257-dim byte-equality parity failed:\n"
            f"{divergence}\n\n"
            f"This indicates the encoder unification (S1W2) missed a dim or wrapper "
            f"path. Do not patch this test -- route the finding back to encoder work."
        )

    assert total >= 100, (
        f"[{wrapper_label}] Only compared {total} states; need >= 100. "
        f"Increase _SEEDS or seed-game lengths."
    )
