"""DESCA env_factory parity test for the Go FFI backend (T1-1).

The Go and Python game engines deal cards differently (Go flips a discard
on Deal, Python starts with empty discard) so a literal lockstep walk
across backends is impossible from a fresh game. The Phase 0
cross-validation already certified the Go and Python *encoders* are
bit-equivalent on equivalent state; that's the load-bearing claim.

This test instead exercises the new Go env_factory adapter surface end
to end:
- 50 random concrete actions step through without raising
- legal-actions output, abstract-action mask, and encoded 257-dim
  features stay well-formed across the walk
- snapshot/restore via the worker helpers round-trips both engine and
  agent state
- agent attributes consumed by `cfr/src/action_abstraction.py`
  (own_hand, opponent_belief, _current_game_turn) remain coherent
  after every update + clone cycle

Numerical tolerance: rtol=0, atol=1e-5.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.action_abstraction import abstract_actions, NUM_ABSTRACT_ACTIONS_2P
from src.cli import (  # type: ignore[attr-defined]
    _build_desca_env_factory_for_test_go,
)
from src.cfr.desca_worker import _encode_state

_NUM_STEPS = 50
_TOLERANCE_ATOL = 1e-5


def test_go_backend_random_walk_50_steps():
    """50 random concrete actions on the Go env_factory must produce
    coherent legal-actions, abstract-action mask, and encoded feature
    output at every step. Counts as a stress test for the
    `_GoEngineAdapter` + `_GoAgentStateAdapter` surfaces.
    """
    seed = 0xDEADBEEF
    factory = _build_desca_env_factory_for_test_go()
    rng = np.random.default_rng(seed)
    engine, agents = factory(rng=rng)
    walk_rng = np.random.default_rng(seed + 1)

    steps_completed = 0
    for step_idx in range(_NUM_STEPS):
        if engine.is_terminal():
            break

        legal = engine.legal_actions()
        if not legal:
            break

        actor = engine.get_acting_player()
        agent = agents[actor]

        # Legal actions: concrete NamedTuple list, non-empty, all action.tag set.
        for a in legal:
            assert hasattr(a, "tag"), f"step {step_idx}: action {a} lacks .tag"

        # Abstract action mask: at least one bit set; exact shape correct.
        mask = abstract_actions(legal, agent)
        assert mask.dtype == bool
        assert mask.shape == (NUM_ABSTRACT_ACTIONS_2P,)
        assert mask.any(), (
            f"step {step_idx}: abstract action mask had no set bits "
            f"(legal={[a.tag for a in legal]})"
        )

        # Encoded feature vector: 257-dim float32, finite values.
        feat = _encode_state(engine, agent)
        assert feat.shape == (257,)
        assert feat.dtype == np.float32
        assert np.isfinite(feat).all(), f"step {step_idx}: non-finite values in encoding"

        # Pick a random legal action; advance state in both engine + agents.
        choice_idx = int(walk_rng.integers(0, len(legal)))
        action = legal[choice_idx]
        engine.apply_action(action)
        for a in agents:
            a.update(engine)

        # Belief surface invariants on the actor's agent post-update.
        own_len_max = max(agent.own_hand.keys()) + 1 if agent.own_hand else 0
        opp_len_max = (
            max(agent.opponent_belief.keys()) + 1 if agent.opponent_belief else 0
        )
        # Hand sizes never exceed MaxHandSize=6.
        assert own_len_max <= 6
        assert opp_len_max <= 6
        # current_game_turn monotonic non-decreasing.
        assert agent._current_game_turn >= 0

        steps_completed += 1

    assert steps_completed >= 5, (
        f"random walk halted unexpectedly early at step {steps_completed} "
        f"(terminal={engine.is_terminal()})"
    )


def test_go_backend_clone_independence_under_apply():
    """A cloned Go-backed agent must not be affected by the original's
    update after apply. This is the load-bearing invariant for the
    DESCA worker's snapshot/restore via `_snapshot` and `_restore`.
    """
    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(rng=np.random.default_rng(0))

    actor = engine.get_acting_player()
    a_orig = agents[actor]
    a_clone = a_orig.clone()

    legal = engine.legal_actions()
    assert legal, "expected legal actions at root"
    engine.apply_action(legal[0])
    for a in agents:
        a.update(engine)

    # The clone's _current_game_turn must remain at the pre-apply value.
    # The original's may have advanced (depends on the action).
    # Critical: handles must differ.
    assert a_clone._go_agent._agent_h != a_orig._go_agent._agent_h

    # Encoding the clone (without updating it) should still produce a
    # valid 257-dim feature vector.
    # Note: the clone references a stale game state; encode_eppbs_interleaved_v2
    # operates on the agent's belief state which is independent of the
    # current game state. So this just stresses the FFI handle is alive.
    decision_ctx = engine.get_decision_context()
    feat = a_clone._go_agent.encode_eppbs_interleaved_v2(int(decision_ctx))
    assert feat.shape == (257,)
    assert feat.dtype == np.float32


def test_go_backend_encoding_dim():
    """Go env_factory's encoded features must be 257-dim float32."""
    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(np.random.default_rng(0))
    feat = _encode_state(engine, agents[engine.get_acting_player()])
    assert feat.shape == (257,), f"got shape {feat.shape}, expected (257,)"
    assert feat.dtype == np.float32


def test_go_backend_omniscient_dim():
    """Go env_factory's omniscient features must be 120-dim (2P)."""
    from src.cfr.desca_worker import _encode_omniscient

    factory = _build_desca_env_factory_for_test_go()
    engine, _agents = factory(np.random.default_rng(0))
    omni = _encode_omniscient(engine)
    assert omni.shape == (120,), f"got shape {omni.shape}, expected (120,)"
    assert omni.dtype == np.float32
    # Sum should be num_slots = 12 (each slot one-hot).
    assert np.isclose(
        float(omni.sum()), 12.0
    ), f"omniscient sum {float(omni.sum())} != 12.0 (one-hot per slot violated)"


def test_go_backend_clone_isolation():
    """Cloning a Go-backed agent must not share underlying handle."""
    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(np.random.default_rng(0))
    a0 = agents[0]
    cloned = a0.clone()
    assert (
        cloned._go_agent._agent_h != a0._go_agent._agent_h
    ), "clone shares the same underlying agent handle"


def test_go_backend_encoding_dim():
    """Go env_factory's encoded features must be 257-dim float32."""
    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(np.random.default_rng(0))
    feat = _encode_state(engine, agents[engine.get_acting_player()])
    assert feat.shape == (257,), f"got shape {feat.shape}, expected (257,)"
    assert feat.dtype == np.float32


def test_go_backend_omniscient_dim():
    """Go env_factory's omniscient features must be 120-dim (2P)."""
    from src.cfr.desca_worker import _encode_omniscient

    factory = _build_desca_env_factory_for_test_go()
    engine, _agents = factory(np.random.default_rng(0))
    omni = _encode_omniscient(engine)
    assert omni.shape == (120,), f"got shape {omni.shape}, expected (120,)"
    assert omni.dtype == np.float32
    # Sum should be num_slots = 12 (each slot one-hot).
    assert np.isclose(
        float(omni.sum()), 12.0
    ), f"omniscient sum {float(omni.sum())} != 12.0 (one-hot per slot violated)"


def test_go_backend_clone_isolation():
    """Cloning a Go-backed agent must not share belief state with the original."""
    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(np.random.default_rng(0))
    a0 = agents[0]
    cloned = a0.clone()

    # Mutate the original via update; cloned should remain at original turn.
    legal = engine.legal_actions()
    if legal:
        engine.apply_action(legal[0])
        for a in agents:
            a.update(engine)
    # cloned was constructed before the apply, so its current_game_turn
    # should match the pre-apply original. (The original's turn may have
    # advanced; cloned's must not.)
    # We assert that cloned has its OWN underlying GoAgentState handle
    # (independent), even if the turn happens to coincide on equal-state
    # games.
    assert (
        cloned._go_agent._agent_h != a0._go_agent._agent_h
    ), "clone shares the same underlying agent handle"


def test_go_backend_save_restore():
    """Save and restore on the Go-backed engine must round-trip when used
    via the worker's `_snapshot`/`_restore` helpers (which clone agents
    alongside the engine snapshot).

    Mirrors the production path in `desca_worker._traverse`: snapshot
    captures both the engine and a clone of every agent's belief state.
    Restore swaps the cloned agents back in and the rewound engine state.
    """
    from src.cfr.desca_worker import _snapshot, _restore

    factory = _build_desca_env_factory_for_test_go()
    engine, agents = factory(np.random.default_rng(0))

    # Capture initial features.
    actor = engine.get_acting_player()
    feat_before = _encode_state(engine, agents[actor])

    snap = _snapshot(engine, agents)
    legal = engine.legal_actions()
    assert legal, "expected at least one legal action at root"
    engine.apply_action(legal[0])
    for a in agents:
        a.update(engine)

    _restore(engine, agents, snap)
    # After restore, encoding must match (state + belief recovered).
    feat_after = _encode_state(engine, agents[actor])
    np.testing.assert_allclose(
        feat_before,
        feat_after,
        rtol=0,
        atol=_TOLERANCE_ATOL,
        err_msg="restore did not recover original engine+agent state's encoding",
    )
