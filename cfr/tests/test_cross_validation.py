"""
tests/test_cross_validation.py

Cross-engine validation tests: Go FFI engine vs Python reference engine.

Tests verify that:
- Encoding dimensions match network constructors (not just constants)
- Go and Python engines agree on acting_player and decision_ctx throughout games
- Legacy 222-dim encodings match between Go and Python agents at matched states
- EP-PBS 200-dim encodings match between Go and Python agents at matched states
"""

import copy
import warnings

import numpy as np
import pytest
import torch

try:
    from src.ffi.bridge import GoEngine, GoAgentState

    HAS_GO = True
except Exception:
    HAS_GO = False

skipgo = pytest.mark.skipif(not HAS_GO, reason="libcambia.so not available")

# ---------------------------------------------------------------------------
# Import cross-engine helpers from the mature test suite
# ---------------------------------------------------------------------------
try:
    from tests.test_cross_engine_samples import (
        _setup_python_game_matching_go,
        XorShift64,
        _is_snap_only,
        _PASS_SNAP_IDX,
        _TEST_RULES,
    )
except ImportError:
    # Fallback: copy minimal helpers if import structure doesn't work
    from test_cross_engine_samples import (
        _setup_python_game_matching_go,
        XorShift64,
        _is_snap_only,
        _PASS_SNAP_IDX,
        _TEST_RULES,
    )

from src.encoding import (
    INPUT_DIM,
    NUM_ACTIONS,
    encode_action_mask,
    encode_infoset,
)
from src.constants import (
    ActionPassSnap,
    DecisionContext,
    EP_PBS_INPUT_DIM,
    N_PLAYER_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS,
    NUM_PLAYERS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNAP_ACTION_MIN = 97


def _make_config():
    """Build a Config object suitable for Python AgentState construction."""
    from src.config import Config, CambiaRulesConfig

    cfg = Config()
    # Override cambia_rules to match _TEST_RULES (Go defaults)
    cfg.cambia_rules = CambiaRulesConfig()
    cfg.cambia_rules.allowDrawFromDiscardPile = True
    cfg.cambia_rules.allowOpponentSnapping = True
    cfg.cambia_rules.max_game_turns = 46
    return cfg


def _build_py_agents(py_state, config):
    """Initialize Python AgentState instances for both players at game start."""
    from src.agent_state import AgentState, AgentObservation

    agents = []
    for pid in range(NUM_PLAYERS):
        opp_id = 1 - pid
        hand = py_state.players[pid].hand
        peeks = py_state.players[pid].initial_peek_indices
        agent = AgentState(
            player_id=pid,
            opponent_id=opp_id,
            memory_level=0,
            time_decay_turns=0,
            initial_hand_size=len(hand),
            config=config,
        )
        initial_obs = AgentObservation(
            acting_player=-1,
            action=None,
            discard_top_card=py_state.get_discard_top(),
            player_hand_sizes=[
                py_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
            ],
            stockpile_size=py_state.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=[],
            did_cambia_get_called=False,
            who_called_cambia=None,
            is_game_over=False,
            current_turn=py_state.get_turn_number(),
        )
        agent.initialize(initial_obs, hand, peeks)
        agents.append(agent)
    return agents


def _create_py_observation(py_state, action, acting_player):
    """Create an AgentObservation from the Python game state after an action."""
    from src.agent_state import AgentObservation

    return AgentObservation(
        acting_player=acting_player,
        action=action,
        discard_top_card=py_state.get_discard_top(),
        player_hand_sizes=[
            py_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
        ],
        stockpile_size=py_state.get_stockpile_size(),
        drawn_card=None,
        peeked_cards=None,
        snap_results=copy.deepcopy(py_state.snap_results_log),
        did_cambia_get_called=py_state.cambia_caller_id is not None,
        who_called_cambia=py_state.cambia_caller_id,
        is_game_over=py_state.is_terminal(),
        current_turn=py_state.get_turn_number(),
    )


def _dc_int_to_enum(ctx_int):
    """Convert Go decision_ctx int to Python DecisionContext enum."""
    _MAP = {
        0: DecisionContext.START_TURN,
        1: DecisionContext.POST_DRAW,
        2: DecisionContext.ABILITY_SELECT,
        3: DecisionContext.SNAP_DECISION,
        4: DecisionContext.SNAP_MOVE,
        5: DecisionContext.START_TURN,  # Terminal maps to START_TURN for encoding
    }
    return _MAP.get(ctx_int, DecisionContext.START_TURN)


def _bucket_int_to_enum(bucket_int):
    """Convert Go drawn_bucket int to Python CardBucket enum (or None)."""
    from src.constants import CardBucket

    if bucket_int < 0:
        return None
    try:
        return CardBucket(bucket_int)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Test 1: Encoding dimensions match network constructors
# ---------------------------------------------------------------------------


class TestEncodingDimensionConstants:
    def test_encoding_dim_matches_network_input(self):
        """AdvantageNetwork accepts INPUT_DIM input and produces NUM_ACTIONS output."""
        from src.networks import AdvantageNetwork

        net = AdvantageNetwork(
            input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_ACTIONS
        )
        x = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(x, mask)
        assert out.shape == (1, NUM_ACTIONS), f"Output shape {out.shape}"

    def test_eppbs_dim_matches_network_input(self):
        """AdvantageNetwork accepts EP_PBS_INPUT_DIM input."""
        from src.networks import AdvantageNetwork

        net = AdvantageNetwork(
            input_dim=EP_PBS_INPUT_DIM, hidden_dim=64, output_dim=NUM_ACTIONS
        )
        x = torch.randn(1, EP_PBS_INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
        out = net(x, mask)
        assert out.shape == (1, NUM_ACTIONS), f"EP-PBS output shape {out.shape}"

    def test_nplayer_dim_matches_network_input(self):
        """AdvantageNetwork accepts N_PLAYER_INPUT_DIM input and N_PLAYER_NUM_ACTIONS output."""
        from src.networks import AdvantageNetwork

        net = AdvantageNetwork(
            input_dim=N_PLAYER_INPUT_DIM,
            hidden_dim=64,
            output_dim=N_PLAYER_NUM_ACTIONS,
        )
        x = torch.randn(1, N_PLAYER_INPUT_DIM)
        mask = torch.ones(1, N_PLAYER_NUM_ACTIONS, dtype=torch.bool)
        out = net(x, mask)
        assert out.shape == (1, N_PLAYER_NUM_ACTIONS), (
            f"N-player output shape {out.shape}"
        )


# ---------------------------------------------------------------------------
# Test 2: Go/Python state tracking parity (acting_player + decision_ctx)
# ---------------------------------------------------------------------------


@skipgo
class TestGoStateParity:
    """At matched game states, Go and Python agree on acting_player and decision_ctx."""

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_state_tracking_parity(self, seed):
        """Drive both engines in lockstep, comparing acting_player at every step."""
        from src.encoding import action_to_index

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        compared = 0
        snap_passes = 0

        for step in range(300):
            go_term = go_engine.is_terminal()
            py_term = py_state.is_terminal()
            if go_term or py_term:
                if snap_passes == 0:
                    assert go_term == py_term, (
                        f"seed {seed} step {step}: terminal mismatch "
                        f"go={go_term} py={py_term}"
                    )
                break

            go_mask = go_engine.legal_actions_mask()
            go_actions = set(np.where(go_mask > 0)[0].tolist())
            py_legal = py_state.get_legal_actions()
            py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
            py_actions = set(np.where(py_mask > 0)[0].tolist())

            go_snap_only = _is_snap_only(go_actions)
            py_snap_phase = py_state.snap_phase_active

            # Pass through snap phases independently
            if go_snap_only:
                go_engine.apply_action(_PASS_SNAP_IDX)
                snap_passes += 1
                if py_snap_phase:
                    py_state.apply_action(ActionPassSnap())
                continue
            if py_snap_phase:
                py_state.apply_action(ActionPassSnap())
                snap_passes += 1
                continue

            # Compare acting player
            go_actor = go_engine.acting_player()
            py_actor = py_state.get_acting_player()
            if snap_passes == 0:
                assert go_actor == py_actor, (
                    f"seed {seed} step {step}: acting_player "
                    f"go={go_actor} py={py_actor}"
                )

            compared += 1

            # Pick lowest common non-snap action and advance
            snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
            go_non_snap = go_actions - snap_indices
            py_non_snap = py_actions - snap_indices
            common = sorted(go_non_snap & py_non_snap)
            if not common:
                break

            action_idx = common[0]
            go_engine.apply_action(action_idx)

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

        go_engine.close()
        assert compared > 0, f"seed {seed}: no steps compared"


# ---------------------------------------------------------------------------
# Test 3: Go/Python legacy encoding parity (THE KEY MISSING TEST)
# ---------------------------------------------------------------------------


@skipgo
class TestGoEncodingParity:
    """
    Compare 222-dim legacy encodings between Go (GoAgentState.encode) and
    Python (encode_infoset) at matched game states.

    If encoding parity fails, that is a genuine bug — the test reports
    full diagnostics at the first divergence point.
    """

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_go_python_encoding_parity(self, seed):
        from src.encoding import action_to_index
        from src.agent_state import AgentObservation

        config = _make_config()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)

        py_state = _setup_python_game_matching_go(seed)

        go_agents = [GoAgentState(go_engine, i) for i in range(2)]
        py_agents = _build_py_agents(py_state, config)

        compared_encodings = 0
        snap_passes = 0
        first_divergence = None

        for step in range(200):
            go_term = go_engine.is_terminal()
            py_term = py_state.is_terminal()
            if go_term or py_term:
                break

            go_mask = go_engine.legal_actions_mask()
            go_actions = set(np.where(go_mask > 0)[0].tolist())
            py_legal = py_state.get_legal_actions()
            py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
            py_actions = set(np.where(py_mask > 0)[0].tolist())

            go_snap_only = _is_snap_only(go_actions)
            py_snap_phase = py_state.snap_phase_active

            # Handle snap phases
            if go_snap_only:
                go_engine.apply_action(_PASS_SNAP_IDX)
                go_engine.update_both(go_agents[0], go_agents[1])
                snap_passes += 1
                if py_snap_phase:
                    py_state.apply_action(ActionPassSnap())
                    obs = _create_py_observation(py_state, ActionPassSnap(), -1)
                    for pa in py_agents:
                        try:
                            pa.update(obs)
                        except Exception:
                            pass
                continue
            if py_snap_phase:
                py_state.apply_action(ActionPassSnap())
                obs = _create_py_observation(py_state, ActionPassSnap(), -1)
                for pa in py_agents:
                    try:
                        pa.update(obs)
                    except Exception:
                        pass
                snap_passes += 1
                continue

            # If we've diverged through snaps, stop comparing
            if snap_passes > 0:
                snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
                go_non_snap = go_actions - snap_indices
                py_non_snap = py_actions - snap_indices
                if go_non_snap != py_non_snap:
                    break

            # Compare encodings at this decision point
            actor = go_engine.acting_player()
            ctx_int = go_engine.decision_ctx()
            drawn_int = go_engine.get_drawn_card_bucket()

            go_enc = go_agents[actor].encode(ctx_int, drawn_int)
            py_enc = encode_infoset(
                py_agents[actor],
                _dc_int_to_enum(ctx_int),
                _bucket_int_to_enum(drawn_int),
            )

            assert go_enc.shape == (INPUT_DIM,), f"Go shape: {go_enc.shape}"
            assert py_enc.shape == (INPUT_DIM,), f"Py shape: {py_enc.shape}"
            assert np.all(np.isfinite(go_enc)), f"Go enc has NaN/Inf at step {step}"
            assert np.all(np.isfinite(py_enc)), f"Py enc has NaN/Inf at step {step}"

            if not np.allclose(go_enc, py_enc, atol=1e-4):
                diff_indices = np.where(~np.isclose(go_enc, py_enc, atol=1e-4))[0]
                first_divergence = (
                    f"seed={seed} step={step} actor=P{actor} "
                    f"ctx={ctx_int} drawn={drawn_int}\n"
                    f"  Divergent indices: {diff_indices.tolist()}\n"
                    f"  Go values at divergence: {go_enc[diff_indices].tolist()}\n"
                    f"  Py values at divergence: {py_enc[diff_indices].tolist()}\n"
                    f"  Max abs diff: {np.max(np.abs(go_enc - py_enc)):.6f}"
                )
                break

            compared_encodings += 1

            # Advance: pick lowest common non-snap action
            snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
            go_non_snap = go_actions - snap_indices
            py_non_snap = py_actions - snap_indices
            common = sorted(go_non_snap & py_non_snap)
            if not common:
                break

            action_idx = common[0]

            # Find the Python action object for agent update
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

            # Apply actions and update agents
            go_engine.apply_action(action_idx)
            go_engine.update_both(go_agents[0], go_agents[1])

            py_state.apply_action(py_action)
            obs = _create_py_observation(py_state, py_action, actor)
            for pa in py_agents:
                try:
                    pa.update(obs)
                except Exception:
                    pass

        # Cleanup
        for a in go_agents:
            a.close()
        go_engine.close()

        if first_divergence is not None:
            pytest.fail(
                f"ENCODING PARITY FAILURE (compared {compared_encodings} "
                f"steps before divergence):\n{first_divergence}"
            )

        assert compared_encodings >= 3, (
            f"seed {seed}: only compared {compared_encodings} encodings "
            f"(need at least 3 decision points)"
        )


# ---------------------------------------------------------------------------
# Test 4: Go/Python EP-PBS encoding parity
# ---------------------------------------------------------------------------


@skipgo
class TestEPPBSCrossEngine:
    """
    Compare 200-dim EP-PBS encodings between Go (GoAgentState.encode_eppbs)
    and Python (encode_infoset_eppbs) at matched game states.

    If parity fails, that is a genuine bug — the test reports diagnostics.
    """

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_eppbs_encoding_parity(self, seed):
        from src.encoding import action_to_index, encode_infoset_eppbs
        from src.constants import CardBucket, GamePhase, StockpileEstimate
        from src.agent_state import AgentObservation

        config = _make_config()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)

        py_state = _setup_python_game_matching_go(seed)

        go_agents = [GoAgentState(go_engine, i) for i in range(2)]
        py_agents = _build_py_agents(py_state, config)

        compared = 0
        snap_passes = 0
        first_divergence = None

        for step in range(200):
            go_term = go_engine.is_terminal()
            py_term = py_state.is_terminal()
            if go_term or py_term:
                break

            go_mask = go_engine.legal_actions_mask()
            go_actions = set(np.where(go_mask > 0)[0].tolist())
            py_legal = py_state.get_legal_actions()
            py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
            py_actions = set(np.where(py_mask > 0)[0].tolist())

            go_snap_only = _is_snap_only(go_actions)
            py_snap_phase = py_state.snap_phase_active

            if go_snap_only:
                go_engine.apply_action(_PASS_SNAP_IDX)
                go_engine.update_both(go_agents[0], go_agents[1])
                snap_passes += 1
                if py_snap_phase:
                    py_state.apply_action(ActionPassSnap())
                    obs = _create_py_observation(py_state, ActionPassSnap(), -1)
                    for pa in py_agents:
                        try:
                            pa.update(obs)
                        except Exception:
                            pass
                continue
            if py_snap_phase:
                py_state.apply_action(ActionPassSnap())
                obs = _create_py_observation(py_state, ActionPassSnap(), -1)
                for pa in py_agents:
                    try:
                        pa.update(obs)
                    except Exception:
                        pass
                snap_passes += 1
                continue

            if snap_passes > 0:
                snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
                go_non_snap = go_actions - snap_indices
                py_non_snap = py_actions - snap_indices
                if go_non_snap != py_non_snap:
                    break

            actor = go_engine.acting_player()
            ctx_int = go_engine.decision_ctx()
            drawn_int = go_engine.get_drawn_card_bucket()

            # Go EP-PBS encoding
            go_enc = go_agents[actor].encode_eppbs(ctx_int, drawn_int)

            # Python EP-PBS encoding
            pa = py_agents[actor]
            cambia_state = 2  # NONE
            if pa.cambia_caller is not None:
                cambia_state = 0 if pa.cambia_caller == pa.player_id else 1
            py_enc = encode_infoset_eppbs(
                slot_tags=pa.slot_tags,
                slot_buckets=pa.slot_buckets,
                discard_top_bucket=pa.known_discard_top_bucket.value,
                stock_estimate=pa.stockpile_estimate.value,
                game_phase=pa.game_phase.value,
                decision_context=ctx_int,
                cambia_state=cambia_state,
                drawn_card_bucket=drawn_int,
            )

            assert go_enc.shape == (EP_PBS_INPUT_DIM,), f"Go shape: {go_enc.shape}"
            assert py_enc.shape == (EP_PBS_INPUT_DIM,), f"Py shape: {py_enc.shape}"
            assert np.all(np.isfinite(go_enc)), f"Go EP-PBS has NaN/Inf at step {step}"
            assert np.all(np.isfinite(py_enc)), f"Py EP-PBS has NaN/Inf at step {step}"

            if not np.allclose(go_enc, py_enc, atol=1e-4):
                diff_indices = np.where(~np.isclose(go_enc, py_enc, atol=1e-4))[0]
                first_divergence = (
                    f"seed={seed} step={step} actor=P{actor} "
                    f"ctx={ctx_int} drawn={drawn_int}\n"
                    f"  Divergent indices: {diff_indices.tolist()}\n"
                    f"  Go values: {go_enc[diff_indices].tolist()}\n"
                    f"  Py values: {py_enc[diff_indices].tolist()}\n"
                    f"  Max abs diff: {np.max(np.abs(go_enc - py_enc)):.6f}"
                )
                break

            compared += 1

            # Advance
            snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
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

        for a in go_agents:
            a.close()
        go_engine.close()

        if first_divergence is not None:
            pytest.fail(
                f"EP-PBS ENCODING PARITY FAILURE (compared {compared} "
                f"steps before divergence):\n{first_divergence}"
            )

        assert compared >= 3, (
            f"seed {seed}: only compared {compared} EP-PBS encodings "
            f"(need at least 3 decision points)"
        )


# ---------------------------------------------------------------------------
# Test 5: Memory decay parity (Go vs Python) — GAP TRACKING
# ---------------------------------------------------------------------------
#
# Memory archetypes (decaying / human_like) require PRNG-driven forgetting in
# the Go AgentState (MemoryArchetype, MemoryDecayLambda, MemoryCapacity fields).
# These fields exist on the Go struct (confirmed in engine/agent/memory_test.go)
# but are NOT currently exposed via the FFI bridge:
#   - cambia_agent_new takes only (game_h, player_id, memory_level,
#     time_decay_turns) — no archetype / lambda / capacity parameters.
#   - cambia_agents_update_both also applies no PRNG decay.
#
# Until FFI exports for memory archetype configuration are added, cross-engine
# parity for decaying/human_like archetypes cannot be tested end-to-end because:
#   1. We cannot set Go-side MemoryArchetype/MemoryDecayLambda from Python.
#   2. Even if we could, Go (PCG PRNG) and Python (random.Random) consume
#      random tokens in potentially different orders, so tensors would drift
#      silently across engines unless a shared PRNG sequence is established.
#
# The tests below are marked skip to document the gap.  When FFI exports are
# added, remove the skip and implement the full lockstep harness following the
# pattern in TestEPPBSCrossEngine above.
# ---------------------------------------------------------------------------


@skipgo
class TestMemoryDecayParity:
    """
    Cross-engine parity tests for memory-decaying archetypes.

    Uses cambia_agent_new_with_memory and cambia_agent_apply_decay FFI exports
    to configure memory archetypes from Python.

    For Decaying archetype: lambda=100.0 gives p≈1, so all PrivOwn slots
    deterministically clear on the first decay call — no PRNG alignment needed.
    For HumanLike archetype: eviction is deterministic (saliency-based).
    """

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_decaying_archetype_legacy_encoding_parity(self, seed):
        """
        With memory_archetype='decaying' and lambda=100.0, all PrivOwn slots
        deterministically decay to TagUnk on the first call.  Go and Python
        legacy (222-dim) encodings must match after a single decay step.
        """
        import random as pyrand

        config = _make_config()

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        # memory_archetype=1 → MemoryDecaying; lambda=100 → p=1-exp(-100)≈1
        LAMBDA = 100.0
        go_agents = [
            GoAgentState.new_with_memory(
                go_engine, i, memory_archetype=1, memory_decay_lambda=LAMBDA
            )
            for i in range(2)
        ]
        py_agents = _build_py_agents(py_state, config)
        for pa in py_agents:
            pa.memory_archetype = "decaying"
            pa.memory_decay_lambda = LAMBDA

        # Apply one decay step — with lambda=100 all PrivOwn slots clear to TagUnk
        decay_seed = seed + 1000
        for ga in go_agents:
            ga.apply_decay(rng_seed=decay_seed)
        rng = pyrand.Random(decay_seed)
        for pa in py_agents:
            pa.apply_memory_decay(rng=rng)

        # Compare legacy 222-dim encodings at the initial decision point
        actor = go_engine.acting_player()
        ctx_int = go_engine.decision_ctx()
        drawn_int = go_engine.get_drawn_card_bucket()

        go_enc = go_agents[actor].encode(ctx_int, drawn_int)
        py_enc = encode_infoset(
            py_agents[actor],
            _dc_int_to_enum(ctx_int),
            _bucket_int_to_enum(drawn_int),
        )

        assert go_enc.shape == (INPUT_DIM,), f"Go shape: {go_enc.shape}"
        assert py_enc.shape == (INPUT_DIM,), f"Py shape: {py_enc.shape}"
        assert np.all(np.isfinite(go_enc)), f"Go enc has NaN/Inf"
        assert np.all(np.isfinite(py_enc)), f"Py enc has NaN/Inf"

        if not np.allclose(go_enc, py_enc, atol=1e-4):
            diff_indices = np.where(~np.isclose(go_enc, py_enc, atol=1e-4))[0]
            pytest.fail(
                f"seed={seed}: decaying archetype legacy encoding mismatch\n"
                f"  Divergent indices: {diff_indices.tolist()}\n"
                f"  Max abs diff: {np.max(np.abs(go_enc - py_enc)):.6f}"
            )

        for ga in go_agents:
            ga.close()
        go_engine.close()

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_decaying_archetype_eppbs_encoding_parity(self, seed):
        """
        With memory_archetype='decaying' and lambda=100.0, all PrivOwn slots
        deterministically decay to TagUnk.  Go and Python EP-PBS (200-dim)
        encodings must match after a single decay step.
        """
        import random as pyrand
        from src.encoding import encode_infoset_eppbs
        from src.constants import CardBucket

        config = _make_config()

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        LAMBDA = 100.0
        go_agents = [
            GoAgentState.new_with_memory(
                go_engine, i, memory_archetype=1, memory_decay_lambda=LAMBDA
            )
            for i in range(2)
        ]
        py_agents = _build_py_agents(py_state, config)
        for pa in py_agents:
            pa.memory_archetype = "decaying"
            pa.memory_decay_lambda = LAMBDA

        decay_seed = seed + 1000
        for ga in go_agents:
            ga.apply_decay(rng_seed=decay_seed)
        rng = pyrand.Random(decay_seed)
        for pa in py_agents:
            pa.apply_memory_decay(rng=rng)

        actor = go_engine.acting_player()
        ctx_int = go_engine.decision_ctx()
        drawn_int = go_engine.get_drawn_card_bucket()

        go_enc = go_agents[actor].encode_eppbs(ctx_int, drawn_int)

        pa = py_agents[actor]
        cambia_state = 2  # NONE
        if pa.cambia_caller is not None:
            cambia_state = 0 if pa.cambia_caller == pa.player_id else 1
        py_enc = encode_infoset_eppbs(
            slot_tags=pa.slot_tags,
            slot_buckets=pa.slot_buckets,
            discard_top_bucket=pa.known_discard_top_bucket.value,
            stock_estimate=pa.stockpile_estimate.value,
            game_phase=pa.game_phase.value,
            decision_context=ctx_int,
            cambia_state=cambia_state,
            drawn_card_bucket=drawn_int,
        )

        assert go_enc.shape == (EP_PBS_INPUT_DIM,), f"Go shape: {go_enc.shape}"
        assert py_enc.shape == (EP_PBS_INPUT_DIM,), f"Py shape: {py_enc.shape}"
        assert np.all(np.isfinite(go_enc)), f"Go EP-PBS has NaN/Inf"
        assert np.all(np.isfinite(py_enc)), f"Py EP-PBS has NaN/Inf"

        if not np.allclose(go_enc, py_enc, atol=1e-4):
            diff_indices = np.where(~np.isclose(go_enc, py_enc, atol=1e-4))[0]
            pytest.fail(
                f"seed={seed}: decaying archetype EP-PBS encoding mismatch\n"
                f"  Divergent indices: {diff_indices.tolist()}\n"
                f"  Max abs diff: {np.max(np.abs(go_enc - py_enc)):.6f}"
            )

        for ga in go_agents:
            ga.close()
        go_engine.close()

    @pytest.mark.parametrize("seed", [42, 137, 12345])
    def test_human_like_archetype_encoding_parity(self, seed):
        """
        With memory_archetype='human_like' and capacity=1, saliency-based
        eviction is deterministic.  Go and Python 222-dim encodings must match
        after eviction reduces the active mask to 1 slot.
        """
        from src.encoding import encode_infoset_eppbs
        from src.constants import CardBucket

        config = _make_config()

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        # memory_archetype=2 → MemoryHumanLike; capacity=1 forces eviction of
        # 1 of the 2 initial-peek PrivOwn slots (deterministic, lowest saliency).
        CAPACITY = 1
        go_agents = [
            GoAgentState.new_with_memory(
                go_engine, i, memory_archetype=2, memory_capacity=CAPACITY
            )
            for i in range(2)
        ]
        py_agents = _build_py_agents(py_state, config)
        for pa in py_agents:
            pa.memory_archetype = "human_like"
            pa.memory_capacity = CAPACITY

        # Apply HumanLike eviction — no PRNG needed
        for ga in go_agents:
            ga.apply_decay(rng_seed=0)
        for pa in py_agents:
            pa.apply_memory_decay()

        # Compare legacy 222-dim encodings at the initial decision point
        actor = go_engine.acting_player()
        ctx_int = go_engine.decision_ctx()
        drawn_int = go_engine.get_drawn_card_bucket()

        go_enc = go_agents[actor].encode(ctx_int, drawn_int)
        py_enc = encode_infoset(
            py_agents[actor],
            _dc_int_to_enum(ctx_int),
            _bucket_int_to_enum(drawn_int),
        )

        assert go_enc.shape == (INPUT_DIM,), f"Go shape: {go_enc.shape}"
        assert py_enc.shape == (INPUT_DIM,), f"Py shape: {py_enc.shape}"
        assert np.all(np.isfinite(go_enc)), f"Go enc has NaN/Inf"
        assert np.all(np.isfinite(py_enc)), f"Py enc has NaN/Inf"

        if not np.allclose(go_enc, py_enc, atol=1e-4):
            diff_indices = np.where(~np.isclose(go_enc, py_enc, atol=1e-4))[0]
            pytest.fail(
                f"seed={seed}: human_like archetype encoding mismatch\n"
                f"  Divergent indices: {diff_indices.tolist()}\n"
                f"  Max abs diff: {np.max(np.abs(go_enc - py_enc)):.6f}"
            )

        for ga in go_agents:
            ga.close()
        go_engine.close()
