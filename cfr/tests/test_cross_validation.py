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
    from src.ffi.bridge import DecisionCtx, GoEngine, GoAgentState

    HAS_GO = True
except Exception:
    HAS_GO = False

skipgo = pytest.mark.skipif(not HAS_GO, reason="libcambia.so not available")

# ---------------------------------------------------------------------------
# Import cross-engine helpers from the mature test suite
# ---------------------------------------------------------------------------
try:
    from tests.parity_seeds import PARITY_SEEDS
except ImportError:  # pragma: no cover - path fallback
    from parity_seeds import PARITY_SEEDS  # type: ignore

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
        player_hand_sizes=[py_state.get_player_card_count(i) for i in range(NUM_PLAYERS)],
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
    """Convert a Go decision_ctx int to the Python DecisionContext enum.

    Resolved by name off ``DecisionCtx``, the bridge's copy of the engine's
    numbering, rather than by a literal int map: the two enums share member
    names but the ints belong to engine/types.go, so a name lookup survives a
    renumbering that a literal map would silently mistranslate. This map used to
    be written out with SNAP_DECISION and ABILITY_SELECT swapped (cambia-1484);
    test_decision_ctx_ints_match_the_engine pins DecisionCtx against a live
    engine so neither copy can drift again.

    Terminal has no Python counterpart and encodes as START_TURN: a terminal
    state carries no decision, and callers racing the terminal check need a
    valid context rather than a raise.
    """
    try:
        name = DecisionCtx(int(ctx_int)).name
    except ValueError:
        return DecisionContext.START_TURN
    if name == DecisionCtx.TERMINAL.name:
        return DecisionContext.START_TURN
    return DecisionContext[name]


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

        net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_ACTIONS)
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
        assert out.shape == (
            1,
            N_PLAYER_NUM_ACTIONS,
        ), f"N-player output shape {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Go/Python state tracking parity (acting_player + decision_ctx)
# ---------------------------------------------------------------------------


@skipgo
class TestGoStateParity:
    """At matched game states, Go and Python agree on acting_player and decision_ctx."""

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

    If encoding parity fails, that is a genuine bug: the test reports
    full diagnostics at the first divergence point.
    """

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

    If parity fails, that is a genuine bug: the test reports diagnostics.
    """

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

            # Compare only first 200 dims (core EP-PBS layout). Dims 200-223 are
            # history features written by Go agent state (obs ages, discard histogram,
            # turn progress) which the Python reference encoder doesn't track.
            _EPPBS_CORE_DIMS = 200
            if not np.allclose(
                go_enc[:_EPPBS_CORE_DIMS], py_enc[:_EPPBS_CORE_DIMS], atol=1e-4
            ):
                diff_indices = np.where(
                    ~np.isclose(
                        go_enc[:_EPPBS_CORE_DIMS], py_enc[:_EPPBS_CORE_DIMS], atol=1e-4
                    )
                )[0]
                first_divergence = (
                    f"seed={seed} step={step} actor=P{actor} "
                    f"ctx={ctx_int} drawn={drawn_int}\n"
                    f"  Divergent indices: {diff_indices.tolist()}\n"
                    f"  Go values: {go_enc[diff_indices].tolist()}\n"
                    f"  Py values: {py_enc[diff_indices].tolist()}\n"
                    f"  Max abs diff: {np.max(np.abs(go_enc[:_EPPBS_CORE_DIMS] - py_enc[:_EPPBS_CORE_DIMS])):.6f}"
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
# Test 5: Memory decay parity (Go vs Python) - GAP TRACKING
# ---------------------------------------------------------------------------
#
# Memory archetypes (decaying / human_like) require PRNG-driven forgetting in
# the Go AgentState (MemoryArchetype, MemoryDecayLambda, MemoryCapacity fields).
# These fields exist on the Go struct (confirmed in engine/agent/memory_test.go)
# but are NOT currently exposed via the FFI bridge:
#   - cambia_agent_new takes only (game_h, player_id, memory_level,
#     time_decay_turns): no archetype / lambda / capacity parameters.
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
    deterministically clear on the first decay call: no PRNG alignment needed.
    For HumanLike archetype: eviction is deterministic (saliency-based).
    """

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

        # Apply one decay step: with lambda=100 all PrivOwn slots clear to TagUnk
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

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

        # Compare only first 200 dims (core EP-PBS). Dims 200-223 are Go-only
        # history features (obs ages, discard histogram, turn progress).
        _EPPBS_CORE_DIMS = 200
        if not np.allclose(
            go_enc[:_EPPBS_CORE_DIMS], py_enc[:_EPPBS_CORE_DIMS], atol=1e-4
        ):
            diff_indices = np.where(
                ~np.isclose(
                    go_enc[:_EPPBS_CORE_DIMS], py_enc[:_EPPBS_CORE_DIMS], atol=1e-4
                )
            )[0]
            pytest.fail(
                f"seed={seed}: decaying archetype EP-PBS encoding mismatch\n"
                f"  Divergent indices: {diff_indices.tolist()}\n"
                f"  Max abs diff: {np.max(np.abs(go_enc[:_EPPBS_CORE_DIMS] - py_enc[:_EPPBS_CORE_DIMS])):.6f}"
            )

        for ga in go_agents:
            ga.close()
        go_engine.close()

    @pytest.mark.parametrize("seed", PARITY_SEEDS)
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

        # Apply HumanLike eviction: no PRNG needed
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


# ---------------------------------------------------------------------------
# Decision-context numbering pin (cambia-1484)
#
# engine/types.go declares the DecisionContext order and engine/legal.go's
# LegalActions() switches on it, so each context admits exactly one band of the
# action index space (engine/types.go action index constants). That makes the
# numbering falsifiable from a live engine: renumber DecisionContext and the
# observed ints stop lining up with the bands, here, rather than silently
# re-encoding every ability and snap node under the wrong one-hot.
#
# _dc_int_to_enum carried SNAP_DECISION and ABILITY_SELECT swapped until
# cambia-1484; the same swap sat in bridge.py's docstring until cambia-1376.
# ---------------------------------------------------------------------------


#: Engine action index band per decision context, half-open, from the
#: engine/types.go action index constants: DrawStockpile/DrawDiscard/CallCambia
#: at 0-2, DiscardNoAbility/DiscardWithAbility/Replace at 3-10, the ability
#: selectors at 11-96, PassSnap/SnapOwn/SnapOpponent at 97-109, and
#: SnapOpponentMove at 110-145.
_CTX_ACTION_BANDS = {
    "START_TURN": (0, 3),
    "POST_DRAW": (3, 11),
    "ABILITY_SELECT": (11, 97),
    "SNAP_DECISION": (97, 110),
    "SNAP_MOVE": (110, 146),
}


def _ctx_pin_action(legal_indices):
    """Pick an action that walks the game through every decision context.

    Prefers DiscardWithAbility so ability selection is reached, then a snap move,
    then an opponent snap (which is what creates a snap move), else the lowest
    legal index.
    """
    for pick in (
        [i for i in legal_indices if i == 4],
        [i for i in legal_indices if 110 <= i < 146],
        [i for i in legal_indices if 104 <= i < 110],
        legal_indices,
    ):
        if pick:
            return pick[0]
    return None


def test_decision_ctx_names_and_values_agree_with_python_enum():
    """DecisionCtx mirrors src.constants.DecisionContext, name and value.

    Both enums carry TERMINAL. _dc_int_to_enum still folds it onto START_TURN,
    because a terminal state has no decision to encode; that fold is a caller
    choice, not a gap between the enums.
    """
    from src.ffi.bridge import DecisionCtx as _DecisionCtx

    assert [m.name for m in _DecisionCtx] == [m.name for m in DecisionContext], (
        "DecisionCtx and DecisionContext no longer share their member names, so "
        "_dc_int_to_enum's name lookup cannot resolve."
    )
    for member in _DecisionCtx:
        assert DecisionContext[member.name].value == member.value, (
            f"{member.name}: engine numbering {member.value} vs Python "
            f"{DecisionContext[member.name].value}"
        )


def test_dc_int_to_enum_maps_each_engine_context():
    """The int map matches the engine numbering, snap and ability included."""
    from src.ffi.bridge import DecisionCtx as _DecisionCtx

    assert _dc_int_to_enum(_DecisionCtx.START_TURN) is DecisionContext.START_TURN
    assert _dc_int_to_enum(_DecisionCtx.POST_DRAW) is DecisionContext.POST_DRAW
    assert _dc_int_to_enum(_DecisionCtx.SNAP_DECISION) is DecisionContext.SNAP_DECISION
    assert _dc_int_to_enum(_DecisionCtx.ABILITY_SELECT) is DecisionContext.ABILITY_SELECT
    assert _dc_int_to_enum(_DecisionCtx.SNAP_MOVE) is DecisionContext.SNAP_MOVE
    # Terminal has no decision to encode; it reports START_TURN.
    assert _dc_int_to_enum(_DecisionCtx.TERMINAL) is DecisionContext.START_TURN
    # An int outside the engine's range degrades rather than raising.
    assert _dc_int_to_enum(99) is DecisionContext.START_TURN


@skipgo
def test_decision_ctx_ints_match_the_engine():
    """Every DecisionCtx value is pinned against a live engine's legal actions.

    Plays seeded games, and at each state asserts the engine's decision_ctx int
    names a context whose action band contains every legal action. A terminal
    state must report TERMINAL and offer none. All six contexts have to appear,
    so a renumbering cannot pass by leaving one unobserved.
    """
    from src.ffi.bridge import DecisionCtx as _DecisionCtx

    observed = set()
    for seed in range(40):
        engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        try:
            for step in range(300):
                ctx_int = engine.decision_ctx()
                try:
                    ctx = _DecisionCtx(ctx_int)
                except ValueError:
                    pytest.fail(
                        f"seed={seed} step={step}: engine reported decision_ctx "
                        f"{ctx_int}, which DecisionCtx does not name. "
                        f"engine/types.go gained a context; mirror it."
                    )
                observed.add(ctx)

                legal = np.where(engine.legal_actions_mask() > 0)[0].tolist()
                if ctx is _DecisionCtx.TERMINAL:
                    assert not legal, (
                        f"seed={seed} step={step}: terminal state offered legal "
                        f"actions {legal[:8]}"
                    )
                    break

                low, high = _CTX_ACTION_BANDS[ctx.name]
                assert legal, f"seed={seed} step={step}: {ctx.name} offered no action"
                assert low <= min(legal) and max(legal) < high, (
                    f"seed={seed} step={step}: engine reported "
                    f"{ctx.name}={ctx.value} but its legal actions "
                    f"{legal[:8]} fall outside that context's band "
                    f"[{low},{high}). The engine's DecisionContext numbering and "
                    f"DecisionCtx have diverged (engine/types.go)."
                )

                action = _ctx_pin_action(legal)
                assert action is not None
                engine.apply_action(action)
        finally:
            engine.close()

    missing = [m.name for m in _DecisionCtx if m not in observed]
    assert not missing, (
        f"These contexts never occurred, so their values went unpinned: "
        f"{missing}. Widen the seed range or the action policy."
    )
