"""
tests/test_deep_worker_go.py

Tests for the Go engine backend integration in the Deep CFR worker (Task 18).

Requires libcambia.so to be built and available (skipped otherwise).
"""

import pytest
import numpy as np
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Skip guard: skip all tests if libcambia.so is unavailable
# ---------------------------------------------------------------------------


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine  # noqa: PLC0415

        e = GoEngine(seed=0)
        e.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def _real_deep_cfr_config():
    """Return the real DeepCfrConfig, bypassing the conftest stub."""
    import importlib
    import sys

    _orig = sys.modules.pop("src.config", None)
    try:
        return importlib.import_module("src.config").DeepCfrConfig
    finally:
        if _orig is not None:
            sys.modules["src.config"] = _orig


class TestGoBackendConfig:
    def test_real_config_defaults_to_go(self):
        """Real DeepCfrConfig defaults engine_backend to 'go' (cambia-1783)."""
        assert _real_deep_cfr_config()().engine_backend == "go"

    def test_real_config_go_backend(self):
        """Real DeepCfrConfig accepts an explicit 'go'."""
        cfg = _real_deep_cfr_config()(engine_backend="go")
        assert cfg.engine_backend == "go"

    def test_real_config_refuses_python_backend(self):
        """The retired Python backend is refused with a named message."""
        with pytest.raises(Exception) as excinfo:
            _real_deep_cfr_config()(engine_backend="python")
        message = str(excinfo.value)
        assert "Python" in message and "retired" in message
        assert "go" in message


# ---------------------------------------------------------------------------
# _infer_decision_context tests
# ---------------------------------------------------------------------------


class TestInferDecisionContext:
    def test_start_turn_draw_stockpile(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[0] = 1  # DrawStockpile
        assert _infer_decision_context(mask) == 0

    def test_start_turn_draw_discard(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[1] = 1  # DrawDiscard
        assert _infer_decision_context(mask) == 0

    def test_start_turn_call_cambia(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[2] = 1  # CallCambia
        assert _infer_decision_context(mask) == 0

    def test_post_draw_discard_no_ability(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[3] = 1  # DiscardNoAbility
        assert _infer_decision_context(mask) == 1

    def test_post_draw_discard_with_ability(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[4] = 1  # DiscardWithAbility
        assert _infer_decision_context(mask) == 1

    def test_post_draw_replace(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[7] = 1  # ActionReplace(2)
        assert _infer_decision_context(mask) == 1

    def test_ability_select_peek_own(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[11] = 1  # ActionPeekOwn(0)
        assert _infer_decision_context(mask) == 3

    def test_ability_select_peek_other(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[17] = 1  # ActionPeekOther(0)
        assert _infer_decision_context(mask) == 3

    def test_ability_select_blind_swap(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[23] = 1  # ActionBlindSwap
        assert _infer_decision_context(mask) == 3

    def test_ability_select_king_swap(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[95] = 1  # ActionKingSwapNo
        assert _infer_decision_context(mask) == 3

    def test_snap_decision_pass(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[97] = 1  # PassSnap
        assert _infer_decision_context(mask) == 2

    def test_snap_decision_own(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[98] = 1  # SnapOwn(0)
        assert _infer_decision_context(mask) == 2

    def test_snap_move(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        mask[110] = 1  # SnapOpponentMove
        assert _infer_decision_context(mask) == 4

    def test_empty_mask_fallback(self):
        from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415

        mask = np.zeros(146, dtype=np.uint8)
        assert _infer_decision_context(mask) == 0  # fallback


# ---------------------------------------------------------------------------
# Decision-context numbering pin (cambia-1688)
#
# _infer_decision_context used to return the SnapDecision and AbilitySelect
# ints reversed relative to engine/types.go (2=CtxSnapDecision,
# 3=CtxAbilitySelect there; this function returned 2 for AbilitySelect and 3
# for SnapDecision). cambia-1484 fixed the same swap in
# tests/test_cross_validation.py's engine-side map; this pins the
# mask-inference path the same way, against bridge.DecisionCtx rather than a
# bare literal, so a future engine renumbering fails here instead of silently
# re-encoding every ability and snap node under the wrong one-hot.
# ---------------------------------------------------------------------------


#: Engine action index band per decision context, half-open, matching the
#: engine/types.go action index constants used by _infer_decision_context's
#: docstring: StartTurn 0-2, PostDraw 3-10, AbilitySelect 11-96, SnapDecision
#: 97-109, SnapMove 110-145.
_INFER_CTX_ACTION_BANDS = {
    "START_TURN": (0, 3),
    "POST_DRAW": (3, 11),
    "ABILITY_SELECT": (11, 97),
    "SNAP_DECISION": (97, 110),
    "SNAP_MOVE": (110, 146),
}


def test_infer_decision_context_bands_match_bridge_decision_ctx():
    """Each action band resolves to the DecisionCtx value the engine assigns.

    Sets one legal action per band and asserts _infer_decision_context returns
    DecisionCtx[name].value rather than a hand-copied int, so this fails if the
    SnapDecision/AbilitySelect swap (cambia-1688) or any other reordering of
    engine/types.go's DecisionContext reappears.
    """
    from src.cfr.deep_worker import _infer_decision_context  # noqa: PLC0415
    from src.ffi.bridge import DecisionCtx  # noqa: PLC0415

    for name, (low, _high) in _INFER_CTX_ACTION_BANDS.items():
        mask = np.zeros(146, dtype=np.uint8)
        mask[low] = 1
        got = _infer_decision_context(mask)
        want = DecisionCtx[name].value
        assert got == want, (
            f"action index {low} (start of the {name} band) resolved to "
            f"decision context {got}, expected {want} (DecisionCtx.{name})"
        )


# ---------------------------------------------------------------------------
# Go traversal tests (require libcambia.so)
# ---------------------------------------------------------------------------


# Marked slow at the class, not per method: these drive _deep_traverse_go, which
# walks the whole game tree, and four of them measured 41.8s of this module's
# 42.0s (cambia-1920). The mark keeps them out of the pull-request run and puts
# them in ci.yml's nightly cfr-slow job; a traversal test added here inherits
# that placement instead of quietly costing every pull request minutes.
@pytest.mark.slow
@skip_if_no_go
class TestGoTraversal:
    def _make_config(self, recursion_limit: int = 12) -> SimpleNamespace:
        """Build a minimal config namespace suitable for Go backend traversal tests."""
        config = SimpleNamespace()
        config.deep_cfr = SimpleNamespace(engine_backend="go")
        config.system = SimpleNamespace(recursion_limit=recursion_limit)
        config.agent_params = SimpleNamespace(memory_level=0, time_decay_turns=0)
        config.cambia_rules = SimpleNamespace(
            allowDrawFromDiscardPile=False,
            allowReplaceAbilities=False,
            snapRace=False,
            penaltyDrawCount=2,
            use_jokers=2,
            cards_per_player=4,
            initial_view_count=2,
            cambia_allowed_round=0,
            allowOpponentSnapping=False,
            max_game_turns=300,
            lockCallerHand=True,
        )
        config.logging = SimpleNamespace(
            log_max_bytes=1024 * 1024,
            log_backup_count=1,
            log_file_prefix="test",
            log_archive_enabled=False,
            log_archive_max_archives=0,
            log_archive_dir="",
            log_size_update_interval_sec=60,
            log_simulation_traces=False,
            simulation_trace_filename_prefix="sim",
            get_worker_log_level=lambda wid, ntotal: "WARNING",
        )
        config.cfr_training = SimpleNamespace(num_workers=1)
        return config

    def test_go_traversal_produces_samples(self):
        """Run a short Go-backend traversal and verify it produces valid samples."""
        from src.cfr.deep_worker import _deep_traverse_go  # noqa: PLC0415
        from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        from src.networks import AdvantageNetwork  # noqa: PLC0415
        from src.encoding import INPUT_DIM, NUM_ACTIONS  # noqa: PLC0415
        from src.reservoir import ReservoirSample  # noqa: PLC0415
        from src.utils import WorkerStats, SimulationNodeData  # noqa: PLC0415

        config = self._make_config()

        network = AdvantageNetwork(INPUT_DIM, 64, NUM_ACTIONS)

        engine = GoEngine(seed=42)
        agents = [GoAgentState(engine, pid) for pid in range(2)]

        advantage_samples = []
        strategy_samples = []
        stats = WorkerStats()

        try:
            utility = _deep_traverse_go(
                engine=engine,
                agent_states=agents,
                updating_player=0,
                network=network,
                iteration=1,
                config=config,
                advantage_samples=advantage_samples,
                strategy_samples=strategy_samples,
                depth=0,
                worker_stats=stats,
                progress_queue=None,
                worker_id=0,
                min_depth_after_bottom_out_tracker=[float("inf")],
                has_bottomed_out_tracker=[False],
                simulation_nodes=[],
            )

            assert isinstance(utility, np.ndarray)
            assert utility.shape == (2,)
            assert len(advantage_samples) > 0
            assert stats.nodes_visited > 0

            # --- Behavioral assertion: 2P zero-sum utility ---
            # Score-margin utility: continuous in [-1, 1], zero-sum.
            assert (
                abs(utility[0] + utility[1]) < 1e-5
            ), f"2P utility not zero-sum: {utility}"

            # --- Behavioral assertion: utility in valid range ---
            # Score-margin utilities are clamped to [-1, 1].
            for i in range(2):
                assert (
                    -1.01 <= utility[i] <= 1.01
                ), f"Utility[{i}]={utility[i]} outside [-1, 1]"

            # --- Behavioral assertion: strategy samples sum to ~1.0 ---
            # In ES-MCCFR, opponent nodes store σ(I) as strategy targets.
            # Since σ is a probability distribution over legal actions,
            # the sum over the legal (masked) entries must equal 1.0.
            for sample in strategy_samples:
                mask = sample.action_mask
                strategy_sum = sample.target[mask].sum()
                assert (
                    abs(strategy_sum - 1.0) < 0.01
                ), f"Strategy sample sum {strategy_sum} != 1.0"

            # --- Behavioral assertion: regret mask consistency ---
            # Regret targets must be zero for illegal actions. The traversal
            # only writes regrets at legal action indices; the target vector
            # is zero-initialized, so illegal slots must remain zero.
            for sample in advantage_samples:
                illegal_mask = ~sample.action_mask
                assert (
                    sample.target[illegal_mask] == 0
                ).all(), "Non-zero regret found for illegal action"
        finally:
            for a in agents:
                a.close()
            engine.close()

    def test_go_traversal_samples_valid_dimensions(self):
        """Verify samples from Go traversal have correct tensor dimensions."""
        from src.cfr.deep_worker import _deep_traverse_go  # noqa: PLC0415
        from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        from src.encoding import INPUT_DIM, NUM_ACTIONS  # noqa: PLC0415
        from src.utils import WorkerStats  # noqa: PLC0415

        config = self._make_config()

        engine = GoEngine(seed=7)
        agents = [GoAgentState(engine, pid) for pid in range(2)]

        advantage_samples = []
        strategy_samples = []
        stats = WorkerStats()

        try:
            _deep_traverse_go(
                engine=engine,
                agent_states=agents,
                updating_player=0,
                network=None,
                iteration=1,
                config=config,
                advantage_samples=advantage_samples,
                strategy_samples=strategy_samples,
                depth=0,
                worker_stats=stats,
                progress_queue=None,
                worker_id=0,
                min_depth_after_bottom_out_tracker=[float("inf")],
                has_bottomed_out_tracker=[False],
                simulation_nodes=[],
            )

            for sample in advantage_samples:
                assert sample.features.shape == (
                    INPUT_DIM,
                ), f"Expected ({INPUT_DIM},), got {sample.features.shape}"
                assert sample.target.shape == (
                    NUM_ACTIONS,
                ), f"Expected ({NUM_ACTIONS},), got {sample.target.shape}"
                assert sample.action_mask.shape == (
                    NUM_ACTIONS,
                ), f"Expected ({NUM_ACTIONS},), got {sample.action_mask.shape}"
        finally:
            for a in agents:
                a.close()
            engine.close()

    def test_go_traversal_multiple_seeds(self):
        """Run Go traversals with different seeds to check robustness."""
        from src.cfr.deep_worker import _deep_traverse_go  # noqa: PLC0415
        from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        from src.utils import WorkerStats  # noqa: PLC0415

        config = self._make_config()

        for seed in [1, 42, 100, 999]:
            engine = GoEngine(seed=seed)
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            advantage_samples = []
            strategy_samples = []
            stats = WorkerStats()

            try:
                utility = _deep_traverse_go(
                    engine=engine,
                    agent_states=agents,
                    updating_player=seed % 2,
                    network=None,
                    iteration=1,
                    config=config,
                    advantage_samples=advantage_samples,
                    strategy_samples=strategy_samples,
                    depth=0,
                    worker_stats=stats,
                    progress_queue=None,
                    worker_id=0,
                    min_depth_after_bottom_out_tracker=[float("inf")],
                    has_bottomed_out_tracker=[False],
                    simulation_nodes=[],
                )
                assert utility.shape == (2,), f"seed={seed}: wrong utility shape"
                assert len(advantage_samples) > 0, f"seed={seed}: no advantage samples"

                # 2P zero-sum: u0 + u1 = 0 exactly
                assert (
                    abs(utility[0] + utility[1]) < 1e-9
                ), f"seed={seed}: 2P utility not zero-sum: {utility}"
                # Utility bounded in [-1, 1]
                for i in range(2):
                    assert (
                        -1.01 <= utility[i] <= 1.01
                    ), f"seed={seed}: utility[{i}]={utility[i]} out of range"
            finally:
                for a in agents:
                    a.close()
                engine.close()

    def test_go_traversal_both_updating_players(self):
        """Verify traversal works for both updating_player=0 and updating_player=1."""
        from src.cfr.deep_worker import _deep_traverse_go  # noqa: PLC0415
        from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        from src.utils import WorkerStats  # noqa: PLC0415

        config = self._make_config()

        for updating_player in [0, 1]:
            engine = GoEngine(seed=123)
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            advantage_samples = []
            strategy_samples = []
            stats = WorkerStats()

            try:
                utility = _deep_traverse_go(
                    engine=engine,
                    agent_states=agents,
                    updating_player=updating_player,
                    network=None,
                    iteration=1,
                    config=config,
                    advantage_samples=advantage_samples,
                    strategy_samples=strategy_samples,
                    depth=0,
                    worker_stats=stats,
                    progress_queue=None,
                    worker_id=0,
                    min_depth_after_bottom_out_tracker=[float("inf")],
                    has_bottomed_out_tracker=[False],
                    simulation_nodes=[],
                )
                assert utility.shape == (2,)
                assert len(advantage_samples) > 0
            finally:
                for a in agents:
                    a.close()
                engine.close()

    def test_run_deep_cfr_worker_go_backend(self):
        """Integration test: run_deep_cfr_worker with engine_backend='go' returns result."""
        from src.cfr.deep_worker import run_deep_cfr_worker  # noqa: PLC0415
        from src.encoding import INPUT_DIM, NUM_ACTIONS  # noqa: PLC0415

        config = self._make_config(recursion_limit=12)

        worker_args = (
            1,  # iteration
            config,
            None,  # no network weights
            {"input_dim": INPUT_DIM, "hidden_dim": 64, "output_dim": NUM_ACTIONS},
            None,  # no progress_queue
            None,  # no archive_queue
            0,  # worker_id
            "/tmp/test_go_worker_logs",  # run_log_dir
            "test",  # run_timestamp
        )

        result = run_deep_cfr_worker(worker_args)

        assert result is not None
        # With Go backend, at least some samples should be produced (game goes to terminal)
        # The worker should not crash
        assert result.stats.nodes_visited > 0

    def test_os_traversal_behavioral_assertions(self):
        """OS-MCCFR traversal: strategy sums, regret masks, utility bounds."""
        from src.cfr.deep_worker import _deep_traverse_os_go  # noqa: PLC0415
        from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        from src.encoding import INPUT_DIM, NUM_ACTIONS  # noqa: PLC0415
        from src.utils import WorkerStats  # noqa: PLC0415

        config = self._make_config()

        for seed in [42, 77, 200]:
            engine = GoEngine(seed=seed)
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            advantage_samples = []
            strategy_samples = []
            stats = WorkerStats()

            try:
                utility = _deep_traverse_os_go(
                    engine=engine,
                    agent_states=agents,
                    updating_player=0,
                    network=None,
                    iteration=1,
                    config=config,
                    advantage_samples=advantage_samples,
                    strategy_samples=strategy_samples,
                    depth=0,
                    worker_stats=stats,
                    progress_queue=None,
                    worker_id=0,
                    min_depth_after_bottom_out_tracker=[float("inf")],
                    has_bottomed_out_tracker=[False],
                    simulation_nodes=[],
                    exploration_epsilon=0.6,
                )

                # --- 2P zero-sum: u0 + u1 = 0 exactly ---
                # Cambia 2P utilities are {+1,-1}, {-1,+1}, or {0,0}.
                assert (
                    abs(utility[0] + utility[1]) < 1e-9
                ), f"seed={seed}: OS utility not zero-sum: {utility}"

                # --- Utility range: each u_i in [-1, 1] ---
                for i in range(2):
                    assert (
                        -1.01 <= utility[i] <= 1.01
                    ), f"seed={seed}: OS utility[{i}]={utility[i]} out of range"

                # --- Strategy samples: probability distribution sums to 1 ---
                # OS-MCCFR stores σ(I) at opponent nodes. The strategy over
                # legal actions is a valid probability distribution.
                for sample in strategy_samples:
                    mask = sample.action_mask
                    strategy_sum = sample.target[mask].sum()
                    assert (
                        abs(strategy_sum - 1.0) < 0.01
                    ), f"seed={seed}: OS strategy sum {strategy_sum} != 1.0"

                # --- Regret mask consistency ---
                # Regret targets are zero-initialized; only legal action slots
                # are written. Illegal action slots must remain exactly zero.
                for sample in advantage_samples:
                    illegal_mask = ~sample.action_mask
                    assert (
                        sample.target[illegal_mask] == 0
                    ).all(), f"seed={seed}: Non-zero regret for illegal action"
            finally:
                for a in agents:
                    a.close()
                engine.close()
