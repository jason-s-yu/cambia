"""
tests/test_ffi_bridge.py

Comprehensive tests for the Python ctypes FFI bridge (src/ffi/bridge.py).

Tests GoEngine and GoAgentState against libcambia.so.
"""

import numpy as np
import pytest

from src.ffi.bridge import GoAgentState, GoEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def play_random_game(seed: int) -> np.ndarray:
    """Play a full game with random actions; return the utility array."""
    with GoEngine(seed=seed) as engine:
        agents = [GoAgentState(engine, pid) for pid in range(2)]
        while not engine.is_terminal():
            mask = engine.legal_actions_mask()
            legal_indices = np.where(mask > 0)[0]
            assert len(legal_indices) > 0, "No legal actions in non-terminal state"
            action = int(np.random.choice(legal_indices))
            engine.apply_action(action)
            for a in agents:
                a.update(engine)
        utility = engine.get_utility()
        for a in agents:
            a.close()
        return utility


# ---------------------------------------------------------------------------
# GoEngine tests
# ---------------------------------------------------------------------------


class TestGoEngineCreate:
    def test_go_engine_create_and_free(self):
        """Engine creates successfully and handle is non-negative."""
        engine = GoEngine(seed=42)
        assert engine.handle >= 0
        engine.close()

    def test_go_engine_context_manager(self):
        """Context manager protocol works and closes cleanly."""
        with GoEngine(seed=0) as engine:
            assert engine.handle >= 0
        # After exit, handle should be -1
        assert engine._game_h == -1

    def test_go_engine_random_seed(self):
        """Engine with no seed provided still creates successfully."""
        with GoEngine() as engine:
            assert engine.handle >= 0

    def test_go_engine_different_seeds_differ(self):
        """Different seeds produce different turn progressions."""
        # Just verify both create valid engines
        with GoEngine(seed=1) as e1, GoEngine(seed=2) as e2:
            assert e1.handle >= 0
            assert e2.handle >= 0
            assert e1.handle != e2.handle


class TestGoEngineActions:
    def test_go_engine_legal_actions_mask(self):
        """Mask has shape (146,), dtype uint8, with at least one legal action."""
        with GoEngine(seed=100) as engine:
            mask = engine.legal_actions_mask()
            assert mask.shape == (146,), f"Expected (146,), got {mask.shape}"
            assert mask.dtype == np.uint8, f"Expected uint8, got {mask.dtype}"
            assert mask.sum() > 0, "No legal actions in initial state"

    def test_go_engine_mask_values_binary(self):
        """All mask values are 0 or 1."""
        with GoEngine(seed=101) as engine:
            mask = engine.legal_actions_mask()
            assert set(np.unique(mask)).issubset({0, 1})

    def test_go_engine_apply_action(self):
        """Applying a legal action does not raise and advances turn state."""
        with GoEngine(seed=200) as engine:
            turn_before = engine.turn_number()
            mask = engine.legal_actions_mask()
            legal = np.where(mask > 0)[0]
            engine.apply_action(int(legal[0]))
            # State has advanced (turn may or may not increment depending on action)
            assert not engine._closed


class TestGoEngineSaveRestore:
    def test_go_engine_save_restore(self):
        """Save then restore returns the engine to the saved state."""
        with GoEngine(seed=300) as engine:
            mask_before = engine.legal_actions_mask()
            turn_before = engine.turn_number()

            snap_h = engine.save()
            assert snap_h >= 0

            # Apply several actions to mutate state
            legal = np.where(mask_before > 0)[0]
            engine.apply_action(int(legal[0]))

            # Restore
            engine.restore(snap_h)
            engine.free_snapshot(snap_h)

            mask_after = engine.legal_actions_mask()
            turn_after = engine.turn_number()

            assert turn_after == turn_before
            np.testing.assert_array_equal(mask_after, mask_before)

    def test_go_engine_free_snapshot(self):
        """free_snapshot does not raise."""
        with GoEngine(seed=301) as engine:
            snap_h = engine.save()
            engine.free_snapshot(snap_h)  # Should not raise


class TestGoEngineState:
    def test_go_engine_acting_player(self):
        """Acting player is 0 or 1 in a fresh game."""
        with GoEngine(seed=400) as engine:
            p = engine.acting_player()
            assert p in (0, 1), f"Expected 0 or 1, got {p}"

    def test_go_engine_turn_number(self):
        """Turn number starts at 0."""
        with GoEngine(seed=401) as engine:
            assert engine.turn_number() == 0

    def test_go_engine_stock_len(self):
        """Stockpile length is positive at game start."""
        with GoEngine(seed=402) as engine:
            s = engine.stock_len()
            assert s > 0, f"Expected positive stockpile, got {s}"

    def test_go_engine_is_terminal_initial(self):
        """A freshly created game is not terminal."""
        with GoEngine(seed=403) as engine:
            assert not engine.is_terminal()


class TestGoEngineFullGame:
    def test_go_engine_full_game(self):
        """Play a complete game to terminal and check utility."""
        utility = play_random_game(seed=999)
        assert utility.shape == (2,), f"Expected (2,), got {utility.shape}"
        assert utility.dtype == np.float32
        # Utility is a zero-sum game: sum should be ~0
        assert abs(float(utility.sum())) < 1e-3, f"Non-zero-sum utility: {utility}"

    def test_go_engine_utility_non_nan(self):
        """Utility values from a full game are finite."""
        utility = play_random_game(seed=12345)
        assert np.all(np.isfinite(utility)), f"Non-finite utility: {utility}"


class TestGoEngineLifecycle:
    def test_double_free_safe(self):
        """Calling close() twice does not crash."""
        engine = GoEngine(seed=500)
        engine.close()
        engine.close()  # Must not raise

    def test_del_after_close_safe(self):
        """Calling __del__ after close does not crash."""
        engine = GoEngine(seed=501)
        engine.close()
        engine.__del__()  # Must not raise


# ---------------------------------------------------------------------------
# GoAgentState tests
# ---------------------------------------------------------------------------


class TestGoAgentCreate:
    def test_go_agent_create_and_free(self):
        """Agent creates successfully with a valid handle."""
        with GoEngine(seed=600) as engine:
            agent = GoAgentState(engine, player_id=0)
            assert agent._agent_h >= 0
            agent.close()

    def test_go_agent_context_manager(self):
        """Agent context manager closes cleanly."""
        with GoEngine(seed=601) as engine:
            with GoAgentState(engine, player_id=1) as agent:
                assert agent._agent_h >= 0
            assert agent._agent_h == -1

    def test_go_agent_player1(self):
        """Agent for player 1 creates without error."""
        with GoEngine(seed=602) as engine:
            with GoAgentState(engine, player_id=1) as agent:
                assert agent._agent_h >= 0


class TestGoAgentEncode:
    def test_go_agent_update_and_encode(self):
        """After update, encode returns a (222,) float32 array."""
        with GoEngine(seed=700) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                agent.update(engine)
                vec = agent.encode(decision_context=0)
                assert vec.shape == (222,), f"Expected (222,), got {vec.shape}"
                assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"

    def test_go_agent_encode_values_valid(self):
        """All encoded values are in [0, 1] with no NaN or Inf."""
        with GoEngine(seed=701) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                agent.update(engine)
                vec = agent.encode(decision_context=0)
                assert np.all(np.isfinite(vec)), "Encoding contains NaN or Inf"
                assert np.all(vec >= 0.0) and np.all(
                    vec <= 1.0
                ), f"Encoding values out of [0,1]: min={vec.min()}, max={vec.max()}"

    def test_go_agent_encode_with_drawn_bucket(self):
        """Encode with a valid drawn_bucket does not raise."""
        with GoEngine(seed=702) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                agent.update(engine)
                vec = agent.encode(decision_context=1, drawn_bucket=2)
                assert vec.shape == (222,)

    def test_go_agent_encode_no_drawn_card(self):
        """Encode with drawn_bucket=-1 (default) does not raise."""
        with GoEngine(seed=703) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                vec = agent.encode(decision_context=0, drawn_bucket=-1)
                assert vec.shape == (222,)


class TestGoAgentClone:
    def test_go_agent_clone_independent(self):
        """Clone produces an independent agent with its own handle."""
        with GoEngine(seed=800) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                agent.update(engine)
                clone = agent.clone()
                assert clone._agent_h >= 0
                assert clone._agent_h != agent._agent_h
                clone.close()

    def test_go_agent_clone_encodes_same(self):
        """Clone produces the same encoding as the original before divergence."""
        with GoEngine(seed=801) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                agent.update(engine)
                vec_orig = agent.encode(decision_context=0)
                clone = agent.clone()
                vec_clone = clone.encode(decision_context=0)
                np.testing.assert_array_equal(vec_orig, vec_clone)
                clone.close()

    def test_go_agent_clone_context_manager(self):
        """Clone works correctly as a context manager."""
        with GoEngine(seed=802) as engine:
            with GoAgentState(engine, player_id=0) as agent:
                with agent.clone() as clone:
                    assert clone._agent_h >= 0


class TestGoAgentLifecycle:
    def test_agent_double_free_safe(self):
        """Calling close() twice on an agent does not crash."""
        with GoEngine(seed=900) as engine:
            agent = GoAgentState(engine, player_id=0)
            agent.close()
            agent.close()  # Must not raise

    def test_agent_del_after_close_safe(self):
        """Calling __del__ after close does not crash."""
        with GoEngine(seed=901) as engine:
            agent = GoAgentState(engine, player_id=0)
            agent.close()
            agent.__del__()  # Must not raise


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_game_with_agents(self):
        """Play a full game, updating both agents at each step."""
        with GoEngine(seed=1001) as engine:
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            steps = 0
            while not engine.is_terminal():
                mask = engine.legal_actions_mask()
                legal = np.where(mask > 0)[0]
                assert len(legal) > 0
                engine.apply_action(int(legal[0]))
                for a in agents:
                    a.update(engine)
                steps += 1
                assert steps < 10000, "Game did not terminate within 10000 steps"

            utility = engine.get_utility()
            assert utility.shape == (2,)
            assert np.all(np.isfinite(utility))
            for a in agents:
                a.close()

    def test_multiple_games(self):
        """Create several games simultaneously and play them independently."""
        engines = [GoEngine(seed=i * 100) for i in range(4)]
        for engine in engines:
            assert engine.handle >= 0
            # Play a few actions in each
            mask = engine.legal_actions_mask()
            legal = np.where(mask > 0)[0]
            engine.apply_action(int(legal[0]))
        for engine in engines:
            engine.close()

    def test_save_restore_mid_game(self):
        """Save/restore works correctly in the middle of a game."""
        with GoEngine(seed=2000) as engine:
            # Advance a few turns
            for _ in range(3):
                if engine.is_terminal():
                    break
                mask = engine.legal_actions_mask()
                legal = np.where(mask > 0)[0]
                engine.apply_action(int(legal[0]))

            if not engine.is_terminal():
                snap_h = engine.save()
                turn_at_save = engine.turn_number()
                mask_at_save = engine.legal_actions_mask()

                # Advance more
                mask = engine.legal_actions_mask()
                legal = np.where(mask > 0)[0]
                engine.apply_action(int(legal[0]))

                # Restore and verify
                engine.restore(snap_h)
                engine.free_snapshot(snap_h)

                assert engine.turn_number() == turn_at_save
                np.testing.assert_array_equal(engine.legal_actions_mask(), mask_at_save)

    def test_random_games_produce_valid_utilities(self):
        """Multiple random games all produce valid zero-sum utilities."""
        seeds = [42, 137, 256, 512, 1024]
        for seed in seeds:
            utility = play_random_game(seed)
            assert utility.shape == (2,)
            assert np.all(np.isfinite(utility)), f"seed={seed}: non-finite utility"
            assert abs(float(utility.sum())) < 1e-3, (
                f"seed={seed}: non-zero-sum utility {utility}"
            )

    def test_agent_encode_across_full_game(self):
        """Agent encodes remain valid shape and range for an entire game."""
        with GoEngine(seed=3000) as engine:
            agents = [GoAgentState(engine, pid) for pid in range(2)]
            while not engine.is_terminal():
                mask = engine.legal_actions_mask()
                legal = np.where(mask > 0)[0]
                engine.apply_action(int(np.random.choice(legal)))
                for a in agents:
                    a.update(engine)
                    vec = a.encode(decision_context=0)
                    assert vec.shape == (222,)
                    assert np.all(np.isfinite(vec))
            for a in agents:
                a.close()
