"""Tests for the Go-backed PPO gym environment (cambia-1376).

Covers the three things the port has to get right: the env steps on GoEngine
with a legal-only action path, the seat count is a real parameter (2 seats keep
the 2-player space, 4 seats use the N-player space), and the Go handle pools do
not leak across the thousands of resets a training run performs.

These tests need libcambia.so; they skip when it cannot be loaded.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_CFR_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(_CFR_ROOT / "config" / "ppo_encoding_v2.yaml")

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium required for the PPO env")

try:
    from src.ffi.bridge import GoEngine, get_handle_pool_stats

    GoEngine.N_PLAYER_INPUT_DIM  # noqa: B018 - touch it so a broken load raises here
    _lib_ok = True
    _lib_err = ""
except Exception as exc:  # noqa: BLE001 - reported as a skip reason
    _lib_ok = False
    _lib_err = f"{type(exc).__name__}: {exc}"

skiplib = pytest.mark.skipif(not _lib_ok, reason=f"libcambia unavailable ({_lib_err})")


def _load_lib_or_skip():
    from src.ffi.bridge import _get_lib

    try:
        _get_lib()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"libcambia.so not loadable: {exc}")


@pytest.fixture
def env_factory():
    """Build CambiaEnv instances and guarantee they are closed.

    Every env allocates Go game and agent handles, so a test that raises
    mid-way would otherwise leak them into the next test's pool reading.
    """
    _load_lib_or_skip()
    from src.ppo_env import CambiaEnv

    built = []

    def _make(**kwargs):
        kwargs.setdefault("config_path", CONFIG_PATH)
        kwargs.setdefault("opponent_type", "random_legal")
        kwargs.setdefault("seed", 12345)
        env = CambiaEnv(**kwargs)
        built.append(env)
        return env

    yield _make

    for env in built:
        env.close()


def _drive(env, steps, rng, assert_space=True):
    """Step the env with uniform-random legal actions; return terminal rewards."""
    rewards = []
    for _ in range(steps):
        mask = np.asarray(env.action_masks())
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            env.reset()
            continue
        obs, reward, term, trunc, _ = env.step(int(rng.choice(legal)))
        if assert_space:
            assert env.observation_space.contains(obs), (
                f"obs outside observation_space: shape={obs.shape} "
                f"dtype={obs.dtype} min={obs.min()} max={obs.max()}"
            )
        if term or trunc:
            rewards.append(reward)
            env.reset()
    return rewards


# ---------------------------------------------------------------------------
# The env runs on the Go engine, not the Python one
# ---------------------------------------------------------------------------


def test_ppo_env_does_not_import_the_python_engine():
    """ppo_env must not import src.game or the Python belief state.

    Walks the module's import statements rather than grepping the text, so the
    docstring may still name CambiaGameState when explaining why the fixed
    baselines are refused. sys.modules is no good here: another test in the
    same session may have imported src.game for its own reasons.
    """
    import ast

    source = (_CFR_ROOT / "src" / "ppo_env.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    offenders = sorted(
        name
        for name in imported
        if name.startswith(("src.game", "src.agent_state", "src.agents"))
    )
    assert not offenders, f"ppo_env still imports the Python engine path: {offenders}"


@skiplib
def test_env_steps_on_goengine(env_factory):
    """A 2-seat env produces the 2-player space and steps to termination."""
    from src.encoding import NUM_ACTIONS
    from src.constants import EP_PBS_V2_INPUT_DIM

    env = env_factory(num_players=2)
    obs, info = env.reset(seed=99)

    assert env.action_space.n == NUM_ACTIONS
    assert env.observation_space.shape == (EP_PBS_V2_INPUT_DIM,)
    assert obs.shape == (EP_PBS_V2_INPUT_DIM,)
    assert obs.dtype == np.float32
    assert info == {}
    # The engine really is the Go one.
    assert isinstance(env._engine, GoEngine)

    rewards = _drive(env, 600, np.random.default_rng(0))
    assert rewards, "no episode terminated in 600 steps"
    assert all(np.isfinite(r) for r in rewards)


@skiplib
def test_action_masks_are_bool_and_agree_with_the_engine(env_factory):
    """action_masks must be the engine's legality, as bool for MaskablePPO."""
    env = env_factory(num_players=2)
    env.reset(seed=3)
    rng = np.random.default_rng(1)

    for _ in range(300):
        mask = env.action_masks()
        assert mask.dtype == np.bool_, f"MaskablePPO needs bool, got {mask.dtype}"
        assert mask.shape == (env.action_space.n,)
        engine_mask = env._engine.legal_actions_mask().astype(bool)
        np.testing.assert_array_equal(mask, engine_mask)
        assert mask.any(), "no legal action at a live state"
        legal = np.flatnonzero(mask)
        _, _, term, _, _ = env.step(int(rng.choice(legal)))
        if term:
            env.reset()


@skiplib
def test_illegal_action_falls_back_to_a_legal_one(env_factory):
    """An out-of-mask or out-of-range action must not reach the engine.

    MaskablePPO should never emit one, but the env is the last line of defence:
    the Go apply path rejects an illegal index with a RuntimeError, which would
    kill the SubprocVecEnv worker.
    """
    env = env_factory(num_players=2)
    env.reset(seed=5)

    for _ in range(120):
        mask = np.asarray(env.action_masks())
        illegal = np.flatnonzero(~mask)
        if illegal.size == 0:
            break
        # Both an illegal-but-in-range index and an out-of-range one.
        env.step(int(illegal[0]))
        if env._engine.is_terminal():
            env.reset()
            continue
        env.step(env.action_space.n + 50)
        if env._engine.is_terminal():
            env.reset()
        env.step(-7)
        if env._engine.is_terminal():
            env.reset()


@skiplib
def test_terminal_reward_is_the_agent_seats_engine_utility(env_factory):
    """Reward at termination is the engine's utility for the agent's seat."""
    env = env_factory(num_players=2)
    env.reset(seed=11)
    rng = np.random.default_rng(2)

    checked = 0
    for _ in range(1500):
        mask = np.asarray(env.action_masks())
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            env.reset()
            continue
        seat = env._agent_seat
        _, reward, term, _, _ = env.step(int(rng.choice(legal)))
        if term:
            util = env._engine.get_utility()
            assert reward == pytest.approx(float(util[seat]), abs=1e-6)
            assert float(util.sum()) == pytest.approx(0.0, abs=1e-5)
            checked += 1
            env.reset()
    assert checked >= 5, f"only {checked} terminations observed"


@skiplib
def test_non_terminal_steps_give_zero_reward(env_factory):
    """Cambia's reward is terminal-only; a mid-game step must pay nothing."""
    env = env_factory(num_players=2)
    env.reset(seed=13)
    rng = np.random.default_rng(3)

    for _ in range(600):
        mask = np.asarray(env.action_masks())
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            env.reset()
            continue
        _, reward, term, _, _ = env.step(int(rng.choice(legal)))
        if not term:
            assert reward == 0.0
        else:
            env.reset()


# ---------------------------------------------------------------------------
# Seat count is a parameter (AC3)
# ---------------------------------------------------------------------------


@skiplib
@pytest.mark.parametrize("seats", [3, 4, 6])
def test_n_seat_env_constructs_and_steps(env_factory, seats):
    """3+ seats select the N-player space and play to termination."""
    env = env_factory(num_players=seats)
    obs, _ = env.reset(seed=21)

    assert env.action_space.n == GoEngine.N_PLAYER_NUM_ACTIONS
    assert env.observation_space.shape == (GoEngine.N_PLAYER_INPUT_DIM,)
    assert obs.shape == (GoEngine.N_PLAYER_INPUT_DIM,)
    assert env._engine.num_players() == seats
    assert len(env._agents) == seats

    rewards = _drive(env, 800, np.random.default_rng(seats))
    assert rewards, f"no episode terminated at {seats} seats"

    util = env._engine.get_nplayer_utility()
    assert util.shape == (seats,)


@skiplib
def test_four_seat_env_is_the_ac3_configuration(env_factory):
    """AC3: the 4-seat env constructs and every seat gets its own belief state."""
    env = env_factory(num_players=4)
    env.reset(seed=4)

    assert env._num_players == 4
    assert env._engine.num_players() == 4
    assert len(env._agents) == 4
    # Each seat reads its own belief, so at least two seats differ. A shared or
    # mis-indexed agent state would make these identical.
    encodings = [env._get_obs(seat=s) for s in range(4)]
    assert any(
        not np.array_equal(encodings[0], encodings[s]) for s in range(1, 4)
    ), "every seat produced the same observation; belief states are not per-seat"


@skiplib
def test_two_seats_and_n_seats_use_different_spaces(env_factory):
    """The 2-player and N-player spaces differ, so checkpoints do not transfer."""
    two = env_factory(num_players=2)
    four = env_factory(num_players=4)
    assert two.action_space.n != four.action_space.n
    assert two.observation_space.shape != four.observation_space.shape


@skiplib
def test_num_players_below_two_is_rejected(env_factory):
    for bad in (1, 0, -3):
        with pytest.raises(ValueError, match="at least 2"):
            env_factory(num_players=bad)


# ---------------------------------------------------------------------------
# Handle hygiene: a training run resets thousands of times per worker
# ---------------------------------------------------------------------------


@skiplib
def test_repeated_resets_do_not_leak_go_handles(env_factory):
    """Resets must free the previous episode's game and agent handles.

    The Go side serves handles from fixed-size pools, so a per-reset leak
    exhausts them partway through a run. Relying on __del__ is not enough:
    CPython would eventually collect, but not deterministically.
    """
    env = env_factory(num_players=2)
    env.reset(seed=1)
    baseline = get_handle_pool_stats()

    for i in range(400):
        env.reset(seed=1000 + i)

    after = get_handle_pool_stats()
    assert after["games"] == baseline["games"], (
        f"game handles grew from {baseline['games']} to {after['games']} "
        "over 400 resets"
    )
    assert after["agents"] == baseline["agents"], (
        f"agent handles grew from {baseline['agents']} to {after['agents']} "
        "over 400 resets"
    )


@skiplib
def test_reset_rejects_deals_that_end_before_the_agent_acts(env_factory):
    """A deal whose opponents finish the game leaves PPO no decision to make.

    reset() must retry rather than return a terminal observation, which step()
    would answer with a zero reward instead of the seat's real utility. Forced
    here by making opponent advancement play the game out.
    """
    env = env_factory(num_players=2)
    env.reset(seed=71)
    baseline = get_handle_pool_stats()

    rng = np.random.default_rng(0)

    def _play_out(self=env):
        while not self._engine.is_terminal():
            mask = self._legal_mask()
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            self._apply(int(legal[int(rng.integers(legal.size))]))
            self._update_agents()

    env._advance_opponent = _play_out
    with pytest.raises(RuntimeError, match="ended before the agent's first turn"):
        env.reset(seed=72)

    # None of the discarded deals may strand handles in the Go pools. The env
    # gave up its own game and agents on the way out, so the counts land at or
    # below the one live env the baseline was read with, never above it.
    after = get_handle_pool_stats()
    assert after["games"] <= baseline["games"], after
    assert after["agents"] <= baseline["agents"], after


@skiplib
def test_close_releases_handles(env_factory):
    """close() frees the handles and is safe to call twice."""
    from src.ppo_env import CambiaEnv

    _load_lib_or_skip()
    before = get_handle_pool_stats()
    env = CambiaEnv(
        opponent_type="random_legal", seed=2, config_path=CONFIG_PATH, num_players=3
    )
    env.reset(seed=2)
    assert get_handle_pool_stats()["games"] > before["games"]
    env.close()
    env.close()
    assert get_handle_pool_stats() == before


# ---------------------------------------------------------------------------
# Opponent regimes
# ---------------------------------------------------------------------------


@skiplib
@pytest.mark.parametrize(
    "opponent",
    ["imperfect_greedy", "memory_heuristic", "aggressive_snap", "random", "greedy"],
)
def test_python_engine_baselines_are_refused_with_a_pointer(env_factory, opponent):
    """The fixed baselines are not runnable here yet; the env must say so.

    They are written against the Python CambiaGameState and are being ported to
    the GameView protocol under cambia-1426. Accepting them quietly would mean
    reviving the Python engine behind the env.
    """
    with pytest.raises(NotImplementedError, match="cambia-1426"):
        env_factory(opponent_type=opponent)


@skiplib
def test_self_play_requires_a_snapshot_path(env_factory):
    with pytest.raises(ValueError, match="selfplay_snapshot_path"):
        env_factory(opponent_type="self_play", selfplay_snapshot_path=None)


@skiplib
def test_self_play_warmup_plays_without_a_snapshot_file(env_factory, tmp_path):
    """Before the first snapshot exists the opponent plays random legal moves.

    Episodes must still terminate and produce reward, otherwise the opening
    refresh interval of every self-play run would stall.
    """
    missing = str(tmp_path / "never_written")
    env = env_factory(
        opponent_type="self_play", selfplay_snapshot_path=missing, num_players=2
    )
    env.reset(seed=31)
    rewards = _drive(env, 600, np.random.default_rng(7))
    assert rewards, "self-play warm-up produced no terminated episode"


@skiplib
def test_self_play_randomizes_the_agent_seat(env_factory, tmp_path):
    """The learner must play every seat over a run, not just seat 0."""
    env = env_factory(
        opponent_type="self_play",
        selfplay_snapshot_path=str(tmp_path / "snap"),
        num_players=4,
    )
    seats = set()
    for i in range(80):
        env.reset(seed=500 + i)
        seats.add(env._agent_seat)
    assert seats == {0, 1, 2, 3}, f"agent only ever sat at {sorted(seats)}"


@skiplib
def test_self_play_opponent_is_asked_on_its_own_seat(env_factory, tmp_path):
    """The snapshot must be queried with the acting opponent seat's belief.

    A seat-indexing slip here would hand the opponent the learner's own belief
    encoding, which is both an information leak and a silent self-play bug: the
    run would still train and still look healthy.
    """
    env = env_factory(
        opponent_type="self_play",
        selfplay_snapshot_path=str(tmp_path / "snap"),
        num_players=4,
    )
    env.reset(seed=81)

    seen = []

    class _Recorder:
        def predict_index(self, obs, action_mask):
            # Checked here, not afterwards: the belief states advance with every
            # applied action, so a comparison made once the episode has moved on
            # would be against a different game state.
            seat = env._engine.acting_player()
            own = env._get_obs(seat=seat)
            learner = env._get_obs(seat=env._agent_seat)
            seen.append(
                (
                    seat,
                    np.array_equal(obs, own),
                    np.array_equal(obs, learner),
                )
            )
            # Decline, so the env takes its random-legal fallback.
            return None

    env._opponent = _Recorder()
    rng = np.random.default_rng(3)
    for _ in range(400):
        mask = np.asarray(env.action_masks())
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        _, _, term, _, _ = env.step(int(rng.choice(legal)))
        if term:
            break

    assert seen, "the self-play opponent was never consulted"
    for seat, matched_own, matched_learner in seen:
        assert seat != env._agent_seat, "opponent asked on the learner's seat"
        assert matched_own, f"seat {seat} was not shown its own belief encoding"
        assert not matched_learner, f"seat {seat} was shown the learner's belief"


@skiplib
def test_random_legal_opponent_keeps_the_agent_on_its_seat(env_factory):
    """Without self-play the agent stays where it was configured."""
    env = env_factory(num_players=4, agent_seat=2)
    for i in range(20):
        env.reset(seed=700 + i)
        assert env._agent_seat == 2


@skiplib
def test_agent_seat_is_wrapped_into_range(env_factory):
    env = env_factory(num_players=3, agent_seat=7)
    assert env._agent_seat == 7 % 3


# ---------------------------------------------------------------------------
# Observation wiring
# ---------------------------------------------------------------------------


@skiplib
def test_observation_comes_from_the_acting_seats_own_belief(env_factory):
    """_get_obs(seat) must encode that seat's GoAgentState, not the learner's."""
    env = env_factory(num_players=2)
    env.reset(seed=41)

    own = env._get_obs()
    same = env._get_obs(seat=env._agent_seat)
    other = env._get_obs(seat=1 - env._agent_seat)

    np.testing.assert_array_equal(own, same)
    assert not np.array_equal(own, other), "both seats produced one belief encoding"


@skiplib
def test_encoding_version_one_selects_the_224_dim_space(env_factory, tmp_path):
    """encoding_version 1 must size the obs space to the v1 layout."""
    from src.constants import EP_PBS_INPUT_DIM

    cfg = (
        Path(CONFIG_PATH)
        .read_text(encoding="utf-8")
        .replace("encoding_version: 2", "encoding_version: 1")
    )
    path = tmp_path / "ppo_v1.yaml"
    path.write_text(cfg, encoding="utf-8")

    env = env_factory(num_players=2, config_path=str(path))
    obs, _ = env.reset(seed=51)
    assert env.observation_space.shape == (EP_PBS_INPUT_DIM,)
    assert obs.shape == (EP_PBS_INPUT_DIM,)


@skiplib
def test_step_before_reset_raises(env_factory):
    env = env_factory(num_players=2)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


@skiplib
def test_stepping_a_terminal_state_is_idempotent(env_factory):
    """Extra steps after termination return the terminal flag, not an error."""
    env = env_factory(num_players=2)
    env.reset(seed=61)
    rng = np.random.default_rng(9)

    for _ in range(2000):
        mask = np.asarray(env.action_masks())
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        _, _, term, _, _ = env.step(int(rng.choice(legal)))
        if term:
            break
    else:
        pytest.skip("no termination reached")

    for _ in range(3):
        obs, reward, term, trunc, _ = env.step(0)
        assert term is True
        assert trunc is False
        assert reward == 0.0
        assert obs.shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# make_env / trainer wiring
# ---------------------------------------------------------------------------


@skiplib
def test_make_env_threads_num_players(tmp_path):
    _load_lib_or_skip()
    from src.ppo_env import make_env

    env = make_env("random_legal", seed=1, config_path=CONFIG_PATH, num_players=4)()
    try:
        assert env._num_players == 4
        assert env.action_space.n == GoEngine.N_PLAYER_NUM_ACTIONS
    finally:
        env.close()


def test_train_ppo_rejects_python_engine_baselines(tmp_path):
    """The trainer must fail before spawning workers, where the traceback is opaque."""
    pytest.importorskip("sb3_contrib", reason="sb3-contrib required")
    from src.ppo_train import train_ppo

    with pytest.raises(NotImplementedError, match="cambia-1426"):
        train_ppo(
            opponent="imperfect_greedy",
            timesteps=64,
            save_path=str(tmp_path / "runs" / "x" / "checkpoints" / "m"),
            n_envs=1,
            eval_freq=0,
            net_arch=[8],
            config_path=CONFIG_PATH,
        )


def test_train_ppo_rejects_one_seat(tmp_path):
    pytest.importorskip("sb3_contrib", reason="sb3-contrib required")
    from src.ppo_train import train_ppo

    with pytest.raises(ValueError, match="at least 2"):
        train_ppo(
            opponent="random_legal",
            timesteps=64,
            save_path=str(tmp_path / "runs" / "x" / "checkpoints" / "m"),
            n_envs=1,
            eval_freq=0,
            net_arch=[8],
            num_players=1,
            config_path=CONFIG_PATH,
        )
