"""tests/test_policy_rng_2022.py

Every policy draw descends from the run seed (cambia-2022).

The defect these pin: an agent's own draws -- a wrapper's illegal-action
fallback, the PRT-CFR snapshot and action sampling, DESCA's abstract-action
sample, a random baseline's every move -- came off Python's global ``random``
module and numpy's global RNG. Neither is seeded by an evaluation run, and both
are reset to a fixed state by ``MaskablePPO.load``: after one, ``random.random()
== 0.6394267984578837`` and ``np.random.random() == 0.3745401188473625``, the
signature of seed 0. A match with a PPO side therefore ran the other side's
policy on one fixed variate stream no matter what seed the run was given
(cambia-1807 F2), and no run replayed another.

Covered here:

1. A random baseline and a wrapper fallback draw from a per-seat stream derived
   from the run seed, and a mid-run reseed of either global stream does not move
   them.
2. ``collect_infosets`` no longer seeds the global module, and still reproduces
   from its own seed.
3. Constructing a PPO wrapper leaves both global streams as it found them.
4. A DESCA match built after a PPO wrapper still moves with the run seed.
5. A PPO seat with no belief raises rather than playing uniform-random under the
   PPO name (cambia-1981), and a PPO seat above two seats no longer raises in
   the shared transition step (cambia-711 F3).
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.baseline_agents import BaseAgent, RandomAgent  # noqa: E402
from src.constants import (  # noqa: E402
    ActionCallCambia,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.ffi.bridge import GoEngine

    GoEngine.N_PLAYER_INPUT_DIM  # noqa: B018 - touch it so a broken load raises here
    _lib_ok = True
    _lib_err = ""
except Exception as exc:  # noqa: BLE001 - reported as a skip reason
    _lib_ok = False
    _lib_err = f"{type(exc).__name__}: {exc}"

skiplib = pytest.mark.skipif(not _lib_ok, reason=f"libcambia unavailable ({_lib_err})")

#: A legal-action list shaped like the engine's: ascending, no duplicates.
_ACTIONS = [
    ActionDrawStockpile(),
    ActionDrawDiscard(),
    ActionCallCambia(),
    ActionPassSnap(),
]


@pytest.fixture
def short_config(tmp_path):
    """The PRT production config with a short turn cap, so a game is quick."""
    from src.config import load_config

    base = os.path.join(PROJECT_ROOT, "config", "prtcfr_production.yaml")
    cfgpath = tmp_path / "policy_rng_config.yaml"
    cfgpath.write_text(f"_base: {base}\ncambia_rules:\n  max_game_turns: 30\n")
    loaded = load_config(str(cfgpath))
    assert loaded is not None
    return loaded


def _reseed_both_globals(seed: int) -> None:
    """What an SB3 model load does to the process, in one line."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. The per-seat stream
# ---------------------------------------------------------------------------


class _FallbackOnlyAgent(BaseAgent):
    """An agent whose whole policy is the uniform fallback every wrapper has."""

    accepts_game_view = True

    def choose_action(self, game_state, legal_actions):
        return self.uniform_action(legal_actions)


def _sequence(agent, actions, draws=40):
    return [repr(agent.choose_action(None, actions)) for _ in range(draws)]


def test_a_seats_policy_stream_follows_the_run_seed(short_config):
    """Same run seed replays; a different one plays differently."""
    same_a = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=4242), _ACTIONS)
    same_b = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=4242), _ACTIONS)
    other = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=4243), _ACTIONS)

    assert same_a == same_b, "the same run seed did not replay the same draws"
    assert same_a != other, "the run seed does not reach the draws at all"


def test_two_seats_of_one_run_seed_draw_independently(short_config):
    """Seat is folded into the derivation, so two seats do not mirror."""
    seat0 = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=99), _ACTIONS)
    seat1 = _sequence(_FallbackOnlyAgent(1, short_config, policy_seed=99), _ACTIONS)
    assert seat0 != seat1


def test_a_mid_run_reseed_of_either_global_stream_does_not_move_a_policy(short_config):
    """The regression proper: an SB3 load between two matches changes nothing."""
    _reseed_both_globals(1234)
    before = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=7), _ACTIONS)
    _reseed_both_globals(0)  # the state MaskablePPO.load leaves behind
    after = _sequence(_FallbackOnlyAgent(0, short_config, policy_seed=7), _ACTIONS)
    assert before == after


def test_a_random_baseline_follows_the_run_seed(short_config):
    """RandomAgent's every move is a policy draw, so it obeys the same rule."""
    same_a = _sequence(RandomAgent(0, short_config, seed=31337), _ACTIONS)
    _reseed_both_globals(0)
    same_b = _sequence(RandomAgent(0, short_config, seed=31337), _ACTIONS)
    other = _sequence(RandomAgent(0, short_config, seed=31338), _ACTIONS)

    assert same_a == same_b, "a global reseed moved a seeded baseline's stream"
    assert same_a != other


def test_a_set_of_legal_actions_is_ordered_before_the_draw(short_config):
    """A set carries no order, so one is imposed rather than hash order taken.

    ``GameAction`` is a NamedTuple carrying a string tag, so its hash -- and a
    set's iteration order with it -- moves with PYTHONHASHSEED (cambia-444).
    Drawing against that order would leave the seeded stream irreproducible
    across processes.
    """
    from_set = _sequence(
        _FallbackOnlyAgent(0, short_config, policy_seed=5), set(_ACTIONS)
    )
    from_sorted = _sequence(
        _FallbackOnlyAgent(0, short_config, policy_seed=5),
        sorted(_ACTIONS, key=repr),
    )
    assert from_set == from_sorted


# ---------------------------------------------------------------------------
# 2. collect_infosets leaves the global module alone
# ---------------------------------------------------------------------------


class _UniformProbe:
    """A policy with no state, for driving a collection cheaply."""

    accepts_game_view = True

    def __init__(self, player_id: int, seed: int = 0):
        self.player_id = player_id
        self._rng = np.random.default_rng(seed)

    def choose_action(self, view, legal_actions):
        actions = list(legal_actions)
        return actions[int(self._rng.integers(len(actions)))]


@skiplib
def test_collect_infosets_does_not_seed_the_global_module(short_config):
    """AC2: nothing there reads the global module, so nothing there seeds it.

    Seeding it used to be the service ``collect_infosets`` did for the policies
    that drew from it. Those draws moved to per-agent streams, and a seed of a
    stream nothing reads only suggests a guarantee this function cannot make:
    an SB3 load anywhere in the process resets it again.
    """
    from src.cfr.lbr import collect_infosets

    random.seed(20260902)
    before = random.getstate()
    sampled = collect_infosets(
        _UniformProbe(0), short_config, num_infosets=5, seed=1234, max_games=20
    )
    assert sampled, "the collection produced no infosets, so it proves nothing"
    assert random.getstate() == before, "collect_infosets moved the global module"


@skiplib
def test_collect_infosets_still_reproduces_from_its_own_seed(short_config):
    """Dropping the global seed did not cost the function its determinism."""
    from src.cfr.lbr import collect_infosets

    def _collect():
        return [
            (s.deal.seed, s.action_prefix, s.agent_action_pos)
            for s in collect_infosets(
                _UniformProbe(0), short_config, num_infosets=5, seed=99, max_games=20
            )
        ]

    first = _collect()
    _reseed_both_globals(0)
    assert _collect() == first


# ---------------------------------------------------------------------------
# 3-4. PPO construction, and a DESCA match built after one
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ppo_model(tmp_path_factory):
    """A saved MaskablePPO with no training: only the load side effect matters."""
    sb3_contrib = pytest.importorskip("sb3_contrib", reason="sb3-contrib required")
    from src.ppo_env import CambiaEnv

    tmpdir = tmp_path_factory.mktemp("ppo")
    path = str(tmpdir / "tiny_ppo")
    env = CambiaEnv(opponent_type="random_legal", seed=42)
    try:
        model = sb3_contrib.MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",
            seed=42,
            policy_kwargs={"net_arch": [16, 16]},
            n_steps=16,
            batch_size=8,
        )
        model.save(path)
    finally:
        env.close()
    return path


@pytest.fixture
def desca_checkpoint(tmp_path):
    """An untrained DESCA checkpoint: the sampling path, not the strategy."""
    from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P
    from src.constants import EP_PBS_V2_INPUT_DIM
    from src.desca_networks import AvgStrategyNetwork

    torch.manual_seed(2022)
    net = AvgStrategyNetwork(
        input_dim=EP_PBS_V2_INPUT_DIM,
        hidden_dim=64,
        num_actions=NUM_ABSTRACT_ACTIONS_2P,
    )
    path = str(tmp_path / "desca_checkpoint.pt")
    torch.save(
        {
            "avg_strategy_state_dict": net.state_dict(),
            "desca_config": {
                "hidden_dim": 64,
                "encoding_dim": EP_PBS_V2_INPUT_DIM,
                "num_abstract_actions": NUM_ABSTRACT_ACTIONS_2P,
            },
            "iteration": 0,
        },
        path,
    )
    return path


@skiplib
def test_ppo_construction_leaves_both_global_streams_as_it_found_them(
    tiny_ppo_model, short_config
):
    """AC3: the load's reseed is undone, so construction is not a side effect.

    The direct load is asserted first: without it a passing test could only mean
    this Stable-Baselines3 build stopped reseeding, not that the wrapper puts
    the streams back.
    """
    from sb3_contrib import MaskablePPO

    from src.evaluate_agents import PPOAgentWrapper

    _reseed_both_globals(777)
    py_before, np_before = random.getstate(), np.random.get_state()
    MaskablePPO.load(tiny_ppo_model, device="cpu")
    assert random.getstate() != py_before and not np.array_equal(
        np.random.get_state()[1], np_before[1]
    ), "this sb3 build does not reseed the global streams; the guard is untested"

    _reseed_both_globals(777)
    py_before, np_before = random.getstate(), np.random.get_state()
    PPOAgentWrapper(0, short_config, tiny_ppo_model, device="cpu", policy_seed=5)
    assert random.getstate() == py_before, "PPO construction moved the global module"
    assert np.array_equal(
        np.random.get_state()[1], np_before[1]
    ), "PPO construction moved numpy's global RNG"


def _play_desca_game(config, desca_ckpt, policy_seed, deck_seed, ppo_model=None):
    """One recorded game: DESCA at seat 0, a fixed uniform baseline at seat 1.

    Returns both halves of the DESCA seat's decision: the abstract action it
    sampled from the network (the draw the PPO load used to pin) and the
    concrete action it unabstracted to (the draw that used to be keyed on an
    object address). ``ppo_model``, when given, is loaded into a wrapper BEFORE
    the DESCA seat is built, which is the ordering the defect needed.
    """
    from src import action_abstraction
    from src.evaluate_agents import (
        DESCAAgentWrapper,
        PPOAgentWrapper,
        _GoEvalGame,
    )

    if ppo_model is not None:
        PPOAgentWrapper(0, config, ppo_model, device="cpu", policy_seed=policy_seed)

    agents = [
        DESCAAgentWrapper(
            0, config, desca_ckpt, device="cpu", use_argmax=False, policy_seed=policy_seed
        ),
        RandomAgent(1, config, seed=555),
    ]
    session = _GoEvalGame(config.cambia_rules, deck_seed, 2, agents)
    trace = []
    real_unabstract = action_abstraction.unabstract

    def _recording_unabstract(abstract_idx, legal_actions, agent_state, seed):
        trace.append(("abstract", int(abstract_idx)))
        return real_unabstract(abstract_idx, legal_actions, agent_state, seed)

    action_abstraction.unabstract = _recording_unabstract
    try:
        turn = 0
        while not session.is_terminal() and turn < 200:
            turn += 1
            seat = session.acting_player()
            if seat < 0:
                break
            legal = session.legal_actions()
            if not legal:
                break
            chosen = agents[seat].choose_action(session.engine, legal)
            if seat == 0:
                trace.append(("concrete", repr(chosen)))
            session.apply(chosen)
    finally:
        action_abstraction.unabstract = real_unabstract
        session.close()
    return trace


@skiplib
def test_a_desca_match_after_a_ppo_build_moves_with_the_run_seed(
    tiny_ppo_model, desca_checkpoint, short_config
):
    """AC4: the PPO load no longer decides how the other side plays.

    Before cambia-2022 neither half of the DESCA decision answered to the run
    seed. The abstract action was sampled with ``np.random.choice`` against
    numpy's global RNG, which the PPO load had just set to seed 0, so a match
    drew the same variates whatever seed it was given; the concrete action
    inside that abstract class was picked with a seed built from ``id(engine)``,
    an object address, so it did not repeat either. Measured on the pre-fix code
    with the same harness: two matches at one deal produced different traces.
    """
    deck_seed = 424242
    a = _play_desca_game(
        short_config, desca_checkpoint, 11, deck_seed, ppo_model=tiny_ppo_model
    )
    b = _play_desca_game(
        short_config, desca_checkpoint, 12, deck_seed, ppo_model=tiny_ppo_model
    )
    again = _play_desca_game(
        short_config, desca_checkpoint, 11, deck_seed, ppo_model=tiny_ppo_model
    )

    assert a, "the DESCA seat never acted"
    assert a == again, "the same run seed did not replay the same DESCA actions"
    assert a != b, (
        "two run seeds produced the identical DESCA action stream on one deal; "
        "the sampling is still on a stream the run seed does not reach"
    )


# ---------------------------------------------------------------------------
# 5. The PPO seat: no belief, and above two seats
# ---------------------------------------------------------------------------


@skiplib
def test_a_ppo_seat_without_a_belief_raises(tiny_ppo_model, short_config):
    """cambia-1981: an unset belief used to play uniform-random as PPO."""
    from src.cfr.lbr import PolicyError
    from src.evaluate_agents import PPOAgentWrapper

    wrapper = PPOAgentWrapper(0, short_config, tiny_ppo_model, device="cpu")
    engine = GoEngine(seed=7, house_rules=short_config.cambia_rules, num_players=2)
    try:
        from src.agents import action_codec

        legal = action_codec.actions_from_mask(engine.legal_actions_mask())
        with pytest.raises(PolicyError, match="belief not attached"):
            wrapper.choose_action(engine, legal)
    finally:
        engine.close()


@skiplib
def test_a_ppo_seat_plays_at_three_seats(tiny_ppo_model, short_config):
    """cambia-711 F3: the N-player transition reaches PPO's belief by name.

    ``TransitionBroadcaster`` advances an N-player belief through
    ``agent_state.update_nplayer``. While this wrapper kept its belief under a
    private name that call raised for a PPO seat above two seats, and the loop
    counted the game as an error.
    """
    from src.agents.transition import TransitionBroadcaster
    from src.evaluate_agents import PPOAgentWrapper
    from src.play import _legal_actions_for

    wrapper = PPOAgentWrapper(0, short_config, tiny_ppo_model, device="cpu")
    engine = GoEngine(seed=3, house_rules=short_config.cambia_rules, num_players=3)
    try:
        broadcast = TransitionBroadcaster(engine, [wrapper, None, None], 3)
        assert wrapper.belief_handle() >= 0, "the PPO seat attached no belief"
        applied = 0
        for _ in range(12):
            if engine.is_terminal():
                break
            seat = engine.acting_player()
            if seat < 0:
                break
            actions, index = _legal_actions_for(engine, seat, 3)
            assert actions
            broadcast.apply(actions[0], index, seat)
            applied += 1
        assert applied >= 3, "too few actions applied to have exercised the branch"
        broadcast.release()
    finally:
        engine.close()
