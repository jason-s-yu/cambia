"""tests/test_transition_broadcast_711.py

Every game loop applies through the one per-transition step (cambia-711).

The defect this guards: a game loop that advances only the engine leaves a
belief- or token-carrying seat on the state it was dealt, and nothing raises.
A PRT-CFR seat then chooses every action off its opening peek prefix and the
loop scores that as if it were the game. ``src/play.py`` had exactly that loop
until the shared ``TransitionBroadcaster`` replaced its private apply; the
head-to-head loops reach the same step through ``_GoEvalGame``.

The token cursor is the measurement because it is the strictest one available:
the engine appends a frame for EVERY applied action by either seat, so a seat
whose stream stops growing is a seat the loop stopped feeding.
"""

import contextlib
import io
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr.prtcfr_net import PRTCFRNet  # noqa: E402
from src.cfr.prtcfr_stability import (  # noqa: E402
    BestSnapshotController,
    write_deployable_manifest,
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


# ---------------------------------------------------------------------------
# Fixtures: a two-snapshot PRT-CFR run dir and a short-game config
# ---------------------------------------------------------------------------


def _tiny_net(seed: int) -> PRTCFRNet:
    torch.manual_seed(seed)
    return PRTCFRNet(
        embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        head_hidden_dim=16,
        dropout=0.0,
        device="cpu",
    )


@pytest.fixture
def prt_checkpoint(tmp_path):
    """A deployable snapshot set the PRT-CFR wrapper can be built from."""
    snapdir = tmp_path / "snapshots"
    snapdir.mkdir()
    iters = [1, 2]
    for t in iters:
        net = _tiny_net(t)
        torch.save(
            {
                "encoder_state_dict": net.encoder_state_dict(),
                "head_state_dict": net.head_state_dict(),
                "iteration": t,
            },
            str(snapdir / f"prtcfr_snapshot_iter_{t}.pt"),
        )
    ctrl = BestSnapshotController()
    ctrl.best_iteration = 2
    write_deployable_manifest(str(snapdir), ctrl, iters)
    return str(snapdir / "prtcfr_snapshot_iter_2.pt")


@pytest.fixture
def prt_config(tmp_path):
    """The PRT production config with a short turn cap, so a game is quick."""
    from src.config import load_config

    base = os.path.join(PROJECT_ROOT, "config", "prtcfr_production.yaml")
    cfgpath = tmp_path / "broadcast_config.yaml"
    cfgpath.write_text(f"_base: {base}\ncambia_rules:\n  max_game_turns: 15\n")
    loaded = load_config(str(cfgpath))
    assert loaded is not None
    return loaded


def _recording_prt_class():
    """PRTCFRAgentWrapper that logs its own token-stream length per decision.

    Subclassed rather than wrapped so it is the same object the loops build and
    seat: what is measured is the stream the wrapper would actually have queried
    its snapshot with.
    """
    from src.evaluate_agents import PRTCFRAgentWrapper

    class RecordingPRT(PRTCFRAgentWrapper):
        seen: list = []

        def initialize_state(self, initial_game_state):
            super().initialize_state(initial_game_state)
            # One episode per game. The stream is re-seeded at every deal, so a
            # length is only comparable against another length from the same
            # game by the same instance; a match reuses these instances.
            self._episode = getattr(self, "_episode", 0) + 1

        def choose_action(self, game_state, legal_actions):
            RecordingPRT.seen.append(
                (
                    (id(self), getattr(self, "_episode", 0)),
                    self.player_id,
                    self.agent_state.token_len(),
                )
            )
            return super().choose_action(game_state, legal_actions)

    RecordingPRT.seen = []
    return RecordingPRT


def _assert_stream_advanced(seen, label):
    """In every game, a seat that acted more than once saw its prefix grow."""
    assert seen, f"{label}: the PRT seat never chose an action"
    by_episode = {}
    for episode, seat, token_len in seen:
        by_episode.setdefault((episode, seat), []).append(token_len)
    grew = False
    for (_, seat), lens in by_episode.items():
        assert lens == sorted(lens), (
            f"{label}: seat {seat} token prefix went backwards ({lens}); the "
            "stream is append-only"
        )
        if len(lens) > 1:
            assert lens[-1] > lens[0], (
                f"{label}: seat {seat} played {len(lens)} decisions of one game "
                f"on a frozen {lens[0]}-token prefix; the loop never fed the "
                "transition"
            )
            grew = True
    assert grew, (
        f"{label}: no seat acted twice in any one game, so a frozen stream "
        "would not have shown up; the games were too short to be evidence"
    )


# ---------------------------------------------------------------------------
# Head-to-head: both loops
# ---------------------------------------------------------------------------


@skiplib
def test_head_to_head_advances_a_prt_seat_token_stream(
    monkeypatch, prt_checkpoint, prt_config
):
    """run_head_to_head seats PRT-CFR through AGENT_REGISTRY; its stream grows."""
    from src import evaluate_agents

    recording = _recording_prt_class()
    monkeypatch.setitem(evaluate_agents.AGENT_REGISTRY, "prt_cfr", recording)

    results = evaluate_agents.run_head_to_head(
        checkpoint_a=prt_checkpoint,
        checkpoint_b=prt_checkpoint,
        num_games=2,
        config=prt_config,
        device="cpu",
        agent_type="prt_cfr",
        seed=711,
    )

    assert results["errors"] == 0, "a PRT seat raised inside the head-to-head loop"
    _assert_stream_advanced(recording.seen, "run_head_to_head")


@skiplib
def test_head_to_head_typed_advances_a_prt_seat_token_stream(
    monkeypatch, prt_checkpoint, prt_config
):
    """The typed variant builds its seats through get_agent; same guarantee."""
    from src import evaluate_agents

    recording = _recording_prt_class()
    monkeypatch.setattr(evaluate_agents, "PRTCFRAgentWrapper", recording)

    results = evaluate_agents.run_head_to_head_typed(
        agent_a_type="prt_cfr",
        checkpoint_a=prt_checkpoint,
        agent_b_type="prt_cfr",
        checkpoint_b=prt_checkpoint,
        num_games=2,
        config=prt_config,
        device="cpu",
        seed=711,
    )

    assert results["errors"] == 0, "a PRT seat raised inside the typed loop"
    _assert_stream_advanced(recording.seen, "run_head_to_head_typed")


# ---------------------------------------------------------------------------
# Interactive play: the loop that carried the defect
# ---------------------------------------------------------------------------


@skiplib
def test_play_game_advances_a_prt_seat_token_stream(prt_checkpoint, prt_config):
    """play_game's AI seat is fed every transition.

    This is the regression proper: before cambia-711 play_game applied through
    the engine alone, so both seats here would have reported the same opening
    prefix length at every decision.
    """
    from src.play import SeatConfig, play_game

    recording = _recording_prt_class()
    seats = [
        SeatConfig(
            seat_id=i,
            is_human=False,
            name=f"prt_cfr(P{i})",
            agent_type="prt_cfr",
            agent=recording(i, prt_config, prt_checkpoint, device="cpu"),
        )
        for i in range(2)
    ]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        play_game(seats, prt_config.cambia_rules, num_players=2, seed=711)

    assert "Game Over" in buf.getvalue()
    _assert_stream_advanced(recording.seen, "play_game")


@skiplib
def test_play_game_releases_the_beliefs_it_attached(prt_checkpoint, prt_config):
    """The belief handles a game attached are given back when it ends.

    They come from a finite pool, so an interactive session that leaked one per
    game would eventually refuse to deal.
    """
    from src.play import SeatConfig, play_game

    agent = _recording_prt_class()(0, prt_config, prt_checkpoint, device="cpu")
    seats = [
        SeatConfig(
            seat_id=0, is_human=False, name="prt", agent_type="prt_cfr", agent=agent
        ),
        SeatConfig(
            seat_id=1,
            is_human=False,
            name="baseline",
            agent_type="random_no_cambia",
            agent=_baseline(1, prt_config),
        ),
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        play_game(seats, prt_config.cambia_rules, num_players=2, seed=712)

    assert agent.agent_state is None, "play_game left a belief handle attached"


def _baseline(seat: int, config):
    from src.evaluate_agents import get_agent

    return get_agent("random_no_cambia", player_id=seat, config=config)


# ---------------------------------------------------------------------------
# The observe_transition half of the contract
# ---------------------------------------------------------------------------


class _FrameProbe:
    """A seat that keeps its own Python-side stream, the shape lbr.py documents.

    Not a NeuralAgentWrapper: what is under test is the loop's fan-out, and the
    hook is duck-typed on purpose so a wrapper opts in by defining it rather
    than by belonging to a class list.
    """

    decides_engine_side = False

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.frames: list = []

    def choose_action(self, game_state, legal_actions):
        return sorted(legal_actions, key=repr)[0]

    def observe_transition(self, view, action, actor) -> None:
        self.frames.append((int(actor), type(action).__name__))


@skiplib
def test_eval_session_feeds_observe_transition_one_frame_per_action(prt_config):
    """_GoEvalGame fires the hook once per applied action, labelled by actor."""
    from src.evaluate_agents import _GoEvalGame

    probes = [_FrameProbe(0), _FrameProbe(1)]
    session = _GoEvalGame(prt_config.cambia_rules, 711, 2, probes)
    applied = 0
    try:
        while not session.is_terminal() and applied < 200:
            acting = session.acting_player()
            legal = session.legal_actions()
            if not legal:
                break
            chosen = probes[acting].choose_action(session.engine, list(legal))
            session.apply(chosen)
            applied += 1
    finally:
        session.close()

    assert applied > 0
    for probe in probes:
        assert len(probe.frames) == applied, (
            f"P{probe.player_id} saw {len(probe.frames)} frames for {applied} "
            "applied actions; a seat with its own stream needs every one"
        )
        assert {actor for actor, _ in probe.frames} <= {0, 1}


@skiplib
def test_play_game_feeds_observe_transition_one_frame_per_action(prt_config):
    """play_game fires the same hook, for the same reason."""
    from src.play import SeatConfig, play_game

    probes = [_FrameProbe(0), _FrameProbe(1)]
    seats = [
        SeatConfig(
            seat_id=i,
            is_human=False,
            name=f"probe{i}",
            agent_type="probe",
            agent=probes[i],
        )
        for i in range(2)
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        play_game(seats, prt_config.cambia_rules, num_players=2, seed=711)

    assert probes[0].frames, "play_game fed no post-action frames at all"
    assert len(probes[0].frames) == len(probes[1].frames), (
        "the two seats saw different numbers of frames; the fan-out is per "
        "transition, not per acting seat"
    )
    # The actor label is the seat that chose, so a wrapper can tell its own
    # moves from its opponent's.
    assert {actor for actor, _ in probes[0].frames} == {0, 1}


# ---------------------------------------------------------------------------
# No second apply path survives
# ---------------------------------------------------------------------------


def test_play_module_has_no_private_apply_path():
    """The interactive loop's own apply is gone, not merely bypassed."""
    import src.play as play_module

    assert not hasattr(play_module, "_apply_action"), (
        "src/play.py still defines a private apply; two apply paths is how the "
        "engine-only one survived unnoticed (cambia-711)"
    )
    assert play_module.TransitionBroadcaster is not None


@skiplib
def test_baseline_only_play_still_applies_without_a_batch_crossing(prt_config):
    """A table with no belief-carrying seat takes the plain engine apply.

    The batched path costs a handle vector per action and buys nothing when no
    seat has a belief to advance, so the broadcaster must not force it.
    """
    from src.agents.transition import TransitionBroadcaster
    from src.ffi.bridge import GoEngine
    from src.agents import action_codec
    import numpy as np

    engine = GoEngine(seed=713, house_rules=prt_config.cambia_rules, num_players=2)
    try:
        broadcast = TransitionBroadcaster(engine, [None, None], 2)
        assert broadcast.belief_agents == []
        rng = random.Random(713)
        applied = 0
        while not engine.is_terminal() and applied < 200:
            mask = engine.legal_actions_mask()
            actions = action_codec.actions_from_mask(mask)
            if not actions:
                break
            index = {a: int(i) for a, i in zip(actions, np.flatnonzero(mask))}
            broadcast.apply(rng.choice(actions), index, engine.acting_player())
            applied += 1
        assert applied > 0
        broadcast.release()
    finally:
        engine.close()
