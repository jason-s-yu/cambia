"""test_eval_opponent_belief_feed.py

The eval belief-feed contract: every applied action reaches every stateful
agent's belief model, in application order, regardless of who acted. A stateful
wrapper facing a stateless baseline must still see the baseline's moves; a
belief that only ever saw its own actions is not the belief training produced.

Coverage:
  1. run_evaluation delivers the full applied-action sequence (both seats) to a
     belief wrapper.
  2. The belief reached through the eval feed equals a from-scratch
     reconstruction replayed over an independently recorded transcript that
     includes the opponent's actions, and differs from an own-actions-only
     replay (the frames carry information, so dropping them is observable).
"""

import copy
import random
from typing import List, Tuple
from unittest.mock import patch

import pytest

from src.agent_state import AgentObservation, AgentState
from src.agents.baseline_agents import RandomNoCambiaAgent
from src.config import load_config
from src.constants import NUM_PLAYERS
from src.evaluate_agents import (
    NeuralAgentWrapper,
    _feed_agent_beliefs,
    get_agent,
    run_evaluation,
)
from src.game.engine import CambiaGameState

CONFIG_PATH = "runs/eppbs-2p/config.yaml"


class _ProbeBeliefAgent(NeuralAgentWrapper):
    """Belief wrapper double: real AgentState, uniform play, no network.

    Avoids calling Cambia while another action is available so games run long
    enough for the opponent to act repeatedly.
    """

    def __init__(self, player_id, config, device="cpu", use_argmax=False, **kwargs):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
        self._rng = random.Random(20250828 + player_id)
        self.delivered: List[Tuple[int, str]] = []

    def choose_action(self, game_state, legal_actions):
        pool = [a for a in legal_actions if type(a).__name__ != "ActionCallCambia"]
        return self._rng.choice(sorted(pool or list(legal_actions), key=repr))

    def update_state(self, observation):
        self.delivered.append(
            (observation.acting_player, type(observation.action).__name__)
        )
        super().update_state(observation)


def _public_obs(
    game_state: CambiaGameState, action, acting_player: int
) -> AgentObservation:
    """Independent transcript frame, built in the test rather than by the code
    under test, so the reconstruction is not a restatement of the feed."""
    return AgentObservation(
        acting_player=acting_player,
        action=action,
        discard_top_card=game_state.get_discard_top(),
        player_hand_sizes=[
            game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
        ],
        stockpile_size=game_state.get_stockpile_size(),
        drawn_card=None,
        peeked_cards=None,
        snap_results=copy.deepcopy(game_state.snap_results_log),
        did_cambia_get_called=game_state.cambia_caller_id is not None,
        who_called_cambia=game_state.cambia_caller_id,
        is_game_over=game_state.is_terminal(),
        current_turn=game_state.get_turn_number(),
    )


def _fresh_state(config, game_state: CambiaGameState, player_id: int) -> AgentState:
    """A belief state initialized at the start of `game_state`, built here from
    the engine directly."""
    state = AgentState(
        player_id=player_id,
        opponent_id=1 - player_id,
        memory_level=config.agent_params.memory_level,
        time_decay_turns=config.agent_params.time_decay_turns,
        initial_hand_size=len(game_state.players[player_id].hand),
        config=config,
    )
    state.initialize(
        _public_obs(game_state, None, -1),
        game_state.players[player_id].hand,
        game_state.players[player_id].initial_peek_indices,
    )
    return state


def _signature(state: AgentState):
    return (
        dict(state.own_hand),
        dict(state.opponent_belief),
        state.opponent_card_count,
        dict(state.opponent_last_seen_turn),
        {k: list(v) for k, v in state.action_history.items()},
    )


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def test_run_evaluation_delivers_every_applied_action(config):
    """Both seats' actions reach the belief wrapper, in application order."""
    applied: List[Tuple[int, str]] = []
    probes: List[_ProbeBeliefAgent] = []

    def _probe_get_agent(agent_type, player_id, config, **kwargs):
        if agent_type == "probe_belief":
            agent = _ProbeBeliefAgent(player_id, config)
            probes.append(agent)
            return agent
        return get_agent(agent_type, player_id=player_id, config=config, **kwargs)

    original_apply = CambiaGameState.apply_action

    def _recording_apply(self, action):
        applied.append((self.get_acting_player(), type(action).__name__))
        return original_apply(self, action)

    with (
        patch("src.evaluate_agents.load_config", return_value=config),
        patch("src.evaluate_agents.get_agent", side_effect=_probe_get_agent),
        patch.object(CambiaGameState, "apply_action", _recording_apply),
    ):
        run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="probe_belief",
            agent2_type="random_no_cambia",
            num_games=2,
            strategy_path=None,
            seat_scheme="fixed",
            crn_seed_base=709,
            crn_identity="cambia-709-feed",
        )

    assert probes, "probe wrapper was never instantiated"
    delivered = [d for probe in probes for d in probe.delivered]

    opponent_applied = [a for a in applied if a[0] == 1]
    assert opponent_applied, "opponent never acted; test cannot discriminate"
    assert delivered == applied, (
        "belief feed does not match the applied-action sequence: "
        f"{len(delivered)} delivered vs {len(applied)} applied "
        f"({len([d for d in delivered if d[0] == 1])} of "
        f"{len(opponent_applied)} opponent actions delivered)"
    )


def test_belief_matches_from_scratch_reconstruction(config):
    """The fed belief equals a replay of the full transcript, and an
    own-actions-only replay diverges."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=709_2026)

    agent = _ProbeBeliefAgent(0, config)
    opponent = RandomNoCambiaAgent(1, config, seed=4242)
    agents = [agent, opponent]
    agent.initialize_state(game_state)

    reference = _fresh_state(config, game_state, player_id=0)
    own_only = _fresh_state(config, game_state, player_id=0)

    transcript: List[Tuple[int, AgentObservation]] = []
    max_actions = 400
    acted = 0
    while not game_state.is_terminal() and acted < max_actions:
        acting = game_state.get_acting_player()
        if acting == -1:
            break
        legal = game_state.get_legal_actions()
        if not legal:
            break
        action = agents[acting].choose_action(game_state, legal)
        game_state.apply_action(action)
        acted += 1
        transcript.append((acting, _public_obs(game_state, action, acting)))
        _feed_agent_beliefs(agents, game_state, action, acting)

    assert any(actor == 1 for actor, _ in transcript), "opponent never acted"

    for _, obs in transcript:
        reference.update(obs)
    for actor, obs in transcript:
        if actor == 0:
            own_only.update(obs)

    assert _signature(agent.agent_state) == _signature(reference), (
        "belief reached through the eval feed diverges from a from-scratch "
        "replay of the same transitions"
    )
    assert _signature(own_only) != _signature(reference), (
        "own-actions-only replay is indistinguishable from the full replay; "
        "this game carries no opponent information, so the test has no teeth"
    )


def test_belief_feed_tolerates_stateless_actor(config):
    """A stateless actor's move still updates the belief wrapper."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=709_2027)
    agent = _ProbeBeliefAgent(0, config)
    opponent = RandomNoCambiaAgent(1, config, seed=99)
    agents = [agent, opponent]
    agent.initialize_state(game_state)

    opponent_frames = 0
    acted = 0
    while not game_state.is_terminal() and acted < 60:
        acting = game_state.get_acting_player()
        if acting == -1:
            break
        legal = game_state.get_legal_actions()
        if not legal:
            break
        action = agents[acting].choose_action(game_state, legal)
        game_state.apply_action(action)
        acted += 1
        _feed_agent_beliefs(agents, game_state, action, acting)
        if acting == 1:
            opponent_frames += 1

    assert opponent_frames > 0
    assert [a for a, _ in agent.delivered].count(1) == opponent_frames
