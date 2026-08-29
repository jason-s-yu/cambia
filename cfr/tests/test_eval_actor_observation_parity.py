"""test_eval_actor_observation_parity.py

The eval belief feed must mask observations the way the TRAINING traversal does
(cambia-1038): the acting seat keeps its own ``drawn_card`` and ``peeked_cards``,
every other seat sees neither.

Before the fix, ``_feed_agent_beliefs`` built a public-only observation and every
``AgentState`` wrapper nulled both fields again, so an evaluated agent never
learned what it drew or peeked. Training (``worker._create_observation`` ->
``worker._filter_observation`` per agent) keeps both for the actor, so every
AgentState-based agent was evaluated on strictly less own-card information than
it was trained with.

Every assertion below reads the observation that actually reaches ``AgentState``
(captured by patching ``AgentState.update``), not a re-derivation of the filter,
so the tests measure the delivered belief input rather than restating the code.

Coverage:
  1. Targeted transitions (own stockpile draw, own replace, own 7/8 peek): what
     an eval wrapper hands its belief equals the training-filtered observation
     for the same transition, field for field.
  2. Whole-game parity: every observation reaching every ``AgentState`` over a
     driven game equals the independently built training-side filtered frame.
  3. No cross-seat leak: no observation delivered to a non-acting seat ever
     carries a drawn or peeked card.
  4. Teeth: the actor's frames really do carry private cards (otherwise 1-3
     would pass vacuously), and the belief reached differs from the one the
     pre-fix public-only feed produced.
"""

import copy
import random
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

from src.agent_state import AgentObservation, AgentState
from src.cfr.worker import _create_observation, _filter_observation
from src.config import load_config
from src.constants import (
    ActionAbilityPeekOwnSelect,
    ActionDrawStockpile,
    ActionReplace,
)
from src.evaluate_agents import (
    NeuralAgentWrapper,
    _build_public_observation,
    _build_transition_observation,
    _feed_agent_beliefs,
)
from src.game.engine import CambiaGameState

CONFIG_PATH = "runs/eppbs-2p/config.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


class _ProbeBeliefAgent(NeuralAgentWrapper):
    """Belief wrapper double: real AgentState and the real inherited
    ``update_state``, uniform play, no network. Avoids calling Cambia so games
    run long enough for both seats to draw, replace and use abilities."""

    def __init__(self, player_id, config, seed: int = 0):
        super().__init__(player_id, config, device="cpu", use_argmax=False)
        self._rng = random.Random(seed + player_id)

    def choose_action(self, game_state, legal_actions):
        pool = [a for a in legal_actions if type(a).__name__ != "ActionCallCambia"]
        return self._rng.choice(sorted(pool or list(legal_actions), key=repr))


def _training_frame(
    game_state, action, acting_player: int, observer: int
) -> Optional[AgentObservation]:
    """The observation the TRAINING traversal would hand ``observer``'s belief
    for this transition, built here from the worker's own two functions."""
    full = _create_observation(
        None,
        action,
        game_state,
        acting_player,
        copy.deepcopy(game_state.snap_results_log),
    )
    if full is None:
        return None
    return _filter_observation(full, observer)


def _prefix_strip(obs: AgentObservation) -> AgentObservation:
    """The pre-cambia-1038 eval filter: null both private fields for every seat."""
    stripped = copy.copy(obs)
    stripped.drawn_card = None
    stripped.peeked_cards = None
    return stripped


def _recorder():
    """(list, patch) recording every (seat, observation) reaching an AgentState."""
    recorded: List[Tuple[int, AgentObservation]] = []
    original = AgentState.update

    def _recording_update(self, observation):
        recorded.append((self.player_id, copy.copy(observation)))
        return original(self, observation)

    return recorded, patch.object(AgentState, "update", _recording_update)


def _delivered(agent, observation) -> AgentObservation:
    """The observation ``agent.update_state(observation)`` hands its AgentState."""
    recorded, patcher = _recorder()
    with patcher:
        agent.update_state(observation)
    assert len(recorded) == 1, f"expected one belief update, got {len(recorded)}"
    assert recorded[0][0] == agent.player_id
    return recorded[0][1]


def _fresh_belief(config, seed: int, player_id: int = 0) -> AgentState:
    """A seat-0 belief initialized at the start of the game with this deck seed."""
    state = CambiaGameState(house_rules=config.cambia_rules, seed=seed)
    belief = AgentState(
        player_id=player_id,
        opponent_id=1 - player_id,
        memory_level=config.agent_params.memory_level,
        time_decay_turns=config.agent_params.time_decay_turns,
        initial_hand_size=len(state.players[player_id].hand),
        config=config,
    )
    belief.initialize(
        _build_public_observation(state, None, -1),
        state.players[player_id].hand,
        state.players[player_id].initial_peek_indices,
    )
    return belief


def _signature(state: AgentState):
    return (
        dict(state.own_hand),
        dict(state.opponent_belief),
        state.opponent_card_count,
        dict(state.opponent_last_seen_turn),
    )


def _new_agents(config, game_state, seed: int) -> List[_ProbeBeliefAgent]:
    agents = [_ProbeBeliefAgent(0, config, seed), _ProbeBeliefAgent(1, config, seed)]
    for agent in agents:
        agent.initialize_state(game_state)
    return agents


def _drive_until(config, seed: int, predicate, max_actions: int = 300):
    """Play a seeded uniform-random game, feeding beliefs, and stop right after
    the first action satisfying ``predicate`` is applied but BEFORE feeding it.

    Returns (game_state, action, acting_seat, agents) or None if the game ends
    without one.
    """
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=seed)
    agents = _new_agents(config, game_state, seed)
    rng = random.Random(seed)
    for _ in range(max_actions):
        if game_state.is_terminal():
            return None
        acting = game_state.get_acting_player()
        if acting == -1:
            return None
        legal = sorted(game_state.get_legal_actions(), key=repr)
        if not legal:
            return None
        action = rng.choice(legal)
        game_state.apply_action(action)
        if predicate(action):
            return game_state, action, acting, agents
        _feed_agent_beliefs(agents, game_state, action, acting)
    return None


def _find(config, predicate, seeds=range(2000, 2400)):
    for seed in seeds:
        found = _drive_until(config, seed, predicate)
        if found is not None:
            return found
    pytest.fail("no matching transition reached in the searched seed range")


# --- 1. Targeted transitions -------------------------------------------------


def test_own_draw_reaches_the_actor_belief(config):
    """The actor's own drawn card survives the eval filter on a stockpile draw."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=10381)
    actor = game_state.get_acting_player()
    agents = _new_agents(config, game_state, seed=10381)
    draw = next(
        a for a in game_state.get_legal_actions() if isinstance(a, ActionDrawStockpile)
    )
    game_state.apply_action(draw)

    shared = _build_transition_observation(game_state, draw, actor)
    assert shared is not None
    assert shared.drawn_card is not None, "engine did not surface the drawn card"
    assert _prefix_strip(shared).drawn_card is None, "pre-fix filter dropped it"

    actor_view = _delivered(agents[actor], shared)
    assert actor_view == _training_frame(game_state, draw, actor, actor)
    assert actor_view.drawn_card == shared.drawn_card

    opp_view = _delivered(agents[1 - actor], shared)
    assert opp_view == _training_frame(game_state, draw, actor, 1 - actor)
    assert opp_view.drawn_card is None


def test_own_replace_reaches_the_actor_belief(config):
    """The card placed into the actor's own hand survives the eval filter."""
    found = _drive_until(
        config, 10381, lambda a: isinstance(a, ActionReplace), max_actions=60
    )
    assert found is not None, "no replace reached"
    game_state, action, actor, agents = found

    shared = _build_transition_observation(game_state, action, actor)
    assert shared is not None
    assert shared.drawn_card is not None
    assert _prefix_strip(shared).drawn_card is None

    actor_view = _delivered(agents[actor], shared)
    assert actor_view == _training_frame(game_state, action, actor, actor)
    assert actor_view.drawn_card == shared.drawn_card
    assert _delivered(agents[1 - actor], shared).drawn_card is None


def test_own_peek_reaches_the_actor_belief(config):
    """The actor's own 7/8 peek survives the eval filter; the opponent sees none."""
    game_state, action, actor, agents = _find(
        config, lambda a: isinstance(a, ActionAbilityPeekOwnSelect)
    )

    shared = _build_transition_observation(game_state, action, actor)
    assert shared is not None
    assert shared.peeked_cards, "engine did not surface the peeked card"
    assert _prefix_strip(shared).peeked_cards is None

    actor_view = _delivered(agents[actor], shared)
    assert actor_view == _training_frame(game_state, action, actor, actor)
    assert actor_view.peeked_cards == shared.peeked_cards

    opp_view = _delivered(agents[1 - actor], shared)
    assert opp_view == _training_frame(game_state, action, actor, 1 - actor)
    assert opp_view.peeked_cards is None


def test_public_builder_still_carries_no_private_cards(config):
    """``_build_public_observation`` keeps its public-only contract (it still
    builds the pre-first-action initial observation)."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=10381)
    actor = game_state.get_acting_player()
    draw = next(
        a for a in game_state.get_legal_actions() if isinstance(a, ActionDrawStockpile)
    )
    game_state.apply_action(draw)
    public = _build_public_observation(game_state, draw, actor)
    assert public is not None
    assert public.drawn_card is None
    assert public.peeked_cards is None


# --- 2/3/4. Whole-game parity, leak check and teeth --------------------------


def _drive_game(config, agents, seed: int, max_actions: int = 400):
    """Play one game through the eval feed, returning the independently built
    training-side expectation per transition (one frame per seat)."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=seed)
    for agent in agents:
        agent.initialize_state(game_state)

    expected: List[Tuple[int, List[Optional[AgentObservation]]]] = []
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
        expected.append(
            (
                acting,
                [_training_frame(game_state, action, acting, seat) for seat in (0, 1)],
            )
        )
        _feed_agent_beliefs(agents, game_state, action, acting)
    return game_state, expected


def test_whole_game_actor_filtering_matches_training(config):
    """Every observation reaching every AgentState equals the training-side
    filtered frame for the same transition, and no private card crosses seats."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=1038_2026)
    agents = _new_agents(config, game_state, seed=1038)
    recorded, patcher = _recorder()
    with patcher:
        del recorded[:]  # drop anything initialization produced
        _, expected = _drive_game(config, agents, seed=1038_2026)

    assert expected, "game produced no transitions"
    assert len(recorded) == 2 * len(expected), (
        f"{len(recorded)} belief updates for {len(expected)} transitions; "
        "the feed did not deliver one frame per seat per transition"
    )

    actor_drawn = 0
    actor_peeked = 0
    for i, (acting, per_seat) in enumerate(expected):
        for seat in (0, 1):
            got_seat, got_obs = recorded[2 * i + seat]
            assert got_seat == seat
            assert got_obs == per_seat[seat], (
                f"transition {i} (actor P{acting}) delivered to P{seat} diverges "
                f"from the training-filtered frame:\n"
                f"  eval : drawn={got_obs.drawn_card} peeked={got_obs.peeked_cards}\n"
                f"  train: drawn={per_seat[seat].drawn_card} "
                f"peeked={per_seat[seat].peeked_cards}"
            )
            if seat != acting:
                assert (
                    got_obs.drawn_card is None
                ), f"transition {i}: P{acting}'s drawn card leaked to P{seat}"
                assert (
                    got_obs.peeked_cards is None
                ), f"transition {i}: P{acting}'s peeked cards leaked to P{seat}"
            else:
                actor_drawn += got_obs.drawn_card is not None
                actor_peeked += bool(got_obs.peeked_cards)

    assert actor_drawn > 0, (
        "no actor frame carried a drawn card; the parity and leak assertions "
        "would pass vacuously"
    )
    assert actor_peeked > 0, (
        "no actor frame carried a peeked card; the parity and leak assertions "
        "would pass vacuously"
    )


def test_actor_private_info_changes_the_belief(config):
    """The fix is observable in the belief itself, not just in the observation
    object: replaying the same transitions through the pre-fix public-only
    filter reaches a different AgentState."""
    game_state = CambiaGameState(house_rules=config.cambia_rules, seed=1038_2027)
    agents = _new_agents(config, game_state, seed=99)
    recorded, patcher = _recorder()
    with patcher:
        del recorded[:]
        _drive_game(config, agents, seed=1038_2027)

    seat0 = [obs for seat, obs in recorded if seat == 0]
    assert seat0, "no frames delivered to seat 0"
    assert any(
        obs.drawn_card is not None or obs.peeked_cards for obs in seat0
    ), "seat 0 never received private information; the comparison has no teeth"

    fixed = _fresh_belief(config, 1038_2027)
    prefix = _fresh_belief(config, 1038_2027)
    diverged_at = None
    for i, obs in enumerate(seat0):
        fixed.update(obs)
        prefix.update(_prefix_strip(obs))
        if diverged_at is None and _signature(fixed) != _signature(prefix):
            diverged_at = i

    assert diverged_at is not None, (
        "the public-only replay tracks the fixed feed's belief exactly; the "
        "actor's private cards are not reaching the belief"
    )
    assert _signature(fixed) == _signature(agents[0].agent_state), (
        "replaying the delivered frames from scratch does not reproduce the "
        "belief the eval feed built"
    )
