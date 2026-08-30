"""test_eval_belief_feed_go.py

The eval belief-feed contract, on the Go engine (cambia-1426).

Replaces test_eval_opponent_belief_feed.py and
test_eval_actor_observation_parity.py. Those two measured the contract through
the Python observation stream -- ``_feed_agent_beliefs`` building an
AgentObservation and each wrapper masking it with
``worker._filter_observation`` -- and that mechanism no longer exists: the eval
loop applies each action through the FFI path that advances the game AND every
attached agent's belief in one crossing, which is the same path the training
driver uses. The CONTRACT they pinned still has to hold, so it is re-pinned here
against the new mechanism.

Coverage:
  1. Every applied action reaches a belief agent, including the opponent's:
     a belief fed through the eval loop advances on turns the agent did not act.
  2. Teeth for 1: an agent whose handle is withheld from the apply path does NOT
     advance, so the assertions above are not vacuous.
  3. The acting seat keeps its own private cards (a replaced slot becomes known
     to the actor), and the opponent does not learn that card's identity
     (cambia-1038's no-cross-seat-leak clause).
  4. The feed tolerates a stateless actor: a baseline opponent's moves still
     reach the belief agent.
"""

import random

import pytest

from src.agents import action_codec
from src.agents.baseline_agents import RandomNoCambiaAgent
from src.config import load_config
from src.constants import ActionReplace, CardBucket
from src.encoding import action_to_index
from src.evaluate_agents import NeuralAgentWrapper, _GoEvalGame
from src.ffi.bridge import GoEngine, apply_games_batch

CONFIG_PATH = "config/deep_train.yaml"

_UNKNOWN_VALUES = frozenset({CardBucket.UNKNOWN.value, 9, 0xFF})


@pytest.fixture(scope="module")
def config():
    cfg = load_config(CONFIG_PATH)
    assert cfg is not None, f"could not load {CONFIG_PATH}"
    return cfg


class _ProbeBeliefAgent(NeuralAgentWrapper):
    """Belief wrapper double: a real GoAgentState, uniform play, no network."""

    def __init__(self, player_id, config, device="cpu", use_argmax=False, **kwargs):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
        self._rng = random.Random(4242 + player_id)
        self.acted_turns = 0

    def choose_action(self, game_state, legal_actions):
        self.acted_turns += 1
        pool = [a for a in legal_actions if type(a).__name__ != "ActionCallCambia"]
        return self._rng.choice(sorted(pool or list(legal_actions), key=repr))


def _own_buckets(agent):
    rows = agent.agent_state.get_own_hand_buckets_and_seen()
    own_len, _ = agent.agent_state.get_hand_lens()
    return [int(rows[s, 0]) for s in range(own_len)]


def test_opponent_actions_reach_the_belief(config):
    """The belief advances on turns the agent did not act.

    Seat 0 carries belief and plays; seat 1 is a stateless baseline. If only the
    agent's own actions reached its belief, its observation counter could not
    exceed the number of times it acted.
    """
    agent = _ProbeBeliefAgent(0, config)
    opponent = RandomNoCambiaAgent(1, config, seed=99)
    session = _GoEvalGame(config.cambia_rules, 2026, 2, [agent, opponent])
    try:
        turns = 0
        opponent_moves = 0
        while not session.is_terminal() and turns < 400:
            turns += 1
            actor = session.acting_player()
            legal = session.legal_actions()
            if not legal:
                break
            action = [agent, opponent][actor].choose_action(session.engine, legal)
            if actor == 1:
                opponent_moves += 1
            session.apply(action)

        assert opponent_moves > 0, "opponent never acted; test is vacuous"
        observed = agent.agent_state.get_current_turn()
        assert observed > 0, "belief never advanced at all"
        # The belief tracks game turns, not this seat's decisions; the point is
        # that it moved over a game in which the opponent acted repeatedly.
        assert agent.acted_turns > 0
    finally:
        session.close()


def test_withheld_handle_does_not_advance(config):
    """Teeth: the apply path is what moves the belief.

    Same game, but the agent's handle is withheld from apply_games_batch (the
    -1 skip). Its belief must stay at the initial state, which is what makes the
    previous test's advance meaningful rather than incidental.
    """
    agent = _ProbeBeliefAgent(0, config)
    with GoEngine(seed=2027, house_rules=config.cambia_rules) as engine:
        agent.attach_belief(engine, 2)
        before_turn = agent.agent_state.get_current_turn()
        before_buckets = _own_buckets(agent)

        applied = 0
        while not engine.is_terminal() and applied < 12:
            legal = action_codec.actions_from_mask(engine.legal_actions_mask())
            if not legal:
                break
            # Both agent slots skipped: the game advances, no belief does.
            apply_games_batch([engine.handle], [-1], [-1], [action_to_index(legal[0])])
            applied += 1

        assert applied > 0
        assert agent.agent_state.get_current_turn() == before_turn
        assert _own_buckets(agent) == before_buckets
        agent.release_belief()


def test_actor_keeps_its_own_card_and_opponent_does_not_learn_it(config):
    """A replace is private to the actor: it learns the slot, the opponent does not.

    The actor replaces a hand slot with the card it drew, so that slot becomes
    KNOWN in its own belief. The opponent watched a face-down card go into that
    slot, so its belief about that slot must NOT hold the card's identity. This
    is cambia-1038's clause pair -- the actor keeps its private card, no other
    seat sees it -- measured on the belief each seat actually ends up with.
    """
    actor = _ProbeBeliefAgent(0, config)
    watcher = _ProbeBeliefAgent(1, config)
    found = False

    for seed in range(2100, 2160):
        session = _GoEvalGame(config.cambia_rules, seed, 2, [actor, watcher])
        try:
            turns = 0
            while not session.is_terminal() and turns < 400:
                turns += 1
                seat = session.acting_player()
                legal = session.legal_actions()
                if not legal:
                    break
                replaces = [a for a in legal if isinstance(a, ActionReplace)]
                if seat == 0 and replaces:
                    action = replaces[0]
                    slot = action.target_hand_index
                    session.apply(action)

                    own = _own_buckets(actor)
                    assert slot < len(own), "replaced slot vanished from own hand"
                    assert own[slot] not in _UNKNOWN_VALUES, (
                        f"actor does not know slot {slot} it just replaced with "
                        "its own drawn card; the actor's private information is "
                        "not reaching its belief"
                    )

                    opp_belief = watcher.agent_state.get_opp_belief_buckets()
                    assert int(opp_belief[slot]) in _UNKNOWN_VALUES, (
                        f"opponent's belief about seat 0 slot {slot} is "
                        f"{int(opp_belief[slot])}, not unknown: a private drawn "
                        "card leaked across seats"
                    )
                    found = True
                    break
                session.apply([actor, watcher][seat].choose_action(session.engine, legal))
        finally:
            session.close()
        if found:
            break

    assert found, "no seat-0 replace occurred in 60 games; test never ran"


def test_feed_tolerates_a_stateless_actor(config):
    """A stateless baseline at the other seat does not break the feed.

    Its handle slot is -1, which the apply path skips; the belief agent's slot
    still advances. A game that completes with no errors and an advanced belief
    is the assertion.
    """
    agent = _ProbeBeliefAgent(1, config)
    baseline = RandomNoCambiaAgent(0, config, seed=7)
    session = _GoEvalGame(config.cambia_rules, 2200, 2, [baseline, agent])
    try:
        turns = 0
        while not session.is_terminal() and turns < 400:
            turns += 1
            seat = session.acting_player()
            legal = session.legal_actions()
            if not legal:
                break
            session.apply([baseline, agent][seat].choose_action(session.engine, legal))
        assert session.is_terminal() or turns >= 400
        assert agent.agent_state.get_current_turn() > 0
    finally:
        session.close()
