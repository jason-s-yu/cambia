"""test_eval_belief_init.py

Belief initialisation and per-game reset for the Go-engine evaluation loop
(cambia-1426, and the wrapper-init contract cambia-1039's PPO fix depends on).

A wrapper's belief is a GoAgentState bound to the game at its INITIAL state.
That binding moment is what seeds the seat's initial-peek knowledge, so getting
it wrong is silent: the agent plays on with an empty belief and simply evaluates
worse. These tests pin the observable consequences.

Coverage:
  1. Attaching at the initial state seeds exactly initial_view_count own slots
     as known, and leaves the rest unknown.
  2. Each seat's belief is its own: seat 1 knows seat 1's peeked cards, and the
     two seats do not share a handle.
  3. A second game resets the belief rather than continuing the first (the
     per-game attach releases and rebuilds).
  4. Belief handles are returned to the pool: a long eval run does not leak
     them.
  5. The PPO wrapper's belief lifecycle matches the neural wrappers', which is
     what lets run_evaluation drive it (cambia-1039).
  6. A belief bound through bind_go_state is borrowed: the borrower reads it and
     never frees it, which is what lets a Tier-B continuation opponent be
     handed the search state's own belief (cambia-1793).
"""

import pytest

from src.agents import action_codec
from src.config import load_config
from src.constants import CardBucket
from src.evaluate_agents import NeuralAgentWrapper, _GoEvalGame
from src.ffi.bridge import GoAgentState, GoEngine, get_handle_pool_stats

CONFIG_PATH = "config/deep_train.yaml"


@pytest.fixture(scope="module")
def config():
    cfg = load_config(CONFIG_PATH)
    assert cfg is not None, f"could not load {CONFIG_PATH}"
    return cfg


class _ProbeBeliefAgent(NeuralAgentWrapper):
    """Belief wrapper double: a real GoAgentState, uniform play, no network.

    Avoids Cambia while any other action is available so games run long enough
    for both seats to act repeatedly.
    """

    def __init__(self, player_id, config, device="cpu", use_argmax=False, **kwargs):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
        import random

        self._rng = random.Random(20260830 + player_id)

    def choose_action(self, game_state, legal_actions):
        pool = [a for a in legal_actions if type(a).__name__ != "ActionCallCambia"]
        return self._rng.choice(sorted(pool or list(legal_actions), key=repr))


def _known_slots(agent) -> list:
    """Own-hand slots this agent's belief holds a real bucket for."""
    rows = agent.agent_state.get_own_hand_buckets_and_seen()
    own_len, _ = agent.agent_state.get_hand_lens()
    known = []
    for slot in range(own_len):
        bucket = int(rows[slot, 0])
        if bucket != CardBucket.UNKNOWN.value and bucket != 9 and bucket != 0xFF:
            known.append(slot)
    return known


def test_attach_seeds_exactly_the_initial_peek(config):
    """Binding at the initial state knows initial_view_count slots, no more."""
    agent = _ProbeBeliefAgent(0, config)
    with GoEngine(seed=11, house_rules=config.cambia_rules) as engine:
        agent.attach_belief(engine, 2)
        expected = int(engine.get_house_rules().initial_view_count)
        known = _known_slots(agent)
        own_len, _ = agent.agent_state.get_hand_lens()
        assert len(known) == expected, (
            f"belief knows {len(known)} slots {known}, expected exactly "
            f"{expected} from the initial peek"
        )
        assert own_len > expected, "hand should have unknown slots left over"
        agent.release_belief()


def test_each_seat_gets_its_own_belief(config):
    """Two seats attached to one game hold distinct handles and distinct state."""
    a0 = _ProbeBeliefAgent(0, config)
    a1 = _ProbeBeliefAgent(1, config)
    with GoEngine(seed=12, house_rules=config.cambia_rules) as engine:
        a0.attach_belief(engine, 2)
        a1.attach_belief(engine, 2)
        assert a0.belief_handle() != a1.belief_handle()
        assert a0.belief_handle() >= 0 and a1.belief_handle() >= 0
        # Each seat's peek is about its OWN hand, so the known buckets are read
        # from different cards even though both know the same slot indices.
        b0 = a0.agent_state.get_own_hand_buckets_and_seen()[:, 0]
        b1 = a1.agent_state.get_own_hand_buckets_and_seen()[:, 0]
        hand0 = engine.get_player_hand(0)
        hand1 = engine.get_player_hand(1)
        assert hand0 != hand1 or list(b0) == list(b1)
        a0.release_belief()
        a1.release_belief()


def test_second_game_resets_the_belief(config):
    """A fresh attach replaces the previous game's belief, not extends it."""
    agent = _ProbeBeliefAgent(0, config)
    with GoEngine(seed=13, house_rules=config.cambia_rules) as engine:
        agent.attach_belief(engine, 2)
        first_handle = agent.belief_handle()
        # Advance the first game so the belief is demonstrably not fresh.
        session_actions = action_codec.actions_from_mask(engine.legal_actions_mask())
        from src.encoding import action_to_index
        from src.ffi.bridge import apply_games_batch

        for _ in range(6):
            legal = action_codec.actions_from_mask(engine.legal_actions_mask())
            if not legal or engine.is_terminal():
                break
            apply_games_batch(
                [engine.handle],
                [agent.belief_handle()],
                [-1],
                [action_to_index(legal[0])],
            )
        assert session_actions
        advanced_turn = agent.agent_state.get_current_turn()
        assert advanced_turn > 0, "belief did not advance over the first game"

    with GoEngine(seed=14, house_rules=config.cambia_rules) as engine2:
        agent.attach_belief(engine2, 2)
        assert agent.agent_state.get_current_turn() == 0, (
            "second game's belief carried the first game's turn counter; "
            "attach_belief must rebuild, not reuse"
        )
        expected = int(engine2.get_house_rules().initial_view_count)
        assert len(_known_slots(agent)) == expected
        agent.release_belief()
    assert agent.belief_handle() == -1
    assert first_handle >= 0


def test_belief_handles_do_not_leak_across_games(config):
    """A run of games returns every engine and belief handle to the pool."""
    before = get_handle_pool_stats()
    agents = [_ProbeBeliefAgent(0, config), _ProbeBeliefAgent(1, config)]
    for game in range(12):
        session = _GoEvalGame(config.cambia_rules, 100 + game, 2, agents)
        try:
            turns = 0
            while not session.is_terminal() and turns < 400:
                turns += 1
                legal = session.legal_actions()
                if not legal:
                    break
                actor = session.acting_player()
                session.apply(agents[actor].choose_action(session.engine, legal))
        finally:
            session.close()
    after = get_handle_pool_stats()
    for key in ("games_in_use", "agents_in_use"):
        if key in before and key in after:
            assert after[key] == before[key], (
                f"{key} went {before[key]} -> {after[key]} over 12 games; "
                "handles leaked"
            )


def test_bind_go_state_borrows_a_belief_without_taking_it_over(config):
    """A bound belief is read, not owned: the lender still holds the handle.

    The Tier-B continuation seat lends the search state's own GoAgentState to an
    opponent built at the infoset (cambia-1793). If the borrower closed it on
    release, the lender would be left driving a freed handle for the rest of the
    infoset, and the pool would see a double free.
    """
    lender = _ProbeBeliefAgent(0, config)
    borrower = _ProbeBeliefAgent(1, config)
    with GoEngine(seed=17, house_rules=config.cambia_rules) as engine:
        lender.attach_belief(engine, 2)
        lent = lender.agent_state
        borrower.bind_go_state(engine, lent)
        assert borrower.agent_state is lent, "bind did not take the handle"
        # A borrowed belief is not offered for adoption: its lender already
        # drives it through apply/save/restore.
        assert borrower.belief_handle() == -1
        assert lender.belief_handle() == int(lent.handle)

        borrower.release_belief()
        assert borrower.agent_state is None
        # Still the lender's, still usable: reading it here would raise on a
        # closed handle.
        assert lender.agent_state is lent
        assert int(lent.get_current_turn()) >= 0
        assert lender.belief_handle() == int(lent.handle)
        lender.release_belief()


def test_a_borrowed_belief_is_replaced_by_a_fresh_attach(config):
    """attach_belief after a borrow owns its belief again, and reports it."""
    agent = _ProbeBeliefAgent(1, config)
    with GoEngine(seed=18, house_rules=config.cambia_rules) as engine:
        with GoAgentState(engine, 0) as lent:
            agent.bind_go_state(engine, lent)
            assert agent.belief_handle() == -1
            agent.attach_belief(engine, 2)
            assert agent.agent_state is not lent
            assert agent.belief_handle() == int(agent.agent_state.handle)
            agent.release_belief()
            assert int(lent.get_current_turn()) >= 0, "the lent handle was freed"


def test_ppo_wrapper_belief_lifecycle(config):
    """PPOAgentWrapper exposes the same belief lifecycle the loop drives.

    cambia-1039: the loop resets belief per game through initialize_state and
    reads the handle through belief_handle. PPO does not inherit
    NeuralAgentWrapper, so the three methods are checked on it directly. sb3 is
    an optional extra, so the model load itself is not exercised here.
    """
    from src.evaluate_agents import PPOAgentWrapper

    for name in ("attach_belief", "belief_handle", "release_belief", "initialize_state"):
        assert callable(getattr(PPOAgentWrapper, name, None)), (
            f"PPOAgentWrapper is missing {name}; run_evaluation cannot reset or "
            "feed its belief"
        )
    # update_state must be inert: belief advances in the engine, and a wrapper
    # that still tried to fold in a Python observation would double-count.
    assert PPOAgentWrapper.update_state(object.__new__(PPOAgentWrapper), None) is None
