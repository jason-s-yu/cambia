"""
tests/test_self_snap_belief_parity.py

A self-snap that closes the snap window keeps the survivor (cambia-1985).

When a seat snaps one of its own cards and that snap closes the window, both
engines clear the snap-results log inside the same apply that appended the
entry. That clear is deliberate and mirrors the Go tokenizer, which emits public
snap frames only while ``Snap.Active``. The Python belief was reading the same
channel, so the entry naming ``removed_own_index`` never reached it: told only
that the hand had shrunk, it truncated from the end, keeping the removed card's
bucket at the vacated slot and dropping the card that actually survived. The Go
agent reads the engine directly and was right all along, which is the ruling
this pins.

The fix carries those entries on ``AgentObservation.closing_snap_results``, a
belief-only channel, so not a single token moves. These tests pin all three
halves of that: the two backends agree over real games, the surviving card is
the one the belief keeps, and the Python engine publishes the entry its own
clear would have destroyed.

Agreement alone would pass if both backends were wrong together, so the survivor
assertions below name the true card rather than comparing the two beliefs to
each other.
"""

import random

import pytest

CONFIG = "config/tiny_cambia_tabular.yaml"

#: Deals to walk. The tiny config's windows are single-snapper
#: (allowOpponentSnapping is off), which is exactly the shape that loses its
#: entry, and 60 deals of random play reach a few dozen of them.
DEALS = 60

#: Go's BucketUnknown, which the Python belief spells 99.
GO_UNKNOWN = 9


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        engine = GoEngine.from_deck(list(range(54)))
        engine.close()
        return True
    except Exception:
        return False


skip_if_no_go = pytest.mark.skipif(
    not _go_available(), reason="libcambia.so not available"
)


def _config():
    from src.config import load_config

    return load_config(CONFIG)


def _go_belief(agent):
    """The Go agent's own-hand and opponent beliefs in Python's bucket domain."""
    from src.constants import CardBucket

    unknown = CardBucket.UNKNOWN.value
    own_len, opp_len = agent.get_hand_lens()
    own_raw = agent.get_own_hand_buckets_and_seen()
    own = tuple(
        unknown if int(own_raw[i, 0]) == GO_UNKNOWN else int(own_raw[i, 0])
        for i in range(own_len)
    )
    opp_raw = agent.get_opp_belief_buckets()
    opp = tuple(
        unknown if int(opp_raw[i]) == GO_UNKNOWN else int(opp_raw[i])
        for i in range(opp_len)
    )
    return own, opp, int(opp_len)


def _new_python_beliefs(cfg, state):
    """Two Python AgentStates seeded exactly as the traversal seeds them."""
    from src.agent_state import AgentState

    initial_obs = state.initial_observation(ability_reveals=True)
    hands = [state.hand(i) for i in range(2)]
    peeks = state.initial_peek_indices()
    agents = []
    for seat in range(2):
        agent = AgentState(
            player_id=seat,
            opponent_id=1 - seat,
            memory_level=cfg.agent_params.memory_level,
            time_decay_turns=cfg.agent_params.time_decay_turns,
            initial_hand_size=len(hands[seat]),
            config=cfg,
        )
        agent.initialize(initial_obs, hands[seat], peeks)
        agents.append(agent)
    return agents


def _deal(cfg, seed):
    from src.cfr.br_state import DealSpec
    from src.ffi.bridge import extract_deck_from_python_game
    from src.game.engine import CambiaGameState

    pygame = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(seed))
    deck, start = extract_deck_from_python_game(pygame)
    return DealSpec(deck=tuple(int(c) for c in deck), starting_player=int(start))


def _updated(agent, observation):
    clone = agent.clone()
    clone.update(observation)
    return clone


def _walk(cfg, seed, rng, on_node=None, on_self_snap=None):
    """One random playout on the Go engine with both beliefs live.

    ``on_node`` sees (seat, python_agent, go_agent) before every action;
    ``on_self_snap`` sees (seat, python_agent, go_agent, true_hand) right after a
    successful own snap that closed the window.
    """
    from src.cfr.br_state import GoBrState
    from src.cfr.worker import _filter_observation
    from src.constants import ActionSnapOwn
    from src.ffi.bridge import GoAgentState

    state = GoBrState.new(cfg.cambia_rules, _deal(cfg, seed))
    go_agents = []
    try:
        py_agents = _new_python_beliefs(cfg, state)
        go_agents = [
            GoAgentState(
                state.engine,
                seat,
                memory_level=cfg.agent_params.memory_level,
                time_decay_turns=cfg.agent_params.time_decay_turns,
            )
            for seat in range(2)
        ]

        for _ in range(200):
            if state.is_terminal():
                break
            if on_node is not None:
                for seat in range(2):
                    on_node(seat, py_agents[seat], go_agents[seat])

            pairs = state.legal_actions()
            action_idx, action = pairs[rng.randrange(len(pairs))]
            actor = state.acting_player()
            window_open = state.decision_context().name == "SNAP_DECISION"
            hand_before = len(state.hand(actor))

            state.apply(action_idx)
            observation = state.observation(action, actor, ability_reveals=True)
            py_agents = [
                _updated(py_agents[seat], _filter_observation(observation, seat))
                for seat in range(2)
            ]
            state.engine.update_both(go_agents[0], go_agents[1])

            closed_self_snap = (
                isinstance(action, ActionSnapOwn)
                and window_open
                and len(state.hand(actor)) < hand_before
                and not state.engine.get_snap_state().active
            )
            # Terminal states are excluded: the traversal returns utilities there
            # and never builds an infoset key, and the two backends do diverge on a
            # snap that empties the hand and ends the game, where Go shrinks the
            # belief to no slots and the Python reconciliation keeps a phantom one
            # (cambia-1985 F1, filed, not fixed here). Every state a belief is
            # actually read at is covered.
            if closed_self_snap and on_self_snap is not None and not state.is_terminal():
                on_self_snap(actor, py_agents[actor], go_agents[actor], state.hand(actor))
    finally:
        for agent in go_agents:
            agent.close()
        state.close()


@skip_if_no_go
def test_the_two_beliefs_agree_over_real_games():
    """No seat-node disagrees on the infoset key's belief components."""
    cfg = _config()
    rng = random.Random(90210)
    mismatches = []
    nodes = 0

    def compare(seat, py_agent, go_agent):
        nonlocal nodes
        nodes += 1
        python = py_agent.get_infoset_key()[:3]
        go = _go_belief(go_agent)
        if python != go:
            mismatches.append((seat, python, go))

    for seed in range(DEALS):
        _walk(cfg, 90210 + seed, rng, on_node=compare)

    assert nodes > 500, f"only {nodes} seat-nodes walked; the games did not advance"
    assert (
        not mismatches
    ), f"{len(mismatches)} of {nodes} seat-nodes disagree; first: {mismatches[0]}"


@skip_if_no_go
def test_a_window_closing_self_snap_is_actually_reached():
    """Teeth for the test above: the defective case really occurs in these deals."""
    cfg = _config()
    rng = random.Random(90210)
    seen = 0

    def count(*_args):
        nonlocal seen
        seen += 1

    for seed in range(DEALS):
        _walk(cfg, 90210 + seed, rng, on_self_snap=count)

    assert seen > 0, (
        "no successful own snap closed a window across the walked deals, so the "
        "agreement test above proves nothing about this path"
    )
    assert seen >= 20, f"only {seen} window-closing self snaps; the sample thinned" '"'


@skip_if_no_go
def test_the_belief_keeps_the_card_that_survived_the_self_snap():
    """The surviving card's bucket, not the removed card's, on both backends.

    Agreement is not enough on its own: this names the true hand, so the pair
    cannot pass by being wrong together.
    """
    from src.abstraction import get_card_bucket
    from src.constants import CardBucket

    cfg = _config()
    rng = random.Random(90210)
    checked = 0

    def check(seat, py_agent, go_agent, true_hand):
        nonlocal checked
        checked += 1
        truth = tuple(get_card_bucket(card).value for card in true_hand)
        python = py_agent.get_infoset_key()[0]
        go = _go_belief(go_agent)[0]
        assert len(python) == len(
            truth
        ), f"seat {seat}: belief holds {len(python)} slots for a {len(truth)}-card hand"
        for slot, (believed, actual) in enumerate(zip(python, truth)):
            # A slot the seat never saw stays unknown; a slot it believes it knows
            # must name the card that is actually there.
            assert believed in (actual, CardBucket.UNKNOWN.value), (
                f"seat {seat} slot {slot}: belief says {believed}, "
                f"hand holds {actual}"
            )
        assert go == python, f"seat {seat}: Go {go} and Python {python} disagree"

    for seed in range(DEALS):
        _walk(cfg, 90210 + seed, rng, on_self_snap=check)

    assert checked > 0, "no window-closing self snap was reached"


def test_the_python_engine_publishes_the_entry_its_own_clear_destroys():
    """The Python engine's side: snap_results_at_close carries removed_own_index.

    The log itself still clears at the window edge, which is what keeps the
    tokenizer channel identical to Go's Observe().
    """
    from src.constants import ActionSnapOwn
    from src.game.engine import CambiaGameState

    cfg = _config()
    rng = random.Random(4242)
    published = 0

    for seed in range(DEALS):
        game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(seed))
        for _ in range(200):
            if game.is_terminal():
                break
            legal = game.get_legal_actions()
            if not legal:
                break
            action = legal[rng.randrange(len(legal))]
            was_open = game.snap_phase_active
            actor = game.get_acting_player()
            if actor < 0:
                break
            before = len(game.players[actor].hand)
            game.apply_action(action)
            if (
                isinstance(action, ActionSnapOwn)
                and was_open
                and not game.snap_phase_active
                and len(game.players[actor].hand) < before
            ):
                entries = game.snap_results_at_close
                assert entries, (
                    "the window-closing own snap published nothing, so the belief "
                    "still cannot learn which slot left"
                )
                assert (
                    game.snap_results_log == []
                ), "the tokenizer channel must still be cleared at the window edge"
                assert any(
                    entry.get("action_type") == "ActionSnapOwn"
                    and entry.get("success")
                    and entry.get("removed_own_index") is not None
                    for entry in entries
                ), f"no removal index in the published entries: {entries}"
                published += 1

    assert published > 0, "no window-closing own snap occurred on the Python engine"
