"""Probe: GoAgentState's belief is NOT ppo_env's old public-only contract.

Evidence behind the cambia-1376 observation-contract finding. Not collected by
pytest (no ``test_`` prefix) and not a gate: it is the measurement that says how
far the ported PPO observation moved, so the number in the report can be
re-derived instead of taken on trust.

Runs a seeded random legal walk in lockstep on the Go engine and the Python
reference engine over the same deck. At every decision point it compares:

  A) GoAgentState.encode_eppbs_interleaved_v2, i.e. the belief
     cambia_agent_update maintains, which is what the ported env now feeds the
     policy. cambia_agent_update reads the game state directly and records what
     the rules reveal to the acting seat.
  B) encode_infoset_eppbs_interleaved_v2 over a Python AgentState fed the
     public-only observation the pre-port ppo_env._update_states built, with
     drawn_card and peeked_cards stripped for every seat including the actor.
     That is the contract ratified in note cambia-1074.

Reading under config/ppo_encoding_v2.yaml (the PPO-200k config), 400 games,
about 2,200 decision points: 46.6 / 46.4 / 45.6 per cent of decision points
diverge over three runs. The first divergence in a game is overwhelmingly
preceded by ActionReplace, where the Go belief learns the card the seat just
placed and the public-only belief does not; peeks and own-snaps account for the
rest.

The reading is a sampled estimate, not a fixed number: the Python engine
iterates hash-salted sets internally, so the trajectory shifts between
interpreter runs even with the deal and the action choice seeded. Three runs at
this sample size agree to within a point, which is enough to act on.

There is no public-only mode on the FFI agent surface, so the two cannot be
reconciled from cfr/ alone.

This probe needs both engines, so it retires with src/game under cambia-1430.

usage (from cfr/, with PYTHONPATH and LIBCAMBIA_PATH pinned):
    python tests/probe_ppo_obs_contract.py
"""

import copy
import random
from collections import Counter

import numpy as np

from src.agent_state import AgentState, AgentObservation
from src.constants import DecisionContext, NUM_PLAYERS
from src.encoding import (
    action_to_index,
    encode_action_mask,
    encode_infoset_eppbs_interleaved_v2,
)
from src.ffi.bridge import GoEngine, GoAgentState, extract_deck_from_python_game
from src.game.engine import CambiaGameState


def py_ctx(gs):
    from src.constants import (
        ActionDiscard,
        ActionAbilityPeekOwnSelect,
        ActionAbilityPeekOtherSelect,
        ActionAbilityBlindSwapSelect,
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionSnapOpponentMove,
    )

    if gs.snap_phase_active:
        return DecisionContext.SNAP_DECISION
    if gs.pending_action:
        p = gs.pending_action
        if isinstance(p, ActionDiscard):
            return DecisionContext.POST_DRAW
        if isinstance(
            p,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            return DecisionContext.ABILITY_SELECT
        if isinstance(p, ActionSnapOpponentMove):
            return DecisionContext.SNAP_MOVE
    return DecisionContext.START_TURN


def public_obs(gs, action, actor):
    return AgentObservation(
        acting_player=actor,
        action=action,
        discard_top_card=gs.get_discard_top(),
        player_hand_sizes=[gs.get_player_card_count(i) for i in range(NUM_PLAYERS)],
        stockpile_size=gs.get_stockpile_size(),
        drawn_card=None,
        peeked_cards=None,
        snap_results=copy.deepcopy(gs.snap_results_log),
        did_cambia_get_called=gs.cambia_caller_id is not None,
        who_called_cambia=gs.cambia_caller_id,
        is_game_over=gs.is_terminal(),
        current_turn=gs.get_turn_number(),
    )


def run(seed, cfg):
    rng = np.random.default_rng(seed)
    # CambiaGameState deals from the global random module, so the deal has to be
    # seeded here too or successive probe runs disagree by several points.
    random.seed(seed)
    py = CambiaGameState(house_rules=cfg.cambia_rules)
    deck, start = extract_deck_from_python_game(py)
    go = GoEngine.from_deck(deck, starting_player=start, house_rules=cfg.cambia_rules)

    py_agents = []
    for pid in range(NUM_PLAYERS):
        st = AgentState(
            player_id=pid,
            opponent_id=1 - pid,
            memory_level=cfg.agent_params.memory_level,
            time_decay_turns=cfg.agent_params.time_decay_turns,
            initial_hand_size=len(py.players[pid].hand),
            config=cfg,
        )
        st.initialize(
            public_obs(py, None, -1),
            py.players[pid].hand,
            py.players[pid].initial_peek_indices,
        )
        py_agents.append(st)
    go_agents = [
        GoAgentState(
            go, i, cfg.agent_params.memory_level, cfg.agent_params.time_decay_turns
        )
        for i in range(NUM_PLAYERS)
    ]

    cmp_n = div_n = 0
    div_after = Counter()
    last_action_kind = "<none: divergent at initialize>"
    run.first_seen = False
    for _ in range(400):
        if py.is_terminal() or go.is_terminal():
            break
        if py.get_acting_player() != go.acting_player():
            print(f"  seed {seed}: seat desync, stopping")
            break

        actor = go.acting_player()
        ctx_i = go.decision_ctx()
        drawn_i = go.get_drawn_card_bucket()
        g_enc = go_agents[actor].encode_eppbs_interleaved_v2(ctx_i, drawn_i)
        p_enc = encode_infoset_eppbs_interleaved_v2(
            py_agents[actor], py_ctx(py), drawn_card_bucket=int(drawn_i)
        )
        cmp_n += 1
        if not np.allclose(g_enc, p_enc, atol=1e-4):
            div_n += 1
            if not run.first_seen:
                run.first_seen = True
                div_after["FIRST:" + last_action_kind] += 1

        py_legal = list(py.get_legal_actions())
        go_mask = go.legal_actions_mask()
        py_idx = {}
        for a in py_legal:
            try:
                py_idx[action_to_index(a)] = a
            except Exception:
                pass
        common = sorted(set(np.where(go_mask > 0)[0].tolist()) & set(py_idx))
        if not common:
            break
        pick = int(common[int(rng.integers(len(common)))])
        act_obj = py_idx[pick]
        last_action_kind = type(act_obj).__name__

        py.apply_action(act_obj)
        go.apply_action(pick)
        obs = public_obs(py, act_obj, actor)
        for st in py_agents:
            try:
                st.update(obs)
            except Exception:
                pass
        go.update_both(go_agents[0], go_agents[1])

    go.close()
    for a in go_agents:
        a.close()
    return cmp_n, div_n, div_after


def main():
    from src.config import load_config

    cfg = load_config("config/ppo_encoding_v2.yaml")
    tot_c = tot_d = 0
    agg = Counter()
    for seed in range(400):
        c, d, after = run(seed, cfg)
        tot_c += c
        tot_d += d
        agg.update(after)
    print(f"decision points compared : {tot_c}")
    print(f"encoding divergences     : {tot_d}  ({100.0*tot_d/max(tot_c,1):.1f}%)")
    print("divergences by preceding action class:")
    for k, v in agg.most_common():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
