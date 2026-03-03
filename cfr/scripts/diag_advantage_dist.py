#!/usr/bin/env python3
"""Diagnostic: Analyze advantage value distributions from Phase 2 checkpoint.

Loads the best checkpoint (iter 450), creates game states at various phases
by playing random moves, encodes them, passes through the network, and
reports statistics on advantage outputs to diagnose degeneracy.
"""

import sys
import os
import random
import copy
import math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.engine import CambiaGameState
from src.agent_state import AgentState, AgentObservation
from src.encoding import (
    encode_infoset_eppbs_interleaved,
    encode_action_mask,
    action_to_index,
    NUM_ACTIONS,
)
from src.networks import build_advantage_network, get_strategy_from_advantages
from src.constants import (
    DecisionContext,
    EpistemicTag,
    NUM_PLAYERS,
    GamePhase,
    ActionCallCambia,
    ActionDrawStockpile,
    ActionDrawDiscard,
)

# -------------------------------------------------------------------
# Action index constants (from encoding.py)
# -------------------------------------------------------------------
IDX_DRAW_STOCKPILE = 0
IDX_DRAW_DISCARD = 1
IDX_CALL_CAMBIA = 2
IDX_DISCARD_NO_ABILITY = 3
IDX_DISCARD_ABILITY = 4
IDX_REPLACE_BASE = 5  # 5-10
IDX_PEEK_OWN_BASE = 11  # 11-16
IDX_PEEK_OTHER_BASE = 17  # 17-22
IDX_BLIND_SWAP_BASE = 23  # 23-58
IDX_KING_LOOK_BASE = 59  # 59-94
IDX_KING_SWAP_FALSE = 95
IDX_KING_SWAP_TRUE = 96
IDX_PASS_SNAP = 97
IDX_SNAP_OWN_BASE = 98  # 98-103
IDX_SNAP_OPP_BASE = 104  # 104-109
IDX_SNAP_OPP_MOVE_BASE = 110  # 110-145

ACTION_NAMES = {}
ACTION_NAMES[0] = "DrawStockpile"
ACTION_NAMES[1] = "DrawDiscard"
ACTION_NAMES[2] = "CallCambia"
ACTION_NAMES[3] = "Discard(noAbil)"
ACTION_NAMES[4] = "Discard(abil)"
for i in range(6):
    ACTION_NAMES[5 + i] = f"Replace({i})"
for i in range(6):
    ACTION_NAMES[11 + i] = f"PeekOwn({i})"
for i in range(6):
    ACTION_NAMES[17 + i] = f"PeekOther({i})"
for own in range(6):
    for opp in range(6):
        ACTION_NAMES[23 + own * 6 + opp] = f"BlindSwap({own},{opp})"
for own in range(6):
    for opp in range(6):
        ACTION_NAMES[59 + own * 6 + opp] = f"KingLook({own},{opp})"
ACTION_NAMES[95] = "KingSwap(No)"
ACTION_NAMES[96] = "KingSwap(Yes)"
ACTION_NAMES[97] = "PassSnap"
for i in range(6):
    ACTION_NAMES[98 + i] = f"SnapOwn({i})"
for i in range(6):
    ACTION_NAMES[104 + i] = f"SnapOpp({i})"
for own in range(6):
    for s in range(6):
        ACTION_NAMES[110 + own * 6 + s] = f"SnapOppMove({own},{s})"


def action_category(idx):
    if idx == 2:
        return "cambia"
    if idx in (0, 1):
        return "draw"
    if idx in (3, 4):
        return "discard"
    if 5 <= idx <= 10:
        return "replace"
    if 11 <= idx <= 22:
        return "peek"
    if 23 <= idx <= 96:
        return "swap/king"
    if 97 <= idx <= 145:
        return "snap"
    return "other"


def get_decision_context(game_state):
    """Determine decision context from game state."""
    if game_state.snap_phase_active:
        return DecisionContext.SNAP_DECISION
    if game_state.pending_action is not None:
        from src.constants import (
            ActionDiscard,
            ActionAbilityPeekOwnSelect,
            ActionAbilityPeekOtherSelect,
            ActionAbilityBlindSwapSelect,
            ActionAbilityKingLookSelect,
            ActionAbilityKingSwapDecision,
            ActionSnapOpponentMove,
        )
        pa = game_state.pending_action
        if isinstance(pa, ActionDiscard):
            return DecisionContext.POST_DRAW
        if isinstance(
            pa,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            return DecisionContext.ABILITY_SELECT
        if isinstance(pa, ActionSnapOpponentMove):
            return DecisionContext.SNAP_MOVE
    return DecisionContext.START_TURN


def encode_state_eppbs(agent_state, decision_context):
    """Encode an AgentState using interleaved EP-PBS layout."""
    st = agent_state
    if st.cambia_caller is None:
        cambia_state = 2
    elif st.cambia_caller == st.player_id:
        cambia_state = 0
    else:
        cambia_state = 1

    drawn_bucket = -1
    if hasattr(st, "drawn_card_bucket") and st.drawn_card_bucket is not None:
        drawn_bucket = (
            st.drawn_card_bucket.value
            if hasattr(st.drawn_card_bucket, "value")
            else int(st.drawn_card_bucket)
        )

    return encode_infoset_eppbs_interleaved(
        slot_tags=[t.value if hasattr(t, "value") else int(t) for t in st.slot_tags],
        slot_buckets=[int(b) for b in st.slot_buckets],
        discard_top_bucket=(
            st.known_discard_top_bucket.value
            if hasattr(st.known_discard_top_bucket, "value")
            else int(st.known_discard_top_bucket)
        ),
        stock_estimate=(
            st.stockpile_estimate.value
            if hasattr(st.stockpile_estimate, "value")
            else int(st.stockpile_estimate)
        ),
        game_phase=(
            st.game_phase.value
            if hasattr(st.game_phase, "value")
            else int(st.game_phase)
        ),
        decision_context=(
            decision_context.value
            if hasattr(decision_context, "value")
            else int(decision_context)
        ),
        cambia_state=cambia_state,
        drawn_card_bucket=drawn_bucket,
        own_hand_size=len(st.own_hand),
        opp_hand_size=st.opponent_card_count,
    )


def create_agent_state(game_state, player_id, config=None):
    """Create an AgentState from a game state."""
    opp_id = 1 - player_id
    st = AgentState(
        player_id=player_id,
        opponent_id=opp_id,
        memory_level="perfect",
        time_decay_turns=100,
        initial_hand_size=len(game_state.players[player_id].hand),
        config=config,
    )
    initial_obs = AgentObservation(
        acting_player=-1,
        action=None,
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
    initial_hand = game_state.players[player_id].hand
    initial_peeks = game_state.players[player_id].initial_peek_indices
    st.initialize(initial_obs, initial_hand, initial_peeks)
    return st


def play_random_turns(game_state, agent_states, n_turns, rng):
    """Play random actions for n_turns, updating agent states.
    Returns the game state after n_turns (or terminal), plus the agent states.
    """
    for _ in range(n_turns):
        if game_state.is_terminal():
            break
        cp = game_state.current_player_index
        legal = game_state.get_legal_actions()
        if not legal:
            break
        # Filter out Cambia for first few turns to avoid instant-end
        non_cambia = [a for a in legal if not isinstance(a, ActionCallCambia)]
        if non_cambia:
            action = rng.choice(non_cambia)
        else:
            action = rng.choice(list(legal))

        delta, undo = game_state.apply_action(action)

        # Create observation for agents
        obs = AgentObservation(
            acting_player=cp,
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
        for ast in agent_states.values():
            try:
                filtered = copy.copy(obs)
                filtered.drawn_card = None
                filtered.peeked_cards = None
                ast.update(filtered)
            except Exception:
                pass
    return game_state, agent_states


def collect_samples(net, n_samples, target_turn_range, device, rng_seed_base):
    """Collect network output samples at specific game phases.

    target_turn_range: (min_turn, max_turn) — we play random actions until
    game turn is in range, then sample from the current player's perspective.
    """
    results = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1
        rng = random.Random(rng_seed_base + attempts)

        gs = CambiaGameState()
        gs._rng = rng

        agent_states = {}
        for pid in range(NUM_PLAYERS):
            agent_states[pid] = create_agent_state(gs, pid)

        min_turn, max_turn = target_turn_range
        # Play random actions to reach target turn range
        # Each "turn" in the engine is an action, not a full player-turn
        # We need to reach roughly min_turn actions
        if min_turn > 0:
            gs, agent_states = play_random_turns(gs, agent_states, min_turn * 3, rng)

        if gs.is_terminal():
            continue

        # Check current turn
        turn = gs.get_turn_number()
        if turn < min_turn or turn > max_turn:
            continue

        cp = gs.current_player_index
        legal = gs.get_legal_actions()
        if not legal:
            continue

        decision_ctx = get_decision_context(gs)
        ast = agent_states[cp]

        try:
            features = encode_state_eppbs(ast, decision_ctx)
        except Exception as e:
            continue

        legal_list = list(legal)
        action_mask = encode_action_mask(legal_list)

        # Truncate/pad to 200 dims (checkpoint input_dim)
        if len(features) > 200:
            features = features[:200]
        elif len(features) < 200:
            features = np.pad(features, (0, 200 - len(features)))

        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).bool().to(device)

        with torch.inference_mode():
            raw_advantages = net(feat_t, mask_t).squeeze(0).cpu().numpy()
            strategy = get_strategy_from_advantages(
                torch.from_numpy(raw_advantages).unsqueeze(0),
                mask_t.cpu(),
            ).squeeze(0).numpy()

        legal_indices = np.where(action_mask)[0]
        legal_advs = raw_advantages[legal_indices]
        legal_strat = strategy[legal_indices]

        # Cambia-specific
        cambia_adv = raw_advantages[IDX_CALL_CAMBIA] if action_mask[IDX_CALL_CAMBIA] else None
        non_cambia_legal = [i for i in legal_indices if i != IDX_CALL_CAMBIA]
        non_cambia_advs = raw_advantages[non_cambia_legal] if non_cambia_legal else np.array([])

        # Strategy entropy
        probs = legal_strat[legal_strat > 0]
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(legal_indices)) if len(legal_indices) > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy = 0.0
            norm_entropy = 0.0

        # Max vs second-max gap
        sorted_advs = np.sort(legal_advs)[::-1]
        max_adv = sorted_advs[0] if len(sorted_advs) > 0 else 0.0
        second_max = sorted_advs[1] if len(sorted_advs) > 1 else 0.0
        gap = max_adv - second_max

        # Cambia probability
        cambia_prob = strategy[IDX_CALL_CAMBIA] if action_mask[IDX_CALL_CAMBIA] else None

        # Top action
        top_idx = legal_indices[np.argmax(legal_advs)]

        results.append({
            "turn": turn,
            "decision_ctx": decision_ctx.name,
            "n_legal": len(legal_indices),
            "legal_adv_mean": float(np.mean(legal_advs)),
            "legal_adv_std": float(np.std(legal_advs)),
            "legal_adv_min": float(np.min(legal_advs)),
            "legal_adv_max": float(np.max(legal_advs)),
            "max_minus_2nd": float(gap),
            "entropy": float(entropy),
            "norm_entropy": float(norm_entropy),
            "cambia_adv": float(cambia_adv) if cambia_adv is not None else None,
            "cambia_prob": float(cambia_prob) if cambia_prob is not None else None,
            "non_cambia_adv_mean": float(np.mean(non_cambia_advs)) if len(non_cambia_advs) > 0 else None,
            "non_cambia_adv_std": float(np.std(non_cambia_advs)) if len(non_cambia_advs) > 0 else None,
            "top_action": ACTION_NAMES.get(int(top_idx), f"idx_{top_idx}"),
            "top_action_idx": int(top_idx),
            "top_action_cat": action_category(int(top_idx)),
            "top_action_prob": float(strategy[top_idx]),
            "cambia_is_legal": bool(action_mask[IDX_CALL_CAMBIA]),
            # Per-category advantage means
            "draw_adv_mean": float(np.mean([raw_advantages[i] for i in legal_indices if action_category(i) == "draw"])) if any(action_category(i) == "draw" for i in legal_indices) else None,
            "replace_adv_mean": float(np.mean([raw_advantages[i] for i in legal_indices if action_category(i) == "replace"])) if any(action_category(i) == "replace" for i in legal_indices) else None,
            "discard_adv_mean": float(np.mean([raw_advantages[i] for i in legal_indices if action_category(i) == "discard"])) if any(action_category(i) == "discard" for i in legal_indices) else None,
        })

    return results


def main():
    ckpt_path = "runs/interleaved-resnet-adaptive/checkpoints/deep_cfr_checkpoint_iter_450.pt"
    device = "cpu"
    n_samples = 500

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    dcfr_config = ckpt.get("dcfr_config", {})

    net = build_advantage_network(
        input_dim=dcfr_config.get("input_dim", 200),
        hidden_dim=dcfr_config.get("hidden_dim", 256),
        output_dim=NUM_ACTIONS,
        dropout=dcfr_config.get("dropout", 0.1),
        validate_inputs=False,
        num_hidden_layers=dcfr_config.get("num_hidden_layers", 3),
        use_residual=dcfr_config.get("use_residual", True),
        network_type=dcfr_config.get("network_type", "residual"),
    )
    net.load_state_dict(ckpt["advantage_net_state_dict"])
    net.eval()

    print(f"Network: {type(net).__name__}, input_dim={dcfr_config.get('input_dim')}, "
          f"hidden_dim={dcfr_config.get('hidden_dim')}, layers={dcfr_config.get('num_hidden_layers')}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Phase definitions: (label, min_turn, max_turn)
    phases = [
        ("Turn 0-1 (START)", 0, 1),
        ("Turn 2-4 (EARLY)", 2, 4),
        ("Turn 5-9 (MID)", 5, 9),
        ("Turn 10-14 (LATE)", 10, 14),
        ("Turn 15-20 (V.LATE)", 15, 20),
    ]

    all_phase_data = {}

    for label, tmin, tmax in phases:
        print(f"--- Collecting {n_samples} samples for {label} (turns {tmin}-{tmax}) ---")
        samples = collect_samples(net, n_samples, (tmin, tmax), device, rng_seed_base=tmin * 10000)
        all_phase_data[label] = samples
        print(f"  Collected {len(samples)} samples")

    # Print summary table
    print("\n" + "=" * 120)
    print("ADVANTAGE DISTRIBUTION SUMMARY")
    print("=" * 120)
    header = f"{'Phase':<22} {'N':>4} {'AdvMean':>8} {'AdvStd':>8} {'Max-2nd':>8} {'NormEnt':>8} {'CambAdv':>8} {'CambProb':>8} {'NCambStd':>8} {'NLegal':>6}"
    print(header)
    print("-" * 120)

    for label, _, _ in phases:
        samples = all_phase_data[label]
        if not samples:
            print(f"{label:<22} {'0':>4} {'N/A':>8}")
            continue

        n = len(samples)
        adv_mean = np.mean([s["legal_adv_mean"] for s in samples])
        adv_std = np.mean([s["legal_adv_std"] for s in samples])
        gap = np.mean([s["max_minus_2nd"] for s in samples])
        nent = np.mean([s["norm_entropy"] for s in samples])
        n_legal = np.mean([s["n_legal"] for s in samples])

        cambia_advs = [s["cambia_adv"] for s in samples if s["cambia_adv"] is not None]
        cambia_adv_mean = np.mean(cambia_advs) if cambia_advs else float("nan")

        cambia_probs = [s["cambia_prob"] for s in samples if s["cambia_prob"] is not None]
        cambia_prob_mean = np.mean(cambia_probs) if cambia_probs else float("nan")

        nc_stds = [s["non_cambia_adv_std"] for s in samples if s["non_cambia_adv_std"] is not None]
        nc_std_mean = np.mean(nc_stds) if nc_stds else float("nan")

        print(f"{label:<22} {n:>4} {adv_mean:>8.4f} {adv_std:>8.4f} {gap:>8.4f} {nent:>8.4f} "
              f"{cambia_adv_mean:>8.4f} {cambia_prob_mean:>8.4f} {nc_std_mean:>8.4f} {n_legal:>6.1f}")

    # Detailed: top action distribution per phase
    print("\n" + "=" * 120)
    print("TOP ACTION CATEGORY DISTRIBUTION (% of samples where category has highest advantage)")
    print("=" * 120)
    header2 = f"{'Phase':<22} {'draw':>8} {'cambia':>8} {'discard':>8} {'replace':>8} {'peek':>8} {'swap/king':>8} {'snap':>8}"
    print(header2)
    print("-" * 120)

    for label, _, _ in phases:
        samples = all_phase_data[label]
        if not samples:
            continue
        n = len(samples)
        from collections import Counter
        cat_counts = Counter(s["top_action_cat"] for s in samples)
        cats = ["draw", "cambia", "discard", "replace", "peek", "swap/king", "snap"]
        pcts = [100.0 * cat_counts.get(c, 0) / n for c in cats]
        print(f"{label:<22} " + " ".join(f"{p:>8.1f}" for p in pcts))

    # Degeneracy diagnosis
    print("\n" + "=" * 120)
    print("DEGENERACY DIAGNOSIS")
    print("=" * 120)

    for label, _, _ in phases:
        samples = all_phase_data[label]
        if not samples:
            continue

        stds = [s["legal_adv_std"] for s in samples]
        nents = [s["norm_entropy"] for s in samples]
        gaps = [s["max_minus_2nd"] for s in samples]

        degenerate_std = sum(1 for s in stds if s < 0.1) / len(stds) * 100
        near_uniform = sum(1 for e in nents if e > 0.9) / len(nents) * 100
        tiny_gap = sum(1 for g in gaps if g < 0.05) / len(gaps) * 100

        # Cambia dominance
        cambia_samples = [s for s in samples if s["cambia_is_legal"]]
        if cambia_samples:
            cambia_top = sum(1 for s in cambia_samples if s["top_action_cat"] == "cambia") / len(cambia_samples) * 100
            cambia_high_prob = sum(1 for s in cambia_samples if s["cambia_prob"] is not None and s["cambia_prob"] > 0.5) / len(cambia_samples) * 100
        else:
            cambia_top = 0
            cambia_high_prob = 0

        print(f"\n{label}:")
        print(f"  Degenerate (std<0.1):    {degenerate_std:5.1f}% of states")
        print(f"  Near-uniform (ent>0.9):  {near_uniform:5.1f}% of states")
        print(f"  Tiny gap (gap<0.05):     {tiny_gap:5.1f}% of states")
        if cambia_samples:
            print(f"  Cambia is top action:    {cambia_top:5.1f}% (of {len(cambia_samples)} states where Cambia legal)")
            print(f"  Cambia prob > 50%:       {cambia_high_prob:5.1f}%")

    # Detailed advantage stats per decision context
    print("\n" + "=" * 120)
    print("PER-DECISION-CONTEXT BREAKDOWN")
    print("=" * 120)
    all_samples = []
    for samples in all_phase_data.values():
        all_samples.extend(samples)

    from collections import defaultdict
    by_ctx = defaultdict(list)
    for s in all_samples:
        by_ctx[s["decision_ctx"]].append(s)

    for ctx, samples in sorted(by_ctx.items()):
        n = len(samples)
        adv_std = np.mean([s["legal_adv_std"] for s in samples])
        nent = np.mean([s["norm_entropy"] for s in samples])
        gap = np.mean([s["max_minus_2nd"] for s in samples])
        print(f"  {ctx:<20} N={n:>4}  AdvStd={adv_std:.4f}  NormEnt={nent:.4f}  Gap={gap:.4f}")

    # Show example advantage vectors from a few early-game states
    print("\n" + "=" * 120)
    print("EXAMPLE: First 3 START_TURN states from Turn 0-1")
    print("=" * 120)
    start_samples = [s for s in all_phase_data.get("Turn 0-1 (START)", []) if s["decision_ctx"] == "START_TURN"][:3]
    for i, s in enumerate(start_samples):
        print(f"\n  State {i}: turn={s['turn']}, n_legal={s['n_legal']}, "
              f"adv_std={s['legal_adv_std']:.4f}, norm_entropy={s['norm_entropy']:.4f}")
        print(f"    Top: {s['top_action']} (prob={s['top_action_prob']:.4f})")
        if s["cambia_adv"] is not None:
            print(f"    Cambia adv={s['cambia_adv']:.4f}, prob={s['cambia_prob']:.4f}")
            print(f"    Non-Cambia adv mean={s['non_cambia_adv_mean']:.4f}, std={s['non_cambia_adv_std']:.4f}")


if __name__ == "__main__":
    main()
