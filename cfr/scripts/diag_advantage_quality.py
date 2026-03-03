#!/usr/bin/env python3
"""Advantage prediction quality diagnostic.

Constructs canonical mid-game states by hand, runs them through the trained
advantage network, and inspects whether outputs are structured or noise.

Phase A item #3 of ceiling investigation.

Usage:
    cd cfr && python scripts/diag_advantage_quality.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from src.constants import (
    EP_PBS_INPUT_DIM,
    EpistemicTag,
    CardBucket,
    DecisionContext,
    GamePhase,
    StockpileEstimate,
)
from src.encoding import NUM_ACTIONS, encode_infoset_eppbs_interleaved
from src.networks import ResidualAdvantageNetwork, get_strategy_from_advantages

# ── Action index reference ──
IDX_DRAW_STOCK = 0
IDX_DRAW_DISCARD = 1
IDX_CAMBIA = 2
IDX_DISCARD_NO_ABILITY = 3
IDX_DISCARD_ABILITY = 4
IDX_REPLACE_BASE = 5       # 5-10: replace slot 0-5
IDX_PEEK_OWN_BASE = 11     # 11-16: peek own 0-5
IDX_PEEK_OTHER_BASE = 17   # 17-22: peek other 0-5
IDX_BLIND_SWAP_BASE = 23   # 23-58: blind swap own*6+opp
IDX_PASS_SNAP = 97
IDX_SNAP_OWN_BASE = 98

ACTION_NAMES = {}
ACTION_NAMES[0] = "DrawStock"
ACTION_NAMES[1] = "DrawDiscard"
ACTION_NAMES[2] = "CallCambia"
ACTION_NAMES[3] = "Discard(no_ability)"
ACTION_NAMES[4] = "Discard(ability)"
for i in range(6):
    ACTION_NAMES[5 + i] = f"Replace(slot{i})"
    ACTION_NAMES[11 + i] = f"PeekOwn(slot{i})"
    ACTION_NAMES[17 + i] = f"PeekOther(slot{i})"
    ACTION_NAMES[98 + i] = f"SnapOwn(slot{i})"
    ACTION_NAMES[104 + i] = f"SnapOpp(slot{i})"
for own in range(6):
    for opp in range(6):
        ACTION_NAMES[23 + own * 6 + opp] = f"BlindSwap({own},{opp})"
        ACTION_NAMES[59 + own * 6 + opp] = f"KingLook({own},{opp})"
        ACTION_NAMES[110 + own * 6 + opp] = f"SnapOppMove({own},{opp})"
ACTION_NAMES[95] = "KingSwap(no)"
ACTION_NAMES[96] = "KingSwap(yes)"
ACTION_NAMES[97] = "PassSnap"


def action_name(idx):
    return ACTION_NAMES.get(idx, f"action_{idx}")


# ── Canonical State Builders ──
# Using the interleaved encoding directly. We craft 224-dim vectors by hand
# for clarity and auditability.

def _make_mask(indices):
    """Create a 146-dim bool mask from a list of legal action indices."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for i in indices:
        mask[i] = True
    return mask


def _build_start_turn_state(
    own_tags, own_buckets, opp_tags, opp_buckets,
    discard_bucket, game_phase, stock_est=StockpileEstimate.HIGH.value,
    own_hand_size=4, opp_hand_size=4, turn_progress=0.0,
):
    """Build a START_TURN state. Legal actions: DrawStock(0), DrawDiscard(1), Cambia(2)."""
    slot_tags = list(own_tags) + [EpistemicTag.UNK] * (6 - len(own_tags)) + \
                list(opp_tags) + [EpistemicTag.UNK] * (6 - len(opp_tags))
    slot_buckets = list(own_buckets) + [0] * (6 - len(own_buckets)) + \
                   list(opp_buckets) + [0] * (6 - len(opp_buckets))
    features = encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags,
        slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket,
        stock_estimate=stock_est,
        game_phase=game_phase,
        decision_context=DecisionContext.START_TURN.value,
        cambia_state=2,  # NONE
        drawn_card_bucket=-1,
        own_hand_size=own_hand_size,
        opp_hand_size=opp_hand_size,
        turn_progress=turn_progress,
    )
    mask = _make_mask([IDX_DRAW_STOCK, IDX_DRAW_DISCARD, IDX_CAMBIA])
    return features, mask


def _build_post_draw_state(
    own_tags, own_buckets, opp_tags, opp_buckets,
    drawn_card_bucket, discard_bucket, game_phase,
    has_ability=False, ability_type=None,
    own_hand_size=4, opp_hand_size=4, turn_progress=0.0,
):
    """Build a POST_DRAW state after drawing from stockpile."""
    slot_tags = list(own_tags) + [EpistemicTag.UNK] * (6 - len(own_tags)) + \
                list(opp_tags) + [EpistemicTag.UNK] * (6 - len(opp_tags))
    slot_buckets = list(own_buckets) + [0] * (6 - len(own_buckets)) + \
                   list(opp_buckets) + [0] * (6 - len(opp_buckets))
    features = encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags,
        slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket,
        stock_estimate=StockpileEstimate.HIGH.value,
        game_phase=game_phase,
        decision_context=DecisionContext.POST_DRAW.value,
        cambia_state=2,  # NONE
        drawn_card_bucket=drawn_card_bucket,
        own_hand_size=own_hand_size,
        opp_hand_size=opp_hand_size,
        turn_progress=turn_progress,
    )
    # Legal: discard (3), discard+ability (4 if applicable), replace slot 0..hand_size-1
    legal = [IDX_DISCARD_NO_ABILITY]
    if has_ability:
        legal.append(IDX_DISCARD_ABILITY)
    for s in range(own_hand_size):
        legal.append(IDX_REPLACE_BASE + s)
    mask = _make_mask(legal)
    return features, mask


def _build_ability_peek_own_state(
    own_tags, own_buckets, opp_tags, opp_buckets,
    discard_bucket, game_phase, own_hand_size=4, opp_hand_size=4,
    turn_progress=0.0,
):
    """Build an ABILITY_SELECT state for peek-own (7/8 discarded)."""
    slot_tags = list(own_tags) + [EpistemicTag.UNK] * (6 - len(own_tags)) + \
                list(opp_tags) + [EpistemicTag.UNK] * (6 - len(opp_tags))
    slot_buckets = list(own_buckets) + [0] * (6 - len(own_buckets)) + \
                   list(opp_buckets) + [0] * (6 - len(opp_buckets))
    features = encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags,
        slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket,
        stock_estimate=StockpileEstimate.HIGH.value,
        game_phase=game_phase,
        decision_context=DecisionContext.ABILITY_SELECT.value,
        cambia_state=2,  # NONE
        drawn_card_bucket=-1,
        own_hand_size=own_hand_size,
        opp_hand_size=opp_hand_size,
        turn_progress=turn_progress,
    )
    legal = [IDX_PEEK_OWN_BASE + s for s in range(own_hand_size)]
    mask = _make_mask(legal)
    return features, mask


def _build_snap_decision_state(
    own_tags, own_buckets, opp_tags, opp_buckets,
    discard_bucket, game_phase, own_hand_size=4, opp_hand_size=4,
    turn_progress=0.0,
):
    """Build a SNAP_DECISION state. Legal: PassSnap(97), SnapOwn(98-103)."""
    slot_tags = list(own_tags) + [EpistemicTag.UNK] * (6 - len(own_tags)) + \
                list(opp_tags) + [EpistemicTag.UNK] * (6 - len(opp_tags))
    slot_buckets = list(own_buckets) + [0] * (6 - len(own_buckets)) + \
                   list(opp_buckets) + [0] * (6 - len(opp_buckets))
    features = encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags,
        slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket,
        stock_estimate=StockpileEstimate.HIGH.value,
        game_phase=game_phase,
        decision_context=DecisionContext.SNAP_DECISION.value,
        cambia_state=2,  # NONE
        drawn_card_bucket=-1,
        own_hand_size=own_hand_size,
        opp_hand_size=opp_hand_size,
        turn_progress=turn_progress,
    )
    legal = [IDX_PASS_SNAP]
    for s in range(own_hand_size):
        legal.append(IDX_SNAP_OWN_BASE + s)
    mask = _make_mask(legal)
    return features, mask


# ── Canonical Scenarios ──

def build_scenarios():
    """Return list of (name, features, mask, expected_description) tuples."""
    P = EpistemicTag.PRIV_OWN
    U = EpistemicTag.UNK
    B = CardBucket

    scenarios = []

    # ── START_TURN scenarios ──

    # S1: Excellent hand (A+A+2+3 = 6), turn 1. Should NOT call Cambia yet (too early to know).
    # Actually at sum=6, Nash T1C is conditional on discard value. With discard=5 (MID_NUM), borderline.
    # The key question: does the network differentiate by hand quality?
    scenarios.append((
        "S1: START_TURN, excellent hand (A,A,2,3), turn 1, early game",
        *_build_start_turn_state(
            own_tags=[P, P, U, U], own_buckets=[B.ACE.value, B.ACE.value, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.MID_NUM.value, game_phase=GamePhase.EARLY.value,
            turn_progress=0.05,
        ),
        "T1C plausible (sum≤6). Expect elevated Cambia advantage.",
    ))

    # S2: Terrible hand (K,Q,T,9 = 13+12+10+9 = 44), turn 1. Should NOT call Cambia.
    scenarios.append((
        "S2: START_TURN, terrible hand (K,Q,T,9=44), turn 1",
        *_build_start_turn_state(
            own_tags=[P, P, P, P],
            own_buckets=[B.HIGH_KING.value, B.SWAP_BLIND.value, B.PEEK_OTHER.value, B.PEEK_OTHER.value],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.MID_NUM.value, game_phase=GamePhase.EARLY.value,
            turn_progress=0.05,
        ),
        "Should strongly NOT Cambia. Draw is dominant.",
    ))

    # S3: Perfect hand (Joker+RedK+A+A = 0+(-1)+1+1 = 1), turn 1. Clear T1C.
    scenarios.append((
        "S3: START_TURN, near-perfect hand (0,-1,1,1 = 1), turn 1",
        *_build_start_turn_state(
            own_tags=[P, P, P, P],
            own_buckets=[B.ZERO.value, B.NEG_KING.value, B.ACE.value, B.ACE.value],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.MID_NUM.value, game_phase=GamePhase.EARLY.value,
            turn_progress=0.05,
        ),
        "Clear T1C — Cambia should dominate strongly.",
    ))

    # S4: Average hand, mid game, no known cards. Should draw.
    scenarios.append((
        "S4: START_TURN, all unknown hand, mid game, turn 5",
        *_build_start_turn_state(
            own_tags=[U, U, U, U], own_buckets=[0, 0, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.LOW_NUM.value, game_phase=GamePhase.MID.value,
            stock_est=StockpileEstimate.MEDIUM.value,
            turn_progress=0.3,
        ),
        "All unknown — should draw (need information). Cambia suicidal.",
    ))

    # S5: Known low hand (2,3,A,2 = 8), mid game, discard is HIGH_KING.
    scenarios.append((
        "S5: START_TURN, good known hand (2,3,1,2=8), mid game",
        *_build_start_turn_state(
            own_tags=[P, P, P, P],
            own_buckets=[B.LOW_NUM.value, B.LOW_NUM.value, B.ACE.value, B.LOW_NUM.value],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.HIGH_KING.value, game_phase=GamePhase.MID.value,
            turn_progress=0.3,
        ),
        "Good hand, mid-game. Cambia reasonable but DrawDiscard bad (high K).",
    ))

    # S6: Known good hand, late game. Should strongly Cambia.
    scenarios.append((
        "S6: START_TURN, good hand (1,2,2,3=8), late game, low stock",
        *_build_start_turn_state(
            own_tags=[P, P, P, P],
            own_buckets=[B.ACE.value, B.LOW_NUM.value, B.LOW_NUM.value, B.LOW_NUM.value],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.MID_NUM.value, game_phase=GamePhase.LATE.value,
            stock_est=StockpileEstimate.LOW.value,
            turn_progress=0.6,
        ),
        "Late game, good hand, low stock → strong Cambia signal.",
    ))

    # ── POST_DRAW scenarios ──

    # S7: Drew Ace (low), have one known high card. Should replace the high card.
    scenarios.append((
        "S7: POST_DRAW, drew ACE, have HIGH_KING in slot 2",
        *_build_post_draw_state(
            own_tags=[P, U, P, U],
            own_buckets=[B.LOW_NUM.value, 0, B.HIGH_KING.value, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            drawn_card_bucket=B.ACE.value,
            discard_bucket=B.MID_NUM.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "Should replace slot 2 (HIGH_KING). Discard is terrible.",
    ))

    # S8: Drew a 7 (PEEK_SELF), have unknown cards. Should use ability.
    scenarios.append((
        "S8: POST_DRAW, drew 7 (PEEK_SELF), 2 unknown slots",
        *_build_post_draw_state(
            own_tags=[P, P, U, U],
            own_buckets=[B.LOW_NUM.value, B.ACE.value, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            drawn_card_bucket=B.PEEK_SELF.value,
            discard_bucket=B.MID_NUM.value,
            game_phase=GamePhase.MID.value,
            has_ability=True,
            turn_progress=0.25,
        ),
        "7 drawn, unknown slots exist → Discard(ability) to peek. Value of info.",
    ))

    # S9: Drew HIGH_KING, all cards low. Should discard (don't replace).
    scenarios.append((
        "S9: POST_DRAW, drew HIGH_KING, all own slots are low",
        *_build_post_draw_state(
            own_tags=[P, P, P, P],
            own_buckets=[B.ACE.value, B.LOW_NUM.value, B.LOW_NUM.value, B.ACE.value],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            drawn_card_bucket=B.HIGH_KING.value,
            discard_bucket=B.MID_NUM.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "HIGH_KING drawn, all slots low → discard, never replace.",
    ))

    # S10: Drew LOW_NUM (2-4), have one unknown. Replace unknown or keep?
    scenarios.append((
        "S10: POST_DRAW, drew LOW_NUM, 2 known low + 2 unknown",
        *_build_post_draw_state(
            own_tags=[P, P, U, U],
            own_buckets=[B.ACE.value, B.LOW_NUM.value, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            drawn_card_bucket=B.LOW_NUM.value,
            discard_bucket=B.MID_NUM.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "LOW drawn, unknown slots → replace unknown (risk-neutral) or discard.",
    ))

    # ── ABILITY_SELECT (peek own) scenarios ──

    # S11: Peek own — 2 known slots, 2 unknown. Should peek unknown.
    scenarios.append((
        "S11: PEEK_OWN, slots 0,1 known, slots 2,3 unknown",
        *_build_ability_peek_own_state(
            own_tags=[P, P, U, U],
            own_buckets=[B.LOW_NUM.value, B.ACE.value, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.PEEK_SELF.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "Should peek slot 2 or 3 (unknown). Peeking known slots is wasted.",
    ))

    # S12: Peek own — all unknown. Any slot equally good.
    scenarios.append((
        "S12: PEEK_OWN, all 4 slots unknown",
        *_build_ability_peek_own_state(
            own_tags=[U, U, U, U],
            own_buckets=[0, 0, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.PEEK_SELF.value,
            game_phase=GamePhase.EARLY.value,
            turn_progress=0.1,
        ),
        "All unknown → uniform over 4 slots is correct.",
    ))

    # ── SNAP_DECISION scenarios ──

    # S13: Discard is ACE, own slot 0 is known ACE. Should snap.
    scenarios.append((
        "S13: SNAP, discard=ACE, own slot 0 = known ACE",
        *_build_snap_decision_state(
            own_tags=[P, U, U, U],
            own_buckets=[B.ACE.value, 0, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.ACE.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "Known match at slot 0 → SnapOwn(0) strongly dominates PassSnap.",
    ))

    # S14: Discard is HIGH_KING, no known cards. Should pass.
    scenarios.append((
        "S14: SNAP, discard=HIGH_KING, all unknown",
        *_build_snap_decision_state(
            own_tags=[U, U, U, U],
            own_buckets=[0, 0, 0, 0],
            opp_tags=[U, U, U, U], opp_buckets=[0, 0, 0, 0],
            discard_bucket=B.HIGH_KING.value,
            game_phase=GamePhase.MID.value,
            turn_progress=0.25,
        ),
        "No info, snapping is blind gamble → PassSnap should dominate.",
    ))

    return scenarios


def load_network(checkpoint_path, device="cpu"):
    """Load a ResidualAdvantageNetwork from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dcfr_config = ckpt.get("dcfr_config", {})
    hidden_dim = dcfr_config.get("hidden_dim", 256)
    num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
    input_dim = dcfr_config.get("input_dim", EP_PBS_INPUT_DIM)

    net = ResidualAdvantageNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
        validate_inputs=False,
    )
    net.load_state_dict(ckpt["advantage_net_state_dict"])
    net.eval()
    # Store the actual input_dim the network expects
    net._ckpt_input_dim = input_dim
    return net, ckpt


def load_ema_network(ema_path, dcfr_config, device="cpu"):
    """Load EMA weights into a fresh ResidualAdvantageNetwork."""
    ema_data = torch.load(ema_path, map_location=device, weights_only=True)
    hidden_dim = dcfr_config.get("hidden_dim", 256)
    num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
    input_dim = dcfr_config.get("input_dim", EP_PBS_INPUT_DIM)

    net = ResidualAdvantageNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
        validate_inputs=False,
    )
    net.load_state_dict(ema_data["ema_state_dict"])
    net.eval()
    net._ckpt_input_dim = input_dim
    return net


def make_random_network(dcfr_config):
    """Create a fresh random-init network (no training)."""
    hidden_dim = dcfr_config.get("hidden_dim", 256)
    num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
    input_dim = dcfr_config.get("input_dim", EP_PBS_INPUT_DIM)

    net = ResidualAdvantageNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
        validate_inputs=False,
    )
    net.eval()
    net._ckpt_input_dim = input_dim
    return net


def evaluate_scenario(net, features, mask, net_label=""):
    """Run one scenario through the network. Return (advantages, strategy) for legal actions."""
    # Slice features to match network input_dim (handles 200 vs 224)
    net_input_dim = getattr(net, "_ckpt_input_dim", net._input_dim)
    feat = features[:net_input_dim]
    with torch.inference_mode():
        feat_t = torch.from_numpy(feat).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        adv_t = net(feat_t, mask_t)
        strat_t = get_strategy_from_advantages(adv_t, mask_t)
        adv = adv_t.squeeze(0).numpy()
        strat = strat_t.squeeze(0).numpy()
    return adv, strat


def print_scenario_result(name, adv, strat, mask, expected, net_label):
    """Pretty-print the result for one scenario."""
    legal_indices = np.where(mask)[0]
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"  Network: {net_label}")
    print(f"  Expected: {expected}")
    print(f"  Legal actions: {len(legal_indices)}")
    print(f"  {'─' * 70}")

    # Sort legal actions by strategy probability (descending)
    entries = []
    for idx in legal_indices:
        entries.append((idx, action_name(idx), adv[idx], strat[idx]))
    entries.sort(key=lambda x: -x[3])

    print(f"  {'Action':<28} {'Advantage':>10} {'Strategy':>10} {'Rank':>5}")
    print(f"  {'─' * 56}")
    for rank, (idx, name_str, a, s) in enumerate(entries, 1):
        marker = " ◀" if s > 0.3 else ""
        print(f"  {name_str:<28} {a:>10.4f} {s:>10.4f} {rank:>5}{marker}")

    # Summary statistics for legal actions
    legal_advs = adv[legal_indices]
    legal_advs_finite = legal_advs[np.isfinite(legal_advs)]
    if len(legal_advs_finite) > 1:
        print(f"\n  Adv stats: mean={legal_advs_finite.mean():.4f}, "
              f"std={legal_advs_finite.std():.4f}, "
              f"range=[{legal_advs_finite.min():.4f}, {legal_advs_finite.max():.4f}]")
        # Entropy of strategy
        legal_strats = strat[legal_indices]
        legal_strats = legal_strats[legal_strats > 0]
        entropy = -np.sum(legal_strats * np.log(legal_strats + 1e-10))
        max_entropy = np.log(len(legal_indices))
        print(f"  Strategy entropy: {entropy:.4f} / {max_entropy:.4f} "
              f"(ratio={entropy / max_entropy:.3f})")


def sensitivity_analysis(net, base_features, base_mask, vary_dim, vary_range, name, net_label):
    """Vary one input dimension and see how strategy changes."""
    print(f"\n{'─' * 60}")
    print(f"  Sensitivity: varying {name} (dim {vary_dim})")
    print(f"  Network: {net_label}")

    cambia_probs = []
    draw_stock_probs = []
    for val in vary_range:
        f = base_features.copy()
        f[vary_dim] = val
        _, strat = evaluate_scenario(net, f, base_mask)
        legal = np.where(base_mask)[0]
        cambia_probs.append(strat[IDX_CAMBIA] if IDX_CAMBIA in legal else 0.0)
        draw_stock_probs.append(strat[IDX_DRAW_STOCK] if IDX_DRAW_STOCK in legal else 0.0)

    print(f"  {'Input val':>10} {'P(Cambia)':>10} {'P(DrawStock)':>12}")
    for val, cp, dp in zip(vary_range, cambia_probs, draw_stock_probs):
        print(f"  {val:>10.2f} {cp:>10.4f} {dp:>12.4f}")


def cross_network_comparison(nets, scenarios):
    """Compare the same scenario across multiple networks."""
    print("\n" + "=" * 80)
    print("  CROSS-NETWORK COMPARISON: Top action and Cambia probability")
    print("=" * 80)

    # For each scenario, show top action and cambia prob for each network
    header = f"  {'Scenario':<50}"
    for label, _ in nets:
        header += f" {label:>12}"
    print(header)
    print("  " + "─" * (50 + 13 * len(nets)))

    for scenario_name, features, mask, expected in scenarios:
        short_name = scenario_name[:48]
        row_top = f"  {short_name:<50}"
        row_cam = f"  {'  P(Cambia)':<50}"
        for label, net in nets:
            adv, strat = evaluate_scenario(net, features, mask)
            legal = np.where(mask)[0]
            top_idx = legal[np.argmax(strat[legal])]
            top_name = action_name(top_idx)[:10]
            row_top += f" {top_name:>12}"
            if IDX_CAMBIA in legal:
                row_cam += f" {strat[IDX_CAMBIA]:>12.4f}"
            else:
                row_cam += f" {'n/a':>12}"
        print(row_top)
        if IDX_CAMBIA in np.where(mask)[0]:
            print(row_cam)


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "runs", "interleaved-resnet-adaptive", "checkpoints")
    best_ckpt = os.path.join(base, "best.pt")  # symlink → iter 450
    ema_ckpt = os.path.join(base, "deep_cfr_checkpoint_ema.pt")

    if not os.path.exists(best_ckpt):
        print(f"ERROR: checkpoint not found at {best_ckpt}")
        sys.exit(1)

    print("Loading networks...")
    raw_net, ckpt = load_network(best_ckpt)
    dcfr_config = ckpt.get("dcfr_config", {})

    ema_net = None
    if os.path.exists(ema_ckpt):
        ema_net = load_ema_network(ema_ckpt, dcfr_config)
        print(f"  Loaded EMA network from {ema_ckpt}")
    else:
        print(f"  No EMA checkpoint found at {ema_ckpt}")

    random_net = make_random_network(dcfr_config)
    print(f"  Raw net: iter 450 (best)")
    print(f"  Random net: fresh initialization")

    scenarios = build_scenarios()
    print(f"\nBuilt {len(scenarios)} canonical scenarios.\n")

    # ── Detailed per-scenario analysis (EMA or raw) ──
    primary_net = ema_net if ema_net else raw_net
    primary_label = "EMA" if ema_net else "Raw-450"

    print("\n" + "#" * 80)
    print(f"  DETAILED SCENARIO ANALYSIS — {primary_label} network")
    print("#" * 80)

    for name, features, mask, expected in scenarios:
        adv, strat = evaluate_scenario(primary_net, features, mask, primary_label)
        print_scenario_result(name, adv, strat, mask, expected, primary_label)

    # ── Cross-network comparison ──
    nets = [(primary_label, primary_net), ("Random", random_net)]
    if ema_net and primary_label != "EMA":
        nets.append(("EMA", ema_net))
    if primary_label != "Raw-450":
        nets.append(("Raw-450", raw_net))
    cross_network_comparison(nets, scenarios)

    # ── Aggregate statistics ──
    print("\n" + "#" * 80)
    print("  AGGREGATE STATISTICS")
    print("#" * 80)

    for label, net in nets:
        cambia_dominant_count = 0
        start_turn_count = 0
        all_entropies = []
        for name, features, mask, expected in scenarios:
            adv, strat = evaluate_scenario(net, features, mask)
            legal = np.where(mask)[0]
            legal_strats = strat[legal]
            legal_strats = legal_strats[legal_strats > 0]
            entropy = -np.sum(legal_strats * np.log(legal_strats + 1e-10))
            max_entropy = np.log(len(legal))
            all_entropies.append(entropy / max_entropy if max_entropy > 0 else 0)

            if IDX_CAMBIA in legal:
                start_turn_count += 1
                if strat[IDX_CAMBIA] > 0.4:
                    cambia_dominant_count += 1

        print(f"\n  {label}:")
        print(f"    Mean normalized entropy: {np.mean(all_entropies):.3f}")
        if start_turn_count > 0:
            print(f"    Cambia dominant (>40%) in {cambia_dominant_count}/{start_turn_count} START_TURN scenarios")

    # ── Sensitivity: Does the network respond to hand quality changes? ──
    print("\n" + "#" * 80)
    print("  SENSITIVITY ANALYSIS")
    print("#" * 80)

    # Build a base START_TURN state with 2 known low cards, vary turn_progress
    base_feat, base_mask = _build_start_turn_state(
        own_tags=[EpistemicTag.PRIV_OWN, EpistemicTag.PRIV_OWN, EpistemicTag.UNK, EpistemicTag.UNK],
        own_buckets=[CardBucket.ACE.value, CardBucket.LOW_NUM.value, 0, 0],
        opp_tags=[EpistemicTag.UNK] * 4, opp_buckets=[0] * 4,
        discard_bucket=CardBucket.MID_NUM.value,
        game_phase=GamePhase.MID.value,
        turn_progress=0.3,
    )
    sensitivity_analysis(
        primary_net, base_feat, base_mask,
        vary_dim=222, vary_range=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        name="turn_progress", net_label=primary_label,
    )

    # Sensitivity: swap slot 0 known card bucket through all values
    print(f"\n  Sensitivity: varying own slot 0 bucket identity")
    print(f"  Network: {primary_label}")
    # Slot 0 identity starts at dim 42 + 0*13 + 4 = 46 (in interleaved: dim 42 is start of slot 0,
    # first 4 dims are tag, next 9 are identity).
    # Actually in interleaved: public=42, then slot i starts at 42 + i*13.
    # Tag is 4 dims, identity is 9 dims.
    slot0_identity_start = 42 + 4  # slot 0, skip 4-dim tag, identity starts at dim 46
    print(f"  {'Bucket':<18} {'P(Cambia)':>10} {'P(DrawStock)':>12} {'P(DrawDiscard)':>14}")
    for bucket_name, bucket_val in [
        ("NEG_KING(-1)", CardBucket.NEG_KING.value),
        ("ZERO(0)", CardBucket.ZERO.value),
        ("ACE(1)", CardBucket.ACE.value),
        ("LOW_NUM(2-4)", CardBucket.LOW_NUM.value),
        ("MID_NUM(5-6)", CardBucket.MID_NUM.value),
        ("PEEK_SELF(7-8)", CardBucket.PEEK_SELF.value),
        ("PEEK_OTHER(9-T)", CardBucket.PEEK_OTHER.value),
        ("SWAP_BLIND(J-Q)", CardBucket.SWAP_BLIND.value),
        ("HIGH_KING(13)", CardBucket.HIGH_KING.value),
    ]:
        f = base_feat.copy()
        # Zero out slot 0 identity, set the right bucket
        f[slot0_identity_start:slot0_identity_start + 9] = 0.0
        f[slot0_identity_start + bucket_val] = 1.0
        _, strat = evaluate_scenario(primary_net, f, base_mask)
        print(f"  {bucket_name:<18} {strat[IDX_CAMBIA]:>10.4f} "
              f"{strat[IDX_DRAW_STOCK]:>12.4f} {strat[IDX_DRAW_DISCARD]:>14.4f}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
