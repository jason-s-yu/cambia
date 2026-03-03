#!/usr/bin/env python3
"""Checkpoint progression diagnostic.

Runs a fixed set of canonical scenarios across multiple checkpoints from
a training run, producing a table showing how advantage quality evolves
over iterations. Supports both ResNet and MLP checkpoints.

Phase A+ item #1 (checkpoint progression) and #2 (cross-architecture).

Usage:
    cd cfr && python scripts/diag_checkpoint_progression.py [run_dir] [--iters 25,100,200,300,450,600]
    cd cfr && python scripts/diag_checkpoint_progression.py runs/interleaved-resnet-adaptive
    cd cfr && python scripts/diag_checkpoint_progression.py runs/prod-full-333 --legacy
"""

import sys
import os
import argparse
import glob
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from src.encoding import NUM_ACTIONS
from src.networks import build_advantage_network, get_strategy_from_advantages
from src.constants import (
    EP_PBS_INPUT_DIM,
    EpistemicTag,
    CardBucket,
    DecisionContext,
    GamePhase,
    StockpileEstimate,
)

# ── Encoding helpers ──

def encode_eppbs_interleaved(slot_tags, slot_buckets, discard_bucket, stock_est,
                              game_phase, decision_ctx, cambia_state=2,
                              drawn_card_bucket=-1, own_hand_size=4,
                              opp_hand_size=4, turn_progress=0.0):
    """Wrap the EP-PBS interleaved encoder."""
    from src.encoding import encode_infoset_eppbs_interleaved
    return encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags, slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket, stock_estimate=stock_est,
        game_phase=game_phase, decision_context=decision_ctx,
        cambia_state=cambia_state, drawn_card_bucket=drawn_card_bucket,
        own_hand_size=own_hand_size, opp_hand_size=opp_hand_size,
        turn_progress=turn_progress,
    )


def encode_legacy(slot_tags, slot_buckets, discard_bucket, stock_est,
                   game_phase, decision_ctx, cambia_state=2,
                   drawn_card_bucket=-1, own_hand_size=4,
                   opp_hand_size=4, turn_progress=0.0):
    """Wrap the legacy encoder (222-dim)."""
    from src.encoding import encode_infoset
    from src.agent_state import AgentState
    # Legacy encoding needs a full AgentState. Build a minimal stub.
    # This is fragile but we only need it for a few canonical states.
    return encode_infoset(
        slot_tags=slot_tags, slot_buckets=slot_buckets,
        discard_top_bucket=discard_bucket, stock_estimate=stock_est,
        game_phase=game_phase, decision_context=decision_ctx,
        cambia_state=cambia_state, drawn_card_bucket=drawn_card_bucket,
        own_hand_size=own_hand_size, opp_hand_size=opp_hand_size,
    )


# ── Action indices ──

IDX_DRAW_STOCK = 0
IDX_DRAW_DISCARD = 1
IDX_CAMBIA = 2
IDX_DISCARD_NO_ABILITY = 3
IDX_DISCARD_ABILITY = 4
IDX_REPLACE_BASE = 5
IDX_PEEK_OWN_BASE = 11
IDX_PASS_SNAP = 97
IDX_SNAP_OWN_BASE = 98

ACTION_NAMES = {}
ACTION_NAMES[0] = "DrawStock"
ACTION_NAMES[1] = "DrawDisc"
ACTION_NAMES[2] = "Cambia"
ACTION_NAMES[3] = "Disc(no)"
ACTION_NAMES[4] = "Disc(abl)"
for i in range(6):
    ACTION_NAMES[5 + i] = f"Repl({i})"
    ACTION_NAMES[11 + i] = f"PkOwn({i})"
    ACTION_NAMES[17 + i] = f"PkOpp({i})"
    ACTION_NAMES[98 + i] = f"SnpOwn({i})"
ACTION_NAMES[97] = "PassSnap"


def _mask(indices):
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    for i in indices:
        m[i] = True
    return m


# ── Canonical Scenarios (compact set for progression tracking) ──

def build_scenarios(encode_fn, input_dim):
    """Return list of (name, features, mask, correct_action_description, key_metric_fn).

    key_metric_fn(strategy, raw_advantages) -> (metric_name, metric_value)
    Each metric captures the ONE number that tells us if the network is right or wrong.
    """
    P = EpistemicTag.PRIV_OWN
    U = EpistemicTag.UNK
    B = CardBucket

    def _pad(features):
        if len(features) > input_dim:
            return features[:input_dim]
        elif len(features) < input_dim:
            return np.pad(features, (0, input_dim - len(features)))
        return features

    def _st(own_tags, own_buckets, opp_tags, opp_buckets, discard, phase, stock=StockpileEstimate.HIGH.value, tp=0.05):
        slot_tags = list(own_tags) + [U] * (6 - len(own_tags)) + list(opp_tags) + [U] * (6 - len(opp_tags))
        slot_bkts = list(own_buckets) + [0] * (6 - len(own_buckets)) + list(opp_buckets) + [0] * (6 - len(opp_buckets))
        f = encode_fn(slot_tags, slot_bkts, discard, stock, phase,
                       DecisionContext.START_TURN.value, turn_progress=tp)
        return _pad(f), _mask([IDX_DRAW_STOCK, IDX_DRAW_DISCARD, IDX_CAMBIA])

    def _pd(own_tags, own_buckets, opp_tags, opp_buckets, drawn, discard, phase, has_abl=False, tp=0.25):
        slot_tags = list(own_tags) + [U] * (6 - len(own_tags)) + list(opp_tags) + [U] * (6 - len(opp_tags))
        slot_bkts = list(own_buckets) + [0] * (6 - len(own_buckets)) + list(opp_buckets) + [0] * (6 - len(opp_buckets))
        f = encode_fn(slot_tags, slot_bkts, discard, StockpileEstimate.HIGH.value, phase,
                       DecisionContext.POST_DRAW.value, drawn_card_bucket=drawn, turn_progress=tp)
        legal = [IDX_DISCARD_NO_ABILITY]
        if has_abl:
            legal.append(IDX_DISCARD_ABILITY)
        for s in range(4):
            legal.append(IDX_REPLACE_BASE + s)
        return _pad(f), _mask(legal)

    def _pk(own_tags, own_buckets, opp_tags, opp_buckets, discard, phase, tp=0.25):
        slot_tags = list(own_tags) + [U] * (6 - len(own_tags)) + list(opp_tags) + [U] * (6 - len(opp_tags))
        slot_bkts = list(own_buckets) + [0] * (6 - len(own_buckets)) + list(opp_buckets) + [0] * (6 - len(opp_buckets))
        f = encode_fn(slot_tags, slot_bkts, discard, StockpileEstimate.HIGH.value, phase,
                       DecisionContext.ABILITY_SELECT.value, turn_progress=tp)
        return _pad(f), _mask([IDX_PEEK_OWN_BASE + s for s in range(4)])

    def _sn(own_tags, own_buckets, opp_tags, opp_buckets, discard, phase, tp=0.25):
        slot_tags = list(own_tags) + [U] * (6 - len(own_tags)) + list(opp_tags) + [U] * (6 - len(opp_tags))
        slot_bkts = list(own_buckets) + [0] * (6 - len(own_buckets)) + list(opp_buckets) + [0] * (6 - len(opp_buckets))
        f = encode_fn(slot_tags, slot_bkts, discard, StockpileEstimate.HIGH.value, phase,
                       DecisionContext.SNAP_DECISION.value, turn_progress=tp)
        return _pad(f), _mask([IDX_PASS_SNAP] + [IDX_SNAP_OWN_BASE + s for s in range(4)])

    scenarios = []

    # S1: Perfect hand, T1C should dominate
    f, m = _st([P,P,P,P], [B.ZERO.value, B.NEG_KING.value, B.ACE.value, B.ACE.value],
               [U,U,U,U], [0,0,0,0], B.MID_NUM.value, GamePhase.EARLY.value)
    scenarios.append(("Perfect(1) T1C", f, m, "P(Cambia)",
                      lambda s, a: s[IDX_CAMBIA]))

    # S2: Terrible hand, should NOT Cambia
    f, m = _st([P,P,P,P], [B.HIGH_KING.value, B.SWAP_BLIND.value, B.PEEK_OTHER.value, B.PEEK_OTHER.value],
               [U,U,U,U], [0,0,0,0], B.MID_NUM.value, GamePhase.EARLY.value)
    scenarios.append(("Terrible(44) T1", f, m, "P(Cambia)",
                      lambda s, a: s[IDX_CAMBIA]))

    # S3: Good hand, late game — should strongly Cambia
    f, m = _st([P,P,P,P], [B.ACE.value, B.LOW_NUM.value, B.LOW_NUM.value, B.LOW_NUM.value],
               [U,U,U,U], [0,0,0,0], B.MID_NUM.value, GamePhase.LATE.value,
               stock=StockpileEstimate.LOW.value, tp=0.6)
    scenarios.append(("Good(8) late", f, m, "P(Cambia)",
                      lambda s, a: s[IDX_CAMBIA]))

    # S4: All unknown mid-game — Cambia is suicidal
    f, m = _st([U,U,U,U], [0,0,0,0], [U,U,U,U], [0,0,0,0],
               B.LOW_NUM.value, GamePhase.MID.value,
               stock=StockpileEstimate.MEDIUM.value, tp=0.3)
    scenarios.append(("Unknown mid", f, m, "P(Cambia)",
                      lambda s, a: s[IDX_CAMBIA]))

    # S5: Drew ACE, slot 2 has HIGH_KING — should replace slot 2
    f, m = _pd([P,U,P,U], [B.LOW_NUM.value, 0, B.HIGH_KING.value, 0],
               [U,U,U,U], [0,0,0,0], B.ACE.value, B.MID_NUM.value, GamePhase.MID.value)
    scenarios.append(("Drew A,has K", f, m, "P(Repl2)",
                      lambda s, a: s[IDX_REPLACE_BASE + 2]))

    # S6: Drew HIGH_KING, all low — should discard
    f, m = _pd([P,P,P,P], [B.ACE.value, B.LOW_NUM.value, B.LOW_NUM.value, B.ACE.value],
               [U,U,U,U], [0,0,0,0], B.HIGH_KING.value, B.MID_NUM.value, GamePhase.MID.value)
    scenarios.append(("Drew K,all low", f, m, "P(Disc)",
                      lambda s, a: s[IDX_DISCARD_NO_ABILITY]))

    # S7: Peek own — 2 known, 2 unknown — should peek unknown (slot 2 or 3)
    f, m = _pk([P,P,U,U], [B.LOW_NUM.value, B.ACE.value, 0, 0],
               [U,U,U,U], [0,0,0,0], B.PEEK_SELF.value, GamePhase.MID.value)
    scenarios.append(("Peek 2kn/2unk", f, m, "P(unk slot)",
                      lambda s, a: s[IDX_PEEK_OWN_BASE+2] + s[IDX_PEEK_OWN_BASE+3]))

    # S8: Snap — known ACE match at slot 0, discard=ACE
    f, m = _sn([P,U,U,U], [B.ACE.value, 0, 0, 0],
               [U,U,U,U], [0,0,0,0], B.ACE.value, GamePhase.MID.value)
    scenarios.append(("Snap ACE s0", f, m, "P(SnapOwn0)",
                      lambda s, a: s[IDX_SNAP_OWN_BASE]))

    # S9: Snap — HIGH_KING discard, all unknown — should pass
    f, m = _sn([U,U,U,U], [0,0,0,0], [U,U,U,U], [0,0,0,0],
               B.HIGH_KING.value, GamePhase.MID.value)
    scenarios.append(("Snap K blind", f, m, "P(Pass)",
                      lambda s, a: s[IDX_PASS_SNAP]))

    return scenarios


def detect_arch_from_state_dict(state_dict):
    """Detect network architecture from state dict keys."""
    keys = list(state_dict.keys())
    if any(k.startswith("input_proj") for k in keys):
        if any(k.startswith("tag_embed") for k in keys):
            return "slot_film"
        return "residual"
    if any(k.startswith("net.") for k in keys):
        return "mlp"
    return "residual"  # fallback


def load_checkpoint(path, device="cpu"):
    """Load any advantage network checkpoint. Returns (network, dcfr_config)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("dcfr_config", {})
    state_dict = ckpt["advantage_net_state_dict"]

    # Auto-detect architecture from state dict (old checkpoints lack network_type)
    detected_type = detect_arch_from_state_dict(state_dict)
    net_type = cfg.get("network_type") or detected_type
    input_dim = cfg.get("input_dim", 200)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_layers = cfg.get("num_hidden_layers") or 3

    use_residual = net_type == "residual"
    net = build_advantage_network(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=NUM_ACTIONS,
        dropout=0.1, validate_inputs=False,
        num_hidden_layers=num_layers, use_residual=use_residual,
        network_type=net_type,
    )
    net.load_state_dict(state_dict)
    net.eval()
    cfg["_detected_type"] = detected_type
    return net, cfg


def load_ema(run_dir, device="cpu"):
    """Load EMA checkpoint if it exists."""
    ema_path = os.path.join(run_dir, "checkpoints", "deep_cfr_checkpoint_ema.pt")
    if not os.path.exists(ema_path):
        return None, None
    ema_data = torch.load(ema_path, map_location=device, weights_only=True)
    # Need dcfr_config from a regular checkpoint
    any_ckpt = glob.glob(os.path.join(run_dir, "checkpoints", "deep_cfr_checkpoint_iter_*.pt"))
    if not any_ckpt:
        return None, None
    cfg = torch.load(any_ckpt[0], map_location=device, weights_only=True).get("dcfr_config", {})
    net_type = cfg.get("network_type", "residual")
    use_residual = cfg.get("use_residual", True)
    input_dim = cfg.get("input_dim", 200)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_layers = cfg.get("num_hidden_layers", 3)

    net = build_advantage_network(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=NUM_ACTIONS,
        dropout=0.1, validate_inputs=False,
        num_hidden_layers=num_layers, use_residual=use_residual,
        network_type=net_type,
    )
    net.load_state_dict(ema_data["ema_state_dict"])
    net.eval()
    return net, cfg


def evaluate_scenarios(net, scenarios, device="cpu"):
    """Run all scenarios through the network. Returns list of metric values."""
    results = []
    for name, features, mask, metric_label, metric_fn in scenarios:
        feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).bool().to(device)
        with torch.inference_mode():
            raw = net(feat_t, mask_t).squeeze(0).cpu()
            strat = get_strategy_from_advantages(raw.unsqueeze(0), mask_t.cpu()).squeeze(0).numpy()
            raw_np = raw.numpy()
        val = metric_fn(strat, raw_np)
        results.append(float(val))
    return results


def find_checkpoints(run_dir, iters=None):
    """Find checkpoint files. Returns sorted list of (iter_num, path)."""
    pattern = os.path.join(run_dir, "checkpoints", "deep_cfr_checkpoint_iter_*.pt")
    paths = glob.glob(pattern)
    items = []
    for p in paths:
        m = re.search(r"iter_(\d+)\.pt$", p)
        if m:
            it = int(m.group(1))
            if iters is None or it in iters:
                items.append((it, p))
    return sorted(items)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint progression diagnostic")
    parser.add_argument("run_dir", nargs="?", default="runs/interleaved-resnet-adaptive",
                        help="Path to run directory")
    parser.add_argument("--iters", type=str, default=None,
                        help="Comma-separated iteration numbers (default: all at 100-iter intervals)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy 222-dim encoding for canonical states")
    parser.add_argument("--all", action="store_true",
                        help="Include all checkpoints (not just 100-iter intervals)")
    args = parser.parse_args()

    run_dir = args.run_dir
    device = "cpu"

    # Determine which iterations to test
    if args.iters:
        target_iters = set(int(x) for x in args.iters.split(","))
    elif args.all:
        target_iters = None  # all
    else:
        target_iters = None  # will filter to reasonable set below

    checkpoints = find_checkpoints(run_dir, target_iters)
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}/checkpoints/")
        sys.exit(1)

    # If no explicit iters and not --all, pick reasonable subset
    if target_iters is None and not args.all:
        all_iters = [it for it, _ in checkpoints]
        # Include: 25, 50, 100, then every 100
        keep = set()
        for it in all_iters:
            if it <= 100 and it in (2, 25, 50, 75, 100):
                keep.add(it)
            elif it % 100 == 0:
                keep.add(it)
            elif it == max(all_iters):
                keep.add(it)
        # Always include known best iterations
        for special in [450, 500, 1075]:
            if special in all_iters:
                keep.add(special)
        checkpoints = [(it, p) for it, p in checkpoints if it in keep]

    # Detect encoding from first checkpoint
    _, first_cfg = load_checkpoint(checkpoints[0][1], device)
    input_dim = first_cfg.get("input_dim", 200)
    net_type = first_cfg.get("network_type", "residual")
    use_residual = first_cfg.get("use_residual", True)
    encoding_mode = first_cfg.get("encoding_mode", "legacy")
    encoding_layout = first_cfg.get("encoding_layout", "auto")

    is_legacy = args.legacy or encoding_mode == "legacy" or input_dim == 222
    if is_legacy:
        try:
            encode_fn = encode_legacy
        except Exception:
            print("Legacy encoding not available. Cannot run canonical tests on legacy checkpoints.")
            sys.exit(1)
    else:
        encode_fn = encode_eppbs_interleaved

    print(f"Run: {run_dir}")
    print(f"Network: {net_type} (use_residual={use_residual}), input_dim={input_dim}")
    print(f"Encoding: {'legacy' if is_legacy else f'{encoding_mode}/{encoding_layout}'}")
    print(f"Checkpoints: {len(checkpoints)} ({checkpoints[0][0]} to {checkpoints[-1][0]})")
    print()

    # Build scenarios
    scenarios = build_scenarios(encode_fn, input_dim)
    scenario_names = [s[0] for s in scenarios]
    metric_labels = [s[3] for s in scenarios]

    # Create random-init baseline
    random_net = build_advantage_network(
        input_dim=input_dim, hidden_dim=first_cfg.get("hidden_dim", 256),
        output_dim=NUM_ACTIONS, dropout=0.1, validate_inputs=False,
        num_hidden_layers=first_cfg.get("num_hidden_layers", 3),
        use_residual=use_residual, network_type=net_type,
    )
    random_net.eval()
    random_results = evaluate_scenarios(random_net, scenarios, device)

    # Evaluate all checkpoints
    all_results = [("Random", random_results)]
    for it, path in checkpoints:
        net, _ = load_checkpoint(path, device)
        results = evaluate_scenarios(net, scenarios, device)
        all_results.append((f"Iter {it}", results))

    # Try EMA
    ema_net, _ = load_ema(run_dir, device)
    if ema_net is not None:
        ema_results = evaluate_scenarios(ema_net, scenarios, device)
        all_results.append(("EMA", ema_results))

    # Print compact table
    # Header
    col_w = 12
    name_w = 10
    print(f"{'Ckpt':<{name_w}}", end="")
    for sname in scenario_names:
        # Truncate scenario names for table
        short = sname[:col_w-1]
        print(f" {short:>{col_w}}", end="")
    print()

    print(f"{'Metric':<{name_w}}", end="")
    for ml in metric_labels:
        short = ml[:col_w-1]
        print(f" {short:>{col_w}}", end="")
    print()

    print("-" * (name_w + (col_w + 1) * len(scenarios)))

    # Correct direction indicators
    # S1 Perfect T1C: high is correct
    # S2 Terrible: low is correct
    # S3 Good late: high is correct
    # S4 Unknown mid: low is correct
    # S5 Drew A has K: high is correct (replace)
    # S6 Drew K all low: high is correct (discard)
    # S7 Peek unk: high is correct
    # S8 Snap match: high is correct
    # S9 Snap blind pass: high is correct
    correct_high = [True, False, True, False, True, True, True, True, True]

    for label, results in all_results:
        print(f"{label:<{name_w}}", end="")
        for i, val in enumerate(results):
            print(f" {val:>{col_w}.3f}", end="")
        print()

    # Correctness summary
    print()
    print("Correct direction: ", end="")
    for ch in correct_high:
        print(f" {'HIGH':>{col_w}}" if ch else f" {'LOW':>{col_w}}", end="")
    print()

    # Inversion detection
    print()
    print("INVERSION ANALYSIS (comparing each checkpoint to correct direction):")
    print("-" * 80)
    for label, results in all_results:
        inversions = 0
        for i, val in enumerate(results):
            if correct_high[i] and val < 0.15:
                inversions += 1
            elif not correct_high[i] and val > 0.4:
                inversions += 1
        n = len(results)
        print(f"  {label:<{name_w}}: {inversions}/{n} inversions "
              f"({'OK' if inversions == 0 else 'INVERTED' if inversions >= 3 else 'PARTIAL'})")

    # Key diagnostic: Cambia signal correlation with hand quality
    print()
    print("CAMBIA SIGNAL VS HAND QUALITY:")
    print("  (Perfect=1 should have HIGHEST P(Cambia), Terrible=44 should have LOWEST)")
    print("-" * 80)
    for label, results in all_results:
        # S1=Perfect, S2=Terrible, S3=Good late, S4=Unknown mid
        perfect, terrible, good_late, unknown = results[0], results[1], results[2], results[3]
        correct_order = perfect > good_late > unknown > terrible
        ordering = "CORRECT" if correct_order else "INVERTED"
        print(f"  {label:<{name_w}}: Perfect={perfect:.3f}  Good-late={good_late:.3f}  "
              f"Unknown={unknown:.3f}  Terrible={terrible:.3f}  [{ordering}]")


if __name__ == "__main__":
    main()
