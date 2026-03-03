#!/usr/bin/env python3
"""Diagnostic script comparing ESCHER vs Phase 2 checkpoints."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, "/home/agent/dev/cambia/cfr/src")

ESCHER_PATH = "/home/agent/dev/cambia/cfr/runs/escher-interleaved/checkpoints/deep_cfr_checkpoint.pt"
PHASE2_PATH = "/home/agent/dev/cambia/cfr/runs/interleaved-resnet-adaptive/checkpoints/deep_cfr_checkpoint.pt"
EP_PBS_INPUT_DIM = 224
NUM_ACTIONS = 146

def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def print_weight_stats(state_dict, label, max_layers=8):
    print(f"\n  {label} weight stats (first {max_layers} layers):")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= max_layers:
            print(f"    ... ({len(state_dict)} total layers)")
            break
        if v.dtype.is_floating_point:
            print(f"    {k}: shape={list(v.shape)} mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")

def analyze_checkpoint(path, name):
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {name}")
    print(f"{'='*60}")
    ckpt = load_ckpt(path)

    print(f"\nTop-level keys: {sorted(ckpt.keys())}")

    # Metadata
    for key in ["training_step", "iteration", "advantage_buffer_size", "value_buffer_size",
                 "snapshot_count", "_snapshot_count"]:
        if key in ckpt:
            print(f"  {key}: {ckpt[key]}")

    # dcfr_config
    if "dcfr_config" in ckpt:
        cfg = ckpt["dcfr_config"]
        print(f"\n  dcfr_config keys: {sorted(cfg.keys()) if isinstance(cfg, dict) else dir(cfg)}")
        if isinstance(cfg, dict):
            for k in ["network_type", "encoding_layout", "input_dim", "traversal_method",
                      "sd_cfr", "escher_mode", "use_value_net", "sampling_method",
                      "exploration_epsilon", "value_hidden", "value_lr", "value_buffer_size",
                      "value_target_buffer_passes", "target_buffer_passes"]:
                if k in cfg:
                    print(f"    {k}: {cfg[k]}")

    # Advantage network
    if "advantage_net_state_dict" in ckpt:
        sd = ckpt["advantage_net_state_dict"]
        print(f"\n  Advantage net: {len(sd)} layers")
        print_weight_stats(sd, "adv_net")

    # Value network
    if "value_net_state_dict" in ckpt:
        sd = ckpt["value_net_state_dict"]
        print(f"\n  Value net: {len(sd)} layers")
        print_weight_stats(sd, "val_net")
    else:
        print("\n  NO value_net_state_dict found")

    # SD-CFR snapshots
    for snap_key in ["sd_snapshots", "sd_cfr_snapshots"]:
        if snap_key in ckpt:
            snaps = ckpt[snap_key]
            print(f"\n  {snap_key}: type={type(snaps)}, len={len(snaps) if hasattr(snaps, '__len__') else 'N/A'}")

    snap_path = path.replace(".pt", "_sd_snapshots.pt")
    if os.path.exists(snap_path):
        snaps = torch.load(snap_path, map_location="cpu", weights_only=False)
        print(f"\n  External SD snapshots file: type={type(snaps)}")
        if isinstance(snaps, list):
            print(f"    Count: {len(snaps)}")
            if snaps:
                first = snaps[0]
                print(f"    First entry type: {type(first)}")
                if isinstance(first, (tuple, list)):
                    print(f"    First entry len: {len(first)}, types: {[type(x).__name__ for x in first]}")
                    if len(first) >= 2:
                        weight = first[0] if isinstance(first[0], (int, float)) else first[-1]
                        print(f"    Sample weight: {first[0]} (type={type(first[0]).__name__})")
                # Show weights of all snapshots
                if len(snaps) > 1 and isinstance(snaps[0], (tuple, list)):
                    weights = [s[0] for s in snaps if isinstance(s[0], (int, float))]
                    if weights:
                        print(f"    Weights: min={min(weights):.4f} max={max(weights):.4f} mean={np.mean(weights):.4f}")

    # EMA
    ema_path = path.replace(".pt", "_ema.pt")
    if os.path.exists(ema_path):
        ema = torch.load(ema_path, map_location="cpu", weights_only=False)
        print(f"\n  EMA checkpoint keys: {sorted(ema.keys()) if isinstance(ema, dict) else type(ema)}")
        if isinstance(ema, dict) and "ema_net_state_dict" in ema:
            sd = ema["ema_net_state_dict"]
            print(f"  EMA net layers: {len(sd)}")
            print_weight_stats(sd, "ema_net")

    return ckpt

def compare_predictions(ckpt_escher, ckpt_phase2):
    print(f"\n{'='*60}")
    print("FORWARD PASS COMPARISON")
    print(f"{'='*60}")

    from networks import build_advantage_network

    torch.manual_seed(42)
    x = torch.randn(1, EP_PBS_INPUT_DIM)
    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    # Random legal actions: ~20 of 146
    legal_indices = torch.randperm(NUM_ACTIONS)[:20]
    mask[0, legal_indices] = True

    for name, ckpt in [("ESCHER", ckpt_escher), ("Phase2", ckpt_phase2)]:
        cfg = ckpt.get("dcfr_config", {})
        network_type = cfg.get("network_type", "residual") if isinstance(cfg, dict) else "residual"
        input_dim = cfg.get("input_dim", EP_PBS_INPUT_DIM) if isinstance(cfg, dict) else EP_PBS_INPUT_DIM

        net = build_advantage_network(
            input_dim=input_dim,
            num_actions=NUM_ACTIONS,
            use_residual=(network_type == "residual"),
        )
        if "advantage_net_state_dict" in ckpt:
            net.load_state_dict(ckpt["advantage_net_state_dict"])
        net.eval()

        with torch.no_grad():
            # Use input_dim for the network
            x_in = x[:, :input_dim]
            logits = net(x_in)
            legal_logits = logits[mask].cpu().numpy()

            # QRE softmax over legal actions
            adv = logits.clone()
            adv[~mask] = float('-inf')
            probs = torch.softmax(adv, dim=-1)
            legal_probs = probs[mask].cpu().numpy()

        print(f"\n  {name} (network_type={network_type}, input_dim={input_dim}):")
        print(f"    Raw logits (legal): min={legal_logits.min():.4f} max={legal_logits.max():.4f} mean={legal_logits.mean():.4f} std={legal_logits.std():.4f}")
        print(f"    Prob (legal): min={legal_probs.min():.4f} max={legal_probs.max():.4f} max_action_prob={legal_probs.max():.4f}")
        print(f"    Top action prob: {legal_probs.max():.4f}, entropy: {-(legal_probs * np.log(legal_probs + 1e-10)).sum():.4f}")

        # Check Cambia action (action 145 is typically the cambia call)
        cambia_action = 145
        cambia_logit = logits[0, cambia_action].item()
        print(f"    Cambia action (idx=145) logit: {cambia_logit:.4f}")

def main():
    ckpt_escher = analyze_checkpoint(ESCHER_PATH, "ESCHER (escher-interleaved)")
    ckpt_phase2 = analyze_checkpoint(PHASE2_PATH, "Phase 2 (interleaved-resnet-adaptive)")
    compare_predictions(ckpt_escher, ckpt_phase2)

    # Check value buffer
    val_buf_path = ESCHER_PATH.replace(".pt", "_value_buffer.npz")
    if os.path.exists(val_buf_path):
        data = np.load(val_buf_path)
        print(f"\n{'='*60}")
        print("ESCHER VALUE BUFFER")
        print(f"{'='*60}")
        print(f"  Keys: {list(data.keys())}")
        for k in data.keys():
            arr = data[k]
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}", end="")
            if arr.dtype.kind == 'f':
                print(f" min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
            else:
                print()

if __name__ == "__main__":
    main()
