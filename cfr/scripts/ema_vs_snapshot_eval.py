"""
One-shot comparison: EMA path vs snapshot-averaging path for SD-CFR.
Uses the interleaved-resnet-adaptive final checkpoint (step 602).

Both modes use the same checkpoint. The only difference is use_ema flag in
SDCFRAgentWrapper:
- EMA: loads _ema.pt weights, single forward pass (theoretically incorrect Jensen's)
- Snapshot: loads all 200 snapshots, averages strategies per call (theoretically correct)

We monkey-patch get_agent to inject use_ema=False for snapshot mode.
"""
import sys
import os
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.evaluate_agents as ea

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "interleaved-resnet-adaptive")
CHECKPOINT = os.path.abspath(os.path.join(RUN_DIR, "checkpoints", "deep_cfr_checkpoint.pt"))
CONFIG_PATH = os.path.abspath(os.path.join(RUN_DIR, "config.yaml"))
BASELINES = ["imperfect_greedy", "memory_heuristic", "aggressive_snap"]
NUM_GAMES = 1000


def win_rate(results: Counter, p0_key: str = "P0 Wins") -> float:
    total = sum(results.values())
    return results.get(p0_key, 0) / total if total > 0 else 0.0


def evaluate_mode(use_ema: bool) -> tuple:
    label = "EMA" if use_ema else "Snapshot"
    print(f"\n=== {label} mode (use_ema={use_ema}) ===")

    # Monkey-patch get_agent so sd_cfr uses the desired use_ema value
    original_get_agent = ea.get_agent

    def patched_get_agent(agent_type, player_id, config, **kwargs):
        if agent_type.lower() == "sd_cfr":
            return ea.SDCFRAgentWrapper(
                player_id=player_id,
                config=config,
                checkpoint_path=kwargs["checkpoint_path"],
                device=kwargs.get("device", "cpu"),
                use_ema=use_ema,
                use_argmax=kwargs.get("use_argmax", False),
            )
        return original_get_agent(agent_type, player_id, config, **kwargs)

    ea.get_agent = patched_get_agent
    t0 = time.time()
    wr_list = []
    for baseline in BASELINES:
        results = ea.run_evaluation(
            config_path=CONFIG_PATH,
            agent1_type="sd_cfr",
            agent2_type=baseline,
            num_games=NUM_GAMES,
            strategy_path=None,
            checkpoint_path=CHECKPOINT,
            device="cpu",
        )
        wr = win_rate(results)
        wr_list.append(wr)
        wins = results.get("P0 Wins", 0)
        total = sum(results.values())
        print(f"  vs {baseline}: {wr:.1%}  ({wins}/{total}  ties={results.get('Ties',0)})")
    ea.get_agent = original_get_agent

    mean = sum(wr_list) / len(wr_list)
    elapsed = time.time() - t0
    print(f"  mean_imp(3): {mean:.1%}  [{elapsed:.0f}s]")
    return mean, wr_list


if __name__ == "__main__":
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Games per baseline: {NUM_GAMES}")

    ema_mean, ema_wrs = evaluate_mode(use_ema=True)
    snap_mean, snap_wrs = evaluate_mode(use_ema=False)

    print(f"\n{'='*50}")
    print(f"SUMMARY (mean_imp(3), {NUM_GAMES} games each)")
    print(f"  EMA:      {ema_mean:.1%}")
    print(f"  Snapshot: {snap_mean:.1%}")
    print(f"  Delta:    {snap_mean - ema_mean:+.1%}")
    for i, b in enumerate(BASELINES):
        print(f"  {b}: EMA={ema_wrs[i]:.1%} Snap={snap_wrs[i]:.1%} delta={snap_wrs[i]-ema_wrs[i]:+.1%}")
