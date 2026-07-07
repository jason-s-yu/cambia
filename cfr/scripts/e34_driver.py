"""
cfr/scripts/e34_driver.py

Driver for experiments E0 (mean_imp quick pass), E3 (sampled LBR exploitability),
and E4 (head-to-head DESCA vs PPO). Thin entry point over evaluate_agents +
sampled_lbr so the production `cambia evaluate` surface stays untouched.

Subcommands:
  mean-imp   : multi-baseline win-rate (mean_imp5) for a checkpoint agent.
  lbr        : sampled LBR exploitability for a checkpoint agent or a heuristic.
  h2h        : seat-alternated head-to-head between two typed agents.

All outputs are JSON on stdout (one object) plus a human-readable summary on stderr.
Run from cfr/ with the cfr pyenv:
  python3 scripts/e34_driver.py <subcommand> [args]
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from typing import Dict, List, Optional


def _wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()


def cmd_mean_imp(args):
    from src.evaluate_agents import run_evaluation_multi_baseline, MEAN_IMP_BASELINES

    baselines = list(MEAN_IMP_BASELINES)
    t0 = time.time()
    results_map: Dict[str, Counter] = run_evaluation_multi_baseline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        baselines=baselines,
        device=args.device,
        output_dir=None,
        use_argmax=args.argmax,
        agent_type=args.agent_type,
        max_workers=args.workers,
    )
    elapsed = time.time() - t0

    per_baseline = {}
    wrs = []
    for b in baselines:
        c = results_map.get(b, Counter())
        p0 = c.get("P0 Wins", 0)
        p1 = c.get("P1 Wins", 0)
        # Published convention folds MaxTurnTies (turn-cap games) into ties so the
        # denominator matches games_played (persist_eval_results, evaluate_agents.py:3029).
        ties = c.get("Ties", 0) + c.get("MaxTurnTies", 0)
        errs = c.get("Errors", 0)
        total = p0 + p1 + ties
        wr, lo, hi = _wilson_ci(p0, total)
        per_baseline[b] = {
            "win_rate": round(wr, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "p0_wins": p0,
            "p1_wins": p1,
            "ties": ties,
            "max_turn_ties": c.get("MaxTurnTies", 0),
            "terminal_ties": c.get("Ties", 0),
            "errors": errs,
            "games_scored": total,
        }
        wrs.append(wr)

    mean_imp5 = sum(wrs) / len(wrs) if wrs else 0.0
    strat = [
        per_baseline[b]["win_rate"]
        for b in ("imperfect_greedy", "memory_heuristic", "aggressive_snap")
        if b in per_baseline
    ]
    mean_imp3 = sum(strat) / len(strat) if strat else 0.0

    out = {
        "experiment": "E0-mean_imp",
        "agent_type": args.agent_type,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "games_per_baseline": args.games,
        "selection_mode": "argmax" if args.argmax else "sampling",
        "seat_mode": "P0-pinned (run_evaluation)",
        "mean_imp5": round(mean_imp5, 6),
        "mean_imp3": round(mean_imp3, 6),
        "per_baseline": per_baseline,
        "elapsed_sec": round(elapsed, 1),
    }
    _eprint(f"[E0] {args.agent_type} {args.checkpoint}")
    _eprint(
        f"[E0] mean_imp5={mean_imp5:.4f} mean_imp3={mean_imp3:.4f} "
        f"mode={out['selection_mode']} games/baseline={args.games} elapsed={elapsed:.1f}s"
    )
    for b in baselines:
        pb = per_baseline[b]
        _eprint(
            f"    {b:20s} wr={pb['win_rate']:.4f} "
            f"[{pb['ci_low']:.4f},{pb['ci_high']:.4f}] "
            f"ties={pb['ties']} err={pb['errors']}"
        )
    print(json.dumps(out))


def _build_agent(agent_type, config, checkpoint, device, argmax):
    from src.evaluate_agents import get_agent

    kwargs = {"device": device}
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    if agent_type in ("deep_cfr", "escher", "sd_cfr", "desca", "dense-escher", "prt_cfr"):
        kwargs["use_argmax"] = argmax
    return get_agent(agent_type, player_id=0, config=config, **kwargs)


def cmd_lbr(args):
    from src.config import load_config

    cfg = load_config(args.config)
    if cfg is None:
        _eprint("[E3] ERROR: config load failed")
        sys.exit(1)

    tier = (getattr(args, "tier", "A") or "A").strip().upper()
    if tier not in ("A", "B"):
        _eprint(f"[E3] ERROR: unknown --tier {tier!r}")
        sys.exit(1)

    agent = _build_agent(
        args.agent_type, cfg, args.checkpoint, args.device, args.argmax
    )

    t0 = time.time()
    if tier == "B":
        # Tier B: agent-policy continuation rollouts vs a strong opponent.
        from src.cfr.lbr import tier_b_lbr

        res = tier_b_lbr(
            agent,
            cfg,
            num_infosets=args.infosets,
            br_rollouts_per_infoset=args.rollouts,
            seed=args.seed,
        )
        rollout_opponent = f"{res.get('rollout_opponent', '?')} (tier B, agent-policy)"
        experiment = "E3-LBR-tierB"
    else:
        from src.cfr.sampled_lbr import sampled_lbr

        res = sampled_lbr(
            agent,
            cfg,
            num_infosets=args.infosets,
            br_rollouts_per_infoset=args.rollouts,
            seed=args.seed,
        )
        rollout_opponent = "RandomAgent (tier A)"
        experiment = "E3-LBR-tierA"
    elapsed = time.time() - t0
    out = {
        "experiment": experiment,
        "tier": tier,
        "agent_type": args.agent_type,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "infosets_requested": args.infosets,
        "infosets_sampled": res["num_infosets_sampled"],
        "rollouts_per_action": args.rollouts,
        "seed": args.seed,
        "selection_mode": "argmax" if args.argmax else "sampling",
        "exploitability": round(res["exploitability"], 6),
        "std_err": round(res["std_err"], 6),
        "ci95_half": round(1.96 * res["std_err"], 6),
        "rollout_opponent": rollout_opponent,
        "elapsed_sec": round(elapsed, 1),
    }
    _eprint(
        f"[E3] tier={tier} {args.agent_type} expl={out['exploitability']:.4f} "
        f"+/-{out['ci95_half']:.4f} (n={out['infosets_sampled']}, "
        f"seed={args.seed}, {elapsed:.1f}s)"
    )
    print(json.dumps(out))


def cmd_h2h(args):
    from src.config import load_config
    from src.evaluate_agents import run_head_to_head_typed

    cfg = load_config(args.config)
    if cfg is None:
        _eprint("[E4] ERROR: config load failed")
        sys.exit(1)

    t0 = time.time()
    res = run_head_to_head_typed(
        agent_a_type=args.agent_a,
        checkpoint_a=args.checkpoint_a,
        agent_b_type=args.agent_b,
        checkpoint_b=args.checkpoint_b,
        num_games=args.games,
        config=cfg,
        device=args.device,
        use_argmax_a=args.argmax_a,
        use_argmax_b=args.argmax_b,
    )
    elapsed = time.time() - t0

    wins_a = res["wins_a"]
    wins_b = res["wins_b"]
    draws = res["draws"]
    scored = wins_a + wins_b + draws
    wr_a, lo_a, hi_a = _wilson_ci(wins_a, scored)
    wr_b, lo_b, hi_b = _wilson_ci(wins_b, scored)
    out = {
        "experiment": "E4-head-to-head",
        "agent_a": args.agent_a,
        "checkpoint_a": args.checkpoint_a,
        "agent_b": args.agent_b,
        "checkpoint_b": args.checkpoint_b,
        "games": args.games,
        "selection_mode_a": "argmax" if args.argmax_a else "sampling",
        "selection_mode_b": ("argmax" if args.argmax_b else "sampling")
        + (" (PPO wrapper is argmax-only)" if args.agent_b == "ppo" else ""),
        "seat_mode": "alternated (run_head_to_head_typed)",
        "tie_semantics": "ties are non-wins for both sides (draws counted separately)",
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "errors": res.get("errors", 0),
        "win_rate_a": round(wr_a, 6),
        "ci_a": [round(lo_a, 6), round(hi_a, 6)],
        "win_rate_b": round(wr_b, 6),
        "ci_b": [round(lo_b, 6), round(hi_b, 6)],
        "avg_game_turns": round(res.get("avg_game_turns", 0.0), 2),
        "elapsed_sec": round(elapsed, 1),
    }
    _eprint(
        f"[E4] A={args.agent_a} vs B={args.agent_b}: "
        f"A {wins_a} ({wr_a:.4f}) | B {wins_b} ({wr_b:.4f}) | draws {draws} "
        f"({elapsed:.1f}s)"
    )
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description="E0/E3/E4 driver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mi = sub.add_parser("mean-imp")
    mi.add_argument("--agent-type", required=True)
    mi.add_argument("--checkpoint", required=True)
    mi.add_argument("--config", required=True)
    mi.add_argument("--games", type=int, default=1000)
    mi.add_argument("--device", default="cpu")
    mi.add_argument("--workers", type=int, default=5)
    mi.add_argument("--argmax", action="store_true")
    mi.set_defaults(func=cmd_mean_imp)

    lb = sub.add_parser("lbr")
    lb.add_argument("--agent-type", required=True)
    lb.add_argument("--checkpoint", default="")
    lb.add_argument("--config", required=True)
    lb.add_argument("--infosets", type=int, default=10000)
    lb.add_argument("--rollouts", type=int, default=100)
    lb.add_argument("--seed", type=int, default=42)
    lb.add_argument("--device", default="cpu")
    lb.add_argument("--argmax", action="store_true")
    lb.add_argument(
        "--tier",
        choices=["A", "B"],
        default="A",
        help="A: random rollouts (loose). B: agent-policy rollouts vs strong opp (tighter).",
    )
    lb.set_defaults(func=cmd_lbr)

    h = sub.add_parser("h2h")
    h.add_argument("--agent-a", required=True)
    h.add_argument("--checkpoint-a", default="")
    h.add_argument("--agent-b", required=True)
    h.add_argument("--checkpoint-b", default="")
    h.add_argument("--config", required=True)
    h.add_argument("--games", type=int, default=5000)
    h.add_argument("--device", default="cpu")
    h.add_argument("--argmax-a", action="store_true", help="Side A uses argmax (default sampling)")
    h.add_argument("--argmax-b", action="store_true", help="Side B uses argmax (default sampling)")
    h.set_defaults(func=cmd_h2h)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
