#!/usr/bin/env python3
"""Aggregate per-game JSONL eval files into metrics.jsonl per run."""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

RUNS_BASE = Path(__file__).parent.parent / "runs"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records: list[dict], baseline: str) -> dict:
    total = len(records)
    p0_wins = sum(1 for r in records if r.get("winner") == "p0")
    p1_wins = sum(1 for r in records if r.get("winner") == "p1")
    ties = total - p0_wins - p1_wins

    turns = [r["turns"] for r in records if "turns" in r]
    avg_turns = sum(turns) / len(turns) if turns else None

    # score_margin: p0_score - p1_score if available
    margins = []
    for r in records:
        if "p0_score" in r and "p1_score" in r:
            margins.append(r["p0_score"] - r["p1_score"])
    avg_score_margin = sum(margins) / len(margins) if margins else None

    return {
        "baseline": baseline,
        "win_rate": round(p0_wins / total, 6) if total > 0 else None,
        "games_played": total,
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "ties": ties,
        "avg_game_turns": round(avg_turns, 3) if avg_turns is not None else None,
        "avg_score_margin": round(avg_score_margin, 3) if avg_score_margin is not None else None,
    }


def load_losses(checkpoint_path: Path) -> tuple[float | None, float | None]:
    if not checkpoint_path.exists():
        return None, None
    try:
        ck = torch.load(checkpoint_path, weights_only=True)
        adv = ck.get("advantage_loss_history") or ck.get("adv_loss_history")
        strat = ck.get("strategy_loss_history") or ck.get("strat_loss_history")
        def extract_last(history):
            if not history:
                return None
            val = history[-1]
            # history entries may be (iter, loss) tuples or plain floats
            if isinstance(val, (list, tuple)):
                return float(val[-1])
            return float(val)

        adv_last = extract_last(adv)
        strat_last = extract_last(strat)
        return adv_last, strat_last
    except Exception as e:
        print(f"  Warning: could not load checkpoint {checkpoint_path}: {e}")
        return None, None


def find_checkpoint(run_dir: Path, iter_num: int) -> Path | None:
    """Find the best matching checkpoint for a given iteration number."""
    ck_dir = run_dir / "checkpoints"
    if not ck_dir.exists():
        return None

    # Explicit iter file
    candidates = [
        ck_dir / f"deep_cfr_checkpoint_iter_{iter_num}.pt",
        # os-full special case
        ck_dir / f"deep_cfr_checkpoint_baseline_test_iter_{iter_num}.pt",
    ]
    # Also check archive subdirs
    for subdir in run_dir.glob("archive_*/checkpoints"):
        candidates.append(subdir / f"deep_cfr_checkpoint_iter_{iter_num}.pt")

    for c in candidates:
        if c.exists():
            return c

    # Fall back to deep_cfr_checkpoint.pt if loss history covers iter_num
    latest = ck_dir / "deep_cfr_checkpoint.pt"
    if latest.exists():
        try:
            ck = torch.load(latest, weights_only=True)
            adv = ck.get("advantage_loss_history") or []
            # Check if the last entry covers iter_num
            if adv:
                last_entry = adv[-1]
                last_iter = last_entry[0] if isinstance(last_entry, (list, tuple)) else iter_num
                if last_iter >= iter_num:
                    return latest
        except Exception:
            pass

    return None


def parse_iter_dir(name: str) -> tuple[int, bool] | None:
    """Parse iter dir name -> (iter_num, is_1k). Returns None if not parseable."""
    m = re.match(r"iter_(\d+)(_1k)?$", name)
    if not m:
        return None
    return int(m.group(1)), m.group(2) is not None


def scan_run(run_dir: Path) -> list[dict]:
    rows = []
    run_name = run_dir.name
    eval_root = run_dir / "evaluations"
    if not eval_root.exists():
        return rows

    # Group iter dirs by iter number, prefer _1k
    iter_dirs: dict[int, Path] = {}
    for d in eval_root.iterdir():
        if not d.is_dir():
            continue
        parsed = parse_iter_dir(d.name)
        if parsed is None:
            continue
        iter_num, is_1k = parsed
        if iter_num not in iter_dirs:
            iter_dirs[iter_num] = d
        else:
            # Prefer _1k over non-1k
            existing_parsed = parse_iter_dir(iter_dirs[iter_num].name)
            if existing_parsed and not existing_parsed[1] and is_1k:
                iter_dirs[iter_num] = d

    timestamp = datetime.now(timezone.utc).isoformat()

    for iter_num, iter_dir in sorted(iter_dirs.items()):
        ck_path = find_checkpoint(run_dir, iter_num)
        adv_loss, strat_loss = load_losses(ck_path) if ck_path else (None, None)
        if ck_path is None:
            print(f"  [{run_name}/iter_{iter_num}] No checkpoint found")

        for jsonl_file in sorted(iter_dir.glob("*.jsonl")):
            baseline = jsonl_file.stem
            records = load_jsonl(jsonl_file)
            if not records:
                continue
            stats = compute_stats(records, baseline)
            row = {
                "run": run_name,
                "iter": iter_num,
                **stats,
                "adv_loss": adv_loss,
                "strat_loss": strat_loss,
                "timestamp": timestamp,
            }
            # Remove None values for cleanliness (keep explicit None for avg_score_margin if missing)
            rows.append(row)
            print(f"  [{run_name}/iter_{iter_num}/{baseline}] games={stats['games_played']} win_rate={stats['win_rate']}")

    return rows


def main():
    if not RUNS_BASE.exists():
        print(f"Runs directory not found: {RUNS_BASE}")
        return

    run_dirs = sorted(d for d in RUNS_BASE.iterdir() if d.is_dir() and (d / "evaluations").exists())
    print(f"Found {len(run_dirs)} run(s) with evaluations: {[d.name for d in run_dirs]}")

    all_rows_by_run: dict[str, list[dict]] = defaultdict(list)

    for run_dir in run_dirs:
        print(f"\nScanning {run_dir.name}...")
        rows = scan_run(run_dir)
        all_rows_by_run[run_dir.name].extend(rows)

    # Write metrics.jsonl per run
    for run_name, rows in all_rows_by_run.items():
        out_path = RUNS_BASE / run_name / "metrics.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary
    print("\n=== Summary ===")
    for run_name, rows in sorted(all_rows_by_run.items()):
        iters = sorted(set(r["iter"] for r in rows))
        baselines = sorted(set(r["baseline"] for r in rows))
        print(f"  {run_name}: {len(rows)} rows, iters={iters}, baselines={baselines}")


if __name__ == "__main__":
    main()
