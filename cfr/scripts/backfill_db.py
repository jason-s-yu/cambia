#!/usr/bin/env python3
"""
scripts/backfill_db.py

One-shot migration: scan existing run directories, populate the SQLite run DB.

Reads config.yaml, checkpoints/*.pt, metrics.jsonl, and head_to_head.jsonl
from each run directory and populates the DB idempotently.

Usage:
    python scripts/backfill_db.py
    python scripts/backfill_db.py --runs-dir cfr/runs --db-path cfr/runs/cambia_runs.db
    python scripts/backfill_db.py --runs-dir cfr/runs --include-archive
"""

import argparse
import json
import logging
import sys
from glob import glob
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

from src.run_db import (
    get_db,
    upsert_run,
    register_checkpoint,
    compute_retention_flags,
    insert_eval_result,
    insert_head_to_head,
    write_run_meta_json,
    write_eval_summary_jsonl,
    infer_algorithm,
    update_best_metric,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_yaml_as_dict(yaml_path: str):
    """Load a YAML file, returning (text, dict). Uses PyYAML if available, else None."""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
        return text, yaml.safe_load(text) or {}
    except ImportError:
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
        return text, {}
    except Exception:
        return None, {}


def _parse_iter(path: str) -> int:
    """Extract iteration number from checkpoint filename."""
    import re
    m = re.search(r"_iter_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def backfill_run(db, run_dir: Path) -> None:
    """Backfill a single run directory into the DB."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        logger.warning("No config.yaml in %s, skipping.", run_dir)
        return

    run_name = run_dir.name
    logger.info("Backfilling run: %s", run_name)

    yaml_text, config_dict = _load_yaml_as_dict(str(config_path))
    algorithm = infer_algorithm(config_dict)

    run_id = upsert_run(
        db,
        name=run_name,
        algorithm=algorithm,
        config_yaml=yaml_text,
        config_dict=config_dict,
        status="completed",
    )
    logger.info("  run_id=%d, algorithm=%s", run_id, algorithm)

    # Register checkpoints
    ckpt_files = sorted(glob(str(run_dir / "checkpoints" / "deep_cfr_checkpoint_iter_*.pt")))
    ckpt_id_map = {}  # iter -> ckpt_id
    for ckpt_file in ckpt_files:
        it = _parse_iter(ckpt_file)
        if it < 0:
            continue
        ckpt_id = register_checkpoint(db, run_id, it, ckpt_file)
        ckpt_id_map[it] = ckpt_id

    if ckpt_id_map:
        compute_retention_flags(db, run_id)
        logger.info("  registered %d checkpoints", len(ckpt_id_map))

    # Parse metrics.jsonl → eval_results
    metrics_path = run_dir / "metrics.jsonl"
    eval_count = 0
    best_mean_imp = None
    best_mean_imp_iter = None

    if metrics_path.exists():
        # Group by iter to compute mean_imp
        from collections import defaultdict
        iter_rows = defaultdict(dict)
        with open(metrics_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                it = row.get("iter", -1)
                baseline = row.get("baseline", "")
                ckpt_id = ckpt_id_map.get(it)
                insert_eval_result(db, run_id, ckpt_id, row)
                eval_count += 1
                iter_rows[it][baseline] = row.get("win_rate", 0.0)

        # Compute mean_imp and find best
        MEAN_IMP = [
            "random_no_cambia", "random_late_cambia", "imperfect_greedy",
            "memory_heuristic", "aggressive_snap",
        ]
        for it, baselines in iter_rows.items():
            wrs = [baselines[bl] for bl in MEAN_IMP if bl in baselines]
            if wrs:
                mi = sum(wrs) / len(wrs)
                if best_mean_imp is None or mi > best_mean_imp:
                    best_mean_imp = mi
                    best_mean_imp_iter = it

        if best_mean_imp is not None and best_mean_imp_iter is not None:
            update_best_metric(db, run_id, "mean_imp", best_mean_imp, best_mean_imp_iter)
            if best_mean_imp_iter in ckpt_id_map:
                from src.run_db import mark_best_checkpoint
                mark_best_checkpoint(db, run_id, ckpt_id_map[best_mean_imp_iter])

        logger.info("  inserted %d eval results (best mean_imp=%.4f at iter %s)",
                    eval_count, best_mean_imp or 0.0, best_mean_imp_iter)

    # Parse head_to_head.jsonl
    h2h_path = run_dir / "head_to_head.jsonl"
    h2h_count = 0
    if h2h_path.exists():
        with open(h2h_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                insert_head_to_head(db, run_id, row)
                h2h_count += 1
        logger.info("  inserted %d H2H rows", h2h_count)

    # Write run_meta.json and eval_summary.jsonl
    try:
        write_run_meta_json(db, run_id, str(run_dir))
        write_eval_summary_jsonl(db, run_id, str(run_dir))
        logger.info("  wrote run_meta.json and eval_summary.jsonl")
    except Exception as e:
        logger.warning("  failed to write JSON outputs: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill SQLite run database from existing run directories."
    )
    parser.add_argument(
        "--runs-dir",
        default=str(_CFR_ROOT / "runs"),
        help="Root runs directory (default: cfr/runs)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite DB (default: cfr/runs/cambia_runs.db)",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also scan cfr/runs/_archive/",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        logger.error("Runs directory not found: %s", runs_root)
        sys.exit(1)

    db = get_db(args.db_path)
    logger.info("Opened DB at %s", args.db_path or "default path")

    # Collect run dirs (skip _archive itself and hidden dirs)
    run_dirs = [
        d for d in sorted(runs_root.iterdir())
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    if args.include_archive:
        archive = runs_root / "_archive"
        if archive.exists():
            run_dirs += [
                d for d in sorted(archive.iterdir())
                if d.is_dir() and not d.name.startswith(".")
            ]

    logger.info("Found %d run directories to backfill.", len(run_dirs))

    success = 0
    failed = 0
    for run_dir in run_dirs:
        try:
            backfill_run(db, run_dir)
            success += 1
        except Exception:
            logger.exception("Failed to backfill %s", run_dir)
            failed += 1

    db.close()
    logger.info("Backfill complete: %d succeeded, %d failed.", success, failed)


if __name__ == "__main__":
    main()
