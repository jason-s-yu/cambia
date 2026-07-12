#!/usr/bin/env python3
"""
scripts/backfill_imported_checkpoints.py

Backfill run_db checkpoint rows for pre-cambia-389 imported warm-start
snapshots.

Root cause (cambia-389): PRTCFRTinyTrainer._import_prior_snapshots copies a
prior run's prtcfr_snapshot_iter_{i}.pt files into a warm-started
continuation's snapshot dir and lists them in resume_state.json's "snapshots"
ledger, but before cambia-389 never called register_checkpoint for them. The
harness pull include-set is derived from checkpoint rows, so an imported
snapshot with no checkpoints row was silently skipped by artifact sync, even
though the file is present on disk and the ledger knows about it.

This script reconciles the two: for each given run directory, it reads the
resume_state.json ledger, checks which ledger iterations already have a
checkpoints row, and registers a row for every ledger iteration whose
snapshot file exists on disk but has no row yet. Iterations absent from the
ledger are never touched, and ledger iterations whose file is missing from
disk are reported, not registered (there is nothing to point a row at).

Usage:
    python scripts/backfill_imported_checkpoints.py cfr/runs/v0.4-x2-ext-1000
    python scripts/backfill_imported_checkpoints.py cfr/runs/run-a cfr/runs/run-b
    python scripts/backfill_imported_checkpoints.py --dry-run cfr/runs/run-a
    python scripts/backfill_imported_checkpoints.py --db /path/to/run_db.sqlite \\
        --run-name v0.4-x2-ext-1000 cfr/runs/v0.4-x2-ext-1000
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

from src.run_db import get_db, register_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SNAPSHOT_NAME_FMT = "prtcfr_snapshot_iter_{iteration}.pt"


class RunNotFoundError(Exception):
    """Raised when a run_name has no matching row in the run_db."""


def snapshot_file_path(run_dir: Path, iteration: int) -> Path:
    """The on-disk path of a PRT-CFR per-iteration snapshot for `run_dir`."""
    return Path(run_dir) / "snapshots" / SNAPSHOT_NAME_FMT.format(iteration=iteration)


def load_ledger_iters(run_dir: Path) -> List[int]:
    """Read resume_state.json's "snapshots" ledger, sorted ascending.

    Raises FileNotFoundError if the run dir has no resume_state.json (nothing
    to backfill from -- this run never wrote the production resume format).
    """
    resume_state_path = Path(run_dir) / "resume_state.json"
    if not resume_state_path.exists():
        raise FileNotFoundError(f"no resume_state.json under {run_dir}")
    with open(resume_state_path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    return sorted(int(i) for i in state.get("snapshots", []))


@dataclass
class BackfillSummary:
    run_name: str
    ledger_count: int
    already_registered: int
    backfilled: int
    backfilled_iters: List[int] = field(default_factory=list)
    missing: List[int] = field(default_factory=list)

    def report(self, dry_run: bool = False) -> str:
        verb = "would backfill" if dry_run else "backfilled"
        return (
            f"[{self.run_name}] ledger={self.ledger_count} "
            f"already_registered={self.already_registered} "
            f"{verb}={self.backfilled} {self.backfilled_iters} "
            f"missing_file={len(self.missing)} {self.missing}"
        )


def backfill_run(db, run_dir: Path, run_name: str, dry_run: bool = False) -> BackfillSummary:
    """Backfill checkpoint rows for one run's imported-snapshot ledger gap.

    Never touches iterations absent from the ledger, and never re-registers
    (or otherwise modifies) an iteration that already has a checkpoints row.
    """
    run_dir = Path(run_dir)
    ledger_iters = load_ledger_iters(run_dir)

    row = db.execute("SELECT id FROM runs WHERE name=?", (run_name,)).fetchone()
    if row is None:
        raise RunNotFoundError(f"no run named {run_name!r} in the run_db")
    run_id = row["id"]

    existing_rows = db.execute(
        "SELECT iteration FROM checkpoints WHERE run_id=?", (run_id,)
    ).fetchall()
    existing_iters = {r["iteration"] for r in existing_rows}

    already_registered = 0
    backfilled_iters: List[int] = []
    missing: List[int] = []

    for it in ledger_iters:
        if it in existing_iters:
            already_registered += 1
            continue
        fp = snapshot_file_path(run_dir, it)
        if not fp.exists():
            missing.append(it)
            continue
        if not dry_run:
            register_checkpoint(db, run_id, it, str(fp))
        backfilled_iters.append(it)

    return BackfillSummary(
        run_name=run_name,
        ledger_count=len(ledger_iters),
        already_registered=already_registered,
        backfilled=len(backfilled_iters),
        backfilled_iters=backfilled_iters,
        missing=missing,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill run_db checkpoint rows for imported warm-start snapshots "
            "that predate cambia-389 (ledger-listed iterations with a snapshot "
            "file on disk but no checkpoints row)."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories containing resume_state.json and snapshots/.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Path to a run_db.sqlite to register into for every given run_dir "
            "(default: <run_dir>/run_db.sqlite if present, else the central "
            "cfr/runs/cambia_runs.db / CAMBIA_RUN_DB override)."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Override the run_db run name to look up (default: the run_dir's "
            "basename). Only valid with a single run_dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be registered without writing to the db.",
    )
    args = parser.parse_args(argv)

    if args.run_name is not None and len(args.run_dirs) != 1:
        parser.error("--run-name is only valid with a single run_dir")

    failures = 0
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        run_name = args.run_name or run_dir.name

        if args.db:
            db_path = args.db
        else:
            local_db = run_dir / "run_db.sqlite"
            db_path = str(local_db) if local_db.exists() else None

        try:
            db = get_db(db_path)
            summary = backfill_run(db, run_dir, run_name, dry_run=args.dry_run)
        except (FileNotFoundError, RunNotFoundError) as e:
            logger.error("[%s] skipped: %s", run_name, e)
            failures += 1
            continue

        # print, not logger.info: an imported module's logging config can set
        # the root level above INFO, which silently swallows the summary.
        print(summary.report(dry_run=args.dry_run))
        if summary.missing:
            logger.warning(
                "[%s] %d ledger-listed iteration(s) have no snapshot file on "
                "disk and were NOT registered: %s",
                run_name, len(summary.missing), summary.missing,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
