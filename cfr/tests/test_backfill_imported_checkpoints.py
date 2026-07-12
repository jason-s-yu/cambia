"""Tests for scripts/backfill_imported_checkpoints.py.

Covers the pre-cambia-389 gap: warm-start continuations imported prior-run
snapshot files (via PRTCFRTinyTrainer._import_prior_snapshots) but never
registered them as run_db checkpoint rows, so a ledger-listed iteration
(resume_state.json's "snapshots" list) with a snapshot file on disk had no
corresponding checkpoints row. The harness pull include-set derives from
checkpoint rows, so those files were silently skipped.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_db import get_db, register_checkpoint, upsert_run
from scripts.backfill_imported_checkpoints import (
    RunNotFoundError,
    backfill_run,
    load_ledger_iters,
    snapshot_file_path,
)


def _make_run_dir(tmp_path: Path, name: str, ledger, present_iters) -> Path:
    """Build a fake run dir: snapshots/ with empty .pt files for present_iters,
    and a resume_state.json ledger listing `ledger` iterations (which may
    include iters with no file on disk, simulating a missing snapshot)."""
    run_dir = tmp_path / name
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    for it in present_iters:
        snapshot_file_path(run_dir, it).touch()
    resume_state = {
        "schema": 1,
        "iteration": max(ledger) if ledger else 0,
        "total_iterations": max(ledger) if ledger else 0,
        "snapshots": list(ledger),
        "controller": None,
    }
    (run_dir / "resume_state.json").write_text(json.dumps(resume_state))
    return run_dir


def test_load_ledger_iters_reads_snapshots_list(tmp_path):
    run_dir = _make_run_dir(tmp_path, "r1", ledger=[1, 2, 3], present_iters=[1, 2, 3])
    assert load_ledger_iters(run_dir) == [1, 2, 3]


def test_backfill_run_registers_gap_reports_missing_leaves_existing(tmp_path):
    # Ledger lists 1..6; files present for 1..5; 6 is ledger-listed but missing
    # on disk. checkpoints table already has rows for 3,4,5 (with a distinct
    # file_path, so we can prove they are untouched) but is missing 1,2 (the
    # cambia-389-shaped gap).
    run_dir = _make_run_dir(
        tmp_path, "myrun", ledger=[1, 2, 3, 4, 5, 6], present_iters=[1, 2, 3, 4, 5]
    )
    db_path = tmp_path / "cambia_runs.db"
    db = get_db(str(db_path))
    run_id = upsert_run(db, name="myrun", algorithm="prt-cfr")
    for it in (3, 4, 5):
        register_checkpoint(db, run_id, it, f"custom/path/iter_{it}.pt")

    summary = backfill_run(db, run_dir, "myrun")

    assert summary.ledger_count == 6
    assert summary.already_registered == 3
    assert summary.backfilled == 2
    assert sorted(summary.backfilled_iters) == [1, 2]
    assert summary.missing == [6]

    rows = {
        r["iteration"]: r["file_path"]
        for r in db.execute(
            "SELECT iteration, file_path FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchall()
    }
    # Newly backfilled rows point at the real snapshot files.
    assert rows[1] == str(snapshot_file_path(run_dir, 1))
    assert rows[2] == str(snapshot_file_path(run_dir, 2))
    # Pre-existing rows are untouched (still the distinct path set above).
    assert rows[3] == "custom/path/iter_3.pt"
    assert rows[4] == "custom/path/iter_4.pt"
    assert rows[5] == "custom/path/iter_5.pt"
    # The ledger-missing iteration was never registered.
    assert 6 not in rows


def test_backfill_run_dry_run_writes_nothing(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "myrun", ledger=[1, 2, 3, 4, 5, 6], present_iters=[1, 2, 3, 4, 5]
    )
    db_path = tmp_path / "cambia_runs.db"
    db = get_db(str(db_path))
    run_id = upsert_run(db, name="myrun", algorithm="prt-cfr")
    for it in (3, 4, 5):
        register_checkpoint(db, run_id, it, f"custom/path/iter_{it}.pt")

    summary = backfill_run(db, run_dir, "myrun", dry_run=True)

    assert summary.backfilled == 2
    assert sorted(summary.backfilled_iters) == [1, 2]
    assert summary.missing == [6]

    count = db.execute(
        "SELECT COUNT(*) AS n FROM checkpoints WHERE run_id=?", (run_id,)
    ).fetchone()["n"]
    assert count == 3  # only the pre-seeded 3,4,5 -- dry-run registered nothing


def test_backfill_run_idempotent_second_run_backfills_zero(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "myrun", ledger=[1, 2, 3, 4, 5, 6], present_iters=[1, 2, 3, 4, 5]
    )
    db_path = tmp_path / "cambia_runs.db"
    db = get_db(str(db_path))
    upsert_run(db, name="myrun", algorithm="prt-cfr")

    first = backfill_run(db, run_dir, "myrun")
    assert first.backfilled == 5  # nothing pre-registered this time: 1..5 all gap

    second = backfill_run(db, run_dir, "myrun")
    assert second.backfilled == 0
    assert second.already_registered == 5
    assert second.missing == [6]


def test_backfill_run_ignores_checkpoint_iters_outside_ledger(tmp_path):
    run_dir = _make_run_dir(tmp_path, "myrun", ledger=[1, 2], present_iters=[1, 2])
    db_path = tmp_path / "cambia_runs.db"
    db = get_db(str(db_path))
    run_id = upsert_run(db, name="myrun", algorithm="prt-cfr")
    # A checkpoint row for an iteration the ledger never mentions (e.g. a
    # natively-written checkpoint past the warm-start import horizon).
    register_checkpoint(db, run_id, 999, "native/iter_999.pt")

    summary = backfill_run(db, run_dir, "myrun")

    assert summary.ledger_count == 2
    assert summary.backfilled == 2
    row = db.execute(
        "SELECT file_path FROM checkpoints WHERE run_id=? AND iteration=999", (run_id,)
    ).fetchone()
    assert row["file_path"] == "native/iter_999.pt"


def test_backfill_run_missing_run_raises(tmp_path):
    run_dir = _make_run_dir(tmp_path, "myrun", ledger=[1], present_iters=[1])
    db_path = tmp_path / "cambia_runs.db"
    db = get_db(str(db_path))
    with pytest.raises(RunNotFoundError):
        backfill_run(db, run_dir, "nonexistent-run")


def test_backfill_run_missing_resume_state_raises(tmp_path):
    run_dir = tmp_path / "norun"
    run_dir.mkdir()
    db = get_db(str(tmp_path / "cambia_runs.db"))
    upsert_run(db, name="norun", algorithm="prt-cfr")
    with pytest.raises(FileNotFoundError):
        backfill_run(db, run_dir, "norun")


def test_cli_help_runs():
    script = Path(__file__).resolve().parent.parent / "scripts" / "backfill_imported_checkpoints.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(script.parent.parent),
    )
    assert result.returncode == 0
    assert "run_dirs" in result.stdout or "run dir" in result.stdout.lower()
