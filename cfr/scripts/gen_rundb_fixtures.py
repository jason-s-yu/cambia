#!/usr/bin/env python3
"""
scripts/gen_rundb_fixtures.py

Builds the shared run_db.sqlite fixture corpus consumed by both the Go
journal validator (runnerd/nashnet/quarantine/rundbcheck.go, D55) and the
Python reconciler corpus test (cfr/tests/test_harness_rundb_corpus.py, D61).

Every fixture is built through src.run_db's real schema and helpers (get_db,
upsert_run, register_checkpoint, insert_eval_result) rather than hand-rolled
SQL, so the corpus tracks the schema instead of a copy of it that can drift.

The oversized fixture is not produced here: the D55 size cap is a byte check
the validator runs before ever opening the file, so both suites synthesize a
sparse truncated file at test time instead of committing a large blob. It is
still listed in manifest.json (file: null) for documentation.

The helpers' clock and engine-commit stamps are pinned while the corpus is
built, so a rerun against an unchanged schema reproduces every file byte for
byte (with the same SQLite library) and a regenerated corpus differs from the
checked-in one only where the schema moved. Rerun this script whenever
src/run_db.py's _DDL or _COLUMN_MIGRATIONS changes: the Go suite's
TestCorpusMatchesRunDBSchema fails until the accept fixtures carry the
current columns (cambia-2358).

Usage:
    python scripts/gen_rundb_fixtures.py
    python scripts/gen_rundb_fixtures.py --out runnerd/harness/testdata/rundb
"""

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

import src.run_db as _run_db  # noqa: E402
from src.run_db import (
    get_db,
    insert_eval_result,
    register_checkpoint,
    upsert_run,
)  # noqa: E402

_REPO_ROOT = _CFR_ROOT.parent
_DEFAULT_OUT = _REPO_ROOT / "runnerd" / "harness" / "testdata" / "rundb"

# Kept in sync by hand with the reconciler's enum (src/harness/reconciler.py
# _ALLOWED_STATUS) and the Go validator's mirror (rundbcheck.go
# allowedStatus). Any drift between the three is exactly what this corpus
# exists to catch (D55/D61): a fixture would start disagreeing between suites.
_VALID_STATUS = "completed"
_INVALID_STATUS = "not_a_real_status"

_FIXTURE_COMMIT = "4af0f825c576d6d86feb22c9e08e892d877168b4"
_FIXTURE_NOW = "2026-09-01T00:00:00Z"


@contextmanager
def _pinned_stamps() -> Iterator[None]:
    """Pin run_db's wall clock and its `git rev-parse` commit stamp, which
    upsert_run and register_checkpoint read at call time, for the duration of
    a build."""
    saved = (_run_db._now, _run_db._get_engine_commit)
    _run_db._now = lambda: _FIXTURE_NOW
    _run_db._get_engine_commit = lambda: _FIXTURE_COMMIT
    try:
        yield
    finally:
        _run_db._now, _run_db._get_engine_commit = saved


def _fresh(path: Path) -> sqlite3.Connection:
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    return get_db(str(path))


def _seal(conn: sqlite3.Connection, path: Path) -> None:
    """Fold the WAL into the main file and close, so exactly one file crosses
    the wire (D22: a node checkpoints its own journal before hashing it)."""
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    conn.close()
    for suffix in ("-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


def _build_valid_train(out_dir: Path) -> dict:
    name = "job-train-0001"
    path = out_dir / "valid_train.sqlite"
    conn = _fresh(path)
    run_id = upsert_run(
        conn,
        name=name,
        algorithm="prt-cfr",
        status=_VALID_STATUS,
        engine_commit_hash=_FIXTURE_COMMIT,
        tags=["v0.4", "pool"],
        notes="fixture: valid train journal",
    )
    ckpt_id = register_checkpoint(
        conn, run_id, 10, "prtcfr_checkpoint_iter_10.pt", file_size_bytes=1024
    )
    insert_eval_result(
        conn,
        run_id,
        ckpt_id,
        {
            "iteration": 10,
            "baseline": "random_no_cambia",
            "win_rate": 0.55,
            "games_played": 500,
            "p0_wins": 275,
            "p1_wins": 220,
            "ties": 5,
            "timestamp": "2026-09-01T00:00:00Z",
        },
    )
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": name,
        "verdict": "accept",
        "reason": "",
    }


def _build_valid_evaluate(out_dir: Path) -> dict:
    # An evaluate job's journal is named for spec.target, not the job id
    # (D64): the coordinator passes the resolved target name as expectedName.
    target_name = "v0.4-x2r-target-run"
    path = out_dir / "valid_evaluate.sqlite"
    conn = _fresh(path)
    run_id = upsert_run(
        conn,
        name=target_name,
        algorithm="prt-cfr",
        status=_VALID_STATUS,
        engine_commit_hash=_FIXTURE_COMMIT,
        notes="fixture: valid evaluate journal, named for spec.target",
    )
    # The eval-hygiene provenance a current `cambia evaluate` row carries
    # (evaluate_agents.persist_eval_results), so the fixture exercises the
    # migrated columns with values rather than only with NULLs.
    insert_eval_result(
        conn,
        run_id,
        None,
        {
            "iteration": 20,
            "baseline": "imperfect_greedy",
            "win_rate": 0.61,
            "ci_low": 0.5965,
            "ci_high": 0.6233,
            "games_played": 5000,
            "p0_wins": 3050,
            "p1_wins": 1900,
            "ties": 50,
            "seat_balanced": 1,
            "selection_mode": "stochastic",
            "seat_scheme": "alternated",
            "crn_seed": 20260901,
            "run_seed": 18446744073709551557,
            "engine_errors": 0,
            "served_policy": "average_strategy",
            "timestamp": "2026-09-01T00:05:00Z",
        },
    )
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "evaluate",
        "expected_name": target_name,
        "verdict": "accept",
        "reason": "",
    }


def _build_corrupt(out_dir: Path) -> dict:
    name = "job-train-0001"
    path = out_dir / "corrupt.sqlite"
    conn = _fresh(path)
    upsert_run(conn, name=name, algorithm="prt-cfr", status=_VALID_STATUS)
    _seal(conn, path)

    # Overwrite the 16-byte "SQLite format 3\0" magic header. This is a
    # structural corruption (SQLITE_NOTADB), not a content mutation deep in a
    # page: flipping arbitrary payload bytes elsewhere in the file is not
    # reliably caught by PRAGMA integrity_check, since it validates b-tree
    # structure, not that TEXT/BLOB payload bytes are "sane". The magic-header
    # corruption fails the same way at the first query either way (open
    # succeeding lazily but the first real access reporting "not a database"),
    # which is what the validator's integrity_check step is guarding against.
    raw = bytearray(path.read_bytes())
    if len(raw) < 16:
        raise RuntimeError(f"corrupt fixture is only {len(raw)} bytes, too small")
    raw[0:16] = b"NOT A SQLITE DB!"
    path.write_bytes(bytes(raw))
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": name,
        "verdict": "reject",
        "reason": "integrity_check",
    }


def _build_extra_table(out_dir: Path) -> dict:
    name = "job-train-0001"
    path = out_dir / "extra_table.sqlite"
    conn = _fresh(path)
    upsert_run(conn, name=name, algorithm="prt-cfr", status=_VALID_STATUS)
    conn.execute("CREATE TABLE evil (id INTEGER PRIMARY KEY, payload TEXT)")
    conn.execute("INSERT INTO evil (payload) VALUES ('not part of the schema')")
    conn.commit()
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": name,
        "verdict": "reject",
        "reason": "schema",
    }


def _build_second_row(out_dir: Path) -> dict:
    name = "job-train-0001"
    path = out_dir / "second_row.sqlite"
    conn = _fresh(path)
    upsert_run(conn, name=name, algorithm="prt-cfr", status=_VALID_STATUS)
    upsert_run(conn, name="some-other-run-999", algorithm="prt-cfr", status=_VALID_STATUS)
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": name,
        "verdict": "reject",
        "reason": "identity",
    }


def _build_wrong_name(out_dir: Path) -> dict:
    expected_name = "job-train-0001"
    path = out_dir / "wrong_name.sqlite"
    conn = _fresh(path)
    upsert_run(
        conn, name="a-completely-different-run", algorithm="prt-cfr", status=_VALID_STATUS
    )
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": expected_name,
        "verdict": "reject",
        "reason": "identity",
    }


def _build_out_of_enum(out_dir: Path) -> dict:
    name = "job-train-0001"
    path = out_dir / "out_of_enum.sqlite"
    conn = _fresh(path)
    upsert_run(conn, name=name, algorithm="prt-cfr", status=_VALID_STATUS)
    # upsert_run only ever writes real lifecycle values; force an
    # out-of-enum value directly, the way a hostile node's own writer could.
    conn.execute("UPDATE runs SET status=? WHERE name=?", (_INVALID_STATUS, name))
    conn.commit()
    _seal(conn, path)
    return {
        "file": path.name,
        "kind": "train",
        "expected_name": name,
        "verdict": "reject",
        "reason": "enum",
    }


def build_all(out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    with _pinned_stamps():
        fixtures = [
            _build_valid_train(out_dir),
            _build_valid_evaluate(out_dir),
            _build_corrupt(out_dir),
            _build_extra_table(out_dir),
            _build_second_row(out_dir),
            _build_wrong_name(out_dir),
            _build_out_of_enum(out_dir),
        ]
    fixtures.append(
        {
            "file": None,
            "kind": "train",
            "expected_name": "job-train-0001",
            "verdict": "reject",
            "reason": "size_cap",
            "note": (
                "not checked in: the size cap is checked before the file is "
                "ever opened, so both suites synthesize a sparse oversized "
                "file at test time instead of committing a large blob"
            ),
        }
    )
    manifest = {"fixtures": fixtures}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    fixtures = build_all(args.out)
    for f in fixtures:
        label: Optional[str] = f["file"] or "(generated at test time)"
        print(f"{label:28s} {f['verdict']:7s} {f['reason']}")


if __name__ == "__main__":
    main()
