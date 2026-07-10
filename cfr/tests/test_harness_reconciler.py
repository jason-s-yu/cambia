"""
tests/test_harness_reconciler.py

Coverage for the serving-harness reconciler (cambia-256, design 4.2/4.3/5.7) and
the run_db origin_host migration + upsert_run engine_commit_hash/origin_host
params.

Sources are built as raw sqlite files (no WAL, DELETE journal) so the tests
control every cell verbatim, including adversarial values the run_db helpers
would otherwise sanitize.
"""

import sqlite3

import pytest

from src import run_db
from src.harness.reconciler import (
    ReconcilerCollisionError,
    ReconcilerError,
    ReconcilerValidationError,
    replay,
)

_NOW = "2026-07-09T00:00:00Z"
_RUNNER_SHA = "runner-sha-xyz"


# ---------------------------------------------------------------------------
# Raw source-db builders (a runner's run_db.sqlite journal)
# ---------------------------------------------------------------------------


def _new_source(path):
    # A full pull replaces the synced run_db.sqlite wholesale; model that by
    # removing any prior file (and WAL siblings) so a re-build starts clean.
    from pathlib import Path as _Path

    for suffix in ("", "-wal", "-shm"):
        p = _Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(run_db._DDL)  # modern schema incl. origin_host
    return conn


def _insert_run(
    conn,
    name="v0.4-prtcfr-r1",
    algorithm="prt-cfr",
    status="completed",
    engine_commit_hash=_RUNNER_SHA,
    tags='["alpha", "beta"]',
    notes="runner run",
    origin_host=None,
):
    cur = conn.execute(
        "INSERT INTO runs (name, algorithm, status, engine_commit_hash, tags, notes, "
        "origin_host, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, algorithm, status, engine_commit_hash, tags, notes, origin_host, _NOW, _NOW),
    )
    conn.commit()
    return cur.lastrowid


def _insert_ckpt(
    conn,
    run_id,
    iteration,
    file_path,
    file_size_bytes=123,
    is_best=0,
    is_retained=1,
    compressed=0,
):
    cur = conn.execute(
        "INSERT INTO checkpoints (run_id, iteration, file_path, file_size_bytes, "
        "created_at, is_best, is_retained, compressed) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, iteration, file_path, file_size_bytes, _NOW, is_best, is_retained, compressed),
    )
    conn.commit()
    return cur.lastrowid


def _insert_eval(conn, run_id, iteration, baseline, checkpoint_id=999, **over):
    vals = dict(
        run_id=run_id,
        checkpoint_id=checkpoint_id,  # runner surrogate id; reconciler must ignore it
        iteration=iteration,
        baseline=baseline,
        win_rate=0.5,
        ci_low=0.48,
        ci_high=0.52,
        games_played=5000,
        p0_wins=2500,
        p1_wins=2400,
        ties=100,
        avg_game_turns=40.0,
        t1_cambia_rate=0.1,
        avg_score_margin=-1.5,
        adv_loss=0.2,
        strat_loss=0.3,
        seat_balanced=1,
        selection_mode="best",
        crn_seed="seed-1",
        seat_scheme="rotate",
        timestamp=_NOW,
    )
    vals.update(over)
    cols = ",".join(vals.keys())
    ph = ",".join(["?"] * len(vals))
    conn.execute(
        f"INSERT INTO eval_results ({cols}) VALUES ({ph})", tuple(vals.values())
    )
    conn.commit()


def _build_full_run(
    run_dir, name="v0.4-prtcfr-r1", iterations=(10, 20), write_files=True
):
    """A realistic run: one run, 2 checkpoints (snapshot files present), evals for
    each iteration across two baselines. Returns the run_dir path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = run_dir / "snapshots"
    snap.mkdir(exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    try:
        rid = _insert_run(conn, name=name)
        for idx, it in enumerate(iterations):
            fname = f"prtcfr_snapshot_iter_{it}.pt"
            if write_files:
                (snap / fname).write_bytes(b"x" * (10 * (idx + 1)))
            is_best = 1 if idx == len(iterations) - 1 else 0
            _insert_ckpt(
                conn, rid, it, str(snap / fname), is_best=is_best, is_retained=1
            )
            for bl in ("random_no_cambia", "memory_heuristic"):
                _insert_eval(conn, rid, it, bl)
    finally:
        conn.close()
    return run_dir


def _dest_path(tmp_path):
    return str(tmp_path / "dest" / "cambia_runs.db")


def _open_dest(path):
    return run_db.get_db(path)


# ---------------------------------------------------------------------------
# run_db schema / upsert_run param changes
# ---------------------------------------------------------------------------


def test_origin_host_in_fresh_ddl(tmp_path):
    db = _open_dest(str(tmp_path / "fresh.db"))
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(runs)").fetchall()}
        assert "origin_host" in cols
    finally:
        db.close()


def test_origin_host_additive_migration(tmp_path):
    """A db created before origin_host existed gains the column via _migrate_schema
    without losing rows."""
    path = str(tmp_path / "old.db")
    legacy_ddl = """
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        algorithm TEXT,
        status TEXT NOT NULL DEFAULT 'created',
        engine_commit_hash TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """
    old = sqlite3.connect(path)
    old.executescript(legacy_ddl)
    old.execute(
        "INSERT INTO runs (name, status, created_at, updated_at) VALUES (?,?,?,?)",
        ("legacy-run", "completed", _NOW, _NOW),
    )
    old.commit()
    old.close()

    db = _open_dest(path)
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(runs)").fetchall()}
        assert "origin_host" in cols
        row = db.execute(
            "SELECT origin_host FROM runs WHERE name='legacy-run'"
        ).fetchone()
        assert row is not None
        assert row["origin_host"] is None  # legacy row reads back NULL = local
    finally:
        db.close()


def test_upsert_run_engine_commit_default_stamps_local(tmp_path):
    db = _open_dest(str(tmp_path / "u.db"))
    try:
        run_db.upsert_run(db, name="r1", algorithm="prt-cfr")
        row = db.execute(
            "SELECT engine_commit_hash, origin_host FROM runs WHERE name='r1'"
        ).fetchone()
        assert row["engine_commit_hash"] == run_db._get_engine_commit()
        assert row["origin_host"] is None
    finally:
        db.close()


def test_upsert_run_engine_commit_verbatim_and_origin_host(tmp_path):
    db = _open_dest(str(tmp_path / "u.db"))
    try:
        run_db.upsert_run(
            db,
            name="r2",
            algorithm="prt-cfr",
            engine_commit_hash="deadbeef",
            origin_host="runner",
        )
        row = db.execute(
            "SELECT engine_commit_hash, origin_host FROM runs WHERE name='r2'"
        ).fetchone()
        assert row["engine_commit_hash"] == "deadbeef"
        assert row["origin_host"] == "runner"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# mark_run_pushed: ownership transfer on push (cambia-338, design 4.3)
# ---------------------------------------------------------------------------


def test_mark_run_pushed_transfers_ownership_preserving_provenance(tmp_path):
    db = _open_dest(str(tmp_path / "m.db"))
    try:
        run_db.upsert_run(
            db,
            name="pushme",
            algorithm="prt-cfr",
            engine_commit_hash="cafef00d",
        )
        before = db.execute(
            "SELECT origin_host, algorithm, engine_commit_hash FROM runs "
            "WHERE name='pushme'"
        ).fetchone()
        assert before["origin_host"] is None  # local run

        n = run_db.mark_run_pushed(db, "pushme", "runner")
        assert n == 1

        after = db.execute(
            "SELECT origin_host, algorithm, engine_commit_hash FROM runs "
            "WHERE name='pushme'"
        ).fetchone()
        assert after["origin_host"] == "runner"  # ownership transferred
        # A targeted UPDATE, not upsert_run: algorithm + engine_commit_hash are
        # preserved verbatim (upsert_run would re-stamp both).
        assert after["algorithm"] == "prt-cfr"
        assert after["engine_commit_hash"] == "cafef00d"
    finally:
        db.close()


def test_mark_run_pushed_idempotent_and_noop_when_absent(tmp_path):
    db = _open_dest(str(tmp_path / "m.db"))
    try:
        # No such run row: a no-op returning 0.
        assert run_db.mark_run_pushed(db, "ghost", "runner") == 0

        run_db.upsert_run(db, name="pushme", algorithm="prt-cfr")
        assert run_db.mark_run_pushed(db, "pushme", "runner") == 1
        # Re-marking the same host stays idempotent (a re-push).
        assert run_db.mark_run_pushed(db, "pushme", "runner") == 1
        row = db.execute(
            "SELECT origin_host FROM runs WHERE name='pushme'"
        ).fetchone()
        assert row["origin_host"] == "runner"
    finally:
        db.close()


def test_pushed_row_then_pull_reconciles_but_unpushed_refused(tmp_path):
    # The round trip: a local run marked pushed reconciles a same-name pull from
    # that host; a never-pushed local run of the same name still refuses.
    dest_path = _dest_path(tmp_path)
    dest = _open_dest(dest_path)
    try:
        run_db.upsert_run(dest, name="v0.4-prtcfr-r1", algorithm="prt-cfr")

        run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
        # Never pushed (origin_host NULL) -> still a hard collision.
        with pytest.raises(ReconcilerCollisionError):
            replay(run_dir, dest, origin_host="runner")

        # Push transfers ownership; the same pull now reconciles.
        run_db.mark_run_pushed(dest, "v0.4-prtcfr-r1", "runner")
        summary = replay(run_dir, dest, origin_host="runner")
        assert summary["runs"] == 1
        row = dest.execute(
            "SELECT origin_host FROM runs WHERE name='v0.4-prtcfr-r1'"
        ).fetchone()
        assert row["origin_host"] == "runner"
    finally:
        dest.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_replay_roundtrip(tmp_path):
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    dest_path = _dest_path(tmp_path)

    summary = replay(run_dir, dest_path, origin_host="runner")
    assert summary == {"runs": 1, "checkpoints": 2, "evals": 4}

    db = _open_dest(dest_path)
    try:
        run = db.execute("SELECT * FROM runs WHERE name='v0.4-prtcfr-r1'").fetchone()
        assert run is not None
        assert run["origin_host"] == "runner"
        assert run["engine_commit_hash"] == _RUNNER_SHA  # preserved, not re-stamped
        assert run["algorithm"] == "prt-cfr"
        assert run["status"] == "completed"
        assert run["notes"] == "runner run"

        run_id = run["id"]
        ckpts = db.execute(
            "SELECT iteration, file_path, is_best, is_retained FROM checkpoints "
            "WHERE run_id=? ORDER BY iteration",
            (run_id,),
        ).fetchall()
        assert [c["iteration"] for c in ckpts] == [10, 20]
        # is_best preserved (last checkpoint marked best in the source)
        assert [c["is_best"] for c in ckpts] == [0, 1]
        assert all(c["is_retained"] == 1 for c in ckpts)
        # file_path re-derived under the local synced run dir
        for c in ckpts:
            assert c["file_path"].startswith(str(run_dir.resolve()))

        evals = db.execute(
            "SELECT iteration, baseline, checkpoint_id, win_rate FROM eval_results "
            "WHERE run_id=? ORDER BY iteration, baseline",
            (run_id,),
        ).fetchall()
        assert len(evals) == 4
        # checkpoint_id re-resolved LOCALLY (never the runner surrogate 999)
        iter_to_ckpt = {
            c["iteration"]: db.execute(
                "SELECT id FROM checkpoints WHERE run_id=? AND iteration=?",
                (run_id, c["iteration"]),
            ).fetchone()["id"]
            for c in ckpts
        }
        for e in evals:
            assert e["checkpoint_id"] == iter_to_ckpt[e["iteration"]]
            assert e["checkpoint_id"] != 999
    finally:
        db.close()


def test_idempotent_rereplay(tmp_path):
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    dest_path = _dest_path(tmp_path)

    replay(run_dir, dest_path, origin_host="runner")
    summary2 = replay(run_dir, dest_path, origin_host="runner")
    assert summary2 == {"runs": 1, "checkpoints": 2, "evals": 4}

    db = _open_dest(dest_path)
    try:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 4
    finally:
        db.close()


def test_partial_pull_convergence(tmp_path):
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    dest_path = _dest_path(tmp_path)

    # First: a truncated pull (only iteration 10, one eval baseline).
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_ckpt(conn, rid, 10, str(run_dir / "snapshots" / "prtcfr_snapshot_iter_10.pt"))
    _insert_eval(conn, rid, 10, "random_no_cambia")
    conn.close()

    s1 = replay(run_dir, dest_path, origin_host="runner")
    assert s1 == {"runs": 1, "checkpoints": 1, "evals": 1}

    # Then: the full pull overwrites the synced db with more rows.
    _build_full_run(run_dir)  # iterations (10, 20) x 2 baselines
    s2 = replay(run_dir, dest_path, origin_host="runner")
    assert s2 == {"runs": 1, "checkpoints": 2, "evals": 4}

    db = _open_dest(dest_path)
    try:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 4
    finally:
        db.close()


def test_engine_commit_preserved_vs_local_stamp(tmp_path):
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    dest_path = _dest_path(tmp_path)
    replay(run_dir, dest_path, origin_host="runner")

    db = _open_dest(dest_path)
    try:
        stored = db.execute(
            "SELECT engine_commit_hash FROM runs WHERE name='v0.4-prtcfr-r1'"
        ).fetchone()["engine_commit_hash"]
        assert stored == _RUNNER_SHA
        # A locally-stamped run would carry the client's HEAD, which differs.
        assert stored != run_db._get_engine_commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Collision guard
# ---------------------------------------------------------------------------


def test_local_name_collision_refused(tmp_path):
    dest_path = _dest_path(tmp_path)
    db = _open_dest(dest_path)
    # A pre-existing LOCAL run (origin_host NULL) with the same name.
    run_db.upsert_run(db, name="v0.4-prtcfr-r1", algorithm="prt-cfr")
    db.close()

    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    with pytest.raises(ReconcilerCollisionError):
        replay(run_dir, dest_path, origin_host="runner")

    # The local run is untouched: still NULL origin_host, no checkpoints ingested.
    db = _open_dest(dest_path)
    try:
        row = db.execute(
            "SELECT origin_host FROM runs WHERE name='v0.4-prtcfr-r1'"
        ).fetchone()
        assert row["origin_host"] is None
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    finally:
        db.close()


def test_cross_host_same_name_refused(tmp_path):
    dest_path = _dest_path(tmp_path)
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    replay(run_dir, dest_path, origin_host="hostA")
    with pytest.raises(ReconcilerCollisionError):
        replay(run_dir, dest_path, origin_host="hostB")


# ---------------------------------------------------------------------------
# Adversarial / untrusted input (design 5.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious",
    ["/etc/passwd", "../../../../etc/passwd", "/tmp/evil.pt", "..", "."],
)
def test_adversarial_file_path_rederived_not_trusted(tmp_path, malicious):
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_ckpt(conn, rid, 10, malicious)
    conn.close()
    dest_path = _dest_path(tmp_path)

    # Replay succeeds: the malicious path is re-derived (ignored), not rejected.
    replay(run_dir, dest_path, origin_host="runner")

    db = _open_dest(dest_path)
    try:
        fp = db.execute(
            "SELECT file_path FROM checkpoints WHERE iteration=10"
        ).fetchone()["file_path"]
        assert fp != malicious
        assert "/etc/passwd" not in fp
        assert "/tmp/evil.pt" not in fp
        # Re-derived path is contained in the local synced run dir.
        assert fp.startswith(str(run_dir.resolve()))
    finally:
        db.close()


@pytest.mark.parametrize("bad_name", ["../evil", "a/b", "", "..", "/abs", "has space"])
def test_adversarial_bad_run_name_rejected(tmp_path, bad_name):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    _insert_run(conn, name=bad_name)
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


def test_adversarial_bad_status_enum_rejected(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    _insert_run(conn, status="pwned; DROP TABLE runs")
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


def test_adversarial_unknown_algorithm_rejected(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    _insert_run(conn, algorithm="rm-rf-slash")
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


@pytest.mark.parametrize(
    "field,value",
    [
        ("win_rate", 5.0),
        ("win_rate", -0.1),
        ("games_played", -1),
        ("t1_cambia_rate", 42.0),
        ("ci_low", 9.9),
    ],
)
def test_adversarial_out_of_range_eval_numeric_rejected(tmp_path, field, value):
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_eval(conn, rid, 10, "random_no_cambia", **{field: value})
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


def test_adversarial_nonfinite_numeric_rejected(tmp_path):
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_eval(conn, rid, 10, "random_no_cambia", avg_score_margin=float("inf"))
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


# ---------------------------------------------------------------------------
# Stability-metric journal rows (cambia-363)
# ---------------------------------------------------------------------------


def test_stability_metric_row_above_one_reconciles_clean(tmp_path):
    """Exact NashConv on the tiny gate routinely exceeds 1.0 (live-fire value
    1.2581860616518317, cambia-363); the reconciler must not reject a
    stability-metric journal row (baseline == STABILITY_NASHCONV_BASELINE) on
    the win_rate 0..1 bound, since it carries a metric value, not a
    probability."""
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_eval(
        conn, rid, 10, run_db.STABILITY_NASHCONV_BASELINE, win_rate=1.2581860616518317
    )
    conn.close()

    summary = replay(run_dir, _dest_path(tmp_path), origin_host="runner")
    assert summary["evals"] == 1

    dest = _open_dest(_dest_path(tmp_path))
    try:
        row = dest.execute(
            "SELECT baseline, win_rate FROM eval_results WHERE baseline=?",
            (run_db.STABILITY_NASHCONV_BASELINE,),
        ).fetchone()
    finally:
        dest.close()
    assert row["baseline"] == "nashconv"
    assert row["win_rate"] == pytest.approx(1.2581860616518317)


def test_real_baseline_win_rate_above_one_still_rejected(tmp_path):
    """The stability-metric exemption (cambia-363) is scoped to
    STABILITY_METRIC_BASELINES only: a genuine agent win-rate row (e.g.
    random_no_cambia) with the same out-of-range value must still reject."""
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_eval(conn, rid, 10, "random_no_cambia", win_rate=1.26)
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


def test_adversarial_negative_checkpoint_iteration_rejected(tmp_path):
    run_dir = tmp_path / "runs" / "v0.4-prtcfr-r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = _new_source(run_dir / "run_db.sqlite")
    rid = _insert_run(conn)
    _insert_ckpt(conn, rid, -5, "prtcfr_snapshot_iter_-5.pt")
    conn.close()
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host="runner")


@pytest.mark.parametrize("bad_host", ["", "has space", "../evil", "a/b"])
def test_invalid_origin_host_rejected(tmp_path, bad_host):
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    with pytest.raises(ReconcilerValidationError):
        replay(run_dir, _dest_path(tmp_path), origin_host=bad_host)


def test_missing_run_db_raises(tmp_path):
    empty = tmp_path / "runs" / "no-db"
    empty.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ReconcilerError):
        replay(empty, _dest_path(tmp_path), origin_host="runner")


def test_replay_accepts_open_connection(tmp_path):
    """dest_db may be an already-open connection (from get_db); the reconciler
    must not close a connection it does not own."""
    run_dir = _build_full_run(tmp_path / "runs" / "v0.4-prtcfr-r1")
    dest_path = _dest_path(tmp_path)
    db = _open_dest(dest_path)
    try:
        replay(run_dir, db, origin_host="runner")
        # connection still usable after replay
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    finally:
        db.close()
