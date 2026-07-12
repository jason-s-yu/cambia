"""Tests for run_db best-metric / best-checkpoint mirroring (cambia-390).

The PRT-CFR stability controller tracks best_iteration/best_metric in memory
(and in resume_state.json) but historically never wrote it to run_db --
checkpoints.is_best stayed 0 and runs.best_metric_* stayed NULL for every run,
so a consumer reading run_db for the winning checkpoint got nothing. These
tests cover the run_db-level primitives the trainer wiring uses to mirror the
controller's best pointer: ``set_best_metric`` (unconditional, mode-agnostic
set) and ``mark_best_checkpoint`` (exclusive is_best flip, already existed but
was unused).
"""

import tempfile
from pathlib import Path

import pytest

import src.run_db as run_db


def _fresh_db():
    tmp = tempfile.mktemp(suffix=".db")
    return run_db.get_db(tmp), tmp


def test_set_best_metric_writes_fields():
    db, tmp = _fresh_db()
    try:
        rid = run_db.upsert_run(db, name="r1", algorithm="prt-cfr")
        run_db.set_best_metric(db, rid, "nashconv", 0.19808, 580)
        row = db.execute(
            "SELECT best_metric_name, best_metric_value, best_metric_iter "
            "FROM runs WHERE id=?",
            (rid,),
        ).fetchone()
        assert row["best_metric_name"] == "nashconv"
        assert row["best_metric_value"] == pytest.approx(0.19808)
        assert row["best_metric_iter"] == 580
    finally:
        db.close()
        Path(tmp).unlink(missing_ok=True)


def test_set_best_metric_overwrites_on_lower_value():
    """set_best_metric mirrors an externally-decided best (e.g. a min-mode
    stability controller, where lower is better); it must NOT reapply
    update_best_metric's higher-is-better comparison, or a min-mode metric's
    improving (lower) values would never overwrite the stored best."""
    db, tmp = _fresh_db()
    try:
        rid = run_db.upsert_run(db, name="r1", algorithm="prt-cfr")
        run_db.set_best_metric(db, rid, "nashconv", 0.5, 10)
        run_db.set_best_metric(db, rid, "nashconv", 0.2, 20)  # improvement, min-mode
        row = db.execute(
            "SELECT best_metric_value, best_metric_iter FROM runs WHERE id=?",
            (rid,),
        ).fetchone()
        assert row["best_metric_value"] == pytest.approx(0.2)
        assert row["best_metric_iter"] == 20
    finally:
        db.close()
        Path(tmp).unlink(missing_ok=True)


def test_mark_best_checkpoint_moves_flag_to_later_iteration():
    db, tmp = _fresh_db()
    try:
        rid = run_db.upsert_run(db, name="r1", algorithm="prt-cfr")
        id_a = run_db.register_checkpoint(db, rid, 10, "/tmp/a.pt")
        id_b = run_db.register_checkpoint(db, rid, 20, "/tmp/b.pt")

        run_db.mark_best_checkpoint(db, rid, id_a)
        rows = db.execute(
            "SELECT iteration, is_best FROM checkpoints WHERE run_id=? "
            "ORDER BY iteration",
            (rid,),
        ).fetchall()
        assert {r["iteration"]: r["is_best"] for r in rows} == {10: 1, 20: 0}

        run_db.mark_best_checkpoint(db, rid, id_b)  # a later, better iteration
        rows = db.execute(
            "SELECT iteration, is_best FROM checkpoints WHERE run_id=? "
            "ORDER BY iteration",
            (rid,),
        ).fetchall()
        assert {r["iteration"]: r["is_best"] for r in rows} == {10: 0, 20: 1}
    finally:
        db.close()
        Path(tmp).unlink(missing_ok=True)
