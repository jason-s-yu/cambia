"""
tests/test_harness_pull.py

Pull-loop coverage (design 4.1/4.4):
  - include-set derivation (retained vs --all-checkpoints)
  - reservoir exclusion (always)
  - replay invocation + harness_sync upsert
  - terminal-pull retry with backoff
  - push-run direction
  - real-rsync integration proving reservoir + non-retained exclusion

The reconciler replay core is consumed as-is (not reimplemented).
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from src import run_db
from src.harness import pull as pullmod
from src.harness.pull import (
    PullCoordinator,
    PullError,
    Runner,
    RsyncRunner,
    build_pull_filters,
    derive_include_set,
    is_terminal,
    is_valid_run_name,
    read_run_status,
    sanitize_last_status,
)

_RETAINED = (5, 10)
_NONRETAINED = (1,)


def _ckpt_name(it):
    return f"prtcfr_checkpoint_iter_{it}.pt"


def _build_remote_run(remote_root: Path, name="v0.4-prtcfr-r1", status="completed"):
    """Build a runner-side run dir: a plain (non-WAL) run_db.sqlite, snapshots/
    (retained + non-retained + best symlink), a reservoir/ dir, logs/, metrics."""
    run_dir = remote_root / name
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / "training.log").write_text("line1\nline2\n")
    (run_dir / "metrics.jsonl").write_text('{"iter": 10}\n')
    reservoir = run_dir / "reservoir" / "player_0"
    reservoir.mkdir(parents=True, exist_ok=True)
    (reservoir / "block.int16").write_bytes(b"\x00" * 4096)

    # Plain sqlite (default DELETE journal, no WAL) so a mode=ro replay reads it.
    conn = sqlite3.connect(str(run_dir / "run_db.sqlite"))
    conn.row_factory = sqlite3.Row
    conn.executescript(run_db._DDL)
    rid = run_db.upsert_run(conn, name=name, algorithm="prt-cfr", status=status)
    for it in _RETAINED + _NONRETAINED:
        fp = snaps / _ckpt_name(it)
        fp.write_bytes(b"ckpt")
        run_db.register_checkpoint(conn, rid, it, str(fp))
    for it in _RETAINED:
        conn.execute(
            "UPDATE checkpoints SET is_retained=1 WHERE run_id=? AND iteration=?",
            (rid, it),
        )
    for it in _NONRETAINED:
        conn.execute(
            "UPDATE checkpoints SET is_retained=0 WHERE run_id=? AND iteration=?",
            (rid, it),
        )
    conn.commit()
    conn.close()
    # best.pt symlink into the retained set (best-effort; skip if unsupported)
    try:
        (snaps / "best.pt").symlink_to(_ckpt_name(_RETAINED[-1]))
    except OSError:
        pass
    return run_dir


# ---------------------------------------------------------------------------
# Pure filter construction (reservoir always excluded)
# ---------------------------------------------------------------------------


def test_filters_exclude_reservoir_both_modes():
    default = build_pull_filters({"a.pt"}, all_checkpoints=False)
    widened = build_pull_filters(None, all_checkpoints=True)
    for f in (default, widened):
        assert "--exclude=/reservoir/" in f
        assert "--exclude=/reservoir/**" in f


def test_filters_retained_only_default():
    f = build_pull_filters({"prtcfr_checkpoint_iter_5.pt"}, all_checkpoints=False)
    assert "--include=/snapshots/prtcfr_checkpoint_iter_5.pt" in f
    assert "--exclude=/snapshots/**" in f  # drops the rest


def test_filters_all_checkpoints_no_snapshot_excludes():
    f = build_pull_filters(None, all_checkpoints=True)
    assert not any("snapshots" in arg for arg in f)  # whole tree pulled


# ---------------------------------------------------------------------------
# Include-set derivation + status read from a synced run_db
# ---------------------------------------------------------------------------


def test_derive_include_set_retained_only(tmp_path):
    run_dir = _build_remote_run(tmp_path / "remote")
    db_path = run_dir / "run_db.sqlite"
    got = derive_include_set(db_path, all_checkpoints=False)
    assert got == {_ckpt_name(5), _ckpt_name(10)}
    assert _ckpt_name(1) not in got  # non-retained excluded


def test_derive_include_set_all_checkpoints_returns_none(tmp_path):
    run_dir = _build_remote_run(tmp_path / "remote")
    assert derive_include_set(run_dir / "run_db.sqlite", all_checkpoints=True) is None


def test_read_run_status(tmp_path):
    run_dir = _build_remote_run(tmp_path / "remote", status="running")
    assert read_run_status(run_dir / "run_db.sqlite", "v0.4-prtcfr-r1") == "running"


def test_is_terminal():
    assert is_terminal("completed")
    assert is_terminal("crashed")
    assert not is_terminal("running")
    assert not is_terminal(None)


# ---------------------------------------------------------------------------
# Fake runner: pull_once replay + harness_sync + include-set feeding
# ---------------------------------------------------------------------------


class FakeRunner(Runner):
    """Copies a prebuilt remote run dir into the local synced dir, recording
    calls and honoring an injected failure count for retry tests."""

    def __init__(self, remote_root: Path, fail_pull_db=0):
        self.remote_root = Path(remote_root)
        self.calls = []
        self.fail_pull_db = fail_pull_db

    def pull_db(self, run_name, local_run_dir):
        self.calls.append(("pull_db", run_name))
        if self.fail_pull_db > 0:
            self.fail_pull_db -= 1
            raise PullError("simulated rsync failure")
        local_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.remote_root / run_name / "run_db.sqlite",
            local_run_dir / "run_db.sqlite",
        )

    def pull_run(self, run_name, local_run_dir, filters):
        self.calls.append(("pull_run", run_name, list(filters)))
        # Mirror the remote run dir minus reservoir and non-retained snapshots
        # (the filters say what to keep; here we just copy snapshots that appear
        # as an --include so replay has real files).
        remote = self.remote_root / run_name
        (local_run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        includes = {
            f.split("/snapshots/")[-1]
            for f in filters
            if f.startswith("--include=/snapshots/") and f.endswith(".pt")
        }
        for name in includes:
            src = remote / "snapshots" / name
            if src.exists():
                shutil.copy2(src, local_run_dir / "snapshots" / name)

    def push_run(self, run_name, local_run_dir):
        self.calls.append(("push_run", run_name, str(local_run_dir)))


def _dest_db(tmp_path):
    return run_db.get_db(str(tmp_path / "dest" / "cambia_runs.db"))


def test_pull_once_replays_and_upserts_harness_sync(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    status = coord.pull_once("v0.4-prtcfr-r1")
    assert status == "completed"

    # replay landed the run into the authoritative db with origin_host stamped.
    row = dest.execute(
        "SELECT origin_host, status FROM runs WHERE name=?", ("v0.4-prtcfr-r1",)
    ).fetchone()
    assert row["origin_host"] == "runner"
    assert row["status"] == "completed"

    # harness_sync current-state row upserted (design 4.5).
    hs = dest.execute(
        "SELECT origin_host, last_status, last_sync_at FROM harness_sync WHERE run_name=?",
        ("v0.4-prtcfr-r1",),
    ).fetchone()
    assert hs["origin_host"] == "runner"
    assert hs["last_status"] == "completed"
    assert hs["last_sync_at"]  # timestamp recorded
    dest.close()


def test_pull_once_feeds_retained_include_set(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    coord.pull_once("v0.4-prtcfr-r1")
    pull_run_call = next(c for c in runner.calls if c[0] == "pull_run")
    filters = pull_run_call[2]
    # retained checkpoints in the include set, reservoir excluded, non-retained not included.
    assert f"--include=/snapshots/{_ckpt_name(5)}" in filters
    assert f"--include=/snapshots/{_ckpt_name(10)}" in filters
    assert f"--include=/snapshots/{_ckpt_name(1)}" not in filters
    assert "--exclude=/reservoir/**" in filters
    dest.close()


def test_pull_once_all_checkpoints_widens(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    coord.pull_once("v0.4-prtcfr-r1", all_checkpoints=True)
    filters = next(c for c in runner.calls if c[0] == "pull_run")[2]
    assert not any("snapshots" in f for f in filters)  # no snapshot filtering
    assert "--exclude=/reservoir/**" in filters  # reservoir still excluded
    dest.close()


def test_pull_invokes_replay_with_expected_args(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    seen = {}

    def spy_replay(run_dir, conn, origin_host):
        seen["run_dir"] = Path(run_dir)
        seen["conn"] = conn
        seen["origin_host"] = origin_host
        return {"runs": 1, "checkpoints": 0, "evals": 0}

    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        replay_fn=spy_replay,
    )
    coord.pull_once("v0.4-prtcfr-r1")
    assert seen["run_dir"] == tmp_path / "local" / "v0.4-prtcfr-r1"
    assert seen["conn"] is dest
    assert seen["origin_host"] == "runner"
    dest.close()


# ---------------------------------------------------------------------------
# Terminal-pull retry with backoff
# ---------------------------------------------------------------------------


def test_pull_with_retry_recovers(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote", fail_pull_db=2)  # fail twice, then ok
    sleeps = []
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        sleep_fn=sleeps.append,
    )
    status = coord.pull_with_retry("v0.4-prtcfr-r1", attempts=5, base_backoff=1.0)
    assert status == "completed"
    assert sleeps == [1.0, 2.0]  # exponential backoff between the 3 attempts
    dest.close()


def test_pull_with_retry_exhausts(tmp_path):
    _build_remote_run(tmp_path / "remote")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote", fail_pull_db=99)  # always fails
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        sleep_fn=lambda s: None,
    )
    with pytest.raises(PullError):
        coord.pull_with_retry("v0.4-prtcfr-r1", attempts=3, base_backoff=0.1)
    dest.close()


# ---------------------------------------------------------------------------
# watch loop: terminal transition triggers immediate pull, run drops once synced
# ---------------------------------------------------------------------------


def test_watch_terminal_transition_pulls_and_drops(tmp_path):
    _build_remote_run(tmp_path / "remote", status="completed")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        sleep_fn=lambda s: None,
    )
    events = []
    # job_lister reports the run terminal on the first tick.
    coord_pulled = {"count": 0}
    orig_pull_once = coord.pull_once

    def counting_pull_once(name, all_checkpoints=False):
        coord_pulled["count"] += 1
        return orig_pull_once(name, all_checkpoints)

    coord.pull_once = counting_pull_once

    pullmod.watch(
        coord,
        job_lister=lambda: [{"name": "v0.4-prtcfr-r1", "status": "completed"}],
        interval_seconds=0,
        on_event=events.append,
        max_ticks=1,
    )
    # harness_sync recorded a terminal status -> run reconciled + dropped.
    hs = dest.execute(
        "SELECT last_status FROM harness_sync WHERE run_name=?", ("v0.4-prtcfr-r1",)
    ).fetchone()
    assert hs["last_status"] == "completed"
    assert any("dropped" in e for e in events)
    dest.close()


# ---------------------------------------------------------------------------
# push-run direction
# ---------------------------------------------------------------------------


def test_push_run_direction(tmp_path):
    dest = _dest_db(tmp_path)
    local = tmp_path / "local" / "myrun"
    local.mkdir(parents=True)
    (local / "run_db.sqlite").write_bytes(b"db")
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    coord.push_run("myrun")
    push = next(c for c in runner.calls if c[0] == "push_run")
    assert push[1] == "myrun"
    assert push[2] == str(local)  # the client-local dir goes up
    dest.close()


def test_push_run_refuses_missing_dir(tmp_path):
    dest = _dest_db(tmp_path)
    coord = PullCoordinator(
        runner=FakeRunner(tmp_path / "remote"),
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    with pytest.raises(PullError):
        coord.push_run("nonexistent")
    dest.close()


# ---------------------------------------------------------------------------
# Real-rsync integration: reservoir + non-retained exclusion proven end-to-end
# ---------------------------------------------------------------------------


def _have_rsync():
    return shutil.which("rsync") is not None


@pytest.mark.skipif(not _have_rsync(), reason="rsync not installed")
def test_rsync_excludes_reservoir_and_nonretained(tmp_path):
    remote_root = tmp_path / "remote"
    _build_remote_run(remote_root)
    local_root = tmp_path / "local"
    # Local rsync mode: empty ssh_alias -> no "host:" prefix.
    runner = RsyncRunner(ssh_alias="", runner_runs_dir=str(remote_root))
    local_dir = local_root / "v0.4-prtcfr-r1"

    runner.pull_db("v0.4-prtcfr-r1", local_dir)
    includes = derive_include_set(local_dir / "run_db.sqlite", all_checkpoints=False)
    filters = build_pull_filters(includes, all_checkpoints=False)
    runner.pull_run("v0.4-prtcfr-r1", local_dir, filters)

    snaps = local_dir / "snapshots"
    assert (snaps / _ckpt_name(5)).exists()  # retained pulled
    assert (snaps / _ckpt_name(10)).exists()
    assert not (snaps / _ckpt_name(1)).exists()  # non-retained NOT pulled
    assert not (local_dir / "reservoir").exists()  # reservoir NEVER pulled
    assert (local_dir / "metrics.jsonl").exists()  # metadata pulled
    assert (local_dir / "logs" / "training.log").exists()


@pytest.mark.skipif(not _have_rsync(), reason="rsync not installed")
def test_rsync_all_checkpoints_pulls_everything_but_reservoir(tmp_path):
    remote_root = tmp_path / "remote"
    _build_remote_run(remote_root)
    local_dir = tmp_path / "local" / "v0.4-prtcfr-r1"
    runner = RsyncRunner(ssh_alias="", runner_runs_dir=str(remote_root))
    runner.pull_db("v0.4-prtcfr-r1", local_dir)
    filters = build_pull_filters(None, all_checkpoints=True)
    runner.pull_run("v0.4-prtcfr-r1", local_dir, filters)

    snaps = local_dir / "snapshots"
    assert (snaps / _ckpt_name(1)).exists()  # non-retained now included
    assert (snaps / _ckpt_name(5)).exists()
    assert not (local_dir / "reservoir").exists()  # reservoir still excluded


# ---------------------------------------------------------------------------
# H1: hostile runner-supplied names never reach the filesystem or rsync
# ---------------------------------------------------------------------------

_HOSTILE_NAMES = [
    "../../etc/passwd",
    "..",
    ".",
    "/abs/path",
    "a/b",
    "run;rm -rf x",
    "run name",
    "run\x00null",
    "",
]


def test_is_valid_run_name_matches_reconciler_rules():
    for good in ("v0.4-prtcfr-r1", "myrun", "a.b_c-1"):
        assert is_valid_run_name(good)
    for bad in _HOSTILE_NAMES:
        assert not is_valid_run_name(bad)
    assert not is_valid_run_name(None)


@pytest.mark.parametrize("bad", _HOSTILE_NAMES)
def test_coordinator_rejects_hostile_name_before_fs_touch(tmp_path, bad):
    """local_run_dir is the chokepoint: an unsafe name raises before any pull,
    push, or filesystem access (H1). No local dir is created for the bad name."""
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
    )
    with pytest.raises(PullError):
        coord.pull_once(bad)
    with pytest.raises(PullError):
        coord.pull_with_retry(bad, attempts=1)
    with pytest.raises(PullError):
        coord.push_run(bad)
    # The runner was never invoked and no directory was created.
    assert runner.calls == []
    assert not (tmp_path / "local").exists() or list((tmp_path / "local").iterdir()) == []
    dest.close()


def test_watch_skips_hostile_names(tmp_path):
    """A hostile name in the job list is skipped + logged; a sibling good run in
    the same tick is still pulled (H1 at the watch ingress)."""
    _build_remote_run(tmp_path / "remote", status="completed")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        sleep_fn=lambda s: None,
    )
    events = []
    pullmod.watch(
        coord,
        job_lister=lambda: [
            {"name": "../../etc/passwd", "status": "completed"},
            {"name": "v0.4-prtcfr-r1", "status": "completed"},
        ],
        interval_seconds=0,
        on_event=events.append,
        max_ticks=1,
    )
    # The hostile name was skipped loudly and never became a pull.
    assert any("unsafe name" in e and "etc/passwd" in e for e in events)
    assert not (tmp_path / "local" / "etc").exists()
    assert ("pull_db", "../../etc/passwd") not in [(c[0], c[1]) for c in runner.calls]
    # The good sibling still reconciled.
    hs = dest.execute(
        "SELECT last_status FROM harness_sync WHERE run_name=?", ("v0.4-prtcfr-r1",)
    ).fetchone()
    assert hs is not None and hs["last_status"] == "completed"
    dest.close()


# ---------------------------------------------------------------------------
# L8: every rsync invocation carries --safe-links (both directions)
# ---------------------------------------------------------------------------


def test_rsync_argv_includes_safe_links(tmp_path, monkeypatch):
    captured = []

    def fake_run(self, args):
        captured.append(list(args))

    monkeypatch.setattr(RsyncRunner, "_run", fake_run)
    runner = RsyncRunner(ssh_alias="runner", runner_runs_dir="/remote/runs")
    local = tmp_path / "run1"

    runner.pull_db("run1", local)  # 3 argv (db + -wal + -shm)
    runner.pull_run("run1", local, ["--exclude=/reservoir/"])  # down
    runner.push_run("run1", local)  # up

    assert len(captured) == 5
    for args in captured:
        assert "--safe-links" in args


# ---------------------------------------------------------------------------
# L10: last_status is clamped to the known enum before harness_sync upsert
# ---------------------------------------------------------------------------


def test_sanitize_last_status():
    # Known run_db lifecycle + runnerd process states pass through.
    for known in ("running", "completed", "queued", "preparing", "starting", "crashed"):
        assert sanitize_last_status(known) == known
    # Anything else is clamped to "unknown".
    for bad in ("pwned; DROP TABLE runs", "weird", "RUNNING", "1"):
        assert sanitize_last_status(bad) == "unknown"
    # None (no status observed yet) is preserved as absence.
    assert sanitize_last_status(None) is None


def test_pull_once_sanitizes_unknown_status_into_harness_sync(tmp_path, monkeypatch):
    """A pulled db whose status is garbage stores 'unknown' in harness_sync,
    never the raw text (L10). Replay is stubbed and read_run_status forced to an
    unknown value so the untrusted status reaches the upsert path unfiltered."""
    _build_remote_run(tmp_path / "remote", status="running")
    dest = _dest_db(tmp_path)
    runner = FakeRunner(tmp_path / "remote")
    coord = PullCoordinator(
        runner=runner,
        local_runs_dir=tmp_path / "local",
        dest_conn=dest,
        origin_host="runner",
        replay_fn=lambda *a: {"runs": 0, "checkpoints": 0, "evals": 0},
    )
    monkeypatch.setattr(
        pullmod, "read_run_status", lambda db_path, run_name: "totally-bogus"
    )
    status = coord.pull_once("v0.4-prtcfr-r1")
    # pull_once returns the raw observed status ...
    assert status == "totally-bogus"
    # ... but harness_sync only ever holds a known enum value or "unknown".
    hs = dest.execute(
        "SELECT last_status FROM harness_sync WHERE run_name=?", ("v0.4-prtcfr-r1",)
    ).fetchone()
    assert hs["last_status"] == "unknown"
    dest.close()
