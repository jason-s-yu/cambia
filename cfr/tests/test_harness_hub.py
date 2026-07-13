"""
tests/test_harness_hub.py

Codebridge hub reflection (cambia-353). Offline: an injected recorder poster for
the reflector logic, and a threaded self-signed HTTPS sink (harness_tls_util) for
the end-to-end signature/pin path. The hub verify is reimplemented here (the real
codebridge.domain.webhooks pulls asyncpg) and pinned against the hub's own test
vector so the wire contract is proven byte-for-byte.
"""

import hmac
import json
from hashlib import sha256
from pathlib import Path

import pytest

from src import run_db
from src.harness import config as cfgmod
from src.harness import hub


# --- hub verify reimplementation (copied scheme from codebridge webhooks) -----
def _verify_inbound(secret, timestamp, body, signature, *, now, window=300):
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(now - ts) > window:
        return False
    presented = (
        signature[len("sha256=") :] if signature.startswith("sha256=") else signature
    )
    expected = hmac.new(
        secret.encode("utf-8"), f"{ts}.{body}".encode("utf-8"), sha256
    ).hexdigest()
    return hmac.compare_digest(presented, expected)


def _db(tmp_path):
    return run_db.get_db(str(tmp_path / "cambia_runs.db"))


def _hub_cfg(**kw):
    base = dict(
        url="https://hub.example.com",
        slug="cambia-harness-reflect",
        collector_item="cambia-439",
        secret_env="CAMBIA_HUB_TEST_SECRET",
    )
    base.update(kw)
    return cfgmod.HubConfig(**base)


# ===========================================================================
# Signing / serialization / payload shape (wire contract)
# ===========================================================================
def test_sign_matches_hub_test_vector():
    # The exact vector from codebridge tests/domain/test_webhooks.py.
    assert (
        hub.sign("sek", 1700000000, '{"id":1}')
        == "9381590abcda65ed83dda791f134a209c0b6c41ad612c16f9c700278cddea2c8"
    )


def test_sign_covers_timestamp_and_body():
    assert hub.sign("s", 1, "b") != hub.sign("s", 2, "b")
    assert hub.sign("s", 1, "a") != hub.sign("s", 1, "b")


def test_serialize_payload_is_canonical():
    body = hub.serialize_payload({"b": 1, "a": 2})
    assert body == '{"a":2,"b":1}'  # sorted keys, no spaces


def test_build_payload_shape():
    p = hub.build_payload("cambia-439", "harness:nash:job1", "body-text")
    assert p == {
        "ops": [
            {
                "op": "set",
                "item_handle": "cambia-439",
                "key": "harness:nash:job1",
                "role": "work",
                "body": "body-text",
            }
        ]
    }


def test_note_key():
    assert hub.note_key("nash", "job1") == "harness:nash:job1"


# ===========================================================================
# End-to-end POST: real self-signed HTTPS sink, fingerprint pin, verify roundtrip
# ===========================================================================
def test_post_note_signs_pins_and_posts(tmp_path, monkeypatch):
    from tests.harness_tls_util import RecordingServer, make_self_signed

    cert, key, fp = make_self_signed(tmp_path)
    routes = {("POST", "/api/hooks/"): (200, {"ok": True})}
    monkeypatch.setenv("CAMBIA_HUB_TEST_SECRET", "trigger-secret")
    with RecordingServer(cert, key, routes) as server:
        cfg = _hub_cfg(url=server.base_url, slug="the-slug", cert_fingerprint=fp)
        payload = hub.build_payload("cambia-439", "harness:nash:j", "hi")
        clock = lambda: 1700000000.0
        ok = hub.post_note(cfg, payload, clock=clock)
        assert ok is True
    assert len(server.requests) == 1
    req = server.requests[0]
    assert req["method"] == "POST"
    assert req["path"] == "/api/hooks/the-slug"
    sig = req["headers"]["X-Codebridge-Signature"]
    ts = req["headers"]["X-Codebridge-Timestamp"]
    assert sig.startswith("sha256=")
    assert ts == "1700000000"
    # The captured body is the exact signed bytes; verify with the hub scheme.
    assert _verify_inbound("trigger-secret", ts, req["body"], sig, now=1700000000)
    # A tampered body fails, confirming the MAC covers the exact body.
    assert not _verify_inbound(
        "trigger-secret", ts, req["body"] + "x", sig, now=1700000000
    )
    assert json.loads(req["body"]) == payload


def test_post_note_wrong_fingerprint_is_dropped(tmp_path, monkeypatch):
    from tests.harness_tls_util import RecordingServer, make_self_signed

    cert, key, _ = make_self_signed(tmp_path)
    routes = {("POST", "/api/hooks/"): (200, {"ok": True})}
    monkeypatch.setenv("CAMBIA_HUB_TEST_SECRET", "trigger-secret")
    with RecordingServer(cert, key, routes) as server:
        # A pin that does not match the server cert -> transport error -> dropped.
        cfg = _hub_cfg(url=server.base_url, cert_fingerprint="ab" * 32)
        assert hub.post_note(cfg, {"ops": []}) is False


# ===========================================================================
# Containment: any poster/sender failure is swallowed, returns False
# ===========================================================================
def test_post_note_containment_on_sender_exception(monkeypatch):
    monkeypatch.setenv("CAMBIA_HUB_TEST_SECRET", "s")

    def boom(*a, **k):
        raise ConnectionError("hub down")

    assert hub.post_note(_hub_cfg(), {"ops": []}, sender=boom) is False


def test_post_note_non_2xx_is_false(monkeypatch):
    monkeypatch.setenv("CAMBIA_HUB_TEST_SECRET", "s")
    assert hub.post_note(_hub_cfg(), {"ops": []}, sender=lambda *a: (500, "err")) is False


def test_post_note_missing_secret_is_false():
    # secret_env points at an unset var -> resolve_secret raises -> dropped.
    assert hub.post_note(_hub_cfg(secret_env="NOPE_UNSET"), {"ops": []}) is False


# ===========================================================================
# Reflector: monotonic ordering + dedup + item resolution
# ===========================================================================
class _Recorder:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, payload):
        self.calls.append(payload)
        return self.ok

    @property
    def last_op(self):
        return self.calls[-1]["ops"][0]


def _reflector(tmp_path, rec, origin_host="nash"):
    return hub.HubReflector(_hub_cfg(), origin_host, _db(tmp_path), poster=rec)


def test_submit_then_monotonic_progression(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    assert r.reflect_submit(
        {"name": "j1", "kind": "train", "commit": "abc", "hub_item": "cambia-359"}
    )
    assert r.reflect_view({"job_id": "j1", "state": "running"})
    # same state again -> idempotent, dropped
    assert not r.reflect_view({"job_id": "j1", "state": "running"})
    # terminal advances
    assert r.reflect_view({"job_id": "j1", "state": "completed"})
    assert len(rec.calls) == 3
    # keyed note stable across the lifecycle
    keys = {c["ops"][0]["key"] for c in rec.calls}
    assert keys == {"harness:nash:j1"}


def test_stale_transition_dropped(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    assert r.reflect_view({"job_id": "j1", "state": "running"})
    # a stale/out-of-order poll showing an earlier state regresses -> dropped
    assert not r.reflect_view({"job_id": "j1", "state": "queued"})
    assert len(rec.calls) == 1


def test_row_updated_only_on_success(tmp_path):
    # cooldown_seconds=0: isolate the row-write-on-success behavior from the
    # circuit breaker (covered separately below), so a failure never gates the
    # very next attempt in this test.
    rec = _Recorder(ok=False)
    db = _db(tmp_path)
    r = hub.HubReflector(_hub_cfg(), "nash", db, poster=rec, cooldown_seconds=0.0)
    assert not r.reflect_view({"job_id": "j1", "state": "running"})
    # failed post left no reflection row -> next attempt retries (not deduped)
    assert run_db.get_harness_reflection(db, "nash", "j1") is None
    rec.ok = True
    assert r.reflect_view({"job_id": "j1", "state": "running"})
    row = run_db.get_harness_reflection(db, "nash", "j1")
    assert row["last_reflected_state"] == "running"


def test_hub_item_then_collector(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "linked", "state": "running", "hub_item": "cambia-359"})
    assert rec.last_op["item_handle"] == "cambia-359"
    r.reflect_view({"job_id": "unlinked", "state": "running"})
    assert rec.last_op["item_handle"] == "cambia-439"  # collector


def test_stored_item_reused_when_hub_item_absent(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "running", "hub_item": "cambia-359"})
    # a later transition whose view lost hub_item still targets the same item
    r.reflect_view({"job_id": "j1", "state": "completed"})
    assert rec.last_op["item_handle"] == "cambia-359"


def test_resume_bypasses_rank_guard(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})
    # resume legitimately moves terminal -> running (a rank regression) via force
    assert r.reflect_resume({"job_id": "j1"})
    assert rec.last_op["body"].splitlines()[2] == "event: resume"


def test_purge_reflects_terminal(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})
    assert r.reflect_purge("j1")
    op = rec.last_op
    assert "state: purged" in op["body"]
    assert op["item_handle"] == "cambia-439"


# ===========================================================================
# Review finding 1 (HIGH): same-name resubmit after purge/terminal must not be
# locked out of reflection (a terminal-or-purged last_state observed with a new
# pending/running-rank state is a lifecycle reset, not a regression).
# ===========================================================================
def test_resubmit_after_purge_reflects_full_new_lifecycle(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})
    r.reflect_purge("j1")  # last_reflected_state = "purged" (rank 4)
    # A same-name resubmit's submit/running/terminal must all land.
    assert r.reflect_submit({"name": "j1", "kind": "train"})
    assert r.reflect_view({"job_id": "j1", "state": "running"})
    assert r.reflect_view({"job_id": "j1", "state": "completed"})
    events = [c["ops"][0]["body"].splitlines()[2] for c in rec.calls[-3:]]
    assert events == ["event: submit", "event: running", "event: terminal"]


def test_resubmit_after_plain_terminal_reflects_full_new_lifecycle(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})  # last_state rank 3
    # No purge this time -- just a same-name resubmit after a terminal.
    assert r.reflect_submit({"name": "j1", "kind": "train"})
    assert r.reflect_view({"job_id": "j1", "state": "running"})
    assert r.reflect_view({"job_id": "j1", "state": "completed"})


def test_reconcile_heals_terminal_to_running_without_force(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})
    rec.calls.clear()
    # A plain (force=False) drift-reconcile poll observing the job running again
    # (an external resume, or a resubmit the watcher missed) must heal, not drop.
    posted = r.reconcile([{"job_id": "j1", "state": "running"}])
    assert posted == 1
    assert rec.last_op["body"].splitlines()[2] == "event: running"


def test_running_to_submitted_same_lifecycle_still_dropped(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "running"})  # rank 2, not terminal
    # A same-lifecycle regression (running observed, then a stale "submitted"
    # poll) is still a drop -- only a terminal/purged last_state resets.
    assert not r.reflect_view({"job_id": "j1", "state": "submitted"})
    assert len(rec.calls) == 1


def test_equal_state_still_deduped_after_lifecycle_reset_fix(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    r.reflect_view({"job_id": "j1", "state": "completed"})
    assert not r.reflect_view({"job_id": "j1", "state": "completed"})
    assert len(rec.calls) == 1


# ===========================================================================
# Review finding 2 (MEDIUM): reflector-wide failure circuit breaker. A dead hub
# must cost at most one blocking poster attempt per cooldown window, not one per
# unreflected job per tick.
# ===========================================================================
class _FakeClock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_breaker_opens_after_failure_and_skips_within_cooldown(tmp_path):
    clock = _FakeClock()
    rec = _Recorder(ok=False)
    r = hub.HubReflector(
        _hub_cfg(), "nash", _db(tmp_path), poster=rec, clock=clock, cooldown_seconds=300.0
    )
    assert not r.reflect_view({"job_id": "j1", "state": "running"})
    assert len(rec.calls) == 1  # the one attempt that armed the breaker
    # Within the cooldown, a distinct job's post is skipped WITHOUT invoking the
    # poster at all (the breaker is reflector-wide, not per-job).
    clock.advance(10.0)
    assert not r.reflect_view({"job_id": "j2", "state": "running"})
    assert len(rec.calls) == 1  # poster not called again


def test_breaker_reattempts_after_cooldown_expiry(tmp_path):
    clock = _FakeClock()
    rec = _Recorder(ok=False)
    r = hub.HubReflector(
        _hub_cfg(), "nash", _db(tmp_path), poster=rec, clock=clock, cooldown_seconds=300.0
    )
    assert not r.reflect_view({"job_id": "j1", "state": "running"})
    assert len(rec.calls) == 1
    clock.advance(300.0)  # cooldown elapsed
    assert not r.reflect_view({"job_id": "j2", "state": "running"})
    assert len(rec.calls) == 2  # attempted again (still failing, re-arms)


def test_breaker_success_resets_it(tmp_path):
    clock = _FakeClock()
    rec = _Recorder(ok=False)
    r = hub.HubReflector(
        _hub_cfg(), "nash", _db(tmp_path), poster=rec, clock=clock, cooldown_seconds=300.0
    )
    assert not r.reflect_view({"job_id": "j1", "state": "running"})
    clock.advance(300.0)
    rec.ok = True
    assert r.reflect_view({"job_id": "j2", "state": "running"})  # re-attempt succeeds
    # A success re-arms the breaker immediately: no cooldown before the next post.
    assert r.reflect_view({"job_id": "j3", "state": "running"})
    assert len(rec.calls) == 3


def test_breaker_checked_before_metrics_read(tmp_path, monkeypatch):
    clock = _FakeClock()
    db = _db(tmp_path)
    rec = _Recorder(ok=False)
    r = hub.HubReflector(
        _hub_cfg(), "nash", db, poster=rec, clock=clock, cooldown_seconds=300.0
    )
    r.reflect_view({"job_id": "j1", "state": "running"})  # arms the breaker

    calls = {"n": 0}
    orig = hub.read_metrics_tail

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(hub, "read_metrics_tail", counting)
    # Gated by the breaker: with_metrics=True must NOT read the metrics tail.
    assert not r.reflect_view({"job_id": "j2", "state": "completed"}, with_metrics=True)
    assert calls["n"] == 0


def test_breaker_gates_force_true_posts_too(tmp_path):
    clock = _FakeClock()
    rec = _Recorder(ok=False)
    r = hub.HubReflector(
        _hub_cfg(), "nash", _db(tmp_path), poster=rec, clock=clock, cooldown_seconds=300.0
    )
    assert not r.reflect_submit({"name": "j1", "kind": "train"})  # arms the breaker
    assert len(rec.calls) == 1
    # A force=True post (resume/purge/submit) is still gated by the breaker.
    assert not r.reflect_purge("j2")
    assert not r.reflect_resume({"job_id": "j3"})
    assert len(rec.calls) == 1


def test_watch_loop_dead_hub_called_at_most_once_per_cooldown_window(tmp_path):
    """watch-loop-level: many ticks with many unreflected jobs, a dead poster is
    invoked at most once per cooldown window, not once per job per tick."""
    from src.harness import pull as pullmod

    clock = _FakeClock()
    rec = _Recorder(ok=False)
    reflector = hub.HubReflector(
        _hub_cfg(), "nash", _db(tmp_path), poster=rec, clock=clock, cooldown_seconds=300.0
    )
    coord = _FakeCoord()
    jobs = [{"job_id": f"j{i}", "state": "running"} for i in range(5)]
    events = []
    pullmod.watch(
        coord,
        job_lister=lambda: jobs,
        interval_seconds=0,
        on_event=events.append,
        max_ticks=10,
        reflector=reflector,
    )
    # 10 ticks x 5 jobs = 50 potential reflections, but the breaker (armed on the
    # very first failure, clock never advances across ticks in this test) caps
    # the poster to exactly one invocation.
    assert len(rec.calls) == 1


# ===========================================================================
# Drift reconcile
# ===========================================================================
def test_reconcile_reposts_only_diffs(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    # j1 already reflected at running; j2 never reflected
    r.reflect_view({"job_id": "j1", "state": "running"})
    rec.calls.clear()
    posted = r.reconcile(
        [
            {"job_id": "j1", "state": "running"},  # unchanged -> skip
            {"job_id": "j2", "state": "completed"},  # new -> post
        ]
    )
    assert posted == 1
    assert rec.last_op["key"] == "harness:nash:j2"


def test_reconcile_survives_a_bad_view(tmp_path):
    rec = _Recorder()
    r = _reflector(tmp_path, rec)
    posted = r.reconcile([{"no_id": True}, {"job_id": "ok", "state": "running"}])
    assert posted == 1


# ===========================================================================
# Metrics tail (read from the reconciled run_db)
# ===========================================================================
def test_metrics_tail_in_terminal_body(tmp_path):
    db = _db(tmp_path)
    rid = run_db.upsert_run(db, name="j1", algorithm="prt-cfr", status="completed")
    for it in (10, 20, 30):
        run_db.register_checkpoint(db, rid, it, str(tmp_path / f"c{it}.pt"))
    # retain two of three
    db.execute("UPDATE checkpoints SET is_retained=0 WHERE iteration=10")
    db.commit()
    from src.evaluate_agents import MEAN_IMP_BASELINES

    for bl in MEAN_IMP_BASELINES:
        run_db.insert_eval_result(
            db, rid, None, {"iteration": 30, "baseline": bl, "win_rate": 0.5}
        )
    rec = _Recorder()
    r = hub.HubReflector(_hub_cfg(), "nash", db, poster=rec)
    assert r.reflect_view({"job_id": "j1", "state": "completed"}, with_metrics=True)
    body = rec.last_op["body"]
    assert "latest_iteration: 30" in body
    assert "retained_checkpoints: 2" in body
    assert "latest_mean_imp: 0.5000" in body


def test_metrics_tail_absent_run_returns_none(tmp_path):
    db = _db(tmp_path)
    assert hub.read_metrics_tail(db, "nope") is None


# ===========================================================================
# Watch containment: a raising reflector never breaks the pull loop
# ===========================================================================
class _FakeCoord:
    """Minimal watch()-facing double. `dest`/`origin_host` mirror the real
    PullCoordinator's attributes: watch() (cambia-449) consults the persisted
    harness_sync row via `coordinator.dest` before re-adding an already-terminal
    job to its active set, so a fake standing in for the coordinator needs a
    real db connection even when it never calls into it in a given test."""

    def __init__(self, dest=None, origin_host="runner"):
        self.pulled = []
        self.sleep_fn = lambda s: None
        self.dest = dest if dest is not None else run_db.get_db(":memory:")
        self.origin_host = origin_host

    def pull_with_retry(self, name, all_checkpoints=False):
        self.pulled.append(name)
        return "completed"

    def pull_once(self, name, all_checkpoints=False):
        return "completed"

    def local_run_dir(self, name):
        return Path("/nonexistent") / name


class _BoomReflector:
    def reflect_view(self, *a, **k):
        raise RuntimeError("hub down")

    def reconcile(self, views):
        raise RuntimeError("hub down")


def test_watch_unaffected_by_hub_outage(tmp_path):
    from src.harness import pull as pullmod

    coord = _FakeCoord()
    events = []
    # A terminal job + a raising reflector: the pull still happens, no exception
    # escapes, and the loop reaches max_ticks.
    pullmod.watch(
        coord,
        job_lister=lambda: [{"job_id": "j1", "state": "completed"}],
        interval_seconds=0,
        on_event=events.append,
        max_ticks=1,
        reflector=_BoomReflector(),
    )
    assert coord.pulled == ["j1"]  # pull unaffected by the hub outage
    assert any("dropped" in e for e in events)  # reflection failure logged, not raised


# ===========================================================================
# build_reflector wiring + disabled path
# ===========================================================================
def test_build_reflector_none_when_no_hub(tmp_path):
    from src.harness.config import from_dict

    data = {
        "runner": {"url": "https://x:8090", "cert_fingerprint": "aa" * 32},
        "auth": {"private_key_path": "/tmp/k"},
        "data_plane": {
            "ssh_alias": "nash",
            "runner_runs_dir": "/srv/runs",
            "mirror_remote_url": "u@h:/m.git",
        },
    }
    cfg = from_dict(data)
    assert hub.build_reflector(cfg, _db(tmp_path)) is None
