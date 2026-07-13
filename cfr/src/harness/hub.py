"""
src/harness/hub.py

Codebridge hub reflection for the `cambia harness` client (cambia-353). Job
lifecycle transitions (submit / running / terminal incl. skipped_dependency /
resume / purge) upsert a single keyed telemetry note into the Codebridge hub via
its inbound webhook, so the hub mirrors runner job/run state without a hub
credential ever living on the runner. Client-side only: the runner emits nothing.

Wire contract (copied EXACTLY from the hub source so a Codebridge-style sender
round-trips; see codebridge/serving/inbound_hooks.py and
codebridge/domain/webhooks.py):

  POST {hub.url}/api/hooks/{hub.slug}
  body    = serialize_payload(params)  # compact JSON, sorted keys, ensure_ascii=False
  headers:
    X-Codebridge-Signature: sha256=<hmac_sha256(secret, "{ts}.{body}")>  (hex)
    X-Codebridge-Timestamp: {ts}                                         (unix s)
    Content-Type: application/json

The signed string is "{timestamp}.{body}" (Stripe-style: the timestamp is inside
the MAC), keyed by the per-trigger secret. The hub recomputes the same MAC and
compares constant-time within a +/-300s replay window.

Payload = the manage_notes params the hub lifts after verification:
  {"ops": [{"op": "set", "item_handle": <hub item>, "key": <note key>,
            "role": "work", "body": <note body>}]}
One keyed note per job (key "harness:<origin_host>:<job_name>"), upserted on every
transition. Linked jobs attach to their hub_item; unlinked jobs to the config
collector item. Reflection NEVER advances a work item (set only).

Failure semantics: every hub call is best-effort. Any error (network, 4xx, 5xx,
missing secret) is logged and dropped; a hub outage or refusal NEVER affects
submit/pull/watch/job behavior or exit codes. The client-side reflection row
(run_db.harness_reflection) is written ONLY after a POST succeeds, so a dropped
post is re-tried by the next poll or drift-reconcile pass. There is no durable
retry spool.
"""

import hmac
import json
import logging
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, urlparse

from src.run_db import get_harness_reflection, upsert_harness_reflection

logger = logging.getLogger(__name__)

# Header names + signature prefix, copied verbatim from the hub source
# (codebridge/serving/inbound_hooks.py SIGNATURE_HEADER/TIMESTAMP_HEADER,
# codebridge/domain/webhooks.py _SIGNATURE_PREFIX). HTTP header names are
# case-insensitive; the canonical mixed-case form matches the hub's own sender.
SIGNATURE_HEADER = "X-Codebridge-Signature"
TIMESTAMP_HEADER = "X-Codebridge-Timestamp"
_SIGNATURE_PREFIX = "sha256="

_BANNER = "auto-reflected by cambia harness; telemetry only"

# Terminal job states (design 2.3 terminal set + run_db completions + the
# skipped-dependency and interrupted terminals). A state here means the job has no
# further transitions coming (resume starts a new launch, not a transition).
_TERMINAL_STATES = frozenset(
    {
        "stopped",
        "crashed",
        "canceled",
        "cancelled",
        "failed",
        "completed",
        "finished",
        "done",
        "skipped",
        "interrupted",
    }
)

# Monotonic rank for the client-side ordering guard (design 7). A poll-observed
# transition posts only when its rank does not regress below the last reflected
# state (a stale/out-of-order poll is dropped). Explicit user actions (resume,
# purge) bypass the rank check via force=True.
_PENDING_STATES = frozenset({"submitted", "created", "queued", "preparing"})
_RUNNING_STATES = frozenset({"starting", "running", "stopping"})


def _state_rank(state: Optional[str]) -> int:
    if not state:
        return 0
    s = state.lower()
    if s in _PENDING_STATES:
        return 1
    if s in _RUNNING_STATES:
        return 2
    if s == "purged":
        return 4
    if s in _TERMINAL_STATES:
        return 3
    return 2  # unknown live state: treat as running-tier, never a regression floor


def is_terminal(state: Optional[str]) -> bool:
    return bool(state) and state.lower() in _TERMINAL_STATES


# ---------------------------------------------------------------------------
# Signing scheme (copied EXACTLY from codebridge/domain/webhooks.py)
# ---------------------------------------------------------------------------


def serialize_payload(payload: Dict[str, Any]) -> str:
    """Canonical JSON body: compact + sorted keys so the signer and the hub
    receiver compute the same "{ts}.{body}" MAC input. ensure_ascii=False keeps
    non-ASCII as the UTF-8 bytes the HMAC and transport use."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sign(secret: str, timestamp: Any, body: str) -> str:
    """Hex HMAC-SHA256 of "{timestamp}.{body}" keyed by secret (hub scheme)."""
    signed_payload = f"{timestamp}.{body}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), signed_payload, sha256).hexdigest()


# ---------------------------------------------------------------------------
# Note construction
# ---------------------------------------------------------------------------


def note_key(origin_host: str, job_name: str) -> str:
    """The stable per-job note key: one keyed note per job lifecycle."""
    return f"harness:{origin_host}:{job_name}"


@dataclass
class MetricsTail:
    latest_iteration: Optional[int] = None
    retained_checkpoints: Optional[int] = None
    latest_mean_imp: Optional[float] = None


@dataclass
class ReflectionRecord:
    """The fields rendered into a job's reflected note body."""

    event: str
    job_id: str
    origin_host: str
    run_name: str
    state: str
    hub_item: Optional[str] = None
    kind: Optional[str] = None
    commit: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    last_error: Optional[str] = None
    metrics: Optional[MetricsTail] = None


def _exit_summary(rec: ReflectionRecord) -> Optional[str]:
    if rec.exit_code is None and not rec.last_error:
        return None
    parts: List[str] = []
    if rec.exit_code is not None:
        parts.append(f"exit_code={rec.exit_code}")
    if rec.last_error:
        parts.append(f"error={rec.last_error}")
    return ", ".join(parts)


def build_note_body(rec: ReflectionRecord) -> str:
    """Render the note body: banner first, then the job telemetry lines."""
    lines: List[str] = [_BANNER, ""]
    lines.append(f"event: {rec.event}")
    lines.append(f"job_id: {rec.job_id}")
    lines.append(f"origin_host: {rec.origin_host}")
    lines.append(f"run_name: {rec.run_name}")
    if rec.kind:
        lines.append(f"kind: {rec.kind}")
    lines.append(f"state: {rec.state}")
    if rec.commit:
        lines.append(f"commit: {rec.commit}")
    if rec.created_at:
        lines.append(f"created_at: {rec.created_at}")
    if rec.started_at:
        lines.append(f"started_at: {rec.started_at}")
    if rec.ended_at:
        lines.append(f"ended_at: {rec.ended_at}")
    summary = _exit_summary(rec)
    if summary:
        lines.append(f"exit: {summary}")
    m = rec.metrics
    if m is not None and (
        m.latest_iteration is not None
        or m.retained_checkpoints is not None
        or m.latest_mean_imp is not None
    ):
        if m.latest_iteration is not None:
            lines.append(f"latest_iteration: {m.latest_iteration}")
        if m.retained_checkpoints is not None:
            lines.append(f"retained_checkpoints: {m.retained_checkpoints}")
        if m.latest_mean_imp is not None:
            lines.append(f"latest_mean_imp: {m.latest_mean_imp:.4f}")
    lines.append(f"hub_item: {rec.hub_item or '(collector)'}")
    return "\n".join(lines)


def build_payload(item_handle: str, key: str, body: str) -> Dict[str, Any]:
    """The manage_notes params the hub lifts: a single set op onto item_handle."""
    return {
        "ops": [
            {
                "op": "set",
                "item_handle": item_handle,
                "key": key,
                "role": "work",
                "body": body,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Metrics tail (read from the client's authoritative run_db after a pull)
# ---------------------------------------------------------------------------


def read_metrics_tail(db, run_name: str) -> Optional[MetricsTail]:
    """Read a reconciled run's metrics tail from the client's run_db.

    Returns None when the run is not (yet) in the db (e.g. at submit, before any
    pull). latest_iteration = the max checkpoint iteration; retained_checkpoints =
    the count flagged is_retained; latest_mean_imp = the mean over MEAN_IMP
    baselines at the highest eval iteration that has any of them (best-effort).
    """
    try:
        row = db.execute("SELECT id FROM runs WHERE name=?", (run_name,)).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    run_id = row["id"]
    tail = MetricsTail()
    try:
        r = db.execute(
            "SELECT MAX(iteration) AS m FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()
        tail.latest_iteration = r["m"] if r is not None else None
        r = db.execute(
            "SELECT COUNT(*) AS c FROM checkpoints WHERE run_id=? AND is_retained=1",
            (run_id,),
        ).fetchone()
        tail.retained_checkpoints = r["c"] if r is not None else None
        tail.latest_mean_imp = _latest_mean_imp(db, run_id)
    except Exception:
        # Any read hiccup degrades to a partial tail, never a raised error.
        pass
    return tail


def _latest_mean_imp(db, run_id: int) -> Optional[float]:
    """Mean win-rate over the MEAN_IMP baselines at the latest eval iteration
    that carries any of them. Imported lazily so a submit (no metrics) never pays
    the evaluate_agents/torch import."""
    try:
        from src.evaluate_agents import MEAN_IMP_BASELINES
    except Exception:
        return None
    rows = db.execute(
        "SELECT iteration, baseline, win_rate FROM eval_results WHERE run_id=?",
        (run_id,),
    ).fetchall()
    by_iter: Dict[int, Dict[str, float]] = {}
    for r in rows:
        if r["win_rate"] is None:
            continue
        by_iter.setdefault(r["iteration"], {})[r["baseline"]] = r["win_rate"]
    for it in sorted(by_iter.keys(), reverse=True):
        present = [by_iter[it][b] for b in MEAN_IMP_BASELINES if b in by_iter[it]]
        if present:
            return round(sum(present) / len(present), 6)
    return None


# ---------------------------------------------------------------------------
# Best-effort webhook POST
# ---------------------------------------------------------------------------


def _send_post(
    url: str,
    body: str,
    headers: Dict[str, str],
    cert_fingerprint: Optional[str],
    timeout: float = 10.0,
):
    """POST body to url over HTTPS, optionally pinning the peer cert fingerprint
    (mirror of the RunnerConfig TLS pin). Returns (status, text). Plaintext is
    refused. Raises on transport failure (the caller drops it)."""
    import http.client

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"hub.url must be https://, got {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"hub.url has no host: {url!r}")
    port = parsed.port or 443
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    if cert_fingerprint:
        from src.harness.transport import _PinnedHTTPSConnection, normalize_fingerprint

        conn = _PinnedHTTPSConnection(
            host, port, normalize_fingerprint(cert_fingerprint), timeout
        )
    else:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    data = body.encode("utf-8")
    try:
        conn.request("POST", path, body=data, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", errors="replace")
        return resp.status, text
    finally:
        conn.close()


def post_note(
    hub_cfg,
    payload: Dict[str, Any],
    *,
    clock: Callable[[], float] = time.time,
    sender: Callable[..., Any] = _send_post,
) -> bool:
    """Sign + POST payload to the hub webhook. Best-effort: returns True on a 2xx,
    False on any non-2xx or exception (network, missing secret, bad config). Never
    raises -- a hub outage must not touch the caller."""
    try:
        secret = hub_cfg.resolve_secret()
        body = serialize_payload(payload)
        ts = int(clock())
        headers = {
            "Content-Type": "application/json",
            SIGNATURE_HEADER: f"{_SIGNATURE_PREFIX}{sign(secret, ts, body)}",
            TIMESTAMP_HEADER: str(ts),
        }
        url = hub_cfg.url.rstrip("/") + "/api/hooks/" + quote(hub_cfg.slug, safe="")
        status, text = sender(url, body, headers, hub_cfg.cert_fingerprint)
        if 200 <= status < 300:
            return True
        logger.warning(
            "hub reflection POST slug=%s -> HTTP %d (dropped): %s",
            hub_cfg.slug,
            status,
            text[:200],
        )
        return False
    except Exception as exc:
        logger.warning("hub reflection POST failed (dropped): %s", exc)
        return False


# ---------------------------------------------------------------------------
# The reflector
# ---------------------------------------------------------------------------


def _view_get(view: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = view.get(k)
        if v not in (None, ""):
            return v
    return None


class HubReflector:
    """Posts job lifecycle transitions to the hub, guarded by the client-side
    monotonic reflection store (run_db.harness_reflection).

    poster(payload) -> bool is the injection seam: production binds it to
    post_note over hub_cfg; tests inject a recorder. The db is the client's
    authoritative run_db (for the reflection store and the metrics tail).
    """

    def __init__(
        self,
        hub_cfg,
        origin_host: str,
        db,
        poster: Optional[Callable[[Dict[str, Any]], bool]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.hub = hub_cfg
        self.origin_host = origin_host
        self.db = db
        self.clock = clock
        self._poster = poster or (
            lambda payload: post_note(hub_cfg, payload, clock=clock)
        )

    # -- guard ------------------------------------------------------------
    def _should_post(
        self, last_state: Optional[str], new_state: str, force: bool
    ) -> bool:
        if force:
            return True
        if last_state is None:
            return True
        if new_state == last_state:
            return False  # idempotent: already reflected this state
        return _state_rank(new_state) >= _state_rank(last_state)  # drop regressions

    def _item_handle(self, hub_item: Optional[str]) -> str:
        return hub_item or self.hub.collector_item

    # -- core -------------------------------------------------------------
    def _post_record(self, rec: ReflectionRecord, force: bool) -> bool:
        last = get_harness_reflection(self.db, self.origin_host, rec.run_name)
        last_state = last["last_reflected_state"] if last is not None else None
        if not self._should_post(last_state, rec.state, force):
            return False
        # Resolve the hub item: the job's own hub_item, else the item a prior
        # transition already landed on, else the config collector item.
        prior_item = last["item_handle"] if last is not None else None
        item = self._item_handle(rec.hub_item or prior_item)
        payload = build_payload(
            item, note_key(self.origin_host, rec.job_id), build_note_body(rec)
        )
        if not self._poster(payload):
            return False  # dropped: leave the row stale so a later pass re-posts
        upsert_harness_reflection(
            self.db, self.origin_host, rec.run_name, rec.state, item
        )
        return True

    # -- public reflection entry points -----------------------------------
    def reflect_submit(self, payload: Dict[str, Any]) -> bool:
        """Reflect a submit off the accepted job payload (spec.to_payload)."""
        rec = ReflectionRecord(
            event="submit",
            job_id=payload.get("name", ""),
            origin_host=self.origin_host,
            run_name=payload.get("name", ""),
            state="submitted",
            hub_item=payload.get("hub_item"),
            kind=payload.get("kind"),
            commit=payload.get("commit"),
        )
        return self._post_record(rec, force=False)

    def reflect_view(
        self,
        view: Dict[str, Any],
        *,
        force: bool = False,
        with_metrics: bool = False,
        event: Optional[str] = None,
        state_override: Optional[str] = None,
    ) -> bool:
        """Reflect a runner JobView (from list_jobs). state_override lets an
        explicit action (resume) post a state the poll has not caught up to yet;
        with_metrics reads the reconciled metrics tail (used post terminal pull)."""
        job_id = _view_get(view, "job_id", "name")
        if not job_id:
            return False
        state = state_override or _view_get(view, "state", "status") or "unknown"
        ev = event or ("terminal" if is_terminal(state) else str(state))
        metrics = read_metrics_tail(self.db, job_id) if with_metrics else None
        rec = ReflectionRecord(
            event=ev,
            job_id=job_id,
            origin_host=self.origin_host,
            run_name=job_id,
            state=state,
            hub_item=_view_get(view, "hub_item"),
            kind=_view_get(view, "kind"),
            commit=_view_get(view, "commit"),
            created_at=_view_get(view, "created_at"),
            started_at=_view_get(view, "started_at"),
            ended_at=_view_get(view, "finished_at", "ended_at"),
            exit_code=view.get("exit_code"),
            last_error=_view_get(view, "last_error"),
            metrics=metrics,
        )
        return self._post_record(rec, force=force)

    def reflect_resume(self, view: Dict[str, Any]) -> bool:
        """Reflect an explicit resume (terminal -> running); bypasses the rank
        guard since resume legitimately moves a job backward in rank."""
        return self.reflect_view(
            view, force=True, event="resume", state_override="running"
        )

    def reflect_purge(self, job_id: str, hub_item: Optional[str] = None) -> bool:
        """Reflect an explicit purge (run dir removed). Minimal record: the run is
        gone, so there is no view to read; hub_item falls back to the stored item
        or the collector."""
        rec = ReflectionRecord(
            event="purge",
            job_id=job_id,
            origin_host=self.origin_host,
            run_name=job_id,
            state="purged",
            hub_item=hub_item,
        )
        return self._post_record(rec, force=True)

    def reconcile(self, views: List[Dict[str, Any]]) -> int:
        """Drift healer: re-post any job whose live state differs from (or was
        never) reflected. Purely client-side + idempotent (the scoped webhook
        credential cannot read hub notes). Returns the count posted."""
        posted = 0
        for view in views:
            try:
                if self.reflect_view(view, force=False):
                    posted += 1
            except Exception as exc:  # one bad view never stops the sweep
                logger.warning("drift reflect skipped a view: %s", exc)
        return posted


def build_reflector(cfg, db, clock: Callable[[], float] = time.time):
    """Build a HubReflector from a HarnessConfig, or None when hub reflection is
    not configured (absent hub section -> zero behavior change)."""
    if getattr(cfg, "hub", None) is None:
        return None
    return HubReflector(cfg.hub, cfg.data_plane.origin_host, db, clock=clock)
