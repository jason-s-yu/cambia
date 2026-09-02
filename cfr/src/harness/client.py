"""
src/harness/client.py

High-level client for the runner control plane (cambia-256, design 2.4). Wraps
ControlPlaneTransport with per-invocation token minting and maps the runner's
status codes to typed errors:

  409 -> name collision (never forceable)
  400 -> invalid name / kind
  429 -> concurrency cap reached
  412 -> preflight failure (per-check detail carried through)
"""

from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from src.harness.transport import ControlPlaneTransport


class HarnessAPIError(Exception):
    """A control-plane call returned a non-success status."""

    def __init__(self, status: int, message: str, detail: Any = None):
        self.status = status
        self.detail = detail
        super().__init__(message)


class UnsupportedFeatureError(Exception):
    """A job spec needs a daemon capability the target does not advertise
    (design D30, cambia-1713). Raised locally, before the request reaches the
    control plane: a daemon with no `features` field on GET /harness/health
    advertises nothing, so a new client submitting e.g. a fan-in `after` list
    against it fails here with a named message instead of the field silently
    dropping on the wire."""


def required_features(payload: Dict[str, Any]) -> List[str]:
    """Daemon feature names `payload` (a POST /harness/jobs body) needs (D30).

    Only the `after` list shape is checked here: a bare string is the pre-r2
    wire shape and needs nothing. `requires`/`kind=measure` extend this list
    when those spec fields land (W1-T3, W1-T6); this function is the single
    place a future feature-gated field registers its requirement.
    """
    needed = []
    if isinstance(payload.get("after"), list):
        needed.append("fan-in")
    return needed


def check_feature_support(payload: Dict[str, Any], health: Dict[str, Any]) -> None:
    """Raise UnsupportedFeatureError when `payload` needs a feature `health`
    (a GET /harness/health body) does not advertise. A daemon with no
    `features` key advertises nothing, so every gated payload fails against it.
    """
    advertised = set(health.get("features") or [])
    missing = [f for f in required_features(payload) if f not in advertised]
    if missing:
        raise UnsupportedFeatureError(
            f"target daemon does not advertise required feature(s): {missing} "
            f"(advertised: {sorted(advertised)})"
        )


def parents_of(job: Dict[str, Any]) -> List[str]:
    """Return every parent of a JobView/queue-snapshot entry (design D29).

    `after_all` carries every parent when the daemon advertises `fan-in`
    (runnerd/harness/views.go viewAfterFields); a daemon or job with no
    dependency has neither key, and the pre-r2 shape carries a single parent in
    `after` alone. Preferring after_all keeps this correct against a fan-in
    job even if a caller reads a job dict that also carries the legacy
    single-parent `after` string alongside it.
    """
    after_all = job.get("after_all")
    if isinstance(after_all, list):
        return list(after_all)
    after = job.get("after")
    if isinstance(after, str) and after:
        return [after]
    return []


def _extract_detail(payload: Any) -> Any:
    if isinstance(payload, dict):
        parts = []
        for key in ("detail", "error", "message"):
            if payload.get(key) not in (None, ""):
                parts.append(str(payload[key]))
                break
        # A preflight 412 carries the failed checks as {name, ok, detail}
        # rows alongside the error label; render them or the caller only
        # sees an opaque "preflight_failed".
        checks = payload.get("checks")
        if isinstance(checks, list):
            rendered = "; ".join(
                f"{c.get('name', '?')}: {c.get('detail', '')}".rstrip(": ")
                for c in checks
                if isinstance(c, dict)
            )
            if rendered:
                parts.append(f"[{rendered}]")
        if parts:
            return " ".join(parts)
    return payload


def _map_error(status: int, payload: Any) -> HarnessAPIError:
    detail = _extract_detail(payload)
    known = {
        400: "invalid job spec (rejected by the runner)",
        409: "name collision: a job with this name already exists (never forceable)",
        412: "preflight failed on the runner",
        429: "runner at concurrency cap; retry when a slot frees",
        401: "unauthorized: token rejected (check key path / clock skew)",
        403: "forbidden",
        404: "no such job on the runner",
    }
    base = known.get(status, f"control-plane error (HTTP {status})")
    if detail not in (None, ""):
        message = f"{base}: {detail}"
    else:
        message = base
    return HarnessAPIError(status, message, detail)


class HarnessClient:
    """Typed wrapper over the control-plane transport.

    token_provider is called once per request and returns a freshly minted
    short-lived JWT (design 5.2). Tokens are never cached here.
    """

    def __init__(
        self,
        transport: ControlPlaneTransport,
        token_provider: Callable[[], str],
    ):
        self._t = transport
        self._token = token_provider

    def _call(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        ok=(200, 201),
    ) -> Any:
        token = self._token()
        status, payload = self._t.request(method, path, token, body)
        if status in ok:
            return payload
        raise _map_error(status, payload)

    def submit(self, payload: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        body = dict(payload)
        if force:
            body["force"] = True
        # Client-side capability gate (design D30): only when body actually
        # needs a gated feature does this cost a health round trip, so a
        # plain single-parent/no-dependency submit (the common case) pays
        # nothing extra.
        needed = required_features(body)
        if needed:
            check_feature_support(body, self.health())
        return self._call("POST", "/harness/jobs", body=body, ok=(201,))

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/harness/jobs/{quote(job_id, safe='')}")

    def list_jobs(self) -> List[Dict[str, Any]]:
        payload = self._call("GET", "/harness/jobs")
        if isinstance(payload, dict) and "jobs" in payload:
            return payload["jobs"]
        return payload if isinstance(payload, list) else []

    def cancel(self, job_id: str, force: bool = False, purge: bool = False) -> Any:
        qid = quote(job_id, safe="")
        params = []
        if force:
            params.append("force=true")
        if purge:
            params.append("purge=true")
        path = f"/harness/jobs/{qid}"
        if params:
            path += "?" + "&".join(params)
        return self._call("DELETE", path, ok=(200, 202, 204))

    def resume(self, job_id: str) -> Any:
        return self._call(
            "POST", f"/harness/jobs/{quote(job_id, safe='')}/resume", ok=(200, 201, 202)
        )

    def artifacts(self, job_id: str) -> Any:
        return self._call("GET", f"/harness/jobs/{quote(job_id, safe='')}/artifacts")

    def rundb_checkpoint(self, job_id: str) -> Any:
        """Ask the runner to WAL-checkpoint runs/<job_id>/run_db.sqlite before a
        pull (cambia-295 item 5), folding the WAL into the main db file so the
        synced file is current on its own. Callers treat this best-effort: a 404
        (no run_db yet, or an older daemon without the route) is a normal
        outcome, not a hard failure."""
        return self._call(
            "POST", f"/harness/jobs/{quote(job_id, safe='')}/rundb-checkpoint", ok=(200,)
        )

    def health(self) -> Dict[str, Any]:
        return self._call("GET", "/harness/health")
