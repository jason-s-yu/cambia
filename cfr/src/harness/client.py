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
