"""
tests/test_harness_client.py

Control-plane client coverage (design 2.4): submit/status/list/cancel against a
fake pinned HTTPS server, plus the 409/400/429/412 error mapping. The Bearer
token is asserted on the wire.
"""

import json

import pytest

from src.harness.client import HarnessAPIError, HarnessClient
from src.harness.transport import ControlPlaneTransport
from tests.harness_tls_util import RecordingServer, make_self_signed


def _client(srv, fp, token="minted-token"):
    transport = ControlPlaneTransport(srv.base_url, fp)
    return HarnessClient(transport, token_provider=lambda: token)


def test_submit_success_and_bearer_header(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/harness/jobs"): (
            201,
            {"job_id": "r1", "state": "queued", "queue_pos": 2},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        resp = client.submit({"kind": "train", "name": "r1", "commit": "a" * 40})
    assert resp == {"job_id": "r1", "state": "queued", "queue_pos": 2}
    req = srv.requests[-1]
    assert req["method"] == "POST"
    assert req["path"] == "/harness/jobs"
    assert req["headers"].get("Authorization") == "Bearer minted-token"
    assert json.loads(req["body"])["name"] == "r1"


def test_status_get_single(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/jobs/r1"): (200, {"job_id": "r1", "state": "running"})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        job = client.get_job("r1")
    assert job["state"] == "running"


def test_list_jobs_unwraps_envelope(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("GET", "/harness/jobs"): (200, {"jobs": [{"job_id": "a"}, {"job_id": "b"}]})
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        jobs = client.list_jobs()
    assert [j["job_id"] for j in jobs] == ["a", "b"]


def test_cancel_delete_with_flags(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("DELETE", "/harness/jobs/r1"): (200, {"canceled": True})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        client.cancel("r1", force=True, purge=False)
    req = srv.requests[-1]
    assert req["method"] == "DELETE"
    assert "force=true" in req["path"]


@pytest.mark.parametrize(
    "status,needle",
    [
        (409, "collision"),
        (400, "invalid"),
        (429, "cap"),
        (412, "preflight"),
    ],
)
def test_submit_error_mapping(tmp_path, status, needle):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("POST", "/harness/jobs"): (status, {"detail": "boom"})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(HarnessAPIError) as exc:
            client.submit({"kind": "train", "name": "r1"})
    assert exc.value.status == status
    assert needle in str(exc.value).lower()
    assert "boom" in str(exc.value)


def test_preflight_detail_carried(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/harness/jobs"): (
            412,
            {"checks": [{"name": "min_free_ram", "ok": False, "detail": "need 8G"}]},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(HarnessAPIError) as exc:
            client.submit({"kind": "train", "name": "r1"})
    assert exc.value.status == 412
    assert "min_free_ram" in str(exc.value)


def test_preflight_error_label_and_checks_both_rendered(tmp_path):
    # The runnerd 412 body carries an error label AND the failed checks; the
    # message must include both, not just the opaque label (live-fire find:
    # the first-key extraction dropped the checks when "error" was present).
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/harness/jobs"): (
            412,
            {
                "error": "preflight_failed",
                "checks": [
                    {"name": "xpu_render_node", "ok": False, "detail": "no node"}
                ],
                "override": "force (gpu_vram only)",
            },
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(HarnessAPIError) as exc:
            client.submit({"kind": "train", "name": "r1"})
    assert exc.value.status == 412
    msg = str(exc.value)
    assert "preflight_failed" in msg
    assert "xpu_render_node" in msg
    assert "no node" in msg


def test_health(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/health"): (200, {"queue_depth": 0, "jobs_running": 1})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        h = client.health()
    assert h["jobs_running"] == 1
