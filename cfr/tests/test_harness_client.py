"""
tests/test_harness_client.py

Control-plane client coverage (design 2.4): submit/status/list/cancel against a
fake pinned HTTPS server, plus the 409/400/429/412 error mapping. The Bearer
token is asserted on the wire.
"""

import json

import pytest

from src.harness.client import (
    HarnessAPIError,
    HarnessClient,
    UnsupportedFeatureError,
    check_feature_support,
    parents_of,
    required_features,
)
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
                "checks": [{"name": "xpu_render_node", "ok": False, "detail": "no node"}],
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


def test_rundb_checkpoint_success(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/harness/jobs/r1/rundb-checkpoint"): (
            200,
            {"job_id": "r1", "busy": 0, "log_frames": 0, "checkpointed": 3},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        resp = client.rundb_checkpoint("r1")
    assert resp["job_id"] == "r1"
    assert resp["checkpointed"] == 3
    req = srv.requests[-1]
    assert req["method"] == "POST"
    assert req["path"] == "/harness/jobs/r1/rundb-checkpoint"
    assert req["headers"].get("Authorization") == "Bearer minted-token"


def test_rundb_checkpoint_404_raises(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/harness/jobs/r1/rundb-checkpoint"): (404, {"error": "not_found"})
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(HarnessAPIError) as exc:
            client.rundb_checkpoint("r1")
    assert exc.value.status == 404


def test_health(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/health"): (200, {"queue_depth": 0, "jobs_running": 1})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        h = client.health()
    assert h["jobs_running"] == 1


def test_nodes_unwraps_envelope(tmp_path):
    # GET /nashnet/nodes (cambia-1722, D23): the operator listing route.
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("GET", "/nashnet/nodes"): (
            200,
            {"nodes": [{"node_id": "node-a"}, {"node_id": "node-b"}]},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        nodes = client.nodes()
    assert [n["node_id"] for n in nodes] == ["node-a", "node-b"]
    req = srv.requests[-1]
    assert req["method"] == "GET"
    assert req["path"] == "/nashnet/nodes"
    assert req["headers"].get("Authorization") == "Bearer minted-token"


def test_nodes_empty_pool(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/nashnet/nodes"): (200, {"nodes": []})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        nodes = client.nodes()
    assert nodes == []


# ---------------------------------------------------------------------------
# Client-side capability gate (design D30, cambia-1713): a spec needing a
# daemon feature (currently: an `after` list, the fan-in wire shape) is
# refused locally when the target's advertised features do not include it.
# ---------------------------------------------------------------------------


def test_required_features_empty_for_plain_payload():
    assert required_features({"kind": "train", "name": "r1"}) == []
    assert required_features({"kind": "train", "name": "r1", "after": "p1"}) == []


def test_required_features_fan_in_for_after_list():
    assert required_features({"kind": "train", "after": ["p1", "p2"]}) == ["fan-in"]


def test_check_feature_support_passes_when_advertised():
    check_feature_support({"after": ["p1", "p2"]}, {"features": ["fan-in"]})


def test_check_feature_support_raises_when_missing():
    with pytest.raises(UnsupportedFeatureError, match="fan-in"):
        check_feature_support({"after": ["p1", "p2"]}, {"features": []})


def test_check_feature_support_raises_when_features_key_absent():
    # A daemon with no `features` field advertises nothing (D30): the gate
    # must fail closed, not treat an absent key as "everything supported".
    with pytest.raises(UnsupportedFeatureError):
        check_feature_support({"after": ["p1", "p2"]}, {"queue_depth": 0})


def test_submit_after_list_requires_fan_in_feature(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    # No /harness/jobs route registered: if the gate leaked the request through
    # to the control plane, RecordingServer would 404/error instead of the
    # refusal happening locally, so this also proves the refusal never made
    # the POST.
    routes = {("GET", "/harness/health"): (200, {"queue_depth": 0})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(UnsupportedFeatureError, match="fan-in"):
            client.submit({"kind": "train", "name": "r1", "after": ["p1", "p2"]})
    assert all(r["method"] != "POST" for r in srv.requests)


def test_submit_after_list_allowed_when_feature_advertised(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("GET", "/harness/health"): (200, {"features": ["fan-in"]}),
        ("POST", "/harness/jobs"): (201, {"job_id": "r1", "state": "queued"}),
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        resp = client.submit({"kind": "train", "name": "r1", "after": ["p1", "p2"]})
    assert resp["job_id"] == "r1"


def test_submit_single_after_string_skips_health_round_trip(tmp_path):
    # The pre-r2 wire shape needs no capability check at all, so it must not
    # even call health -- only /harness/jobs is registered here.
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("POST", "/harness/jobs"): (201, {"job_id": "r1", "state": "queued"})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        resp = client.submit({"kind": "train", "name": "r1", "after": "p1"})
    assert resp["job_id"] == "r1"
    assert all(r["path"] != "/harness/health" for r in srv.requests)


# ---------------------------------------------------------------------------
# parents_of (design D29): reads a JobView/queue-snapshot entry's parent list.
# The dict literals below mirror runnerd/harness/views.go's JobView JSON
# shape exactly (job_id/state/after/after_all field names), the same shape
# runnerd/harness/views_after_test.go's TestJobViewRendersFanInAfterAll
# asserts the Go marshaler emits, so this is the cross-language half of that
# assertion (AC7).
# ---------------------------------------------------------------------------


def test_parents_of_prefers_after_all_for_fan_in_job():
    job_view = {
        "job_id": "fav-child",
        "state": "running",
        "after": "fav-p1",
        "after_all": ["fav-p1", "fav-p2", "fav-p3"],
    }
    assert parents_of(job_view) == ["fav-p1", "fav-p2", "fav-p3"]


def test_parents_of_falls_back_to_single_after_string():
    # A daemon that predates fan-in (or a single-parent job even on a fan-in
    # daemon, per viewAfterFields) carries only the legacy string field.
    job_view = {"job_id": "c1", "state": "queued", "after": "p1"}
    assert parents_of(job_view) == ["p1"]


def test_parents_of_empty_for_no_dependency():
    job_view = {"job_id": "c1", "state": "queued"}
    assert parents_of(job_view) == []


# ---------------------------------------------------------------------------
# nashnet node acting routes (design D3/D46/D60, cambia-1725): the
# operator-token client half of GET /nashnet/nodes/{id},
# POST /nashnet/nodes/{id}/drain, POST /nashnet/nodes/{id}/revoke.
# nodes() (the listing route) is covered above (cambia-1722).
# ---------------------------------------------------------------------------


def test_get_node_unwraps_envelope(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("GET", "/nashnet/nodes/n-a"): (
            200,
            {"node": {"node_id": "n-a", "revoked": False}},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        node = client.get_node("n-a")
    assert node == {"node_id": "n-a", "revoked": False}


def test_drain_node_sends_drain_and_clear_breaker(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/nashnet/nodes/n-a/drain"): (
            200,
            {"node": {"node_id": "n-a", "drained": True}},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        node = client.drain_node("n-a", drain=True, clear_breaker=True)
    assert node["drained"] is True
    req = srv.requests[-1]
    body = json.loads(req["body"])
    assert body == {"drain": True, "clear_breaker": True}


def test_drain_node_off_lifts_the_hold(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/nashnet/nodes/n-a/drain"): (
            200,
            {"node": {"node_id": "n-a", "drained": False}},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        client.drain_node("n-a", drain=False, clear_breaker=False)
    req = srv.requests[-1]
    assert json.loads(req["body"]) == {"drain": False, "clear_breaker": False}


def test_revoke_node_posts_to_revoke_route(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {
        ("POST", "/nashnet/nodes/n-a/revoke"): (
            200,
            {"node_id": "n-a", "revoked": True, "node_epoch": 2, "revoked_leases": []},
        )
    }
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        resp = client.revoke_node("n-a")
    assert resp["revoked"] is True
    req = srv.requests[-1]
    assert req["method"] == "POST"
    assert req["headers"].get("Authorization") == "Bearer minted-token"


def test_required_features_nashnet_pool_for_requires_block():
    assert required_features({"kind": "train", "requires": {"node": "n-a"}}) == [
        "nashnet-pool"
    ]
    assert required_features({"kind": "train"}) == []
    assert required_features({"kind": "train", "requires": {}}) == []


def test_submit_requires_block_needs_nashnet_pool_feature(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/health"): (200, {"queue_depth": 0})}
    with RecordingServer(cert, key, routes) as srv:
        client = _client(srv, fp)
        with pytest.raises(UnsupportedFeatureError, match="nashnet-pool"):
            client.submit({"kind": "train", "name": "r1", "requires": {"node": "n-a"}})
    assert all(r["method"] != "POST" for r in srv.requests)
