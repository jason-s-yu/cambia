"""
tests/test_harness_cli_render.py

CLI-level coverage for `cambia harness status` and `cambia harness nodes`
(cambia-1722): both call the client and print through src/harness/views.py.
Follows the direct-call + monkeypatch pattern of test_harness_spec.py's submit
tests (cli._build_client/_load_cfg stubbed with a fake client, no real
network).
"""

import json

import src.harness.cli as cli


class _FakeCfg:
    pass


def test_status_list_mode_renders_placement(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())

    class FakeClient:
        def list_jobs(self):
            return [
                {
                    "job_id": "j1",
                    "state": "running",
                    "node": "n-abc",
                    "placement": "placed",
                    "phase": "running",
                },
                {"job_id": "j2", "state": "queued", "placement": "reservoir_unavailable"},
            ]

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    cli.status(job_id=None, config=None)
    out = capsys.readouterr().out
    assert "j1" in out and "node=n-abc" in out and "placement=placed" in out
    assert "j2" in out and "reservoir_unavailable" in out


def test_status_single_job_names_reservoir_unavailable(monkeypatch, capsys):
    # AC1: harness status names reservoir_unavailable for a resume pinned to
    # an absent node, rendering the wire value verbatim (nothing sets the hold
    # yet per W3-T15b; this exercises the rendering path against a stubbed
    # wire payload shaped like handleGetJob's wrapped response).
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())

    class FakeClient:
        def get_job(self, job_id):
            return {
                "job": {
                    "job_id": job_id,
                    "state": "queued",
                    "placement": "reservoir_unavailable",
                    "placement_detail": ["next_eligible_at=2026-09-02T00:00:00Z"],
                },
                "resolved_sha": "a" * 40,
            }

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    cli.status(job_id="resume-job", config=None)
    out = capsys.readouterr().out
    assert "reservoir_unavailable" in out
    # The raw JSON dump is still printed underneath the summary line.
    payload = json.loads(out[out.index("{") :])
    assert payload["job"]["placement"] == "reservoir_unavailable"


def test_status_single_job_renders_executed_on(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())

    class FakeClient:
        def get_job(self, job_id):
            return {
                "job": {
                    "job_id": job_id,
                    "state": "running",
                    "node": "n-abc",
                    "placement": "placed",
                },
                "env": {"origin_host": "nash", "executed_on": "n-abc"},
            }

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    cli.status(job_id="j1", config=None)
    out = capsys.readouterr().out
    assert "executed_on=n-abc" in out


def test_nodes_cmd_renders_pool_state(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())

    class FakeClient:
        def nodes(self):
            return [
                {
                    "node_id": "n-9c1f2a7b0d44",
                    "node_epoch": 3,
                    "agent_version": "1.1.0",
                    "platform_tag": "linux-x86_64",
                    "slots": 2,
                    "slots_free": 1,
                    "kinds": ["train", "evaluate"],
                    "presence": "online",
                    "stale_seconds": 2,
                    "gate_report": {
                        "admit": False,
                        "slots_offered": 0,
                        "next_eligible_at": "2026-09-01T22:00:00-07:00",
                        "checks": [
                            {"gate": "windows", "ok": False, "detail": "outside window"}
                        ],
                    },
                    "leases": [
                        {
                            "job_id": "v0.4-prtcfr-r13",
                            "lease_id": "01J...",
                            "lease_epoch": 7,
                            "state": "active",
                            "phase": "running",
                        }
                    ],
                    "drained": True,
                    "breaker_trips": 1,
                    "degraded_jobs": {"job-a": 90},
                }
            ]

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    cli.nodes_cmd(config=None)
    out = capsys.readouterr().out
    assert "n-9c1f2a7b0d44" in out
    assert "admit=False" in out and "next_eligible_at=2026-09-01T22:00:00-07:00" in out
    assert "v0.4-prtcfr-r13" in out
    assert "hold: drain" in out
    assert "breaker_trips=1" in out
    assert "job-a" in out and "90s" in out


def test_nodes_cmd_empty_pool(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())

    class FakeClient:
        def nodes(self):
            return []

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    cli.nodes_cmd(config=None)
    out = capsys.readouterr().out
    assert "no nodes enrolled" in out
