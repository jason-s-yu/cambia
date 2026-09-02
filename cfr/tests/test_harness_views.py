"""
tests/test_harness_views.py

Coverage for src/harness/views.py (cambia-1722): the client-side rendering of
JobView.node/placement/phase and NodeView (declaration, gate report, session
state, leases, drain hold, breaker, degraded marks). Pure-function tests over
dict literals shaped exactly like the wire (runnerd/harness/views.go's
JobView JSON tags, runnerd/nashnet/nodes.go's NodeRecord JSON tags, and
runnerd/nashnet/gates/evaluate.go's Report JSON tags).
"""

from src.harness.views import (
    format_job_placement,
    placement_label,
    render_breaker,
    render_declaration,
    render_degraded,
    render_hold,
    render_gate_report,
    render_job_row,
    render_leases,
    render_node_block,
    render_nodes,
    render_session,
)

# ---------------------------------------------------------------------------
# Job placement (JobView.node/placement/placement_detail/phase, D23)
# ---------------------------------------------------------------------------


def test_placement_label_known_reasons_are_human_readable():
    assert "waiting" in placement_label("waiting_for_node")
    assert "gated" in placement_label("waiting_for_node_gate")


def test_placement_label_names_reservoir_unavailable():
    # AC1: a resume pinned to an absent node must name reservoir_unavailable,
    # not just render an opaque enum value.
    label = placement_label("reservoir_unavailable")
    assert "reservoir_unavailable" in label
    assert "resume" in label.lower()


def test_placement_label_unknown_reason_falls_back_to_raw_value():
    assert placement_label("some_future_reason") == "some_future_reason"


def test_format_job_placement_none_for_untouched_job():
    assert format_job_placement({"job_id": "j1", "state": "queued"}) is None


def test_format_job_placement_placed_job():
    job = {
        "job_id": "j1",
        "state": "running",
        "node": "n-abc123",
        "placement": "placed",
        "phase": "running",
    }
    summary = format_job_placement(job)
    assert "node=n-abc123" in summary
    assert "placement=placed" in summary
    assert "phase=running" in summary


def test_format_job_placement_reservoir_unavailable_names_itself():
    job = {
        "job_id": "resume-job",
        "state": "queued",
        "placement": "reservoir_unavailable",
        "placement_detail": ["next_eligible_at=2026-09-02T00:00:00Z"],
    }
    summary = format_job_placement(job)
    assert "reservoir_unavailable" in summary
    assert "next_eligible_at=2026-09-02T00:00:00Z" in summary


def test_render_job_row_includes_placement_when_present():
    row = render_job_row(
        {
            "job_id": "j1",
            "state": "queued",
            "queue_pos": 3,
            "placement": "waiting_for_node",
        }
    )
    assert "j1" in row
    assert "queued" in row
    assert "q=3" in row
    assert "waiting_for_node" in row


def test_render_job_row_untouched_job_has_no_placement_suffix():
    row = render_job_row({"job_id": "j1", "state": "completed"})
    assert "node=" not in row
    assert "placement=" not in row


# ---------------------------------------------------------------------------
# Node rendering (GET /nashnet/nodes, D9/D46/D3/D45/D63/D8)
# ---------------------------------------------------------------------------


def test_render_declaration_summarizes_slots_and_devices():
    node = {
        "agent_version": "1.1.0",
        "platform_tag": "linux-x86_64",
        "slots": 2,
        "slots_free": 1,
        "kinds": ["train", "evaluate"],
        "capabilities": {
            "devices": [
                {"id": "cpu", "cores": 32, "ram_total_gb": 62.0},
                {"id": "cuda:0", "vram_total_gb": 24.0},
            ]
        },
    }
    lines = render_declaration(node)
    assert any("1/2 free" in l for l in lines)
    assert any("cuda:0" in l and "24.0GB" in l for l in lines)


def test_render_session_reports_presence_and_staleness():
    line = render_session({"presence": "online", "stale_seconds": 5})
    assert "presence=online" in line
    assert "unseen_for=5s" in line


def test_render_gate_report_empty():
    assert render_gate_report(None) == ["  gate: (no report yet)"]
    assert render_gate_report({}) == ["  gate: (no report yet)"]


def test_render_gate_report_renders_checks_with_next_eligible_at():
    report = {
        "admit": False,
        "slots_offered": 0,
        "next_eligible_at": "2026-09-01T22:00:00-07:00",
        "checks": [
            {"gate": "windows", "ok": False, "detail": "outside 22:00-07:00"},
            {
                "gate": "floors.free_vram_gb.cuda:0",
                "ok": False,
                "observed": 6.1,
                "required": 12,
                "next_eligible_at": "2026-09-01T23:00:00-07:00",
            },
        ],
    }
    lines = render_gate_report(report)
    assert "admit=False" in lines[0]
    assert "next_eligible_at=2026-09-01T22:00:00-07:00" in lines[0]
    joined = "\n".join(lines)
    assert "windows" in joined and "outside 22:00-07:00" in joined
    assert "floors.free_vram_gb.cuda:0" in joined
    assert "next_eligible_at=2026-09-01T23:00:00-07:00" in joined
    assert "observed=6.1" in joined and "required=12" in joined


def test_render_leases_none():
    assert render_leases({}) == ["  leases: none"]


def test_render_leases_renders_job_lease_epoch_state_phase():
    node = {
        "leases": [
            {
                "job_id": "v0.4-prtcfr-r13",
                "lease_id": "01J...",
                "lease_epoch": 7,
                "state": "active",
                "phase": "running",
                "attempt": 1,
            }
        ]
    }
    lines = render_leases(node)
    joined = "\n".join(lines)
    assert "v0.4-prtcfr-r13" in joined
    assert "lease=01J..." in joined
    assert "epoch=7" in joined
    assert "state=active" in joined
    assert "phase=running" in joined


def test_render_hold_names_which_hold_stands():
    assert "breaker" in render_hold({"hold": "breaker"})
    assert "drain" in render_hold({"hold": "drain", "drained": True})
    assert "none" in render_hold({"drained": False})
    assert "none" in render_hold({})
    # The drain flag is the fallback for a coordinator that predates the field.
    assert "drain" in render_hold({"drained": True})


def test_render_breaker_defaults_to_zero_trips():
    assert render_breaker({}) == "  breaker: breaker_trips=0"


def test_render_breaker_renders_trips():
    assert render_breaker({"breaker_trips": 2}) == "  breaker: breaker_trips=2"


def test_render_breaker_picks_up_any_future_breaker_field():
    # W3-T14 (a sibling ticket) fills the remaining-cooldown field; this
    # renderer must surface it with no code change the day it lands, per the
    # "render whatever the wire carries" instruction.
    node = {"breaker_trips": 3, "breaker_cooldown_remaining_seconds": 240}
    line = render_breaker(node)
    assert "breaker_trips=3" in line
    assert "breaker_cooldown_remaining_seconds=240" in line


def test_render_degraded_none():
    assert render_degraded({}) == "  degraded jobs: none"


def test_render_degraded_renders_remaining_hold():
    line = render_degraded({"degraded_jobs": {"job-a": 137}})
    assert "job-a" in line
    assert "137s" in line
    assert "remaining" in line


def test_render_node_block_combines_every_section():
    node = {
        "node_id": "n-9c1f2a7b0d44",
        "node_epoch": 3,
        "agent_version": "1.1.0",
        "platform_tag": "linux-x86_64",
        "slots": 2,
        "slots_free": 2,
        "kinds": ["train"],
        "presence": "online",
        "stale_seconds": 1,
        "gate_report": {"admit": True, "slots_offered": 2, "checks": []},
        "leases": [],
        "drained": False,
        "breaker_trips": 0,
        "degraded_jobs": {},
    }
    lines = render_node_block(node)
    joined = "\n".join(lines)
    assert "node n-9c1f2a7b0d44" in joined
    assert "epoch=3" in joined
    assert "declaration:" in joined
    assert "presence=online" in joined
    assert "gate: admit=True" in joined
    assert "leases: none" in joined
    assert "hold: none" in joined
    assert "breaker_trips=0" in joined
    assert "degraded jobs: none" in joined


def test_render_nodes_empty_pool():
    assert render_nodes([]) == ["no nodes enrolled"]


def test_render_nodes_separates_multiple_nodes_with_blank_line():
    nodes = [
        {"node_id": "node-a", "node_epoch": 1, "presence": "online"},
        {"node_id": "node-b", "node_epoch": 1, "presence": "stale"},
    ]
    lines = render_nodes(nodes)
    assert "" in lines
    assert any("node-a" in l for l in lines)
    assert any("node-b" in l for l in lines)
