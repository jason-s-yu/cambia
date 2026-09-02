"""
src/harness/views.py

Client-side rendering of nashnet placement and pool state (cambia-1722, design
D23/D45). Pure functions over the parsed JSON dicts the control plane returns:
the human-readable layer over runnerd/harness/views.go's JobView
(node/placement/placement_detail/phase) and QueueSnapshot.Nodes, and
runnerd/nashnet/nodes.go + nashnet_pool.go's NodeView (the D9 declaration, the
D46 gate_report, D3/D45 session state, leases, the coordinator-side drain hold,
the D63 breaker, and the D8 per-job degraded marks). Nothing here performs I/O,
so every renderer is unit-testable with a plain dict literal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Placement hold reasons a JobView carries once a ready job has matched no node
# past the unplaceable grace (runnerd/harness/nashnet_placement.go D14).
# reservoir_unavailable is the D12 resume-pin hold: the coordinator names the
# constant (PlacementReservoirGone) today, but nothing sets it until W3-T15b
# lands the resume re-place path, so this renders the moment the wire starts
# carrying it -- no CLI change required when it does.
_PLACEMENT_LABELS = {
    "waiting_for_node": "waiting for a matching node",
    "waiting_for_node_gate": "waiting: every matching node is gated",
    "reservoir_unavailable": "waiting: resume pinned to an absent node (reservoir_unavailable)",
    "seed_missing": "waiting: a required seed could not be resolved (seed_missing)",
    "placed": "placed",
}


def placement_label(placement: str) -> str:
    """Human text for a JobView.placement value. Falls back to the raw value
    for a hold reason this module does not recognize, so a future reason still
    renders instead of raising or going blank."""
    return _PLACEMENT_LABELS.get(placement, placement)


def format_job_placement(job: Dict[str, Any]) -> Optional[str]:
    """One-line placement summary for a JobView dict: node, placement (with its
    human label and, for an unplaceable job, the accumulated rejection detail),
    and phase. None for a job the pool never touched (no node/placement/phase
    field at all), which is every job on a zero-node daemon and every job
    predating the pool."""
    node = job.get("node")
    placement = job.get("placement")
    phase = job.get("phase")
    if not node and not placement and not phase:
        return None
    parts = []
    if node:
        parts.append(f"node={node}")
    if placement:
        label = placement_label(placement)
        if label != placement:
            parts.append(f"placement={placement} ({label})")
        else:
            parts.append(f"placement={placement}")
    detail = job.get("placement_detail")
    if detail:
        parts.append(f"detail=[{', '.join(detail)}]")
    if phase:
        parts.append(f"phase={phase}")
    return " ".join(parts)


def render_job_row(job: Dict[str, Any]) -> str:
    """One line for `harness status` (list mode) and `list-remote`: name,
    state, queue position, and the placement summary when the pool has
    touched this job."""
    name = job.get("job_id") or job.get("name") or "?"
    state = job.get("state") or job.get("status") or "?"
    qp = job.get("queue_pos")
    qp_str = f" q={qp}" if qp is not None else ""
    line = f"  {name:32s} {state}{qp_str}"
    placement = format_job_placement(job)
    if placement:
        line += f"  {placement}"
    return line


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{int(value)}s"
    except (TypeError, ValueError):
        return str(value)


def _render_device(d: Dict[str, Any]) -> str:
    bits = []
    if d.get("vram_total_gb") is not None:
        bits.append(f"{d['vram_total_gb']}GB")
    if d.get("cores") is not None:
        bits.append(f"{d['cores']}c")
    if d.get("ram_total_gb") is not None:
        bits.append(f"{d['ram_total_gb']}GB ram")
    suffix = f"({','.join(bits)})" if bits else ""
    return f"{d.get('id', '?')}{suffix}"


def render_declaration(node: Dict[str, Any]) -> List[str]:
    """The D9 declaration: the NodeRecord summary fields (agent_version,
    platform_tag, slots/slots_free, kinds) plus every device in the full
    declaration's `capabilities.devices` list, when the node carries one."""
    agent = node.get("agent_version") or "?"
    platform = node.get("platform_tag") or "?"
    slots = node.get("slots")
    slots_free = node.get("slots_free")
    if slots_free is not None and slots is not None:
        slot_str = f"{slots_free}/{slots} free"
    elif slots is not None:
        slot_str = str(slots)
    else:
        slot_str = "?"
    kinds = node.get("kinds") or []
    lines = [
        f"  declaration: agent={agent} platform={platform} slots={slot_str} kinds={list(kinds)}"
    ]
    caps = node.get("capabilities")
    devices = caps.get("devices") if isinstance(caps, dict) else None
    if devices:
        lines.append("    devices: " + ", ".join(_render_device(d) for d in devices))
    return lines


def render_session(node: Dict[str, Any]) -> str:
    """D3/D45 session/liveness rendering: presence (online/disconnected/
    stale/revoked), how long the node has been unseen, and its held
    events-poll deadline when it has one."""
    presence = node.get("presence") or "?"
    bits = [f"presence={presence}"]
    if node.get("stale_seconds") is not None:
        bits.append(f"unseen_for={_fmt_seconds(node['stale_seconds'])}")
    if node.get("connected_until"):
        bits.append(f"connected_until={node['connected_until']}")
    return "  session: " + " ".join(bits)


def render_gate_report(report: Optional[Dict[str, Any]]) -> List[str]:
    """D46 gate verdicts: the overall admit/slots_offered/next_eligible_at
    line, then each check with its own observed/required/next_eligible_at
    when the node reported one."""
    if not report:
        return ["  gate: (no report yet)"]
    head = (
        f"  gate: admit={report.get('admit')} slots_offered={report.get('slots_offered')}"
    )
    if report.get("next_eligible_at"):
        head += f" next_eligible_at={report['next_eligible_at']}"
    lines = [head]
    for check in report.get("checks") or []:
        bits = [f"ok={check.get('ok')}"]
        if check.get("observed") is not None:
            bits.append(f"observed={check['observed']}")
        if check.get("required") is not None:
            bits.append(f"required={check['required']}")
        if check.get("next_eligible_at"):
            bits.append(f"next_eligible_at={check['next_eligible_at']}")
        if check.get("detail"):
            bits.append(check["detail"])
        lines.append(f"    - {check.get('gate', '?')}: {' '.join(bits)}")
    return lines


def render_leases(node: Dict[str, Any]) -> List[str]:
    """The node's live leases (job, lease id, epoch, state, phase, attempt)."""
    leases = node.get("leases") or []
    if not leases:
        return ["  leases: none"]
    lines = ["  leases:"]
    for lease in leases:
        bits = [
            lease.get("job_id", "?"),
            f"lease={lease.get('lease_id', '?')}",
            f"epoch={lease.get('lease_epoch', '?')}",
            f"state={lease.get('state', '?')}",
        ]
        if lease.get("phase"):
            bits.append(f"phase={lease['phase']}")
        if lease.get("attempt"):
            bits.append(f"attempt={lease['attempt']}")
        lines.append("    - " + " ".join(bits))
    return lines


def render_drain(node: Dict[str, Any]) -> str:
    """The coordinator-side hold (an operator drain or the D63 breaker), not
    the node's own drain gate, which arrives only inside the gate report."""
    return f"  drain: {'yes' if node.get('drained') else 'no'}"


def render_breaker(node: Dict[str, Any]) -> str:
    """Every field the wire carries under a breaker_* key (D63): today that is
    only breaker_trips, so this renders the trip count; when W3-T14 adds a
    remaining-cooldown field it shows up automatically with no renderer
    change, per the ticket's "render whatever the wire carries" instruction."""
    bits = [f"{k}={node[k]}" for k in sorted(node) if k.startswith("breaker")]
    if not bits:
        bits = ["breaker_trips=0"]
    return "  breaker: " + " ".join(bits)


def render_degraded(node: Dict[str, Any]) -> str:
    """Per-job degraded marks (D8): a job this node nacked three times, held
    against it for the remaining hold duration."""
    degraded = node.get("degraded_jobs") or {}
    if not degraded:
        return "  degraded jobs: none"
    parts = [f"{job} ({_fmt_seconds(secs)} remaining)" for job, secs in degraded.items()]
    return "  degraded jobs: " + ", ".join(parts)


def render_node_block(node: Dict[str, Any]) -> List[str]:
    """The full multi-line block for one node: declaration, session state,
    gate verdicts, leases, drain hold, breaker state, and degraded marks."""
    node_id = node.get("node_id") or "?"
    epoch = node.get("node_epoch")
    lines = [f"node {node_id}  epoch={epoch}"]
    lines.extend(render_declaration(node))
    lines.append(render_session(node))
    lines.extend(render_gate_report(node.get("gate_report")))
    lines.extend(render_leases(node))
    lines.append(render_drain(node))
    lines.append(render_breaker(node))
    lines.append(render_degraded(node))
    return lines


def render_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
    """Every node in the pool, blank-line separated. `cambia harness nodes`
    prints this verbatim."""
    if not nodes:
        return ["no nodes enrolled"]
    out: List[str] = []
    for i, node in enumerate(nodes):
        if i:
            out.append("")
        out.extend(render_node_block(node))
    return out
