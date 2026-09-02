"""
src/harness/cli.py

Client-side `cambia harness` sub-app (cambia-256, design 2.5). Verbs: init,
submit, status, list-remote, logs, cancel, resume, pull, push-run, watch,
nodes, node (cambia-1725: nashnet node enrollment listing and acting verbs --
grant/revoke/drain). The data plane is ssh/rsync + git push; the control
plane is the TLS-pinned, JWT-authed runnerd API. Spelling throughout: plural
`nodes` lists, singular `node` acts (design D3, cambia-1719).
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Union

import typer

harness_app = typer.Typer(
    help="Serving-harness control (init/submit/status/logs/pull) for the remote runner",
    no_args_is_help=True,
)

_CONFIG_OPT = typer.Option(
    None, "--config", help="Path to harness.yaml (else $CAMBIA_HARNESS_CONFIG or default)"
)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _load_cfg(config: Optional[str]):
    from src.harness import config as cfgmod

    try:
        return cfgmod.load(config)
    except cfgmod.HarnessConfigError as exc:
        _fail(str(exc))


def _token_provider(cfg):
    from src.harness.transport import load_ed25519_private_key, mint_token

    try:
        key = load_ed25519_private_key(cfg.auth.private_key_path)
    except Exception as exc:
        _fail(f"failed to load signing key {cfg.auth.private_key_path}: {exc}")

    def provider() -> str:
        return mint_token(key, cfg.auth.subject, cfg.auth.token_ttl_seconds)

    return provider


def _build_client(cfg):
    from src.harness.client import HarnessClient
    from src.harness.transport import ControlPlaneTransport

    transport = ControlPlaneTransport(cfg.runner.url, cfg.runner.cert_fingerprint)
    return HarnessClient(transport, _token_provider(cfg))


def _build_coordinator(cfg):
    from src.harness.pull import PullCoordinator, RsyncRunner
    from src.run_db import get_db

    runner = RsyncRunner(cfg.data_plane.ssh_alias, cfg.data_plane.runner_runs_dir)
    dest = get_db()
    coordinator = PullCoordinator(
        runner=runner,
        local_runs_dir=Path(cfg.sync.local_runs_dir),
        dest_conn=dest,
        origin_host=cfg.data_plane.origin_host,
        # Best-effort WAL-checkpoint request before each pull (cambia-295 item
        # 5); the coordinator tolerates a 404/405 or any failure on its own.
        checkpoint_client=_build_client(cfg),
    )
    return coordinator, dest


def _maybe_reflect(cfg, action) -> None:
    """Run action(reflector) if hub reflection is configured (cambia-353).

    Best-effort and fully contained: opens the run_db, builds the reflector, runs
    the action, and swallows every error so a hub outage or a config gap never
    changes the verb's behavior or exit code. A no-op when no [hub] section is set.
    """
    if getattr(cfg, "hub", None) is None:
        return
    try:
        from src.harness.hub import build_reflector
        from src.run_db import get_db

        dest = get_db()
        try:
            reflector = build_reflector(cfg, dest)
            if reflector is not None:
                action(reflector)
        finally:
            dest.close()
    except Exception:
        pass


def _git(args: List[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _repo_root() -> Path:
    try:
        return Path(_git(["rev-parse", "--show-toplevel"], Path.cwd()))
    except RuntimeError as exc:
        _fail(f"not inside a git repository: {exc}")


def _is_dirty(repo: Path) -> bool:
    return bool(_git(["status", "--porcelain"], repo))


def _is_commit_signed(sha: str, repo: Path) -> bool:
    """Return True iff `git verify-commit <sha>` succeeds in `repo`.

    Used only by the optional `require_signed_commit` pre-push gate
    (cambia-551 W2); a False here means the commit has no verifiable
    signature (unsigned, or signed by a key not in the local trust store),
    not that verification itself failed to run.
    """
    proc = subprocess.run(
        ["git", "verify-commit", sha], cwd=str(repo), capture_output=True, text=True
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# init (first-time client bootstrap: keypair + config scaffold)
# ---------------------------------------------------------------------------


@harness_app.command("init")
def init_cmd(
    runner_url: Optional[str] = typer.Option(
        None, "--runner-url", help="runnerd control-plane URL (placeholder if omitted)"
    ),
    ssh_target: Optional[str] = typer.Option(
        None,
        "--ssh-target",
        help="ssh alias for the runner (fills ssh_alias and origin_host; placeholder if omitted)",
    ),
    mirror_url: Optional[str] = typer.Option(
        None, "--mirror-url", help="git mirror remote URL (placeholder if omitted)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing key pair or harness.yaml"
    ),
):
    """Bootstrap a first-time client: generate the ed25519 signing key pair and
    scaffold ~/.config/cambia/harness.yaml. Non-interactive; flags only."""
    from src.harness.bootstrap import BootstrapError, run_init

    try:
        result = run_init(
            runner_url=runner_url,
            ssh_target=ssh_target,
            mirror_url=mirror_url,
            force=force,
        )
    except BootstrapError as exc:
        _fail(str(exc))

    typer.secho("harness init complete:", fg=typer.colors.GREEN)
    typer.echo(f"  private key: {result.private_key_path} (0600)")
    typer.echo(f"  public key:  {result.public_key_path}")
    typer.echo(f"  config:      {result.config_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  1. Ship the public key to the runner: {result.public_key_path}")
    typer.echo(
        "     (see docs/serving-harness/deploy.md and docs/serving-harness/keys-and-tls.md)"
    )
    typer.echo("  2. Fetch the runner's TLS fingerprint and pin it in the config:")
    typer.echo(
        "     openssl s_client -connect <runner-host>:8090 </dev/null 2>/dev/null "
        "| openssl x509 -fingerprint -sha256 -noout"
    )
    typer.echo(f"  3. Fill in any remaining placeholders in {result.config_path}.")


# ---------------------------------------------------------------------------
# submit (design 2.5: ssh push, then HTTPS POST)
# ---------------------------------------------------------------------------


@harness_app.command("submit")
def submit(
    spec_file: Path = typer.Argument(..., exists=True, help="Job spec YAML (design 2.6)"),
    force: bool = typer.Option(
        False, "--force", help="Set spec.force (only gpu_vram is forceable in v1)"
    ),
    after: Optional[List[str]] = typer.Option(
        None,
        "--after",
        help="Gate this job on another job id finishing first (cambia-352). "
        "Repeatable for a fan-in AND-join over 2+ parents (design D29, "
        "cambia-1713): the job launches only once every named parent has "
        "reached a clean terminal.",
    ),
    on_failure: Optional[str] = typer.Option(
        None,
        "--on-failure",
        help="Parent-failure policy: skip (default) | run | fail. Requires --after.",
    ),
    exclusive: bool = typer.Option(
        False,
        "--exclusive",
        help="Run this job alone: the runner launches it only into an idle daemon "
        "and holds every other job while it runs (cambia-655). For timing-"
        "sensitive measurement jobs.",
    ),
    require_node: Optional[str] = typer.Option(
        None,
        "--require-node",
        help="Pin this job to one enrolled node id (design D12), e.g. for "
        "affinity or measurement. Sets/overrides requires.node; other "
        "requires.* constraints still come from the spec file.",
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Push the pinned commit to the runner mirror and submit a job."""
    from src.harness.client import HarnessAPIError
    from src.harness.spec import HarnessSpecError, JobSpec, parse_spec_file

    cfg = _load_cfg(config)
    # isinstance(str) rather than `is not None`: a direct call (tests) leaves an
    # unpassed typer Option as its OptionInfo sentinel, which must count as "not
    # provided" just like a real CLI invocation's None default.
    # --after may repeat (design D29, cambia-1713): a real CLI invocation always
    # hands a list here (empty/None when the flag is never given); a direct call
    # (tests, and the pre-r2 single-parent callers) may still pass a bare string.
    # Exactly one value keeps the pre-r2 wire shape (a string) so a daemon that
    # predates fan-in sees an unchanged payload; 2+ values become the AND-join
    # list (cambia-1725 matches the wire shape client.parents_of already reads).
    if isinstance(after, list):
        after_values = [a for a in after if isinstance(a, str)]
    elif isinstance(after, str):
        after_values = [after]
    else:
        after_values = []
    after_flag: Optional[Union[str, List[str]]]
    if not after_values:
        after_flag = None
    elif len(after_values) == 1:
        after_flag = after_values[0]
    else:
        after_flag = after_values
    on_failure_flag = on_failure if isinstance(on_failure, str) else None
    # --exclusive is a set-only flag: it forces exclusivity on, additive over the
    # spec file (there is no way to unset a spec-file exclusive:true from the CLI).
    # The OptionInfo sentinel of a direct call is not a bool, so it counts as unset.
    exclusive_flag = bool(exclusive) if isinstance(exclusive, bool) else False
    require_node_flag = require_node if isinstance(require_node, str) else None
    try:
        if (
            after_flag is not None
            or on_failure_flag is not None
            or exclusive_flag
            or require_node_flag is not None
        ):
            # CLI flags override the spec-file keys; re-parse the raw mapping with
            # them applied so the same validation runs.
            import yaml

            with open(spec_file, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if raw is None:
                raise HarnessSpecError(f"spec file is empty: {spec_file}")
            if after_flag is not None:
                raw["after"] = after_flag
            if on_failure_flag is not None:
                raw["on_failure"] = on_failure_flag
            if exclusive_flag:
                raw["exclusive"] = True
            if require_node_flag is not None:
                # --require-node sets requires.node without discarding any other
                # requires.* constraint the spec file already carries.
                existing = raw.get("requires")
                requires = dict(existing) if isinstance(existing, dict) else {}
                requires["node"] = require_node_flag
                raw["requires"] = requires
            spec = JobSpec.parse(raw)
        else:
            spec = parse_spec_file(str(spec_file))
    except (HarnessSpecError, OSError) as exc:
        _fail(f"invalid spec: {exc}")

    repo = _repo_root()
    if _is_dirty(repo):
        _fail(
            "refusing to submit: working tree is dirty. The harness is commit-pinned "
            "(design 3.1); commit or stash first so run_db records exactly what ran."
        )
    try:
        sha = _git(["rev-parse", "HEAD"], repo)
    except RuntimeError as exc:
        _fail(str(exc))
    if spec.commit and spec.commit.lower() != sha.lower():
        _fail(f"spec commit {spec.commit} does not match HEAD {sha}")

    # Optional pre-push verify gate (cambia-551 W2), default off. The client
    # pushes the developer's existing HEAD commit as-is and creates no new
    # commit, so this only checks a signature that's already there (repo-config
    # concern, e.g. commit.gpgsign=true) -- it never signs anything itself.
    if getattr(cfg, "require_signed_commit", False) and not _is_commit_signed(
        sha, repo
    ):
        _fail(
            f"commit {sha} is not a verifiable signed commit; sign it or "
            "disable require_signed_commit"
        )

    job_id = spec.name
    ref = f"refs/harness/{job_id}"
    typer.echo(f"pushing {sha[:12]} -> {ref} on {cfg.data_plane.mirror_remote_url}")
    try:
        _git(["push", cfg.data_plane.mirror_remote_url, f"{sha}:{ref}"], repo)
    except RuntimeError as exc:
        _fail(f"mirror push failed: {exc}")

    client = _build_client(cfg)
    try:
        resp = client.submit(spec.to_payload(sha), force=force)
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"submit failed: {exc}")

    jid = resp.get("job_id", job_id) if isinstance(resp, dict) else job_id
    state = resp.get("state", "?") if isinstance(resp, dict) else "?"
    qpos = resp.get("queue_pos", "?") if isinstance(resp, dict) else "?"
    typer.secho(f"submitted {jid}: state={state} queue_pos={qpos}", fg=typer.colors.GREEN)

    # Reflect the submit into the Codebridge hub (cambia-353); best-effort.
    _maybe_reflect(cfg, lambda r: r.reflect_submit(spec.to_payload(sha)))


# ---------------------------------------------------------------------------
# status / list-remote (design 2.5: HTTPS GET)
# ---------------------------------------------------------------------------


def _render_job_row(job: dict) -> str:
    name = job.get("job_id") or job.get("name") or "?"
    state = job.get("state") or job.get("status") or "?"
    qp = job.get("queue_pos")
    qp_str = f" q={qp}" if qp is not None else ""
    return f"  {name:32s} {state}{qp_str}"


@harness_app.command("status")
def status(
    job_id: Optional[str] = typer.Argument(None, help="Job id (omit to list all)"),
    config: Optional[str] = _CONFIG_OPT,
):
    """Show one job's full state, or list all jobs when no id is given."""
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        if job_id:
            job = client.get_job(job_id)
            import json

            typer.echo(json.dumps(job, indent=2, default=str))
        else:
            jobs = client.list_jobs()
            if not jobs:
                typer.echo("no jobs on the runner")
                return
            for job in jobs:
                typer.echo(_render_job_row(job))
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"status failed: {exc}")


@harness_app.command("list-remote")
def list_remote(config: Optional[str] = _CONFIG_OPT):
    """List all jobs (live + terminal) on the runner."""
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        jobs = client.list_jobs()
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"list failed: {exc}")
    if not jobs:
        typer.echo("no jobs on the runner")
        return
    for job in jobs:
        typer.echo(_render_job_row(job))


# ---------------------------------------------------------------------------
# logs (design 2.5: HTTPS WS tail)
# ---------------------------------------------------------------------------


@harness_app.command("logs")
def logs(
    job_id: str = typer.Argument(..., help="Job id"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Stream new log lines"),
    config: Optional[str] = _CONFIG_OPT,
):
    """Tail a job's training.log over the pinned WS control plane."""
    from src.harness.transport import open_log_stream

    cfg = _load_cfg(config)
    token = _token_provider(cfg)()
    try:
        conn = open_log_stream(cfg.runner.url, cfg.runner.cert_fingerprint, token, job_id)
    except Exception as exc:
        _fail(f"log stream failed: {exc}")

    import websockets.exceptions as wse

    try:
        if follow:
            for message in conn:
                typer.echo(message, nl=False if str(message).endswith("\n") else True)
        else:
            # No follow: drain the backfill burst, then stop on idle.
            while True:
                try:
                    message = conn.recv(timeout=2.0)
                except TimeoutError:
                    break
                typer.echo(message, nl=False if str(message).endswith("\n") else True)
    except (KeyboardInterrupt, wse.ConnectionClosed):
        pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# cancel / resume (design 2.5: HTTPS DELETE / POST)
# ---------------------------------------------------------------------------


@harness_app.command("cancel")
def cancel(
    job_id: str = typer.Argument(..., help="Job id"),
    force: bool = typer.Option(
        False, "--force", help="SIGKILL a running job (skip grace)"
    ),
    purge: bool = typer.Option(
        False, "--purge", help="Remove a terminal job's run dir to free the name"
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Cancel a queued/running job (or purge a terminal one)."""
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        client.cancel(job_id, force=force, purge=purge)
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"cancel failed: {exc}")
    typer.secho(f"canceled {job_id}", fg=typer.colors.GREEN)

    # A purge removes the run dir; reflect it so the hub note records the terminal
    # purge (cambia-353). Plain cancels keep flowing through the watch poll.
    if purge:
        _maybe_reflect(cfg, lambda r: r.reflect_purge(job_id))


@harness_app.command("resume")
def resume(
    job_id: str = typer.Argument(..., help="Job id"),
    config: Optional[str] = _CONFIG_OPT,
):
    """Resume a terminal train job on the runner (run dir must be runner-local)."""
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        client.resume(job_id)
    except HarnessAPIError as exc:
        if exc.status in (404, 409, 412):
            _fail(
                f"{exc}. If this run's dir is not runner-local, push it first: "
                f"cambia harness push-run {job_id}"
            )
        _fail(str(exc))
    except Exception as exc:
        _fail(f"resume failed: {exc}")
    typer.secho(f"resumed {job_id}", fg=typer.colors.GREEN)

    # Reflect the resume (terminal -> running) into the hub (cambia-353); the job
    # view carries the hub link + kind/commit. Best-effort.
    _maybe_reflect(cfg, lambda r: r.reflect_resume(client.get_job(job_id)))


# ---------------------------------------------------------------------------
# pull / push-run / watch (design 2.5: ssh data plane + reconciler replay)
# ---------------------------------------------------------------------------


@harness_app.command("pull")
def pull(
    job_id: str = typer.Argument(..., help="Run/job id to pull"),
    all_checkpoints: bool = typer.Option(
        False,
        "--all-checkpoints",
        help="Pull the whole snapshots/ tree, not just retained",
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Pull one run's artifacts and reconcile them into the local run db."""
    cfg = _load_cfg(config)
    coordinator, dest = _build_coordinator(cfg)
    try:
        status = coordinator.pull_once(job_id, all_checkpoints=all_checkpoints)
    except Exception as exc:
        dest.close()
        _fail(f"pull failed: {exc}")
    dest.close()
    typer.secho(f"pulled {job_id}: synced status={status}", fg=typer.colors.GREEN)


@harness_app.command("push-run")
def push_run(
    job_id: str = typer.Argument(..., help="Local run id to push up for remote resume"),
    config: Optional[str] = _CONFIG_OPT,
):
    """Push a client-local run dir up to the runner (explicit locality transfer)."""
    cfg = _load_cfg(config)
    coordinator, dest = _build_coordinator(cfg)
    try:
        coordinator.push_run(job_id)
    except Exception as exc:
        dest.close()
        _fail(f"push-run failed: {exc}")
    dest.close()
    typer.secho(
        f"pushed {job_id} up to {cfg.data_plane.ssh_alias}", fg=typer.colors.GREEN
    )


@harness_app.command("watch")
def watch_cmd(
    all_checkpoints: bool = typer.Option(
        False, "--all-checkpoints", help="Widen every pull to the full snapshots/ tree"
    ),
    interval: Optional[int] = typer.Option(
        None, "--interval", help="Pull cadence seconds (default from config)"
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Run the foreground pull loop: periodic delta pulls + reconcile (design 4.1)."""
    from src.harness.client import HarnessAPIError
    from src.harness.hub import build_reflector
    from src.harness.pull import is_valid_run_name, watch

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    coordinator, dest = _build_coordinator(cfg)
    # Hub reflection (cambia-353): None when no [hub] section, so watch behaves
    # exactly as before. Shares the coordinator's run_db for the reflection store
    # and the reconciled metrics tail.
    reflector = build_reflector(cfg, dest)
    tick = interval if interval is not None else cfg.sync.interval_seconds

    def job_lister():
        # H1: names arrive here from the untrusted control plane. Drop any whose
        # name fails the canonical validator before they reach the pull loop, and
        # warn loudly; the pull coordinator re-checks as a hard backstop.
        try:
            jobs = client.list_jobs()
        except HarnessAPIError:
            return []
        safe = []
        for job in jobs:
            name = job.get("name") if isinstance(job, dict) else None
            if name is not None and not is_valid_run_name(name):
                typer.secho(
                    f"warning: skipping job with unsafe name {name!r}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
                continue
            safe.append(job)
        return safe

    typer.echo(
        f"watching {cfg.data_plane.origin_host} every {tick}s "
        f"(all_checkpoints={all_checkpoints}); Ctrl-C to stop"
    )
    try:
        watch(
            coordinator,
            job_lister,
            interval_seconds=tick,
            all_checkpoints=all_checkpoints,
            on_event=lambda msg: typer.echo(msg),
            reflector=reflector,
        )
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        dest.close()


@harness_app.command("reflect")
def reflect_cmd(config: Optional[str] = _CONFIG_OPT):
    """One-shot hub drift reconcile (cambia-353): re-post every job whose live
    runner state differs from (or was never) its last hub reflection. Idempotent;
    works purely from client-side state (the webhook credential cannot read hub
    notes). Best-effort per job; a hub error is logged and skipped."""
    from src.harness.client import HarnessAPIError
    from src.harness.hub import build_reflector
    from src.run_db import get_db

    cfg = _load_cfg(config)
    if getattr(cfg, "hub", None) is None:
        _fail("no [hub] section in harness config; reflection is disabled")
    client = _build_client(cfg)
    try:
        jobs = client.list_jobs()
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"list failed: {exc}")

    dest = get_db()
    try:
        reflector = build_reflector(cfg, dest)
        posted = reflector.reconcile(jobs)
    finally:
        dest.close()
    typer.secho(f"reflected {posted} job(s) to hub {cfg.hub.slug}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# nodes / node <verb> <id> (design D3/D9/D46/D60, cambia-1725): the nashnet
# node-enrollment listing and acting verbs. Spelling throughout: plural
# `nodes` lists, singular `node` acts.
# ---------------------------------------------------------------------------

node_app = typer.Typer(
    help="Act on one enrolled nashnet node: grant, revoke, drain",
    no_args_is_help=True,
)
harness_app.add_typer(node_app, name="node")


def _gate_summary(node: dict) -> str:
    report = node.get("gate_report")
    if not isinstance(report, dict):
        return "-"
    checks = report.get("checks")
    n_checks = len(checks) if isinstance(checks, list) else 0
    admit_str = "admit" if report.get("admit") else "hold"
    return f"{admit_str} ({n_checks} checks)"


def _render_node_row(node: dict) -> str:
    """One `cambia harness nodes` row: the enrollment-facing summary.

    Renders every top-level field the NodeView wire (GET /nashnet/nodes,
    runnerd/harness/nashnet_pool.go) carries -- state (presence), a gate-report
    summary, staleness, and the live lease count -- at the raw/compact
    altitude. Deeper interpretation (per-check `next_eligible_at`, remaining
    breaker cooldown, degraded-job detail) is the dashboard/placement view's
    job (cambia-1722), not duplicated here.
    """
    node_id = node.get("node_id") or "?"
    presence = str(node.get("presence") or "?")
    stale = node.get("stale_seconds")
    stale_str = f"{stale}s" if stale is not None else "?"
    slots_free = node.get("slots_free")
    slots = node.get("slots")
    slots_str = f"{slots_free}/{slots}" if slots is not None else "?"
    holds = []
    if node.get("revoked"):
        holds.append("revoked")
    if node.get("drained"):
        holds.append("drained")
    hold_str = "+".join(holds) if holds else "-"
    leases = node.get("leases")
    lease_str = str(len(leases)) if isinstance(leases, list) else "0"
    return (
        f"  {node_id:20s} {presence:12s} slots={slots_str:7s} stale={stale_str:6s} "
        f"hold={hold_str:14s} leases={lease_str:>2s} gate={_gate_summary(node)}"
    )


@harness_app.command("nodes")
def nodes_cmd(
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw GET /nashnet/nodes list as JSON"
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """List every enrolled nashnet node: id, session state, gate-report
    summary, staleness, and live lease count (design D3/D45/D46/D60)."""
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        nodes = client.list_nodes()
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"nodes failed: {exc}")

    if as_json:
        import json

        typer.echo(json.dumps(nodes, indent=2, default=str))
        return
    if not nodes:
        typer.echo("no enrolled nodes")
        return
    for node in nodes:
        typer.echo(_render_node_row(node))


@node_app.command("grant")
def node_grant(
    pubkey: Optional[str] = typer.Option(
        None,
        "--pubkey",
        help="Node's ed25519 public key, base64 (the value "
        "`cambia-runnerd node init` or a node's first `--role node` run prints)",
    ),
    pubkey_file: Optional[Path] = typer.Option(
        None,
        "--pubkey-file",
        help="Path to the node's public key file instead of --pubkey "
        "(raw 32 bytes, or base64 text)",
    ),
    slots: Optional[int] = typer.Option(
        None, "--slots", help="Clamp the node's declared max_slots"
    ),
    kinds: Optional[List[str]] = typer.Option(
        None, "--kinds", help="Clamp allowed job kinds; repeatable"
    ),
    device_kinds: Optional[List[str]] = typer.Option(
        None,
        "--device-kinds",
        help="Clamp allowed device kinds (cpu/cuda/xpu); repeatable",
    ),
    max_vram_gb: Optional[List[str]] = typer.Option(
        None,
        "--max-vram-gb",
        help="Clamp per-device-kind VRAM as DEVICE=GB, e.g. cuda=24; repeatable",
    ),
    max_cores: Optional[int] = typer.Option(None, "--max-cores", help="Clamp max_cores"),
    max_ram_gb: Optional[float] = typer.Option(
        None, "--max-ram-gb", help="Clamp max_ram_gb"
    ),
    max_disk_gb: Optional[float] = typer.Option(
        None, "--max-disk-gb", help="Clamp max_disk_gb"
    ),
    labels: Optional[List[str]] = typer.Option(
        None, "--labels", help="Clamp the grant's label set; repeatable"
    ),
    max_lease_bytes: Optional[int] = typer.Option(
        None, "--max-lease-bytes", help="Clamp max_lease_bytes"
    ),
    expires: str = typer.Option(
        "90d", "--expires", help="Grant validity: <n>[s|m|h|d] (default 90d)"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write the grant here (default ./<node_id>.grant)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing file at the write path"
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Mint an enrollment grant for a node's public key (design D60).

    Purely local and offline: signs with the operator's own key
    (auth.private_key_path) and writes the grant to disk. Nothing is sent
    over the network -- drop the file onto the coordinator's nodes directory
    yourself, over your own ssh/admin access, never a node's path (D61). See
    docs/serving-harness/node-enrollment.md for the full walkthrough.
    """
    from src.harness.nashnet import (
        NashnetError,
        decode_node_public_key,
        derive_node_id,
        grant_filename,
        load_node_public_key,
        mint_grant,
        parse_duration,
    )
    from src.harness.transport import load_ed25519_private_key

    if bool(pubkey) == bool(pubkey_file):
        _fail("exactly one of --pubkey or --pubkey-file is required")

    cfg = _load_cfg(config)
    try:
        key = load_ed25519_private_key(cfg.auth.private_key_path)
    except Exception as exc:
        _fail(f"failed to load signing key {cfg.auth.private_key_path}: {exc}")

    try:
        raw = (
            decode_node_public_key(pubkey)
            if pubkey
            else load_node_public_key(pubkey_file)
        )
    except NashnetError as exc:
        _fail(str(exc))

    caps: dict = {}
    if slots is not None:
        caps["max_slots"] = slots
    if kinds:
        caps["kinds"] = list(kinds)
    if device_kinds:
        caps["device_kinds"] = list(device_kinds)
    if max_vram_gb:
        vram: dict = {}
        for pair in max_vram_gb:
            if "=" not in pair:
                _fail(f"--max-vram-gb must be DEVICE=GB, got {pair!r}")
            dev, val = pair.split("=", 1)
            try:
                vram[dev] = float(val)
            except ValueError:
                _fail(f"--max-vram-gb value must be a number: {pair!r}")
        caps["max_vram_gb"] = vram
    if max_cores is not None:
        caps["max_cores"] = max_cores
    if max_ram_gb is not None:
        caps["max_ram_gb"] = max_ram_gb
    if max_disk_gb is not None:
        caps["max_disk_gb"] = max_disk_gb
    if labels:
        caps["labels"] = list(labels)
    if max_lease_bytes is not None:
        caps["max_lease_bytes"] = max_lease_bytes

    try:
        lifetime = parse_duration(expires)
    except NashnetError as exc:
        _fail(str(exc))

    try:
        token = mint_grant(key, raw, caps=caps or None, lifetime=lifetime)
    except NashnetError as exc:
        _fail(str(exc))

    node_id = derive_node_id(raw)
    out_path = Path(out) if out else Path.cwd() / grant_filename(node_id)
    if out_path.exists() and not force:
        _fail(f"{out_path} already exists (use --force to overwrite)")
    out_path.write_text(token + "\n", encoding="utf-8")

    typer.secho(f"minted grant for {node_id}", fg=typer.colors.GREEN)
    typer.echo(f"  wrote:   {out_path}")
    typer.echo(f"  expires: {expires}")
    typer.echo("")
    typer.echo(
        "Drop it onto the coordinator over your own ssh/admin access (never a "
        "node's ssh path -- design D61):"
    )
    typer.echo(
        f"  scp {out_path} <coordinator-ssh-target>:"
        f"$RUNNERD_NASHNET_NODES_DIR/{grant_filename(node_id)}"
    )


@node_app.command("revoke")
def node_revoke(
    node_id: str = typer.Argument(..., help="Node id (n-<12hex>)"),
    config: Optional[str] = _CONFIG_OPT,
):
    """Revoke one node's credential (design D60).

    The coordinator writes the node's `<node_id>.revoked` tombstone, bumps
    its node epoch (superseding every lease it held), and settles those jobs
    against whatever was already promoted. There is no grace period -- an
    operator revoke means the node is untrusted -- and no undo short of
    minting and dropping a fresh grant.
    """
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    try:
        resp = client.revoke_node(node_id)
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"revoke failed: {exc}")
    typer.secho(f"revoked {node_id}", fg=typer.colors.GREEN)
    revoked_leases = resp.get("revoked_leases") if isinstance(resp, dict) else None
    if revoked_leases:
        typer.echo(f"  settled leases: {', '.join(revoked_leases)}")


@node_app.command("drain")
def node_drain(
    node_id: str = typer.Argument(..., help="Node id (n-<12hex>)"),
    off: bool = typer.Option(
        False, "--off", help="Lift a prior drain hold instead of setting one"
    ),
    clear_breaker: bool = typer.Option(
        False,
        "--clear-breaker",
        help="Also reset the D63 circuit breaker's trip count and any per-job "
        "degraded marks (D8)",
    ),
    config: Optional[str] = _CONFIG_OPT,
):
    """Hold or release one node's placement eligibility (design D46).

    Two-way over one route: the default sets the drain hold, `--off` lifts
    it. `--clear-breaker` is independent of the hold direction and can
    accompany either.
    """
    from src.harness.client import HarnessAPIError

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    # isinstance guards: a direct call (tests) leaves an unpassed typer Option
    # as its OptionInfo sentinel, which must count as the flag's default.
    off_flag = bool(off) if isinstance(off, bool) else False
    clear_flag = bool(clear_breaker) if isinstance(clear_breaker, bool) else False
    drain = not off_flag
    try:
        client.drain_node(node_id, drain=drain, clear_breaker=clear_flag)
    except HarnessAPIError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"drain failed: {exc}")
    state = "drained" if drain else "undrained"
    suffix = ", breaker cleared" if clear_flag else ""
    typer.secho(f"{node_id}: {state}{suffix}", fg=typer.colors.GREEN)
