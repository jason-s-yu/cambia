"""
src/harness/cli.py

Client-side `cambia harness` sub-app (cambia-256, design 2.5). Verbs: init,
submit, status, list-remote, logs, cancel, resume, pull, push-run, watch. The
data plane is ssh/rsync + git push; the control plane is the TLS-pinned,
JWT-authed runnerd API.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

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
    after: Optional[str] = typer.Option(
        None, "--after", help="Gate this job on another job id finishing first (cambia-352)"
    ),
    on_failure: Optional[str] = typer.Option(
        None,
        "--on-failure",
        help="Parent-failure policy: skip (default) | run | fail. Requires --after.",
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
    after_flag = after if isinstance(after, str) else None
    on_failure_flag = on_failure if isinstance(on_failure, str) else None
    try:
        if after_flag is not None or on_failure_flag is not None:
            # CLI flags override the spec-file keys for the dependency gate; re-parse
            # the raw mapping with them applied so the same validation runs.
            import yaml

            with open(spec_file, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if raw is None:
                raise HarnessSpecError(f"spec file is empty: {spec_file}")
            if after_flag is not None:
                raw["after"] = after_flag
            if on_failure_flag is not None:
                raw["on_failure"] = on_failure_flag
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
    from src.harness.pull import is_valid_run_name, watch

    cfg = _load_cfg(config)
    client = _build_client(cfg)
    coordinator, dest = _build_coordinator(cfg)
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
        )
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        dest.close()
