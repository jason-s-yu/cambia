# runnerd

`runnerd` is the serving-harness runner daemon: a single Go binary that accepts train/evaluate/head-to-head/bench jobs against a pinned git commit, stages an isolated environment for each one, launches and supervises the process, and exposes status, logs, and artifacts over HTTPS. It never serves plaintext and never auto-launches queued or resumable work at startup; on boot it only reconciles inherited state and reports.

Core pieces:

- **Control plane**: HTTPS + Bearer-JWT REST API, plus two WebSocket streams (queue snapshots, per-job log tail).
- **Dispatcher**: bounded in-memory FIFO queue, a concurrency cap, and the job state machine (`queued` -> `preparing` -> running/terminal).
- **Ingest pipeline** (`ingest/`): per-job staging from a bare git mirror -- detached worktree, cached per-lock `uv` virtualenv, cached per-engine-tree `libcambia.so` build, rendered config, and a write-once provenance record (`env.json`).
- **procmgr** (`procmgr/`): process supervision, `process.json` state persistence, and preflight checks (disk, RAM, GPU VRAM).

## Build

```bash
cd runnerd
CGO_ENABLED=0 go build -o cambia-runnerd ./cmd/runnerd
```

The binary has no cgo dependencies of its own (the `libcambia.so` build it triggers per job runs as a subprocess, not linked in).

## Running

```bash
./cambia-runnerd --listen 0.0.0.0:8090
```

Flags:

| Flag | Default | Description |
|-|-|-|
| `--listen` | `$RUNNERD_LISTEN` or `127.0.0.1:8090` | control-plane listen address |

Four environment variables are required; the daemon exits immediately if any is unset.

## Environment variables

| Variable | Required | Default | Description |
|-|-|-|-|
| `RUNNERD_LISTEN` | no | `127.0.0.1:8090` | fallback for `--listen` when the flag is not passed |
| `RUNNERD_BASE_DIR` | no | `/srv/cambia` | base directory for ingest caches (git mirror, venvs, libcambia builds) |
| `RUNNERD_RUNS_DIR` | no | `/srv/cambia/runs` | root directory for per-job run directories |
| `RUNNERD_CFR_DIR` | no | `/srv/cambia/cfr` | working directory for the fixed-binary launch fallback (used only when a job's environment did not stage its own worktree/venv, e.g. a stub environment) |
| `RUNNERD_CAMBIA_BIN` | no | `cambia` | `cambia` executable name/path for the same fixed-binary fallback |
| `RUNNERD_ALLOWED_ORIGIN` | **yes** | -- | the single allowed `Origin` for browser-originated WebSocket connections; never a wildcard. The CLI client sends no `Origin` header and authenticates purely via Bearer token, so this gate only matters for a browser consumer. |
| `RUNNERD_JWT_PUBKEY` | **yes** | -- | path to the verify-only ed25519 public key (raw 32 bytes, not PEM) used to validate Bearer JWTs |
| `RUNNERD_TLS_CERT` | **yes** | -- | path to the TLS certificate; the control plane is HTTPS-only |
| `RUNNERD_TLS_KEY` | **yes** | -- | path to the TLS private key |
| `RUNNERD_MAX_CONCURRENT_JOBS` | no | `1` | concurrency cap; non-positive or unparseable values fall back to the default |
| `RUNNERD_MAX_QUEUE_DEPTH` | no | `128` | queue depth cap; `POST /harness/jobs` returns 429 once reached |
| `RUNNERD_MIN_FREE_RAM_GB` | no | `8.0` | admission preflight floor for available RAM |
| `RUNNERD_MIN_FREE_DISK_GB` | no | `20.0` | admission preflight floor for free disk on the runs directory's filesystem |
| `RUNNERD_ALLOWED_DEVICES` | no | `cpu` | comma-separated device capability gate (`cpu`, `cuda`, `xpu`); a job whose device is not in this set is rejected at submit as `device_unsupported`, not forceable |
| `RUNNERD_ORIGIN_HOST` | no | the daemon's hostname | overrides the `origin_host` value stamped into each job's `env.json` provenance record |
| `RUNNERD_KILL_JOBS_ON_STOP` | no | unset (false) | when true, `SIGTERM` kills the job process groups instead of detaching from them; see Restart semantics |

## TLS and authentication

The control plane serves HTTPS only; there is no plaintext fallback. Bearer tokens are ed25519 JWTs minted by the client and verified against `RUNNERD_JWT_PUBKEY` (runnerd holds the public half only and never signs). Verification requires:

- signing method `EdDSA`
- audience claim `aud == "cambia-runnerd"` (rejects a token minted for an unrelated purpose even if it happens to share the same keypair)
- a non-empty string `sub` claim
- up to 30 seconds of clock-skew leeway on `exp`/`nbf`/`iat`

## systemd unit

A unit file is provided at `deploy/cambia-runnerd.service`. Install it as `/etc/systemd/system/cambia-runnerd.service`, adjust the paths/listen address for your host, and fill in the placeholder values below:

```
ExecStart=/usr/local/bin/cambia-runnerd --listen 192.0.2.10:8090
Environment=RUNNERD_ALLOWED_ORIGIN=https://your-dashboard-host.example
```

Notable settings baked into the unit: `KillMode=process`, `PrivateTmp=no`, and `TimeoutStopSec=35` are what make a restart job-preserving (see Restart semantics below) -- do not change `KillMode` or `PrivateTmp` without reading that section; `NoNewPrivileges=yes` sandboxes the process; `Restart=on-failure` with a 5s backoff recovers from crashes, relying on reconcile-on-boot rather than in-place state recovery.

## API endpoints

| Method + path | Purpose |
|-|-|
| `POST /harness/jobs` | submit a job spec; `201` with `job_id`/`state`/`queue_pos`, or `400`/`409`/`412`/`429` on validation, collision, preflight, or queue-full |
| `GET /harness/jobs` | list every known job (queued, preparing, running, terminal) |
| `GET /harness/jobs/{id}` | full job view, resolved commit sha, and the `env.json` provenance summary if present |
| `DELETE /harness/jobs/{id}` | cancel (`?force=true` for SIGKILL instead of SIGINT + 30s grace) or, with `?purge=true`, remove a terminal job's run directory |
| `POST /harness/jobs/{id}/resume` | re-enqueue a terminal train job as a new launch from its rolling checkpoint |
| `GET /harness/jobs/{id}/artifacts` | manifest of every file under the job's run directory: relative path, size, sha256, mtime |
| `GET /harness/health` | `reconciled_at`, `jobs_running`, `queue_depth`, `free_ram_gb`, `free_disk_gb`, `restart_preserves_jobs`, `build_commit` |
| `GET /ws/harness/queue` | WebSocket: a queue/active snapshot on connect, then a fresh snapshot on every state change |
| `GET /ws/harness/jobs/{id}/logs` | WebSocket: backfills recent `training.log` lines, then streams new ones as they're written |

## Queue, admission, and state machine

Submission runs a fixed validation order: name shape -> name collision -> kind allowlist (`train`, `evaluate`, `head-to-head`, `bench`; v1 `train` only launches the `prtcfr` algorithm) -> device shape (`cpu`, `cuda`, or `xpu`) -> device capability gate (`RUNNERD_ALLOWED_DEVICES`) -> lexical path guards on `config`/`checkpoint_*`/`target`/`warm_start` (reject absolute paths and `..` segments) -> containment guards (checkpoints, an evaluate target, and a train job's `warm_start` must resolve inside the runs directory) -> resource preflights (disk, RAM, and a device-routed GPU check).

A `train` job may optionally set `warm_start` to another run's staged snapshot, relative to the runs directory (e.g. `prior-run/snapshots/prtcfr_snapshot_iter_530.pt`), to initialize from it instead of starting cold; `evaluate`, `head-to-head`, and `bench` forbid the field (`400 invalid_warm_start`), and a `warm_start` that resolves but names a file that does not exist is rejected at submit (`400 warm_start_not_found`) rather than failing the job after it launches.

Launch argv by kind: `train` runs `cambia train prtcfr --config <rendered> --run-name <name> --save-path runs/<name>`; `evaluate` runs `cambia evaluate <resolved target> --latest --games N --device D` (no `--config`: run-dir mode reads the target's own `config.yaml`); `head-to-head` runs `cambia head-to-head --config <rendered> --checkpoint-a <resolved> --checkpoint-b <resolved> --games N --device D` (`config` is required for this kind, unlike evaluate, since two bare checkpoints carry no rules/agent-type of their own); `bench` runs `cambia benchmark all --config <rendered> --output-dir runs/<name> --device D`.

The device-routed preflight: `cpu` skips the GPU check entirely; `cuda` runs the existing `nvidia-smi` VRAM check; `xpu` hard-requires an Intel GPU render node (`/dev/dri/renderD*`) and, when the `xpu-smi` binary is present, additionally checks free VRAM against the same floor -- absent `xpu-smi`, the render-node pass alone stands in and the result notes the VRAM floor as unverifiable. The render-node check is reported as `xpu_render_node`, a name outside the force matrix, so it can never be bypassed with `force=true`.

Job state is split across two owners: `queued` and `preparing` are runnerd-level and held only in memory, so a daemon restart loses them and they get swept as orphans on reconcile. `running`, `stopped`, and `crashed` live in each job's `process.json` and survive a restart. `canceled` and `failed` are runnerd-level but persisted into `process.json` so they also survive a restart.

`force` (set via the job spec, surfaced by the client's `--force`) can only override the `gpu_vram` preflight. Disk-space and RAM floors, and the device capability gate, are operator-set via the environment variables above and are never forceable over the API.

Each job's per-lock `uv` venv is additionally keyed by its device: a `cuda` or `xpu` job installs the matching `gpu`/`xpu` dependency extra into its own cached venv (suffixed `-gpu`/`-xpu`), while a `cpu` job's cache key is unchanged from before device support.

## Purge and resume

`DELETE .../{id}?purge=true` removes a terminal job's run directory to free its name for reuse, and drops the job's pinned git ref so the mirror can eventually garbage-collect the commit. It refuses (`409 not_terminal`) on a job that is still queued, preparing, or running.

`POST .../{id}/resume` requires both `snapshots/prtcfr_checkpoint.pt` and `resume_state.json` in the run directory; absent either, it returns `409 no_resumable_state`. Resume is a new launch, not a state transition -- it goes back through the queue -- and only train jobs are resumable in v1.

## Restart semantics

Restarting the daemon does not stop the jobs it supervises. Training runs here are measured in days or weeks, so a redeploy, a `systemctl restart`, or a unit reload leaves the job processes running and hands them to the next daemon incarnation.

**Signals.** On `SIGTERM` -- what systemd sends on every stop and restart -- runnerd logs how many job process groups it is leaving behind and exits without signalling them. On `SIGINT` (an interactive Ctrl-C on a foreground dev daemon, where an orphaned job would be a surprise) it SIGKILLs every job process group as before. Setting `RUNNERD_KILL_JOBS_ON_STOP=1` makes `SIGTERM` behave like `SIGINT`; `GET /harness/health` reports the active policy as `restart_preserves_jobs`, alongside a `build_commit` identifying the serving binary.

**The unit file is load-bearing.** `KillMode=process` is what keeps systemd from SIGKILLing the leftover jobs itself: under `KillMode=mixed` (or the `control-group` default) systemd kills everything remaining in the service cgroup once the main process exits, which is every job. `PrivateTmp=no` matters for the same reason: a private `/tmp` is a per-incarnation mount namespace whose backing directory systemd deletes on stop, which would leave every surviving job with a `/tmp` pointing at a deleted directory, so the daemon and its jobs share the host `/tmp` (the runner is a single-purpose unprivileged container).

**Reattach.** At startup the dispatcher adopts every local `process.json` row still live: each one claims a concurrency slot (so admission does not over-subscribe the runner with jobs this daemon did not launch) and, if it was submitted `exclusive`, re-raises the exclusive hold. A watcher polls each adopted job's PID and finalizes the row when the process exits, then frees the slot and re-dispatches, so a queued dependent gated on `after: <job>` still launches.

**Exit-status inference.** An adopted process was forked by a previous incarnation, so this daemon can never `waitpid` it and its true exit code is unrecoverable. The outcome is read off two witnesses, in order:

1. A row sitting at `stopping` means an operator stopped the job through the API (`DELETE .../{id}` records the request before signalling the process group). It is finalized `canceled` with no exit code -- the same terminal any other operator-requested stop produces -- rather than looking like a crash.
2. Otherwise the run's own journal (`runs/<name>/run_db.sqlite`) decides, read read-only: a `runs.status` of `completed` means the trainer finished cleanly, and the job is recorded `stopped` with `exit_code: 0` so a dependent's success gate fires. Anything else -- `interrupted`, `running`, no journal, an unreadable one -- is recorded `crashed` with **no** exit code rather than a fabricated one, the conservative verdict under every `on_failure` policy.

In every case `last_error` says the status was inferred and quotes what was observed, and dependents gate on the result by their `on_failure` policy exactly as they would on a normally-terminated parent.

Jobs are stopped deliberately, through the control-plane API (`DELETE .../{id}`), never as a side effect of a daemon restart.

## Reconciliation

On startup, before serving any request, the dispatcher reconciles inherited state: any `process.json` row left `running`/`starting`/`stopping` with no live PID is flipped to `crashed`; any `created` row with no live process and no pending queue entry is flipped to `failed` as orphaned. It never auto-launches a queued or resumable job -- an operator resumes explicitly via the API once the daemon is confirmed healthy.

## Related

- [../README.md](../README.md) and [../cfr/README.md](../cfr/README.md) -- project and client overview
- [../docs/serving-harness/architecture.md](../docs/serving-harness/architecture.md) -- full design
- [../cfr/config/harness-spec.example.yaml](../cfr/config/harness-spec.example.yaml) -- example client-side job spec
- [../cfr/config/harness.example.yaml](../cfr/config/harness.example.yaml) -- example client-side connection config
