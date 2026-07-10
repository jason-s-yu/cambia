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
| `RUNNERD_ORIGIN_HOST` | no | the daemon's hostname | overrides the `origin_host` value stamped into each job's `env.json` provenance record |

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

Notable settings baked into the unit: `KillMode=mixed` and `TimeoutStopSec=35` back up the daemon's own signal handling (see Shutdown below); `NoNewPrivileges=yes` and `PrivateTmp=yes` sandbox the process; `Restart=on-failure` with a 5s backoff recovers from crashes, relying on reconcile-on-boot rather than in-place state recovery.

## API endpoints

| Method + path | Purpose |
|-|-|
| `POST /harness/jobs` | submit a job spec; `201` with `job_id`/`state`/`queue_pos`, or `400`/`409`/`412`/`429` on validation, collision, preflight, or queue-full |
| `GET /harness/jobs` | list every known job (queued, preparing, running, terminal) |
| `GET /harness/jobs/{id}` | full job view, resolved commit sha, and the `env.json` provenance summary if present |
| `DELETE /harness/jobs/{id}` | cancel (`?force=true` for SIGKILL instead of SIGINT + 30s grace) or, with `?purge=true`, remove a terminal job's run directory |
| `POST /harness/jobs/{id}/resume` | re-enqueue a terminal train job as a new launch from its rolling checkpoint |
| `GET /harness/jobs/{id}/artifacts` | manifest of every file under the job's run directory: relative path, size, sha256, mtime |
| `GET /harness/health` | `reconciled_at`, `jobs_running`, `queue_depth`, `free_ram_gb`, `free_disk_gb` |
| `GET /ws/harness/queue` | WebSocket: a queue/active snapshot on connect, then a fresh snapshot on every state change |
| `GET /ws/harness/jobs/{id}/logs` | WebSocket: backfills recent `training.log` lines, then streams new ones as they're written |

## Queue, admission, and state machine

Submission runs a fixed validation order: name shape -> name collision -> kind allowlist (`train`, `evaluate`, `head-to-head`, `bench`; v1 `train` only launches the `prtcfr` algorithm) -> lexical path guards on `config`/`checkpoint_*`/`target` (reject absolute paths and `..` segments) -> containment guards (checkpoints and an evaluate target must resolve inside the runs directory) -> resource preflights (disk, RAM, and GPU VRAM when `device != cpu`).

Job state is split across two owners: `queued` and `preparing` are runnerd-level and held only in memory, so a daemon restart loses them and they get swept as orphans on reconcile. `running`, `stopped`, and `crashed` live in each job's `process.json` and survive a restart. `canceled` and `failed` are runnerd-level but persisted into `process.json` so they also survive a restart.

`force` (set via the job spec, surfaced by the client's `--force`) can only override the `gpu_vram` preflight. Disk-space and RAM floors are operator-set via the environment variables above and are never forceable over the API.

## Purge and resume

`DELETE .../{id}?purge=true` removes a terminal job's run directory to free its name for reuse, and drops the job's pinned git ref so the mirror can eventually garbage-collect the commit. It refuses (`409 not_terminal`) on a job that is still queued, preparing, or running.

`POST .../{id}/resume` requires both `snapshots/prtcfr_checkpoint.pt` and `resume_state.json` in the run directory; absent either, it returns `409 no_resumable_state`. Resume is a new launch, not a state transition -- it goes back through the queue -- and only train jobs are resumable in v1.

## Shutdown and reconciliation

On SIGINT or SIGTERM, runnerd SIGKILLs every job process group immediately rather than attempting a graceful stop; recovery happens on the next boot's reconcile, not in place. The systemd unit's `KillMode=mixed` and `TimeoutStopSec=35` sit above this as a backstop.

On startup, before serving any request, the dispatcher reconciles inherited state: any `process.json` row left `running`/`starting`/`stopping` with no live PID is flipped to `crashed`; any `created` row with no live process and no pending queue entry is flipped to `failed` as orphaned. It never auto-launches a queued or resumable job -- an operator resumes explicitly via the API once the daemon is confirmed healthy.

## Related

- [../README.md](../README.md) and [../cfr/README.md](../cfr/README.md) -- project and client overview
- [../docs/serving-harness/architecture.md](../docs/serving-harness/architecture.md) -- full design
- [../cfr/config/harness-spec.example.yaml](../cfr/config/harness-spec.example.yaml) -- example client-side job spec
- [../cfr/config/harness.example.yaml](../cfr/config/harness.example.yaml) -- example client-side connection config
