# Serving Harness: Host Requirements

This is the provider-neutral contract for what a runner host must provide.
It applies to any machine intended to run `cambia-runnerd`: a spare server, a
cloud VM, a container host with a real init system, or a bare-metal box.
Nothing here assumes a particular cloud provider, virtualization layer, or
distro beyond "a Linux host running systemd."

Requirements are split into two tiers:

- **Hard** -- the daemon will not start, or will refuse work outright,
  without this. Missing a hard requirement is visible immediately (the
  process exits, or every job submission fails the same way).
- **Soft** -- the daemon starts fine; a specific job fails at the point it
  needs the missing piece. Soft requirements are still necessary for the
  harness to be useful, they just fail later and more narrowly.

## 1. Operating system and init

- Linux with systemd. The daemon ships as a `systemd` unit
  (`runnerd/deploy/cambia-runnerd.service`) that relies on
  `NoNewPrivileges`, `PrivateTmp`, `KillMode=mixed`, and `Restart=on-failure`
  for its process-sandboxing and crash-recovery story. **Hard** if you want
  the daemon supervised the way it's designed to run; `cambia-runnerd` is a
  single static Go binary and can technically run under another supervisor,
  but the unit file is the only supported path and the rest of this
  document assumes it.
- No container runtime is required to run the daemon itself; `runnerd`
  launches job processes directly as child processes of itself, not inside
  containers.

## 2. Required host software

All of the following must be on the `PATH` of the unprivileged user the
daemon runs as (see [section 3](#3-dedicated-user-and-base-directory)),
since job staging shells out to them directly rather than through a build
system.

| Tool | Used for | Tier |
|-|-|-|
| `git` | bare mirror creation, worktree checkout per job, ref bookkeeping | Hard |
| `uv` | per-job Python virtualenv creation and dependency sync (`uv venv`, `uv sync --frozen`, `uv lock --check`) | Hard |
| A Go toolchain satisfying `go1.26.0` | building the native engine library (`libcambia.so`) per job | Hard |
| A C compiler (`cc`/`gcc`) | the engine build is `CGO_ENABLED=1`; `go build -buildmode=c-shared` needs a working cgo toolchain | Hard |
| `python3` (or whatever interpreter `RUNNERD_PYTHON_BIN`-equivalent config points at) | the interpreter `uv venv --python` targets when building a job's environment | Soft -- if a compatible interpreter (>=3.11) isn't already present, `uv` will download and manage its own unless it's explicitly configured not to; see [section 4](#4-network-surface) |
| `openssl` | generating the self-signed TLS cert and computing its fingerprint at deploy time (see [keys-and-tls.md](keys-and-tls.md)) | Hard, but only at deploy/rotation time, not at daemon runtime |
| `sshd` reachable from the client workstation | data-plane transport: the client pushes commits via `git push` over ssh and pulls results back via `rsync` over ssh | Hard |
| `rsync` | the data-plane pull side; standard `rsync`-over-`ssh` requires the binary on both ends | Hard |

Building `cambia-runnerd` itself does **not** require any of this on the
runner: the daemon binary is built with `CGO_ENABLED=0` and is deployed as a
static binary (see [deploy.md](deploy.md)). The Go toolchain and C compiler
requirements above are for the *per-job* engine build the daemon triggers at
job-staging time, not for building the daemon.

**Go toolchain pin.** The engine build pins `GOTOOLCHAIN=go1.26.0` as an
explicit environment variable on every build invocation, so a different
system Go version cannot silently change what compiler produced the
artifact. If the host's `go` binary is not exactly that version, Go's
toolchain manager will attempt to fetch `go1.26.0` on first use (module
cache permitting, or over the network -- see
[section 4](#4-network-surface)). Pre-installing `go1.26.0` directly avoids
that fetch on the first job.

## 3. Dedicated user and base directory

The daemon expects to run as a **dedicated, unprivileged user** that owns a
single base directory (`RUNNERD_BASE_DIR`, default `/srv/cambia`) containing
the git mirror, worktrees, cached virtualenvs, cached engine builds, TLS/JWT
key material, and every run's output directory. This is **hard**: the
daemon itself never requests elevated privileges (`NoNewPrivileges=yes` in
the unit), and nothing in the ingest or process-management code assumes
root.

Provisioning that user and directory, and making sure the data-plane ssh
target lands in that user's account, is a one-time host setup step covered
in [deploy.md](deploy.md). The daemon does not create the user for you.

## 4. Network surface

Two planes, both already covered at the architecture level in
[architecture.md](architecture.md):

- **Control plane**: one HTTPS port (default `8090`, configurable via the
  unit's `--listen` flag) that the client workstation reaches directly.
  **Hard.** This is the only listener the daemon itself opens.
- **Data plane**: ssh (whatever port `sshd` is configured for, typically
  22) for `git push` (commit staging) and `rsync` (artifact pull-back).
  **Hard.** No separate listener is needed for this; it rides the host's
  existing ssh service.

**Soft, situational:** outbound internet access from the runner host. The
daemon's control and data planes are entirely inbound from the client's
perspective, but two staging steps can reach out if their target isn't
already cached locally: `uv` fetching Python interpreters or package wheels
it doesn't have cached, and Go fetching the pinned `go1.26.0` toolchain if
the host's installed Go doesn't already match. Pre-warming the `uv` cache
and installing the pinned Go version ahead of time removes this dependency
entirely; an air-gapped runner is workable as long as those caches are
seeded first.

## 5. Sizing guidance

CFR generation -- the harness's actual workload -- is **CPU-bound**, not
GPU-bound. v1 runners are CPU-only in practice: the daemon has a GPU-VRAM
preflight check, but it is exercised only when a job's rendered config sets
`device` to something other than `cpu`; a host with no GPU at all (no
`nvidia-smi`) passes that check automatically, treated as a CPU host. Size
the runner for CPU and RAM, not VRAM.

Default admission floors, both operator-configurable and **not**
overridable per-request over the API (raising them is a host-config change,
not something a client token can bypass):

| Floor | Env var | Default |
|-|-|-|
| Free RAM | `RUNNERD_MIN_FREE_RAM_GB` | 8 GB |
| Free disk (on the runs-directory filesystem) | `RUNNERD_MIN_FREE_DISK_GB` | 20 GB |

These are **hard** admission gates: a job submission that would violate
them is refused (`412`) rather than launched into contention. The disk
floor in particular should be sized generously above the default if you
plan to run several jobs' worth of history before pruning -- checkpoint and
reservoir artifacts accumulate per run and are not proactively trimmed by
the daemon itself.

CPU count matters for throughput, not admission: the daemon does not gate
on core count, but it does clamp each job's internal worker count to
`(host cores - 2)` so the daemon and host stay responsive under load. More
cores means more throughput per concurrent job, not a requirement floor.

## Optional: GPU support

By default a runner only accepts `device: cpu` jobs. Accepting `cuda` or
`xpu` jobs needs two things: the host has to provide the prerequisites
below, and the operator has to extend `RUNNERD_ALLOWED_DEVICES`
(comma-separated, default `cpu`) to include the device. A job submitted
with a `device` outside the runner's advertised list is rejected at
validation (`device_unsupported`) before it ever reaches the queue, and
that rejection is not forceable.

**NVIDIA (`cuda`).** A driver with `nvidia-smi` visible to the daemon's
unprivileged user. This is what the free-VRAM preflight shells out to; a
`cuda` job cannot pass preflight without it, even once the device is
allowlisted.

**Intel (`xpu`).** A `/dev/dri` render node accessible to the daemon's
user, plus the Intel compute runtime (the level-zero loader and an OpenCL
ICD) so torch's XPU backend can enumerate the device. For a containerized
runner this means passing the render node into the container with its
group permissions mapped through, not just bind-mounting the device file.
The free-VRAM preflight uses `xpu-smi` when it's present; render-node
presence alone admits the job if it isn't.

The RAM and disk floors in [section 5](#5-sizing-guidance) are unchanged
by any of this: GPU support adds a capability and preflight check, not a
different sizing regime.

## 6. Summary: what happens if a requirement is missing

| Missing | Effect | Tier |
|-|-|-|
| systemd | daemon still runs if launched another way, but loses the unit's sandboxing/restart behavior | Hard (for the supported path) |
| `git` / `uv` / Go toolchain / C compiler | daemon starts; every job submission fails at staging with a specific ingest error | Hard (job-time) |
| `openssl` | no TLS cert can be generated at deploy time; daemon cannot start without `RUNNERD_TLS_CERT`/`RUNNERD_TLS_KEY` | Hard (deploy-time) |
| `sshd` / `rsync` reachable | control plane still works; nothing can ever be submitted or pulled back | Hard |
| Dedicated user / base directory ownership | daemon fails to create its runs directory and exits at startup | Hard |
| RAM/disk below floor | daemon runs; job submissions against those floors are refused with `412` until resolved | Hard (per-submission) |
| GPU absent | no effect for v1 CPU-only jobs; the VRAM preflight treats a missing GPU as a pass | N/A |
| Outbound internet for `uv`/Go fetches | daemon runs; first job needing an uncached interpreter or toolchain fails or blocks until it can be fetched | Soft |

See [keys-and-tls.md](keys-and-tls.md) for the authentication and TLS
material a runner also needs, and [deploy.md](deploy.md) for the concrete
provisioning steps that satisfy this contract end to end.
