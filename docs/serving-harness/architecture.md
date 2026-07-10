# Serving Harness: Architecture

The serving harness lets a developer offload CFR training and evaluation jobs from
a local workstation onto a dedicated runner host, while keeping that workstation's
run database as the single source of truth for results. It is a small distributed
system with a narrow job model, not a general remote-execution service: the set of
things a job can be is fixed, and the trust boundary is placed deliberately at the
one place arbitrary code is allowed to enter.

Code lives in two places: `runnerd/` (the Go daemon and client-side Go packages it
shares) and `cfr/src/harness/` (the Python client: CLI, transport, and reconciler).

## 1. System shape

Two independent planes connect the client workstation to the runner host:

- **Control plane** (HTTPS + WebSocket, Bearer JWT): job submit, status, cancel,
  resume, artifact manifest, health, and two streaming endpoints (queue
  transitions, log tail). Served by `runnerd`, a Go daemon running on the runner
  host.
- **Data plane** (ssh-based git push and rsync): the actual code that runs
  (a commit-pinned snapshot of the repository, pushed to a mirror on the runner
  host) and the actual bytes that come back (artifacts pulled down by rsync). No
  new network listener beyond ssh is needed for this plane.

End-to-end flow: the client's `cambia harness` CLI pushes a pinned commit to the
runner's git mirror, then calls the control-plane API to submit a job referencing
that commit. `runnerd` checks out an isolated worktree, prepares a cached
Python environment and engine build, renders and validates the job's config, and
launches it under a process supervisor. While the job runs, the client polls
status and can stream logs over the control plane. As the job produces results,
the client pulls them back over rsync and a reconciler replays them into the
client's local, authoritative run database.

```
client workstation                          runner host
-------------------                          -----------
cambia harness CLI  --- HTTPS+JWT (control) --->  runnerd
                     <--- WS (queue/logs)   ----     |
                     --- git push (data)    --->  git mirror --> worktree --> job process
                     <--- rsync (data)      ----  run directory (journal + artifacts)
reconciler -> local run_db.sqlite (authoritative)
```

## 2. Control-plane API

All routes require a Bearer JWT (see Security below) and are served over HTTPS
only.

| Method | Path | Purpose |
|-|-|-|
| POST | `/harness/jobs` | Submit a job spec; returns the job id and its queue position, or a structured rejection (name collision, invalid kind/name, queue-full, or a failed resource preflight). |
| GET | `/harness/jobs` | List jobs, live and terminal. |
| GET | `/harness/jobs/{id}` | Full state for one job: process status, queue metadata, resolved commit, and a summary of its provenance record. |
| DELETE | `/harness/jobs/{id}` | Cancel a queued or running job (with an optional forceful kill), or, for an already-terminal job, purge its run directory to free the name. |
| POST | `/harness/jobs/{id}/resume` | Resume a job from its last checkpoint, gated on resumable state actually being present on disk. |
| GET | `/harness/jobs/{id}/artifacts` | An artifact manifest (path, size, checksum, modified time) for the run directory; the manifest is fetched over this API, the bytes move over rsync. |
| GET | `/harness/health` | Daemon health snapshot: last reconciliation time, jobs currently running, queue depth, free RAM, free disk. |
| GET (WS) | `/ws/harness/queue` | A stream of queue and job-state transitions. |
| GET (WS) | `/ws/harness/jobs/{id}/logs` | A tail of the job's log file, with a short backfill on connect. |

## 3. Ingest pipeline

Getting a submitted job from "a commit reference" to "a running, isolated
process" goes through several caching and staging layers on the runner:

- **Repo mirror.** A single persistent bare git mirror lives on the runner host.
  Submitting a job pushes the pinned commit to a job-scoped ref
  (`refs/harness/<job-id>`); the runner verifies the pushed object is actually
  present before using it, and that ref is never force-pushed or reused, so a
  job's pinned commit cannot be silently swapped out from under it after
  submission. A fallback bundle-fetch transport exists for environments where a
  direct git push isn't available.
- **Per-run worktrees.** Each job gets a detached git worktree checked out at its
  pinned commit, sharing the mirror's object store (cheap to create, isolated
  per job). Worktrees are removed once a job reaches a terminal state and its
  artifacts have synced back to the client; a worktree that fails without a
  confirmed sync is kept briefly for post-mortem debugging under a bounded
  retention window before a periodic sweep reclaims it.
- **Python environment cache (uv).** Virtual environments are keyed by a hash of
  the dependency lock file plus interpreter and platform, and reused across any
  jobs that share that exact lock: a job never touches the network to reinstall
  dependencies it already has cached. Each device maps to its own dependency
  extra (`cpu`, `cuda`, or `xpu`, each pulling from its own wheel index) that
  is part of the cache key, so a `cpu` job and a `cuda` or `xpu` job never
  share a venv even against the same lock file. Jobs never use an editable install; the
  first-party Python package is made importable via `PYTHONPATH` pointed at the
  pinned worktree, with an interpreter-startup guard that hard-fails a job if
  its import of that package would resolve outside the pinned worktree (guarding
  against an ambient copy leaking in from elsewhere on the runner).
- **Engine build cache.** The native engine library is built once per engine-tree
  commit and cached by that hash; jobs sharing an engine tree share the same
  build artifact rather than rebuilding it. The Python FFI bridge checks this
  cache location first before any other resolution path.
- **Config render, on the runner.** A job's config is rendered and validated on
  the runner host itself, not trusted verbatim from the submitter. User-supplied
  overrides are applied first; a fixed set of harness-owned rails (device
  selection, worker-count clamping, output paths) is appended afterward so they
  always win, and a submitter attempting to set one of those harness-owned keys
  directly is rejected at validation rather than silently overridden. The
  rendered file is what actually runs, so the run directory always records
  exactly what happened.
- **Write-once provenance (`env.json`).** Every run directory gets a
  provenance record, written once at launch and never modified afterward:
  the exact commit, engine-tree hash, dependency lock hash, resolved package
  versions, interpreter and toolchain versions, platform, and creation time.
  This is the audit trail for reproducing or debugging a specific run later.

## 4. Job model

A job is one of a fixed, server-enforced set of kinds: `train` (v1 restricts this
to a single supported training algorithm), `evaluate`, `head-to-head`, and
`bench`. Anything outside that allowlist is rejected before any other
processing happens.

**Queue.** Submissions enter a bounded, in-memory FIFO queue; a submission past
the configured depth cap is rejected outright rather than silently queued
forever. A dispatcher admits queued jobs up to a concurrency cap and launches
them in submission order. Because the queue is in-memory, a daemon restart
clears it: jobs that were only queued (never launched) are not silently
resumed; see the reconciliation sweep below.

**State machine.** A job is admitted to disk immediately on submission, then
moves through preparing (worktree checkout, environment and build cache
resolution, config render) to running, and from there to one of several
terminal states: stopped (clean exit), crashed, canceled, or failed. Resuming a
job always produces a new launch rather than reopening the old one; a terminal
job's identity persists so its history stays attributable, but its execution
does not restart in place.

**Startup reconciliation.** On daemon restart, a sweep closes the crash window
left by an ungraceful shutdown: any job that was mid-transition with no live
process behind it is resolved to crashed, and any job that was still preparing
(with, by construction, an empty queue at startup) is resolved to failed. The
same sweep reclaims that job's worktree and any cache references it held. The
daemon never auto-resumes a job on its own; resuming is always an explicit,
operator-initiated action.

**Resume.** Resuming is gated strictly on the presence of the training
algorithm's own resume-state record and a checkpoint actually on disk for that
run: nothing is inferred or assumed; a job is either resumable or it isn't, and
the API says so.

**Purge.** A terminal job's run directory can be explicitly purged to free its
name for reuse. Purge refuses to act on any job that is not yet terminal, so a
live run's data can never be reclaimed out from under it.

**Validation order at submit.** Name well-formedness, then a name-collision
check (a job name can never be silently reused while another run of that name
exists), then the kind allowlist, then per-kind field scoping (for example, an
`evaluate` job requires a target and a `train` job forbids one; conversely, a
`train` job may optionally carry a `warm_start` referencing another run's
staged snapshot, which `evaluate`/`head-to-head`/`bench` forbid, and which is
containment-guarded the same way the evaluate target is), then path
guards on every spec field that names a file (see Security below), then
resource preflights (disk space, free memory, a device-capability check against
the runner's advertised device list, and a device-aware GPU preflight for
non-`cpu` jobs), and finally config render and validation. A job whose `device`
isn't in the runner's advertised list is rejected outright (`device_unsupported`,
never queued, not forceable); the GPU preflight itself is backend-specific -- a
free-VRAM check via `nvidia-smi` for `cuda`, via `xpu-smi` when present for
`xpu` -- and is skipped entirely for `cpu` jobs.

## 5. Artifact return and reconciliation

The runner never pushes; the client always pulls. A periodic pull runs on a
short interval for any run that's still live (an rsync delta transfer, so only
changed bytes move), and a terminal state transition triggers an immediate pull
with retry, so a client never has to wait out the full periodic interval to see
a run's final result.

**Wire format.** Rather than a lossy summary export, the reconciliation
transport is the runner's own local run-tracking database for that run
directory, synced down as-is and treated as the full-fidelity source for
replay. The reconciler opens the synced copy read-only and replays it into the
client's authoritative database through idempotent upserts keyed on stable
natural keys (run name; run + checkpoint iteration; run + iteration +
evaluation baseline), so a partial or repeated pull converges safely on the
next tick rather than duplicating or corrupting rows. A collision guard refuses
to replay a pulled run name that collides with an existing local (non-remote)
run of the same name, since same-named runs originating on different hosts are
not supported.

The origin field is the single ownership authority, and `harness push-run`
respects it: pushing a client-local run up to the runner for a remote resume or
evaluate stamps that run's origin to the runner once the transfer succeeds. The
runner now holds the live copy, so a later pull of the same run reconciles
(the guard sees a matching owner, not a local collision), the dashboard renders
it read-only, and a local start/resume on it is refused. A run that was never
pushed keeps a null origin and is still refused, so the guard only accepts a
same-name pull for a run the client actually handed to the runner.

**Dashboard integration.** Synced remote runs render read-only in the existing
run dashboard, alongside local runs, distinguished by an origin field rather
than by any special directory layout. Liveness for a remote run is always
answered by asking the runner over the control plane, never by a local process
check (a process id observed locally has no relationship to a process id on a
different host); the dashboard instead shows a bounded-staleness indicator
based on how long ago the last successful sync landed.

## 6. Security model

**Two-plane trust model.** The control plane accepts no arbitrary code: job
kinds are a fixed, server-defined allowlist, and every field that could name a
file or steer execution is validated and guarded before it is used for
anything (see Path guards, below). Arbitrary code enters the system through
exactly one place, by design: the git data plane. The commit that actually runs
on the runner is whatever was pushed to the mirror under a given job-scoped
ref. That makes pushing to the mirror, not calling the control-plane API,
the real trust boundary. Holding a control-plane token lets someone submit,
query, or cancel jobs against code that has already been pushed; it does not,
by itself, let them get new code executed.

**Authentication.** Control-plane access is a Bearer JWT, verified with a
verify-only ed25519 key: the runner holds only the public half and never a
signing key, so a compromised runner cannot mint tokens of its own. The
private signing key stays exclusively on the client side, which mints
short-lived, per-session tokens. Every valid token additionally carries a
runner-specific audience claim; a token minted for an unrelated purpose (say,
a user-facing session token that happens to reuse the same underlying keypair)
is rejected even though its signature would otherwise verify, because it
lacks that audience.

**Transport.** Every control-plane route is TLS-only; there is no plaintext
fallback. Because the deployment certificate is self-signed rather than issued
by a trusted CA, the client pins the runner's certificate fingerprint out of
band at deploy time and verifies it on every connection instead of walking a
certificate chain. WebSocket upgrades are restricted to the expected client
origin and ride the same pinned TLS connection that was already verified for
the initiating HTTPS request, so there is no separate unauthenticated upgrade
path.

**Path guards.** Every job-spec field that names a file (a config path, a
checkpoint reference, an evaluation target) is checked lexically first
(no absolute paths, no `..` segments anywhere in the path) and then, once its
expected base directory is known, resolved with symlinks followed and
re-checked for containment inside that base. The symlink-aware second pass
matters because a lexical check alone can't see a symlink planted inside
otherwise-legitimate, submitter-influenced content that points outside the
intended sandbox; resolving and re-checking closes that gap.

**The return path is untrusted input.** Even though results come back as
data rather than code, they are not trusted as such. The reconciler that
replays a synced run journal opens it read-only, applies a schema and field
whitelist rather than accepting arbitrary columns, range-checks numeric and
enum fields, and re-derives any checkpoint file path from the client's own
known local layout rather than trusting a path pulled from the runner (an
attacker-chosen absolute path pulled from the runner could otherwise reach a
deserializer that trusts its input). Run names are re-validated before being
joined into any local filesystem path, on both sides of the sync.
