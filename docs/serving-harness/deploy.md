# Serving Harness: Deploy

This walks through `scripts/deploy-runnerd.sh`, the one-paste deployment
script that takes a bare Linux host meeting
[host-requirements.md](host-requirements.md) to a running `cambia-runnerd`
serving the control plane over TLS. It assumes you've already generated the
JWT keypair (see [keys-and-tls.md](keys-and-tls.md), or run
`cambia harness init` first).

## What it needs

```bash
RUNNER_SSH=user@192.0.2.10 \
RUNNER_HOST=192.0.2.10 \
scripts/deploy-runnerd.sh
```

| Env var | Required | Purpose |
|-|-|-|
| `RUNNER_SSH` | yes | ssh target for the runner host: an ssh-config alias or `user@host`. Everything privileged the script does happens by sshing to this target. |
| `RUNNER_HOST` | yes | the runner's IP or hostname, used as the TLS certificate's `subjectAltName` and for the final acceptance-probe URL. Can equal `RUNNER_SSH`'s host, or differ if ssh reaches it through an alias while clients reach it by IP. |
| `REPO` | no (defaults to `~/dev/cambia`) | local repo checkout, used to build `runnerd` and to locate `runnerd/deploy/cambia-runnerd.service`. |
| `JWT_PUB` | no (defaults to `~/.config/cambia/jwt_ed25519.pub`) | the raw 32-byte JWT public key to install on the runner. |
| `RUNNERD_BIN` | no | a pre-built `runnerd` binary to ship instead of building one in-script. |
| `RUNNERD_BUILD_SHA` | no | the commit sha stamped into that pre-built binary, so the stage-5 `build_commit` check can still pin the live daemon to the binary you shipped. Only meaningful with `RUNNERD_BIN`; the in-script build sets it itself. Without it, stage 5 only checks that `build_commit` is non-empty. |

The account behind `RUNNER_SSH` must have passwordless sudo on the runner
host for stages 1 and 4 (both are one-shot privileged setup: creating
`/srv/cambia` and installing the systemd unit). Nothing runs privileged on
the client workstation; everything the script does locally (stage 0, the
`scp` calls) runs as the invoking user.

## Stages

**Stage 0 -- build (local, unprivileged).** If `RUNNERD_BIN` isn't set, the
script builds `cambia-runnerd` from `$REPO/runnerd` with
`CGO_ENABLED=0 go build`, producing a static binary with no runtime cgo
dependency. It links with
`-ldflags "-s -w -X main.buildCommit=$(git -C $REPO rev-parse HEAD)"`, which
stamps the source commit into the binary; the daemon reports it as
`build_commit` on `GET /harness/health`, and stage 5 compares the two to
prove the daemon answering afterward is the binary this run shipped (a
redeploy that silently left the old process serving would otherwise look
identical from outside). A dirty working tree still stamps `HEAD`, and the
script says so. This is the only place the daemon's own build happens; the
per-job engine build the daemon later triggers on the runner is separate
and does need cgo there (see [host-requirements.md](host-requirements.md)).

**Stage 1 -- base layout (remote, privileged once).** Creates `/srv/cambia`
and chowns it to a dedicated `cambia` user, which must already exist on the
runner host. This is the only step in the whole script that touches
anything outside `/srv/cambia`.

**Stage 2 -- mirror and TLS cert (remote, unprivileged, as `cambia`).**
Creates the `keys/` and `runs/` subdirectories, initializes the bare git
mirror (`mirror.git`) if it doesn't already exist, and generates a
self-signed ed25519 TLS certificate (`keys/tls.key`, `keys/tls.crt`) with
`RUNNER_HOST` baked in as the SAN, if one doesn't already exist. Both steps
are idempotent: rerunning the script against an already-initialized runner
leaves the existing mirror and certificate alone rather than regenerating
them. Prints the certificate's SHA256 fingerprint at the end -- this is the
value that goes into the client's `harness.yaml` as
`runner.cert_fingerprint`; see [keys-and-tls.md](keys-and-tls.md) for the
full explanation of that trust model.

**Stage 3 -- ship artifacts (local -> remote).** Copies the JWT public key,
the `runnerd` binary, and the systemd unit file to the runner over `scp`,
then installs the public key into `/srv/cambia/keys/` (owned by `cambia`)
and the binary into `/usr/local/bin/cambia-runnerd` (owned by `root`,
mode 755). The daemon's own binary is the only thing installed with root
ownership; everything else it touches at runtime lives under `/srv/cambia`
owned by the unprivileged `cambia` user.

**Stage 4 -- job-preserving daemon swap (remote, privileged).** Installs
`runnerd/deploy/cambia-runnerd.service` to
`/etc/systemd/system/cambia-runnerd.service`, reloads systemd, `enable`s it
(so it starts on boot), then swaps the running daemon for the new binary
without killing the jobs it supervises, and verifies that nothing died. Its
`.d` drop-in directory (for example the `devices.conf` that grants GPU
access) is never written or removed; drop-ins are only listed in the log.
Every step is timestamped and mirrored to a per-deploy log on the runner.
The mechanics are in [the job-preserving swap](#the-job-preserving-swap-stage-4)
below.

**Stage 5 -- acceptance probes (local).** Hits the control plane from the
client over a TLS connection that skips certificate verification
(`curl -sk`, appropriate here since this is a one-time self-signed-cert
bootstrap probe, not a pinned production connection) and checks four
things:

- `GET /harness/health` with no credentials returns `200`. It is the one
  token-free route (read-only counters for LAN monitoring); the handshake
  succeeding at all confirms the stage-2 certificate is being served.
- `GET /harness/jobs` with no credentials returns `401`, confirming
  Bearer-JWT auth is enforced on the rest of the surface.
- The health body reports `restart_preserves_jobs: true`, i.e. the daemon
  now serving is a build whose `SIGTERM` detaches jobs instead of killing
  them. A `false` here means an older binary is still the live process.
- The health body's `build_commit` equals the sha stage 0 stamped (or is
  merely non-empty when `RUNNERD_BIN` was supplied without
  `RUNNERD_BUILD_SHA`), which pins the answering daemon to the binary this
  run shipped.

It then prints the pre/post job pid table read back from the runner and the
path of the stage-4 log. `DEPLOY OK` and exit `0` require all four checks;
anything else prints the specific failure and exits non-zero.

## The job-preserving swap (stage 4)

A training run on this harness lasts days to weeks, so a deploy that
restarts the daemon must leave the python jobs it supervises running. The
daemon and its jobs are deliberately decoupled for this: jobs are spawned
into their own process groups, their state lives in
`/srv/cambia/runs/<job>/process.json`, and a fresh daemon reattaches to
whatever is still alive during its startup reconcile. Stage 4 is the
sequence that makes a swap safe, and proves it afterward.

**Precondition: `KillMode=process`.** With systemd's default
`KillMode=control-group` (or the `mixed` the unit used to ship), systemd
`SIGKILL`s everything left in the service cgroup once the main process
exits, which is every job. The unit therefore sets `KillMode=process`, and
stage 4 reads back the merged loaded value with
`systemctl show cambia-runnerd -p KillMode --value` and aborts unless it is
exactly `process`. The check runs after `daemon-reload` (so it sees the unit
just installed, including any `.d` drop-in that overrides `KillMode`) and
before anything is stopped, so an abort leaves the running daemon and its
jobs untouched: fix the unit or the drop-in, then rerun. Never restart the
service by hand while the loaded `KillMode` is anything else.

**Pre-swap snapshot.** Stage 4 reads every
`/srv/cambia/runs/*/process.json` and records name, pid, and `start_ticks`
for each row whose status is `starting`, `running`, or `stopping`. Rows
carrying a `host` field are skipped: those are bounded-stale projections of
another machine's run, and their pid names a process in that host's pid
space, so probing it locally would be meaningless. Nothing under
`/srv/cambia/runs` is written or deleted at any point.

**Choosing the swap method.** Stage 4 asks the daemon that is currently
running what it does on `SIGTERM`, via the unauthenticated
`GET https://127.0.0.1:8090/harness/health`:

- `restart_preserves_jobs: true` means this daemon detaches on `SIGTERM`,
  so the swap is a plain `sudo systemctl restart cambia-runnerd`.
- Anything else (an older binary, a build started with
  `RUNNERD_KILL_JOBS_ON_STOP`, or health unreachable) means its `SIGTERM`
  handler would `SIGKILL` every job process group. That daemon is removed
  with `sudo systemctl kill --kill-whom=main --signal=SIGKILL
  cambia-runnerd` instead: `SIGKILL` cannot be handled, so the job-killing
  handler never runs, and `--kill-whom=main` plus `KillMode=process` keeps
  systemd from touching the children. `Restart=on-failure` brings the unit
  back within `RestartSec`; stage 4 waits up to 20 s and falls back to
  `systemctl start` if it does not. On systemd older than 252 the flag is
  spelled `--kill-who`, and stage 4 retries with that spelling if the first
  form is rejected. This path is a one-time migration: once the new binary
  is live, every later deploy takes the plain-restart path.
- If the unit is not active at all, stage 4 logs a cold start and issues
  `systemctl start`.

**Verification.** After the swap, stage 4 requires all of: the unit is
`active`; `MainPID` has changed (a restart that silently did nothing is a
failed deploy); and every pid from the pre-swap snapshot is still alive.
Liveness compares pid and `/proc/<pid>/stat` `starttime` against the
recorded `start_ticks`, so an unrelated process that inherited a recycled
pid does not read as a surviving job. A missing pid prints a
`MISSING AFTER SWAP` line naming each lost job (with the status its
`process.json` now carries), and the deploy exits non-zero without
attempting any repair; stage 5 is skipped. One caveat: a job that finishes
on its own inside the swap window also reads as `GONE`, so confirm against
`journalctl -u cambia-runnerd` and that run's `process.json` before treating
it as a swap failure.

**Deploy log.** The whole of stage 4 is timestamped and mirrored to
`/srv/cambia/deploy/deploy-<UTC-timestamp>.log` on the runner (the directory
is created with sudo and owned by `cambia`), with the pre/post job table
also written to `deploy-<UTC-timestamp>.jobs` beside it. The same output
streams to the client's terminal as the deploy runs, and stage 5 prints both
paths.

## Redeploy and upgrade

Rerunning the script against an already-deployed runner is safe and is the
supported way to upgrade the daemon binary or push a config change:

- Stage 1 (`mkdir -p` / `chown`) and stage 2 (mirror init, cert generation)
  are no-ops against existing state -- the mirror and TLS certificate are
  never regenerated once present, so redeploying never invalidates a
  client's pinned fingerprint or the pushed commit history in the mirror.
- Stage 3 always re-ships and reinstalls the binary and JWT public key, so
  a new build or a rotated JWT key take effect on every run.
- Stage 4 always reinstalls the unit file and swaps the daemon, so a
  redeploy always picks up the new binary and any unit-file changes.
  Rerunning it with a binary that is already current is safe: the daemon is
  swapped again, the jobs are unaffected, and the stage-5 `build_commit`
  check passes as before.

**What a redeploy means for in-flight jobs.** Nothing, when stage 4
succeeds: the running jobs keep running across the swap and the new daemon
reattaches to them at startup reconcile, which is what the pre/post pid
table at the end of the deploy demonstrates. That holds only through this
script, which asserts the `KillMode=process` precondition and picks a swap
method the currently-running daemon can survive; a hand-run
`systemctl restart` under the wrong `KillMode`, or against a daemon whose
`SIGTERM` still kills jobs, takes every live job with it. Deliberately
stopping the jobs too is a separate action: stop them through the API, or
start the daemon with `RUNNERD_KILL_JOBS_ON_STOP=1`. See
`runnerd/README.md`'s Shutdown and reconciliation section for the daemon
side of this.

To rotate the TLS certificate or JWT key deliberately (rather than as a
side effect of redeploy, since redeploy alone won't do it), remove the
relevant file under `/srv/cambia/keys/` on the runner before rerunning the
script, and update the client's `harness.yaml` with the new fingerprint
and/or key path afterward.

## Restricting the data-plane ssh keys (optional hardening)

`scripts/deploy-runnerd.sh` itself only uses `RUNNER_SSH`, an unrestricted
admin ssh target for one-shot privileged setup (stages 1 and 4). Day-to-day
`cambia harness` use is separate: it authenticates to the control plane with
the JWT keypair (see [keys-and-tls.md](keys-and-tls.md)) and talks to the
data plane over two independent client config keys, `data_plane.ssh_alias`
(rsync, runs dir) and `data_plane.mirror_remote_url` (git push, commit
mirror) -- see `cfr/config/harness.example.yaml`. Nothing requires those two
to share a host or key; a more restricted server-side setup binds each to
its own single-purpose ssh key rather than reusing the admin key:

- a **mirror-push key**, restricted in the runner's `authorized_keys` to
  `git-receive-pack '/srv/cambia/mirror.git'` for that exact command only,
  used solely for `harness submit`'s push;
- a **runs-dir key**, restricted to `rrsync /srv/cambia/runs`, used solely
  for `harness pull`/`push-run` rsync traffic.

Each key gets its own `~/.ssh/config` host alias on the client, and
`ssh_alias` / `mirror_remote_url` in `harness.yaml` point at the matching
alias. This is optional: a single shared admin-equivalent key also works,
it's just a larger blast radius if a client key leaks. See also the
optional `require_signed_commit` submit gate in
`cfr/config/harness.example.yaml`, which refuses to push an unsigned HEAD
commit when enabled; it's a client-local check, not part of this deploy
script, and does not require the split above.

## See also

- [host-requirements.md](host-requirements.md) -- what the runner host
  needs before this script can succeed
- [keys-and-tls.md](keys-and-tls.md) -- how the JWT keypair and TLS
  certificate are generated and why the trust model holds without a CA
- `runnerd/README.md` -- the daemon's own environment variables, API
  surface, and shutdown/reconciliation behavior in full
