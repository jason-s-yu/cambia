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

The account behind `RUNNER_SSH` must have passwordless sudo on the runner
host for stages 1 and 4 (both are one-shot privileged setup: creating
`/srv/cambia` and installing the systemd unit). Nothing runs privileged on
the client workstation; everything the script does locally (stage 0, the
`scp` calls) runs as the invoking user.

## Stages

**Stage 0 -- build (local, unprivileged).** If `RUNNERD_BIN` isn't set, the
script builds `cambia-runnerd` from `$REPO/runnerd` with
`CGO_ENABLED=0 go build`, producing a static binary with no runtime cgo
dependency. This is the only place the daemon's own build happens; the
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

**Stage 4 -- systemd unit (remote, privileged once).** Installs
`runnerd/deploy/cambia-runnerd.service` to
`/etc/systemd/system/cambia-runnerd.service`, reloads the daemon,
`enable`s it (so it starts on boot), and `restart`s it, printing a short
status snippet.

**Stage 5 -- acceptance probe (local).** Hits `https://$RUNNER_HOST:8090/harness/health`
with no credentials, over a TLS connection that skips certificate
verification (`curl -sk`, appropriate here since this is a one-time
self-signed-cert bootstrap probe, not a pinned production connection). A
correctly deployed daemon refuses this with `401` -- the TLS handshake
succeeding at all confirms the certificate installed in stage 2 is being
served, and the `401` confirms Bearer-JWT auth is enforced rather than the
health endpoint being open. The script prints `DEPLOY OK` and exits `0`
only when it sees exactly that `401`; anything else (connection refused,
`200`, a different error code) is treated as a failed deploy.

## Redeploy and upgrade

Rerunning the script against an already-deployed runner is safe and is the
supported way to upgrade the daemon binary or push a config change:

- Stage 1 (`mkdir -p` / `chown`) and stage 2 (mirror init, cert generation)
  are no-ops against existing state -- the mirror and TLS certificate are
  never regenerated once present, so redeploying never invalidates a
  client's pinned fingerprint or the pushed commit history in the mirror.
- Stage 3 always re-ships and reinstalls the binary and JWT public key, so
  a new build or a rotated JWT key take effect on every run.
- Stage 4 always reinstalls the unit file and does `systemctl restart`,
  so a redeploy always restarts the daemon, picking up the new binary
  and any unit-file changes.

**What a restart means for in-flight jobs.** `runnerd` has no graceful-stop
path: on `SIGTERM` (which `systemctl restart` sends) it immediately
`SIGKILL`s every running job's process group rather than trying to let them
finish, and recovery happens on the next boot's reconciliation sweep rather
than in place -- any job that was running gets marked crashed, and a train
job in that state can be resumed explicitly afterward from its last
checkpoint. A redeploy is therefore not silently safe to run against a
runner with live jobs; either wait for the queue to drain first, or expect
to resume afterward. See `runnerd/README.md`'s Shutdown and reconciliation
section for the full detail.

To rotate the TLS certificate or JWT key deliberately (rather than as a
side effect of redeploy, since redeploy alone won't do it), remove the
relevant file under `/srv/cambia/keys/` on the runner before rerunning the
script, and update the client's `harness.yaml` with the new fingerprint
and/or key path afterward.

## See also

- [host-requirements.md](host-requirements.md) -- what the runner host
  needs before this script can succeed
- [keys-and-tls.md](keys-and-tls.md) -- how the JWT keypair and TLS
  certificate are generated and why the trust model holds without a CA
- `runnerd/README.md` -- the daemon's own environment variables, API
  surface, and shutdown/reconciliation behavior in full
