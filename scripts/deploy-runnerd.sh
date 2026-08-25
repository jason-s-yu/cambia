#!/usr/bin/env bash
# scripts/deploy-runnerd.sh: one-paste deploy of the cambia serving harness
# runner daemon (cambia-runnerd) onto a runner host.
#
# Run from the client workstation. Stages 1 and 4 use sudo ON the runner host
# (the account behind RUNNER_SSH must have passwordless sudo there); nothing
# runs privileged on the client workstation.
#
# Stage 4 is a job-preserving swap: a redeploy replaces the daemon without
# killing the python training jobs it supervises. See
# docs/serving-harness/deploy.md for the full rationale.
#
# Required env:
#   - RUNNER_SSH: ssh target for the runner host, e.g. an ssh-config alias or
#     user@host.
#   - RUNNER_HOST: the runner host IP or hostname (used for the TLS cert
#     subjectAltName and the acceptance-probe URL). May be the same value as
#     RUNNER_SSH's host, or different if ssh reaches it by alias.
#
# Optional env:
#   - RUNNERD_BIN: ship this pre-built binary instead of building one.
#   - RUNNERD_BUILD_SHA: the commit sha baked into that pre-built binary, so
#     stage 5 can still assert the live daemon is the one just shipped. Only
#     meaningful together with RUNNERD_BIN; the in-script build sets it itself.
#   - REPO, JWT_PUB: see below.
#
# Prereqs:
#   - runnerd binary at $RUNNERD_BIN (built in-script if unset; static, CGO_ENABLED=0)
#   - JWT public key at ~/.config/cambia/jwt_ed25519.pub (raw 32-byte ed25519;
#     see docs/serving-harness/keys-and-tls.md)
#   - repo checkout at $REPO with runnerd/deploy/cambia-runnerd.service
#   - python3 on the runner host (job snapshot, pid liveness, health JSON) and
#     on this client (stage-5 health JSON); jq is not assumed on either
set -euo pipefail

RUNNER_SSH=${RUNNER_SSH:?set RUNNER_SSH to the ssh target for the runner host, e.g. RUNNER_SSH=user@192.0.2.10 or an ssh-config alias}
REPO=${REPO:-$HOME/dev/cambia}
JWT_PUB=${JWT_PUB:-$HOME/.config/cambia/jwt_ed25519.pub}
RUNNER_HOST=${RUNNER_HOST:?set RUNNER_HOST to the runner host IP or hostname, e.g. RUNNER_HOST=192.0.2.10}

BUILD_SHA=${RUNNERD_BUILD_SHA:-}
DEPLOY_TS=$(date -u +%Y%m%dT%H%M%SZ)
DEPLOY_LOG="/srv/cambia/deploy/deploy-${DEPLOY_TS}.log"
DEPLOY_JOBS="/srv/cambia/deploy/deploy-${DEPLOY_TS}.jobs"

if [ -z "${RUNNERD_BIN:-}" ]; then
  echo "== stage 0: building cambia-runnerd (RUNNERD_BIN unset)"
  BUILD_SHA=$(git -C "$REPO" rev-parse HEAD)
  if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    echo "   note: $REPO is dirty; the binary is stamped with HEAD ($BUILD_SHA) anyway"
  fi
  RUNNERD_BUILD_DIR=$(mktemp -d)
  (cd "$REPO/runnerd" && CGO_ENABLED=0 go build \
    -ldflags "-s -w -X main.buildCommit=$BUILD_SHA" \
    -o "$RUNNERD_BUILD_DIR/cambia-runnerd" ./cmd/runnerd)
  RUNNERD_BIN="$RUNNERD_BUILD_DIR/cambia-runnerd"
  echo "   build_commit = $BUILD_SHA"
fi

echo "== stage 1: /srv/cambia layout (sudo on the runner host)"
ssh "$RUNNER_SSH" 'sudo mkdir -p /srv/cambia && sudo chown cambia:cambia /srv/cambia'

echo "== stage 2: unprivileged layout, mirror, TLS cert (as cambia)"
ssh "$RUNNER_SSH" 'sudo -u cambia bash -s' <<EOS
set -euo pipefail
cd /srv/cambia
mkdir -p keys runs
if [ ! -d mirror.git ]; then
  git init --bare --quiet mirror.git
  git -C mirror.git config gc.auto 0
fi
if [ ! -f keys/tls.key ]; then
  openssl req -x509 -newkey ed25519 -keyout keys/tls.key -out keys/tls.crt \
    -days 825 -nodes -subj "/CN=cambia-runnerd" \
    -addext "subjectAltName=IP:${RUNNER_HOST}" 2>/dev/null
  chmod 600 keys/tls.key
fi
echo "TLS cert SHA256 fingerprint:"
openssl x509 -in keys/tls.crt -outform DER | sha256sum | cut -d' ' -f1
EOS

echo "== stage 3: ship JWT public key + runnerd binary"
scp -q "$JWT_PUB" "$RUNNER_SSH":/tmp/jwt_ed25519.pub
scp -q "$RUNNERD_BIN" "$RUNNER_SSH":/tmp/cambia-runnerd
scp -q "$REPO/runnerd/deploy/cambia-runnerd.service" "$RUNNER_SSH":/tmp/cambia-runnerd.service
ssh "$RUNNER_SSH" 'sudo install -o cambia -g cambia -m 644 /tmp/jwt_ed25519.pub /srv/cambia/keys/jwt_ed25519.pub \
  && sudo install -o root -g root -m 755 /tmp/cambia-runnerd /usr/local/bin/cambia-runnerd \
  && rm -f /tmp/jwt_ed25519.pub /tmp/cambia-runnerd'


echo "== stage 4: job-preserving daemon swap (sudo on the runner host)"
echo "   remote log: $DEPLOY_LOG"
# A redeploy must not take the training jobs down with the daemon. The order is
# load-bearing:
#   (a) install the unit, daemon-reload, enable
#   (b) assert the LOADED KillMode is process, abort otherwise
#   (c) snapshot the pids of every live job this host owns
#   (d) pick the swap method from the OLD daemon's health report
#   (e) assert active + a new MainPID + every snapshot pid still alive
# (b) runs before anything is stopped: under KillMode=mixed or control-group,
# systemd SIGKILLs whatever is left in the service cgroup once the main process
# exits, which is every job this stage exists to preserve.
stage4_rc=0
ssh "$RUNNER_SSH" "bash -s -- '$DEPLOY_TS'" <<'EOS' || stage4_rc=$?
set -euo pipefail

TS="$1"
SVC=cambia-runnerd
RUNS_DIR=/srv/cambia/runs
DEPLOY_DIR=/srv/cambia/deploy
LOG="$DEPLOY_DIR/deploy-$TS.log"
JOBS="$DEPLOY_DIR/deploy-$TS.jobs"
HEALTH_URL=https://127.0.0.1:8090/harness/health

sudo mkdir -p "$DEPLOY_DIR"
sudo chown cambia:cambia "$DEPLOY_DIR"
sudo chmod 755 "$DEPLOY_DIR"
sudo -u cambia touch "$LOG" "$JOBS"
sudo chmod 644 "$LOG" "$JOBS"

# The job snapshot lives in a temp file. It is created and trapped HERE, at the
# top level, not inside main(): main() runs as a pipeline stage, i.e. in a
# subshell, and an EXIT trap registered inside that subshell replaces its exit
# status with the trap's own (bash), which would silently turn a stage-4 failure
# into a successful deploy.
snapfile=$(mktemp)
trap 'rm -f "$snapfile"' EXIT

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# health_json prints the local daemon's /harness/health body, empty on any
# failure. curl is the documented runner prereq; python3 is the fallback, and it
# is present anyway because the JSON parsing below needs it.
health_json() {
  if command -v curl >/dev/null 2>&1; then
    curl -sk --max-time 5 "$HEALTH_URL" 2>/dev/null || true
    return 0
  fi
  python3 - "$HEALTH_URL" <<'PY' 2>/dev/null || true
import ssl, sys, urllib.request

ctx = ssl._create_unverified_context()
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5, context=ctx) as resp:
        sys.stdout.write(resp.read().decode())
except Exception:
    pass
PY
}

# json_field <key>: JSON object on stdin -> value of <key> on stdout (bools as
# true/false), empty when the key is absent or the body does not parse. jq is
# not a runner prereq, so this goes through python3.
json_field() {
  python3 -c '
import json, sys

try:
    doc = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
val = doc.get(sys.argv[1])
if val is None:
    raise SystemExit(0)
print("true" if val is True else ("false" if val is False else val))
' "$1" 2>/dev/null || true
}

unit_active() { systemctl is-active "$SVC" 2>/dev/null || true; }
main_pid() { systemctl show "$SVC" -p MainPID --value 2>/dev/null || echo 0; }

main() {
  log "stage 4 start (deploy $TS) on $(hostname)"

  # (a) unit file, daemon-reload, enable. The .d drop-in directory is read for
  # the log line and otherwise never touched.
  if ! sudo install -o root -g root -m 644 /tmp/cambia-runnerd.service "/etc/systemd/system/$SVC.service"; then
    log "!! ABORT: could not install /etc/systemd/system/$SVC.service"
    return 1
  fi
  rm -f /tmp/cambia-runnerd.service
  if ! sudo systemctl daemon-reload; then
    log "!! ABORT: systemctl daemon-reload failed"
    return 1
  fi
  if ! sudo systemctl enable "$SVC" >/dev/null; then
    log "!! ABORT: systemctl enable $SVC failed"
    return 1
  fi
  log "unit installed at /etc/systemd/system/$SVC.service; daemon-reload + enable done"
  if [ -d "/etc/systemd/system/$SVC.service.d" ]; then
    log "drop-ins left untouched: $(ls "/etc/systemd/system/$SVC.service.d" | tr '\n' ' ')"
  fi

  # (b) KillMode gate, on the MERGED loaded value, so a drop-in that overrides
  # KillMode is caught here too.
  killmode=$(systemctl show "$SVC" -p KillMode --value)
  log "loaded KillMode = ${killmode:-<unset>}"
  if [ "$killmode" != "process" ]; then
    log "!! ABORT: loaded unit has KillMode=${killmode:-<unset>}, expected process."
    log "!! A restart under that KillMode SIGKILLs everything left in the service"
    log "!! cgroup, i.e. every running job. Nothing was stopped and the running"
    log "!! daemon is untouched. Fix $SVC.service (and any .d drop-in that sets"
    log "!! KillMode), then re-run this deploy."
    return 1
  fi

  # (c) pre-swap snapshot of the live jobs this host owns.
  was_active=$(unit_active)
  old_main=$(main_pid)
  log "pre-swap: is-active=$was_active MainPID=$old_main"

  # An unreadable runs dir or a missing python3 must abort BEFORE the swap: a
  # deploy that cannot enumerate the jobs cannot promise they survived it.
  if ! sudo python3 - "$RUNS_DIR" > "$snapfile" <<'PY'
import glob, json, os, sys

# A live job is one this host owns. A row carrying "host" is a bounded-stale
# projection of another machine's run whose pid names a process in THAT host's
# pid space, so probing it locally would be the cross-host pid-reuse bug.
# "starting" joins running/stopping because it is a spawned child too, and it is
# in the daemon's own liveness set.
LIVE = ("starting", "running", "stopping")

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "process.json"))):
    try:
        with open(path) as fh:
            state = json.load(fh)
    except Exception:
        continue
    if state.get("host"):
        continue
    if state.get("status") not in LIVE:
        continue
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        continue
    if pid <= 0:
        continue
    try:
        ticks = int(state.get("start_ticks") or 0)
    except (TypeError, ValueError):
        ticks = 0
    run_dir = os.path.dirname(path)
    name = state.get("name") or os.path.basename(run_dir)
    print("\t".join([name, str(pid), str(ticks), state.get("status", ""), run_dir]))
PY
  then
    log "!! ABORT: could not snapshot live jobs from $RUNS_DIR; nothing was stopped"
    return 1
  fi

  njobs=$(wc -l < "$snapfile" | tr -d ' ')
  log "pre-swap snapshot: $njobs live job(s) owned by this host"
  while IFS=$'\t' read -r snap_name snap_pid snap_ticks snap_status _; do
    log "  job $snap_name pid=$snap_pid status=$snap_status start_ticks=$snap_ticks"
  done < "$snapfile"

  # (d) swap method, chosen from the OLD daemon's own health report. A daemon
  # that answers restart_preserves_jobs=true detaches its jobs on SIGTERM, so a
  # plain restart is safe. Anything else (older binary, or health unreachable)
  # gets SIGKILL on the main process only: SIGKILL cannot be handled, so the old
  # SIGTERM handler that kills every job process group never runs, and
  # KillMode=process leaves the children alone.
  if [ "$was_active" != "active" ]; then
    log "unit is not active (is-active=$was_active): cold start, nothing to swap"
    sudo systemctl start "$SVC"
  else
    preserves=$(health_json | json_field restart_preserves_jobs)
    log "old daemon reports restart_preserves_jobs=${preserves:-<unknown>}"
    if [ "$preserves" = "true" ]; then
      log "swap method: systemctl restart (its SIGTERM detaches jobs)"
      sudo systemctl restart "$SVC"
    else
      log "swap method: SIGKILL the main process only (this daemon's SIGTERM kills jobs)"
      if ! kill_err=$(sudo systemctl kill --kill-whom=main --signal=SIGKILL "$SVC" 2>&1); then
        log "  --kill-whom rejected (${kill_err}); retrying --kill-who (systemd < 252 spelling)"
        sudo systemctl kill --kill-who=main --signal=SIGKILL "$SVC"
      fi
      log "  waiting up to 20s for Restart=on-failure to bring the unit back"
      for _ in $(seq 1 20); do
        [ "$(unit_active)" = "active" ] && break
        sleep 1
      done
      if [ "$(unit_active)" != "active" ]; then
        log "  unit did not come back on its own; issuing systemctl start"
        sudo systemctl start "$SVC"
      fi
    fi
  fi

  # (e) post-swap assertions: unit active, MainPID moved, every snapshot pid
  # still alive. No repair is attempted on failure; a half-swapped runner is an
  # operator decision, not a script's.
  new_main=$old_main
  act=""
  for _ in $(seq 1 30); do
    act=$(unit_active)
    new_main=$(main_pid)
    if [ "$act" = "active" ] && [ -n "$new_main" ] && [ "$new_main" != "0" ] && [ "$new_main" != "$old_main" ]; then
      break
    fi
    sleep 1
  done
  log "post-swap: is-active=$act MainPID=$new_main (was $old_main)"

  swap_rc=0
  if [ "$act" != "active" ]; then
    log "!! FAILURE: $SVC is not active after the swap (is-active=$act)"
    swap_rc=1
  fi
  if [ -z "$new_main" ] || [ "$new_main" = "0" ] || [ "$new_main" = "$old_main" ]; then
    log "!! FAILURE: MainPID did not change ($old_main -> ${new_main:-0}); the shipped binary is not the one serving"
    swap_rc=1
  fi

  # Liveness is pid + start_ticks (/proc/<pid>/stat field 22), so a recycled pid
  # cannot read as a surviving job.
  jobs_rc=0
  table=$(sudo python3 - "$snapfile" <<'PY'
import json, os, sys


def start_ticks(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            data = fh.read()
    except OSError:
        return None
    # comm (field 2) can hold spaces and parens, so the numeric fields resume
    # after the LAST ')'; starttime is field 22 = index 19 of the remainder.
    return int(data[data.rindex(")") + 2:].split()[19])


def current_status(run_dir):
    try:
        with open(os.path.join(run_dir, "process.json")) as fh:
            return json.load(fh).get("status", "?")
    except Exception:
        return "?"


rows, missing = [], []
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        name, pid_s, ticks_s, status, run_dir = line.split("\t")
        pid, want = int(pid_s), int(ticks_s)
        got = start_ticks(pid)
        if got is None:
            post = "GONE (now %s)" % current_status(run_dir)
        elif want and got != want:
            post = "GONE, pid reused (now %s)" % current_status(run_dir)
        else:
            post = "alive"
        if post != "alive":
            missing.append("%s (pid %d)" % (name, pid))
        rows.append((name, pid, status, post))

print("%-38s %-8s %-9s %s" % ("JOB", "PID", "PRE", "POST"))
for name, pid, status, post in rows:
    print("%-38s %-8d %-9s %s" % (name, pid, status, post))
if not rows:
    print("(no jobs were live at swap time)")
if missing:
    print("MISSING AFTER SWAP: " + ", ".join(missing))
raise SystemExit(1 if missing else 0)
PY
) || jobs_rc=$?
  printf '%s\n' "$table"
  printf '%s\n' "$table" | sudo -u cambia tee "$JOBS" >/dev/null
  if [ "$jobs_rc" != "0" ]; then
    log "!! FAILURE: a job that was live before the swap is gone (see MISSING AFTER SWAP above)."
    log "!! No repair attempted. Check journalctl -u $SVC and that run's process.json"
    log "!! before submitting more work to this runner."
    swap_rc=1
  elif [ "$njobs" != "0" ]; then
    log "all $njobs pre-swap job(s) survived the swap"
  fi

  # Let the new daemon bind and answer health before the client-side probes.
  hbody=""
  for _ in $(seq 1 20); do
    hbody=$(health_json)
    [ -n "$(printf '%s' "$hbody" | json_field build_commit)" ] && break
    sleep 1
  done
  new_commit=$(printf '%s' "$hbody" | json_field build_commit)
  new_preserves=$(printf '%s' "$hbody" | json_field restart_preserves_jobs)
  log "new daemon health: build_commit=${new_commit:-<none>} restart_preserves_jobs=${new_preserves:-<none>}"

  sudo systemctl --no-pager --lines=5 status "$SVC" || true
  log "stage 4 done (rc=$swap_rc); log $LOG, job table $JOBS"
  return "$swap_rc"
}

rc=0
main 2>&1 | sudo -u cambia tee -a "$LOG" || rc=$?
exit "$rc"
EOS
if [ "$stage4_rc" != "0" ]; then
  echo "DEPLOY CHECK FAILED: stage 4 exited $stage4_rc; the swap was aborted or a job did not survive."
  echo "Remote log: $DEPLOY_LOG (on $RUNNER_SSH). Stage 5 probes skipped."
  exit 1
fi

echo "== stage 5: acceptance probes"
# TLS handshake succeeds; /harness/health is the one token-free route
# (read-only counters for LAN monitoring, cambia-330/network-552) and every
# other route refuses an unauthenticated request with 401.
hcode=$(curl -sk -o /dev/null -w '%{http_code}' "https://${RUNNER_HOST}:8090/harness/health" || true)
jcode=$(curl -sk -o /dev/null -w '%{http_code}' "https://${RUNNER_HOST}:8090/harness/jobs" || true)
echo "unauthenticated /harness/health -> HTTP $hcode (expect 200)"
echo "unauthenticated /harness/jobs   -> HTTP $jcode (expect 401)"

# The health body says WHICH daemon answered: build_commit pins it to the binary
# stage 0 built, restart_preserves_jobs pins it to a build whose SIGTERM detaches
# jobs instead of killing them (an old binary answering here means the swap put
# the wrong daemon back).
hbody=$(curl -sk --max-time 5 "https://${RUNNER_HOST}:8090/harness/health" || true)
health_field() {
  printf '%s' "$hbody" | python3 -c '
import json, sys

try:
    doc = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
val = doc.get(sys.argv[1])
if val is None:
    raise SystemExit(0)
print("true" if val is True else ("false" if val is False else val))
' "$1" 2>/dev/null || true
}
live_preserves=$(health_field restart_preserves_jobs)
live_commit=$(health_field build_commit)
echo "health restart_preserves_jobs   -> ${live_preserves:-<absent>} (expect true)"
echo "health build_commit             -> ${live_commit:-<absent>} (expect ${BUILD_SHA:-<any non-empty>})"

echo "pre/post job pid table (from the runner):"
ssh "$RUNNER_SSH" "cat '$DEPLOY_JOBS' 2>/dev/null" || echo "(job table unavailable: $DEPLOY_JOBS)"
echo "stage 4 log on the runner: $DEPLOY_LOG"

probe_rc=0
[ "$hcode" = "200" ] || { echo "FAIL: /harness/health returned $hcode, expected 200"; probe_rc=1; }
[ "$jcode" = "401" ] || { echo "FAIL: /harness/jobs returned $jcode, expected 401"; probe_rc=1; }
[ "$live_preserves" = "true" ] || { echo "FAIL: health restart_preserves_jobs is ${live_preserves:-absent}, expected true (old binary still serving?)"; probe_rc=1; }
if [ -n "$BUILD_SHA" ]; then
  [ "$live_commit" = "$BUILD_SHA" ] || { echo "FAIL: health build_commit is ${live_commit:-absent}, expected $BUILD_SHA (the shipped binary is not the one serving)"; probe_rc=1; }
else
  echo "note: RUNNERD_BIN was pre-built without RUNNERD_BUILD_SHA; build_commit is only checked non-empty"
  [ -n "$live_commit" ] || { echo "FAIL: health build_commit is absent"; probe_rc=1; }
fi
[ "$probe_rc" = "0" ] && echo "DEPLOY OK" || { echo "DEPLOY CHECK FAILED"; exit 1; }
