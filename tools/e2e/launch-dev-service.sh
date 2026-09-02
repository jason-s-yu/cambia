#!/usr/bin/env bash
# tools/e2e/launch-dev-service.sh
# Builds and launches the game service the end-to-end pass (cambia-934) runs against.
#
# Detached on purpose. An agent turn ending reaps the process group it started, which has taken
# the :8088 dev service down twice mid-run (hub note cambia-998), so the service is started with
# setsid and its own nohup and a manifest is written naming the pid, the sha it was built from and
# its log. Nothing else in this repository is allowed to guess which build is answering on :8088.
#
# Usage, from the repository root:
#   tools/e2e/launch-dev-service.sh            # build, launch, wait for /healthz
#   OUT_DIR=/tmp/e2e tools/e2e/launch-dev-service.sh
#   tools/e2e/stop-dev-service.sh              # stop it again
#
# Requires Postgres and Redis. The dev compose stack provides both:
#   cd service && docker compose up -d
# and their ports are read from service/.env, whose values this script defaults to.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/cambia-e2e}"
PORT="${CAMBIA_E2E_PORT:-8088}"
BIN="$OUT_DIR/cambia-server-e2e"
LOG="$OUT_DIR/cambia-server-e2e.log"
MANIFEST="$OUT_DIR/cambia-server-e2e.manifest.json"

# Connection settings. Defaults match service/.env.template with the dev Postgres port, which is
# 5434 here because 5432 belongs to another stack on this host. Override any of them in the
# environment rather than editing this file.
export PORT
export DATA_DIR="${DATA_DIR:-$OUT_DIR/data}"
export POSTGRES_USER="${POSTGRES_USER:-cambia}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-cambia-1}"
export PG_HOST="${PG_HOST:-localhost}"
export PG_PORT="${PG_PORT:-5434}"
export PG_DATABASE="${PG_DATABASE:-cambia-dev}"
export REDIS_ADDR="${REDIS_ADDR:-localhost:6379}"
# The suite pins each seat with ?as=<name>, which needs the dev identity routes registered.
export CAMBIA_DEV_ACCOUNTS=1

mkdir -p "$OUT_DIR" "$DATA_DIR"

if [ -f "$MANIFEST" ]; then
  OLD_PID="$(sed -n 's/.*"pid":\([0-9]*\).*/\1/p' "$MANIFEST")"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "A service from a previous launch is still running (pid $OLD_PID)." >&2
    echo "Stop it with tools/e2e/stop-dev-service.sh before launching another." >&2
    exit 1
  fi
fi

SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "Building the service from $SHA"
(cd "$REPO_ROOT/service" && go build -o "$BIN" ./cmd/server)

# cd out of the repository: the server autoloads a .env from its working directory, and this
# script's explicit settings should be the whole story.
cd "$OUT_DIR"
setsid nohup "$BIN" >"$LOG" 2>&1 &
PID=$!

printf '{"pid":%d,"sha":"%s","port":%s,"log":"%s","bin":"%s","started":"%s"}\n' \
  "$PID" "$SHA" "$PORT" "$LOG" "$BIN" "$(date -Is)" >"$MANIFEST"

for _ in $(seq 1 40); do
  if curl -fsS "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
    echo "Service up on :$PORT"
    curl -fsS "http://localhost:$PORT/healthz"
    echo
    cat "$MANIFEST"
    exit 0
  fi
  sleep 0.5
done

echo "Service did not answer /healthz within 20s. Log tail:" >&2
tail -30 "$LOG" >&2
exit 1
