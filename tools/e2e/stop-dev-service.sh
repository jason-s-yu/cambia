#!/usr/bin/env bash
# tools/e2e/stop-dev-service.sh
# Stops the service tools/e2e/launch-dev-service.sh started, by the pid in its manifest.
#
# The manifest, not a port scan or a pattern match on the process table: the port can be held by
# something else, and killing by pattern reaches builds this script never started.
set -euo pipefail

OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/cambia-e2e}"
MANIFEST="$OUT_DIR/cambia-server-e2e.manifest.json"

if [ ! -f "$MANIFEST" ]; then
  echo "No manifest at $MANIFEST; nothing this script started is running." >&2
  exit 0
fi

PID="$(sed -n 's/.*"pid":\([0-9]*\).*/\1/p' "$MANIFEST")"
if [ -z "$PID" ]; then
  echo "Manifest at $MANIFEST names no pid." >&2
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Service pid $PID is already gone."
  rm -f "$MANIFEST"
  exit 0
fi

# SIGTERM first: the server drains on it (cmd/server/main.go arms the handler before the listener).
kill "$PID"
for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Service pid $PID stopped."
    rm -f "$MANIFEST"
    exit 0
  fi
  sleep 0.5
done

echo "Service pid $PID did not stop on SIGTERM; sending SIGKILL." >&2
kill -9 "$PID" 2>/dev/null || true
rm -f "$MANIFEST"
