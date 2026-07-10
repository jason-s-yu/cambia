#!/usr/bin/env bash
# One-paste deploy of the cambia serving harness onto the runner host.
# Run from the client workstation as the operator. Stages 1 and 4 use sudo ON THE RUNNER HOST (the operator is
# NOPASSWD there); nothing runs privileged on the client workstation.
#
# Required env:
#   - RUNNER_HOST: the runner host IP or hostname (used for the TLS cert
#     subjectAltName and the acceptance-probe URL).
#
# Prereqs (already done if the autoiterate window staged them):
#   - runnerd binary at $RUNNERD_BIN (built in-script if unset; static, CGO_ENABLED=0)
#   - JWT public key at ~/.config/cambia/jwt_ed25519.pub (raw 32-byte ed25519)
#   - repo checkout at $REPO with runnerd/deploy/cambia-runnerd.service
set -euo pipefail

RUNNER_SSH=${RUNNER_SSH:-runner}
REPO=${REPO:-$HOME/dev/cambia}
JWT_PUB=${JWT_PUB:-$HOME/.config/cambia/jwt_ed25519.pub}
RUNNER_HOST=${RUNNER_HOST:?set RUNNER_HOST to the runner host IP or hostname, e.g. RUNNER_HOST=192.0.2.10}

if [ -z "${RUNNERD_BIN:-}" ]; then
  echo "== stage 0: building cambia-runnerd (RUNNERD_BIN unset)"
  RUNNERD_BUILD_DIR=$(mktemp -d)
  (cd "$REPO/runnerd" && CGO_ENABLED=0 go build -o "$RUNNERD_BUILD_DIR/cambia-runnerd" ./cmd/runnerd)
  RUNNERD_BIN="$RUNNERD_BUILD_DIR/cambia-runnerd"
fi

echo "== stage 1: /srv/cambia layout (sudo on runner)"
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

echo "== stage 4: systemd unit (sudo on runner)"
ssh "$RUNNER_SSH" 'sudo install -o root -g root -m 644 /tmp/cambia-runnerd.service /etc/systemd/system/cambia-runnerd.service \
  && rm -f /tmp/cambia-runnerd.service \
  && sudo systemctl daemon-reload \
  && sudo systemctl enable cambia-runnerd \
  && sudo systemctl restart cambia-runnerd \
  && sleep 2 && sudo systemctl --no-pager --lines=5 status cambia-runnerd'

echo "== stage 5: acceptance probes"
# TLS handshake succeeds and an unauthenticated request is refused with 401.
code=$(curl -sk -o /dev/null -w '%{http_code}' "https://${RUNNER_HOST}:8090/harness/health" || true)
echo "unauthenticated /harness/health -> HTTP $code (expect 401)"
[ "$code" = "401" ] && echo "DEPLOY OK" || { echo "DEPLOY CHECK FAILED"; exit 1; }
