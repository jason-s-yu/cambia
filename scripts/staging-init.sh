#!/usr/bin/env bash
# One-time staging setup: obtain Let's Encrypt cert, then start full stack.
# Prerequisites: DNS A record for $STAGING_DOMAIN → this machine's IP
set -euo pipefail

DOMAIN="${STAGING_DOMAIN:?Set STAGING_DOMAIN env var}"
COMPOSE="docker compose -f docker-compose.staging.yml"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "=== Step 1: Start infra (postgres, redis, service) ==="
$COMPOSE up -d postgres redis service

echo "=== Step 2: Start nginx with HTTP-only config for ACME challenge ==="
$COMPOSE run -d --name cambia-certbot-nginx \
    -p 80:80 \
    -v "$DIR/nginx/nginx-init.conf:/etc/nginx/templates/default.conf.template:ro" \
    -v certbot_webroot:/var/www/certbot \
    web

echo "Waiting for nginx to start..."
sleep 2

echo "=== Step 3: Run certbot ==="
docker run --rm \
    -v cambia_letsencrypt:/etc/letsencrypt \
    -v cambia_certbot_webroot:/var/www/certbot \
    certbot/certbot certonly \
    --webroot -w /var/www/certbot \
    -d "$DOMAIN" \
    --non-interactive --agree-tos \
    --email "${CERTBOT_EMAIL:?Set CERTBOT_EMAIL env var}" \
    --no-eff-email

echo "=== Step 4: Stop temp nginx, start full stack ==="
docker stop cambia-certbot-nginx && docker rm cambia-certbot-nginx
$COMPOSE up -d

echo ""
echo "=== Done! ==="
echo "Site live at https://$DOMAIN"
echo ""
echo "To renew certs later:"
echo "  docker run --rm \\"
echo "    -v cambia_letsencrypt:/etc/letsencrypt \\"
echo "    -v cambia_certbot_webroot:/var/www/certbot \\"
echo "    certbot/certbot renew"
echo "  $COMPOSE exec web nginx -s reload"
