# Deploying cambia.jasonyu.io

Deploy runbook for the production Cambia service at `cambia.jasonyu.io`, driven by `.github/workflows/deploy.yml`.

## Architecture

Production runs on the hawking host (`157.245.172.131`). Nginx on hawking terminates TLS, serves the built web client as static files from `/var/www/cambia.jasonyu.io`, and reverse-proxies API/WebSocket traffic to `127.0.0.1:3200`, which is bound by the `server` container in a Docker Compose stack at `/opt/cambia` (`deploy/hawking/docker-compose.yml`). The stack also runs `historian`, `postgres`, and `redis`; the database containers are not published on the host network and are reachable only from other stack containers.

## Secrets inventory

Set at the repository level (Settings > Secrets and variables > Actions > Secrets).

|Secret|What it is|Where minted|
|-|-|-|
|`CAMBIA_WEB_DEPLOY_KEY`|Private SSH key for an rrsync-restricted account limited to writing `/var/www/cambia.jasonyu.io`|Generated on hawking during network-hawking-2828; the matching public key goes into that account's `authorized_keys` with an `rrsync` `command=` restriction|
|`CAMBIA_DEPLOY_SSH_KEY`|Private SSH key for the deploy account with permission to run `docker compose` in `/opt/cambia`|Generated on hawking during network-hawking-2828; the matching public key goes into the deploy account's `authorized_keys`|
|`HAWKING_DEPLOY_USER`|Unix username on hawking that both the web-asset rsync and the compose SSH command connect as|Set alongside the two keys above during network-hawking-2828|
|`HAWKING_KNOWN_HOSTS`|Output of `ssh-keyscan 157.245.172.131`, used for strict host-key checking on both the rsync and SSH deploy steps|Captured once from hawking and pasted into the secret|

## Prerequisites

Before the deploy workflow can run against hawking:

- **network-hawking-2828** (hawking ingress): DNS A record for `cambia.jasonyu.io`, a TLS certificate, the nginx vhost (static root + reverse proxy to `127.0.0.1:3200`), and the two deploy accounts/keys listed above.
- **network-adleman-2829** (runner registration): this repo added to the `gh-runner` fleet's `repos.txt` so a self-hosted runner is available, per `adleman/gh-runner.md` in the `network` repo.
- **`RUNS_ON_LINUX` repository variable**: unset routes `ci.yml` and `deploy.yml` jobs to GitHub-hosted `ubuntu-latest`. Once network-adleman-2829 lands, flip to self-hosted with:

  ```sh
  gh variable set RUNS_ON_LINUX -R jason-s-yu/cambia --body '["self-hosted","linux","x64"]'
  ```

  This is reversible by deleting the variable; no workflow edit or commit required either direction.

## First-deploy checklist

1. All four secrets above are set on the repository.
2. On hawking, `/opt/cambia/.env` exists with `POSTGRES_PASSWORD` and any other variables `deploy/hawking/docker-compose.yml` requires; `CAMBIA_IMAGE_TAG` does not need to be pre-set, the deploy job appends or updates it.
3. `docker compose -f /opt/cambia/deploy/hawking/docker-compose.yml up -d` succeeds manually once on hawking to confirm the stack starts clean before handing control to CI.
4. Push to `master` touching one of the paths in `deploy.yml`'s trigger list, or run the workflow manually, and confirm `build-server`, `build-web`, and `deploy` all complete.
5. `curl https://cambia.jasonyu.io/healthz` returns healthy.
6. Play a full game end to end over `wss://cambia.jasonyu.io` to confirm the WebSocket path through nginx and the reverse proxy works, not just the health check.

## Rollback

Two options, in order of preference:

- **Re-run a prior successful `deploy.yml` run** from the Actions tab (or `gh run rerun <run-id>`), which redeploys the image tag and web assets from that commit.
- **Manual pin**, if CI is unavailable: SSH to hawking as the deploy user and run

  ```sh
  cd /opt/cambia
  sed -i "s/^CAMBIA_IMAGE_TAG=.*/CAMBIA_IMAGE_TAG=<old-sha>/" .env
  docker compose pull
  docker compose up -d
  ```

  Web static assets are not versioned by tag; recovering a prior frontend build requires re-running the `build-web` job for the target commit (rsync mirrors `web/dist/` with `--delete`, so redeploying the current tree is the only way back).
