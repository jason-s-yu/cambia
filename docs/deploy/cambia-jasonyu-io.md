# Deploying cambia.jasonyu.io

Deploy runbook for the production Cambia service at `cambia.jasonyu.io`, driven by `.github/workflows/deploy.yml`.

## Architecture

Production runs on the hawking host (`157.245.172.131`). Nginx on hawking terminates TLS, serves the built web client as static files from `/var/www/cambia.jasonyu.io`, and reverse-proxies API/WebSocket traffic to `127.0.0.1:3200`, which is bound by the `server` container in a Docker Compose stack at `/opt/cambia` (`deploy/hawking/docker-compose.yml`). The stack also runs `historian`, `postgres`, and `redis`; the database containers are not published on the host network and are reachable only from other stack containers.

## Secrets inventory

Set at the repository level (Settings > Secrets and variables > Actions > Secrets).

All four were minted and set network-side on 2026-08-27 (cambia-767). The private halves exist only in GitHub secrets and pair with public keys installed in hawking's `authorized_keys`: never re-mint either side alone. Rotation must update GitHub secrets and hawking `authorized_keys` together; the procedure lives in the network repo, `hawking/README.md` cambia section.

|Secret|What it is|Where minted|
|-|-|-|
|`CAMBIA_WEB_DEPLOY_KEY`|Private SSH key for an rrsync-restricted account limited to writing `/var/www/cambia.jasonyu.io`|Generated on hawking during network-hawking-2828; the matching public key goes into that account's `authorized_keys` with an `rrsync` `command=` restriction|
|`CAMBIA_DEPLOY_SSH_KEY`|Private SSH key for the deploy account with permission to run `docker compose` in `/opt/cambia`|Generated on hawking during network-hawking-2828; the matching public key goes into the deploy account's `authorized_keys`|
|`HAWKING_DEPLOY_USER`|Unix username on hawking that both the web-asset rsync and the compose SSH command connect as|Set alongside the two keys above during network-hawking-2828|
|`HAWKING_KNOWN_HOSTS`|Output of `ssh-keyscan 157.245.172.131`, used for strict host-key checking on both the rsync and SSH deploy steps|Captured once from hawking and pasted into the secret|

## Deploy execution contract (hawking forced command)

The compose-leg SSH key is bound to a forced command on hawking (`~/bin/cambia-deploy`, owned by the network project). It validates that the client requested `bash -s -- <sha>`, updates `CAMBIA_IMAGE_TAG` in `/opt/cambia/.env`, and runs `docker compose pull && docker compose up -d`. The script content `deploy.yml` pipes over stdin is ignored by design: a leaked key can only deploy a tag, nothing else.

Consequence: editing the `REMOTE` heredoc in `deploy.yml` changes nothing on hawking. Deploy-behavior changes land in the wrapper (network project) first, with the workflow's block kept cosmetically in sync so the file still documents what happens.

## Prerequisites

All satisfied as of 2026-08-27 (network-hawking-2828 + network-adleman-2829 done, `RUNS_ON_LINUX` set to the self-hosted runner). Recorded for rebuild-from-scratch:

- **network-hawking-2828** (hawking ingress): DNS A record for `cambia.jasonyu.io`, a TLS certificate, the nginx vhost (static root + reverse proxy to `127.0.0.1:3200`), and the two deploy accounts/keys listed above.
- **network-adleman-2829** (runner registration): this repo added to the `gh-runner` fleet's `repos.txt` so a self-hosted runner is available, per `adleman/gh-runner.md` in the `network` repo.
- **`RUNS_ON_LINUX` repository variable**: unset routes `ci.yml` and `deploy.yml` jobs to GitHub-hosted `ubuntu-latest`. Once network-adleman-2829 lands, flip to self-hosted with:

  ```sh
  gh variable set RUNS_ON_LINUX -R jason-s-yu/cambia --body '["self-hosted","linux","x64"]'
  ```

  This is reversible by deleting the variable; no workflow edit or commit required either direction.

## First-deploy checklist

Until the first successful pipeline run, `https://cambia.jasonyu.io/` serves 500 (empty webroot) and the API paths serve 502 (no stack): both are the expected pre-deploy state, not faults.

1. Secrets: already set (see Secrets inventory); nothing to do unless rotating.
2. Bootstrap `/opt/cambia` on hawking: copy `deploy/hawking/docker-compose.yml` to `/opt/cambia/docker-compose.yml` and create `/opt/cambia/.env` with `POSTGRES_PASSWORD` (plus any overrides such as `TOKEN_EXPIRE_TIME`). `CAMBIA_IMAGE_TAG` does not need to be pre-set, the deploy wrapper appends or updates it. The wrapper fails with "no configuration file" until this step is done. Compose-file changes are NOT synced by the pipeline (it only flips the image tag): re-copy the file on any future change to `deploy/hawking/docker-compose.yml`.
3. Push to `master` touching one of the paths in `deploy.yml`'s trigger list, or run the workflow manually, and confirm `build-server`, `build-web`, and `deploy` all complete.
4. `curl https://cambia.jasonyu.io/healthz` returns healthy.
5. Play a full game end to end over `wss://cambia.jasonyu.io` to confirm the WebSocket path through nginx and the reverse proxy works, not just the health check.

## Rollback

Two options, in order of preference:

- **Re-run a prior successful `deploy.yml` run** from the Actions tab (or `gh run rerun <run-id>`), which redeploys the image tag and web assets from that commit.
- **Manual pin**, if CI is unavailable: invoke the deploy wrapper with the old sha (the deploy key's forced command only accepts this shape):

  ```sh
  ssh -i <deploy-key> <deploy-user>@157.245.172.131 bash -s -- <old-sha> < /dev/null
  ```

  An operator account on hawking (outside the forced-command key) can instead edit `CAMBIA_IMAGE_TAG` in `/opt/cambia/.env` and run `docker compose pull && docker compose up -d` directly.

  Web static assets are not versioned by tag; recovering a prior frontend build requires re-running the `build-web` job for the target commit (rsync mirrors `web/dist/` with `--delete`, so redeploying the current tree is the only way back).
