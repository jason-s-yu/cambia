# Serving Harness: Quickstart

This is the client-side setup path: getting a workstation from a fresh clone
to a submitted job on an already-deployed runner. It assumes the runner side
is up and running -- see [runnerd/README.md](../../runnerd/README.md) for
running the daemon and [deploy.md](deploy.md) for standing one up from
scratch. For the full system design see
[architecture.md](architecture.md); for exactly what `cambia harness init`
generates and why the trust model holds, see
[keys-and-tls.md](keys-and-tls.md).

## 1. Install the client

The harness CLI and pull loop live behind an optional `harness` extra so the
base `cambia` install stays lightweight:

```bash
cd cfr
pip install -e ".[gpu,harness]" --extra-index-url https://download.pytorch.org/whl/cu128
```

(Swap `gpu` for `cpu` if this workstation has no CUDA device -- the harness
client itself runs no GPU code either way; only the base package's PyTorch
build depends on it.)

## 2. Bootstrap: `cambia harness init`

```bash
cambia harness init \
  --runner-url https://<runner-host>:8090 \
  --ssh-target <runner-ssh-alias> \
  --mirror-url cambia@<runner-host>:/srv/cambia/mirror.git
```

This is a one-shot, non-interactive bootstrap:

- generates an ed25519 signing key pair and writes the private half to
  `~/.config/cambia/jwt_ed25519` (chmod `0600`) and the public half to
  `~/.config/cambia/jwt_ed25519.pub`
- scaffolds `~/.config/cambia/harness.yaml` from the checked-in template
  (`cfr/config/harness.example.yaml`), filling in whichever of
  `--runner-url` / `--ssh-target` / `--mirror-url` you passed
- refuses to overwrite an existing key pair or config; pass `--force` to
  regenerate

Any flag you omit is left as the template's placeholder value -- fill it in
by hand afterward, or run `init` again later with `--force` once you know
it. See [keys-and-tls.md section 2](keys-and-tls.md#2-the-primary-path-cambia-harness-init)
for exactly what gets written and the manual equivalent if you'd rather not
use `init`.

## 3. Ship the public key and pin the TLS fingerprint

The runner trusts exactly one JWT public key at a time (`RUNNERD_JWT_PUBKEY`)
and exactly one pinned TLS fingerprint per client config -- there's no CA
chain to walk, so both of these are manual, out-of-band steps:

1. Get `~/.config/cambia/jwt_ed25519.pub` (the raw 32-byte public key from
   step 2) onto the runner host, installed at the path
   `RUNNERD_JWT_PUBKEY` points to, and restart `runnerd` so it picks up the
   new key. If you also operate the runner, `deploy.md`'s deploy script does
   this as part of a redeploy; otherwise send the file to whoever does.
2. Fetch the runner's TLS certificate fingerprint and paste it into
   `harness.yaml`'s `runner.cert_fingerprint`:

   ```bash
   openssl s_client -connect <runner-host>:8090 </dev/null 2>/dev/null \
     | openssl x509 -fingerprint -sha256 -noout
   ```

Full detail on both steps, including why fingerprint pinning replaces CA
validation here, is in [keys-and-tls.md](keys-and-tls.md).

## 4. ssh and git prerequisites for the data plane

The data plane (commit pushes and artifact pulls) runs over plain ssh, not
the control plane, so it needs its own reachability:

- an ssh alias (or `user@host`) that reaches the runner without a
  passphrase prompt -- add it to `~/.ssh/config` if `--ssh-target` above
  was a bare hostname
- push access to the bare git mirror at `data_plane.mirror_remote_url`
  (`cambia harness submit` pushes your pinned commit there directly, not
  through a named git remote)
- `rsync` installed locally (used by `harness pull` / `harness watch` /
  `harness push-run`)

## 5. Submit a first job

Copy the example spec and adjust it for a real config:

```bash
cp cfr/config/harness-spec.example.yaml my_job.yaml
# edit my_job.yaml: name, config, overrides, etc.
cambia harness submit my_job.yaml
```

(The spec's `device` defaults to `cpu`. `cuda` and `xpu` only work if the
runner advertises support for them -- see
[host-requirements.md's GPU section](host-requirements.md#optional-gpu-support).)

`submit` requires a clean working tree: it pins your HEAD commit, pushes it
to the runner's git mirror, then submits the job referencing that commit, so
`run_db` always records exactly what ran. Check on it with:

```bash
cambia harness status <job_name>
cambia harness logs -f <job_name>
```

## 6. Get results back: `watch` or `pull`

Artifacts and `run_db` rows don't appear locally until you pull them:

```bash
cambia harness watch          # foreground loop: periodic pulls + reconcile
cambia harness pull <job_name>  # one-shot pull for a single job
```

`watch` also does an immediate retry-backed pull on every terminal
transition it observes, so a job that finishes while `watch` is running
syncs promptly rather than waiting for the next periodic tick.

## See also

- [architecture.md](architecture.md) -- full system design: control plane,
  data plane, trust boundary, and the reconciler's untrusted-input handling
- [keys-and-tls.md](keys-and-tls.md) -- exactly what `cambia harness init`
  generates, the manual equivalent, and the TLS/JWT trust model
- [deploy.md](deploy.md) -- standing up (or redeploying) the runner side
- [../../runnerd/README.md](../../runnerd/README.md) -- the runner daemon's
  own environment variables, API surface, and operational behavior
- `cfr/config/harness.example.yaml` -- the client connection config template
- `cfr/config/harness-spec.example.yaml` -- the job spec template
