# Serving Harness: Keys and TLS

This is the key- and certificate-provisioning walkthrough for a runner
deployment: what gets generated, in what format, where each piece lands,
and why the trust model holds together without a certificate authority.

## 1. Security properties, in brief

- **The runner never holds a signing key.** Control-plane authentication is
  a Bearer JWT signed with an ed25519 private key that lives only on the
  client workstation. The runner (`runnerd`) loads only the matching
  *public* half and verifies; it can never mint a token of its own, so a
  compromised runner cannot forge control-plane access.
- **Fingerprint pinning replaces CA validation.** The runner's TLS
  certificate is self-signed, not issued by a trusted CA. Rather than
  walking a certificate chain, the client pins the certificate's SHA256
  fingerprint out of band (captured once, at deploy time) and checks every
  connection's peer certificate against that exact fingerprint. This is
  tighter than trusting a CA that could issue certificates for other
  identities too: the client trusts exactly one certificate, not a whole
  chain of trust.
- **Two independent keys, two independent purposes.** The TLS keypair
  secures the transport (confidentiality/integrity of the connection); the
  JWT keypair authenticates the caller (who is allowed to submit/query/
  cancel jobs). Rotating one never requires rotating the other.

## 2. The primary path: `cambia harness init`

`cambia harness init` generates the JWT keypair and scaffolds
`harness.yaml` for you. It writes:

- the private key to `~/.config/cambia/jwt_ed25519` (raw 32-byte ed25519
  seed)
- the public key to `~/.config/cambia/jwt_ed25519.pub` (raw 32-byte
  ed25519 public key)

Both are the raw byte form, not PEM -- this is one of the three formats the
client-side loader accepts (see [section 4](#4-manual-key-generation) for
the others) and the only format `runnerd`'s public-key loader accepts at
all. Run it, fill in the scaffolded `harness.yaml` with the runner's URL
and TLS fingerprint once you have them (section 3), and copy
`~/.config/cambia/jwt_ed25519.pub` to the runner as described in
[deploy.md](deploy.md). The rest of this document is the manual/reference
path for anyone who wants to generate these by hand or understand exactly
what `init` is doing.

## 3. TLS certificate generation

Generate a self-signed ed25519 certificate on the runner host, with the
runner's IP or hostname as the `subjectAltName` (this is what the client
validates the connection against -- not the CN, since hostname checking is
disabled entirely in favor of fingerprint pinning, but a correct SAN keeps
the cert self-consistent and lets other tooling that *does* check
hostnames still work against it):

```bash
openssl req -x509 -newkey ed25519 -keyout tls.key -out tls.crt \
  -days 825 -nodes -subj "/CN=cambia-runnerd" \
  -addext "subjectAltName=IP:192.0.2.10"
```

Use `-addext "subjectAltName=DNS:runner.example.internal"` instead if the
client will reach the runner by hostname rather than IP.

Extract the SHA256 fingerprint right after generation:

```bash
openssl x509 -in tls.crt -outform DER | sha256sum | cut -d' ' -f1
```

This is the value that goes into `harness.yaml`'s `runner.cert_fingerprint`
(bare hex or colon-separated both work; the client-side config loader
lowercases and strips colons before comparing). If you didn't capture it at
generation time, or want to confirm what's actually deployed matches, fetch
it remotely instead:

```bash
openssl s_client -connect 192.0.2.10:8090 </dev/null 2>/dev/null \
  | openssl x509 -fingerprint -sha256 -noout
```

## 4. Manual key generation

If you'd rather not use `cambia harness init`, or want to understand the
exact formats involved, here is the manual equivalent -- verified against
the real loading code, not assumed from the format description.

**Private key.** The client-side loader
(`cfr/src/harness/transport.py::load_ed25519_private_key`) accepts three
input shapes: a PEM PKCS8 key, a 32-byte raw seed, or a 64-byte raw Go
`ed25519.PrivateKey` (seed followed by public key; this is the shape the
service's own `auth.InitAndSave` writes, included for interop with that
code path). PEM is what `openssl genpkey` produces natively, so it's the
simplest manual choice -- no byte-level extraction needed:

```bash
openssl genpkey -algorithm ed25519 -out jwt_ed25519.pem
chmod 600 jwt_ed25519.pem
```

This file is used directly as `auth.private_key_path` in `harness.yaml`.
It must stay on the client workstation only, `chmod 600`, and must never be
copied to the runner.

**Public key.** `runnerd`'s loader (`runnerd/authtoken.Load`) is stricter:
it accepts *only* the raw 32-byte public key, rejecting PEM/DER outright
(`len(data) != ed25519.PublicKeySize` is a hard error). Extract the raw key
from the PEM-derived DER encoding; an Ed25519 `SubjectPublicKeyInfo` DER
blob is always a fixed 44 bytes with a fixed 12-byte header, so the last 32
bytes are always exactly the raw key, for any ed25519 key:

```bash
openssl pkey -in jwt_ed25519.pem -pubout -outform DER \
  | tail -c 32 > jwt_ed25519.pub
```

**Verification performed while writing this doc.** Both commands above
were run against a throwaway keypair and loaded with the actual production
code, not just eyeballed against the format description:

- `load_ed25519_private_key()` (the Python transport loader) successfully
  loaded both the PEM file directly and a 32-byte raw extraction of it, and
  `mint_token()` signed a real token with each.
- `authtoken.Load()` (the Go verify-only loader) successfully loaded the
  32-byte raw public key produced by the `tail -c 32` extraction, via a
  scoped `go test` run against `runnerd/authtoken` that called `Load` on
  the generated file and asserted no error and a 32-byte key.

## 5. Where files land

**On the runner host**, under the base directory's `keys/` subdirectory
(default `/srv/cambia/keys/`):

| File | Env var (systemd unit) | Contents |
|-|-|-|
| `keys/tls.crt` | `RUNNERD_TLS_CERT` | the self-signed certificate |
| `keys/tls.key` | `RUNNERD_TLS_KEY` | its private key |
| `keys/jwt_ed25519.pub` | `RUNNERD_JWT_PUBKEY` | the raw 32-byte JWT public key |

The runner never receives the TLS private key from anywhere else (it's
generated in place) and never receives the JWT private key at all.

**On the client workstation**, in `harness.yaml`:

| Field | Contents |
|-|-|
| `runner.url` | `https://<runner host>:8090` |
| `runner.cert_fingerprint` | the SHA256 fingerprint from section 3 |
| `auth.private_key_path` | path to the JWT private key (PEM or raw; either loader shape works) |

See [deploy.md](deploy.md) for how these pieces move from generation to
their final locations as part of a full deployment, and
[host-requirements.md](host-requirements.md) for what the runner host needs
to have installed (`openssl`) to generate its half of this material.

## 6. Token validation details

For reference, what `runnerd` actually checks on every Bearer token
(`runnerd/authtoken.Verify`):

- signing method is `EdDSA` (rejects any other algorithm outright, even if
  it would otherwise verify against the same key material)
- the `aud` claim is exactly `cambia-runnerd` -- a token minted for an
  unrelated purpose that happens to reuse the same keypair is rejected,
  because it won't carry this audience
- a non-empty string `sub` claim is present
- up to 30 seconds of clock-skew leeway is applied to `exp`/`nbf`/`iat`,
  so a few seconds of drift between the client's and runner's clocks
  doesn't spuriously reject a freshly minted token; the client additionally
  backdates `nbf` by 30 seconds when minting, as the other half of the same
  tolerance
