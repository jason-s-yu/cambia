# Cambia web client

React 19 + TypeScript + Vite + Tailwind CSS v4 frontend for Cambia. See the
root `CLAUDE.md` for the full monorepo layout and cross-component commands.

```bash
npm install
npm run dev       # Vite dev server, http://localhost:5180
npm run build     # tsc -b && vite build -> dist/
npm run lint      # ESLint
```

The dev server is also reachable over TLS at
`https://app.cambia.pangu.home.jasonyu.io` -- see "Staging hosts on pangu".

## Production URL configuration

`VITE_API_URL`/`VITE_WS_URL` in `.env` are an override, not the primary
config path. Same-origin deployment is the operating topology: the service
sends no CORS headers, so the API and the web client have to share an
origin anyway. `npm run build` (and the `dev:remote-lite` build lane below)
resolves both base URLs at **runtime** from `window.location.origin` (see
`src/lib/runtimeEnv.ts`) unless `.env` sets a non-empty override, in which
case the override wins -- for split-origin deploys where the API genuinely
lives on a different host than the client.

For a same-origin deployment (the default topology), leave both variables
empty or commented out in `.env`:

```
# VITE_API_URL=
# VITE_WS_URL=
```

A stale value here used to get baked as a literal into the production
bundle and would silently break once whatever it pointed at moved or was
reassigned to another tenant (cambia-495). It no longer can: the override
only applies when the value is a non-empty string, and a runtime module --
not a build-time `define` substitution -- is what reads it, so there is
nothing for `vite build`'s stricter `define` validation to reject.

The dev server (`npm run dev`, `npm run dev:remote`) is unaffected either
way -- it always derives origin unconditionally through the same-origin dev
proxy (see below), regardless of `.env`.

## Staging hosts on pangu

Caddy on pangu terminates TLS on :443 with per-host Let's Encrypt certs and
reverse-proxies to the local dev processes:

| Host | Proxies to | Serves |
|-|-|-|
| `https://app.cambia.pangu.home.jasonyu.io` | `172.17.0.1:5180` | `npm run dev` (this client) |
| `https://api.cambia.pangu.home.jasonyu.io` | `172.17.0.1:8088` | Go service, direct |
| `https://preview.cambia.pangu.home.jasonyu.io` | `172.17.0.1:3000` | compose nginx preview, optional |

Work against the **app** host. It carries the same-origin topology
`http://localhost:5180` already has: the browser only ever sees one origin,
and `/user`, `/lobby`, `/ws`, `/training`, and the rest reach the Go service
through the dev proxy. Pointing the client at the api host instead would make every call
cross-origin, and the service sends no CORS headers, so it would fail. The api
host is for hitting REST or a WebSocket by hand (curl, a WS client), not for
the client to talk to.

`app.cambia.pangu.home.jasonyu.io` is listed in `server.allowedHosts` in
`vite.config.js` (`DEV_ALLOWED_HOSTS`); without it Vite's DNS-rebinding guard
answers 403 to both page loads and the HMR WebSocket upgrade. Vite always
allows `localhost` and bare IPv4 literals on its own, so that list does not
affect local or LAN-IP access. Add another staging name to the same array.

HMR needs no extra configuration. `server.hmr` is left unset in the default
lane, so the HMR client derives its protocol and port from the URL the page
was loaded from: `wss://app.cambia.pangu.home.jasonyu.io` on :443 behind the
TLS front, `ws://localhost:5180` locally. Do not add a static
`hmr: { protocol: 'wss', clientPort: 443 }` block here: those values are
injected as literals for every client alike, so the TLS front would work and
localhost would break.

`COOKIE_SECURE` (service side) stays unset by default. Setting it to `true`
adds `Secure` + `SameSite=Lax` to `auth_token`, which the app host wants and
loopback tolerates -- browsers treat `http://localhost` and `http://127.0.0.1`
as trustworthy origins and will store and replay a `Secure` cookie there. A
LAN IP or tailnet name over plain http is not a trustworthy origin, so the
cookie is dropped silently and login appears to do nothing. Turn it on only
when the https host is the only lane in use.

## Dev sessions: several players in one browser

Identity is one HttpOnly `auth_token` cookie scoped to the host, so every tab
on an origin is the same player and a second seat used to need a second
hostname. A tab can instead pin its own identity: the JWT goes in
`sessionStorage`, and the tab sends it explicitly on every carrier (an
`Authorization: Bearer` header on REST, a second `cambia-token.<jwt>` entry in
the WebSocket subprotocol list). The service resolves an explicit token first
and falls back to the cookie, so an unpinned tab behaves exactly as it always
has.

The switcher is the pill at the bottom left, rendered only under
`import.meta.env.DEV`. `vite build` folds that flag to false and drops the
component with it, so its markup and copy are absent from a production bundle.
Open it with a click or Enter, close it with Escape. It offers:

- the current identity and whether it came from this tab or the shared session
- pinning a named dev account, chosen from the list or typed as a new name
- a fresh guest for this tab alone
- opening a new tab already signed in as this user
- unpinning, back to the shared session

Named dev accounts need the service run with `CAMBIA_DEV_ACCOUNTS=1`; without
it `/dev/session` does not exist, the panel says so, and guests still work.
A dev account name is 1 to 32 characters of `a-z`, `0-9`, underscore or dash,
and the account is idempotent per name (`alice` is always the same player).

Two URL forms do the same thing without the panel, which is what a browser
verifier uses:

```
http://localhost:5180/dashboard?as=alice   # pin the named dev account
http://localhost:5180/dashboard?as=guest   # pin a fresh guest
http://localhost:5180/lobby/<id>#tab=<jwt> # pin a token another tab handed over
```

Both are consumed once at boot, before the first `/user/me` probe, and taken
straight back out of the address bar. A request that cannot be honoured (the
flag is unset, the service is down) leaves the tab on the shared session and
says why in the switcher.

Tab semantics follow `sessionStorage`: duplicating a tab (Ctrl+click reload, or
"Duplicate tab") copies the pin, so the copy is the same player; a tab opened
fresh starts on the shared cookie. Closing the tab ends its pin. Logging out of
a pinned tab drops the pin and lands back on the cookie identity: it never
posts `/user/logout`, so the other tabs keep their session.

The older stopgap still works and needs nothing: `alice.localhost:5180` and
`bob.localhost:5180` are separate origins with separate cookie jars. It cannot
pin two tabs on one origin, and it does nothing for the staging vhost or for a
LAN device, which is what the switcher is for.

## Remote development

Three lanes for working over a tailnet or a tethered/hotspot link, where
round-trip time (not bandwidth) usually dominates a cold load.

| Lane | Command | Port | HMR | Best for |
|-|-|-|-|-|
| Normal dev | `npm run dev` | 5180 | yes | Same-machine or low-RTT LAN work |
| Remote dev | `npm run dev:remote` | 5180 | yes (ws) | Tailnet/tethered work, live editing |
| Lite lane | `npm run dev:remote-lite` | 5186 (preview) | no, hand-reload | Weakest links, fewest bytes |

All three proxy `/user`, `/lobby`, `/friends`, `/matchmaking`,
`/leaderboard`, `/training`, and `/ws` to the Go game server so the browser
only ever talks to one origin (the service sends no CORS headers). The
target defaults to `http://localhost:8088`; override with `CAMBIA_API_TARGET`:

```bash
CAMBIA_API_TARGET=http://localhost:8088 npm run dev:remote
```

### Remote dev (`npm run dev:remote`)

`vite --mode remote`. Same live-editing workflow as `npm run dev`, plus:

- **Compression** -- brotli (quality 5) preferred, gzip level 6 fallback,
  mounted as the first dev middleware so dev-served modules, HMR-adjacent
  assets, and proxied API responses all come back compressed. Below the
  compressible-type/~1KB threshold, or with no `Accept-Encoding` sent,
  responses pass through uncompressed. WebSocket upgrades (HMR, the `/ws`
  proxy) are handled on the raw `upgrade` event and are never touched by
  this middleware.
- **Fewer round trips** -- every client-imported runtime dependency is
  force-listed in `optimizeDeps.include` and the two entry modules
  (`src/main.tsx`, `src/App.tsx`) are warmed via `server.warmup`, so nothing
  triggers Vite's dependency-discovery reload mid-session.
- **No embedded sourcemaps** -- measured at ~43% of compressed cold-load
  bytes (see the measurements further down), so remote mode strips the
  embedded `//# sourceMappingURL=data:...` comment from served JS. Devtools can no
  longer map dev-served code back to original source locations in this mode;
  use `npm run dev` for that. (Pre-bundled dependency chunks keep their
  small external `.map`-file-reference comment -- that file is fetched
  lazily by devtools on demand, not as part of a normal page load, so it
  isn't part of this cost and isn't touched.)
- **Tailnet-reachable** -- `server.host` is `0.0.0.0` and `allowedHosts` is
  `true`, so a tailnet IP or MagicDNS name is accepted (Vite's default
  Host-header allowlist otherwise rejects non-localhost names).

To use it: run `npm run dev:remote` on the dev machine, then open
`http://<tailnet-ip-or-magicdns-name>:5180/` from the remote client.

HMR connects back over `ws://` on the same port by default (`clientPort`
left unset so the HMR client infers it from the page's own origin, which is
correct for a bare tailnet/tethered connection). If you front the server
with TLS (see `tailscale serve` below), HMR needs to be told explicitly to
reconnect over `wss://` on port 443 instead, since that's what the public
listener actually terminates -- edit the `hmr` settings in `vite.config.js` to
`{ protocol: 'wss', clientPort: 443 }` for that case.

### Lite lane (`npm run dev:remote-lite`)

`vite build --watch --mode development` (rebuilds `dist/` on file change) run
concurrently with `vite preview --mode remote` (serves `dist/` statically on
port 5186, same compression middleware as remote dev). This is the
fewest-bytes option: one HTML request, one JS bundle, one CSS bundle, no
per-module dev overhead, no HMR socket.

There is no live-reload client in this lane -- after editing, wait for the
`vite build --watch` rebuild to finish (console prints `built in Nms`), then
reload the page by hand.

Like any production build, the lite lane resolves its API/WS base URLs at
runtime rather than baking a `.env` literal into the bundle -- see
"Production URL configuration" below.

### Fronting with `tailscale serve`

Tailscale's HTTPS reverse proxy gives HTTP/2 multiplexing over the public
listener, which matters on a high-RTT link: plain HTTP/1.1 caps a browser at
~6 connections per origin, so on a lossy/high-latency path those 6 slots
queue up fast; HTTP/2 multiplexes many requests over one connection and
avoids that queuing entirely. Run on the dev machine (documented here, not
run by this ticket -- it's host-level shared infra):

```bash
tailscale serve --bg 5180
```

or, for the lite lane:

```bash
tailscale serve --bg 5186
```

`tailscale serve` terminates TLS itself, so the browser connects over
`https://<magicdns-name>/`. When fronted this way, HMR must be told to
reconnect over `wss://` port 443 instead of the raw dev port -- see the
`hmr` note above.

### Measuring transfer (`web/scripts/measure-dev-transfer.mjs`)

`node scripts/measure-dev-transfer.mjs <baseUrl>` crawls a running
server starting from `/`: it discovers entry scripts/stylesheets from the
served HTML, then recursively follows non-dynamic `import`/`from` specifiers
in any JS it fetches (a regex-based crawl, not a real parser -- dynamic
`import()` calls are intentionally not followed, matching what a browser's
initial cold-load waterfall actually requests). It reports total requests,
total bytes with and without compression, the share of compressed bytes
spent on embedded sourcemaps, and a projected cold-load time at 2 Mbps
down / 150 ms RTT / 6 parallel connections
(`requests/6 * RTT + bytes/bandwidth`).

Measured against this app (494 modules) on 2026-07-13:

| Lane | Requests | Bytes (compressed) | Bytes (uncompressed) | Sourcemap share | Projected cold load |
|-|-|-|-|-|-|
| `npm run dev` | 99 | 3,325,401 B | 3,325,401 B | 10.1% (of raw bytes: 30.3%) | 15.78s |
| `npm run dev:remote` | 101 | 454,002 B | 2,339,127 B | 0.0% | 4.34s |
| `dev:remote-lite` preview | 3 | 167,183 B | 535,298 B | 0.0% | 0.74s |

`npm run dev` has no compression middleware at all, so its "compressed"
column is measured with the same `Accept-Encoding` header but reflects an
uncompressed response either way; its sourcemap-compressed-share figure is
therefore a same-quality local brotli estimate divided by that uncompressed
total, not a real wire ratio -- the raw-byte share (30.3%) is the honest
number for that lane. `dev:remote`'s 0.0% share reflects the stripping
plugin actually removing the comments (confirmed directly via curl: no
`sourceMappingURL` in served `src/**` modules or dependency chunks).
