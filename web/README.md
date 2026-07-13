# Cambia web client

React 19 + TypeScript + Vite + Tailwind CSS v4 frontend for Cambia. See the
root `CLAUDE.md` for the full monorepo layout and cross-component commands.

```bash
npm install
npm run dev       # Vite dev server, http://localhost:5180
npm run build     # tsc -b && vite build -> dist/
npm run lint      # ESLint
```

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

Known limitation: unlike remote dev, the lite lane's `VITE_API_URL`/
`VITE_WS_URL` come from `.env` unchanged (same as any production build),
not derived from the page origin -- Vite's build-time `define` replacement
uses esbuild's own validator, which rejects the non-literal
origin-derivation expression remote dev uses at serve time (`Invalid define
value (must be an entity name or JS literal)`). If `.env`'s values aren't
reachable from the remote client, point `CAMBIA_API_TARGET` at a target
that already matches what's baked in, or use `dev:remote` instead.

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
