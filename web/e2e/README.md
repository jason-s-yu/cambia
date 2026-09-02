# End-to-end gameplay pass

Playwright suite for cambia-934, Phase 1's exit criterion: a game played through the shipped
client against a live service, plus one test per surface Phase 1 shipped.

This is not the render rig. `npm test` runs vitest against components in jsdom and answers "does
this component behave"; this suite drives Chromium against a real Go service, a real Postgres and a
real hub, and answers "is the surface reachable in the app, and does a whole game get to a result".

## Running it

Three things have to be up, in this order.

```sh
cd service && docker compose up -d      # Postgres and Redis
tools/e2e/launch-dev-service.sh         # the game service on :8088, detached, with a manifest
cd web && npm run test:e2e              # starts the Vite dev server itself, then runs the suite
```

`npm run test:e2e` starts the Vite dev server on :5180 when nothing already holds the port, and
reuses whatever is there when something does. It does not start the game service: that owns a
database and a queue, and guessing which build is answering on :8088 is how a pass ends up
reporting on code nobody meant to test. `globalSetup` refuses the run when :8088 is dead, has no
database, or is running without `CAMBIA_DEV_ACCOUNTS=1`, and names the launch script in the error.

Stop the service again with `tools/e2e/stop-dev-service.sh`.

Useful flags:

```sh
npm run test:e2e -- --headed                       # watch it play
npm run test:e2e -- -g "error boundary"            # one surface
npm run test:e2e -- --reporter=list gameplay-h2h   # one file
```

Artifacts (frame logs, screenshots, traces, the HTML report) land under `web/e2e/.artifacts/`,
which is git-ignored.

## What it covers

| File | Criterion | What it plays |
|-|-|-|
| `gameplay-h2h.spec.ts` | AC1 | Two seats, lobby to results, clean console |
| `gameplay-ffa4.spec.ts` | AC3 | Four seats, free-for-all, lobby to results |
| `phase1-surfaces.spec.ts` | AC2 | Error boundary, reconnect cap, forfeited seat, post-game host override, lobby nits, card and pile accessibility |

## How it is built

`support/seat.ts` opens one browser context per player, pinned with `?as=<name>`. One context per
seat, not one context with tabs: the identity pin lives in sessionStorage, but the auth cookie is
shared across a context, so seats in one context can silently become each other. Every raw failure
signal is recorded per seat (console, page errors, failed requests, 4xx and 5xx, socket closes,
every frame in both directions) and written to a log on every run, pass or fail, so a defect report
has frames to quote.

`support/game.ts` drives the client the way a player does: the dashboard's create-lobby dialog, the
lobby's own controls, the table's action column. Nothing reaches into a store or sends a WebSocket
frame by hand, so a green run is evidence about the shipped surface rather than about a harness
that agrees with it. Its bot draws from the stock and replaces a slot, which fires no ability with
`allowReplaceAbilities` off (the shipped default), so runs are comparable; ability prompts are
still handled, since the server's turn clock auto-discards for a seat that stalls and that can arm
one.

`support/instrument.ts` is the one exception, and it is opt-in per seat. It wraps `window.WebSocket`
in a Proxy that still constructs real sockets, which is what makes three things possible that the
client offers no seam for: counting dials (the only direct evidence a retry cap caps), refusing
dials (so a budget can be spent without taking the server down for the other seats), and replaying
the last server snapshot with a field bent out of shape (the render throw the error boundary
exists to catch). Faults are delivered as events on the socket object, never by closing it, so the
server's view of the game is untouched.

## Adding a test

Name the surface in the test title and say in a comment what the assertion is worth: which ticket
the behaviour comes from, and what the failure it prevents looks like. A test that only says an
element is visible does not survive its first refactor.

Seats are named `e2e-<surface>-<letter>`; keep the names distinct per spec so a lobby left over
from one test cannot seat itself in the next. The dev accounts are created on first use.
