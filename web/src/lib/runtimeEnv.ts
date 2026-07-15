// src/lib/runtimeEnv.ts
//
// Resolves the API/WS base URLs at runtime instead of baking a build-time
// literal into the bundle.
//
// The service sends no CORS headers, so same-origin deployment is the
// operating topology for both HTTP and WebSocket traffic. `VITE_API_URL`/
// `VITE_WS_URL` in `.env` are treated as an explicit override for
// split-origin deploys and only take effect when set to a non-empty string;
// otherwise both are derived from `window.location.origin` at runtime, so a
// stale or unset `.env` value can never poison a production build.
//
// The dev server (`npm run dev`, `npm run dev:remote`) instead forces
// origin-derivation unconditionally via a `define` text-replacement in
// vite.config.js (`command === 'serve'` only), so the same-origin dev proxy
// always wins there regardless of `.env` content. The fallback path below is
// what actually executes for real production builds and the
// `dev:remote-lite` build lane, where that `define` replacement does not
// apply (esbuild's build-time `define` validator rejects the non-literal
// origin-derivation expression -- see vite.config.js and README.md).
function nonEmpty(value: string | undefined): value is string {
	return typeof value === 'string' && value.trim() !== '';
}

const rawApiUrl = import.meta.env.VITE_API_URL as string | undefined;
const rawWsUrl = import.meta.env.VITE_WS_URL as string | undefined;

/** Base URL for REST API calls. Explicit override wins when set and non-empty; otherwise same-origin. */
export const API_URL: string = nonEmpty(rawApiUrl) ? rawApiUrl : window.location.origin;

/** Base URL for WebSocket connections. Explicit override wins when set and non-empty; otherwise same-origin (ws/wss). */
export const WS_URL: string = nonEmpty(rawWsUrl)
	? rawWsUrl
	: window.location.origin.replace(/^http/, 'ws');
