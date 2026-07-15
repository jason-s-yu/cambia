import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'; // Use the Tailwind CSS v4 Vite plugin
import tsconfigPaths from 'vite-tsconfig-paths';
import compression from 'compression';
import zlib from 'node:zlib';
// [https://vitejs.dev/config/](https://vitejs.dev/config/)

// Runtime deps actually imported by client-side code (src/**), force-included
// in optimizeDeps for remote mode so nothing dep-discovers mid-session over a
// high-RTT tailnet link (each discovery round trip costs a full RTT before
// the module graph can continue). `framer-motion` is declared as a dependency
// in package.json but is not currently imported anywhere under src, so it is
// left out; `immer` is only reached indirectly via `zustand/middleware/immer`
// but is listed explicitly since it is a direct package.json dependency.
const CLIENT_RUNTIME_DEPS = [
    'axios',
    'date-fns',
    'immer',
    'react',
    'react-dom',
    'react-router-dom',
    'uplot',
    'uplot-react',
    'uuid',
    'zustand',
];

// Same-origin proxy paths to the Go game server: the service sends no CORS
// headers, so the browser must see one origin. CAMBIA_API_TARGET overrides
// the target (the server's default :8080 and much of the 80xx range are held
// by co-tenants here; dev runs use PORT=8088). Shared between the dev server
// (always on) and the remote-mode preview server (D7 lite lane).
const API_PROXY_PATHS = ['/user', '/lobby', '/friends', '/matchmaking', '/leaderboard', '/training', '/ws'];
function apiProxy() {
    return Object.fromEntries(
        API_PROXY_PATHS.map((path) => [
            path,
            {
                target: process.env.CAMBIA_API_TARGET || 'http://localhost:8088',
                changeOrigin: false,
                ws: path === '/ws',
            },
        ])
    );
}

// Remote mode only: mounts brotli/gzip compression as the first dev (or
// preview) middleware, ahead of Vite's own stack (proxy, transform, file
// serving). `enforce: 'pre'` plus a non-post configureServer/configurePreviewServer
// hook both run before Vite's internal middlewares are installed, so this is
// first in the chain and every downstream response -- dev-served modules and
// proxied API responses alike -- passes through it. WebSocket upgrades (HMR,
// the /ws proxy) are handled on the raw `upgrade` event, never through this
// connect middleware stack, so they are untouched by construction.
function remoteCompressionPlugin() {
    const middleware = compression({
        threshold: 1024, // ~1KB; compression package also skips non-compressible mime types
        level: 6, // gzip fallback
        brotli: {
            params: {
                [zlib.constants.BROTLI_PARAM_QUALITY]: 5, // balanced ratio/CPU for dev use
            },
        },
    });
    return {
        name: 'cambia-remote-compression',
        enforce: 'pre',
        configureServer(server) {
            server.middlewares.use(middleware);
        },
        configurePreviewServer(server) {
            server.middlewares.use(middleware);
        },
    };
}

// Remote mode only: measured cold-load share of embedded sourcemaps was ~43%
// of compressed bytes (well over the ~25% bar; see README "Remote
// development" section), so this middleware strips them. A `transform` hook
// returning `map: null` does not work here: Vite's own later-registered
// internal plugins (import rewriting for both src modules and pre-bundled dep
// chunks) still attach their own map after any user post-hook runs, and the
// `//# sourceMappingURL=data:...` comment itself is only rendered into the
// response body at send time, not during transform. Patching res.write/
// res.end instead operates on the literal bytes about to go out, after
// everything upstream (Vite core included) has had its say. Mounted after
// the compression middleware below so it runs first at write time (last
// patched = outermost = runs first), stripping the comment before the
// remaining text gets compressed. Only touches JS responses (by
// Content-Type), so proxied API/JSON traffic is untouched. Trades away
// devtools' ability to map dev-served code back to original source
// locations in remote mode; never applied to default `npm run dev`.
function remoteStripSourcemapsPlugin() {
    const stripRe = /\n?\/\/# sourceMappingURL=data:application\/json;base64,[A-Za-z0-9+/=]+\s*/g;
    function stripChunk(chunk) {
        if (chunk == null) return chunk;
        const isBuffer = Buffer.isBuffer(chunk);
        const text = isBuffer ? chunk.toString('utf8') : chunk;
        stripRe.lastIndex = 0;
        if (!stripRe.test(text)) return chunk;
        stripRe.lastIndex = 0;
        const stripped = text.replace(stripRe, '');
        return isBuffer ? Buffer.from(stripped, 'utf8') : stripped;
    }
    function middleware(req, res, next) {
        const _write = res.write;
        const _end = res.end;
        function isJsResponse() {
            const type = res.getHeader('Content-Type');
            return typeof type === 'string' && /javascript/.test(type);
        }
        res.write = function (chunk, ...rest) {
            if (isJsResponse()) chunk = stripChunk(chunk);
            return _write.call(this, chunk, ...rest);
        };
        res.end = function (chunk, ...rest) {
            if (chunk != null && isJsResponse()) chunk = stripChunk(chunk);
            return _end.call(this, chunk, ...rest);
        };
        next();
    }
    return {
        name: 'cambia-remote-strip-sourcemaps',
        configureServer(server) {
            server.middlewares.use(middleware);
        },
        configurePreviewServer(server) {
            server.middlewares.use(middleware);
        },
    };
}

export default defineConfig(({ command, mode }) => {
    const isRemote = mode === 'remote';
    // Dev-server only. Broadening this to also cover `vite build --watch
    // --mode development` (the dev:remote-lite build step, D7) was tried and
    // reverted: esbuild's own `define` validation (used for CJS/ESM interop
    // during the Rollup build, not just Vite's more lenient dev-serve
    // replacement) rejects non-literal expression values like the
    // `.replace(...)` call below with "Invalid define value (must be an
    // entity name or JS literal)". The lite lane's built output therefore
    // bakes in .env's literal VITE_API_URL/VITE_WS_URL same as any
    // production build -- see the README "Remote development" section.
    const useOriginDerivedEnv = command === 'serve';

    return {
        plugins: [
            tsconfigPaths(), // Enables tsconfig path aliases like @/*
            tailwindcss(), // Integrates Tailwind CSS v4 via its Vite plugin
            react(), // Standard React plugin
            ...(isRemote ? [remoteCompressionPlugin(), remoteStripSourcemapsPlugin()] : []),
        ],
        // Dev only: derive API/WS bases from the page origin so requests stay
        // same-origin through the dev proxy below, from any host (localhost or
        // tailnet), overriding .env unconditionally. Production builds and the
        // dev:remote-lite build lane instead resolve API/WS bases at runtime
        // (see src/lib/runtimeEnv.ts): .env's VITE_API_URL/VITE_WS_URL win only
        // when set to a non-empty string, otherwise the built bundle derives
        // same-origin from window.location at load time. That runtime fallback
        // is why a stale .env literal no longer gets baked into prod builds.
        define: useOriginDerivedEnv
            ? {
                  'import.meta.env.VITE_API_URL': 'globalThis.location.origin',
                  'import.meta.env.VITE_WS_URL': '(globalThis.location.origin.replace(/^http/, "ws"))',
              }
            : {},
        server: {
            host: '0.0.0.0',
            // Pinned so VITE_API_URL/VITE_WS_URL in .env stay valid; 5173/5174 are
            // taken by co-tenant dev servers on this host.
            port: 5180,
            strictPort: true,
            proxy: apiProxy(),
            ...(isRemote
                ? {
                      // Tailnet clients arrive via their tailnet IP or MagicDNS
                      // name, never 'localhost' -- Vite's Host-header allowlist
                      // (DNS-rebinding protection) rejects those by default.
                      allowedHosts: true,
                      hmr: {
                          protocol: 'ws',
                          // clientPort left at its origin-relative default: the
                          // HMR client infers the port from the page it loaded
                          // from, which is correct for a bare tailnet/tethered
                          // connection. Fronting with TLS (tailscale serve, see
                          // README) requires protocol: 'wss' and clientPort: 443
                          // instead, since the public listener terminates TLS on
                          // 443 and reverse-proxies to this port.
                      },
                      // Force-warm the two entry modules so their import graph
                      // is walked and ready before the first browser request,
                      // instead of resolving lazily module-by-module.
                      warmup: {
                          clientFiles: ['./src/main.tsx', './src/App.tsx'],
                      },
                  }
                : {}),
        },
        // Only used by `vite preview` (D7 lite lane); the dev server reads
        // `server` above instead, so this has no effect on `npm run dev`.
        preview: isRemote
            ? {
                  port: 5186,
                  host: '0.0.0.0',
                  allowedHosts: true,
                  proxy: apiProxy(),
              }
            : {},
        ...(isRemote
            ? {
                  optimizeDeps: {
                      include: CLIENT_RUNTIME_DEPS,
                  },
              }
            : {}),
    };
});
