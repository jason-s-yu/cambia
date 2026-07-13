import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'; // Use the Tailwind CSS v4 Vite plugin
import tsconfigPaths from 'vite-tsconfig-paths';
// [https://vitejs.dev/config/](https://vitejs.dev/config/)
export default defineConfig(({ command }) => ({
    plugins: [
        tsconfigPaths(), // Enables tsconfig path aliases like @/*
        tailwindcss(), // Integrates Tailwind CSS v4 via its Vite plugin
        react() // Standard React plugin
    ],
    // Dev only: derive API/WS bases from the page origin so requests stay
    // same-origin through the dev proxy below, from any host (localhost or
    // tailnet). Production builds keep the .env VITE_* values untouched.
    define: command === 'serve'
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
        // Same-origin proxy to the Go game server: the service sends no CORS
        // headers, so the browser must see one origin. CAMBIA_API_TARGET
        // overrides the target (the server's default :8080 and much of the
        // 80xx range are held by co-tenants here; dev runs use PORT=8088).
        proxy: Object.fromEntries(
            ['/user', '/lobby', '/friends', '/matchmaking', '/leaderboard', '/training', '/ws'].map((path) => [
                path,
                {
                    target: process.env.CAMBIA_API_TARGET || 'http://localhost:8088',
                    changeOrigin: false,
                    ws: path === '/ws',
                },
            ])
        ),
    },
}));
