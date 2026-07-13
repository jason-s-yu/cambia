#!/usr/bin/env node
// D8 (cambia-490): cold-load transfer measurement for the remote-dev lanes.
//
// Crawls a running dev/preview server starting from `/`, discovers the entry
// scripts and stylesheets from the served HTML, then recursively walks the
// non-dynamic import graph of any JS it fetches (regex-based -- this is a rough
// crawler, not a real parser or bundler; dynamic `import()` calls are
// intentionally not followed, matching what a browser's initial cold-load
// waterfall actually requests before anything is interacted with).
//
// Zero new deps: only Node's built-in http/https/zlib/url modules. Node's
// global `fetch` (undici) is deliberately NOT used here -- it transparently
// decompresses gzip/br response bodies, which would hide the actual
// over-the-wire byte counts this script exists to measure. Raw sockets via
// `http`/`https` give us the untouched compressed bytes instead.
//
// Usage: node scripts/measure-dev-transfer.mjs [baseUrl]
//   node scripts/measure-dev-transfer.mjs http://localhost:5185   # dev / dev:remote
//   node scripts/measure-dev-transfer.mjs http://localhost:5186   # dev:remote-lite preview

/* global process, Buffer, console */

import http from 'node:http';
import https from 'node:https';
import zlib from 'node:zlib';
import { URL } from 'node:url';

const target = process.argv[2] || 'http://localhost:5173';
const baseUrl = new URL(target);

const BANDWIDTH_BYTES_PER_SEC = 2_000_000 / 8; // 2 Mbps down
const RTT_SECONDS = 0.15; // 150ms
const PARALLEL_CONNECTIONS = 6;
const BROTLI_QUALITY = 5; // matches the server's remote-mode compression setting

const SOURCEMAP_RE = /\/\/# sourceMappingURL=data:application\/json;base64,[A-Za-z0-9+/=]+\s*/g;

function rawRequest(pathname, { acceptEncoding } = {}) {
    return new Promise((resolve, reject) => {
        const lib = baseUrl.protocol === 'https:' ? https : http;
        const headers = { host: baseUrl.host };
        if (acceptEncoding) headers['accept-encoding'] = acceptEncoding;
        const req = lib.request(
            {
                hostname: baseUrl.hostname,
                port: baseUrl.port || (baseUrl.protocol === 'https:' ? 443 : 80),
                path: pathname,
                method: 'GET',
                headers,
            },
            (res) => {
                const chunks = [];
                res.on('data', (c) => chunks.push(c));
                res.on('end', () =>
                    resolve({
                        statusCode: res.statusCode,
                        headers: res.headers,
                        body: Buffer.concat(chunks),
                    })
                );
            }
        );
        req.on('error', reject);
        req.end();
    });
}

function decodeBody(body, contentEncoding) {
    if (contentEncoding === 'br') return zlib.brotliDecompressSync(body);
    if (contentEncoding === 'gzip') return zlib.gunzipSync(body);
    if (contentEncoding === 'deflate') return zlib.inflateSync(body);
    return body;
}

function extractHtmlRefs(html) {
    const refs = [];
    const scriptRe = /<script[^>]+src=["']([^"']+)["']/gi;
    const linkRe = /<link[^>]+rel=["'](?:stylesheet|modulepreload)["'][^>]*href=["']([^"']+)["']/gi;
    const linkRe2 = /<link[^>]+href=["']([^"']+)["'][^>]*rel=["'](?:stylesheet|modulepreload)["']/gi;
    let m;
    while ((m = scriptRe.exec(html))) refs.push(m[1]);
    while ((m = linkRe.exec(html))) refs.push(m[1]);
    while ((m = linkRe2.exec(html))) refs.push(m[1]);
    return refs;
}

// Static (not dynamic-import) specifiers: `import ... from "x"`, bare
// `import "x"`, and `export ... from "x"` (which also matches `from "x"`).
function extractJsImports(code) {
    const specs = [];
    const fromRe = /\bfrom\s+["']([^"']+)["']/g;
    const bareImportRe = /\bimport\s+["']([^"']+)["']/g;
    let m;
    while ((m = fromRe.exec(code))) specs.push(m[1]);
    while ((m = bareImportRe.exec(code))) specs.push(m[1]);
    return specs;
}

function resolveSpecifier(spec, fromPath) {
    if (/^https?:\/\//.test(spec) || spec.startsWith('data:')) return null;
    const base = new URL(fromPath, 'http://internal');
    const resolved = new URL(spec, base);
    return resolved.pathname + resolved.search;
}

function isJsLike(pathname, contentType) {
    if (contentType && /javascript/.test(contentType)) return true;
    return /\.(m?[tj]sx?)($|\?)/.test(pathname);
}

async function crawl() {
    const visited = new Set();
    const order = [];
    const records = new Map(); // path -> { uncompressedBytes, contentType, isJs }

    async function fetchAndRecord(path) {
        if (visited.has(path)) return;
        visited.add(path);
        order.push(path);

        const res = await rawRequest(path, {});
        if (res.statusCode >= 400) {
            records.set(path, { uncompressedBytes: res.body.length, contentType: '', isJs: false, missing: true });
            return;
        }
        const contentType = res.headers['content-type'] || '';
        const body = decodeBody(res.body, res.headers['content-encoding']);
        const text = body.toString('utf8');
        records.set(path, { uncompressedBytes: body.length, contentType, isJs: isJsLike(path, contentType), text });

        if (path === '/') {
            for (const ref of extractHtmlRefs(text)) {
                const resolved = resolveSpecifier(ref, '/index.html');
                if (resolved) await fetchAndRecord(resolved);
            }
            return;
        }

        if (isJsLike(path, contentType)) {
            for (const spec of extractJsImports(text)) {
                const resolved = resolveSpecifier(spec, path);
                if (resolved) await fetchAndRecord(resolved);
            }
        }
    }

    await fetchAndRecord('/');
    return { order, records };
}

async function measureCompressed(order) {
    let totalBytesWithAE = 0;
    let totalBytesWithoutAE = 0;
    for (const path of order) {
        const withAE = await rawRequest(path, { acceptEncoding: 'br, gzip' });
        const withoutAE = await rawRequest(path, {});
        totalBytesWithAE += withAE.body.length;
        totalBytesWithoutAE += withoutAE.body.length;
    }
    return { totalBytesWithAE, totalBytesWithoutAE };
}

function estimateSourcemapCompressedBytes(records) {
    let sourcemapCompressedEstimate = 0;
    let sourcemapRawBytes = 0;
    let rawTotalBytes = 0;
    for (const rec of records.values()) {
        if (!rec.text) continue;
        rawTotalBytes += rec.uncompressedBytes;
        const matches = rec.text.match(SOURCEMAP_RE);
        if (!matches || matches.length === 0) continue;
        const stripped = rec.text.replace(SOURCEMAP_RE, '');
        const fullCompressed = zlib.brotliCompressSync(Buffer.from(rec.text, 'utf8'), {
            params: { [zlib.constants.BROTLI_PARAM_QUALITY]: BROTLI_QUALITY },
        });
        const strippedCompressed = zlib.brotliCompressSync(Buffer.from(stripped, 'utf8'), {
            params: { [zlib.constants.BROTLI_PARAM_QUALITY]: BROTLI_QUALITY },
        });
        sourcemapCompressedEstimate += Math.max(0, fullCompressed.length - strippedCompressed.length);
        sourcemapRawBytes += matches.reduce((sum, m) => sum + Buffer.byteLength(m), 0);
    }
    return { sourcemapCompressedEstimate, sourcemapRawBytes, rawTotalBytes };
}

function projectColdLoadSeconds(requestCount, bytes) {
    return (requestCount / PARALLEL_CONNECTIONS) * RTT_SECONDS + bytes / BANDWIDTH_BYTES_PER_SEC;
}

function fmtBytes(n) {
    return `${n.toLocaleString()} B (${(n / 1024).toFixed(1)} KB)`;
}

async function main() {
    console.log(`Target: ${baseUrl.toString()}`);
    const { order, records } = await crawl();
    const { totalBytesWithAE, totalBytesWithoutAE } = await measureCompressed(order);
    const { sourcemapCompressedEstimate, sourcemapRawBytes, rawTotalBytes } = estimateSourcemapCompressedBytes(records);

    const sourcemapShareOfCompressed = totalBytesWithAE > 0 ? sourcemapCompressedEstimate / totalBytesWithAE : 0;
    const sourcemapShareOfRaw = rawTotalBytes > 0 ? sourcemapRawBytes / rawTotalBytes : 0;
    const projectedSeconds = projectColdLoadSeconds(order.length, totalBytesWithAE);

    console.log(`Requests: ${order.length}`);
    for (const path of order) {
        const rec = records.get(path);
        console.log(`  ${rec?.missing ? '[404] ' : ''}${path}${rec?.isJs ? ' (js)' : ''}`);
    }
    console.log(`Total bytes with Accept-Encoding br/gzip: ${fmtBytes(totalBytesWithAE)}`);
    console.log(`Total bytes without Accept-Encoding:      ${fmtBytes(totalBytesWithoutAE)}`);
    console.log(
        `Inline-sourcemap share of compressed bytes: ${(sourcemapShareOfCompressed * 100).toFixed(1)}% ` +
            `(est. ${fmtBytes(sourcemapCompressedEstimate)} of ${fmtBytes(totalBytesWithAE)})`
    );
    console.log(
        `Inline-sourcemap share of raw bytes:         ${(sourcemapShareOfRaw * 100).toFixed(1)}% ` +
            `(${fmtBytes(sourcemapRawBytes)} of ${fmtBytes(rawTotalBytes)})`
    );
    console.log(
        `Projected cold load @ 2Mbps/150ms RTT/6 connections: ${projectedSeconds.toFixed(2)}s ` +
            `(requests/${PARALLEL_CONNECTIONS} * RTT + bytes/bandwidth)`
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
