// Copy guard (cambia-927): fails the build on any em dash (U+2014) under
// web/src, web/index.html, and web/DESIGN.md. CLAUDE.md bans em dashes in
// all generated text (UI copy, comments, docs); this is the mechanical
// backstop so a future edit can't reintroduce one silently.
//
// Run standalone:  npm run check-copy
// Also runs as the last step of `npm run build` (see package.json).

/* global process, console */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const EM_DASH = '—';

const TARGETS = [join(ROOT, 'src'), join(ROOT, 'index.html'), join(ROOT, 'DESIGN.md')];

/** Recursively collects file paths under `path` (or returns `[path]` for a plain file). */
function collectFiles(path) {
    const st = statSync(path, { throwIfNoEntry: false });
    if (!st) return [];
    if (st.isFile()) return [path];
    if (!st.isDirectory()) return [];
    const out = [];
    for (const entry of readdirSync(path)) {
        if (entry === 'node_modules' || entry === 'dist') continue;
        out.push(...collectFiles(join(path, entry)));
    }
    return out;
}

const files = TARGETS.flatMap(collectFiles);
const hits = [];
for (const file of files) {
    const text = readFileSync(file, 'utf8');
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes(EM_DASH)) {
            hits.push(`${relative(ROOT, file)}:${i + 1}: ${lines[i].trim()}`);
        }
    }
}

if (hits.length > 0) {
    console.error(`found ${hits.length} em dash (U+2014) occurrence(s):\n`);
    console.error(hits.join('\n'));
    process.exit(1);
}

console.log(`check-copy: 0 em dashes across ${files.length} files`);
process.exit(0);
