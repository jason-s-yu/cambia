// Token-layer check (cambia-845).
//
// Parses the built stylesheet, resolves the dark (:root) and light
// ([data-theme=light]) custom-property sets, and asserts that:
//   1. every semantic token resolves to a literal color, and
//   2. dark and light differ on every one of them.
//
// Run after `npm run build`:  node scripts/check-theme-tokens.mjs
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const assetsDir = join(process.cwd(), 'dist', 'assets');
const cssFile = readdirSync(assetsDir).find((f) => f.endsWith('.css'));
if (!cssFile) {
    console.error('no built CSS found in dist/assets; run `npm run build` first');
    process.exit(1);
}
const css = readFileSync(join(assetsDir, cssFile), 'utf8');

/** Collects `--name: value` pairs from every rule whose selector list contains `selector`. */
function collect(selector) {
    const out = {};
    const re = /([^{}]+)\{([^{}]*)\}/g;
    let m;
    while ((m = re.exec(css))) {
        const sel = m[1].trim();
        const parts = sel.split(',').map((s) => s.trim());
        if (!parts.includes(selector)) continue;
        for (const decl of m[2].split(';')) {
            const i = decl.indexOf(':');
            if (i < 0) continue;
            const name = decl.slice(0, i).trim();
            if (!name.startsWith('--')) continue;
            out[name] = decl.slice(i + 1).trim();
        }
    }
    return out;
}

const dark = collect(':root');
const light = { ...dark, ...collect('[data-theme=light]'), ...collect('[data-theme="light"]') };

/** Follows var() chains within one theme's map until a literal remains. */
function resolve(map, value, depth = 0) {
    if (depth > 12 || typeof value !== 'string') return value;
    const m = value.match(/^var\((--[a-z0-9-]+)\)$/i);
    if (!m) return value;
    const next = map[m[1]];
    if (next === undefined) return `UNRESOLVED(${m[1]})`;
    return resolve(map, next, depth + 1);
}

const SEMANTIC = [
    'surface-0', 'surface-1', 'surface-2', 'surface-3', 'surface-inset', 'surface-overlay',
    'surface-felt', 'surface-felt-deep',
    'text-primary', 'text-secondary', 'text-tertiary', 'text-disabled', 'text-inverse', 'text-on-gold',
    'border-subtle', 'border-default', 'border-strong', 'border-accent',
    'interactive-hover', 'interactive-active', 'interactive-selected', 'focus-ring-color',
    'accent-gold', 'accent-gold-hover', 'accent-gold-active', 'accent-gold-soft',
    'accent-green', 'accent-green-hover', 'accent-danger', 'accent-danger-hover',
    'status-success', 'status-success-bg', 'status-success-border',
    'status-danger', 'status-danger-bg', 'status-danger-border',
    'status-warning', 'status-warning-bg', 'status-warning-border',
    'status-info', 'status-info-bg', 'status-info-border',
    'card-face', 'card-face-edge', 'card-back', 'card-back-line', 'suit-red',
    'tier-bronze', 'tier-silver', 'tier-gold', 'tier-platinum', 'tier-diamond', 'tier-master', 'tier-grandmaster'
];

let missing = 0;
let identical = 0;
const rows = [];
for (const key of SEMANTIC) {
    const name = '--' + key;
    const d = resolve(dark, dark[name]);
    const l = resolve(light, light[name]);
    if (d === undefined || l === undefined || String(d).startsWith('UNRESOLVED') || String(l).startsWith('UNRESOLVED')) {
        missing++;
        rows.push(`MISSING  ${name}  dark=${d} light=${l}`);
        continue;
    }
    if (d === l) {
        identical++;
        rows.push(`SAME     ${name}  ${d}`);
        continue;
    }
    rows.push(`ok       ${name}  dark=${d}  light=${l}`);
}

console.log(rows.join('\n'));
console.log(`\n${SEMANTIC.length} semantic tokens checked: ${missing} unresolved, ${identical} identical across themes`);

// Font check: no serif family and none of the banned faces may survive
// anywhere in the built CSS, the UI face must be the self-hosted one, and no
// font may be fetched from a remote host. The `serif` lookbehind lets the
// generic `sans-serif` fallback keyword through and nothing else.
const serif = css.match(/Young Serif|Alegreya|Playfair|Lora|Fraunces|Crimson|Georgia|Garamond|(?<!sans-)\bserif\b/gi) || [];
// Verbatim from web/DESIGN.md "Banned faces".
const BANNED = [
    'Inter', 'Manrope', 'Space Grotesk', 'Plus Jakarta Sans', 'Outfit', 'Sora', 'Poppins',
    'Montserrat', 'Lexend', 'Figtree', 'Geist', 'Urbanist', 'Epilogue', 'Nunito', 'Work Sans'
];
const banned = BANNED.filter((f) => new RegExp('["\'\\s]' + f + '["\',]', 'i').test(css));
const hasArchivo = /Archivo Variable/.test(css);
const hasRemoteFont = /fonts\.googleapis\.com|fonts\.gstatic\.com/.test(css);
console.log(`serif references: ${serif.length ? serif.join(', ') : 'none'}`);
console.log(`banned faces present: ${banned.length ? banned.join(', ') : 'none'}`);
console.log(`Archivo Variable declared: ${hasArchivo}`);
console.log(`remote font host referenced: ${hasRemoteFont}`);

const failed = missing > 0 || identical > 0 || serif.length > 0 || banned.length > 0 || !hasArchivo || hasRemoteFont;
process.exit(failed ? 1 : 0);
