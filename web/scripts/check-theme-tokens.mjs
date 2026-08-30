// Token-layer check (cambia-845).
//
// Parses the built stylesheet, resolves the dark (:root) and light
// ([data-theme=light]) custom-property sets, and asserts that:
//   1. every semantic token resolves to a literal color,
//   2. dark and light differ on every one of them,
//   3. the tokens declared theme-stable (SHARED) resolve and stay identical, and
//   4. every documented text/ground pair, and the one state indicator drawn
//      straight on the felt, clears its contrast floor in both themes
//      (cambia-914, cambia-959).
//
// Run after `npm run build`:  npm run check-tokens

/* global process, console */

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
    'surface-0', 'surface-1', 'surface-2', 'surface-3', 'surface-inset', 'surface-disabled', 'surface-overlay',
    'surface-felt', 'surface-felt-deep', 'surface-selected',
    'text-on-felt-muted', 'border-on-felt',
    'text-primary', 'text-secondary', 'text-tertiary', 'text-disabled', 'text-inverse', 'text-on-gold',
    'border-subtle', 'border-default', 'border-strong', 'border-accent',
    'interactive-hover', 'interactive-active', 'interactive-selected', 'focus-ring-color',
    'accent-gold', 'accent-gold-hover', 'accent-gold-active', 'accent-gold-soft',
    'accent-green', 'accent-green-hover', 'accent-danger', 'accent-danger-hover',
    'status-success', 'status-success-bg', 'status-success-border',
    'status-danger', 'status-danger-bg', 'status-danger-border',
    'status-warning', 'status-warning-bg', 'status-warning-border',
    'status-info', 'status-info-bg', 'status-info-border',
    // The three tokens a face-down card is actually painted with (cambia-1096)
    // rode outside this list until cambia-1097, so the plate, its crosshatch and
    // its silhouette were the one part of the card language nothing checked
    // resolved or moved between themes. --card-back is the legacy name they
    // replaced, now aliased onto the plate.
    'card-face', 'card-face-edge', 'card-back', 'card-back-line',
    'card-back-fill', 'card-back-pattern', 'card-back-edge',
    'card-targetable-ring', 'suit-red',
    'tier-bronze', 'tier-silver', 'tier-gold', 'tier-platinum', 'tier-diamond', 'tier-master', 'tier-grandmaster'
];

// Semantic tokens that are deliberately theme-stable: text drawn on a fill that
// keeps its hue in both themes. They still have to resolve to a literal, but
// dark == light is the contract, not a drift. Listing them here is what stops
// the token count from quietly excluding them (cambia-876, DL-4 review F9).
const SHARED = ['text-on-green', 'text-on-danger'];

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

// The SHARED failures keep their own counter: folding them into `missing`
// printed a theme-stable failure under the semantic headline, where the count
// could exceed the number of semantic tokens the line claims to describe
// (cambia-892, DL-7 F4). The light side is checked for UNRESOLVED too, since
// light inherits dark's map and an unresolved pair matches itself.
let shared = 0;
let sharedFailed = 0;
for (const key of SHARED) {
    const name = '--' + key;
    const d = resolve(dark, dark[name]);
    const l = resolve(light, light[name]);
    if (d === undefined || l === undefined || String(d).startsWith('UNRESOLVED') || String(l).startsWith('UNRESOLVED')) {
        sharedFailed++;
        rows.push(`MISSING  ${name}  dark=${d} light=${l}`);
        continue;
    }
    if (d !== l) {
        sharedFailed++;
        rows.push(`SPLIT    ${name}  dark=${d} light=${l} (declared theme-stable)`);
        continue;
    }
    shared++;
    rows.push(`shared   ${name}  ${d}`);
}

console.log(rows.join('\n'));
console.log(`\n${SEMANTIC.length} semantic tokens checked: ${missing} unresolved, ${identical} identical across themes`);
console.log(`${SHARED.length} theme-stable tokens checked: ${shared} identical as declared, ${sharedFailed} unresolved or split`);

// ---- Contrast gate (cambia-914, DL-8 R8) ----
//
// The token layer documents which text tier sits on which ground; nothing
// checked that the pair was legible, so a light-theme CTA shipped with its
// label at 3.35:1 on the hover fill. Every pair DESIGN.md pairs by name is
// measured here with the WCAG 2.1 relative-luminance formula, in both themes.
//
// Floors: 4.5:1, the AA body-text minimum, everywhere except the disabled
// pairs. WCAG 1.4.3 exempts inactive controls from any contrast requirement;
// the house floor for them is 3:1, enough to read a disabled label ("Signing
// in", "Creating") without letting it compete with live text.
//
// The last pair is not text: it is the ring that marks a card or pile as a
// legal target, and it takes the 3:1 non-text floor of WCAG 1.4.11. It is
// measured because the state shipped as a 16% gold tint that came out at
// 1.03:1 on the felt, a state indicator that indicated nothing (cambia-959,
// review F1).
//
// Translucent foregrounds are composited over their ground first, so the
// on-felt tiers are measured as rendered rather than as declared. A translucent
// ground is measured the same way once the pair names the opaque surface it is
// painted on (`over`): the tint is flattened onto that surface, then the text is
// flattened onto the result. Without an `over` a translucent ground stays a hard
// failure, since there is nothing to measure it against (cambia-935, F5).
//
// --accent-green-hover is the one accent fill left out: no surface pairs text
// with it (the green accent appears only as an avatar disc), and its dark value
// sits at 4.07:1 under the shared near-white label. Pair text with it and it
// joins this list, with the fill deepening the way --accent-danger-hover did.

function parseColor(value) {
    const s = String(value).trim();
    // The minifier rewrites rgba() as 4- or 8-digit hex, so alpha arrives in
    // either notation.
    const h = s.match(/^#([0-9a-f]{3,8})$/i);
    if (h && [3, 4, 6, 8].includes(h[1].length)) {
        const d = h[1].length <= 4 ? h[1].split('').map((c) => c + c).join('') : h[1];
        const a = d.length === 8 ? parseInt(d.slice(6, 8), 16) / 255 : 1;
        return [parseInt(d.slice(0, 2), 16), parseInt(d.slice(2, 4), 16), parseInt(d.slice(4, 6), 16), a];
    }
    const f = s.match(/^rgba?\(([^)]+)\)$/i);
    if (f) {
        const parts = f[1].split(/[,/]+/).map((p) => p.trim()).filter(Boolean);
        if (parts.length < 3) return null;
        const [r, g, b] = parts.slice(0, 3).map(Number);
        const a = parts.length > 3 ? Number(parts[3]) : 1;
        if ([r, g, b, a].some((n) => Number.isNaN(n))) return null;
        return [r, g, b, a];
    }
    return null;
}

/** Flattens a translucent foreground onto an opaque ground. */
function composite(fg, bg) {
    const a = fg[3];
    return [0, 1, 2].map((i) => fg[i] * a + bg[i] * (1 - a));
}

function relativeLuminance([r, g, b]) {
    const lin = [r, g, b].map((c) => {
        const s = c / 255;
        return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

function contrast(fg, bg) {
    const a = relativeLuminance(fg);
    const b = relativeLuminance(bg);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

const PAIRS = [
    { fg: 'text-on-gold', bg: 'accent-gold', min: 4.5, use: 'primary CTA label' },
    { fg: 'text-on-gold', bg: 'accent-gold-hover', min: 4.5, use: 'primary CTA, hovered' },
    { fg: 'text-on-gold', bg: 'accent-gold-active', min: 4.5, use: 'primary CTA, pressed' },
    { fg: 'text-primary', bg: 'surface-1', min: 4.5, use: 'body copy on a card' },
    { fg: 'text-secondary', bg: 'surface-1', min: 4.5, use: 'labels on a card' },
    { fg: 'accent-gold-text', bg: 'surface-1', min: 4.5, use: 'gold as text on a card' },
    { fg: 'text-on-green', bg: 'surface-felt', min: 4.5, use: 'primary tier on the felt' },
    { fg: 'text-on-green', bg: 'accent-green', min: 4.5, use: 'label on an affirmative fill' },
    { fg: 'text-on-danger', bg: 'accent-danger', min: 4.5, use: 'Cambia call and destructive label' },
    { fg: 'text-on-danger', bg: 'accent-danger-hover', min: 4.5, use: 'Cambia call, hovered' },
    { fg: 'text-on-felt-muted', bg: 'surface-felt', min: 4.5, use: 'pile labels and counts' },
    { fg: 'text-disabled', bg: 'surface-disabled', min: 3, use: 'disabled control label (WCAG exempts it; house floor 3:1)' },
    { fg: 'text-disabled', bg: 'surface-2', min: 3, use: 'disabled text on a raised row (house floor 3:1)' },
    // The tertiary tier was the one text tier with no pair here, and it sat
    // under AA in both themes (cambia-935, F4). The three grounds below are the
    // ones it is drawn on, enumerated rather than bounded: contrast moves in
    // opposite directions per theme, so --surface-2 is the dark worst case of
    // the three and --surface-inset the light one, with --surface-0 and
    // --surface-1 inside that span. The claim stops there. Grounds outside the
    // list carry no tertiary text: --surface-3 is a button hover fill under
    // --text-primary, and the two selected grounds sit lighter still in dark,
    // so the metadata on them steps up to --text-secondary and is measured as
    // that tier below (cambia-935, R1).
    { fg: 'text-tertiary', bg: 'surface-1', min: 4.5, use: 'eyebrows and metadata on a card' },
    { fg: 'text-tertiary', bg: 'surface-2', min: 4.5, use: 'metadata on a raised row (the lightest ground it sits on in dark)' },
    { fg: 'text-tertiary', bg: 'surface-inset', min: 4.5, use: 'input placeholder (the darkest ground it sits on in light)' },
    // Tinted fills. A badge lands on a card or on a raised row, so the tint is
    // measured over both (cambia-935, F5).
    { fg: 'accent-gold-text', bg: 'accent-gold-soft', over: 'surface-1', min: 4.5, use: 'gold badge label on a card' },
    { fg: 'accent-gold-text', bg: 'accent-gold-soft', over: 'surface-2', min: 4.5, use: 'gold badge label on a raised row' },
    { fg: 'text-primary', bg: 'accent-gold-soft', over: 'surface-1', min: 4.5, use: 'selected text (::selection in index.css)' },
    // Selected grounds. The gold selection tint lands lighter than any neutral
    // dark surface, which put --text-tertiary at 3.74:1 on the active-turn seat
    // and 4.14:1 on the own leaderboard row; both now draw their metadata in
    // --text-secondary (cambia-935, R1). --surface-selected is the tint
    // flattened onto --surface-2 and painted opaque, so the seat stays solid
    // over the felt; the leaderboard keeps the tint live over --surface-1.
    { fg: 'text-primary', bg: 'surface-selected', min: 4.5, use: 'player name on the active-turn seat' },
    { fg: 'text-secondary', bg: 'surface-selected', min: 4.5, use: 'rating and hand size on the active-turn seat' },
    { fg: 'accent-gold-text', bg: 'surface-selected', min: 4.5, use: 'the turn label on the active-turn seat' },
    { fg: 'text-primary', bg: 'interactive-selected', over: 'surface-1', min: 4.5, use: 'own name on the leaderboard row' },
    { fg: 'text-secondary', bg: 'interactive-selected', over: 'surface-1', min: 4.5, use: 'rank and counts on the own leaderboard row' },
    { fg: 'accent-gold-text', bg: 'interactive-selected', over: 'surface-1', min: 4.5, use: 'a top-three rank on the own leaderboard row' },
    // Status tones (cambia-971). Every one of them is read as text at 11-13px,
    // so the floor is 4.5 and not the 3:1 large-text or non-text one; the pill
    // is bold, but bold reaches "large" only at 18.66px.
    //
    // Grounds, enumerated the way the tertiary tier is: the badge tint over the
    // two panel surfaces, then the worst flat ground per theme. A tone's own
    // tint pulls the ground toward the tone, so the tint pair over a surface
    // bounds the same tone drawn flat on it, which covers the opaque
    // --surface-1 chip on the felt (DsGameTable FeltChip). The flat grounds
    // that are not bounded that way are --surface-selected, the lightest ground
    // in dark, and --surface-inset, the darkest in light; both are measured in
    // both themes, and the resting --surface-2 seat sits inside that span.
    { fg: 'status-success', bg: 'status-success-bg', over: 'surface-1', min: 4.5, use: 'success badge label on a card' },
    { fg: 'status-success', bg: 'status-success-bg', over: 'surface-2', min: 4.5, use: 'success badge label on a raised row' },
    { fg: 'status-danger', bg: 'status-danger-bg', over: 'surface-1', min: 4.5, use: 'danger badge label on a card' },
    { fg: 'status-danger', bg: 'status-danger-bg', over: 'surface-2', min: 4.5, use: 'danger badge label on a raised row' },
    { fg: 'status-warning', bg: 'status-warning-bg', over: 'surface-1', min: 4.5, use: 'warning badge label on a card' },
    { fg: 'status-warning', bg: 'status-warning-bg', over: 'surface-2', min: 4.5, use: 'warning badge label on a raised row' },
    { fg: 'status-info', bg: 'status-info-bg', over: 'surface-1', min: 4.5, use: 'info badge label on a card' },
    { fg: 'status-info', bg: 'status-info-bg', over: 'surface-2', min: 4.5, use: 'info badge label on a raised row' },
    { fg: 'status-success', bg: 'surface-selected', min: 4.5, use: 'the Ready line on the active-turn seat' },
    { fg: 'status-danger', bg: 'surface-selected', min: 4.5, use: 'the Called Cambia line on the active-turn seat' },
    { fg: 'status-success', bg: 'surface-inset', min: 4.5, use: 'a positive delta in a score pill' },
    { fg: 'status-danger', bg: 'surface-inset', min: 4.5, use: 'a negative delta in a score pill' },
    { fg: 'text-secondary', bg: 'surface-2', min: 4.5, use: 'the stopped badge, the one status pill drawn on a neutral fill' },
    { fg: 'card-targetable-ring', bg: 'surface-felt', min: 3, use: 'targetable card and pile ring (non-text indicator, WCAG 1.4.11 3:1)' }
];

let contrastFailed = 0;
const contrastRows = [];
for (const theme of ['dark', 'light']) {
    const map = theme === 'dark' ? dark : light;
    for (const pair of PAIRS) {
        const fgRaw = resolve(map, map['--' + pair.fg]);
        const bgRaw = resolve(map, map['--' + pair.bg]);
        const overRaw = pair.over ? resolve(map, map['--' + pair.over]) : null;
        const fg = parseColor(fgRaw);
        const bg = parseColor(bgRaw);
        const over = pair.over ? parseColor(overRaw) : null;
        const label = `${theme.padEnd(5)} ${pair.fg} on ${pair.bg}${pair.over ? ` over ${pair.over}` : ''}`;
        if (!fg || !bg || (pair.over && !over)) {
            contrastFailed++;
            contrastRows.push(`UNPARSED ${label}  fg=${fgRaw} bg=${bgRaw}${pair.over ? ` over=${overRaw}` : ''}`);
            continue;
        }
        let ground = bg;
        if (bg[3] !== 1) {
            if (!over) {
                contrastFailed++;
                contrastRows.push(`TRANSLUCENT-BG ${label}  bg=${bgRaw} (a translucent ground needs an \`over\` surface to be measured)`);
                continue;
            }
            if (over[3] !== 1) {
                contrastFailed++;
                contrastRows.push(`TRANSLUCENT-OVER ${label}  over=${overRaw} (the surface a tint is flattened onto has to be opaque)`);
                continue;
            }
            ground = [...composite(bg, over), 1];
        } else if (over) {
            // An `over` on an opaque ground is a stale declaration, not a
            // no-op: the pair reads as measured on a tint it no longer has.
            contrastFailed++;
            contrastRows.push(`OPAQUE-BG ${label}  bg=${bgRaw} (\`over\` belongs on a translucent ground only)`);
            continue;
        }
        const ratio = contrast(fg[3] === 1 ? fg : composite(fg, ground), ground);
        const shown = ratio.toFixed(2).padStart(5);
        if (ratio < pair.min) {
            contrastFailed++;
            contrastRows.push(`FAIL ${shown}:1 (min ${pair.min}) ${label}  -- ${pair.use}`);
        } else {
            contrastRows.push(`ok   ${shown}:1 (min ${pair.min}) ${label}`);
        }
    }
}
console.log('\n' + contrastRows.join('\n'));
console.log(`\n${PAIRS.length * 2} contrast pairs checked (${PAIRS.length} per theme): ${contrastFailed} under the floor`);

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

const failed = missing > 0 || identical > 0 || sharedFailed > 0 || contrastFailed > 0 || serif.length > 0 || banned.length > 0 || !hasArchivo || hasRemoteFont;
process.exit(failed ? 1 : 0);
