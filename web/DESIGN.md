# Cambia design language

Modern flat card-room style. Deep green-tinted neutral grounds in dark, warm paper neutrals in light, one gold accent family, one sans, 1px borders, restrained shadows.

Implementation: `src/styles/design-tokens.css` (tokens), `src/index.css` (globals and the Tailwind `dark:` binding), `src/components/ds/**` (primitives). Verification: `npm run build && npm run check-tokens`.

## Rules

No gradients as decoration. No glassmorphism, blur panels, or neon. No drop shadow on a resting surface: a card separates from its ground by the surface step plus a 1px border. Shadows belong only to layers that are genuinely detached (dialog, menu, dragged card). No raw hex or rgb in a component; every color comes from a token. No serif anywhere.

## Themes

Dark is the default and lives on `:root`, so an unset `data-theme` attribute renders dark and a first load has no flash. Light is an explicit override under `[data-theme="light"]` that redefines the same semantic names. `hooks/useTheme` resolves the stored preference (`dark` default, `light`, `system`) and writes the attribute on `<html>`; `index.html` ships `data-theme="dark"` as the pre-mount value.

Components read semantic names only. They never branch on theme, and never read a primitive ramp.

Token layers, in `design-tokens.css` order:

1. Primitive ramps (`--felt-*`, `--paper-*`, `--ink-*`, `--green-*`, `--gold-*`, `--cream-*`, `--red-*`, `--amber-*`, `--blue-*`). Raw hues. Not for component use.
2. Semantic tokens. The component contract, listed below.
3. Scale tokens: type, spacing, radius, elevation, motion, control geometry.
4. Legacy aliases. Deprecated, see "Removal checklist".

Names that would collide with a Tailwind v4 default theme key carry a `--ds-` prefix (`--ds-text-sm`, `--ds-radius-lg`, `--ds-ease-out`, `--ds-font-mono`) so both scales coexist. The `@theme` block at the foot of the file exposes the semantic tokens as Tailwind utilities (`bg-surface-1`, `text-text-secondary`, `border-border-default`, `rounded-ds-md`) with a `-ds-` infix where needed.

### Surfaces

|Token|Dark|Light|Use|
|-|-|-|-|
|`--surface-0`|`#0b1210`|`#f1eee6`|App ground. The page background, behind everything.|
|`--surface-1`|`#101a16`|`#fbf9f5`|Default card, panel, top bar.|
|`--surface-2`|`#16231e`|`#ffffff`|Raised: row hover, table header, modal footer, chips.|
|`--surface-3`|`#1c2c26`|`#ffffff`|Highest: menu, popover, hovered raised control.|
|`--surface-inset`|`#070d0b`|`#e9e5db`|Wells: text inputs, log panes, score pills.|
|`--surface-disabled`|`#16231e`|`#e9e5db`|Fill of a disabled control. Dark raises one step; light recesses to the well tone, because its raised step is white and a white fill on warm paper reads as a hole in the card.|
|`--surface-overlay`|`rgba(4,8,7,.72)`|`rgba(22,33,29,.42)`|Scrim behind a dialog.|
|`--surface-felt`|`#14563a`|`#1a6d49`|The game table.|
|`--surface-felt-deep`|`#0f3d2a`|`#14563a`|Table rail and vignette.|

Surfaces step in one direction only. A panel on `--surface-1` puts its nested rows on `--surface-2`, never back down to `--surface-0`.

The felt is the only saturated ground and the only surface that stays green in both themes. The rail is a solid ring of `--surface-felt-deep` around the felt with a 1px border on each edge; there is no gradient vignette. Chips that sit on the felt (a notice, a seat) use `--surface-1` or `--surface-2` with the standard border, so they read as pieces placed on the table rather than as part of it.

### On-felt

Text and hairlines drawn directly on the felt cannot use the neutral tiers, which flip with the theme. These two stay light-on-green in both themes and only shift alpha (cambia-848).

|Token|Dark|Light|Use|
|-|-|-|-|
|`--text-on-felt-muted`|`rgba(240,247,243,.68)`|`rgba(240,247,243,.86)`|Pile labels and counts, secondary lines on the felt. `--text-on-green` is the primary tier.|
|`--border-on-felt`|`rgba(240,247,243,.22)`|`rgba(240,247,243,.30)`|Empty pile slot outline, the felt's own 1px edge.|

### Text tiers

|Token|Dark|Light|Use|
|-|-|-|-|
|`--text-primary`|`#edf2ef`|`#16211d`|Body copy, headings, values.|
|`--text-secondary`|`#a9b8b1`|`#3f4d47`|Labels, supporting lines, inactive nav.|
|`--text-tertiary`|`#7e8f87`|`#5b6962`|Eyebrows, units, metadata, placeholders. Holds AA on the card, the raised row and the input ground.|
|`--text-disabled`|`#63726b`|`#76827b`|Disabled control text. Holds 3:1 on `--surface-disabled` in both themes.|
|`--text-inverse`|`#0b1210`|`#fbf9f5`|Text on a filled neutral of the opposite theme.|
|`--text-on-gold`|`#070d0b`|`#241c05`|Text and icons on `--accent-gold`.|
|`--text-on-danger`|`#fdf1ef`|`#fdf1ef`|Text on `--accent-danger`.|
|`--text-on-green`|`#f0f7f3`|`#f0f7f3`|Text on `--accent-green` and the felt.|

### Borders

|Token|Dark|Light|Use|
|-|-|-|-|
|`--border-subtle`|`rgba(237,242,239,.07)`|`rgba(22,33,29,.09)`|Hairline dividers inside a surface.|
|`--border-default`|`rgba(237,242,239,.14)`|`rgba(22,33,29,.17)`|The standard 1px component border.|
|`--border-strong`|`rgba(237,242,239,.26)`|`rgba(22,33,29,.32)`|Emphasis, container under attention.|
|`--border-accent`|`#c9a227`|`#ab8526`|Gold-outlined surface: primary queue, selected row.|

### Interactive states

|Token|Dark|Light|Use|
|-|-|-|-|
|`--interactive-hover`|`rgba(237,242,239,.06)`|`rgba(22,33,29,.05)`|Hover wash on a transparent or ghost control.|
|`--interactive-active`|`rgba(237,242,239,.11)`|`rgba(22,33,29,.10)`|Pressed wash.|
|`--interactive-selected`|`rgba(201,162,39,.14)`|`rgba(171,133,38,.14)`|Selected or active-turn tint.|
|`--surface-selected`|`#2f351f`|`#f3eee1`|The same tint flattened onto `--surface-2`, opaque. For a resting surface that must not let its ground through: the active-turn player seat sits on the felt.|
|`--focus-ring-color`|`#dcb84a`|`#ab8526`|The focus ring hue.|
|`--focus-ring`|composed|composed|Ready-made `box-shadow`: 2px ground gap, then 2px ring.|

Every focusable control shows a focus ring. Keyboard focus falls back to the global `:focus-visible` outline in `index.css`; controls that already own their `box-shadow` (Input, Select) set `--focus-ring` instead.

A dialog takes focus into its panel on open, never onto the close control. Where it lands is the call site's choice (`initialFocus` on `ds/core/Modal`: `confirm`, `dismiss`, `body`, `panel`, or a ref); the default is the first action-row control that is not destructive. The danger button variants carry `data-destructive`, so the rule reads a marker rather than a label or a position.

Press states change color, never position. The previous language translated buttons downward on press; the flat language does not move them.

A disabled control is a neutral fill, never a faded accent: `--surface-disabled` with `--border-default` and `--text-disabled`. Gold at half opacity still reads as the CTA and put its label at 1.15:1 in light. `ds/core/Button` and `common/Button` own this; a ghost control keeps its transparent shell and only drops to `--text-disabled`.

### Accents

One family. Gold is the CTA and the highlight; green is the table and the affirmative secondary; red is the Cambia call and destructive.

|Token|Dark|Light|Use|
|-|-|-|-|
|`--accent-gold`|`#c9a227`|`#b89026`|Primary button fill, active toggle, wordmark diamond.|
|`--accent-gold-hover`|`#dcb84a`|`#cba32e`|Hover fill. Lighter than the resting fill in both themes.|
|`--accent-gold-active`|`#ab8526`|`#a98325`|Pressed fill. Darker than the resting fill in both themes.|
|`--accent-gold-soft`|`rgba(201,162,39,.16)`|`rgba(171,133,38,.14)`|Gold-tinted fill behind a badge or selection.|
|`--accent-gold-text`|`#c9a227`|`#6f5518`|Gold as text: on `--accent-gold-soft`, on a neutral surface (status strip, gold badge, own name in chat). `--accent-gold` is a fill; as light-theme text it sits under 3:1.|
|`--accent-green`|`#1a6d49`|`#14563a`|Affirmative fill, ready state.|
|`--accent-danger`|`#c4362f`|`#a92c26`|Cambia call, destructive action.|
|`--accent-danger-hover`|`#a92c26`|`#8e2420`|Hover fill. Darker than the resting fill in both themes: `--text-on-danger` is near-white and theme-stable, so contrast comes from a deeper red.|

The light gold trio runs one step brighter than the `--gold-*` ramp and moves the same direction as dark: lighter on hover, darker on press. The earlier light ramp darkened on both, and `--text-on-gold` fell to 3.35:1 on the hover fill and 2.40:1 pressed.

The lift stops at those three fills. Light gold that carries no text stays on the ramp's `--gold-600` `#ab8526`: `--border-accent`, `--focus-ring-color`, `--accent-gold-soft` and the `--interactive-selected` tint. A hairline and a ring are measured against the surface behind them (3:1, WCAG 1.4.11) rather than against a label on top of them, so the two requirements pull opposite ways: `--gold-600` measures 3.27:1 on `--surface-1` and 3.43:1 on `--surface-2`, the lifted `#b89026` only 2.83 and 2.98. Follow the trio with them and the ring loses its edge on paper.

### Contrast

Every text tier paired with a ground by name above holds AA (4.5:1) in both themes, measured with the WCAG 2.1 relative-luminance formula and, for translucent tiers, on the composited color. The exception is the two disabled pairs, which hold 3:1: WCAG exempts inactive controls, but a disabled label is still read ("Signing in", "Creating"), so the house floor keeps it legible without letting it compete with live text.

`npm run check-tokens` measures the pairs and fails under the floor, and CI runs it after the web build, so a token change that breaks one is visible before merge. The pairs it holds: `--text-on-gold` on each of `--accent-gold`, `--accent-gold-hover`, `--accent-gold-active`; `--text-on-danger` on `--accent-danger` and `--accent-danger-hover`; `--text-on-green` on `--accent-green`; `--text-primary`, `--text-secondary` and `--accent-gold-text` on `--surface-1`; `--text-tertiary` on `--surface-1`, `--surface-2` and `--surface-inset`; `--text-on-green` and `--text-on-felt-muted` on `--surface-felt`; `--text-disabled` on `--surface-disabled` and `--surface-2`; `--accent-gold-text` on `--accent-gold-soft` over `--surface-1` and over `--surface-2`, and `--text-primary` on `--accent-gold-soft` over `--surface-1` (the `::selection` fill). `--accent-green-hover` is out: nothing draws text on it.

A tinted fill is a ground like any other: the pair names the opaque surface the tint is painted on (`over`), the gate flattens the tint onto it, then flattens the text onto the result. A translucent ground with no `over` named stays a hard failure, so a pair cannot claim a measurement it has no ground for.

Three grounds bound the tertiary tier because contrast moves in opposite directions per theme: it falls as a dark ground lightens (`--surface-2` is the lightest one an eyebrow sits on) and as a light ground darkens (`--surface-inset`, the input fill under a placeholder). `--surface-0` and `--surface-1` sit inside that span.

### Status

Each status carries three tokens: the foreground, a tinted `-bg`, and a `-border`. Status color reports state; it is never a call to action. Warning is amber, held apart from gold so a warning never reads as a button.

|Token family|Dark foreground|Light foreground|Use|
|-|-|-|-|
|`--status-success-*`|`#35a271`|`#1a6d49`|Running, succeeded, connected, positive delta.|
|`--status-danger-*`|`#dd5a52`|`#a92c26`|Failed, crashed, error text, negative delta.|
|`--status-warning-*`|`#e09a30`|`#a45f0c`|Starting, stopping, degraded.|
|`--status-info-*`|`#6aa5cf`|`#2a5f8f`|Created, queued, neutral notice.|

### Cards and tiers

`--card-face`, `--card-face-edge`, `--card-back`, `--card-back-line`, `--suit-red`, `--suit-black` cover the playing card. The back is flat green with a 1px gold inner frame, not a woven lattice.

`--tier-bronze` through `--tier-grandmaster` are set per theme: the dark values are lifted for a dark ground, the light values darkened for contrast on paper.

## Typography

One family for the whole app: **Archivo Variable**, self-hosted through `@fontsource-variable/archivo` and imported in `src/main.tsx`. Chosen because it is a neutral grotesque with slightly narrow proportions that stay readable at 11 to 13px in dense leaderboards and score readouts, ships a 100 to 900 weight axis so hierarchy needs no second family, and carries a real `tnum` feature for tabular figures. It is not on the AI-default shortlist.

`--font-sans` resolves to `'Archivo Variable', 'Archivo', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif`.

`--ds-font-mono` is the platform monospace stack. It is for code-like content only: training logs, run ids, seeds, hashes. It is not for numbers.

### Banned faces

Verbatim, binding. Do not introduce any of these, in any weight, by any delivery method:

Inter, Manrope, Space Grotesk, Plus Jakarta Sans, Outfit, Sora, Poppins, Montserrat, Lexend, Figtree, Geist, Urbanist, Epilogue, Nunito, Work Sans, and any serif display face (Playfair, Lora, Fraunces, Crimson).

Serifs are out of the app entirely. `npm run check-tokens` fails the build check if any of the above, or a serif family, or a `fonts.googleapis.com` / `fonts.gstatic.com` request, appears in the built stylesheet.

### Scale

|Token|Size|Use|
|-|-|-|
|`--text-2xs`|11px|Eyebrow labels, status pills, unit suffixes.|
|`--ds-text-xs`|12px|Badge text, dense metadata rows.|
|`--ds-text-sm`|13px|Secondary lines, small controls, table cells.|
|`--text-md`|14px|Body default. Controls, labels, most copy.|
|`--ds-text-lg`|16px|Card and dialog titles, large control text.|
|`--ds-text-xl`|20px|Section headings.|
|`--ds-text-2xl`|25px|Page headings.|
|`--ds-text-3xl`|32px|Hero headline.|
|`--ds-text-4xl`|42px|Display number (final score, big rating).|
|`--ds-text-5xl`|56px|Reserved. One per screen at most.|

Weights: `--weight-regular` 400 (body), `--weight-medium` 500 (labels, nav, control text), `--weight-bold` 600 (headings, buttons, active nav), `--weight-black` 700 (wordmark, display numbers, card indices). Nothing above 700.

Line height: `--ds-leading-tight` 1.15 for headings, `--ds-leading-snug` 1.35 for compact multi-line labels, `--ds-leading-normal` 1.55 for body.

Tracking: `--ds-tracking-tight` -0.01em on headings 16px and up, `--tracking-caps` 0.08em on uppercase eyebrows, `--ds-tracking-wide` 0.14em on uppercase display text such as JOKER.

### Eyebrows

One style, exported as `EYEBROW` from `components/ds/eyebrow.ts`: 11px, `--weight-bold`, `--tracking-caps`, uppercase, `--text-tertiary`, and `word-spacing: 0.12em`. Spread it (`{ ...EYEBROW }`) and override only the color or weight a surface genuinely needs; do not restate the five declarations. The word spacing is the reason it is shared: at 11px with 0.08em tracking the inter-word gap disappears into the letter gaps and a two-word eyebrow reads as one run (RULESHEET).

### Numerals

Anything numeric that updates in place gets `font-variant-numeric: tabular-nums`: scores, timers, counts, ratings, deltas, queue sizes, epoch and iteration numbers. Proportional figures reflow as digits change and make a live readout jitter.

`index.css` applies tabular figures globally to `table`, `time`, `input[type=number]`, and the `.tabular-nums` class. Primitives that render numbers (`ScorePill`, `TimerBar`, `StatRow`, `Badge`, `PlayerSeat`, `QueueCard`, `PlayingCard`, the TopBar rating line) set it inline. A new numeric readout must do one of the two.

Do not reach for the mono stack to get alignment. The UI face has tabular figures; mono is a different voice and reads as code.

## Spacing, radius, elevation, motion

Spacing is a 4px base: `--space-1` 4, `--space-2` 8, `--space-3` 12, `--space-4` 16, `--space-5` 20, `--space-6` 24, `--space-8` 32, `--space-10` 40, `--space-12` 48, `--space-16` 64. Panel padding is `--space-4 --space-5`. Gaps between related controls are `--space-2`, between groups `--space-4`, between page sections `--space-6` or more.

Radius: `--ds-radius-sm` 4px (input, checkbox, small chip), `--ds-radius-md` 6px (button, icon button, nav tab), `--ds-radius-lg` 10px (panel, card, dialog), `--ds-radius-xl` 14px (rare, large container), `--radius-pill` 999px (badge, status pill, avatar chip), `--radius-playing-card` 8px.

Borders are 1px. `--line-thin` and `--line-thick` both resolve to 1px so legacy call sites flatten with the token layer rather than needing an edit.

Elevation: `--shadow-piece` and `--shadow-card` are `none` by design. `--shadow-raised` is a 1px hairline shadow for a control that lifts off its ground. `--shadow-overlay` is the only heavy one, for dialogs and menus. `--shadow-playing-card` is a 1px card lift.

Motion: `--dur-fast` 120ms for hover and press, `--dur-med` 200ms for state and theme transitions, `--dur-slow` 320ms for entrances. `--ds-ease-out` for anything entering or settling, `--ease-snap` for a card landing. Nothing bounces past its target.

## Do and don't

Do

- `background: var(--surface-1); border: 1px solid var(--border-default)` for a card.
- `color: var(--status-danger)` for an error message; `background: var(--accent-danger); color: var(--text-on-danger)` for the destructive button.
- `<span style={{ fontVariantNumeric: 'tabular-nums' }}>{score}</span>` for a live score.
- One accent per screen region. A panel with a gold CTA does not also gold-outline its own border.
- Reach for `--text-tertiary` and 11px uppercase for an eyebrow, then let the value below it carry the weight.

Don't

- `background: #101a16` or `rgba(243,236,218,0.07)` inline. Both have tokens.
- `boxShadow: '0 2px 0 var(--outline-ink)'` or any hard offset shadow. That is the previous language.
- `fontFamily: 'var(--ds-font-mono)'` on a rating, score, or countdown. Use the sans with tabular figures.
- `fontFamily: 'var(--font-display)'` for a heading. There is no display face; use `--ds-text-xl` at `--weight-bold`.
- A theme branch in a component (`theme === 'dark' ? ... : ...`). Add or fix a token instead.
- `dark:bg-gray-900` style Tailwind pairs on new markup. They exist on unswept pages and are being removed.

## Removal checklist

Section 4 of `design-tokens.css` keeps the previous language's token names alive, mapped onto the new palette, so surfaces not yet swept render correctly instead of falling back to unset custom properties. Each name below is deleted once no page references it. Do not use any of them in new code.

Ramps: `--ember-300/400/500/600`, `--honey-300/400/500/600`, `--berry-400/500/600`, `--moss-400/500/600/700`, `--dusk-400/500/600`, `--bark-500/600/700/800/850/900/950`, `--parchment-50/100/200/300/400`.

Semantics: `--surface-page`, `--surface-card`, `--surface-raised`, `--surface-table`, `--surface-table-deep`, `--outline-ink`, `--text-on-ember`, `--text-on-honey`.

Fonts: `--font-display`, `--font-ui`. Both resolve to `--font-sans`.

Component API: `Button` `variant="gold"` is an alias of `variant="primary"`; `Badge` `tone="ember"` is an alias of `tone="gold"`; `StatRow` `deltaTone` still takes the legacy `'moss'` and `'berry'` names. Call sites migrate during the page sweeps.
