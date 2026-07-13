# Cambia Design System

Design system for **Cambia** — a real-time multiplayer card game platform (the game is also known as Cabo/Cambio) whose headline feature is AI opponents trained with Deep CFR (counterfactual regret minimization). Players race to the lowest score in an imperfect-information card game built on memory, snap reflexes, and bluffing.

**Design language in one line:** *Root-boardgame-inspired woodland storybook — cartoonish on the surface, competitively serious underneath.* Dark theme first ("night forest"), with a light "daylight parchment" theme toggled via `data-theme="light"` on `<html>` or any subtree.

## Product surfaces

1. **Game platform** (primary) — player homepage/dashboard (queues, lobby list, ratings, friends), lobby creation & house-rule settings, and the live game table (draw/discard/snap/Cambia, special abilities, multi-round circuit scoreboards).
2. **Leaderboard** — ranked ladders per pool (H2H Ranked · Glicko-2, FFA-4 Ranked · OpenSkill), percentile tiers Bronze → Grandmaster.
3. **AI training dashboard** (admin, local hardware) — CFR training runs, checkpoints, eval win-rate charts, GPU/CPU resource monitors, log streaming.

## Sources

- GitHub: [jason-s-yu/cambia](https://github.com/jason-s-yu/cambia) — Go game engine (`engine/`), multiplayer service (`service/`), Deep CFR pipeline (`cfr/`), web client (`web/`). Explore it to ground designs in real enums and flows: `service/internal/game/rules.go` (house rules), `web/src/types/*.ts` (game/lobby/training state), `MATCHMAKING.md` (queues, ratings, tiers), `RULES.md`.
- Full rules + Tournament Circuit spec supplied by the owner (summarized below).
- **The repo's `web/` client is explicitly NOT a style reference** — it is a primitive placeholder this system supersedes. Use it only for flows, enums, and data shapes.
- No brand assets (logo, fonts, illustrations) exist in the sources. **There is no logo: render the wordmark "CAMBIA" in type** (Young Serif, letterspaced). Never invent a mark.

## Game vocabulary (use it verbatim in copy)

- Actions: **Draw**, **Swap**, **Discard**, **Snap**, **Call "Cambia"**.
- Abilities: 7/8 **Peek Own** · 9/10 **Peek Other** · J/Q **Blind Swap** · K **Look & Swap**.
- Values: Red King −1 (best), Joker 0, Ace 1, 2–10 face, J 11, Q 12, Black King 13 (worst).
- Queues: `h2h_quickplay` (hidden MMR), `h2h_blitz`/`h2h_rapid`/`h2h_classical` (Glicko-2), `ffa4_standard`/`ffa4_classical` (OpenSkill). Primary queues: **H2H Rapid**, **FFA-4 Standard**.
- Tiers: Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster.
- House rules (exact keys): `allowDrawFromDiscardPile`, `allowReplaceAbilities`, `snapRace`, `forfeitOnDisconnect`, `penaltyDrawCount`, `autoKickTurnCount`, `turnTimerSec`; circuit: `targetScore`, `winBonus`, `falseCambiaPenalty`, `freezeUserOnDisconnect`.
- Ranked bonus: the **Aggression Subsidy** (−5/−2 FFA-4; −3/0 H2H).

## CONTENT FUNDAMENTALS

- **Tone:** a friendly tavern host running a deadly serious tournament. Warm, wry, never corporate. Whimsy lives in nouns and flavor lines; numbers and rules are stated with precision.
- **Voice:** second person ("Your turn", "You snapped the 7♥"). The system speaks as narrator in game-log lines ("Maple peeks at an opponent's card").
- **Casing:** Sentence case everywhere, including buttons ("Call Cambia", "Create lobby"). UPPERCASE reserved for tiny letterspaced eyebrows/labels ("STOCKPILE", "ROUND 3 OF 8").
- **Card names:** always rank + suit glyph ("7♥", "Black King ♠"). Scores/ratings always in mono ("1520 ± 140").
- **Flavor, not filler:** one short flavor line max per surface ("The forest remembers your discards."). Rules copy stays literal.
- **No emoji.** Whimsy comes from palette, type, and suit glyphs — never emoji.
- **Example strings:** "Cambia called — everyone gets one last turn." · "Snap! −1 card." · "Wrong snap: draw 2." · "Placement: 7/15 games remaining."

## VISUAL FOUNDATIONS

- **Palette:** warm umber "bark" surfaces (dark) / cream "parchment" (light); accents: **ember** rust-orange (primary actions), **moss** green (success/ready + the table felt), **dusk** slate blue (info), **berry** red (Cambia call, danger, red suits), **honey** gold (ranked, wins, focus). All defined in `tokens/colors.css` with semantic aliases that switch per theme.
- **Type:** Young Serif for display/headings and big numerals; Alegreya Sans for UI; Space Mono for all game/stat numbers (scores, timers, ratings, seeds, metrics). Sizes/weights in `tokens/typography.css`.
- **Cartoon outline:** key interactive pieces wear a thick 2px ink outline (`--outline-ink`, `--line-thick`) like tabletop pieces.
- **Shadows:** hard offset shadows (`--shadow-piece`, `0 2px 0 ink`) — pieces on a table, not floating glass. Soft blur only on modals/overlays. Buttons press DOWN: hover lifts 1px, active removes the offset and translates down.
- **Backgrounds:** flat token surfaces; the game table is deep moss green with a subtle inner lip (`--inset-table`). No gradients, no photos, no textures (a paper-grain texture is welcome later if a real asset is supplied).
- **Motion:** quick ease-out (120–220ms) for UI; the springy `--ease-snap` bounce is reserved for card deals and snaps. No infinite loops.
- **Hover:** surface lightens one step or border strengthens; **press:** translate down 2px + shadow removed. Focus: honey-gold double ring (`--focus-ring`).
- **Corners:** generous, toy-like — 10px buttons, 14px panels, 20px modals, 9px playing cards. Pills for badges/tags.
- **Cards (UI panels):** `--surface-card`, 1.5–2px border (`--border-default` or ink for emphasis), radius 14px, hard shadow. Playing cards: parchment face, ink/berry pips, poker ratio 1:1.4, ember-red patterned back.
- **Transparency/blur:** only the modal scrim (`--surface-overlay`); no glassmorphism.
- **Theme toggle:** dark is default on `:root`; set `data-theme="light"` to swap semantic aliases. Base hue ramps never change.

## ICONOGRAPHY

- **Suits are the icon language of the game:** unicode glyphs ♠ ♥ ♦ ♣ (+ ★ for Joker), colored `--suit-red`/`--suit-black` on parchment, `--berry-400`/`--text-primary` on dark surfaces.
- **UI icons: [Lucide](https://lucide.dev) via CDN** (`https://unpkg.com/lucide@latest`) — outline style, 1.5–2px stroke, matches the ink-outline motif. **This is a substitution:** the source repo ships no icon set (only Vite's `react.svg`). Flagged for replacement if a bespoke set is drawn.
- Common mappings: draw=`layers`, snap=`zap`, cambia=`flag`, timer=`timer`, ranked=`trophy`, friends=`users`, settings=`sliders-horizontal`, training=`brain`, GPU=`cpu`.
- No emoji, no hand-rolled SVG illustrations. If illustration is needed (empty states, hero), leave a labeled placeholder and request real art.

## Index

- `styles.css` — global entry; imports everything under `tokens/`.
- `tokens/` — `colors.css`, `typography.css`, `spacing.css`, `effects.css`, `fonts.css`, `base.css`.
- `guidelines/` — foundation specimen cards (Design System tab).
- `components/core/` — Button, IconButton, Input, Select, Checkbox, Switch, Badge, Modal, Spinner.
- `components/game/` — PlayingCard, PlayerSeat, ScorePill, TimerBar.
- `components/data/` — QueueCard, TierBadge, StatusBadge, StatRow.
- `ui_kits/platform/` — player home, lobby, game table, leaderboard (interactive `index.html`).
- `ui_kits/training/` — AI training dashboard (runs, run detail, resources).
- `SKILL.md` — agent-skill entry point.

## Intentional additions

- **TierBadge / ScorePill / TimerBar / PlayerSeat** — no component source defines the new visual language (old web client superseded), so game-critical primitives were authored fresh from the rules + backend types.

## Caveats

- **Fonts are Google Fonts substitutions** (Young Serif / Alegreya Sans / Space Mono) — no brand fonts exist. Supply licensed files to replace `tokens/fonts.css` `@import` with real `@font-face`.
- **No logo/illustrations exist**; wordmark is set in type, illustration slots are placeholders.
- Lucide icons are a CDN substitution, flagged above.
