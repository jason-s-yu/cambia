// web/e2e/support/game.ts
// Drives a lobby and a game through the client the way a player does: dashboard dialog, lobby
// controls, and the table's own action column. Nothing here reaches into a store or fires a
// WebSocket frame by hand, so a pass is evidence about the shipped surface and not about a
// harness that agrees with it.
import { expect, type Browser, type Locator, type Page } from '@playwright/test';
import { openSeat, type Seat } from './seat';

/** Test ids the table puts on its action column (src/components/game/DsGameTable.tsx). */
export const ACTION = {
  drawStock: '[data-testid="action-draw-stock"]',
  takeDiscard: '[data-testid="action-take-discard"]',
  discardDrawn: '[data-testid="action-discard-drawn"]',
  snap: '[data-testid="action-snap"]',
  cancelSnap: '[data-testid="action-cancel-snap"]',
  kingSwap: '[data-testid="action-king-swap"]',
  kingKeep: '[data-testid="action-king-keep"]',
  skipAbility: '[data-testid="action-skip-ability"]',
  cambia: '[data-testid="action-cambia"]',
  leave: '[data-testid="action-leave"]',
  leaveForfeited: '[data-testid="action-leave-forfeited"]',
} as const;

export const PILE = {
  stock: '[data-testid="pile-stock"]',
  discard: '[data-testid="pile-discard"]',
  drawn: '[data-testid="card-drawn"]',
} as const;

/** The results card, whichever of its three titles it is wearing. */
export const RESULTS_TITLE = /^(Game over|Final standings|Game ended by an error)$/;

/** An own-hand card as a control. Only matches while the slot takes a click or holds a pick. */
export function ownCardButton(page: Page, slot: number): Locator {
  return page.getByRole('button', { name: new RegExp(`^Your card ${slot}[:,]`) });
}

/** An own-hand card in any state, control or image. */
export function ownCardAny(page: Page, slot: number): Locator {
  return page.locator(`[aria-label^="Your card ${slot}:"], [aria-label^="Your card ${slot},"]`);
}

/** Creates a lobby from the dashboard dialog and returns its id, with the host already on it. */
export async function createLobby(
  seat: Seat,
  opts: { mode?: 'head_to_head' | 'group_of_4'; type?: 'public' | 'private' } = {}
): Promise<string> {
  const page = seat.page;
  await page.goto('/dashboard');
  await page.getByRole('button', { name: 'Create lobby' }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByRole('heading', { name: 'New lobby' })).toBeVisible();
  await dialog.getByRole('combobox', { name: /Lobby type/ }).selectOption(opts.type ?? 'public');
  await dialog.getByRole('combobox', { name: /Game mode/ }).selectOption(opts.mode ?? 'head_to_head');
  await dialog.getByRole('button', { name: 'Create', exact: true }).click();
  await page.waitForURL(/\/lobby\/[0-9a-f-]{36}/, { timeout: 30_000 });
  const id = page.url().split('/lobby/')[1].split('?')[0];
  await expect(page.getByRole('heading', { name: 'Lobby', exact: true })).toBeVisible();
  return id;
}

/** Marks a seat ready. With auto-start on (the default) the last ready starts the game. */
export async function readyUp(seat: Seat): Promise<void> {
  await seat.page.getByRole('button', { name: 'Ready up' }).click();
}

/** Waits until the felt is up for this seat: the two piles are on screen. */
export async function waitForTable(seat: Seat, timeout = 60_000): Promise<void> {
  await expect(seat.page.locator(PILE.stock)).toBeVisible({ timeout });
  await expect(seat.page.locator(PILE.discard)).toBeVisible({ timeout });
}

/** True while the results card is on screen for this seat. */
export async function resultsShown(page: Page): Promise<boolean> {
  return page.getByRole('heading', { name: RESULTS_TITLE }).isVisible().catch(() => false);
}

/** Whether a locator is on screen right now, without waiting for it to appear. */
async function showing(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).first().isVisible().catch(() => false);
}

export interface PlayOptions {
  /**
   * How many of this seat's own turns to take before calling Cambia. Only the seat named in
   * `cambiaCaller` calls; the rest keep playing, so exactly one call ends the round.
   */
  cambiaAfter?: number;
  cambiaCaller?: string;
  /** Wall-clock budget for the whole game, in ms. */
  deadlineMs?: number;
}

/**
 * Plays one seat until the results card appears.
 *
 * The policy is a replace, not a discard: with `allowReplaceAbilities` off (the shipped default,
 * visible on the lobby's rule sheet) a replace fires no ability, so the loop needs no branch per
 * rank and a run is comparable to the run before it. Ability prompts are still handled, since a
 * turn the clock takes for a seat is auto-discarded by the server and can arm one.
 *
 * Nothing here asserts. A stalled seat is not a test failure by itself: the server's turn clock
 * plays for it, and the assertion that matters is the one the caller makes about the game ending.
 */
export async function playSeat(seat: Seat, opts: PlayOptions = {}): Promise<{ ownTurns: number }> {
  const page = seat.page;
  const cambiaAfter = opts.cambiaAfter ?? 4;
  const isCaller = opts.cambiaCaller === seat.name;
  const deadline = Date.now() + (opts.deadlineMs ?? 180_000);
  let ownTurns = 0;

  while (Date.now() < deadline) {
    if (page.isClosed()) break;
    if (await resultsShown(page)) break;

    // A fill owed for a snapped slot blocks every other action until it is paid.
    if (await page.getByText(/Choose one of your cards to fill the slot/).isVisible().catch(() => false)) {
      const card = ownCardButton(page, 1);
      if (await card.isVisible().catch(() => false)) {
        await card.click({ timeout: 5_000 }).catch(() => undefined);
        continue;
      }
    }

    if (await showing(page, ACTION.skipAbility)) {
      await page.locator(ACTION.skipAbility).click({ timeout: 5_000 }).catch(() => undefined);
      continue;
    }
    if (await showing(page, ACTION.kingKeep)) {
      await page.locator(ACTION.kingKeep).click({ timeout: 5_000 }).catch(() => undefined);
      continue;
    }
    // Holding a drawn card: commit it into a slot. Slot 1 every time, so the hand this seat ends
    // with is a function of the deal alone.
    if (await showing(page, ACTION.discardDrawn)) {
      const card = ownCardButton(page, 1);
      if (await card.isVisible().catch(() => false)) {
        await card.click({ timeout: 5_000 }).catch(() => undefined);
        continue;
      }
    }
    if (isCaller && ownTurns >= cambiaAfter && (await showing(page, ACTION.cambia))) {
      await page.locator(ACTION.cambia).click({ timeout: 5_000 }).catch(() => undefined);
      continue;
    }
    if (await showing(page, ACTION.drawStock)) {
      await page.locator(ACTION.drawStock).click({ timeout: 5_000 }).catch(() => undefined);
      ownTurns += 1;
      continue;
    }
    await page.waitForTimeout(200);
  }
  return { ownTurns };
}

/**
 * Opens two seats, puts them in one public head-to-head lobby and starts the game.
 *
 * Auto-start is on by default, so the second ready is what starts it; no host click is needed.
 * `beforeStart` runs on the host while both seats are seated and neither is ready, which is the
 * only window in which the rule sheet is editable.
 */
export async function startH2H(
  browser: Browser,
  prefix: string,
  opts: { instrumentSockets?: boolean; beforeStart?: (host: Seat) => Promise<void> } = {}
): Promise<{ host: Seat; guest: Seat; lobbyId: string }> {
  const { seats, lobbyId } = await startLobby(browser, prefix, { ...opts, count: 2, mode: 'head_to_head' });
  return { host: seats[0], guest: seats[1], lobbyId };
}

/** The same for any seat count the mode admits. Seats are named `<prefix>-a`, `-b`, and so on. */
export async function startLobby(
  browser: Browser,
  prefix: string,
  opts: {
    count: number;
    mode?: 'head_to_head' | 'group_of_4';
    instrumentSockets?: boolean;
    beforeStart?: (host: Seat) => Promise<void>;
  }
): Promise<{ seats: Seat[]; lobbyId: string }> {
  const letters = 'abcdefgh'.slice(0, opts.count).split('');
  const host = await openSeat(browser, `${prefix}-${letters[0]}`, '/dashboard', opts);
  const lobbyId = await createLobby(host, { mode: opts.mode ?? 'head_to_head', type: 'public' });
  const seats = [host];
  for (const letter of letters.slice(1)) {
    seats.push(await openSeat(browser, `${prefix}-${letter}`, `/lobby/${lobbyId}`, opts));
  }
  await expect(host.page.getByText(`${opts.count} seated`)).toBeVisible({ timeout: 30_000 });
  if (opts.beforeStart) await opts.beforeStart(host);
  for (const seat of seats.slice(1)) await readyUp(seat);
  await readyUp(host);
  for (const seat of seats) await waitForTable(seat);
  return { seats, lobbyId };
}

/** Waits out the pre-game peek, so the table is in its ordinary playing state. */
export async function waitForPlay(seat: Seat, timeout = 30_000): Promise<void> {
  await expect(seat.page.getByText('Pre-game peek.')).toBeHidden({ timeout });
}

/** Plays every seat at once and waits for the results card on each. */
export async function playToResults(seats: Seat[], opts: PlayOptions = {}): Promise<void> {
  await Promise.all(seats.map((s) => playSeat(s, opts)));
  for (const seat of seats) {
    await expect(seat.page.getByRole('heading', { name: RESULTS_TITLE })).toBeVisible({ timeout: 30_000 });
  }
}
