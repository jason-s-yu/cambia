// web/e2e/gameplay-h2h.spec.ts
// AC1 of the cambia-934 pass: a full two-player game played through the client to the results
// card, with no console error and no unhandled rejection on either seat.
//
// The game is played, not simulated: two browser contexts, the real dashboard dialog, the real
// lobby, and the table's own action column. What the assertions here are worth depends on that,
// which is why the driver in support/game.ts touches no store.
import { expect, test } from '@playwright/test';
import { closeSeat, dumpSignals, openSeat, realConsoleErrors, type Seat } from './support/seat';
import { createLobby, ownCardAny, playToResults, readyUp, waitForTable } from './support/game';

test.describe('h2h game to results', () => {
  const seats: Seat[] = [];

  // eslint-disable-next-line no-empty-pattern -- Playwright requires the fixtures argument to be a destructuring pattern, and this hook takes none.
  test.afterEach(async ({}, testInfo) => {
    for (const seat of seats) dumpSignals(seat, 'h2h', testInfo.outputPath());
    for (const seat of seats) await closeSeat(seat);
    seats.length = 0;
  });

  test('two seats play a full game and land on the results card with a clean console', async ({ browser }, testInfo) => {
    const host = await openSeat(browser, 'e2e-h2h-a', '/dashboard');
    seats.push(host);
    const lobbyId = await createLobby(host, { mode: 'head_to_head', type: 'public' });

    const guest = await openSeat(browser, 'e2e-h2h-b', `/lobby/${lobbyId}`);
    seats.push(guest);
    await expect(guest.page.getByText('2 seated')).toBeVisible();

    await readyUp(guest);
    await readyUp(host);

    await waitForTable(host);
    await waitForTable(guest);

    // The opening peek is the one moment an own face is on screen before the round ends: every
    // sync snapshot hides own cards (cambia-1094), so the faces live only in the client's peek
    // and only for the pre-game window. Two, because `initialViewCount` is 2 on the default sheet.
    const peeked = host.page.locator('[aria-label^="Your card "]:not([aria-label*="face down"])');
    await expect(peeked).toHaveCount(2, { timeout: 15_000 });
    await host.page.screenshot({ path: testInfo.outputPath('h2h-pregame-peek.png') });

    // Once the window closes every own slot is face down again, which is what makes the peek a
    // memory game rather than an open hand.
    await expect(peeked).toHaveCount(0, { timeout: 30_000 });
    await expect(ownCardAny(host.page, 1)).toBeVisible();

    await playToResults([host, guest], { cambiaCaller: host.name, cambiaAfter: 4, deadlineMs: 150_000 });

    await host.page.screenshot({ path: testInfo.outputPath('h2h-results-host.png') });
    await guest.page.screenshot({ path: testInfo.outputPath('h2h-results-guest.png') });

    // The card reports a result, not just a title: both seats are named on it, and both seats'
    // hands are revealed under their standings row, which is the round-end reveal (RULES.md 3C)
    // and the half a title alone would not prove.
    for (const seat of [host, guest]) {
      const card = seat.page.getByRole('dialog');
      await expect(card.getByText(/e2e-h2h-a/).first()).toBeVisible();
      await expect(card.getByText(/e2e-h2h-b/).first()).toBeVisible();
      await expect(card.locator('[data-testid^="final-hand-"]')).toHaveCount(2);
    }

    for (const seat of [host, guest]) {
      expect(realConsoleErrors(seat), `console errors on ${seat.name}`).toEqual([]);
      expect(seat.signals.pageErrors, `uncaught errors and rejections on ${seat.name}`).toEqual([]);
      const serverErrors = seat.signals.httpErrors.filter((l) => / 5\d\d /.test(l));
      expect(serverErrors, `5xx responses on ${seat.name}`).toEqual([]);
    }
  });
});
