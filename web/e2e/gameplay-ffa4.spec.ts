// web/e2e/gameplay-ffa4.spec.ts
// AC3 of the cambia-934 pass: a four-seat free-for-all reaches the results card.
//
// Pass 1 never got a four-seat game to the end: both attempts died on the ability panic that
// became cambia-946 (note cambia-960). This is the run that says whether that is closed on the
// shipped surface, so it asserts the end state and the console, not just that four seats sat down.
import { expect, test } from '@playwright/test';
import { closeSeat, dumpSignals, openSeat, realConsoleErrors, type Seat } from './support/seat';
import { createLobby, playToResults, readyUp, waitForTable, RESULTS_TITLE } from './support/game';

const NAMES = ['e2e-ffa-a', 'e2e-ffa-b', 'e2e-ffa-c', 'e2e-ffa-d'];

test.describe('ffa-4 game to results', () => {
  const seats: Seat[] = [];

  // eslint-disable-next-line no-empty-pattern -- Playwright requires the fixtures argument to be a destructuring pattern, and this hook takes none.
  test.afterEach(async ({}, testInfo) => {
    for (const seat of seats) dumpSignals(seat, 'ffa4', testInfo.outputPath());
    for (const seat of seats) await closeSeat(seat);
    seats.length = 0;
  });

  test('four seats play a full game and land on the results card', async ({ browser }, testInfo) => {
    const host = await openSeat(browser, NAMES[0], '/dashboard');
    seats.push(host);
    const lobbyId = await createLobby(host, { mode: 'group_of_4', type: 'public' });

    for (const name of NAMES.slice(1)) {
      seats.push(await openSeat(browser, name, `/lobby/${lobbyId}`));
    }
    await expect(host.page.getByText('4 seated')).toBeVisible({ timeout: 30_000 });

    for (const seat of seats.slice(1)) await readyUp(seat);
    await readyUp(host);

    for (const seat of seats) await waitForTable(seat);

    // Three opponents across the felt, each named and each holding a hand.
    for (const name of NAMES.slice(1)) {
      await expect(host.page.locator(`[aria-label^="${name} card 1"]`)).toBeVisible();
    }
    await host.page.screenshot({ path: testInfo.outputPath('ffa4-table.png') });

    await playToResults(seats, { cambiaCaller: host.name, cambiaAfter: 3, deadlineMs: 180_000 });

    await host.page.screenshot({ path: testInfo.outputPath('ffa4-results.png') });
    for (const seat of seats) {
      await expect(seat.page.getByRole('heading', { name: RESULTS_TITLE })).toBeVisible();
      // Four rows of standings, one per seat, on every seat's own card.
      for (const name of NAMES) {
        await expect(seat.page.getByRole('dialog').getByText(new RegExp(name)).first()).toBeVisible();
      }
    }

    for (const seat of seats) {
      expect(realConsoleErrors(seat), `console errors on ${seat.name}`).toEqual([]);
      expect(seat.signals.pageErrors, `uncaught errors and rejections on ${seat.name}`).toEqual([]);
      const serverErrors = seat.signals.httpErrors.filter((l) => / 5\d\d /.test(l));
      expect(serverErrors, `5xx responses on ${seat.name}`).toEqual([]);
    }
  });
});
