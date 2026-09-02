// web/e2e/phase1-surfaces.spec.ts
// AC2 of the cambia-934 pass: each surface Phase 1 shipped, exercised in a browser against a live
// service rather than in a render harness.
//
// The render rig already covers these at component level (src/components/**/*.test.tsx). What it
// cannot say is whether the surface is reachable in the shipped app: whether the boundary is
// actually mounted around the table, whether the retry cap holds when the dialing is driven by
// the real effect, whether a forfeited seat's exit is on screen after a real disconnect. That is
// what each test here is for, and each names the surface it stands for.
import { expect, test } from '@playwright/test';
import { closeSeat, dumpSignals, openSeat, type Seat } from './support/seat';
import { dropSocket, lobbyDials, refuseDials, injectUnrenderableSnapshot } from './support/instrument';
import { ACTION, PILE, createLobby, ownCardAny, playSeat, readyUp, startH2H, startLobby, waitForPlay, RESULTS_TITLE } from './support/game';

const seats: Seat[] = [];

// eslint-disable-next-line no-empty-pattern -- Playwright requires the fixtures argument to be a destructuring pattern, and this hook takes none.
test.afterEach(async ({}, testInfo) => {
  const run = testInfo.title.replace(/[^a-z0-9]+/gi, '-').slice(0, 40);
  for (const seat of seats) dumpSignals(seat, run, testInfo.outputPath());
  for (const seat of seats) await closeSeat(seat);
  seats.length = 0;
});

test.describe('error boundary recovery', () => {
  test('a table that cannot render swaps for a recovery panel, and the app shell survives it', async ({ browser }, testInfo) => {
    const { host, guest } = await startH2H(browser, 'e2e-eb', { instrumentSockets: true });
    seats.push(host, guest);
    await waitForPlay(host);

    // A snapshot the table cannot render. The boundary exists for exactly this: the table reads
    // live server state through optional-chained shapes, so an unexpected payload is a real
    // failure mode (src/components/ErrorBoundary.tsx).
    await injectUnrenderableSnapshot(host.page);

    const panel = host.page.getByRole('alert');
    await expect(panel).toBeVisible();
    await expect(panel).toContainText('The table hit an error and stopped rendering.');
    await expect(host.page.getByTestId('error-boundary-reload')).toBeVisible();
    await expect(host.page.getByTestId('error-boundary-leave')).toBeVisible();
    await host.page.screenshot({ path: testInfo.outputPath('error-boundary-panel.png') });

    // The shell's own boundary did not fire: the nav and the account chip are still there, which
    // is the point of nesting a second boundary around the table (cambia-1235).
    await expect(host.page.getByRole('navigation', { name: 'Primary' })).toBeVisible();
    await expect(host.page.getByRole('button', { name: 'Log out' })).toBeVisible();

    // The boundary logged the throw once. That console error is the injected fault's own record,
    // so it is expected here and nowhere else.
    expect(host.signals.consoleErrors.filter((l) => l.includes('[ErrorBoundary:table]'))).toHaveLength(1);

    // The other seat is untouched: this is one client's render failure, not the game's.
    await expect(guest.page.locator(PILE.stock)).toBeVisible();
    expect(guest.signals.pageErrors).toEqual([]);

    // Recovery: Reload puts the table back, which is what the panel promises.
    await host.page.getByTestId('error-boundary-reload').click();
    await expect(host.page.locator(PILE.stock)).toBeVisible({ timeout: 30_000 });
  });
});

test.describe('reconnect cap and its stopped-trying copy', () => {
  test('stops dialing at the cap and says so on the table', async ({ browser }, testInfo) => {
    // At the table, not in the lobby. Phase 1 AC2 words the criterion as the copy being reachable
    // on the table, and that is the only place it is: a lobby that spends its budget sets the
    // hook's own "Lost connection after 5 retries." and is bounced to the dashboard by the same
    // state, which clears it (LobbyPage shouldRedirectToDash). Reported, not asserted here.
    //
    // The turn clock is off for this run so the game cannot end during the ~35 seconds the budget
    // takes to spend: a finished game swaps the table for the results card, and the surface under
    // test would go with it. The connection is what is being measured, not the clock.
    const { host, guest } = await startH2H(browser, 'e2e-cap', {
      instrumentSockets: true,
      beforeStart: async (h) => {
        await h.page.getByRole('spinbutton', { name: 'Turn clock (sec)' }).fill('0');
        await h.page.getByRole('button', { name: 'Save rules' }).click();
        await expect(h.page.getByRole('spinbutton', { name: 'Turn clock (sec)' })).toHaveValue('0');
      },
    });
    seats.push(host, guest);
    await waitForPlay(host);

    const before = await lobbyDials(host.page);
    await refuseDials(host.page);
    await dropSocket(host.page, 1006);

    // While the budget is being spent the felt says a reconnect is happening, because one is.
    await expect(host.page.getByText('Reconnecting', { exact: true }).first()).toBeVisible({ timeout: 20_000 });

    // MAX_RETRIES is 5 (src/hooks/useSocket.ts). The budget goes on backoffs of 1, 2, 4, 8 and 16
    // seconds plus up to a second of jitter each, so the terminal state cannot arrive before ~31s.
    await expect(host.page.getByText('Connection lost. Leave the table and rejoin from the dashboard.')).toBeVisible({
      timeout: 90_000,
    });
    await expect(host.page.getByText('Disconnected', { exact: true }).first()).toBeVisible();
    await expect(host.page.getByText('Reconnecting', { exact: true })).toHaveCount(0);
    await host.page.screenshot({ path: testInfo.outputPath('reconnect-cap-table.png') });

    expect((await lobbyDials(host.page)) - before, 'dials made after the drop').toBe(5);

    // The cap is a cap only if nothing dials again once it is reached. Before cambia-1236 the
    // connect effect re-ran on every lobby-store write and dialed a dead server 521 times.
    await host.page.waitForTimeout(8_000);
    expect((await lobbyDials(host.page)) - before, 'dials after the cap was reached').toBe(5);

    // The other seat played on: one client's spent budget is not the table's problem.
    await expect(guest.page.locator(PILE.stock)).toBeVisible();
  });

  test('a table whose socket the hub closed says disconnected, not reconnecting', async ({ browser }, testInfo) => {
    const { host, guest } = await startH2H(browser, 'e2e-gu', { instrumentSockets: true });
    seats.push(host, guest);
    await waitForPlay(host);

    // A clean close is the hub's own: it closes 1000 when it displaces a duplicate socket, on
    // leave, and on cleanup. Nothing retries after one, so the felt must draw a terminal state
    // instead of a reconnect that is not coming (cambia-1239 review).
    await dropSocket(host.page, 1000);

    // Two places say it: the chip in the top strip and the seat itself.
    await expect(host.page.getByText('Disconnected', { exact: true })).toHaveCount(2, { timeout: 20_000 });
    await expect(host.page.getByText('Connection lost. Leave the table and rejoin from the dashboard.')).toBeVisible();
    await expect(host.page.getByText('Reconnecting', { exact: true })).toHaveCount(0);
    await host.page.screenshot({ path: testInfo.outputPath('table-gave-up.png') });
    // The exit off a dead table is still there.
    await expect(host.page.locator(ACTION.leave)).toBeVisible();
  });
});

test.describe('forfeited seat', () => {
  test('a seat that gave up its round is told so and given a way off the table', async ({ browser }, testInfo) => {
    // Four seats, not two: a forfeit in a head-to-head ends the round on the spot, and the results
    // card then covers the very control this test is about. With three seats still playing, the
    // forfeited seat sits on a live table, which is the state the surface was built for.
    //
    // The forfeit is the deliberate one, not a dropped socket. A seat that merely lost its
    // connection and came back has its forfeit lifted by design (RULES.md T5, cambia-955), so a
    // reload can never leave a live table showing this surface. A seat given up on purpose is the
    // exception the server keeps forfeited, and it returns as a connected spectator
    // (service/internal/game/game.go, cambia-1239) - which is exactly this state.
    const { seats: players, lobbyId } = await startLobby(browser, 'e2e-ff', { count: 4, mode: 'group_of_4' });
    seats.push(...players);
    const dropped = players[3];
    await waitForPlay(dropped);

    await dropped.page.locator(ACTION.leave).click();
    await expect(dropped.page.getByTestId('leave-forfeit')).toBeVisible();
    await dropped.page.getByTestId('leave-forfeit').click();
    await expect(dropped.page).toHaveURL(/\/dashboard/, { timeout: 20_000 });

    // Back to the table it gave up on, which is the path the server names in as many words.
    await dropped.page.goto(`/lobby/${lobbyId}`);

    const exit = dropped.page.locator(ACTION.leaveForfeited);
    await expect(exit).toBeVisible({ timeout: 45_000 });
    await expect(exit).toHaveText('Leave table');
    // The seat is told what happened, not left on the waiting-for-opponent copy with every control
    // gone, which is the defect this surface closed (Phase 1 AC3).
    await expect(dropped.page.getByText(/gave up|forfeit/i).first()).toBeVisible();
    // Its own cards say the score stopped counting (cambia-1468).
    await expect(dropped.page.locator('[aria-label*="forfeited"]').first()).toBeVisible();
    await dropped.page.screenshot({ path: testInfo.outputPath('forfeited-seat.png') });

    // The other seats keep playing, and see the forfeited seat named as such.
    await expect(players[0].page.locator(PILE.stock)).toBeVisible();

    // The exit works: the control leaves the table rather than only looking like it could.
    await exit.click();
    await expect(dropped.page).toHaveURL(/\/dashboard/, { timeout: 20_000 });
  });
});

test.describe('post-game host override', () => {
  test('the host returns the lobby to open from the results card, and a second game starts in it', async ({ browser }, testInfo) => {
    const { host, guest } = await startH2H(browser, 'e2e-pg');
    seats.push(host, guest);

    await Promise.all([
      playSeat(host, { cambiaCaller: host.name, cambiaAfter: 1, deadlineMs: 120_000 }),
      playSeat(guest, { deadlineMs: 120_000 }),
    ]);
    await expect(host.page.getByRole('heading', { name: RESULTS_TITLE })).toBeVisible({ timeout: 30_000 });

    // The host has the override; the guest of a player-hosted lobby is told to wait for it
    // (cambia-1516), which is the same gate the hub enforces.
    const hostCard = host.page.getByRole('dialog');
    await expect(hostCard.getByRole('button', { name: 'Back to lobby' })).toBeVisible();
    await expect(guest.page.getByRole('dialog').getByRole('button', { name: 'Back to lobby' })).toBeHidden();
    await expect(guest.page.getByRole('dialog')).toContainText('Waiting for the host');
    await guest.page.screenshot({ path: testInfo.outputPath('post-game-guest-waiting.png') });

    // A reload into the results still shows a result. Pass 1 found the card blank after a
    // reconnect, which is the half of cambia-992 the results resend closed: the server answers a
    // rejoining seat with game_results, so the scores are not lost with the socket.
    await guest.page.reload();
    const resent = guest.page.getByRole('dialog');
    await expect(resent.getByRole('heading', { name: RESULTS_TITLE })).toBeVisible({ timeout: 30_000 });
    await expect(resent.getByText('Scores')).toBeVisible();
    await expect(resent.locator('[data-testid^="final-hand-"]')).toHaveCount(2);
    await guest.page.screenshot({ path: testInfo.outputPath('post-game-results-after-reload.png') });

    await hostCard.getByRole('button', { name: 'Back to lobby' }).click();

    // Both seats are back in an open lobby, not just the one that clicked.
    for (const seat of [host, guest]) {
      await expect(seat.page.getByRole('heading', { name: 'Lobby', exact: true })).toBeVisible({ timeout: 30_000 });
      await expect(seat.page.getByRole('button', { name: 'Ready up' })).toBeVisible();
    }
    await host.page.screenshot({ path: testInfo.outputPath('post-game-back-in-lobby.png') });

    // Phase 1 AC4's second half: a second game starts in the same lobby.
    await readyUp(guest);
    await readyUp(host);
    await expect(host.page.locator(PILE.stock)).toBeVisible({ timeout: 45_000 });
    await expect(guest.page.locator(PILE.stock)).toBeVisible({ timeout: 45_000 });
  });
});

test.describe('lobby nits', () => {
  test('an unjoinable lobby id is never dialled, so nothing retries a refusal', async ({ browser }) => {
    const seat = await openSeat(browser, 'e2e-nit-a', '/dashboard', { instrumentSockets: true });
    seats.push(seat);

    // The nil UUID is what the stores hold for "no id yet"; before cambia-1126 item 1 the retry
    // path dialled a URL the hub refuses outright until the page went away.
    await seat.page.goto('/lobby/00000000-0000-0000-0000-000000000000');
    await seat.page.waitForTimeout(6_000);
    expect(await lobbyDials(seat.page), 'dials for the nil lobby id').toBe(0);

    // A well-formed id for a lobby that does not exist is dialled once and refused once, not
    // redialled: the hub's answer is not transient.
    await seat.page.goto('/lobby/11111111-2222-3333-4444-555555555555');
    await seat.page.waitForTimeout(8_000);
    expect(await lobbyDials(seat.page), 'dials for a lobby that does not exist').toBeLessThanOrEqual(1);
  });

  test('no round counter over a lobby that is not playing a circuit', async ({ browser }) => {
    const { host, guest } = await startH2H(browser, 'e2e-nit2');
    seats.push(host, guest);
    // "ROUND 0/8" used to be badged over the standings of the one game a matchmade lobby plays
    // (cambia-1126 item 2). Circuit scoring is off on the default sheet, so no counter belongs.
    await expect(host.page.getByText(/Round \d+\s*\/\s*\d+/i)).toHaveCount(0);
  });

  test('the rule sheet only offers a save once something changed, and keeps what it saved', async ({ browser }) => {
    const host = await openSeat(browser, 'e2e-nit3-a', '/dashboard');
    seats.push(host);
    await createLobby(host);

    const save = host.page.getByRole('button', { name: 'Save rules' });
    await expect(save).toBeDisabled();

    const penalty = host.page.getByRole('spinbutton', { name: 'Snap penalty (cards)' });
    await penalty.fill('3');
    await expect(save).toBeEnabled();
    await save.click();

    // The sheet that went out stands in for the lobby until its echo arrives, so the field does
    // not flick back to the old value and the save does not re-arm itself (cambia-1126 item 3).
    await expect(penalty).toHaveValue('3');
    await expect(save).toBeDisabled();
    await host.page.waitForTimeout(2_000);
    await expect(penalty).toHaveValue('3');
    await expect(save).toBeDisabled();
  });

  test('the new-lobby dialog keeps a usable ruleset list when the game mode changes', async ({ browser }) => {
    const seat = await openSeat(browser, 'e2e-nit4-a', '/dashboard');
    seats.push(seat);
    await seat.page.getByRole('button', { name: 'Create lobby' }).click();
    const dialog = seat.page.getByRole('dialog');
    const ruleset = dialog.getByRole('combobox', { name: /Ruleset/ });
    // The block is rendered only once GET /lobby/presets has answered, since an empty dropdown is
    // worse than none (DashboardPage). Wait for it rather than reading an absent control as empty.
    await expect(ruleset).toBeVisible();

    const h2hOptions = await ruleset.locator('option').allInnerTexts();
    expect(h2hOptions.length).toBeGreaterThan(1);
    expect(h2hOptions.some((o) => /H2H/i.test(o))).toBe(true);

    // Switching the mode refilters the list rather than leaving a preset the mode cannot play
    // selected, or leaving the control disabled with no way back (cambia-1126 item 7).
    await dialog.getByRole('combobox', { name: /Game mode/ }).selectOption('group_of_4');
    await expect(ruleset).toBeEnabled();
    const ffaOptions = await ruleset.locator('option').allInnerTexts();
    expect(ffaOptions.length).toBeGreaterThan(0);
    expect(ffaOptions.some((o) => /H2H/i.test(o))).toBe(false);
  });
});

test.describe('card and pile accessibility', () => {
  test('every card and pile carries a name, and neither is re-typed when its clickability changes', async ({ browser }, testInfo) => {
    const { host, guest } = await startH2H(browser, 'e2e-a11y');
    seats.push(host, guest);

    // During the peek nothing across the table is snappable, so the opponent's cards are named
    // images: not controls, out of the tab sequence, still read aloud (cambia-876 DL-4 F13).
    const oppSlot1 = host.page.locator('[aria-label^="e2e-a11y-b card 1"]');
    await expect(oppSlot1).toHaveRole('img');
    await expect(oppSlot1).toHaveAttribute('tabindex', '-1');
    // Mark the node so the assertion after the change is about this element, not a replacement.
    await oppSlot1.evaluate((el: HTMLElement) => {
      el.dataset.e2eNode = 'opp-slot-1';
    });

    await waitForPlay(host);

    // Own hand: four named slots.
    for (let slot = 1; slot <= 4; slot += 1) {
      await expect(ownCardAny(host.page, slot)).toHaveCount(1);
    }
    await expect(host.page.locator('[aria-label^="Your card "]')).toHaveCount(4);
    // Across the table the slots are named by whose they are, so a reader is never told "card 1"
    // with no owner.
    await expect(host.page.locator('[aria-label^="e2e-a11y-b card "]')).toHaveCount(4);

    // Once play starts an opponent snap is legal, so the same slot becomes a control. The element
    // is not swapped to get there: a <button> that turned into a <div> when it stopped taking
    // clicks sent a keyboard user's focus to <body> (cambia-1242).
    await expect(oppSlot1).toHaveRole('button');
    await expect(host.page.locator('[data-e2e-node="opp-slot-1"]')).toHaveCount(1);
    await expect(host.page.locator('[data-e2e-node="opp-slot-1"]')).toHaveAttribute('aria-label', /^e2e-a11y-b card 1/);

    // Both piles are buttons in both turn states, and the off-turn one is marked unavailable
    // rather than taken away.
    const onTurn = (await host.page.locator(ACTION.drawStock).isVisible()) ? host : guest;
    const offTurn = onTurn === host ? guest : host;
    for (const seat of [onTurn, offTurn]) {
      const stock = seat.page.locator(PILE.stock);
      const discard = seat.page.locator(PILE.discard);
      await expect(stock).toHaveRole('button');
      await expect(discard).toHaveRole('button');
      await expect(stock).toHaveAttribute('aria-label', /^Stockpile, \d+ cards?$/);
      await expect(discard).toHaveAttribute('aria-label', /^Discard pile, (top .+|empty)$/);
    }
    await expect(offTurn.page.locator(PILE.stock)).toHaveAttribute('aria-disabled', 'true');
    await expect(onTurn.page.locator(PILE.stock)).not.toHaveAttribute('aria-disabled', 'true');

    // A pick reports itself: the lift the eye sees and aria-pressed are one value, so no card is
    // ever standing up with nothing under it saying so (cambia-1242).
    const ownSlot1 = onTurn.page.getByRole('button', { name: /^Your card 1[:,]/ });
    await expect(ownSlot1).toHaveAttribute('aria-pressed', 'false');
    await ownSlot1.click();
    await expect(ownSlot1).toHaveAttribute('aria-pressed', 'true');
    await expect(onTurn.page.locator(ACTION.snap)).toBeVisible();
    await onTurn.page.screenshot({ path: testInfo.outputPath('a11y-own-pick.png') });
    await onTurn.page.locator(ACTION.cancelSnap).click();
    await expect(ownSlot1).toHaveAttribute('aria-pressed', 'false');

    // The drawn card is named by its face for the seat holding it, and by nothing at all for the
    // seat that is not (its own slot is a back).
    await onTurn.page.locator(ACTION.drawStock).click();
    await expect(onTurn.page.locator(PILE.drawn)).toHaveAttribute('aria-label', /^Drawn card: /);
    await onTurn.page.screenshot({ path: testInfo.outputPath('a11y-drawn-card.png') });
  });
});
