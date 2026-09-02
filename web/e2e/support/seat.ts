// web/e2e/support/seat.ts
// A seat: one browser context pinned to one dev account, plus every raw failure signal that
// context produced.
//
// One context per seat, never one context with several tabs: `?as=<name>` pins identity in
// sessionStorage (src/lib/tabSession.ts), and while sessionStorage is already per tab, a shared
// context also shares the auth cookie, so a seat that falls back to the cookie silently becomes
// whichever seat wrote it last. Separate contexts make that impossible.
//
// Every signal is collected, none is filtered at the source. Pass 1's critic (note cambia-960)
// found the run had mapped some raw signals and dropped others, so a socket that closed on its own
// never entered the error set. What counts as a failure is decided in the assertion, where the
// reader can see the rule; what gets recorded is everything.
import { expect, type Browser, type BrowserContext, type ConsoleMessage, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { instrumentSockets } from './instrument';
import { dirname, join } from 'node:path';

/** Everything a seat's page reported that could carry a defect. */
export interface Signals {
  /** console.error text. */
  consoleErrors: string[];
  /** console.warn text. */
  consoleWarnings: string[];
  /** Uncaught exceptions and unhandled promise rejections, which Playwright reports alike. */
  pageErrors: string[];
  /** Requests the browser could not complete at all (DNS, refused, aborted). */
  requestFailures: string[];
  /** Completed HTTP responses with a 4xx or 5xx status. */
  httpErrors: string[];
  /** WebSocket closes, in order. A socket that closes is a failure signal until read otherwise. */
  socketCloses: string[];
  /** Every frame in both directions, for the log a defect report has to quote. */
  frames: string[];
}

export interface Seat {
  name: string;
  context: BrowserContext;
  page: Page;
  signals: Signals;
}

function emptySignals(): Signals {
  return {
    consoleErrors: [],
    consoleWarnings: [],
    pageErrors: [],
    requestFailures: [],
    httpErrors: [],
    socketCloses: [],
    frames: [],
  };
}

/** A short, stable stamp so frames and console lines can be read against each other. */
function stamp(): string {
  return new Date().toISOString().slice(11, 23);
}

/**
 * Opens a browser context pinned to `name` and lands it on `path`.
 *
 * `?as=<name>` needs the service running with CAMBIA_DEV_ACCOUNTS=1; globalSetup has already
 * refused the run if it is not. The boot parameter is stripped from the URL by the client itself
 * once the token is minted, so the page settles on the clean path.
 */
export async function openSeat(
  browser: Browser,
  name: string,
  path = '/',
  opts: { instrumentSockets?: boolean } = {}
): Promise<Seat> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const signals = emptySignals();
  // Off unless a test asks: the wrapper is behaviour-preserving, but a clean-console run should
  // be measuring the client and nothing else.
  if (opts.instrumentSockets) await instrumentSockets(page);

  page.on('console', (msg: ConsoleMessage) => {
    const line = `[${stamp()}] ${msg.text()}`;
    if (msg.type() === 'error') signals.consoleErrors.push(line);
    else if (msg.type() === 'warning') signals.consoleWarnings.push(line);
  });
  page.on('pageerror', (err) => {
    signals.pageErrors.push(`[${stamp()}] ${err.name}: ${err.message}`);
  });
  page.on('requestfailed', (req) => {
    signals.requestFailures.push(`[${stamp()}] ${req.method()} ${req.url()} ${req.failure()?.errorText ?? ''}`);
  });
  page.on('response', (res) => {
    if (res.status() >= 400) signals.httpErrors.push(`[${stamp()}] ${res.status()} ${res.request().method()} ${res.url()}`);
  });
  page.on('websocket', (ws) => {
    signals.frames.push(`[${stamp()}] OPEN ${ws.url()}`);
    ws.on('framesent', (f) => signals.frames.push(`[${stamp()}] > ${f.payload}`));
    ws.on('framereceived', (f) => signals.frames.push(`[${stamp()}] < ${f.payload}`));
    ws.on('socketerror', (e) => signals.frames.push(`[${stamp()}] SOCKET ERROR ${e}`));
    ws.on('close', () => {
      signals.frames.push(`[${stamp()}] CLOSE ${ws.url()}`);
      signals.socketCloses.push(`[${stamp()}] ${ws.url()}`);
    });
  });

  // The pin is taken on the root path first, never straight on the destination: a protected route
  // redirects to /login the moment it renders unauthenticated, and that redirect can outrun the
  // mint, leaving the tab on the sign-in page with its boot intent already consumed.
  await page.goto(`/?as=${encodeURIComponent(name)}`);
  await expect(page.getByRole('button', { name: `Dev session: ${name} (tab)` })).toBeVisible({ timeout: 30_000 });
  await page.waitForURL('**/dashboard', { timeout: 30_000 });
  if (path !== '/' && path !== '/dashboard') await page.goto(path);
  return { name, context, page, signals };
}

/** Closes a seat's context. Safe to call on a seat whose page has already gone. */
export async function closeSeat(seat: Seat): Promise<void> {
  await seat.context.close().catch(() => undefined);
}

/**
 * Writes a seat's frame log and failure signals under the run's artifact directory.
 *
 * Called on every run, not only on failure: a defect report has to quote the frames around the
 * moment, and a log that only exists for red runs cannot be quoted for a green one that still
 * looked wrong.
 */
export function dumpSignals(seat: Seat, runName: string, artifactDir: string): string {
  const file = join(artifactDir, `${runName}-${seat.name}.log`);
  mkdirSync(dirname(file), { recursive: true });
  const body = [
    `# seat ${seat.name} run ${runName}`,
    '',
    '## console errors',
    ...seat.signals.consoleErrors,
    '',
    '## console warnings',
    ...seat.signals.consoleWarnings,
    '',
    '## page errors',
    ...seat.signals.pageErrors,
    '',
    '## request failures',
    ...seat.signals.requestFailures,
    '',
    '## http 4xx/5xx',
    ...seat.signals.httpErrors,
    '',
    '## socket closes',
    ...seat.signals.socketCloses,
    '',
    '## frames',
    ...seat.signals.frames,
    '',
  ].join('\n');
  writeFileSync(file, body, 'utf8');
  return file;
}

/**
 * The console noise this client is known to produce that is not a defect.
 *
 * Kept as an explicit, commented list rather than a regex sprinkled through the assertions: an
 * entry here is a decision that a line is benign, and it should be re-argued when it changes.
 * Empty on purpose right now. Nothing observed in this pass qualified, and an empty list is the
 * honest statement that the shipped surface is quiet.
 */
export const BENIGN_CONSOLE: RegExp[] = [];

/** Console errors that are not on the benign list. */
export function realConsoleErrors(seat: Seat): string[] {
  return seat.signals.consoleErrors.filter((line) => !BENIGN_CONSOLE.some((re) => re.test(line)));
}
