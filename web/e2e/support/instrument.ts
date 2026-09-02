// web/e2e/support/instrument.ts
// Page-side instrumentation for the fault-injection tests, installed before any app script runs.
//
// It wraps `window.WebSocket` in a Proxy that still constructs real sockets, so the client under
// test is unchanged until a test asks for a fault. Three things it makes possible, none of which
// the client offers a seam for:
//
//   1. counting dials, which is the only direct evidence that a retry cap actually caps
//   2. refusing dials, so a cap can be reached without taking the server down for the other seats
//   3. replaying the last server snapshot with a field bent out of shape, which is the render
//      throw the table's error boundary exists to catch
//
// Faults are injected as events on the real socket object rather than by closing it, so the
// server's own view of the game is untouched and the run stays repeatable.
import type { Page } from '@playwright/test';

/** Names the instrumentation hangs on `window`, kept in one place for both sides of the wire. */
export const HOOK = '__cambiaE2E';

/** Installed with addInitScript, so it is in place before the app's first module evaluates. */
function initScript(hook: string): void {
  const w = window as unknown as Record<string, unknown>;
  const Native = window.WebSocket;

  interface State {
    dials: { url: string; at: number }[];
    sockets: WebSocket[];
    lastSync: unknown;
    refuse: boolean;
  }
  const state: State = { dials: [], sockets: [], lastSync: null, refuse: false };
  w[hook] = state;

  /** A socket that never opens and closes unclean, which is what a refused dial looks like. */
  class RefusedSocket extends EventTarget {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;
    readyState = 0;
    url: string;
    onopen: ((e: Event) => void) | null = null;
    onmessage: ((e: MessageEvent) => void) | null = null;
    onerror: ((e: Event) => void) | null = null;
    onclose: ((e: CloseEvent) => void) | null = null;

    constructor(url: string) {
      super();
      this.url = url;
      // A tick, not zero: the hook assigns its handlers after the constructor returns.
      setTimeout(() => {
        this.readyState = 3;
        const ev = new CloseEvent('close', { code: 1006, wasClean: false, reason: 'e2e refused dial' });
        this.onerror?.(new Event('error'));
        this.onclose?.(ev);
        this.dispatchEvent(ev);
      }, 5);
    }

    send(): void {
      /* a refused socket carries nothing */
    }

    close(): void {
      this.readyState = 3;
    }
  }

  window.WebSocket = new Proxy(Native, {
    construct(target, args: [string, (string | string[])?]) {
      const url = String(args[0]);
      state.dials.push({ url, at: Date.now() });
      if (state.refuse && url.includes('/ws/')) {
        return new RefusedSocket(url) as unknown as WebSocket;
      }
      const socket = Reflect.construct(target, args) as WebSocket;
      state.sockets.push(socket);
      socket.addEventListener('message', (event: MessageEvent) => {
        try {
          const envelope = JSON.parse(String(event.data));
          if (envelope && envelope.type === 'private_sync_state') state.lastSync = envelope;
        } catch {
          /* not a frame this hook cares about */
        }
      });
      return socket;
    },
  }) as typeof WebSocket;
}

/** Installs the hook on a page that has not navigated yet. */
export async function instrumentSockets(page: Page): Promise<void> {
  await page.addInitScript(initScript, HOOK);
}

/** How many sockets the page has dialled to a lobby so far. */
export async function lobbyDials(page: Page): Promise<number> {
  return page.evaluate(([hook]) => {
    const state = (window as unknown as Record<string, { dials: { url: string }[] }>)[hook];
    return state.dials.filter((d) => d.url.includes('/ws/')).length;
  }, [HOOK]);
}

/** Makes every further lobby dial fail the way a refused connection does. */
export async function refuseDials(page: Page): Promise<void> {
  await page.evaluate(([hook]) => {
    (window as unknown as Record<string, { refuse: boolean }>)[hook].refuse = true;
  }, [HOOK]);
}

/**
 * Drops the live lobby socket the way a lost network does: an unclean close event, with the real
 * socket left alone so the server keeps the seat and the rest of the table plays on.
 */
export async function dropSocket(page: Page, code = 1006): Promise<void> {
  await page.evaluate(
    ([hook, closeCode]) => {
      const state = (window as unknown as Record<string, { sockets: WebSocket[] }>)[hook as string];
      const live = state.sockets.filter((s) => s.url.includes('/ws/'));
      const socket = live[live.length - 1];
      if (!socket) throw new Error('no lobby socket to drop');
      const ev = new CloseEvent('close', {
        code: closeCode as number,
        wasClean: (closeCode as number) === 1000,
        reason: 'e2e drop',
      });
      (socket as unknown as { onclose?: (e: CloseEvent) => void }).onclose?.(ev);
    },
    [HOOK, code] as [string, number]
  );
}

/**
 * Replays the last server snapshot with every hand replaced by a value the table cannot render.
 *
 * `revealedHand` is read as an array by DsGameTable's own-hand render (`hand.map`), and by nothing
 * in the store on a post-pregame snapshot, so a string lands in state and throws in render. That
 * is the shape of failure the boundary was built for: the table reads live server state through
 * optional-chained shapes, and an unexpected payload is a real failure mode, not a hypothetical.
 */
export async function injectUnrenderableSnapshot(page: Page): Promise<void> {
  await page.evaluate(([hook]) => {
    const state = (window as unknown as Record<string, { sockets: WebSocket[]; lastSync: unknown }>)[hook];
    if (!state.lastSync) throw new Error('no private_sync_state seen yet');
    const live = state.sockets.filter((s) => s.url.includes('/ws/'));
    const socket = live[live.length - 1];
    if (!socket) throw new Error('no lobby socket to inject on');
    const envelope = JSON.parse(JSON.stringify(state.lastSync)) as {
      payload: { state: { players: { revealedHand: unknown }[] } };
    };
    for (const player of envelope.payload.state.players) player.revealedHand = 'not-a-hand';
    const ev = new MessageEvent('message', { data: JSON.stringify(envelope) });
    (socket as unknown as { onmessage?: (e: MessageEvent) => void }).onmessage?.(ev);
  }, [HOOK]);
}
