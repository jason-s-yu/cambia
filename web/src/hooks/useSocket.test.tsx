// src/hooks/useSocket.test.tsx
// Reconnect-state coverage for useSocket (cambia-1236). The hook owns a small state machine over a
// WebSocket the tests stand in for: a fake with the same four readyState values and the same four
// handler properties, plus test-only `accept()` / `fail()` so a handshake can be landed or dropped
// on demand. Every dial the hook makes constructs one, so `FakeWebSocket.instances` is the record
// of what it actually did, and the retry cap is a statement about the length of that list.
//
// Fake timers throughout: the reconnect path is exponential backoff on window.setTimeout, so the
// only way to watch a budget being spent is to run the clock forward between failures.
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NIL as NIL_UUID } from 'uuid';
import { useSocket } from '@/hooks/useSocket';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useGameStore } from '@/stores/gameStore';

/** Matches MAX_RETRIES in the hook. A dial plus this many redials is the whole budget. */
const MAX_RETRIES = 5;
/** Longer than the largest backoff the hook can schedule (2^4 * 1000 + up to 1000 ms of jitter). */
const PAST_ANY_BACKOFF_MS = 120_000;

const LOBBY_ID = '11111111-2222-4333-8444-555555555555';
const OTHER_LOBBY_ID = '99999999-8888-4777-8666-555555555555';

interface CloseCall {
  code?: number;
  reason?: string;
  /** The state the socket was in when close() was called. CONNECTING here is what makes a browser
   *  log "WebSocket is closed before the connection is established", so the tests below are
   *  specific about which teardowns are allowed to produce it (cambia-1236 AC4). */
  readyStateAtCall: number;
}

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readonly protocols: string | string[] | undefined;
  readonly closeCalls: CloseCall[] = [];
  readyState: number = FakeWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    FakeWebSocket.instances.push(this);
  }

  send(): void {}

  close(code?: number, reason?: string): void {
    this.closeCalls.push({ code, reason, readyStateAtCall: this.readyState });
    const established = this.readyState === FakeWebSocket.OPEN;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ wasClean: established, code: code ?? 1000, reason: reason ?? '' } as CloseEvent);
  }

  /** The handshake completes. */
  accept(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  /** The server is unreachable or refuses: an abnormal close with no clean shutdown. */
  fail(code = 1006): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ wasClean: false, code, reason: '' } as CloseEvent);
  }

  /** The hub ends the socket itself: a completed close handshake the client never asked for.
   *  Its defaults are what Connection.Close sends (StatusGoingAway, "connection closed",
   *  service/internal/hub/connection.go). */
  serverClose(code = 1001, reason = 'connection closed'): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ wasClean: true, code, reason } as CloseEvent);
  }
}

const RealWebSocket = globalThis.WebSocket;
const authInitial = useAuthStore.getState();
const lobbyInitial = useCurrentLobbyStore.getState();
const gameInitial = useGameStore.getState();

/** The socket the hook most recently opened. */
function latestSocket(): FakeWebSocket {
  const socket = FakeWebSocket.instances.at(-1);
  if (!socket) throw new Error('the hook opened no socket');
  return socket;
}

/** Drops the current dial and lets the whole backoff elapse, so any scheduled redial has run. */
function dropAndWaitOutBackoff(): void {
  act(() => {
    latestSocket().fail();
  });
  act(() => {
    vi.advanceTimersByTime(PAST_ANY_BACKOFF_MS);
  });
}

/**
 * In these flows every socket the hook opens either fails on its own or is closed after it opened.
 * A close while CONNECTING would mean the hook had opened a socket and abandoned it mid-handshake,
 * which is the orphan dial behind both the 404 retry loop and the browser's "WebSocket is closed
 * before the connection is established" line on Leave (cambia-1236 AC4, cambia-1126 item 1).
 * Deliberately abandoning a handshake is a separate case and is asserted directly where it happens.
 */
function expectNoAbandonedHandshake(): void {
  for (const socket of FakeWebSocket.instances) {
    for (const call of socket.closeCalls) {
      expect(call.readyStateAtCall).not.toBe(FakeWebSocket.CONNECTING);
    }
  }
}

/** Backgrounds or foregrounds the tab without touching the clock. jsdom answers 'visible' from the
 *  prototype, so the override is an own property and afterEach deletes it again. */
function setVisibility(state: 'visible' | 'hidden'): void {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
}

/** The tab goes into the background: the state flips and the browser fires the event either way,
 *  which is why the hook has to check the state rather than trust the event (cambia-1521). */
function leaveForBackground(): void {
  setVisibility('hidden');
  act(() => {
    document.dispatchEvent(new Event('visibilitychange'));
  });
}

/** The user comes back to the tab. */
function returnToForeground(): void {
  setVisibility('visible');
  act(() => {
    document.dispatchEvent(new Event('visibilitychange'));
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  useAuthStore.setState({ user: { id: 'user-1', username: 'You', is_ephemeral: false } });
  useCurrentLobbyStore.setState({ currentLobbyId: LOBBY_ID });
});

afterEach(() => {
  globalThis.WebSocket = RealWebSocket;
  Reflect.deleteProperty(document, 'visibilityState');
  vi.useRealTimers();
  useAuthStore.setState(authInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
  useGameStore.setState(gameInitial, true);
});

describe('useSocket retry budget', () => {
  it('stops dialing after MAX_RETRIES and reports the spent budget as a state', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    expect(FakeWebSocket.instances).toHaveLength(1);

    // Each drop spends one retry and the backoff redials, so the budget buys MAX_RETRIES redials
    // on top of the opening dial.
    for (let i = 0; i < MAX_RETRIES; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);

    // The budget is spent: this drop schedules nothing, however long the clock runs.
    dropAndWaitOutBackoff();
    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);

    // What DsGameTable branches on. The copy below is the player-facing sentence and nothing
    // reads it back: the table used to rebuild this state with a regex over it, and a give-up the
    // regex did not match left the felt reconnecting forever (cambia-1239 review).
    expect(result.current.gaveUp).toBe('retries');
    const lobby = useCurrentLobbyStore.getState();
    expect(lobby.error).toBe(`Lost connection after ${MAX_RETRIES} retries.`);
    expect(lobby.isConnected).toBe(false);
    expect(lobby.isLoading).toBe(false);
    expectNoAbandonedHandshake();
  });

  it('keeps the budget across lobby-store writes and re-renders during a retry (cambia-1236 AC1)', () => {
    const { rerender } = renderHook(() => useSocket(LOBBY_ID));
    expect(FakeWebSocket.instances).toHaveLength(1);

    // A drop with the backoff still pending: the redial belongs to the timer, nobody else.
    act(() => {
      latestSocket().fail();
    });
    expect(FakeWebSocket.instances).toHaveLength(1);

    // Chat, a phase change, a dropped-action notice: ordinary lobby traffic, each one a write that
    // used to change the store object's identity, re-create connectWebSocket, re-run the connect
    // effect and reset the counter before the backoff ever fired.
    act(() => {
      useCurrentLobbyStore.getState().setPhase('countdown');
      useCurrentLobbyStore.getState().addChatMessage({ user_id: 'user-2', username: 'Rival', msg: 'hi', ts: 1 });
      useCurrentLobbyStore.getState().noteDroppedAction();
    });
    rerender();
    expect(FakeWebSocket.instances).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(2);

    // Four more drops, each with a store write and a re-render on top, still exhaust the budget at
    // the same count: the writes bought no extra dials.
    for (let i = 0; i < MAX_RETRIES - 1; i++) {
      act(() => {
        latestSocket().fail();
        useCurrentLobbyStore.getState().noteDroppedAction();
      });
      rerender();
      act(() => {
        vi.advanceTimersByTime(PAST_ANY_BACKOFF_MS);
      });
    }
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);

    dropAndWaitOutBackoff();
    rerender();
    act(() => {
      vi.advanceTimersByTime(PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);
    expect(useCurrentLobbyStore.getState().error).toBe(`Lost connection after ${MAX_RETRIES} retries.`);
    expectNoAbandonedHandshake();
  });

  it('resets the budget on a successful open and on nothing else', () => {
    renderHook(() => useSocket(LOBBY_ID));

    for (let i = 0; i < 3; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(4);

    act(() => {
      latestSocket().accept();
    });
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);

    // A full budget again from here: MAX_RETRIES redials before the hook gives up, not the two the
    // earlier failures would have left.
    for (let i = 0; i < MAX_RETRIES; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(4 + MAX_RETRIES);

    dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(4 + MAX_RETRIES);
    expectNoAbandonedHandshake();
  });

  it('dials again once the hook is pointed at a different lobby', () => {
    const { rerender } = renderHook(({ id }: { id: string }) => useSocket(id), { initialProps: { id: LOBBY_ID } });

    for (let i = 0; i <= MAX_RETRIES; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);

    useCurrentLobbyStore.setState({ currentLobbyId: OTHER_LOBBY_ID });
    rerender({ id: OTHER_LOBBY_ID });

    expect(FakeWebSocket.instances).toHaveLength(2 + MAX_RETRIES);
    expect(latestSocket().url).toContain(OTHER_LOBBY_ID);
    expectNoAbandonedHandshake();
  });
});

/**
 * The backgrounded-tab forfeit (cambia-1521). The server holds a dropped player's seat for
 * DisconnectGraceSec (90 by default, with ForfeitOnDisconnect on) and that is unchanged; what the
 * client owes is getting back inside it. A hidden tab cannot: its timers are throttled to roughly
 * one wake-up a minute, so the backoff above can land after the seat is already forfeited, and a
 * tab restored from the back/forward cache never ran it at all.
 *
 * The clock is deliberately never advanced in these tests. Every dial they assert is one the hook
 * made because the page came back, not because a timer fired: that is the whole property.
 */
describe('useSocket backgrounded tab (cambia-1521)', () => {
  it('dials on return rather than waiting out a backoff a hidden tab never runs', () => {
    renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);

    // The socket dies while the tab is away - a sleeping device, a network change, the browser
    // discarding the page. The backoff is scheduled and then sits there.
    leaveForBackground();
    act(() => {
      latestSocket().fail();
    });
    expect(FakeWebSocket.instances).toHaveLength(1);

    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(2);

    // Inside the grace window the seat is still the player's, so the reconnect resumes it.
    act(() => {
      latestSocket().accept();
    });
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);
    expectNoAbandonedHandshake();
  });

  it('redials on a back/forward-cache restore, where no visibility change fires', () => {
    renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });

    // Entering the cache closes the socket; the page resumes holding a dead one, still "visible".
    act(() => {
      latestSocket().fail();
    });
    expect(FakeWebSocket.instances).toHaveLength(1);

    act(() => {
      window.dispatchEvent(new Event('pageshow'));
    });
    expect(FakeWebSocket.instances).toHaveLength(2);
    expectNoAbandonedHandshake();
  });

  it('collapses a pending backoff without spending extra budget (cambia-1236 AC2)', () => {
    renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().fail();
    });
    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(2);

    // The collapsed attempt was the retry that was already counted, so the budget still ends where
    // it always did: MAX_RETRIES redials on top of the opening dial, and then nothing.
    for (let i = 0; i < MAX_RETRIES - 1; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);

    dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);
    expect(useCurrentLobbyStore.getState().error).toBe(`Lost connection after ${MAX_RETRIES} retries.`);
    expectNoAbandonedHandshake();
  });

  it('buys one fresh budget per return and no more', () => {
    renderHook(() => useSocket(LOBBY_ID));
    for (let i = 0; i <= MAX_RETRIES; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(1 + MAX_RETRIES);
    expect(useCurrentLobbyStore.getState().error).toBe(`Lost connection after ${MAX_RETRIES} retries.`);

    // A budget spent while nobody was looking says the network was unusable then, which returning
    // is evidence against: one dial now, and the stopped-trying copy comes off the table.
    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(2 + MAX_RETRIES);
    expect(useCurrentLobbyStore.getState().error).toBeNull();
    expect(useCurrentLobbyStore.getState().isLoading).toBe(true);

    // That is one budget, not an unbounded supply. It runs out exactly like the first one.
    for (let i = 0; i <= MAX_RETRIES; i++) dropAndWaitOutBackoff();
    expect(FakeWebSocket.instances).toHaveLength(2 + 2 * MAX_RETRIES);

    // Nothing the page does on its own buys another: not the clock, not the tab going away again.
    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    leaveForBackground();
    expect(FakeWebSocket.instances).toHaveLength(2 + 2 * MAX_RETRIES);

    // Only another return does, which is what bounds the dial count by the user's own tab switches
    // rather than by a loop the hook can run for itself (the 521 dials of cambia-1236).
    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(3 + 2 * MAX_RETRIES);
    expectNoAbandonedHandshake();
  });

  it('leaves an open socket alone on return', () => {
    renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });

    leaveForBackground();
    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);
  });

  it('does not redial a lobby the hub refused by name', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    const socket = latestSocket();
    act(() => {
      socket.accept();
    });

    // lobby_not_found is the hub saying the lobby is gone. Coming back to the tab does not bring
    // it back, so this give-up survives a return where a spent retry budget does not.
    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({ seq: 1, type: 'error', payload: { code: 'lobby_not_found', message: 'Lobby not found.' } })
      } as MessageEvent);
    });
    expect(useCurrentLobbyStore.getState().currentLobbyId).toBeNull();
    expect(result.current.gaveUp).toBe('refused');

    returnToForeground();
    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(result.current.gaveUp).toBe('refused');
  });

  it('does not undo an explicit close on return', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });
    act(() => {
      result.current.closeSocket();
    });

    returnToForeground();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(useCurrentLobbyStore.getState().isConnected).toBe(false);
  });
});

/**
 * The close the client did not ask for (cambia-1239 review). The hub ends a socket cleanly in
 * ordinary play: opening the lobby in a second tab displaces the first one's socket, and leave and
 * hub cleanup after a dissolve or an idle reap do the same (service/internal/hub/hub.go). Nothing
 * redials it, so the state the hook reports has to say that; while it was reconstructed from the
 * error copy the felt read this as a reconnect in progress and locked every control for the rest
 * of the round.
 */
describe('useSocket hub-initiated close', () => {
  it('reports a clean close from the hub as terminal, not as a reconnect', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });
    expect(result.current.gaveUp).toBeNull();

    act(() => {
      latestSocket().serverClose();
    });

    expect(result.current.gaveUp).toBe('closed');
    expect(useCurrentLobbyStore.getState().isConnected).toBe(false);
    expect(useCurrentLobbyStore.getState().isLoading).toBe(false);
    // The copy that goes with it, which nothing branches on any more.
    expect(useCurrentLobbyStore.getState().error).toBe('Disconnected: connection closed');
  });

  it('does not dial again after one, on the clock or on a return to the tab', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });
    act(() => {
      latestSocket().serverClose();
    });

    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    leaveForBackground();
    returnToForeground();

    // A displaced socket stays displaced: dialing again would take the lobby back off whichever
    // tab now holds it. Only a fresh decision to connect (reopenSocket, a different lobby) does.
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(result.current.gaveUp).toBe('closed');
  });

  it('clears the state when the connection is deliberately taken back', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      latestSocket().accept();
    });
    act(() => {
      latestSocket().serverClose();
    });
    expect(result.current.gaveUp).toBe('closed');

    // reopenSocket is the refused-leave path (cambia-1520): the player is still at the table and
    // asked for the connection back, so the felt must stop showing a dead socket.
    act(() => {
      result.current.reopenSocket();
    });
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(result.current.gaveUp).toBeNull();

    act(() => {
      latestSocket().accept();
    });
    expect(result.current.gaveUp).toBeNull();
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);
    expectNoAbandonedHandshake();
  });
});

describe('useSocket lobby id gate', () => {
  it.each([
    ['the nil UUID', NIL_UUID],
    ['a non-UUID id', 'not-a-lobby'],
    ['an empty id', ''],
    ['a null id', null],
    ['an undefined id', undefined]
  ])('never opens a socket for %s', (_label, id: string | null | undefined) => {
    useCurrentLobbyStore.setState({ currentLobbyId: id ?? null });
    renderHook(() => useSocket(id));

    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it('reports an id it refused rather than retrying it', () => {
    useCurrentLobbyStore.setState({ currentLobbyId: NIL_UUID });
    renderHook(() => useSocket(NIL_UUID));

    expect(useCurrentLobbyStore.getState().error).toBe('Cannot connect: Invalid lobby ID.');
    expect(useCurrentLobbyStore.getState().isLoading).toBe(false);
  });

  it('drops a live socket when the id it is pointed at goes invalid', () => {
    const { rerender } = renderHook(({ id }: { id: string }) => useSocket(id), { initialProps: { id: LOBBY_ID } });
    const socket = latestSocket();
    act(() => {
      socket.accept();
    });

    rerender({ id: NIL_UUID });

    expect(socket.closeCalls).toHaveLength(1);
    expect(socket.closeCalls[0].readyStateAtCall).toBe(FakeWebSocket.OPEN);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

describe('useSocket teardown', () => {
  it('leaves an established lobby without opening a second socket (cambia-1174, cambia-985 R4)', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    const socket = latestSocket();
    act(() => {
      socket.accept();
    });
    expect(useCurrentLobbyStore.getState().isConnected).toBe(true);

    // The regression: closeSocket writes to the lobby store, and those writes used to re-run the
    // connect effect, which opened a fresh socket to the lobby being left. That orphan was the
    // 404 retry loop (cambia-1126 item 1), and closing it mid-handshake on the redirect that
    // followed was the browser's "closed before the connection is established" line on every
    // Leave. One socket, closed once it was open, is the whole teardown.
    act(() => {
      result.current.closeSocket();
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(socket.closeCalls).toEqual([
      { code: 1000, reason: 'User initiated disconnect', readyStateAtCall: FakeWebSocket.OPEN }
    ]);
    expect(useCurrentLobbyStore.getState().isConnected).toBe(false);

    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    expectNoAbandonedHandshake();
  });

  it('aborts a handshake still in flight rather than letting it land and rejoin the lobby', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    const socket = latestSocket();
    expect(socket.readyState).toBe(FakeWebSocket.CONNECTING);

    // Leaving before the handshake lands is the one case where the hook does close a CONNECTING
    // socket, and it has to: the hub joins the lobby on connect (ws.go step 7 MarkJoinedUnsafe),
    // so a handshake allowed to complete after POST /lobby/{id}/leave would hand the membership
    // straight back. The browser logs a line for the abort; the membership stays released.
    act(() => {
      result.current.closeSocket();
    });
    expect(socket.closeCalls).toEqual([
      { code: 1000, reason: 'User initiated disconnect', readyStateAtCall: FakeWebSocket.CONNECTING }
    ]);
    expect(FakeWebSocket.instances).toHaveLength(1);

    // A server that answers anyway reaches nothing: the handlers came off with the close.
    act(() => {
      socket.accept();
    });
    expect(useCurrentLobbyStore.getState().isConnected).toBe(false);
  });

  it('does not redial after an explicit close', () => {
    const { result } = renderHook(() => useSocket(LOBBY_ID));
    act(() => {
      result.current.closeSocket();
    });
    act(() => {
      vi.advanceTimersByTime(10 * PAST_ANY_BACKOFF_MS);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it('drops the socket when the lobby id goes null', () => {
    const { rerender } = renderHook(({ id }: { id: string | null }) => useSocket(id), { initialProps: { id: LOBBY_ID as string | null } });
    const socket = latestSocket();
    act(() => {
      socket.accept();
    });

    useCurrentLobbyStore.setState({ currentLobbyId: null });
    rerender({ id: null });

    expect(socket.closeCalls).toEqual([
      { code: 1000, reason: 'Lobby ID became null', readyStateAtCall: FakeWebSocket.OPEN }
    ]);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(useCurrentLobbyStore.getState().isConnected).toBe(false);
    expectNoAbandonedHandshake();
  });
});
