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

beforeEach(() => {
  vi.useFakeTimers();
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  useAuthStore.setState({ user: { id: 'user-1', username: 'You', is_ephemeral: false } });
  useCurrentLobbyStore.setState({ currentLobbyId: LOBBY_ID });
});

afterEach(() => {
  globalThis.WebSocket = RealWebSocket;
  vi.useRealTimers();
  useAuthStore.setState(authInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
  useGameStore.setState(gameInitial, true);
});

describe('useSocket retry budget', () => {
  it('stops dialing after MAX_RETRIES and leaves the table its stopped-trying copy', () => {
    renderHook(() => useSocket(LOBBY_ID));
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

    // DsGameTable reads `gaveUp` off this copy (its /stopped|after \d+ retries/i test), so the
    // wording is load-bearing, not cosmetic.
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
