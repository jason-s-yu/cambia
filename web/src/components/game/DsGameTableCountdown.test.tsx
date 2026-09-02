// src/components/game/DsGameTableCountdown.test.tsx
// The reconnect window as the table draws it (cambia-1241). The server publishes an away seat's
// deadline twice over - on the player_reconnecting event and on `reconnectDeadline` in every sync
// snapshot - and the felt used to read neither: an away chair said "Reconnecting" and nothing
// else, so a player deciding whether to wait had no idea how long the table would hold the seat.
//
// The snapshot is the source these assert against on purpose. It is the one a client that joined
// or resynced mid-window has, and the event is the one it does not, so a countdown driven by the
// event alone would leave exactly the reloaded client this ticket is about with a blank chair.
//
// Time is frozen and the clocks are deliberately skewed: a deadline is a stamp from the SERVER's
// clock, and reading it against Date.now() (which is what the felt did) is right only for a client
// whose clock happens to agree with the server's. Held apart from DsGameTableReconnect.test.tsx,
// which owns the own-seat/give-up half of the same felt.
import { act, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGameStore } from '@/stores/gameStore';
import { buildGameState, OPP_ID } from '@/test/fixtures/gameState';
import { renderDsGameTable, resetGameTableStores } from '@/test/renderDsGameTable';

/** A frozen client clock. Every deadline below is expressed against the server's. */
const CLIENT_NOW = 1_700_000_000_000;

/**
 * A table whose single opponent dropped `graceMs` ago-in-the-future, with the client clock running
 * `skewMs` behind the server's. Positive skew is a slow client, which is what
 * `serverClockOffsetMs = serverNow - Date.now()` reports.
 */
function awayOpponent(graceMs: number, skewMs: number) {
  const serverNow = CLIENT_NOW + skewMs;
  return {
    gameState: buildGameState({
      serverNow,
      opponent: { connected: false, reconnectDeadline: serverNow + graceMs }
    }),
    gameStoreState: { serverClockOffsetMs: skewMs }
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(CLIENT_NOW);
});

afterEach(() => {
  resetGameTableStores();
  vi.useRealTimers();
});

describe('DsGameTable reconnect countdown', () => {
  it('counts an away seat down from the deadline the snapshot carries', () => {
    renderDsGameTable(awayOpponent(45_000, 0));

    expect(screen.getByText('Reconnecting 45s')).toBeInTheDocument();
    expect(screen.queryByText('Reconnecting')).not.toBeInTheDocument();
  });

  it('reads the deadline on the server clock, not this one', () => {
    // The client is half a minute slow. Against Date.now() the same 45-second window reads 75s.
    renderDsGameTable(awayOpponent(45_000, 30_000));

    expect(screen.getByText('Reconnecting 45s')).toBeInTheDocument();
    expect(screen.queryByText('Reconnecting 75s')).not.toBeInTheDocument();
    expect(screen.queryByText('Reconnecting 15s')).not.toBeInTheDocument();
  });

  it('keeps counting while the window runs, and stops at zero', () => {
    renderDsGameTable(awayOpponent(3_000, 12_000));

    expect(screen.getByText('Reconnecting 3s')).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(screen.getByText('Reconnecting 1s')).toBeInTheDocument();

    // Past the deadline the seat is still held on screen until the server says otherwise
    // (player_forfeited, or the next snapshot), and a countdown must not run negative there.
    act(() => {
      vi.advanceTimersByTime(4_000);
    });
    expect(screen.getByText('Reconnecting 0s')).toBeInTheDocument();
  });

  it('falls back to the plain label when no window is open', () => {
    // A socket that dropped with no grace at all: `connected` false, no deadline published.
    renderDsGameTable({
      gameState: buildGameState({ opponent: { connected: false } })
    });

    expect(screen.getByText('Reconnecting')).toBeInTheDocument();
    expect(screen.queryByText(/Reconnecting \d/)).not.toBeInTheDocument();
  });

  it('leaves a forfeited seat alone', () => {
    // The window closed and the seat is gone; a countdown on it would be a lie about a seat that
    // is not coming back (seatStateFor puts forfeited ahead of away).
    const serverNow = CLIENT_NOW;
    renderDsGameTable({
      gameState: buildGameState({
        serverNow,
        opponent: { connected: false, forfeited: true, reconnectDeadline: serverNow + 20_000 }
      })
    });

    expect(screen.getByText('Forfeited')).toBeInTheDocument();
    expect(screen.queryByText(/Reconnecting/)).not.toBeInTheDocument();
  });

  it('does not stand in for an ability prompt on the same seat', () => {
    const serverNow = CLIENT_NOW;
    renderDsGameTable({
      gameState: buildGameState({
        serverNow,
        currentPlayerId: OPP_ID,
        opponent: { connected: false, reconnectDeadline: serverNow + 20_000 },
        specialAction: { active: true, playerId: OPP_ID, cardRank: 'K' }
      })
    });

    const seat = screen.getByText('Look and swap');
    expect(seat).toBeInTheDocument();
    expect(screen.queryByText(/Reconnecting/)).not.toBeInTheDocument();
  });
});

describe('DsGameTable reconnect notice', () => {
  it('quotes the window the server set, corrected for a skewed client clock', () => {
    const skewMs = 30_000;
    const serverNow = CLIENT_NOW + skewMs;
    renderDsGameTable({ gameState: buildGameState({ serverNow }) });

    // The notice is the moment the seat changed, so it fires on a presence nonce this mount has
    // not seen: seeding it before render would leave the effect with nothing new to report.
    act(() => {
      useGameStore.setState({
        serverClockOffsetMs: skewMs,
        lastPresence: { nonce: 1, kind: 'reconnecting', playerId: OPP_ID, deadline: serverNow + 90_000, graceSeconds: 90 }
      });
    });

    expect(screen.getByText('Rival dropped. 90s to reconnect.')).toBeInTheDocument();
    // Against Date.now() the same window reads two minutes on this client.
    expect(screen.queryByText('Rival dropped. 120s to reconnect.')).not.toBeInTheDocument();
  });

  it('quotes the house rule when the event carried no deadline', () => {
    renderDsGameTable({ gameState: buildGameState({ serverNow: CLIENT_NOW }) });

    act(() => {
      useGameStore.setState({
        serverClockOffsetMs: 0,
        lastPresence: { nonce: 1, kind: 'reconnecting', playerId: OPP_ID, deadline: null, graceSeconds: 90 }
      });
    });

    expect(screen.getByText('Rival dropped. 90s to reconnect.')).toBeInTheDocument();
  });

  it('says only that they dropped when the window is already spent', () => {
    renderDsGameTable({ gameState: buildGameState({ serverNow: CLIENT_NOW }) });

    // graceSeconds is deliberately set: a spent deadline must not fall back to the rule's full
    // length, which would announce a fresh 90 seconds for a window the server has closed.
    act(() => {
      useGameStore.setState({
        serverClockOffsetMs: 0,
        lastPresence: { nonce: 1, kind: 'reconnecting', playerId: OPP_ID, deadline: CLIENT_NOW - 5_000, graceSeconds: 90 }
      });
    });

    expect(screen.getByText('Rival dropped.')).toBeInTheDocument();
    expect(screen.queryByText('Rival dropped. 90s to reconnect.')).not.toBeInTheDocument();
  });
});
