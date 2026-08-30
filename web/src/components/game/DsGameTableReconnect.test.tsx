// src/components/game/DsGameTableReconnect.test.tsx
// The felt half of cambia-1236: what the table shows once useSocket has spent its retry budget.
// The hook's own coverage (src/hooks/useSocket.test.tsx) proves the cap holds and pins the copy it
// writes; these assert the table turns that copy into the disconnected state a player can act on,
// and that an ordinary retry is still shown as a retry. Held in its own file so the ticket's tests
// do not collide with the general DsGameTable suite.
//
// Queried by the strings unique to each state (the status line's "Disconnected." / "Reconnecting."
// and the hint line) rather than by the felt chip's bare "Reconnecting": PlayerSeat labels a
// disconnected seat "Reconnecting" too, so that word alone is ambiguous on an offline table.
import { screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { renderDsGameTable, resetGameTableStores } from '@/test/renderDsGameTable';

/** Verbatim from useSocket's give-up branch; DsGameTable's `gaveUp` test reads this string. */
const GAVE_UP_COPY = 'Lost connection after 5 retries.';
/** Verbatim from useSocket's retry branch. */
const RETRYING_COPY = 'Connection lost. Retrying... (Attempt 3)';

afterEach(() => {
  resetGameTableStores();
});

describe('DsGameTable connection state', () => {
  it('shows the stopped-trying state once the hook reports a spent retry budget', () => {
    renderDsGameTable({ connected: false, connectionError: GAVE_UP_COPY });

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
    expect(screen.getByText('Disconnected.')).toBeInTheDocument();
    expect(screen.getByText('Connection lost. Leave the table and rejoin from the dashboard.')).toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Reconnecting.')).not.toBeInTheDocument();
  });

  it('shows the reconnecting state while the hook is still spending that budget', () => {
    renderDsGameTable({ connected: false, connectionError: RETRYING_COPY });

    expect(screen.getByText('Reconnecting.')).toBeInTheDocument();
    expect(screen.getByText('Connection lost. Reconnecting.')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    expect(screen.queryByText('Disconnected.')).not.toBeInTheDocument();
  });

  it('shows neither while the socket is up', () => {
    renderDsGameTable({ connected: true, connectionError: null });

    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Reconnecting.')).not.toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Leave the table and rejoin from the dashboard.')).not.toBeInTheDocument();
  });
});
