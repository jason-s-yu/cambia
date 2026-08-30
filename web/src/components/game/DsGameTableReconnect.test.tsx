// src/components/game/DsGameTableReconnect.test.tsx
// The felt half of cambia-1236: what the table shows once useSocket has spent its retry budget.
// The hook's own coverage (src/hooks/useSocket.test.tsx) proves the cap holds and pins the copy it
// writes; these assert the table turns that copy into the disconnected state a player can act on,
// and that an ordinary retry is still shown as a retry. Held in its own file so the ticket's tests
// do not collide with the general DsGameTable suite.
//
// The status line ("Disconnected." / "Reconnecting.") and the hint line are queried by their
// unique, period-suffixed strings. The bare word is not unique: the felt chip and the own seat's
// PlayerSeat both render it once the own seat is passed `gaveUp` (cambia-1473), so the two
// give-up assertions below query the bare word deliberately at both surfaces, scoped to the own
// seat's container (its `data-testid='seat-self'`, cambia-1473) to tell it apart from the chip
// and from the "You (you)" the standings row also renders elsewhere on the felt.
import { screen, within } from '@testing-library/react';
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
  it('shows the stopped-trying state once the hook reports a spent retry budget, on both the chip and the own seat', () => {
    renderDsGameTable({ connected: false, connectionError: GAVE_UP_COPY });

    // Two deliberate sources for "Disconnected": the felt chip and, since cambia-1473, the own
    // seat's PlayerSeat label. Asserted as a pair so a regression collapsing either source back
    // to "Reconnecting" is caught, not just a regression that removes one of the two.
    const disconnectedNodes = screen.getAllByText('Disconnected');
    expect(disconnectedNodes).toHaveLength(2);
    const seat = screen.getByTestId('seat-self');
    expect(disconnectedNodes.some((n) => seat.contains(n))).toBe(true);
    expect(disconnectedNodes.some((n) => !seat.contains(n))).toBe(true);
    expect(within(seat).getByText('Disconnected')).toBeInTheDocument();

    expect(screen.getByText('Disconnected.')).toBeInTheDocument();
    expect(screen.getByText('Connection lost. Leave the table and rejoin from the dashboard.')).toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Reconnecting.')).not.toBeInTheDocument();
    expect(within(seat).queryByText('Reconnecting')).not.toBeInTheDocument();
  });

  it('shows the reconnecting state while the hook is still spending that budget, on both the chip and the own seat', () => {
    renderDsGameTable({ connected: false, connectionError: RETRYING_COPY });

    // Same pairing as the give-up case: the chip and the own seat both still read "Reconnecting"
    // while the retry budget is not spent.
    const reconnectingNodes = screen.getAllByText('Reconnecting');
    expect(reconnectingNodes).toHaveLength(2);
    const seat = screen.getByTestId('seat-self');
    expect(reconnectingNodes.some((n) => seat.contains(n))).toBe(true);
    expect(reconnectingNodes.some((n) => !seat.contains(n))).toBe(true);
    expect(within(seat).getByText('Reconnecting')).toBeInTheDocument();

    expect(screen.getByText('Reconnecting.')).toBeInTheDocument();
    expect(screen.getByText('Connection lost. Reconnecting.')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    expect(screen.queryByText('Disconnected.')).not.toBeInTheDocument();
    expect(within(seat).queryByText('Disconnected')).not.toBeInTheDocument();
  });

  it('shows neither while the socket is up', () => {
    renderDsGameTable({ connected: true, connectionError: null });

    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Reconnecting.')).not.toBeInTheDocument();
    expect(screen.queryByText('Connection lost. Leave the table and rejoin from the dashboard.')).not.toBeInTheDocument();
  });
});
