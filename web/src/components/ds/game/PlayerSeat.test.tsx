// src/components/ds/game/PlayerSeat.test.tsx
// Render-level coverage for the disconnected-seat label (cambia-1473): the `disconnected` state
// reads "Reconnecting" while a caller is still retrying and "Disconnected" once `gaveUp` is set,
// so the seat agrees with the felt's own connection chip (which already makes that distinction;
// see DsGameTableReconnect.test.tsx).
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import PlayerSeat from './PlayerSeat';

describe('PlayerSeat disconnected label', () => {
  it('reads "Reconnecting" while still retrying', () => {
    render(<PlayerSeat username='Alice' state='disconnected' />);

    expect(screen.getByText('Reconnecting')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
  });

  it('reads "Reconnecting" for the own seat too while still retrying', () => {
    render(<PlayerSeat username='Alice' isYou state='disconnected' />);

    expect(screen.getByText('Reconnecting')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
  });

  it('reads "Disconnected" once the caller has given up', () => {
    render(<PlayerSeat username='Alice' state='disconnected' gaveUp />);

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
    expect(screen.queryByText('Reconnecting')).not.toBeInTheDocument();
  });

  it('reads "Disconnected" for the own seat once the caller has given up', () => {
    render(<PlayerSeat username='Alice' isYou state='disconnected' gaveUp />);

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
    expect(screen.queryByText('Reconnecting')).not.toBeInTheDocument();
  });

  it('ignores gaveUp outside the disconnected state', () => {
    render(<PlayerSeat username='Alice' state='forfeited' gaveUp />);

    expect(screen.getByText('Forfeited')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
  });

  it('lets an explicit note override both labels', () => {
    render(<PlayerSeat username='Alice' state='disconnected' gaveUp note='Choosing a target' />);

    expect(screen.getByText('Choosing a target')).toBeInTheDocument();
    expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    expect(screen.queryByText('Reconnecting')).not.toBeInTheDocument();
  });
});
