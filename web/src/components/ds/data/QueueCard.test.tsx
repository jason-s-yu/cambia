// src/components/ds/data/QueueCard.test.tsx
// Render-level coverage for the round-count claim (cambia-1518). The card used to default to 8
// rounds and always render it, advertising a match length that nothing plays: a matchmade lobby
// runs no circuit (Circuit.Enabled only turns on through a host update_rules edit) and plays
// exactly one game regardless of the queue's round count. `rounds` is now omitted rather than
// shown wrong; the card renders no round claim unless a caller has one actually backed.
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import QueueCard from './QueueCard';

describe('QueueCard round claim', () => {
  it('renders no round count when none is given', () => {
    render(<QueueCard name='H2H Rapid' players={2} minutes={3} />);

    expect(screen.queryByText(/round/i)).not.toBeInTheDocument();
  });

  it('still renders players and the minutes estimate', () => {
    render(<QueueCard name='H2H Rapid' players={2} minutes={3} />);

    expect(screen.getByText('2p')).toBeInTheDocument();
    expect(screen.getByText('~3 min')).toBeInTheDocument();
  });

  it('renders a round count a caller explicitly backs', () => {
    render(<QueueCard name='Some circuit' players={2} rounds={4} minutes={10} />);

    expect(screen.getByText('4 rounds')).toBeInTheDocument();
  });

  it('singularizes a one-round claim', () => {
    render(<QueueCard name='H2H Quick' players={2} rounds={1} minutes={3} />);

    expect(screen.getByText('1 round')).toBeInTheDocument();
    expect(screen.queryByText('1 rounds')).not.toBeInTheDocument();
  });
});
