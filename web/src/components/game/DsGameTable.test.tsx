// src/components/game/DsGameTable.test.tsx
// Render-level regression coverage for DsGameTable (cambia-1172): own-hand slot placement, the
// hint line, the single aria-live notice slot, and one user-event interaction. This establishes the
// harness other web tickets test against, not a sweep of the table's full interaction surface.
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { useGameStore } from '@/stores/gameStore';
import { buildGameState, OPP_ID } from '@/test/fixtures/gameState';
import { renderDsGameTable, resetGameTableStores } from '@/test/renderDsGameTable';

afterEach(() => {
  resetGameTableStores();
});

describe('DsGameTable own-hand slot placement', () => {
  it('draws slots 0 and 1 on the bottom row, nearest the owner, with later slots stacked above (cambia-1095)', () => {
    renderDsGameTable();

    const slot0 = screen.getByTestId('card-0-0');
    const slot1 = screen.getByTestId('card-0-1');
    const slot2 = screen.getByTestId('card-0-2');
    const slot3 = screen.getByTestId('card-0-3');

    // A four-card hand takes two grid rows; row 2 is the bottom row, nearest the owner, and is
    // where the engine's pregame peek pair (slots 0 and 1) belongs (handLayout.ts ownHandPlacement).
    expect(slot0.style.gridRow).toBe('2');
    expect(slot1.style.gridRow).toBe('2');
    expect(slot2.style.gridRow).toBe('1');
    expect(slot3.style.gridRow).toBe('1');

    expect(slot0.style.gridColumn).toBe('1');
    expect(slot1.style.gridColumn).toBe('2');
    expect(slot2.style.gridColumn).toBe('1');
    expect(slot3.style.gridColumn).toBe('2');
  });
});

describe('DsGameTable hint line', () => {
  it('tells the caller they can draw or take the discard on their own turn', () => {
    renderDsGameTable();
    expect(screen.getByText('Your turn. Draw from the stock or take the discard.')).toBeInTheDocument();
  });

  it('names the current player while another seat is to act', () => {
    const gameState = buildGameState({ currentPlayerId: OPP_ID, opponent: { username: 'Rival' } });
    renderDsGameTable({ gameState });
    expect(screen.getByText('Waiting for Rival.')).toBeInTheDocument();
  });
});

describe('DsGameTable notice line', () => {
  it('renders exactly one aria-live region and fills it from a dropped-action notice', () => {
    const { container } = renderDsGameTable();

    const liveRegions = container.querySelectorAll('[aria-live="polite"]');
    expect(liveRegions).toHaveLength(1);
    expect(liveRegions[0]).toBeEmptyDOMElement();

    // Bumping the store's droppedActionNonce is what the socket layer does when it discards a
    // stale outbound action (cambia-891); the table's own notice effect turns that into copy.
    act(() => {
      useGameStore.setState({ droppedActionNonce: 1 });
    });

    expect(screen.getByText('That did not go through.')).toBeInTheDocument();
    expect(liveRegions[0]).toHaveTextContent('That did not go through.');
  });
});

describe('DsGameTable interactions', () => {
  it('emits action_draw_stockpile when the caller clicks the stockpile on their own turn', async () => {
    const user = userEvent.setup();
    const { sendMessage } = renderDsGameTable();

    await user.click(screen.getByTestId('pile-stock'));

    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith({ type: 'action_draw_stockpile' });
  });
});

describe('DsGameTable forfeited own seat', () => {
  // The seat whose reconnect window closed (cambia-955): the server drops it from scoring and
  // plays its turns on the clock, so isMyTurn never comes back. Before cambia-1237 the hint line
  // sat on the whose-turn copy for the rest of the round with every control gone, and the only way
  // off the table was the ghost button at the bottom of the side panel.
  const forfeitedSelf = () =>
    buildGameState({
      currentPlayerId: OPP_ID,
      self: { forfeited: true }
    });

  it('names the forfeit instead of the player who is to act (cambia-1237)', () => {
    renderDsGameTable({ gameState: forfeitedSelf() });

    expect(
      screen.getByText('You forfeited. Your score is not counted. You can keep watching, or leave the table.')
    ).toBeInTheDocument();
    expect(screen.queryByText('Waiting for Rival.')).not.toBeInTheDocument();
  });

  it('keeps the forfeit line once the round is over, since that is why the score is missing', () => {
    renderDsGameTable({ gameState: forfeitedSelf(), phase: 'round_end' });
    expect(screen.getByText('You forfeited. Your score is not counted in this round.')).toBeInTheDocument();
  });

  it('offers an exit in the action area, not only in the side panel', async () => {
    const user = userEvent.setup();
    const { onLeave } = renderDsGameTable({ gameState: forfeitedSelf() });

    const exit = screen.getByTestId('action-leave-forfeited');
    await user.click(exit);
    expect(onLeave).toHaveBeenCalledTimes(1);
  });

  it('leaves the round watchable: the piles, the opponent hand and the standings still render', () => {
    renderDsGameTable({ gameState: forfeitedSelf() });

    expect(screen.getByTestId('pile-stock')).toBeInTheDocument();
    expect(screen.getByTestId('pile-discard')).toBeInTheDocument();
    expect(screen.getByTestId('card-1-0')).toBeInTheDocument();
    expect(screen.getByTestId('card-0-0')).toBeInTheDocument();
  });

  it('takes the snap surface off the felt, so no live control contradicts the forfeit line', () => {
    const { sendMessage } = renderDsGameTable({ gameState: forfeitedSelf() });

    // A live seat's own cards are snap picks and an opponent's are snappable out of turn. A
    // forfeited seat is out of the round, so neither is offered: PlayingCard renders a <button>
    // only where it takes a click, and with no pick there is no Snap button to reach.
    expect(screen.getByTestId('card-0-0').tagName).not.toBe('BUTTON');
    expect(screen.getByTestId('card-1-0').tagName).not.toBe('BUTTON');
    expect(screen.queryByTestId('action-snap')).not.toBeInTheDocument();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('shows no forfeit exit for a live seat', () => {
    renderDsGameTable();
    expect(screen.queryByTestId('action-leave-forfeited')).not.toBeInTheDocument();
  });
});
