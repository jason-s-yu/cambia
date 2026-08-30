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
