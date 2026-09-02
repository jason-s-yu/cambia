// src/components/game/DsGameTableCardA11y.test.tsx
// cambia-1242: the three state-dependent a11y regressions on the felt, each of which only shows up
// when the table changes state under a card that is already on screen.
//   - a card that stopped taking a click swapped its element type from <button> to <div>, which
//     React reconciles by unmounting the node, so a keyboard user's focus went to <body>;
//   - an own-hand pick survived a stock draw, keeping its gold lift on a card that had quietly
//     stopped reporting aria-pressed;
//   - the piles moved between role button and role img with the turn, so a control appeared and
//     disappeared under a screen reader instead of going unavailable.
// Held in its own file, the same way DsGameTableKingRemount.test.tsx is, so it does not collide
// with the general DsGameTable suite.
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { useGameStore } from '@/stores/gameStore';
import { buildGameState, OPP_ID } from '@/test/fixtures/gameState';
import { renderDsGameTable, resetGameTableStores } from '@/test/renderDsGameTable';

/** The lift PlayingCard puts on a selected card; the visual half of the aria-pressed pairing. */
const LIFT = 'translateY(-6px)';

afterEach(() => {
  resetGameTableStores();
});

describe('DsGameTable card focus across a clickability change', () => {
  it('keeps focus on a card that goes inert under it, rather than dropping it to the body', () => {
    renderDsGameTable();

    // A live own card is a snap pick, so it is a real control: no role override, in the tab
    // sequence, focusable.
    const card = screen.getByTestId('card-0-1');
    expect(card).not.toHaveAttribute('role');
    card.focus();
    expect(document.activeElement).toBe(card);

    // An action goes in flight, which locks the felt until the server answers: the card takes no
    // click for as long as it lasts.
    act(() => {
      useGameStore.setState({ isProcessingAction: true });
    });

    const after = screen.getByTestId('card-0-1');
    expect(after).toBe(card);
    // Not a control any more, and out of the tab sequence, but the same node, so it still holds
    // the focus the player put on it.
    expect(after).toHaveAttribute('role', 'img');
    expect(after).toHaveAttribute('tabindex', '-1');
    expect(document.activeElement).toBe(after);
    expect(document.activeElement).not.toBe(document.body);
  });
});

describe('DsGameTable own-hand pick through a stock draw', () => {
  it('pairs aria-pressed with the selected lift while the pick stands', async () => {
    const user = userEvent.setup();
    renderDsGameTable();

    const card = screen.getByTestId('card-0-2');
    expect(card).toHaveAttribute('aria-pressed', 'false');
    expect(card.style.transform).toBe('none');

    await user.click(card);

    expect(card).toHaveAttribute('aria-pressed', 'true');
    expect(card.style.transform).toBe(LIFT);
  });

  it('clears the pick when the draw lands, leaving no card lifted without a pressed state', async () => {
    const user = userEvent.setup();
    renderDsGameTable();

    const card = screen.getByTestId('card-0-2');
    await user.click(card);
    expect(card).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('action-snap')).toBeInTheDocument();

    // What the store does once the server answers a draw (gameStore action_draw_stockpile ->
    // pendingAction 'discard_replace'). The own hand stays clickable, since a click now commits
    // the replace, so the card is still a control - but the pick meant "snap this card", and the
    // draw took the snap away with it.
    act(() => {
      useGameStore.setState({ pendingAction: 'discard_replace' });
    });

    const after = screen.getByTestId('card-0-2');
    expect(after).toBe(card);
    expect(after).not.toHaveAttribute('aria-pressed');
    expect(screen.queryByTestId('action-snap')).not.toBeInTheDocument();
    // Nothing in the hand is left wearing the lift, which is the state the dropped aria-pressed
    // used to contradict.
    for (const slot of [0, 1, 2, 3]) {
      expect(screen.getByTestId(`card-0-${slot}`).style.transform).toBe('none');
    }
  });
});

describe('DsGameTable pile roles across turn state', () => {
  it('draws both piles as buttons on the caller own turn', () => {
    renderDsGameTable();

    expect(screen.getByRole('button', { name: /^Stockpile/ })).toBe(screen.getByTestId('pile-stock'));
    expect(screen.getByRole('button', { name: /^Discard pile/ })).toBe(screen.getByTestId('pile-discard'));
    expect(screen.getByTestId('pile-stock')).not.toHaveAttribute('aria-disabled');
    expect(screen.getByTestId('pile-discard')).not.toHaveAttribute('aria-disabled');
  });

  it('keeps them buttons while another seat acts, marked unavailable rather than taken away', async () => {
    const user = userEvent.setup();
    const { sendMessage } = renderDsGameTable({ gameState: buildGameState({ currentPlayerId: OPP_ID }) });

    const stock = screen.getByTestId('pile-stock');
    const discard = screen.getByTestId('pile-discard');

    expect(screen.getByRole('button', { name: /^Stockpile/ })).toBe(stock);
    expect(screen.getByRole('button', { name: /^Discard pile/ })).toBe(discard);
    expect(stock).toHaveAttribute('aria-disabled', 'true');
    expect(discard).toHaveAttribute('aria-disabled', 'true');
    // Still a tab stop, so the tab order does not shuffle every time the turn passes, and still
    // inert: clicking an unavailable pile sends nothing.
    expect(stock).not.toHaveAttribute('tabindex');
    await user.click(stock);
    await user.click(discard);
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('holds the same pile node and role when the turn passes', () => {
    renderDsGameTable();

    const stock = screen.getByTestId('pile-stock');
    expect(stock).not.toHaveAttribute('aria-disabled');

    // isMyTurn comes from the store copy of the game state alone (gameStore selectIsSelfTurn), and
    // it is the only thing the piles read off whose turn it is: their counts and their top card
    // come from the prop, which the turn passing does not touch. So moving the turn in the store
    // is the whole of the change under test here.
    act(() => {
      useGameStore.setState({ gameState: buildGameState({ currentPlayerId: OPP_ID }) });
    });

    const after = screen.getByTestId('pile-stock');
    expect(after).toBe(stock);
    expect(after).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('button', { name: /^Stockpile/ })).toBe(after);
  });
});
