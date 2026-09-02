// src/components/game/DsGameTableCardA11y.test.tsx
// cambia-1242: the three state-dependent a11y regressions on the felt, each of which only shows up
// when the table changes state under a card that is already on screen.
//   - a card that stopped taking a click swapped its element type from <button> to <div>, which
//     React reconciles by unmounting the node, so a keyboard user's focus went to <body>;
//   - an own-hand pick survived a stock draw, keeping its gold lift on a card that had quietly
//     stopped reporting aria-pressed;
//   - the piles moved between role button and role img with the turn, so a control appeared and
//     disappeared under a screen reader instead of going unavailable.
// Plus the two forfeited-seat gaps carried onto this ticket from cambia-1237 (note cambia-1468):
// a forfeited seat's cards were neither dimmed nor named as forfeited, so a hand whose score had
// stopped counting read as an ordinary live one.
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

describe('DsGameTable own-hand pick that outlives its click', () => {
  it('keeps reporting a standing pick while the felt is busy, rather than lifting a silent card', async () => {
    const user = userEvent.setup();
    renderDsGameTable();

    const card = screen.getByTestId('card-0-2');
    await user.click(card);
    expect(card).toHaveAttribute('aria-pressed', 'true');
    expect(card.style.transform).toBe(LIFT);

    // An action goes in flight under the standing pick. Every own-card click is off until the
    // server answers, but nothing has touched the pick itself, so it stays lifted.
    act(() => {
      useGameStore.setState({ isProcessingAction: true });
    });

    const after = screen.getByTestId('card-0-2');
    expect(after).toBe(card);
    expect(after.style.transform).toBe(LIFT);
    // Still a control, so the lift still has a pressed state under it: unavailable rather than
    // demoted to an image, which is what used to strip aria-pressed off a card left lifted.
    expect(after).not.toHaveAttribute('role');
    expect(after).toHaveAttribute('aria-pressed', 'true');
    expect(after).toHaveAttribute('aria-disabled', 'true');
  });
});

describe('DsGameTable forfeited seat cards', () => {
  const forfeitedSelf = () => buildGameState({ currentPlayerId: OPP_ID, self: { forfeited: true } });

  it('dims a forfeited own hand, the way a hand held out of reach is dimmed', () => {
    renderDsGameTable({ gameState: forfeitedSelf() });
    expect(screen.getByTestId('card-0-0').style.opacity).toBe('0.7');
  });

  it('leaves a live own hand undimmed, so the dim keeps saying something', () => {
    renderDsGameTable();
    expect(screen.getByTestId('card-0-0').style.opacity).toBe('1');
  });

  it('names the forfeit on an own slot, since nothing else on the card says the score stopped counting', () => {
    renderDsGameTable({ gameState: forfeitedSelf() });
    expect(screen.getByTestId('card-0-0')).toHaveAttribute('aria-label', 'Your card 1, face down, seat forfeited, not scored');
  });

  it('names a forfeited seat across the table on the same rule', () => {
    renderDsGameTable({ gameState: buildGameState({ opponent: { forfeited: true } }) });
    expect(screen.getByTestId('card-1-0')).toHaveAttribute('aria-label', 'Rival card 1, face down, seat forfeited, not scored');
  });

  it('holds the dim and the name through round end, where the Cambia lock drops both', () => {
    renderDsGameTable({ gameState: forfeitedSelf(), phase: 'round_end' });

    const card = screen.getByTestId('card-0-0');
    // The lock dim means "out of reach until the round ends" and goes when it does; a forfeited
    // seat is still unscored afterwards, and its cards stay face down while the table turns over.
    expect(card.style.opacity).toBe('0.7');
    expect(card).toHaveAttribute('aria-label', 'Your card 1, face down, seat forfeited, not scored');
  });
});
