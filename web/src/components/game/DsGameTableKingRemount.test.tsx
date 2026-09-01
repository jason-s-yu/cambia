// src/components/game/DsGameTableKingRemount.test.tsx
// cambia-1567: a client that mounts fresh into a King's second step (reload, new tab, device
// switch, not a live-socket resync) used to have no way to tell the look step from the swap/keep
// step. specialAction.firstStepDone (service/internal/game/sync_state.go) now carries that, and
// DsGameTable's kingConfirm reads it directly rather than only the locally-held kingPair the
// interactive look flow sets - kingPair is unset on a fresh mount, since the peeked pair is
// deliberately never re-delivered on remount (cambia-763 F1, cambia-1094).
//
// Held in its own file, same pattern as DsGameTableReconnect.test.tsx, so it does not collide with
// the general DsGameTable suite.
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { renderDsGameTable, resetGameTableStores } from '@/test/renderDsGameTable';
import { buildGameState, SELF_ID } from '@/test/fixtures/gameState';

afterEach(() => {
  resetGameTableStores();
});

describe('DsGameTable King second step from a fresh mount', () => {
  it('renders Swap/Keep controls from the snapshot alone, with no local kingPair', () => {
    const gameState = buildGameState({
      specialAction: { active: true, playerId: SELF_ID, cardRank: 'K', firstStepDone: true }
    });

    renderDsGameTable({
      gameState,
      gameStoreState: { pendingAction: 'special_action' }
    });

    expect(screen.getByTestId('action-king-swap')).toBeInTheDocument();
    expect(screen.getByTestId('action-king-keep')).toBeInTheDocument();
  });

  it('sends action_special swap_peek_swap when Swap is clicked with no local kingPair', async () => {
    const user = userEvent.setup();
    const gameState = buildGameState({
      specialAction: { active: true, playerId: SELF_ID, cardRank: 'K', firstStepDone: true }
    });

    const { sendMessage } = renderDsGameTable({
      gameState,
      gameStoreState: { pendingAction: 'special_action' }
    });

    await user.click(screen.getByTestId('action-king-swap'));

    expect(sendMessage).toHaveBeenCalledTimes(1);
    const sent = sendMessage.mock.calls[0][0];
    expect(sent.type).toBe('action_special');
    expect(sent.special).toBe('swap_peek_swap');
  });

  it('sends skip when Keep is clicked with no local kingPair', async () => {
    const user = userEvent.setup();
    const gameState = buildGameState({
      specialAction: { active: true, playerId: SELF_ID, cardRank: 'K', firstStepDone: true }
    });

    const { sendMessage } = renderDsGameTable({
      gameState,
      gameStoreState: { pendingAction: 'special_action' }
    });

    await user.click(screen.getByTestId('action-king-keep'));

    expect(sendMessage).toHaveBeenCalledTimes(1);
    const sent = sendMessage.mock.calls[0][0];
    expect(sent.type).toBe('action_special');
    expect(sent.special).toBe('skip');
  });

  it('does not render Swap/Keep before the look resolves (firstStepDone false/absent)', () => {
    const gameState = buildGameState({
      specialAction: { active: true, playerId: SELF_ID, cardRank: 'K' }
    });

    renderDsGameTable({
      gameState,
      gameStoreState: { pendingAction: 'special_action' }
    });

    expect(screen.queryByTestId('action-king-swap')).not.toBeInTheDocument();
    expect(screen.queryByTestId('action-king-keep')).not.toBeInTheDocument();
  });
});
