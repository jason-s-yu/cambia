// src/pages/LobbyPage.test.tsx
// Render-level coverage for the post-game exit (cambia-1238). "Back to lobby" used to flip the
// phase in the store and tell nobody, which left this client rendering an open lobby while the
// hub was still in post_game dropping every frame it sent; the button now asks the hub for the
// transition and waits for the phase_change that answers it.
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import LobbyPage from '@/pages/LobbyPage';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore } from '@/stores/gameStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import type { LobbyState } from '@/types';

const { sendMessage, closeSocket } = vi.hoisted(() => ({ sendMessage: vi.fn(), closeSocket: vi.fn() }));

// The socket itself is not under test: the page's job is to hand the hub a frame and render what
// comes back, and both halves are driven here directly (sendMessage spy in, store update out).
vi.mock('@/hooks/useSocket', () => ({
  useSocket: () => ({ sendMessage, closeSocket, isConnected: true, isLoading: false, error: null })
}));

const LOBBY_ID = '11111111-1111-4111-8111-111111111111';
const SELF_ID = '22222222-2222-4222-8222-222222222222';
const OPP_ID = '33333333-3333-4333-8333-333333333333';

const authInitial = useAuthStore.getState();
const gameInitial = useGameStore.getState();
const lobbyInitial = useCurrentLobbyStore.getState();

function lobbyDetails(): LobbyState {
  return {
    id: LOBBY_ID,
    lobby_id: LOBBY_ID,
    hostUserID: SELF_ID,
    host_id: SELF_ID,
    your_id: SELF_ID,
    your_is_host: true,
    system_host: false,
    type: 'private',
    gameMode: 'head_to_head',
    inGame: false,
    game_id: null,
    presetId: '',
    rulesLocked: false,
    houseRules: {
      allowDrawFromDiscardPile: false,
      allowReplaceAbilities: false,
      allowOpponentSnapping: false,
      snapRace: false,
      lockCallerHand: false,
      forfeitOnDisconnect: true,
      disconnectGraceSec: 90,
      penaltyDrawCount: 2,
      turnTimerSec: 15,
      maxGameTurns: 0,
      cardsPerPlayer: 4,
      cambiaAllowedRound: 0,
      numJokers: 2,
      numDecks: 1,
      initialViewCount: 2
    },
    circuit: {
      enabled: false,
      mode: '',
      rules: { targetScore: 100, winBonus: 0, falseCambiaPenalty: 0, freezeUserOnDisconnect: false }
    },
    lobbySettings: { autoStart: true },
    settings: { autoStart: true },
    lobby_status: {
      users: [
        { id: SELF_ID, username: 'Self', is_host: true, is_ready: false },
        { id: OPP_ID, username: 'Rival', is_host: false, is_ready: false }
      ]
    }
  } as unknown as LobbyState;
}

/** Mounts LobbyPage on its own route with the stores seeded to a finished casual game. */
function renderResultsScreen() {
  useAuthStore.setState({ isAuthenticated: true, initialised: true, isLoading: false, user: { id: SELF_ID, username: 'Self' } as never });
  useCurrentLobbyStore.setState({
    currentLobbyId: LOBBY_ID,
    lobbyDetails: lobbyDetails(),
    isConnected: true,
    isLoading: false,
    error: null,
    phase: 'post_game'
  });
  return render(
    <MemoryRouter initialEntries={[`/lobby/${LOBBY_ID}`]}>
      <Routes>
        <Route path='/lobby/:lobbyId' element={<LobbyPage />} />
      </Routes>
    </MemoryRouter>
  );
}

afterEach(() => {
  sendMessage.mockClear();
  closeSocket.mockClear();
  useAuthStore.setState(authInitial, true);
  useGameStore.setState(gameInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
});

describe('LobbyPage post-game exit', () => {
  it('asks the hub to reopen the lobby and flips no phase of its own', async () => {
    renderResultsScreen();
    expect(screen.getByText('Game over')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Back to lobby' }));

    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith({ type: 'return_to_lobby' });

    // The server has not answered yet, so the results are still what the player is looking at.
    expect(useCurrentLobbyStore.getState().phase).toBe('post_game');
    expect(screen.getByText('Game over')).toBeInTheDocument();
  });

  it('leaves the results when the hub answers with phase_change', async () => {
    renderResultsScreen();
    await userEvent.click(screen.getByRole('button', { name: 'Back to lobby' }));

    act(() => {
      useCurrentLobbyStore.getState().processLobbyWebSocketMessage('phase_change', { phase: 'open' });
    });

    expect(useCurrentLobbyStore.getState().phase).toBe('open');
    expect(screen.queryByText('Game over')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ready up' })).toBeInTheDocument();
  });
});
