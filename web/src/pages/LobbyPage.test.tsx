// src/pages/LobbyPage.test.tsx
// Render-level coverage for the post-game exit (cambia-1238). "Back to lobby" used to flip the
// phase in the store and tell nobody, which left this client rendering an open lobby while the
// hub was still in post_game dropping every frame it sent; the button now asks the hub for the
// transition and waits for the phase_change that answers it.
//
// And for the leave flow (cambia-1520). A leave the server refused with 409 used to be swallowed
// and the page navigated to the dashboard anyway, so the player's seat stayed in the round until
// the disconnect grace forfeited it a minute later. The page now keeps them where they are and
// says why, and giving a live seat up is asked about first and sent as the player's own answer.
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import LobbyPage from '@/pages/LobbyPage';
import { LeaveRefusedError } from '@/services/lobbyService';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore } from '@/stores/gameStore';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { buildGameState, SELF_ID as SEAT_ID } from '@/test/fixtures/gameState';
import type { LobbyState } from '@/types';

const { sendMessage, closeSocket, reopenSocket, leaveLobbyApi } = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  closeSocket: vi.fn(),
  reopenSocket: vi.fn(),
  leaveLobbyApi: vi.fn()
}));

// The socket itself is not under test: the page's job is to hand the hub a frame and render what
// comes back, and both halves are driven here directly (sendMessage spy in, store update out).
vi.mock('@/hooks/useSocket', () => ({
  useSocket: () => ({ sendMessage, closeSocket, reopenSocket, isConnected: true, isLoading: false, error: null })
}));

// Only the one call is replaced: LeaveRefusedError has to stay the real class, since the page
// tells a refusal from every other failure with instanceof.
vi.mock('@/services/lobbyService', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/lobbyService')>();
  return { ...actual, leaveLobby: leaveLobbyApi };
});

/** Verbatim from handlers.leaveInProgressMessage; the page renders the server's own sentence. */
const REFUSAL = 'Cannot leave a lobby while its game is in progress';

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
      disconnectGraceSec: 60,
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

/**
 * Mounts LobbyPage on a live table, with a `/dashboard` route alongside it so a navigation the
 * page should not have made is visible as a rendered marker rather than as an absence.
 * `forfeited` puts the own seat out of the round, which is the seat cambia-1237 gave a leave
 * control to and the one the server lets go without a question. `phase`/`gameOver` put the
 * table between rounds, where the seat is nobody's to give up either.
 */
function renderLiveTable(
  { forfeited = false, gameOver = false, phase = 'in_game' }:
    { forfeited?: boolean; gameOver?: boolean; phase?: LobbyPhase } = {}
) {
  const gameState = buildGameState({ self: { forfeited }, gameOver });
  useAuthStore.setState({ isAuthenticated: true, initialised: true, isLoading: false, user: { id: SEAT_ID, username: 'You' } as never });
  useGameStore.setState({ gameState });
  useCurrentLobbyStore.setState({
    currentLobbyId: LOBBY_ID,
    lobbyDetails: lobbyDetails(),
    isConnected: true,
    isLoading: false,
    error: null,
    phase
  });
  return render(
    <MemoryRouter initialEntries={[`/lobby/${LOBBY_ID}`]}>
      <Routes>
        <Route path='/lobby/:lobbyId' element={<LobbyPage />} />
        <Route path='/dashboard' element={<div>Dashboard stand-in</div>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  // Default: the server accepts the leave. The tests that need a refusal say so themselves.
  leaveLobbyApi.mockReset();
  leaveLobbyApi.mockResolvedValue(undefined);
});

afterEach(() => {
  sendMessage.mockClear();
  closeSocket.mockClear();
  reopenSocket.mockClear();
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

describe('LobbyPage leave flow', () => {
  it('keeps a refused leave at the table and shows the server reason', async () => {
    // A seat that has already forfeited is the caller that leaves without being asked anything,
    // so this exercises the refusal on its own rather than through the confirmation. The 409 is
    // the race the guard exists for: the seat went live between the render and the click.
    leaveLobbyApi.mockRejectedValue(new LeaveRefusedError(REFUSAL));
    renderLiveTable({ forfeited: true });

    await userEvent.click(screen.getByTestId('action-leave-forfeited'));

    expect(leaveLobbyApi).toHaveBeenCalledWith(LOBBY_ID, { forfeit: false });
    expect(await screen.findByText(REFUSAL)).toBeInTheDocument();
    // Still at the table, not on the dashboard, and the connection the leave closed is back.
    expect(screen.queryByText('Dashboard stand-in')).not.toBeInTheDocument();
    expect(screen.getByTestId('seat-self')).toBeInTheDocument();
    expect(reopenSocket).toHaveBeenCalledTimes(1);
    expect(useCurrentLobbyStore.getState().currentLobbyId).toBe(LOBBY_ID);
  });

  it('asks before giving a live seat up, and sends the forfeit as the answer', async () => {
    renderLiveTable();

    await userEvent.click(screen.getByTestId('action-leave'));

    // Nothing has been sent yet: the question is the point.
    expect(leaveLobbyApi).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog', { name: 'Leave the table?' })).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('leave-forfeit'));

    expect(leaveLobbyApi).toHaveBeenCalledWith(LOBBY_ID, { forfeit: true });
    expect(await screen.findByText('Dashboard stand-in')).toBeInTheDocument();
  });

  it('leaves the seat where it is when the question is declined', async () => {
    renderLiveTable();

    await userEvent.click(screen.getByTestId('action-leave'));
    await userEvent.click(screen.getByTestId('leave-cancel'));

    expect(leaveLobbyApi).not.toHaveBeenCalled();
    expect(closeSocket).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByText('Dashboard stand-in')).not.toBeInTheDocument();
    expect(screen.getByTestId('seat-self')).toBeInTheDocument();
  });

  it('does not ask a forfeited seat, which has nothing left to give up', async () => {
    renderLiveTable({ forfeited: true });

    await userEvent.click(screen.getByTestId('action-leave-forfeited'));

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(leaveLobbyApi).toHaveBeenCalledWith(LOBBY_ID, { forfeit: false });
    expect(await screen.findByText('Dashboard stand-in')).toBeInTheDocument();
  });

  it('does not ask between rounds, where the seat is already nobody\'s to give up', async () => {
    // The table is still mounted at round_end and keeps its leave control, but game_end has
    // flipped gameOver and the server has cleared the lobby's in-game flag with it. Confirming a
    // forfeit here would claim a game is running next to a scoreboard saying the round is over.
    renderLiveTable({ phase: 'round_end', gameOver: true });

    await userEvent.click(screen.getByTestId('action-leave'));

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(leaveLobbyApi).toHaveBeenCalledWith(LOBBY_ID, { forfeit: false });
    expect(await screen.findByText('Dashboard stand-in')).toBeInTheDocument();
  });
});
