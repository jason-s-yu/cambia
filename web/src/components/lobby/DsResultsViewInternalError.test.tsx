// src/components/lobby/DsResultsViewInternalError.test.tsx
// The results screen for a game the service ended on an internal error (cambia-1831). The panic
// guard aborts a game through the same end path an ordinary one takes, so the scores that ride the
// results frames are read off whatever state the fault left behind. They arrive anyway - the
// service records them as evidence - and the `reason` field is what says they are not a result.
// This pins that the view reports the error and draws no scoreboard from them, on both frames: the
// game_end a client at the table receives, and the game_results a reload into the results gets on
// its own.
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import DsResultsView from './DsResultsView';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useGameStore } from '@/stores/gameStore';
import type { LobbyState } from '@/types';

const SELF_ID = '22222222-2222-4222-8222-222222222222';
const OPP_ID = '33333333-3333-4333-8333-333333333333';

const authInitial = useAuthStore.getState();
const lobbyInitial = useCurrentLobbyStore.getState();
const gameInitial = useGameStore.getState();

function seedLobby() {
  useAuthStore.setState({ isAuthenticated: true, initialised: true, isLoading: false, user: { id: SELF_ID, username: 'Self' } as never });
  useCurrentLobbyStore.setState({
    lobbyDetails: {
      your_is_host: true,
      system_host: false,
      lobby_status: {
        users: [
          { id: SELF_ID, username: 'Self', is_host: true, is_ready: false },
          { id: OPP_ID, username: 'Rival', is_host: false, is_ready: false }
        ]
      }
    } as unknown as LobbyState
  });
}

function syncFinishedTable() {
  useGameStore.getState().processGameWebSocketMessage('private_sync_state', {
    state: {
      gameId: 'g1',
      preGameActive: false,
      started: false,
      gameOver: true,
      currentPlayerId: null,
      turnId: 9,
      stockpileSize: 0,
      discardSize: 0,
      players: [
        { playerId: SELF_ID, username: 'Self', handSize: 2, hasCalledCambia: false, connected: true, isCurrentTurn: false },
        { playerId: OPP_ID, username: 'Rival', handSize: 2, hasCalledCambia: false, connected: true, isCurrentTurn: false }
      ],
      cambiaCalled: false,
      houseRules: {}
    }
  });
}

/** Pushes a results frame through the real store, the way useSocket does. */
function deliver(type: 'game_end' | 'game_results', payload: unknown) {
  syncFinishedTable();
  useGameStore.getState().processGameWebSocketMessage(type, payload);
}

afterEach(() => {
  useAuthStore.setState(authInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
  useGameStore.setState(gameInitial, true);
});

describe('DsResultsView on an internal error', () => {
  it('reports the error and shows no scoreboard when game_end names it', () => {
    seedLobby();
    // game_end nests its payload one level deeper, the GameEvent wrapper convention.
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, reason: 'internal_error' } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.getByTestId('results-internal-error')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Game ended by an error' })).toBeInTheDocument();
    // No scoreboard: not the section, not the scores it would have drawn, and no winner claimed.
    expect(screen.queryByText('Scores')).not.toBeInTheDocument();
    expect(screen.queryByText('9')).not.toBeInTheDocument();
    expect(screen.queryByText('1')).not.toBeInTheDocument();
    expect(screen.queryByText('Winner')).not.toBeInTheDocument();
    expect(screen.queryByText(/wins\./)).not.toBeInTheDocument();
  });

  it('reports the error from game_results alone, which is all a reload into the results gets', () => {
    seedLobby();
    // The service drops the game from its store right after emitting game_results, so the hub's
    // held copy of that frame is the only one a returning player ever sees (cambia-1241).
    deliver('game_results', { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, reason: 'internal_error' });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.getByTestId('results-internal-error')).toBeInTheDocument();
    expect(screen.queryByText('Scores')).not.toBeInTheDocument();
    expect(screen.queryByText('9')).not.toBeInTheDocument();
  });

  it('leaves the seat its way out of the finished table', () => {
    seedLobby();
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, reason: 'internal_error' } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    // The error suppresses the result, not the controls: the table is over either way and the
    // host still has to be able to reopen the lobby.
    expect(screen.getByRole('button', { name: 'Back to lobby' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Leave lobby' })).toBeInTheDocument();
  });

  it('shows the ordinary scoreboard when the results carry no reason', () => {
    seedLobby();
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.queryByTestId('results-internal-error')).not.toBeInTheDocument();
    expect(screen.getByText('Scores')).toBeInTheDocument();
    expect(screen.getAllByText('9').length).toBeGreaterThan(0);
  });

  it('suppresses the scoreboard for a reason it has no copy for', () => {
    seedLobby();
    // A newer service naming an ending this client does not know still must not have its scores
    // read as a result: the field being there at all is what decides (gameStore readEndReason).
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, reason: 'some_future_abort' } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.queryByText('Scores')).not.toBeInTheDocument();
    expect(screen.queryByText('9')).not.toBeInTheDocument();
  });
});
