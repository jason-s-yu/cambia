// src/components/lobby/DsResultsView.test.tsx
// Render-level coverage for the post-game "Back to lobby" gate (cambia-1516). The hub only
// admits return_to_lobby from the host, widened to any seated player where the host role
// belongs to the system (Hub.mayReturnToLobby, cambia-1238); before this fix every seat saw
// the button and a non-host click came back as an error envelope. A seat the hub would refuse
// now sees a waiting label instead, naming the results timer only where post_game actually
// arms one (match_end arms none, see service/doc/lobby_actions.md "Post-game exit").
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import DsResultsView from './DsResultsView';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import type { LobbyState } from '@/types';

const SELF_ID = '22222222-2222-4222-8222-222222222222';
const OPP_ID = '33333333-3333-4333-8333-333333333333';

const authInitial = useAuthStore.getState();
const lobbyInitial = useCurrentLobbyStore.getState();

function seedLobby(overrides: Partial<LobbyState>) {
  useAuthStore.setState({ isAuthenticated: true, initialised: true, isLoading: false, user: { id: SELF_ID, username: 'Self' } as never });
  useCurrentLobbyStore.setState({
    lobbyDetails: {
      lobby_status: {
        users: [
          { id: SELF_ID, username: 'Self', is_host: false, is_ready: false },
          { id: OPP_ID, username: 'Rival', is_host: false, is_ready: false }
        ]
      },
      ...overrides
    } as unknown as LobbyState
  });
}

afterEach(() => {
  useAuthStore.setState(authInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
});

describe('DsResultsView post-game exit gate', () => {
  it('shows "Back to lobby" to the host', () => {
    seedLobby({ your_is_host: true, system_host: false });
    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.getByRole('button', { name: 'Back to lobby' })).toBeInTheDocument();
    expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument();
  });

  it('shows "Back to lobby" to any seat of a system-hosted lobby', () => {
    seedLobby({ your_is_host: false, system_host: true });
    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.getByRole('button', { name: 'Back to lobby' })).toBeInTheDocument();
    expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument();
  });

  it('hides "Back to lobby" from a non-host seat and names the results timer in post_game', () => {
    seedLobby({ your_is_host: false, system_host: false });
    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.queryByRole('button', { name: 'Back to lobby' })).not.toBeInTheDocument();
    expect(screen.getByText(/results timer/i)).toBeInTheDocument();
    // Leave lobby stays available regardless of the gate.
    expect(screen.getByRole('button', { name: 'Leave lobby' })).toBeInTheDocument();
  });

  it('hides "Back to lobby" from a non-host seat and claims no timer in match_end', () => {
    seedLobby({ your_is_host: false, system_host: false });
    render(<DsResultsView phase='match_end' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.queryByRole('button', { name: 'Back to lobby' })).not.toBeInTheDocument();
    expect(screen.getByText('Waiting for the host to return to the lobby.')).toBeInTheDocument();
    expect(screen.queryByText(/timer/i)).not.toBeInTheDocument();
  });
});
