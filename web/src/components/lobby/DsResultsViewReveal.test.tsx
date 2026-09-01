// src/components/lobby/DsResultsViewReveal.test.tsx
// The round-end reveal on the results screen (RULES.md 3C, cambia-1542). The service reveals
// every scored seat's hand when a round ends and carries it on game_end and on the game_results a
// reload into the results is answered with; the results view draws those faces beside the score
// each hand produced. Before this the view rendered scores alone and the table under it was
// entirely face-down, so a player could not see what they had lost to.
import { render, screen, within } from '@testing-library/react';
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

// The finalHands list exactly as the service emits it (service/internal/game/game.go
// buildFinalReveal): a card id, its slot, and its face.
const FINAL_HANDS = [
  {
    playerId: SELF_ID,
    cards: [
      { id: 'c1', idx: 0, rank: 'K', suit: 'H', value: -1 },
      { id: 'c2', idx: 1, rank: 'T', suit: 'S', value: 10 }
    ]
  },
  {
    playerId: OPP_ID,
    cards: [
      { id: 'c3', idx: 0, rank: 'A', suit: 'D', value: 1 },
      { id: 'c4', idx: 1, rank: 'O', suit: 'R', value: 0 }
    ]
  }
];

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

/**
 * Seeds the store with a finished table's snapshot. The store drops every frame that arrives
 * before one (see processGameWebSocketMessage), and the server order is the same: a reconnect is
 * sent private_sync_state and then the terminal results frame.
 */
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

describe('DsResultsView round-end reveal', () => {
  it('renders every seat\'s revealed hand from the game_end event', () => {
    seedLobby();
    // game_end nests its results under payload.payload, the GameEvent wrapper convention.
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, finalHands: FINAL_HANDS } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    const own = screen.getByTestId(`final-hand-${SELF_ID}`);
    expect(within(own).getByRole('img', { name: 'king of hearts' })).toBeInTheDocument();
    expect(within(own).getByRole('img', { name: '10 of spades' })).toBeInTheDocument();

    const rival = screen.getByTestId(`final-hand-${OPP_ID}`);
    expect(within(rival).getByRole('img', { name: 'ace of diamonds' })).toBeInTheDocument();
    expect(within(rival).getByRole('img', { name: 'joker' })).toBeInTheDocument();

    // The scores the hands sit beside are still rendered ('9' twice: the standings row and the
    // big "Your score" figure above it).
    expect(screen.getAllByText('9').length).toBeGreaterThan(0);
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
  });

  it('renders the reveal from game_results alone, which is all a reload into post_game gets', () => {
    seedLobby();
    // A client that reloads after the round ended never sees game_end: the service drops the game
    // from its store right after emitting game_results, so the hub's held copy of that frame is
    // the only carrier of the reveal it will ever receive (cambia-955, cambia-1542).
    deliver('game_results', { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID, finalHands: FINAL_HANDS });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(within(screen.getByTestId(`final-hand-${SELF_ID}`)).getByRole('img', { name: 'king of hearts' })).toBeInTheDocument();
    expect(within(screen.getByTestId(`final-hand-${OPP_ID}`)).getByRole('img', { name: 'joker' })).toBeInTheDocument();
  });

  it('draws no hand for a seat the reveal leaves out', () => {
    seedLobby();
    // A forfeited seat is not scored and so is not revealed (service buildFinalReveal).
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9 }, winner: SELF_ID, finalHands: [FINAL_HANDS[0]] } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.getByTestId(`final-hand-${SELF_ID}`)).toBeInTheDocument();
    expect(screen.queryByTestId(`final-hand-${OPP_ID}`)).not.toBeInTheDocument();
  });

  it('renders scores alone when the results carry no reveal', () => {
    seedLobby();
    deliver('game_end', { payload: { scores: { [SELF_ID]: 9, [OPP_ID]: 1 }, winner: OPP_ID } });

    render(<DsResultsView phase='post_game' onReturnToLobby={() => {}} onLeave={() => {}} />);

    expect(screen.queryByTestId(`final-hand-${SELF_ID}`)).not.toBeInTheDocument();
    expect(screen.getAllByText('9').length).toBeGreaterThan(0);
  });
});
