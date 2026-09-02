// src/stores/gameStore.gameStartedExemption.test.ts
// The client half of the game_started-before-sync exemption (cambia-1244).
//
// processGameWebSocketMessage drops every frame while gameState is null, because the store has
// nothing to apply one to; private_sync_state is what fills that field. game_started is the second
// exemption from that gate (cambia-958 D6), and it has to be: the hub emits game_started and only
// then calls BeginPreGame, which sends the round's first sync (hub.createAndStartGame, pinned
// server-side in internal/hub/game_started_order_test.go). So game_started always arrives against a
// null gameState on a session's first round, and against a deliberately nulled one on every round
// after, since its own case is what nulls it.
//
// Losing the exemption has no error and no dropped connection: the store logs a warning and keeps
// the previous round's terminal state. gameState stays non-null with gameOver still set, so
// LobbyPage keeps deriving roundOver from it and the finished round's scoreboard renders over the
// new deal until the next private_sync_state lands.
import { beforeEach, describe, expect, it } from 'vitest';
import { useGameStore } from '@/stores/gameStore';
import { buildGameState } from '@/test/fixtures/gameState';

/** The payload shape hub.Emit sends for game_started: a plain map, no GameEvent wrapper. */
const GAME_STARTED = { game_id: 'game-1', players: ['p1', 'p2'] };

describe('gameStore game_started before the first sync', () => {
  beforeEach(() => {
    useGameStore.setState({
      gameState: null,
      isLoading: false,
      finalScores: null,
      winnerId: null,
      pendingAction: null,
      pregamePeek: []
    });
  });

  it('applies game_started when no sync has landed yet', () => {
    // The session's first round: nothing has ever populated gameState.
    useGameStore.getState().processGameWebSocketMessage('game_started', GAME_STARTED);

    // The case body's own effect, which only runs if the frame was not dropped at the gate.
    expect(useGameStore.getState().isLoading).toBe(true);
    expect(useGameStore.getState().gameState).toBeNull();
  });

  it('clears the previous round from a table that has a state', () => {
    // The multi-round case: startNextRound re-invokes createAndStartGame while the store still
    // holds the finished round, and game_started is the only frame that clears it.
    useGameStore.setState({
      gameState: buildGameState({}),
      finalScores: { p1: 12, p2: 30 },
      winnerId: 'p1',
      isLoading: false
    });

    useGameStore.getState().processGameWebSocketMessage('game_started', GAME_STARTED);

    const state = useGameStore.getState();
    expect(state.gameState).toBeNull();
    expect(state.finalScores).toBeNull();
    expect(state.winnerId).toBeNull();
    expect(state.isLoading).toBe(true);
  });

  it('still drops a non-exempt frame that arrives before the first sync', () => {
    // The gate itself has to survive: exempting game_started must not exempt everything. A turn
    // announcement before any snapshot has no table to apply itself to.
    useGameStore.getState().processGameWebSocketMessage('game_player_turn', {
      payload: { currentPlayerId: 'p1', turn: 0 }
    });

    expect(useGameStore.getState().gameState).toBeNull();
    expect(useGameStore.getState().isLoading).toBe(false);
  });
});
