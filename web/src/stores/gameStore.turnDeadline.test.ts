// src/stores/gameStore.turnDeadline.test.ts
// The client half of cambia-1556: game_turn_deadline moves the countdown TimerBar reads
// (gameState.turnDeadline plus serverClockOffsetMs, wired in DsGameTable) and touches nothing else.
// The frame exists because the server re-arms the turn clock mid-turn - an ability prompt, a King's
// second step, a reconnect onto an unclocked turn - and none of those may announce a turn.
import { beforeEach, describe, expect, it } from 'vitest';
import { useGameStore } from '@/stores/gameStore';
import { buildGameState, SELF_ID, OPP_ID } from '@/test/fixtures/gameState';

/** Seeds the store with a table already counting down, the state a mid-turn re-arm lands on. */
function seedCountingDownTable(turnDeadline: number) {
  useGameStore.setState({
    gameState: buildGameState({ turnId: 7, turnDeadline }),
    serverClockOffsetMs: 0
  });
}

describe('gameStore game_turn_deadline', () => {
  beforeEach(() => {
    useGameStore.setState({ gameState: null, serverClockOffsetMs: 0 });
  });

  it('applies the re-armed deadline and the send time it came with', () => {
    const original = Date.now() + 2_000;
    seedCountingDownTable(original);

    const serverNow = Date.now() + 4_000; // a client clock running 4s slow
    const moved = serverNow + 15_000;
    useGameStore.getState().processGameWebSocketMessage('game_turn_deadline', {
      payload: { turn: 8, turnDeadline: moved, serverNow }
    });

    const state = useGameStore.getState();
    expect(state.gameState?.turnDeadline).toBe(moved);
    expect(state.gameState?.turnId).toBe(8);
    // TimerBar counts (deadline + offset - Date.now()), so the offset has to come off this frame
    // rather than the last snapshot, or the corrected countdown is wrong by the skew.
    expect(state.serverClockOffsetMs).toBeGreaterThan(3_000);
  });

  it('leaves the pending ability prompt and the acting seat alone', () => {
    useGameStore.setState({
      gameState: buildGameState({
        currentPlayerId: SELF_ID,
        turnDeadline: Date.now() + 1_000,
        specialAction: { active: true, playerId: SELF_ID, cardRank: '7' }
      }),
      pendingAction: 'special_action'
    });

    useGameStore.getState().processGameWebSocketMessage('game_turn_deadline', {
      payload: { turn: 7, turnDeadline: Date.now() + 15_000, serverNow: Date.now() }
    });

    const state = useGameStore.getState();
    // Most re-arms happen precisely to give a player time to answer a prompt, so clearing it here
    // would wipe the thing the frame was sent to protect. That is game_player_turn's job.
    expect(state.gameState?.specialAction?.active).toBe(true);
    expect(state.pendingAction).toBe('special_action');
    expect(state.gameState?.currentPlayerId).toBe(SELF_ID);
    expect(state.gameState?.players.find((p) => p.playerId === OPP_ID)?.isCurrentTurn).toBe(false);
  });

  it('clears the countdown when the frame carries no deadline', () => {
    seedCountingDownTable(Date.now() + 5_000);

    useGameStore.getState().processGameWebSocketMessage('game_turn_deadline', {
      payload: { turn: 7, serverNow: Date.now() }
    });

    // Absent turnDeadline means no clock is armed, the same reading game_player_turn and the
    // sync_state snapshot give it; TimerBar falls back to its informational render.
    expect(useGameStore.getState().gameState?.turnDeadline).toBeNull();
  });
});
