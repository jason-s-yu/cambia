// src/test/renderDsGameTable.tsx
// Render-level harness for DsGameTable (cambia-1172): mounts the real component tree against a
// fixture ObfGameState.
//
// DsGameTable takes `gameState` as a prop for almost everything it draws (hand, piles, house
// rules, notice deltas), but a handful of derived reads - `isMyTurn` chief among them, via
// useGameStore(selectIsSelfTurn) - come from a SEPARATE copy of the game state living in
// useGameStore, not the prop. The two are the same store in the running app (gameStore's own
// socket handler writes both), but nothing enforces that here, so this harness always seeds
// useGameStore's `gameState` with the identical object passed as the prop. Skipping that step is
// the harness's own footgun: the felt would render the fixture's hand and piles correctly while
// `isMyTurn` silently reads a stale (null) game state and every turn-gated control goes dead.
//
// useAuthStore/useGameStore/useCurrentLobbyStore are bare zustand create() singletons, not React
// context, so seeding is setState before render and resetting is setState back to the pristine
// snapshot captured at import time - there is no provider to wrap the tree in.
import { render } from '@testing-library/react';
import type { RenderResult } from '@testing-library/react';
import { vi } from 'vitest';
import DsGameTable from '@/components/game/DsGameTable';
import type { ObfGameState } from '@/types/game';
import type { LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore } from '@/stores/gameStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { buildGameState, SELF_ID } from './fixtures/gameState';

const authInitial = useAuthStore.getState();
const gameInitial = useGameStore.getState();
const lobbyInitial = useCurrentLobbyStore.getState();

/** Restores every store DsGameTable reads to its pristine state. Call from afterEach. */
export function resetGameTableStores(): void {
  useAuthStore.setState(authInitial, true);
  useGameStore.setState(gameInitial, true);
  useCurrentLobbyStore.setState(lobbyInitial, true);
}

export interface RenderDsGameTableOptions {
  /** Defaults to `buildGameState()` (SELF_ID to act, turn 3, live discard pile). */
  gameState?: ObfGameState;
  phase?: LobbyPhase;
  connected?: boolean;
  connectionError?: string | null;
  /** The signed-in user's id; defaults to the fixture's own seat (SELF_ID). */
  selfId?: string;
  /** Patches applied to useGameStore after `gameState` is seeded, e.g. `{ pendingAction: 'discard_replace' }`. */
  gameStoreState?: Partial<ReturnType<typeof useGameStore.getState>>;
}

export interface RenderDsGameTableResult extends RenderResult {
  sendMessage: ReturnType<typeof vi.fn>;
  onLeave: ReturnType<typeof vi.fn>;
  gameState: ObfGameState;
}

/** Mounts DsGameTable against a fixture game state, with every store it reads seeded to match. */
export function renderDsGameTable(options: RenderDsGameTableOptions = {}): RenderDsGameTableResult {
  const gameState = options.gameState ?? buildGameState();
  const selfId = options.selfId ?? SELF_ID;

  useAuthStore.setState({ user: { id: selfId, username: 'You', is_ephemeral: false } });
  useGameStore.setState({ gameState });
  if (options.gameStoreState) useGameStore.setState(options.gameStoreState);

  const sendMessage = vi.fn();
  const onLeave = vi.fn();

  const result = render(
    <DsGameTable
      gameState={gameState}
      phase={options.phase ?? 'in_game'}
      sendMessage={sendMessage}
      onLeave={onLeave}
      connected={options.connected ?? true}
      connectionError={options.connectionError ?? null}
    />
  );

  return { ...result, sendMessage, onLeave, gameState };
}
