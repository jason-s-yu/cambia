// src/stores/queueStore.ts
import { create } from 'zustand';
import { getQueues, startSearch as apiStartSearch, cancelSearch as apiCancelSearch, type QueueInfo } from '@/services/matchmakingService';
import { leaveLobby as apiLeaveLobby } from '@/services/lobbyService';
import { useCurrentLobbyStore } from './lobbyStore';

/**
 * Gives up the throwaway lobby a search ran from and drops it as the current lobby, which also
 * closes the socket the dashboard holds open on it. Never throws: the lobby is a side effect of
 * searching, and a failed release is the server's idle reaper's problem, not the player's.
 */
async function releaseSearchLobby(lobbyId: string | null): Promise<void> {
  if (!lobbyId) return;
  try {
    await apiLeaveLobby(lobbyId);
  } catch (err) {
    console.error('Failed to release the search lobby:', err);
  }
  if (useCurrentLobbyStore.getState().currentLobbyId === lobbyId) {
    useCurrentLobbyStore.getState().leaveLobby();
  }
}

interface QueueState {
  queues: QueueInfo[];
  isLoading: boolean;
  error: string | null;

  // Join-search flow (ported from pages/PlayPage.tsx).
  searchingQueueId: string | null;
  searchLobbyId: string | null;
  searchStartTime: number;

  fetchQueues: () => Promise<void>;
  joinQueue: (queue: QueueInfo) => Promise<void>;
  cancelSearch: () => Promise<void>;
  /** Ends the search because the matchmaker resolved it, given the lobby the match is played in. */
  finishSearch: (matchedLobbyId: string) => Promise<void>;
  clearError: () => void;
}

/** Matchmaking queue store for the design-system home screen (cambia-483). */
export const useQueueStore = create<QueueState>((set, get) => ({
  queues: [],
  isLoading: false,
  error: null,

  searchingQueueId: null,
  searchLobbyId: null,
  searchStartTime: 0,

  fetchQueues: async () => {
    set({ isLoading: true, error: null });
    try {
      const queues = await getQueues();
      set({ queues, isLoading: false });
    } catch (err) {
      console.error('Failed to fetch matchmaking queues:', err);
      set({ error: 'Failed to load queues.', isLoading: false });
    }
  },

  joinQueue: async (queue) => {
    if (get().searchingQueueId) return;
    try {
      const createAndJoinLobby = useCurrentLobbyStore.getState().createAndJoinLobby;
      // The queue is what defines a matchmaking lobby: the service derives the game mode and
      // ranked-ness from the queue config and reads the same config again when the lobby enters
      // the queue. Sending the queue id as gameMode (what this did before cambia-933) 400'd on
      // the service's game-mode validation and left the lobby with no queue to search in.
      const lobbyId = await createAndJoinLobby({
        type: 'matchmaking',
        queueID: queue.queueId,
      });
      if (!lobbyId) return;
      set({
        searchingQueueId: queue.queueId,
        searchLobbyId: lobbyId,
        searchStartTime: Date.now(),
      });
      await apiStartSearch(lobbyId);
    } catch (err) {
      console.error('Failed to start matchmaking search:', err);
      // The lobby was created before the search failed, so release it rather than leave a
      // lobby nobody is in queued behind a search that never started.
      await releaseSearchLobby(get().searchLobbyId);
      set({ searchingQueueId: null, searchLobbyId: null, searchStartTime: 0, error: 'Could not start the search.' });
    }
  },

  cancelSearch: async () => {
    const lobbyId = get().searchLobbyId;
    if (!lobbyId) return;
    try {
      await apiCancelSearch(lobbyId);
    } catch (err) {
      console.error('Failed to cancel matchmaking search:', err);
    } finally {
      await releaseSearchLobby(lobbyId);
      set({ searchingQueueId: null, searchLobbyId: null, searchStartTime: 0 });
    }
  },

  finishSearch: async (matchedLobbyId) => {
    const lobbyId = get().searchLobbyId;
    set({ searchingQueueId: null, searchLobbyId: null, searchStartTime: 0 });
    // A match is played in one lobby, and the party that did not host it is moving out of the
    // one it searched from: release that one so it is not left behind as a resumable session
    // (cambia-933).
    if (lobbyId && lobbyId !== matchedLobbyId) {
      await releaseSearchLobby(lobbyId);
    }
  },

  clearError: () => set({ error: null }),
}));
