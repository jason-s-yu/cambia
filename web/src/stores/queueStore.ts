// src/stores/queueStore.ts
import { create } from 'zustand';
import { getQueues, startSearch as apiStartSearch, cancelSearch as apiCancelSearch, type QueueInfo } from '@/services/matchmakingService';
import { useCurrentLobbyStore } from './lobbyStore';

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
      const lobbyId = await createAndJoinLobby({
        type: 'matchmaking',
        gameMode: queue.queueId,
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
      set({ searchingQueueId: null, searchLobbyId: null, searchStartTime: 0 });
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
      set({ searchingQueueId: null, searchLobbyId: null, searchStartTime: 0 });
    }
  },

  clearError: () => set({ error: null }),
}));
