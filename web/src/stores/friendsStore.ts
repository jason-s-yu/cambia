// src/stores/friendsStore.ts
import { create } from 'zustand';
import { getFriends, type FriendEntry } from '@/services/friendsService';

interface FriendsState {
  friends: FriendEntry[];
  isLoading: boolean;
  error: string | null;
  fetchFriends: () => Promise<void>;
  clearError: () => void;
}

/** Friends list store for the design-system home screen (cambia-483). */
export const useFriendsStore = create<FriendsState>((set) => ({
  friends: [],
  isLoading: false,
  error: null,

  fetchFriends: async () => {
    set({ isLoading: true, error: null });
    try {
      const friends = await getFriends();
      set({ friends, isLoading: false });
    } catch (err: unknown) {
      console.error('Failed to fetch friends list:', err);
      let errorMessage = 'Failed to load friends.';
      if (typeof err === 'object' && err !== null && 'response' in err) {
        const response = (err as { response?: { data?: { message?: string }; statusText?: string } }).response;
        if (response?.data?.message) {
          errorMessage = response.data.message;
        } else if (response?.statusText) {
          errorMessage = response.statusText;
        }
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      set({ error: errorMessage, isLoading: false });
    }
  },

  clearError: () => set({ error: null }),
}));
