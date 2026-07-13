// src/stores/leaderboardStore.ts
import { create } from 'zustand';
import type { AxiosError } from 'axios';
import type { ApiErrorResponse } from '@/types';
import { fetchLeaderboard, type LeaderboardRow } from '@/services/leaderboardService';

// Pool identifiers match the service's rating pools (validLeaderboardPools in
// service/internal/handlers/leaderboard.go): the elo_*/phi_* column pairs.
export type LeaderboardPool = '1v1' | '4p' | '7p8p';

interface LeaderboardPoolState {
	rows: LeaderboardRow[];
	you: LeaderboardRow | null;
	isLoading: boolean;
	error: string | null;
	loaded: boolean;
}

interface LeaderboardState {
	pools: Record<LeaderboardPool, LeaderboardPoolState>;
	fetchPool: (pool: LeaderboardPool, limit?: number) => Promise<void>;
}

const emptyPoolState = (): LeaderboardPoolState => ({
	rows: [],
	you: null,
	isLoading: false,
	error: null,
	loaded: false
});

/**
 * Leaderboard store: per-pool ranked rows fetched from GET /leaderboard.
 * Each pool tracks its own loading/error state so switching tabs never
 * bleeds one pool's failure into another's display.
 */
export const useLeaderboardStore = create<LeaderboardState>((set) => ({
	pools: {
		'1v1': emptyPoolState(),
		'4p': emptyPoolState(),
		'7p8p': emptyPoolState()
	},

	fetchPool: async (pool, limit = 50) => {
		set((state) => ({
			pools: {
				...state.pools,
				[pool]: { ...state.pools[pool], isLoading: true, error: null }
			}
		}));

		try {
			const data = await fetchLeaderboard(pool, limit);
			set((state) => ({
				pools: {
					...state.pools,
					[pool]: {
						rows: data.rows ?? [],
						you: data.you ?? null,
						isLoading: false,
						error: null,
						loaded: true
					}
				}
			}));
		} catch (err: unknown) {
			const axErr = err as AxiosError<ApiErrorResponse>;
			let message = 'Failed to load leaderboard.';
			if (axErr?.response?.status === 404) {
				message = 'Leaderboard is not available yet.';
			} else if (axErr?.response?.data?.message) {
				message = axErr.response.data.message;
			} else if (err instanceof Error) {
				message = err.message;
			}
			set((state) => ({
				pools: {
					...state.pools,
					[pool]: { ...state.pools[pool], isLoading: false, error: message, loaded: true }
				}
			}));
		}
	}
}));
