// src/stores/historyStore.ts
import { create } from 'zustand';
import type { AxiosError } from 'axios';
import type { ApiErrorResponse } from '@/types';
import {
	fetchGameHistory,
	fetchRatingSummary,
	type HistoryGame,
	type RatingSummary
} from '@/services/historyService';

/** Games fetched per page, and per "load more" press. */
export const HISTORY_PAGE_SIZE = 10;

interface HistoryState {
	games: HistoryGame[];
	/** Total games the caller has; games.length < total means more are loadable. */
	total: number;
	gamesLoading: boolean;
	gamesError: string | null;
	/** True once a games fetch has settled, so an empty list can be told from an unloaded one. */
	gamesLoaded: boolean;

	ratings: RatingSummary | null;
	ratingsLoading: boolean;
	ratingsError: string | null;
	ratingsLoaded: boolean;

	/** Loads the first page, replacing anything already held. */
	fetchGames: (limit?: number) => Promise<void>;
	/** Appends the next page after what is already held. */
	fetchMoreGames: (limit?: number) => Promise<void>;
	fetchRatings: () => Promise<void>;
	/** Drops everything, for a logout or an account switch. */
	reset: () => void;
}

const errorMessage = (err: unknown, fallback: string): string => {
	const axErr = err as AxiosError<ApiErrorResponse>;
	if (axErr?.response?.status === 404) return 'Match history is not available yet.';
	if (axErr?.response?.data?.message) return axErr.response.data.message;
	if (err instanceof Error) return err.message;
	return fallback;
};

/**
 * Profile history store: the caller's own past games (GET /user/history,
 * paginated) and rating summary (GET /user/ratings). Both endpoints are scoped
 * server-side to the authenticated session, so this store never holds another
 * player's data and takes no user id.
 *
 * Games and ratings track separate loading/error state: one failing must not
 * blank the other on the profile page.
 */
export const useHistoryStore = create<HistoryState>((set, get) => ({
	games: [],
	total: 0,
	gamesLoading: false,
	gamesError: null,
	gamesLoaded: false,

	ratings: null,
	ratingsLoading: false,
	ratingsError: null,
	ratingsLoaded: false,

	fetchGames: async (limit = HISTORY_PAGE_SIZE) => {
		set({ gamesLoading: true, gamesError: null });
		try {
			const data = await fetchGameHistory(limit, 0);
			set({
				games: data.games ?? [],
				total: data.total ?? 0,
				gamesLoading: false,
				gamesError: null,
				gamesLoaded: true
			});
		} catch (err: unknown) {
			set({
				gamesLoading: false,
				gamesError: errorMessage(err, 'Failed to load match history.'),
				gamesLoaded: true
			});
		}
	},

	fetchMoreGames: async (limit = HISTORY_PAGE_SIZE) => {
		const { games, gamesLoading } = get();
		if (gamesLoading) return;
		set({ gamesLoading: true, gamesError: null });
		try {
			const data = await fetchGameHistory(limit, games.length);
			// Re-read rather than closing over `games`: the page was in flight, and a
			// concurrent fetchGames() may have replaced the list underneath it. De-dupe on
			// game id so a game that shifted across the page boundary is never listed twice.
			const existing = get().games;
			const seen = new Set(existing.map((g) => g.gameId));
			const merged = existing.concat((data.games ?? []).filter((g) => !seen.has(g.gameId)));
			set({
				games: merged,
				total: data.total ?? get().total,
				gamesLoading: false,
				gamesError: null,
				gamesLoaded: true
			});
		} catch (err: unknown) {
			set({
				gamesLoading: false,
				gamesError: errorMessage(err, 'Failed to load more games.')
			});
		}
	},

	fetchRatings: async () => {
		set({ ratingsLoading: true, ratingsError: null });
		try {
			const data = await fetchRatingSummary();
			set({ ratings: data, ratingsLoading: false, ratingsError: null, ratingsLoaded: true });
		} catch (err: unknown) {
			set({
				ratingsLoading: false,
				ratingsError: errorMessage(err, 'Failed to load ratings.'),
				ratingsLoaded: true
			});
		}
	},

	reset: () =>
		set({
			games: [],
			total: 0,
			gamesLoading: false,
			gamesError: null,
			gamesLoaded: false,
			ratings: null,
			ratingsLoading: false,
			ratingsError: null,
			ratingsLoaded: false
		})
}));
