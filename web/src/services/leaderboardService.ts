// src/services/leaderboardService.ts
import api from '@/lib/axios';
import type { AxiosError } from 'axios';
import type { ApiErrorResponse } from '@/types';

/** A single ranked leaderboard row as returned by the server. */
export interface LeaderboardRow {
	rank: number;
	userId: string;
	username: string;
	rating: number;
	rd: number;
	games: number;
	/** Peak rating. Omitted by the server when not tracked/available for the pool. */
	peak?: number;
}

export interface LeaderboardResponse {
	pool: string;
	rows: LeaderboardRow[];
	you: LeaderboardRow | null;
}

/**
 * Fetches the ranked leaderboard for a rating pool (e.g. `h2h`, `ffa4`).
 * @throws {Error} If the API request fails (including 404 when the endpoint
 * is not yet deployed); callers should catch and surface an empty/error state.
 */
export const fetchLeaderboard = async (pool: string, limit = 50): Promise<LeaderboardResponse> => {
	try {
		const response = await api.get<LeaderboardResponse>('/leaderboard', { params: { pool, limit } });
		return response.data;
	} catch (error) {
		const err = error as AxiosError<ApiErrorResponse>;
		console.error('Fetch Leaderboard API call failed:', err.response?.data || err.message);
		throw error; // Re-throw for store/UI error handling
	}
};
