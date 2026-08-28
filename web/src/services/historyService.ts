// src/services/historyService.ts
import api from '@/lib/axios';
import type { AxiosError } from 'axios';
import type { ApiErrorResponse } from '@/types';

/** One other seat at the table in a past game. */
export interface HistoryOpponent {
	userId: string;
	username: string;
	/** Final score. Null when the server never recorded one for that seat. */
	score: number | null;
	didWin: boolean | null;
	/** Placement. Currently always null: nothing writes game_results.ranking yet. */
	ranking: number | null;
}

/** The rating change a rated game produced for the caller. */
export interface HistoryRatingChange {
	pool: string;
	old: number;
	new: number;
	delta: number;
}

/** A single past game from the caller's point of view. */
export interface HistoryGame {
	gameId: string;
	/** ISO timestamp of when the game finished. */
	playedAt: string;
	status: string;
	roundIndex: number;
	/** Lobby type the game was played in: private, public or matchmaking. */
	lobbyType: string;
	/** Lobby mode: ranked or casual. Empty when the lobby row carried none. */
	mode: string;
	rated: boolean;
	playerCount: number;
	score: number | null;
	didWin: boolean | null;
	ranking: number | null;
	/** Null for an unrated game. */
	rating: HistoryRatingChange | null;
	opponents: HistoryOpponent[];
}

export interface HistoryResponse {
	games: HistoryGame[];
	limit: number;
	offset: number;
	/** Total games the caller has, independent of this page. */
	total: number;
}

/** The caller's standing in one Glicko-2 rating pool. */
export interface PoolRating {
	pool: string;
	rating: number;
	/** Glicko-2 rating deviation. */
	rd: number;
	volatility: number;
	/** Rated games recorded in this pool. */
	games: number;
	wins: number;
	peak: number;
}

export interface RatingSummary {
	pools: PoolRating[];
	openSkill: { mu: number; sigma: number };
	/** Lifetime totals across every recorded game, rated or not. */
	record: { games: number; wins: number };
}

/**
 * Fetches a page of the authenticated caller's own past games, newest first.
 * The server scopes the result to the caller's session; there is no user
 * parameter and no way to read another player's history.
 * @throws {Error} If the API request fails; callers surface an error state.
 */
export const fetchGameHistory = async (limit = 20, offset = 0): Promise<HistoryResponse> => {
	try {
		const response = await api.get<HistoryResponse>('/user/history', { params: { limit, offset } });
		return response.data;
	} catch (error) {
		const err = error as AxiosError<ApiErrorResponse>;
		console.error('Fetch Game History API call failed:', err.response?.data || err.message);
		throw error;
	}
};

/**
 * Fetches the authenticated caller's rating summary: every Glicko-2 pool, the
 * OpenSkill pair, and lifetime win/loss totals.
 * @throws {Error} If the API request fails; callers surface an error state.
 */
export const fetchRatingSummary = async (): Promise<RatingSummary> => {
	try {
		const response = await api.get<RatingSummary>('/user/ratings');
		return response.data;
	} catch (error) {
		const err = error as AxiosError<ApiErrorResponse>;
		console.error('Fetch Rating Summary API call failed:', err.response?.data || err.message);
		throw error;
	}
};
