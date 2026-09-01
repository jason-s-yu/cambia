/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/axios';
import type { ActiveSession, LobbyState, LobbyListEntry, LobbyPreset } from '@/types';

/**
 * Body of POST /lobby/create: lobby fields, plus the optional id of a ruleset preset the
 * service expands into house rules, lobby settings and the game mode the preset fixes
 * (cambia-1088). `presetId` is request-only, which is why it is not a LobbyState field: the
 * created lobby comes back carrying the expanded values, not the id it was built from.
 */
export type CreateLobbyRequest = Partial<LobbyState> & { presetId?: string };

/**
 * Creates a new lobby via the backend API.
 * @param settings Initial lobby settings (type, gameMode, presetId, houseRules, etc.). Can be a partial object.
 * @returns A promise resolving to the full state of the newly created lobby, or null on failure before throwing.
 * @throws {Error} If the API request fails.
 */
export const createLobby = async (settings: CreateLobbyRequest): Promise<LobbyState | null> => {
	try {
		const response = await api.post<LobbyState>('/lobby/create', settings);
		return response.data;
	} catch (error: any) {
		console.error('Create Lobby API call failed:', error.response?.data || error.message, error);
		throw error; // Re-throw for the store/component to handle UI feedback
	}
};

/**
 * Fetches a map of all currently active lobbies from the backend.
 * Primarily intended for dashboard display or debugging.
 * @returns A promise resolving to a map where keys are lobby UUIDs and values are lobby state objects.
 * @throws {Error} If the API request fails.
 */
export const joinLobby = async (lobbyId: string): Promise<void> => {
	try {
		await api.post(`/lobby/${lobbyId}/join`);
	} catch (error: any) {
		console.error('Join Lobby API call failed:', error.response?.data || error.message, error);
		throw error;
	}
};

/** The default sentence for a refused leave, for a 409 that arrives without a body to quote. */
const LEAVE_REFUSED_FALLBACK = 'This lobby has a game in progress.';

/**
 * A leave the server refused (409). Its own error type because the caller has to tell it from
 * every other failure: a refusal means the membership is still held and the player is still at
 * the table, so navigating away on it is what left the seat to forfeit on the grace timer
 * (cambia-1520). `reason` is the server's own sentence, meant to be shown to the player.
 */
export class LeaveRefusedError extends Error {
	readonly reason: string;

	constructor(reason: string) {
		super(reason);
		this.name = 'LeaveRefusedError';
		this.reason = reason;
	}
}

/** Pulls the refusal sentence out of an axios error. http.Error writes a bare text body. */
function refusalReason(error: any): string {
	const data = error?.response?.data;
	const text = typeof data === 'string' ? data.trim() : '';
	return text || LEAVE_REFUSED_FALLBACK;
}

/**
 * Releases the signed-in user's membership of a lobby. This is the deliberate leave: closing
 * the tab or dropping the WebSocket keeps membership so the session stays resumable, and only
 * this call gives it up. The server tears the lobby down once its last member leaves.
 *
 * Close the lobby WebSocket before calling this: connecting joins the lobby server-side, so a
 * socket left open to auto-reconnect would hand the membership straight back.
 *
 * `forfeit` is the player's consent to give up a live seat, and the server only releases one
 * when it is set. It is sent after a confirmation, never on a plain Leave, since it decides a
 * round rather than a screen (cambia-1520).
 * @throws {LeaveRefusedError} 409: the caller holds a live seat and did not agree to forfeit it.
 * @throws {Error} Any other API failure (404 when the lobby is already gone, which the caller
 * may ignore and navigate away regardless).
 */
export const leaveLobby = async (lobbyId: string, opts?: { forfeit?: boolean }): Promise<void> => {
	try {
		await api.post(`/lobby/${lobbyId}/leave`, { forfeit: opts?.forfeit === true });
	} catch (error: any) {
		if (error?.response?.status === 409) {
			throw new LeaveRefusedError(refusalReason(error));
		}
		console.error('Leave Lobby API call failed:', error.response?.data || error.message, error);
		throw error;
	}
};

/**
 * Fetches the lobby or in-progress game the signed-in user can rejoin, for the home screen's
 * resume affordance. Resolves to null when there is nothing to resume, which is the common
 * case and is not an error.
 * @throws {Error} If the API request fails.
 */
export const getActiveSession = async (): Promise<ActiveSession | null> => {
	try {
		const response = await api.get<{ active: ActiveSession | null }>('/lobby/active');
		return response.data?.active ?? null;
	} catch (error: any) {
		console.error('Active Session API call failed:', error.response?.data || error.message, error);
		throw error;
	}
};

/**
 * Fetches the selectable rulesets (GET /lobby/presets): the default first, then one per
 * matchmaking queue. The service owns these values - the dialog and the lobby rule sheet both
 * fill themselves from this list so they cannot disagree about what a preset name means.
 * @throws {Error} If the API request fails.
 */
export const getLobbyPresets = async (): Promise<LobbyPreset[]> => {
	try {
		const response = await api.get<LobbyPreset[]>('/lobby/presets');
		return response.data || [];
	} catch (error: any) {
		console.error('Lobby Presets API call failed:', error.response?.data || error.message, error);
		throw error;
	}
};

export const listLobbies = async (): Promise<Record<string, LobbyListEntry>> => {
	try {
		const response = await api.get<Record<string, LobbyListEntry>>('/lobby/list');
		return response.data || {}; // Return empty object if data is null/undefined to prevent errors
	} catch (error: any) {
		console.error('List Lobbies API call failed:', error.response?.data || error.message, error);
		throw error; // Re-throw for the store/component to handle UI feedback
	}
};