/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/axios';
import type { ActiveSession, LobbyState, LobbyListEntry } from '@/types';

/**
 * Creates a new lobby via the backend API.
 * @param settings Initial lobby settings (type, gameMode, houseRules, etc.). Can be a partial object.
 * @returns A promise resolving to the full state of the newly created lobby, or null on failure before throwing.
 * @throws {Error} If the API request fails.
 */
export const createLobby = async (settings: Partial<LobbyState>): Promise<LobbyState | null> => {
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

/**
 * Releases the signed-in user's membership of a lobby. This is the deliberate leave: closing
 * the tab or dropping the WebSocket keeps membership so the session stays resumable, and only
 * this call gives it up. The server tears the lobby down once its last member leaves.
 *
 * Close the lobby WebSocket before calling this: connecting joins the lobby server-side, so a
 * socket left open to auto-reconnect would hand the membership straight back.
 * @throws {Error} If the API request fails (404 when the lobby is already gone, 409 while its
 * game is in progress, which the caller may ignore and navigate away regardless).
 */
export const leaveLobby = async (lobbyId: string): Promise<void> => {
	try {
		await api.post(`/lobby/${lobbyId}/leave`);
	} catch (error: any) {
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

export const listLobbies = async (): Promise<Record<string, LobbyListEntry>> => {
	try {
		const response = await api.get<Record<string, LobbyListEntry>>('/lobby/list');
		return response.data || {}; // Return empty object if data is null/undefined to prevent errors
	} catch (error: any) {
		console.error('List Lobbies API call failed:', error.response?.data || error.message, error);
		throw error; // Re-throw for the store/component to handle UI feedback
	}
};