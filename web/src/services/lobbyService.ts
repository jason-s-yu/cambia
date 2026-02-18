/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/axios';
import type { LobbyState } from '@/types';

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
export const listLobbies = async (): Promise<Record<string, LobbyState>> => {
	try {
		const response = await api.get<Record<string, LobbyState>>('/lobby/list');
		return response.data || {}; // Return empty object if data is null/undefined to prevent errors
	} catch (error: any) {
		console.error('List Lobbies API call failed:', error.response?.data || error.message, error);
		throw error; // Re-throw for the store/component to handle UI feedback
	}
};