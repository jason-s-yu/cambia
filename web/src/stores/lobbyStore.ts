/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/lobbyStore.ts
import { create } from 'zustand';
import type { LobbyState, ChatMessage, CircuitSettings, User, LobbyUser } from '@/types/index';
import { listLobbies, createLobby as apiCreateLobby } from '@/services/lobbyService';
import { useAuthStore } from './authStore';
import { NIL as NIL_UUID } from 'uuid';

/** Represents lobby information displayed in the dashboard list */
interface LobbyListEntry {
	lobby: LobbyState;
	playerCount: number;
	maxPlayers: number;
}

/** State for managing the list of available lobbies */
interface LobbyListState {
	lobbies: Record<string, LobbyListEntry>;
	isLoading: boolean;
	error: string | null;
	fetchLobbies: () => Promise<void>;
	clearError: () => void;
	removeLobbyFromList: (lobbyId: string) => void;
}

/** State for managing the currently joined lobby */
interface CurrentLobbyState {
	currentLobbyId: string | null;
	lobbyDetails: LobbyState | null;
	chatMessages: ChatMessage[];
	isConnected: boolean; // WebSocket connection status
	isLoading: boolean;   // Loading state for API calls or initial WS connection/sync
	error: string | null;
	// Countdown state
	countdownStartTime: number | null; // System timestamp (ms) when countdown started
	countdownDuration: number | null; // Duration in seconds
	setCurrentLobbyId: (lobbyId: string | null) => void; // Sets the target lobby ID, affecting connection status
	processLobbyWebSocketMessage: (message: any) => void; // Handles incoming WS messages
	addChatMessage: (message: ChatMessage) => void; // Adds a message to the chat
	setConnected: (status: boolean) => void; // Updates WS connection status
	setLoading: (loading: boolean) => void; // Sets loading indicator state
	setError: (error: string | null) => void; // Sets error message
	clearError: () => void; // Clears the error message
	clearChat: () => void; // Clears chat messages
	createAndJoinLobby: (settings: Partial<LobbyState>) => Promise<string | null>; // Creates a lobby via API and sets it as current
	leaveLobby: () => void; // Clears the current lobby state (client-side only)
}

// --- Lobby List Store Implementation ---
export const useLobbyListStore = create<LobbyListState>((set) => ({
	lobbies: {},
	isLoading: false,
	error: null,
	fetchLobbies: async () => {
		set({ isLoading: true, error: null });
		try {
			const lobbyMap = await listLobbies();
			set({ lobbies: lobbyMap, isLoading: false });
		} catch (err: unknown) { // Use unknown type
			console.error('Failed to fetch lobbies:', err);
			// Type checking for error message
			let errorMessage = 'Failed to load lobbies.';
			if (typeof err === 'object' && err !== null && 'response' in err) {
				const response = (err as any).response; // Basic check
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
	removeLobbyFromList: (lobbyId) => set((state) => {
		const updatedLobbies = { ...state.lobbies };
		delete updatedLobbies[lobbyId];
		return { lobbies: updatedLobbies };
	}),
}));


// --- Current Lobby Store Implementation ---

/**
 * Maps raw user data from WS messages to the LobbyUser type, ensuring username is populated.
 * Tries to get username from message, falls back to current auth user, then generates a default.
 */
const mapLobbyUsers = (users: any[] | undefined): LobbyUser[] => {
	if (!users) return [];
	const currentUser = useAuthStore.getState().user;

	return users.map(u => {
		const userId = u.id ?? NIL_UUID.toString();
		const username = u.username
		|| (currentUser && currentUser.id === userId ? currentUser.username : null)
		|| `User_${userId.substring(0, 4)}`; // Fallback username

		const baseUser: User = {
			id: userId,
			username: username ?? 'Unknown', // Ensure username is never nullish
			is_ephemeral: u.is_ephemeral ?? false,
			is_admin: u.is_admin ?? false
		};
		return {
			...baseUser,
			is_host: u.is_host ?? false,
			is_ready: u.is_ready ?? false
		};
	});
};


export const useCurrentLobbyStore = create<CurrentLobbyState>((set, get) => ({
	currentLobbyId: null,
	lobbyDetails: null,
	chatMessages: [],
	isConnected: false,
	isLoading: false,
	error: null,
	countdownStartTime: null,
	countdownDuration: null,

	setCurrentLobbyId: (lobbyId) => {
		const currentId = get().currentLobbyId;
		if (currentId === lobbyId) {
			// If ID is the same, ensure loading state reflects connection attempt status
			if (!get().isConnected && lobbyId !== null) {
				set({ isLoading: true, error: null }); // Mark as loading if trying to connect
			} else if (get().isConnected) {
				set({ isLoading: false, error: null }); // Mark as not loading if already connected
			}
			return;
		}

		// If ID is changing, reset relevant state for the new lobby context
		set({
			currentLobbyId: lobbyId,
			lobbyDetails: null,
			chatMessages: [],
			isConnected: false,
			isLoading: !!lobbyId, // Start loading only if joining a new lobby (non-null ID)
			error: null,
			countdownStartTime: null, // Reset countdown on lobby change
			countdownDuration: null
		});
	},

	processLobbyWebSocketMessage: (message) => {
		const currentLobbyId = get().currentLobbyId;
		// Ignore messages if no lobby is active or message is for a different lobby
		if (!currentLobbyId || !message || (message.lobby_id && message.lobby_id !== currentLobbyId)) {
			if (message && message.lobby_id && currentLobbyId && message.lobby_id !== currentLobbyId) {
				console.warn(`[Store] Ignoring message for lobby ${message.lobby_id} while current is ${currentLobbyId}`);
			}
			return;
		}

		set((state) => {
			// Double-check state hasn't changed unexpectedly
			if (state.currentLobbyId !== currentLobbyId) {
				console.warn(`[Store] Store ID changed during processing. Ignoring message type ${message?.type}`);
				return {};
			}

			const newLobbyDetails = state.lobbyDetails ? { ...state.lobbyDetails } : null;

			try {
				switch (message.type) {
					case 'lobby_state': {
						const usersState = mapLobbyUsers(message.lobby_status?.users);
						const updatedDetails: LobbyState = {
							id: message.lobby_id,
							hostUserID: message.host_id,
							host_id: message.host_id,
							type: message.lobby_type ?? 'private',
							gameMode: message.game_mode ?? 'unknown',
							inGame: message.in_game ?? false,
							game_id: message.game_id ?? null,
							houseRules: message.house_rules ?? state.lobbyDetails?.houseRules ?? {},
							circuit: message.circuit ?? state.lobbyDetails?.circuit ?? {},
							// Important: Prefer 'settings' from WS message if available for LobbySettings
							lobbySettings: message.settings ?? state.lobbyDetails?.lobbySettings ?? { autoStart: false },
							settings: message.settings ?? state.lobbyDetails?.settings ?? { autoStart: false }, // Keep `settings` for comparison if needed
							lobby_status: { users: usersState },
							your_id: message.your_id,
							your_is_host: message.your_is_host ?? (message.your_id === message.host_id),
							lobby_id: message.lobby_id
						};
						// Update state only if details changed or connection status was previously false/loading
						if (!deepEqual(state.lobbyDetails, updatedDetails) || !state.isConnected || state.isLoading) {
							return {
								lobbyDetails: updatedDetails,
								isLoading: false,
								error: null,
								isConnected: true
							};
						}
						return {};
					}

					case 'lobby_update': {
						if (!newLobbyDetails) return {};
						const newUsersUpdate = mapLobbyUsers(message.lobby_status?.users);
						const newLobbyStatusUpdate = { users: newUsersUpdate };

						// Prefer 'settings' if present in the update message
						const updatedLobbySettings = message.settings ?? newLobbyDetails.lobbySettings;

						const updatedDetails: LobbyState = {
							...newLobbyDetails,
							lobby_status: newLobbyStatusUpdate,
							hostUserID: message.host_id ?? newLobbyDetails.hostUserID,
							host_id: message.host_id ?? newLobbyDetails.host_id,
							your_is_host: message.host_id ? (newLobbyDetails.your_id === message.host_id) : newLobbyDetails.your_is_host,
							// Update both lobbySettings and settings based on the message or existing state
							lobbySettings: updatedLobbySettings,
							settings: message.settings ?? newLobbyDetails.settings // Keep settings consistent
						};
						if (!deepEqual(state.lobbyDetails, updatedDetails)) {
							return { lobbyDetails: updatedDetails };
						}
						return {};
					}

					case 'ready_update': {
						if (!newLobbyDetails?.lobby_status?.users) return {};
						let changed = false;
						const newUsersReady = newLobbyDetails.lobby_status.users.map(u => {
							if (u.id === message.user_id) {
								if (u.is_ready !== message.is_ready) {
									changed = true;
									return { ...u, is_ready: message.is_ready, username: message.username || u.username };
								}
							}
							return u;
						});
						if (changed) {
							return {
								lobbyDetails: {
									...newLobbyDetails,
									lobby_status: { users: newUsersReady }
								},
								// Cancel countdown if someone becomes unready
								countdownStartTime: message.is_ready ? state.countdownStartTime : null,
								countdownDuration: message.is_ready ? state.countdownDuration : null
							};
						}
						return {};
					}

					case 'lobby_rules_updated': {
						if (!newLobbyDetails) return {};
						if (!message.rules || typeof message.rules !== 'object') {
							console.warn('[Store] Received lobby_rules_updated without valid nested "rules" object.');
							return {};
						}

						const incomingRules = message.rules;
						const incomingHouseRules = incomingRules.house_rules || {};
						const incomingCircuit = incomingRules.circuit || {};
						// Expect lobby settings under the 'settings' key from the server
						const incomingSettings = incomingRules.settings || {};

						// Create potential new state based on merging incoming rules with existing ones
						const potentialDetails: LobbyState = { ...newLobbyDetails };
						let updated = false;

						const updatedHouseRules = { ...potentialDetails.houseRules, ...incomingHouseRules };
						if (!deepEqual(potentialDetails.houseRules, updatedHouseRules)) {
							potentialDetails.houseRules = updatedHouseRules;
							updated = true;
						}

						// Deep merge for circuit rules
						const updatedCircuit: CircuitSettings = { ...potentialDetails.circuit, ...incomingCircuit };
						if (incomingCircuit.rules) {
							updatedCircuit.rules = { ...(potentialDetails.circuit?.rules || {}), ...incomingCircuit.rules };
						}
						if (!deepEqual(potentialDetails.circuit, updatedCircuit)) {
							potentialDetails.circuit = updatedCircuit;
							updated = true;
						}

						// Update lobby settings (both 'settings' and 'lobbySettings' for consistency)
						const updatedSettings = { ...(potentialDetails.lobbySettings || {}), ...incomingSettings };
						if (!deepEqual(potentialDetails.lobbySettings, updatedSettings)) {
							potentialDetails.lobbySettings = updatedSettings;
							potentialDetails.settings = updatedSettings; // Keep both consistent
							updated = true;
						}

						if (updated) {
							console.log('[Store] lobby_rules_updated: Detected changes, updating state.');
							return { lobbyDetails: potentialDetails };
						} else {
							console.log('[Store] lobby_rules_updated: No effective changes detected.');
							return {};
						}
					}

					case 'game_start': {
						if (!newLobbyDetails) return {};
						if (!newLobbyDetails.inGame || newLobbyDetails.game_id !== message.game_id) {
							return {
								lobbyDetails: {
									...newLobbyDetails,
									inGame: true,
									game_id: message.game_id
								},
								isLoading: false,
								countdownStartTime: null, // Clear countdown on game start
								countdownDuration: null
							};
						}
						return {};
					}

					case 'error': {
						console.error('[Store] Received server error message:', message.message);
						// Clear countdown if an error related to it occurs? Maybe not necessary.
						return { error: message.message || 'Unknown error from server.', isLoading: false };
					}

					case 'chat': {
						if (message.user_id && message.msg && message.ts) {
							const chatUsername = message.username || `User_${message.user_id.substring(0, 4)}`;
							const chatMsg: ChatMessage = {
								user_id: message.user_id,
								username: chatUsername,
								msg: message.msg,
								ts: message.ts
							};
							if (!state.chatMessages.some(m => m.ts === chatMsg.ts && m.user_id === chatMsg.user_id)) {
								return { chatMessages: [...state.chatMessages, chatMsg].slice(-100) };
							}
						}
						return {};
					}

					// Countdown handling
					case 'lobby_countdown_start': {
						const duration = typeof message.seconds === 'number' ? message.seconds : 0;
						console.log(`[Store] Received countdown start: ${duration}s`);
						return {
							countdownStartTime: Date.now(),
							countdownDuration: duration
						};
					}
					case 'lobby_countdown_cancel': {
						console.log('[Store] Received countdown cancel');
						return {
							countdownStartTime: null,
							countdownDuration: null
						};
					}

					// Info messages, no state change needed in store
					case 'lobby_invite':
						console.log(`[Store] Received info message: ${message.type}`);
						return {};

					default:
						console.warn(`[Store] Unhandled WebSocket message type: ${message.type}`);
						return {};
				}
			} catch (processingError: unknown) { // Use unknown type
				console.error(`[Store] Error processing WebSocket message type ${message?.type}:`, processingError);
				let errorMsg = `Failed to process update (type: ${message?.type})`;
				if (processingError instanceof Error) {
					errorMsg = `${errorMsg}: ${processingError.message}`;
				}
				return { error: errorMsg, isLoading: false };
			}
		});
	},

	addChatMessage: (message) => {
		set((state) => {
			if (state.chatMessages.some(m => m.ts === message.ts && m.user_id === message.user_id)) {
				return {}; // Avoid duplicates
			}
			return { chatMessages: [...state.chatMessages, message].slice(-100) }; // Keep last 100 messages
		});
	},

	setConnected: (status) => set((state) => {
		if (state.isConnected === status) return {};
		// If disconnected, clear countdown
		const countdownState = status ? { countdownStartTime: state.countdownStartTime, countdownDuration: state.countdownDuration } : { countdownStartTime: null, countdownDuration: null };
		return { isConnected: status, error: status ? null : state.error, isLoading: status ? state.isLoading : false, ...countdownState };
	}),

	setLoading: (loading) => set((state) => {
		if (state.isLoading === loading) return {};
		return { isLoading: loading, error: loading ? null : state.error };
	}),

	setError: (error) => set((state) => {
		if (state.error === error) return {};
		return { error, isLoading: false };
	}),

	clearError: () => set((state) => {
		if (state.error === null) return {};
		return { error: null };
	}),

	clearChat: () => set({ chatMessages: [] }),

	createAndJoinLobby: async (settings) => {
		set({ isLoading: true, error: null });
		try {
			const newLobby = await apiCreateLobby(settings);
			if (newLobby?.id) {
				// Trigger connection logic by setting the current lobby ID
				get().setCurrentLobbyId(newLobby.id);
				return newLobby.id;
			} else {
				throw new Error('Lobby creation API call did not return a valid lobby ID.');
			}
		} catch (err: unknown) { // Use unknown type
			console.error('[Store] Failed to create lobby:', err);
			let errorMessage = 'Failed to create lobby.';
			// Extract message similarly to fetchLobbies
			if (typeof err === 'object' && err !== null && 'response' in err) {
				const response = (err as any).response;
				if (response?.data?.message) {
					errorMessage = response.data.message;
				} else if (response?.statusText) {
					errorMessage = response.statusText;
				}
			} else if (err instanceof Error) {
				errorMessage = err.message;
			}
			set({ error: errorMessage, isLoading: false });
			return null;
		}
	},

	leaveLobby: () => {
		// This action only clears the client-side state.
		// WebSocket closure is handled by the useLobbySocket hook reacting to currentLobbyId becoming null.
		set({
			currentLobbyId: null,
			lobbyDetails: null,
			chatMessages: [],
			isConnected: false,
			isLoading: false,
			error: null,
			countdownStartTime: null, // Clear countdown on leave
			countdownDuration: null
		});
	}
}));

// --- Selectors ---
export const useCurrentLobbyDetails = () => {
	return useCurrentLobbyStore((state) => state.lobbyDetails);
};

export const useLobbyConnectionStatus = () => {
	return useCurrentLobbyStore((state) => ({
		isConnected: state.isConnected,
		isLoading: state.isLoading,
		error: state.error
	}));
};

export const useLobbyChatMessages = () => {
	return useCurrentLobbyStore((state) => state.chatMessages);
};

/**
 * Performs a deep comparison between two values to determine if they are
 * equivalent. Handles basic types, arrays, and nested objects.
 * Does not handle functions, Sets, Maps, or complex circular references robustly.
 */
function deepEqual(obj1: any, obj2: any): boolean {
	// Basic checks for identity and type
	if (obj1 === obj2) return true;
	if (obj1 === null || obj2 === null || typeof obj1 !== typeof obj2) return false;

	const t1 = typeof obj1;
	const t2 = typeof obj2;

	// If not objects, perform simple comparison
	if (t1 !== 'object' || t2 !== 'object') {
		return obj1 === obj2;
	}

	// Handle Dates
	if (obj1 instanceof Date && obj2 instanceof Date) {
		return obj1.getTime() === obj2.getTime();
	}

	// Handle Arrays
	if (Array.isArray(obj1) && Array.isArray(obj2)) {
		if (obj1.length !== obj2.length) return false;
		for (let i = 0; i < obj1.length; i++) {
			if (!deepEqual(obj1[i], obj2[i])) return false;
		}
		return true;
	}

	// Handle Objects
	if (obj1 instanceof Object && obj2 instanceof Object) {
		const keys1 = Object.keys(obj1);
		const keys2 = Object.keys(obj2);

		if (keys1.length !== keys2.length) return false;

		// Ensure all keys in obj1 exist in obj2 and their values are equal
		for (const key of keys1) {
			if (!Object.prototype.hasOwnProperty.call(obj2, key) || !deepEqual(obj1[key], obj2[key])) {
				return false;
			}
		}
		return true;
	}

	// If none of the above, types are different or unsupported
	return false;
}