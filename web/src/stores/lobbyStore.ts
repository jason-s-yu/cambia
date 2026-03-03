/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/lobbyStore.ts
import { create } from 'zustand';
import type { LobbyState, ChatMessage, CircuitSettings, User, LobbyUser, MatchState } from '@/types/index';
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

/** Hub phase values matching the server's LobbyPhase string encoding. */
export type LobbyPhase = 'open' | 'searching' | 'ready_check' | 'countdown' | 'in_game' | 'round_end' | 'post_game' | 'match_end';

/** State for managing the currently joined lobby */
interface CurrentLobbyState {
	currentLobbyId: string | null;
	lobbyDetails: LobbyState | null;
	chatMessages: ChatMessage[];
	isConnected: boolean; // WebSocket connection status
	isLoading: boolean;   // Loading state for API calls or initial WS connection/sync
	error: string | null;
	phase: LobbyPhase; // Current hub phase
	seq: number; // Last known server seq for this lobby
	// Countdown state
	countdownStartTime: number | null; // System timestamp (ms) when countdown started
	countdownDuration: number | null; // Duration in seconds
	matchState: MatchState | null;
	setCurrentLobbyId: (lobbyId: string | null) => void;
	processLobbyWebSocketMessage: (type: string, payload: any) => void; // Envelope-unwrapped: type + payload
	addChatMessage: (message: ChatMessage) => void;
	setConnected: (status: boolean) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	clearError: () => void;
	clearChat: () => void;
	setPhase: (phase: LobbyPhase) => void;
	forceSync: (state: any) => void; // Full state replacement from sync_state
	createAndJoinLobby: (settings: Partial<LobbyState>) => Promise<string | null>;
	leaveLobby: () => void;
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
	phase: 'open' as LobbyPhase,
	seq: 0,
	countdownStartTime: null,
	countdownDuration: null,
	matchState: null,

	setPhase: (phase) => set({ phase }),

	forceSync: (payload) => {
		set((state) => {
			const updated: Partial<CurrentLobbyState> = {};
			if (typeof payload?.seq === "number") updated.seq = payload.seq;
			if (payload.phase) updated.phase = payload.phase as LobbyPhase;
			// Rebuild lobby details from sync payload (same shape as lobby_state)
			if (payload.lobby_id || payload.lobby_status) {
				const usersState = mapLobbyUsers(payload.lobby_status?.users);
				updated.lobbyDetails = {
					id: payload.lobby_id ?? state.currentLobbyId,
					hostUserID: payload.host_id,
					host_id: payload.host_id,
					type: payload.lobby_type ?? state.lobbyDetails?.type ?? 'private',
					gameMode: payload.game_mode ?? state.lobbyDetails?.gameMode ?? 'unknown',
					inGame: payload.in_game ?? false,
					game_id: payload.game_id ?? null,
					houseRules: payload.house_rules ?? state.lobbyDetails?.houseRules ?? {},
					circuit: payload.circuit ?? state.lobbyDetails?.circuit ?? {},
					lobbySettings: payload.settings ?? state.lobbyDetails?.lobbySettings ?? { autoStart: false },
					settings: payload.settings ?? state.lobbyDetails?.settings ?? { autoStart: false },
					lobby_status: { users: usersState },
					your_id: payload.your_id,
					your_is_host: payload.your_is_host ?? (payload.your_id === payload.host_id),
					lobby_id: payload.lobby_id ?? state.currentLobbyId,
				};
			}
			if (payload.match_state) {
				updated.matchState = {
					queueId: payload.match_state.queue_id ?? '',
					isRanked: payload.match_state.is_ranked ?? false,
					totalRounds: payload.match_state.total_rounds ?? 1,
					currentRound: payload.match_state.current_round ?? 0,
					roundScores: payload.match_state.round_history ?? [],
					cumulativeScores: payload.match_state.cumulative_scores ?? {},
				};
			}
			updated.isConnected = true;
			updated.isLoading = false;
			updated.error = null;
			return updated;
		});
	},

	setCurrentLobbyId: (lobbyId) => {
		const currentId = get().currentLobbyId;
		if (currentId === lobbyId) {
			if (!get().isConnected && lobbyId !== null) {
				set({ isLoading: true, error: null });
			} else if (get().isConnected) {
				set({ isLoading: false, error: null });
			}
			return;
		}

		set({
			currentLobbyId: lobbyId,
			lobbyDetails: null,
			chatMessages: [],
			isConnected: false,
			isLoading: !!lobbyId,
			error: null,
			phase: 'open' as LobbyPhase,
			countdownStartTime: null,
			countdownDuration: null,
			matchState: null
		});
	},

	processLobbyWebSocketMessage: (type, payload) => {
		const currentLobbyId = get().currentLobbyId;
		if (!currentLobbyId || !type) return;

		// Alias: the payload IS the message data (already unwrapped from envelope)
		const message = payload ?? {};

		set((state) => {
			if (state.currentLobbyId !== currentLobbyId) return {};

			const newLobbyDetails = state.lobbyDetails ? { ...state.lobbyDetails } : null;

			try {
				switch (type) {
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
							lobbySettings: message.settings ?? state.lobbyDetails?.lobbySettings ?? { autoStart: false },
							settings: message.settings ?? state.lobbyDetails?.settings ?? { autoStart: false },
							lobby_status: { users: usersState },
							your_id: message.your_id,
							your_is_host: message.your_is_host ?? (message.your_id === message.host_id),
							lobby_id: message.lobby_id
						};
						const phaseUpdate = message.phase ? { phase: message.phase as LobbyPhase } : {};
						if (!deepEqual(state.lobbyDetails, updatedDetails) || !state.isConnected || state.isLoading) {
							return {
								lobbyDetails: updatedDetails,
								isLoading: false,
								error: null,
								isConnected: true,
								...phaseUpdate
							};
						}
						return phaseUpdate;
					}

					case 'lobby_update': {
						if (!newLobbyDetails) return {};
						const newUsersUpdate = mapLobbyUsers(message.lobby_status?.users);
						const newLobbyStatusUpdate = { users: newUsersUpdate };
						const updatedLobbySettings = message.settings ?? newLobbyDetails.lobbySettings;

						const updatedDetails: LobbyState = {
							...newLobbyDetails,
							lobby_status: newLobbyStatusUpdate,
							hostUserID: message.host_id ?? newLobbyDetails.hostUserID,
							host_id: message.host_id ?? newLobbyDetails.host_id,
							your_is_host: message.host_id ? (newLobbyDetails.your_id === message.host_id) : newLobbyDetails.your_is_host,
							lobbySettings: updatedLobbySettings,
							settings: message.settings ?? newLobbyDetails.settings
						};
						if (!deepEqual(state.lobbyDetails, updatedDetails)) {
							return { lobbyDetails: updatedDetails };
						}
						return {};
					}

					case 'phase_change': {
						const newPhase = message.phase as LobbyPhase;
						if (newPhase && newPhase !== state.phase) {
							const updates: any = { phase: newPhase };
							// Clear countdown on phase transitions out of countdown
							if (newPhase !== 'countdown') {
								updates.countdownStartTime = null;
								updates.countdownDuration = null;
							}
							return updates;
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
								countdownStartTime: message.is_ready ? state.countdownStartTime : null,
								countdownDuration: message.is_ready ? state.countdownDuration : null
							};
						}
						return {};
					}

					case 'lobby_rules_updated': {
						if (!newLobbyDetails) return {};
						if (!message.rules || typeof message.rules !== 'object') return {};

						const incomingRules = message.rules;
						const incomingHouseRules = incomingRules.house_rules || {};
						const incomingCircuit = incomingRules.circuit || {};
						const incomingSettings = incomingRules.settings || {};

						const potentialDetails: LobbyState = { ...newLobbyDetails };
						let updated = false;

						const updatedHouseRules = { ...potentialDetails.houseRules, ...incomingHouseRules };
						if (!deepEqual(potentialDetails.houseRules, updatedHouseRules)) {
							potentialDetails.houseRules = updatedHouseRules;
							updated = true;
						}

						const updatedCircuit: CircuitSettings = { ...potentialDetails.circuit, ...incomingCircuit };
						if (incomingCircuit.rules) {
							updatedCircuit.rules = { ...(potentialDetails.circuit?.rules || {}), ...incomingCircuit.rules };
						}
						if (!deepEqual(potentialDetails.circuit, updatedCircuit)) {
							potentialDetails.circuit = updatedCircuit;
							updated = true;
						}

						const updatedSettings = { ...(potentialDetails.lobbySettings || {}), ...incomingSettings };
						if (!deepEqual(potentialDetails.lobbySettings, updatedSettings)) {
							potentialDetails.lobbySettings = updatedSettings;
							potentialDetails.settings = updatedSettings;
							updated = true;
						}

						if (updated) return { lobbyDetails: potentialDetails };
						return {};
					}

					case 'game_start': {
						if (!newLobbyDetails) return {};
						return {
							lobbyDetails: {
								...newLobbyDetails,
								inGame: true,
								game_id: message.game_id ?? newLobbyDetails.game_id
							},
							phase: 'in_game' as LobbyPhase,
							isLoading: false,
							countdownStartTime: null,
							countdownDuration: null
						};
					}

					case 'error': {
						console.error('[Store] Received server error:', message.error || message.message);
						return { error: message.error || message.message || 'Unknown error from server.', isLoading: false };
					}

					case 'chat': {
						if (message.userID && message.msg) {
							const chatUsername = message.username || `User_${message.userID.substring(0, 4)}`;
							const chatMsg: ChatMessage = {
								user_id: message.userID,
								username: chatUsername,
								msg: message.msg,
								ts: message.ts ?? new Date().toISOString()
							};
							if (!state.chatMessages.some(m => m.ts === chatMsg.ts && m.user_id === chatMsg.user_id)) {
								return { chatMessages: [...state.chatMessages, chatMsg].slice(-100) };
							}
						}
						return {};
					}

					case 'lobby_countdown_start': {
						const duration = typeof message.seconds === 'number' ? message.seconds : 0;
						return {
							countdownStartTime: Date.now(),
							countdownDuration: duration
						};
					}
					case 'lobby_countdown_cancel': {
						return {
							countdownStartTime: null,
							countdownDuration: null
						};
					}

					case 'lobby_invite':
						return {};

					case 'search_status': {
						const searching = message.searching === true;
						return { phase: searching ? 'searching' as LobbyPhase : 'open' as LobbyPhase };
					}

					case 'match_found': {
						return {
							phase: 'ready_check' as LobbyPhase,
							matchState: {
								queueId: message.queue_id ?? '',
								isRanked: message.is_ranked ?? false,
								totalRounds: message.total_rounds ?? 1,
								currentRound: 0,
								roundScores: [],
								cumulativeScores: {},
							},
						};
					}

					case 'round_start': {
						const currentMatchState = state.matchState;
						return {
							phase: 'in_game' as LobbyPhase,
							matchState: currentMatchState ? {
								...currentMatchState,
								currentRound: message.round ?? (currentMatchState.currentRound + 1),
							} : null,
						};
					}

					case 'round_end': {
						const ms = state.matchState;
						const roundScores = message.round_scores ?? {};
						const cumulativeScores = message.cumulative_scores ?? {};
						const subsidies = message.subsidies ?? {};
						return {
							phase: 'round_end' as LobbyPhase,
							matchState: ms ? {
								...ms,
								currentRound: message.round ?? ms.currentRound,
								roundScores: [...ms.roundScores, roundScores],
								cumulativeScores,
								subsidies,
							} : null,
						};
					}

					case 'match_end': {
						const ms2 = state.matchState;
						const finalRoundScores = message.round_scores ?? {};
						const finalCumulative = message.cumulative_scores ?? {};
						const finalSubsidies = message.subsidies ?? {};
						const ratingChanges = message.rating_changes;
						return {
							phase: 'match_end' as LobbyPhase,
							matchState: ms2 ? {
								...ms2,
								roundScores: message.final ? [...ms2.roundScores, finalRoundScores] : ms2.roundScores,
								cumulativeScores: finalCumulative,
								subsidies: finalSubsidies,
								ratingChanges: ratingChanges ?? undefined,
							} : null,
						};
					}

					case 'game_results': {
						// Casual single-game results (existing flow)
						return { phase: 'post_game' as LobbyPhase };
					}

					default:
						console.warn(`[Store] Unhandled lobby message type: ${type}`);
						return {};
				}
			} catch (processingError: unknown) {
				console.error(`[Store] Error processing message type ${type}:`, processingError);
				let errorMsg = `Failed to process update (type: ${type})`;
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
		set({
			currentLobbyId: null,
			lobbyDetails: null,
			chatMessages: [],
			isConnected: false,
			isLoading: false,
			error: null,
			phase: 'open' as LobbyPhase,
			seq: 0,
			countdownStartTime: null,
			countdownDuration: null,
			matchState: null
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