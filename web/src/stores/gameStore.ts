/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/gameStore.ts
import { create } from 'zustand';
import type { ObfGameState, ObfCard, ServerGameEvent } from '@/types/game';
import { immer } from 'zustand/middleware/immer';

interface GameState {
	gameId: string | null;
	gameState: ObfGameState | null;
	isConnected: boolean;
	isLoading: boolean; // For initial connection/sync
	error: string | null;
	displayedDrawnCard: ObfCard | null; // Card magnified on screen after draw
	pendingAction: string | null; // e.g., 'discard_replace', 'special_action'
	isProcessingAction: boolean; // Indicate if client is waiting for server response after sending action
	lastMessageTimestamp: number; // Track last message for debugging/staleness
}

interface GameActions {
	setGameId: (id: string | null) => void;
	setConnected: (status: boolean) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	clearError: () => void;
	processGameWebSocketMessage: (message: ServerGameEvent) => void;
	clearGameState: () => void; // For leaving game
	setDisplayedDrawnCard: (card: ObfCard | null) => void;
	clearDisplayedDrawnCard: () => void;
	setProcessingAction: (isProcessing: boolean) => void;
}

// Initial empty state
const initialState: GameState = {
	gameId: null,
	gameState: null,
	isConnected: false,
	isLoading: false,
	error: null,
	displayedDrawnCard: null,
	pendingAction: null, // e.g., 'discard_replace', 'special_action'
	isProcessingAction: false,
	lastMessageTimestamp: 0
};

export const useGameStore = create<GameState & GameActions>()(
	immer((set, get) => ({
		...initialState,

		setGameId: (id) => {
			set((state) => {
				// Reset state if game ID changes
				if (state.gameId !== id) {
					state.gameState = null;
					state.isConnected = false;
					state.isLoading = !!id; // Start loading if joining a new game
					state.error = null;
					state.displayedDrawnCard = null;
					state.pendingAction = null;
					state.isProcessingAction = false;
				}
				state.gameId = id;
			});
		},

		setConnected: (status) => {
			set((state) => {
				state.isConnected = status;
				if (!status) {
					// Clear loading/error on explicit disconnect? Or handled by hook?
					// Maybe set error if disconnection was unexpected.
					// state.isLoading = false;
				} else {
					state.isLoading = false; // Mark as not loading once connected
					state.error = null;
				}
			});
		},

		setLoading: (loading) => {
			set((state) => {
				state.isLoading = loading;
				if (loading) state.error = null; // Clear error when starting to load
			});
		},

		setError: (error) => {
			set((state) => {
				state.error = error;
				state.isLoading = false; // Stop loading on error
				state.isConnected = false; // Assume disconnected on error
			});
		},

		clearError: () => {
			set((state) => {
				state.error = null;
			});
		},

		clearGameState: () => {
			set(initialState); // Reset to initial state
		},

		setDisplayedDrawnCard: (card) => {
			set((state) => {
				state.displayedDrawnCard = card;
			});
		},

		clearDisplayedDrawnCard: () => {
			set((state) => {
				state.displayedDrawnCard = null;
			});
		},

		setProcessingAction: (isProcessing) => {
			set((state) => {
				state.isProcessingAction = isProcessing;
			});
		},

		processGameWebSocketMessage: (message) => {
			set((state) => {
				state.lastMessageTimestamp = Date.now();
				state.isProcessingAction = false; // Assume action processed on any message receipt

				if (!state.gameState && message.type !== 'private_sync_state') {
					console.warn(`[GameStore] Received message type ${message.type} before initial state sync. Ignoring.`);
					return;
				}

				try { // Add try-catch for safety during state updates
					switch (message.type) {
						case 'private_sync_state':
							state.gameState = message.state;
							state.isLoading = false; // Sync received, no longer loading initial state
							state.isConnected = true; // Mark as connected on successful sync
							state.error = null;
							state.pendingAction = null; // Clear pending actions on full sync
							// Determine pending action based on new state
							const userState = state.gameState.players.find(p => p.playerId === state.gameState?.players.find(pl => pl.revealedHand !== undefined)?.playerId); // Find 'self'
							if (userState?.drawnCard && state.gameState.currentPlayerId === userState.playerId && !state.gameState.gameOver && state.gameState.started) {
								state.pendingAction = 'discard_replace';
							}
							// TODO: Add logic for pending special action based on state.gameState.specialAction
							break;

						case 'private_initial_cards':
							// This is mainly for the client to "remember" cards.
							// We don't usually update the main ObfGameState based on this,
							// but could store it separately if needed for UI hints.
							console.log('[GameStore] Received initial cards:', message.card1, message.card2);
							break;

						case 'game_player_turn':
							if (state.gameState) {
								state.gameState.currentPlayerId = message.user.id;
								state.gameState.turnId = message.turn;
								state.pendingAction = null; // New turn clears pending actions
								state.displayedDrawnCard = null; // Clear magnified card
								// Update isCurrentTurn for all players
								state.gameState.players.forEach(p => {
									p.isCurrentTurn = (p.playerId === message.user.id);
								});
							}
							break;

						case 'player_draw_stockpile':
						case 'private_draw_stockpile': // Treat both similarly for state update, but display logic differs
							if (state.gameState) {
								if (message.type === 'player_draw_stockpile') {
									// Public draw - update stockpile size, show card back magnified for others
									if (message.payload.source === 'stockpile') {
										state.gameState.stockpileSize = message.payload.stockpileSize ?? state.gameState.stockpileSize - 1;
									} else {
										state.gameState.discardSize = message.payload.discardSize ?? state.gameState.discardSize - 1;
										// Update discardTop if drawn from discard
										state.gameState.discardTop = null; // Simplified: assume next sync will fix it
									}
									const player = state.gameState.players.find(p => p.playerId === message.user.id);
									const self = state.gameState.players.find(p => p.revealedHand !== undefined);
									if (player && self && player.playerId !== self.playerId) {
										// Magnify card back for others
										state.displayedDrawnCard = { id: message.card.id, known: false };
									}
								} else { // Private draw
									// Update the specific player's drawnCard state
									const player = state.gameState.players.find(p => p.playerId === state.gameState?.players.find(pl => pl.revealedHand !== undefined)?.playerId); // Find 'self'
									if (player && player.playerId === state.gameState.currentPlayerId) {
										player.drawnCard = message.card;
										state.pendingAction = 'discard_replace'; // Player must now discard/replace
										state.displayedDrawnCard = message.card; // Magnify revealed card for self
									}
									// Update stockpile/discard size based on source
									if (message.payload.source === 'stockpile') {
										state.gameState.stockpileSize--; // Approximate if size not sent
									} else {
										state.gameState.discardSize--; // Approximate
										state.gameState.discardTop = null; // Simplified
									}
								}
							}
							break;

						case 'player_discard':
							if (state.gameState) {
								// Update discard pile
								state.gameState.discardSize++;
								state.gameState.discardTop = message.card;
								// Clear drawn card for the discarding player
								const player = state.gameState.players.find(p => p.playerId === message.user.id);
								if (player) {
									player.drawnCard = null;
								}
								state.displayedDrawnCard = null; // Clear magnified card
								state.pendingAction = null; // Action completed (unless special triggered)
							}
							break;

						case 'player_replace': // Treat as discard for state update simplicity
							if (state.gameState) {
								state.gameState.discardSize++;
								state.gameState.discardTop = message.card; // The replaced card goes to discard
								const player = state.gameState.players.find(p => p.playerId === message.user.id);
								if (player) {
									player.drawnCard = null;
									// If we have revealedHand, update it (tricky without full card info)
									if (player.revealedHand && message.card.idx !== undefined) {
										// We don't know what the drawn card was here easily,
										// rely on next sync or private events for perfect hand state.
										// For now, just clear drawnCard.
									}
								}
								state.displayedDrawnCard = null; // Clear magnified card
								state.pendingAction = null; // Action completed (unless special triggered)
							}
							break;

						case 'player_special_choice':
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.revealedHand !== undefined); // Find 'self'
								if (player && player.playerId === message.user.id) {
									state.pendingAction = 'special_action'; // Player needs to make choice
								}
								// Optionally store special action details
								state.gameState.specialAction = {
									active: true,
									playerId: message.user.id,
									cardRank: message.card.rank
								};
							}
							break;

						case 'player_special_action': // Public confirmation/info
							if (state.gameState) {
								// If the action was 'skip' or a completed swap/peek, clear pending state
								if (message.special === 'skip' || message.special === 'peek_self' || message.special === 'peek_other' || message.special === 'swap_blind' || message.special === 'swap_peek_swap') {
									state.pendingAction = null;
									state.gameState.specialAction = null;
								}
								// If it was swap_peek_reveal, the action is still pending (waiting for swap/skip)
								// UI might update based on card1/card2 info (e.g., highlight targets)
							}
							break;

						// --- Private Events ---
						case 'private_special_action_success':
							// UI might use this to show revealed cards temporarily
							// state updates handled by public events or sync
							break;
						case 'private_special_action_fail':
							// Show error message to the user
							// state.error = `Special action failed: ${message.message}`; // Maybe too aggressive?
							state.pendingAction = 'special_action'; // Allow retry or skip
							state.isProcessingAction = false; // Allow sending new action
							break;

						// --- Snap Events ---
						case 'player_snap_success':
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === message.user.id);
								if (player) {
									player.handSize--; // Update hand size
									// Remove card from revealedHand if it's 'self'
									if (player.revealedHand && message.card.idx !== undefined) {
										player.revealedHand.splice(message.card.idx, 1);
										// Adjust indices of subsequent cards
										for (let i = message.card.idx; i < player.revealedHand.length; i++) {
											if (player.revealedHand[i].idx !== undefined) {
												player.revealedHand[i].idx!--;
											}
										}
									}
								}
								state.gameState.discardSize++;
								state.gameState.discardTop = message.card;
							}
							break;
						case 'player_snap_fail':
							// No immediate state change needed from public fail event
							break;
						case 'player_snap_penalty': // Public penalty draw notification
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === message.user.id);
								if (player) {
									player.handSize++; // Increment hand size
								}
								state.gameState.stockpileSize--; // Decrement stockpile
							}
							break;
						case 'private_snap_penalty': // Private penalty card details
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.revealedHand !== undefined); // Find 'self'
								if (player && player.playerId === state.gameState?.players.find(p => p.playerId === message.card?.id)?.playerId) { // Check if message is for self? No, need user ID
									// This logic is tricky without user id in the private message.
									// Assuming it's for 'self' if received.
									if (player.revealedHand && message.card.idx !== undefined) {
										// Add card at the specified index
										player.revealedHand.splice(message.card.idx, 0, message.card);
										// Adjust subsequent indices (might be off if server adds to end)
										// Relying on sync state is safer here.
									}
								}
							}
							break;

						// --- Cambia & Game End ---
						case 'player_cambia':
							if (state.gameState) {
								state.gameState.cambiaCalled = true;
								state.gameState.cambiaCallerId = message.user.id;
								const player = state.gameState.players.find(p => p.playerId === message.user.id);
								if (player) player.hasCalledCambia = true;
								state.pendingAction = null; // Turn ends immediately
								state.displayedDrawnCard = null; // Clear magnified card
							}
							break;

						case 'game_end':
							if (state.gameState) {
								state.gameState.gameOver = true;
								state.gameState.currentPlayerId = null;
								state.pendingAction = null;
								state.displayedDrawnCard = null;
								// Store final scores/winner if needed, maybe trigger modal
								// state.finalScores = message.payload.scores;
								// state.winnerId = message.payload.winner;
							}
							break;

						case 'error': // Server-sent error message
							state.error = `Game Error: ${message.message}`;
							state.isProcessingAction = false; // Allow new actions after error
							// Should we clear pending state on error? Depends on the error.
							break;

						default:
							console.warn(`[GameStore] Unhandled WebSocket message type: ${message.type}`);
					}
				} catch (e: unknown) {
					console.error(`[GameStore] Error processing message type ${message.type}:`, e);
					state.error = `Client error processing game update: ${(e instanceof Error) ? e.message : 'Unknown error'}`;
				}
			});
		}

	}))
);

// --- Selectors ---
export const selectGameState = (state: GameState) => state.gameState;
export const selectIsConnected = (state: GameState) => state.isConnected;
export const selectIsLoading = (state: GameState) => state.isLoading;
export const selectGameError = (state: GameState) => state.error;
export const selectDisplayedDrawnCard = (state: GameState) => state.displayedDrawnCard;
export const selectPendingAction = (state: GameState) => state.pendingAction;
export const selectIsProcessingAction = (state: GameState) => state.isProcessingAction;
export const selectCurrentPlayerId = (state: GameState) => state.gameState?.currentPlayerId;
export const selectSelfPlayerState = (state: GameState) => state.gameState?.players.find(p => p.revealedHand !== undefined);
export const selectIsSelfTurn = (state: GameState) => {
	const self = selectSelfPlayerState(state);
	const currentId = selectCurrentPlayerId(state);
	return self?.playerId === currentId && !!state.gameState?.started && !state.gameState?.gameOver;
};