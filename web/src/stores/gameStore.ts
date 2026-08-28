/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/gameStore.ts
import { create } from 'zustand';
import type { ObfGameState, ObfCard, EventCard } from '@/types/game';
import { immer } from 'zustand/middleware/immer';
import { useAuthStore } from './authStore';

/** A face shown to this client by an ability (7/8 own card, 9/T opponent card, King both). */
export interface RevealedCard {
	id: string;
	rank?: string;
	suit?: string;
	value?: number;
	idx?: number;
	ownerId?: string;
}

/** The most recent ability reveal, kept so the table can show the faces while the ability plays out. */
export interface AbilityReveal {
	special: string;
	/** Client receipt time; the table holds a peek on screen for a beat past this. */
	at: number;
	cards: RevealedCard[];
}

interface GameState {
	gameId: string | null;
	gameState: ObfGameState | null;
	seq: number;
	isConnected: boolean;
	isLoading: boolean; // For initial connection/sync
	error: string | null;
	displayedDrawnCard: ObfCard | null; // Card magnified on screen after draw
	pendingAction: string | null; // e.g., 'discard_replace', 'special_action'
	isProcessingAction: boolean; // Indicate if client is waiting for server response after sending action
	lastMessageTimestamp: number; // Track last message for debugging/staleness
	// serverNow - clientNow (ms), captured from the serverNow field on sync_state/game_player_turn
	// events. Applied to Date.now() when deriving a live turn countdown from gameState.turnDeadline
	// (cambia-488), so a skewed client clock doesn't distort the remaining-time display.
	serverClockOffsetMs: number;
	// Final adjusted scores/winner from the game_end event (userID -> score), populated once the
	// game ends. Casual (single-game, non-ranked) results have no matchState, so DsResultsView
	// reads these directly instead (cambia-510).
	finalScores: Record<string, number> | null;
	winnerId: string | null;
	// Faces revealed to this client by abilities (cambia-848 F3). The service's
	// private_special_action_success is the only carrier of a peeked face: own cards it names
	// are marked seen server-side (CardTracker.SeenByPlayer) and come back known on the next
	// sync, so they are folded into revealedHand here as well; opponent faces never appear in
	// a sync, so they live only in these two fields. seenFaces accumulates every face by card
	// id for the life of the game so a swap that moves a looked-at card into the own hand
	// keeps its face; abilityReveal is the latest reveal, for the table's transient display.
	seenFaces: Record<string, RevealedCard>;
	abilityReveal: AbilityReveal | null;
}

interface GameActions {
	setGameId: (id: string | null) => void;
	setConnected: (status: boolean) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	clearError: () => void;
	processGameWebSocketMessage: (type: string, payload: any) => void;
	forceSync: (payload: any) => void;
	clearGameState: () => void; // For leaving game
	setDisplayedDrawnCard: (card: ObfCard | null) => void;
	clearDisplayedDrawnCard: () => void;
	setProcessingAction: (isProcessing: boolean) => void;
}

// Initial empty state
const initialState: GameState = {
	gameId: null,
	gameState: null,
	seq: 0,
	isConnected: false,
	isLoading: false,
	error: null,
	displayedDrawnCard: null,
	pendingAction: null, // e.g., 'discard_replace', 'special_action'
	isProcessingAction: false,
	lastMessageTimestamp: 0,
	serverClockOffsetMs: 0,
	finalScores: null,
	winnerId: null,
	seenFaces: {},
	abilityReveal: null
};

export const useGameStore = create<GameState & GameActions>()(
	immer((set) => ({
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
					state.seenFaces = {};
					state.abilityReveal = null;
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

		forceSync: (payload) => {
			set((state) => {
				if (payload?.state) {
					state.gameState = payload.state;
					if (typeof payload.state.serverNow === 'number') {
						state.serverClockOffsetMs = payload.state.serverNow - Date.now();
					}
				}
				if (typeof payload?.seq === 'number') {
					state.seq = payload.seq;
				}
				state.pendingAction = null;
				state.displayedDrawnCard = null;
				state.abilityReveal = null;
				state.isLoading = false;
				state.isConnected = true;
				state.error = null;
			});
		},

		processGameWebSocketMessage: (type, payload) => {
			set((state) => {
				state.lastMessageTimestamp = Date.now();
				state.isProcessingAction = false; // Assume action processed on any message receipt

				// Identify 'self' by the authenticated user id. Opponent hands now also carry a
				// revealedHand (hidden id references so opponent-targeting abilities have real UUIDs,
				// cambia-509), so the old "first player with a revealedHand" heuristic no longer picks
				// out self and must not be used for self-detection.
				const selfPlayerId = useAuthStore.getState().user?.id ?? null;

				if (!state.gameState && type !== 'private_sync_state') {
					console.warn(`[GameStore] Received message type ${type} before initial state sync. Ignoring.`);
					return;
				}

				try { // Add try-catch for safety during state updates
					switch (type) {
						case 'private_sync_state': {
							state.gameState = payload.state;
							state.isLoading = false; // Sync received, no longer loading initial state
							state.isConnected = true; // Mark as connected on successful sync
							state.error = null;
							state.pendingAction = null; // Clear pending actions on full sync
							state.abilityReveal = null;
							// Recompute clock skew from this snapshot's serverNow (cambia-488).
							if (typeof payload.state?.serverNow === 'number') {
								state.serverClockOffsetMs = payload.state.serverNow - Date.now();
							}
							// Determine pending action based on new state
							const gs = state.gameState;
							if (gs) {
								const userState = gs.players.find(p => p.playerId === selfPlayerId); // Find 'self'
								for (const c of userState?.revealedHand ?? []) {
									if (c.known && c.rank) state.seenFaces[c.id] = { id: c.id, rank: c.rank, suit: c.suit, value: c.value, idx: c.idx, ownerId: selfPlayerId ?? undefined };
								}
								if (userState?.drawnCard && gs.currentPlayerId === userState.playerId && !gs.gameOver && gs.started) {
									state.pendingAction = 'discard_replace';
								} else if (gs.specialAction?.active && gs.specialAction.playerId === selfPlayerId && !gs.gameOver && gs.started) {
									// The service's ObfGameState (service/internal/game/sync_state.go) serializes
									// SpecialActionState into private_sync_state (cambia-763 F1), so a client that
									// resyncs mid-action (reconnect, tab refresh) restores pendingAction here.
									state.pendingAction = 'special_action';
								}
							}
							break;
						}

						case 'private_initial_cards': {
							// Pregame peek reveal, own hand only. The event carries one entry per peeked
							// slot under `cards` (cambia-817); the count is the initialViewCount house
							// rule, up to cardsPerPlayer, so nothing here may assume two.
							//
							// The service fires this right after the opening private_sync_state, which it
							// builds before marking these cards seen: that snapshot therefore shows every
							// own slot face-down. Apply the reveal to revealedHand here so the peek is
							// visible for the whole pregame window instead of only from the next full sync
							// at game start.
							const cards = Array.isArray(payload.cards) ? (payload.cards as EventCard[]) : [];
							const self = state.gameState?.players.find(p => p.playerId === selfPlayerId);
							if (self?.revealedHand) {
								for (const card of cards) {
									if (!card || typeof card.idx !== 'number') continue;
									const slot = self.revealedHand.findIndex(c => c.idx === card.idx);
									if (slot < 0) continue;
									self.revealedHand[slot] = {
										id: card.id,
										known: true,
										rank: card.rank,
										suit: card.suit,
										value: card.value,
										idx: card.idx
									};
									state.seenFaces[card.id] = { id: card.id, rank: card.rank, suit: card.suit, value: card.value, idx: card.idx, ownerId: selfPlayerId ?? undefined };
								}
							}
							break;
						}

						case 'game_player_turn':
							if (state.gameState) {
								state.gameState.currentPlayerId = payload.user?.id;
								state.pendingAction = null; // New turn clears pending actions
								state.displayedDrawnCard = null; // Clear magnified card
								// Update isCurrentTurn for all players
								state.gameState.players.forEach(p => {
									p.isCurrentTurn = (p.playerId === payload.user?.id);
								});
								// Turn number, deadline + clock skew (cambia-488). turn/turnDeadline/serverNow
								// live under payload.payload (the service's GameEvent.Payload map, built in
								// engine_adapter.go broadcastPlayerTurnEngine), not at the top level of the
								// envelope's payload, which is the full GameEvent object. turn was being read
								// off the envelope, so turnId stayed undefined and the table's Turn readout
								// never rendered (cambia-876, DL-4 review F7).
								const turnPayload = payload.payload;
								const turnNo = turnPayload?.turn ?? payload.turn;
								if (typeof turnNo === 'number') state.gameState.turnId = turnNo;
								if (turnPayload && typeof turnPayload.serverNow === 'number') {
									state.serverClockOffsetMs = turnPayload.serverNow - Date.now();
								}
								state.gameState.turnDeadline =
									(turnPayload && typeof turnPayload.turnDeadline === 'number')
										? turnPayload.turnDeadline
										: null;
							}
							break;

						case 'player_draw_stockpile':
						case 'private_draw_stockpile': // Treat both similarly for state update, but display logic differs
							if (state.gameState) {
								if (type === 'player_draw_stockpile') {
									// Public draw - update stockpile size, show card back magnified for others
									if (payload.payload?.source === 'stockpile') {
										state.gameState.stockpileSize = payload.payload?.stockpileSize ?? state.gameState.stockpileSize - 1;
									} else {
										state.gameState.discardSize = payload.payload?.discardSize ?? state.gameState.discardSize - 1;
										// Update discardTop if drawn from discard
										state.gameState.discardTop = null; // Simplified: assume next sync will fix it
									}
									const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
									const self = state.gameState.players.find(p => p.playerId === selfPlayerId);
									if (player && self && player.playerId !== self.playerId) {
										// Magnify card back for others
										state.displayedDrawnCard = { id: payload.card?.id, known: false };
										// Hold the id on the player too: a replace moves it into their hand slot
										// (player_discard below), and abilities target opponent cards by id.
										if (payload.card?.id) player.drawnCard = { id: payload.card.id, known: false };
									}
								} else { // Private draw
									// Update the specific player's drawnCard state
									const player = state.gameState.players.find(p => p.playerId === selfPlayerId); // Find 'self'
									if (player && player.playerId === state.gameState.currentPlayerId) {
										player.drawnCard = payload.card;
										state.pendingAction = 'discard_replace'; // Player must now discard/replace
										state.displayedDrawnCard = payload.card; // Magnify revealed card for self
										if (payload.card?.id && payload.card.rank) {
											state.seenFaces[payload.card.id] = { id: payload.card.id, rank: payload.card.rank, suit: payload.card.suit, value: payload.card.value, ownerId: selfPlayerId ?? undefined };
										}
									}
									// Update stockpile/discard size based on source
									if (payload.payload?.source === 'stockpile') {
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
								state.gameState.discardTop = payload.card;
								// Clear drawn card for the discarding player. A replace carries the slot the
								// discarded card left (card.idx, see the engine adapter's replace branch); the
								// drawn card takes that slot, with its face for the own hand and as an id
								// reference for an opponent's. No sync follows a replace, so without this the
								// slot keeps showing (and targeting) the card that just hit the discard pile
								// (cambia-848 F3). A plain discard carries no idx and leaves the hand alone.
								const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
								if (player) {
									const idx = payload.card?.idx;
									const drawn = player.drawnCard;
									if (typeof idx === 'number' && drawn?.id && player.revealedHand && idx >= 0 && idx < player.revealedHand.length) {
										const face = player.playerId === selfPlayerId ? (drawn.rank ? drawn : state.seenFaces[drawn.id]) : undefined;
										player.revealedHand[idx] = face
											? { id: drawn.id, known: true, rank: face.rank, suit: face.suit, value: face.value, idx }
											: { id: drawn.id, known: false, idx };
									}
									player.drawnCard = null;
								}
								state.displayedDrawnCard = null; // Clear magnified card
								state.pendingAction = null; // Action completed (unless special triggered)
							}
							break;

						case 'player_special_choice':
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === selfPlayerId); // Find 'self'
								if (player && player.playerId === payload.user?.id) {
									state.pendingAction = 'special_action'; // Player needs to make choice
								}
								// Optionally store special action details
								state.gameState.specialAction = {
									active: true,
									playerId: payload.user?.id,
									cardRank: payload.card?.rank
								};
							}
							break;

						case 'player_special_action': // Public confirmation/info
							if (state.gameState) {
								// If the action was 'skip' or a completed swap/peek, clear pending state
								if (payload.special === 'skip' || payload.special === 'peek_self' || payload.special === 'peek_other' || payload.special === 'swap_blind' || payload.special === 'swap_peek_swap') {
									state.pendingAction = null;
									state.gameState.specialAction = null;
								}
								// If it was swap_peek_reveal, the action is still pending (waiting for swap/skip)
								// UI might update based on card1/card2 info (e.g., highlight targets)
								if (payload.special === 'swap_blind' || payload.special === 'swap_peek_swap') {
									// The event carries the post-swap slots: card1 is the actor's, card2 the
									// target's, each with the id now sitting there. No sync follows a swap,
									// so move the ids in both hands here or a later snap or ability would
									// target the card that left, and a reveal keyed by id would land on the
									// wrong slot. An own slot keeps a face only when this client has seen it
									// (a King look, or an earlier peek of that card); opponent slots are id
									// references and stay face down (cambia-848 F3).
									for (const c of [payload.card1, payload.card2] as (EventCard | undefined)[]) {
										if (!c?.user?.id || typeof c.idx !== 'number') continue;
										const owner = state.gameState.players.find(p => p.playerId === c.user!.id);
										if (!owner?.revealedHand || c.idx < 0 || c.idx >= owner.revealedHand.length) continue;
										const face = owner.playerId === selfPlayerId ? state.seenFaces[c.id] : undefined;
										owner.revealedHand[c.idx] = face
											? { id: c.id, known: true, rank: face.rank, suit: face.suit, value: face.value, idx: c.idx }
											: { id: c.id, known: false, idx: c.idx };
									}
								}
							}
							break;

						// --- Private Events ---
						case 'private_special_action_success': {
							// The looked-at faces: card1 for a 7/8 or 9/T peek, card1 (own) and card2
							// (opponent) for a King look. See the seenFaces note on the state shape.
							const revealed = ([payload.card1, payload.card2] as (EventCard | undefined)[])
								.filter((c): c is EventCard => !!c && !!c.id)
								.map((c): RevealedCard => ({ id: c.id, rank: c.rank, suit: c.suit, value: c.value, idx: c.idx, ownerId: c.user?.id }));
							for (const c of revealed) state.seenFaces[c.id] = c;
							state.abilityReveal = revealed.length > 0
								? { special: typeof payload.special === 'string' ? payload.special : '', at: Date.now(), cards: revealed }
								: null;
							const self = state.gameState?.players.find(p => p.playerId === selfPlayerId);
							if (self?.revealedHand) {
								for (const c of revealed) {
									if (c.ownerId !== selfPlayerId) continue;
									const slot = self.revealedHand.findIndex(h => h.id === c.id);
									if (slot < 0) continue;
									self.revealedHand[slot] = { ...self.revealedHand[slot], known: true, rank: c.rank, suit: c.suit, value: c.value };
								}
							}
							break;
						}
						case 'private_special_action_fail':
							// Show error message to the user
							// state.error = `Special action failed: ${payload.message}`; // Maybe too aggressive?
							state.pendingAction = 'special_action'; // Allow retry or skip
							state.isProcessingAction = false; // Allow sending new action
							break;

						// --- Snap Events ---
						case 'player_snap_success':
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
								if (player) {
									player.handSize--; // Update hand size
									// Remove card from revealedHand if it's 'self'
									if (player.revealedHand && payload.card?.idx !== undefined) {
										player.revealedHand.splice(payload.card.idx, 1);
										// Adjust indices of subsequent cards
										for (let i = payload.card.idx; i < player.revealedHand.length; i++) {
											if (player.revealedHand[i].idx !== undefined) {
												player.revealedHand[i].idx!--;
											}
										}
									}
								}
								state.gameState.discardSize++;
								state.gameState.discardTop = payload.card;
							}
							break;
						case 'player_snap_fail':
							// No immediate state change needed from public fail event
							break;
						case 'player_snap_penalty': // Public penalty draw notification
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
								if (player) {
									player.handSize++; // Increment hand size
								}
								// Pile sizes are the server's post-draw counts (cambia-821), the same shape
								// game_reshuffle_stockpile carries, so apply them rather than subtracting.
								// A penalty draw is not always one card off the stockpile: the engine can
								// reshuffle the discard pile back in mid-penalty, which grows the stockpile
								// and empties the discard, and neither is something a client can derive from
								// the event alone.
								if (typeof payload.payload?.stockpileSize === 'number') {
									state.gameState.stockpileSize = payload.payload.stockpileSize;
								}
								if (typeof payload.payload?.discardSize === 'number') {
									state.gameState.discardSize = payload.payload.discardSize;
								}
							}
							break;
						case 'private_snap_penalty': // Private penalty card details
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === selfPlayerId); // Find 'self'
								if (player && player.playerId === state.gameState?.players.find(p => p.playerId === payload.card?.id)?.playerId) { // Check if message is for self? No, need user ID
									// This logic is tricky without user id in the private message.
									// Assuming it's for 'self' if received.
									if (player.revealedHand && payload.card?.idx !== undefined) {
										// Add card at the specified index
										player.revealedHand.splice(payload.card.idx, 0, payload.card);
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
								state.gameState.cambiaCallerId = payload.user?.id;
								const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
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
							}
							// Final scores/winner live under the nested GameEvent payload (the service
							// wraps { type, payload: {...} } and the hub re-wraps that as the envelope
							// payload), matching the payload.payload convention used elsewhere in this
							// switch (e.g. player_draw_stockpile's source/stockpileSize).
							state.finalScores = payload.payload?.scores ?? null;
							state.winnerId = payload.payload?.winner ?? null;
							break;

						case 'error': // Server-sent error message
							state.error = `Game Error: ${payload.message ?? payload.error ?? 'Unknown error'}`;
							state.isProcessingAction = false; // Allow new actions after error
							// Should we clear pending state on error? Depends on the error.
							break;

						case 'game_started': {
							// Plain Hub.Emit map (game_id, players at the top level of payload, no
							// GameEvent wrapper - see hub.createAndStartGame). Fires once per game AND
							// again for every subsequent round in multi-round/ranked matches
							// (hub.startNextRound re-invokes createAndStartGame) while this store still
							// holds the previous round's terminal gameState. Reset it so LobbyPage's
							// `!gameState` guard renders a loading state instead of flashing the finished
							// round's game-over UI until the new round's private_sync_state /
							// private_initial_cards arrive.
							state.gameState = null;
							state.pendingAction = null;
							state.displayedDrawnCard = null;
							state.finalScores = null;
							state.winnerId = null;
							state.seenFaces = {};
							state.abilityReveal = null;
							state.isLoading = true;
							state.isProcessingAction = false;
							break;
						}

						// --- Historian-only action markers: never delivered over WebSocket ---
						// These strings are logAction()'d to the historian/Redis pipeline (service
						// game.go / engine_adapter.go) for the game_actions replay table. That pipeline
						// is unrelated to the Emitter/fireEvent path real WS events use, so despite
						// sharing the game_/player_ naming convention with actual GameEventType members,
						// none of these can ever actually reach this switch. Everything they would
						// represent already arrives via events already handled above: players[].connected
						// via private_sync_state (player_add/disconnect/reconnect/reconnect_fail),
						// turn-advance/forced-discard via game_player_turn/player_discard
						// (player_timeout/player_timeout_discard), and per-card penalty details via
						// player_snap_penalty/private_snap_penalty (player_snap_penalty_applied).
						// Documented no-op instead of falling to the default warn.
						case 'game_pregame_start':
						case 'player_add':
						case 'player_disconnect':
						case 'player_reconnect':
						case 'player_reconnect_fail':
						case 'player_timeout':
						case 'player_timeout_discard':
						case 'player_snap_penalty_applied':
							break;

						// game_reshuffle_stockpile (EventGameReshuffleStockpile): fired when a stockpile
						// draw starts with an empty stockpile, forcing the engine to reshuffle the
						// discard pile back into the stockpile first (cambia-763 F3). The counts sent
						// here are already post-reshuffle-and-draw (server engine state), so apply them
						// directly rather than incrementally: without this, discardSize in particular
						// would drift (the plain draw-from-stockpile handling below only ever updates
						// stockpileSize) until the next full sync_state.
						case 'game_reshuffle_stockpile':
							if (state.gameState) {
								if (typeof payload.payload?.stockpileSize === 'number') {
									state.gameState.stockpileSize = payload.payload.stockpileSize;
								}
								if (typeof payload.payload?.discardSize === 'number') {
									state.gameState.discardSize = payload.payload.discardSize;
								}
							}
							break;

						// game_results duplicates game_end's winner/scores (both derived from the same
						// adjustedScores computed once in CambiaGame.endGame, game.go); finalScores/winnerId
						// are already set by game_end, so this store no-ops on it. Its lobby_status
						// snapshot is lobbyStore's domain and is dual-routed there by useSocket.ts
						// (cambia-763 F2).
						case 'game_results':
							break;

						default:
							console.warn(`[GameStore] Unhandled WebSocket message type: ${type}`);
					}
				} catch (e: unknown) {
					console.error(`[GameStore] Error processing message type ${type}:`, e);
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
export const selectServerClockOffsetMs = (state: GameState) => state.serverClockOffsetMs;
export const selectAbilityReveal = (state: GameState) => state.abilityReveal;
export const selectFinalScores = (state: GameState) => state.finalScores;
export const selectCurrentPlayerId = (state: GameState) => state.gameState?.currentPlayerId;
export const selectSelfPlayerState = (state: GameState) => {
	const selfId = useAuthStore.getState().user?.id;
	return state.gameState?.players.find(p => p.playerId === selfId);
};
export const selectIsSelfTurn = (state: GameState) => {
	const self = selectSelfPlayerState(state);
	const currentId = selectCurrentPlayerId(state);
	return self?.playerId === currentId && !!state.gameState?.started && !state.gameState?.gameOver;
};
