/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/gameStore.ts
import { create } from 'zustand';
import type { ObfGameState, ObfCard, EventCard, FinalHand } from '@/types/game';
import { immer } from 'zustand/middleware/immer';
import { useAuthStore } from './authStore';
import { applySnapSuccess } from '@/lib/snapSuccess';
import { applySnapMove } from '@/lib/snapFill';
import { applyPregamePeek, nextPregamePeek } from '@/lib/pregamePeek';
import { applyDrawPileCounts, pendingActionAfterFail, pendingActionForSpecial } from '@/lib/specialPrompt';

/** A face shown to this client by an event: an ability look, a pregame peek, a card drawn in. */
export interface RevealedCard {
	id: string;
	rank?: string;
	suit?: string;
	value?: number;
	idx?: number;
	ownerId?: string;
}

/**
 * The most recent transient reveal, kept so the table can show the faces for the window they are
 * shown in and no longer. Since cambia-1094 this covers own cards too: an ability look at your own
 * card and the card you take from a draw are shown the same way an opponent's peeked card is, held
 * for a beat and then turned back down. `special` names what caused it - the ability's own string
 * for an ability, 'replace' for a card taken in from a draw.
 */
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
	// The round-end reveal (RULES.md 3C, cambia-1542): every scored seat's hand as the round
	// ended. Carried by game_end and by the game_results a client reconnecting into the results
	// is answered with, so the results view has the faces whether it watched the round end or
	// reloaded into it. A forfeited seat is absent, since it is not scored.
	finalHands: FinalHand[] | null;
	// The latest transient reveal, for the table's temporary display (cambia-848 F3, widened by
	// cambia-1094). No face this client is shown is durable any more: sync_state hides every own
	// hand slot in every phase, exactly as it hides every opponent slot, because the physical game
	// turns your pregame peek face-down at the start and leaves you to play on memory. So an
	// ability look (own or opponent) and the card taken in from a draw are all held here for their
	// window and then dropped, rather than being folded into revealedHand for the rest of the round.
	abilityReveal: AbilityReveal | null;
	// The pregame peek, held for the length of the pregame window only. private_initial_cards is
	// the sole carrier of these faces, and any sync during the window (a peer dropping, a repair)
	// replaces the board with one that has them face-down, so they are re-applied to each snapshot
	// while preGameActive rather than being read back off it. Cleared by the sync that ends the
	// pregame phase, which is what turns the peeked cards down at game start (cambia-1094).
	pregamePeek: RevealedCard[];
	// Bumped once per outbound action the hub discarded on its staleness gate that the client
	// could not safely resend (cambia-891). The table watches the number and shows a notice; it
	// carries no text so the surface owns the copy.
	droppedActionNonce: number;
	// The most recent successful snap: who snapped, and whose hand the card left. The two differ
	// on an opponent snap, and a hand-size delta cannot tell them apart, so the table reads this
	// rather than inferring the actor from the seat that shrank (cambia-913). Ids only; the
	// surface owns the copy.
	lastSnap: { nonce: number; snapperId: string | null; ownerId: string | null } | null;
	// The fill this client owes after snapping an opponent's card (RULES.md 5, cambia-936): the
	// victim whose slot it fills, that slot, and the epoch-ms deadline the server settles it at
	// (null on a table with no turn timer). Held apart from pendingAction because the obligation
	// outlives the events that clear that field: it is owed out of turn, so a turn change or a
	// discard by anyone else would wipe the prompt while the server still refuses everything else
	// this client sends.
	pendingSnapMove: { victimId: string; slot: number; deadline: number | null } | null;
	// The most recent fill that landed: who gave the card, who received it, and whether the server
	// chose it when the deadline passed. The table needs it for two things: the notice, and to know
	// that this particular hand growing by one is a card being paid, not a snap penalty being drawn
	// (which is what every other mid-game hand growth is).
	lastSnapMove: { nonce: number; snapperId: string | null; victimId: string | null; auto: boolean } | null;
	// The most recent seat-presence change: a socket dropped and its seat is being held
	// ('reconnecting'), the player came back ('reconnected'), or the window closed and they
	// forfeited ('forfeited') - cambia-955. Ids and a deadline only; the surface owns the copy.
	lastPresence: { nonce: number; kind: 'reconnecting' | 'reconnected' | 'forfeited'; playerId: string; deadline: number | null } | null;
}

interface GameActions {
	setGameId: (id: string | null) => void;
	setConnected: (status: boolean) => void;
	setLoading: (loading: boolean) => void;
	setError: (error: string | null) => void;
	clearError: () => void;
	processGameWebSocketMessage: (type: string, payload: any) => void;
	forceSync: (payload: any) => void;
	noteDroppedAction: () => void;
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
	finalHands: null,
	abilityReveal: null,
	pregamePeek: [],
	droppedActionNonce: 0,
	lastSnap: null,
	pendingSnapMove: null,
	lastSnapMove: null,
	lastPresence: null
};

/**
 * The round-end reveal off a results frame's `finalHands` (RULES.md 3C, cambia-1542). Returns null
 * for anything that is not a list of hands, so a frame from an older server leaves the state alone
 * rather than blanking a reveal an earlier frame already carried. Cards without a rank are dropped:
 * the results render faces, and a faceless entry would draw as a back beside real ones.
 */
function readFinalHands(raw: unknown): FinalHand[] | null {
	if (!Array.isArray(raw)) return null;
	const hands: FinalHand[] = [];
	for (const entry of raw) {
		if (!entry || typeof entry !== 'object') continue;
		const hand = entry as Partial<FinalHand>;
		if (typeof hand.playerId !== 'string' || !Array.isArray(hand.cards)) continue;
		hands.push({
			playerId: hand.playerId,
			cards: hand.cards.filter((c) => !!c && typeof c.rank === 'string' && c.rank.length > 0)
		});
	}
	return hands;
}

/** The fill `selfId` still owes, read out of a state snapshot's snapMoves (cambia-936). */
function ownSnapMove(gs: ObfGameState | null | undefined, selfId: string | null) {
	if (!gs || !selfId || gs.gameOver) return null;
	const mine = (gs.snapMoves ?? []).find((m) => m.snapperId === selfId);
	return mine ? { victimId: mine.victimId, slot: mine.slot, deadline: mine.deadline ?? null } : null;
}

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
					state.abilityReveal = null;
					state.pregamePeek = [];
					state.pendingSnapMove = null;
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
					// Only a payload that actually replaced the board may clear what the board
					// was showing. The hub's sync_state is a lobby snapshot plus a seq (see
					// hub.go sendSyncState -> buildLobbySnapshot); the game side resyncs through
					// private_sync_state. Clearing unconditionally wiped the drawn card and the
					// pending ability of a player whose only sin was sending a frame during a
					// broadcast, leaving a turn that could not be finished (cambia-891).
					state.pendingAction = null;
					state.displayedDrawnCard = null;
					state.abilityReveal = null;
					// A snap fill is not cleared, it is re-read: the obligation lives on the server
					// and a repair that replaced the board carries whatever is still owed
					// (cambia-936).
					const selfId = useAuthStore.getState().user?.id ?? null;
					state.pendingSnapMove = ownSnapMove(payload.state, selfId);
					// A repair that lands mid-peek must not end the peek early (cambia-1094). Only
					// a positive preGameActive re-applies: this payload can be the hub's lobby
					// snapshot, which carries no phase at all, and that is not evidence the window
					// closed. The phase transition is settled by private_sync_state below.
					if (payload.state.preGameActive === true) {
						applyPregamePeek(state.gameState, state.pregamePeek, selfId);
					}
				}
				if (typeof payload?.seq === 'number') {
					state.seq = payload.seq;
				}
				state.isLoading = false;
				state.isConnected = true;
				state.error = null;
			});
		},

		noteDroppedAction: () => {
			set((state) => {
				state.droppedActionNonce += 1;
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

				// game_started is exempt alongside private_sync_state: the hub emits it before
				// BeginPreGame's first private_sync_state on every round, including the first of a
				// session, when state.gameState is still null (hub.go createAndStartGame comment:
				// "game_started precedes BeginPreGame"). Its case below only resets fields and never
				// reads state.gameState first, so letting it through here is safe regardless of
				// whether a prior sync has landed; dropping it here logged this warning on every
				// game start instead of only a genuinely out-of-order message (cambia-958 D6).
				if (!state.gameState && type !== 'private_sync_state' && type !== 'game_started') {
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
							// The snap fill is server state, so the snapshot is its authority both
							// ways: it restores a prompt this client lost and drops one the server
							// has already settled (cambia-936).
							state.pendingSnapMove = ownSnapMove(payload.state, selfPlayerId);
							// Recompute clock skew from this snapshot's serverNow (cambia-488).
							if (typeof payload.state?.serverNow === 'number') {
								state.serverClockOffsetMs = payload.state.serverNow - Date.now();
							}
							// The snapshot hides every own card (cambia-1094), so the pregame peek has
							// to be put back on for as long as the window lasts, and dropped the moment
							// it closes. The sync StartGame broadcasts is what turns the peeked cards
							// down on screen: it is the first snapshot with preGameActive false.
							state.pregamePeek = nextPregamePeek(state.pregamePeek, payload.state);
							if (payload.state?.preGameActive) {
								applyPregamePeek(state.gameState, state.pregamePeek, selfPlayerId);
							}
							// Determine pending action based on new state
							const gs = state.gameState;
							if (gs) {
								const userState = gs.players.find(p => p.playerId === selfPlayerId); // Find 'self'
								if (userState?.drawnCard && gs.currentPlayerId === userState.playerId && !gs.gameOver && gs.started) {
									state.pendingAction = 'discard_replace';
								} else if (!gs.gameOver && gs.started) {
									// The service's ObfGameState (service/internal/game/sync_state.go) serializes
									// SpecialActionState into private_sync_state (cambia-763 F1), so a client that
									// resyncs mid-action (reconnect, tab refresh) restores pendingAction here.
									state.pendingAction = pendingActionForSpecial(gs.specialAction, selfPlayerId);
								}
							}
							break;
						}

						case 'private_initial_cards': {
							// Pregame peek reveal, own hand only. The event carries one entry per peeked
							// slot under `cards` (cambia-817); the count is the initialViewCount house
							// rule, up to cardsPerPlayer, so nothing here may assume two.
							//
							// This is the ONLY frame that ever carries an own face for these cards: the
							// server hides every own hand slot in every sync_state (cambia-1094). So the
							// faces are held in pregamePeek and re-applied to each snapshot that lands
							// during the window, rather than written once and read back. The service
							// re-fires the event to a player who reconnects mid-window, which lands here
							// and simply replaces the held peek with the same faces.
							const cards = Array.isArray(payload.cards) ? (payload.cards as EventCard[]) : [];
							state.pregamePeek = cards
								.filter((c): c is EventCard => !!c && !!c.id)
								.map((c): RevealedCard => ({ id: c.id, rank: c.rank, suit: c.suit, value: c.value, idx: c.idx, ownerId: selfPlayerId ?? undefined }));
							applyPregamePeek(state.gameState, state.pregamePeek, selfPlayerId);
							break;
						}

						case 'game_player_turn':
							if (state.gameState) {
								state.gameState.currentPlayerId = payload.user?.id;
								state.pendingAction = null; // New turn clears pending actions
								// The server never announces a turn over a pending ability (the King's two-step
								// runs without one), so a special action that survives this event is stale and
								// would keep rendering a prompt for an ability nobody owes (cambia-1125).
								state.gameState.specialAction = null;
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

						// The turn clock alone, for a re-arm that moved the deadline without
						// starting a new turn: the ability prompt a discard opens, a King's second
						// step, a timeout that handed the window back, a reconnect onto a turn that
						// had no clock. Those all used to move the deadline and emit nothing, so
						// TimerBar kept counting down to the deadline it was last given and hit
						// 0:00 while the server still held a full window open (cambia-1556).
						//
						// Deliberately narrower than game_player_turn: this says nothing about
						// whose turn it is, so it touches neither currentPlayerId nor
						// specialAction. Clearing the prompt here would wipe the very prompt most
						// of these re-arms exist to give the player time to answer.
						case 'game_turn_deadline': {
							if (state.gameState) {
								const deadlinePayload = payload.payload;
								if (typeof deadlinePayload?.turn === 'number') {
									state.gameState.turnId = deadlinePayload.turn;
								}
								if (typeof deadlinePayload?.serverNow === 'number') {
									state.serverClockOffsetMs = deadlinePayload.serverNow - Date.now();
								}
								state.gameState.turnDeadline =
									(deadlinePayload && typeof deadlinePayload.turnDeadline === 'number')
										? deadlinePayload.turnDeadline
										: null;
							}
							break;
						}

						case 'player_draw_stockpile':
						case 'private_draw_stockpile': // Treat both similarly for state update, but display logic differs
							if (state.gameState) {
								// The counts are the public event's alone; the private twin that follows it is
								// the same draw seen a second time (cambia-1125, lib/specialPrompt.ts).
								applyDrawPileCounts(state.gameState, type, payload.payload);
								if (type === 'player_draw_stockpile') {
									// Public draw - show card back magnified for others
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
								// drawn card takes that slot as an id reference. No sync follows a replace, so
								// without this the slot keeps showing (and targeting) the card that just hit
								// the discard pile (cambia-848 F3). A plain discard carries no idx and leaves
								// the hand alone.
								//
								// The slot goes down face-DOWN even for the player who drew it (cambia-1094):
								// you saw the card as you put it in and then it is one more card you have to
								// remember. That look is the transient reveal below, on the same hold an
								// opponent's peeked face gets.
								const player = state.gameState.players.find(p => p.playerId === payload.user?.id);
								if (player) {
									const idx = payload.card?.idx;
									const drawn = player.drawnCard;
									if (typeof idx === 'number' && drawn?.id && player.revealedHand && idx >= 0 && idx < player.revealedHand.length) {
										player.revealedHand[idx] = { id: drawn.id, known: false, idx };
										if (player.playerId === selfPlayerId && drawn.rank) {
											state.abilityReveal = {
												special: 'replace',
												at: Date.now(),
												cards: [{ id: drawn.id, rank: drawn.rank, suit: drawn.suit, value: drawn.value, idx, ownerId: selfPlayerId ?? undefined }]
											};
										}
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
								// Optionally store special action details. `mandatory` marks an ability the
								// engine armed off a replace, which cannot be declined, so the table asks for
								// a target instead of offering a skip the server would refuse (cambia-1125).
								state.gameState.specialAction = {
									active: true,
									playerId: payload.user?.id,
									cardRank: payload.card?.rank,
									mandatory: payload.payload?.mandatory === true
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
									// wrong slot. Every slot on both sides is an id reference and stays face
									// down: no own card is ever persistently face-up (cambia-1094), and a
									// King's look at the card it is about to move is shown by the transient
									// reveal, not by the slot it lands in (cambia-848 F3).
									for (const c of [payload.card1, payload.card2] as (EventCard | undefined)[]) {
										if (!c?.user?.id || typeof c.idx !== 'number') continue;
										const owner = state.gameState.players.find(p => p.playerId === c.user!.id);
										if (!owner?.revealedHand || c.idx < 0 || c.idx >= owner.revealedHand.length) continue;
										owner.revealedHand[c.idx] = { id: c.id, known: false, idx: c.idx };
									}
								}
							}
							break;

						// --- Private Events ---
						case 'private_special_action_success': {
							// The looked-at faces: card1 for a 7/8 or 9/T peek, card1 (own) and card2
							// (opponent) for a King look. Held for the reveal window and nothing more -
							// an own card the actor just looked at goes back down with the opponent's
							// (cambia-1094). See the abilityReveal note on the state shape.
							const revealed = ([payload.card1, payload.card2] as (EventCard | undefined)[])
								.filter((c): c is EventCard => !!c && !!c.id)
								.map((c): RevealedCard => ({ id: c.id, rank: c.rank, suit: c.suit, value: c.value, idx: c.idx, ownerId: c.user?.id }));
							state.abilityReveal = revealed.length > 0
								? { special: typeof payload.special === 'string' ? payload.special : '', at: Date.now(), cards: revealed }
								: null;
							break;
						}
						case 'private_special_action_fail':
							// Show error message to the user
							// state.error = `Special action failed: ${payload.message}`; // Maybe too aggressive?
							// The server sends this for ANY refused action, not only a refused special one, so
							// it restores the prompt only when one is genuinely pending and never invents one:
							// a plain refusal used to leave a rankless 'special_action' on a table that had no
							// ability to skip, which is how the dead Skip button got on screen (cambia-1125).
							state.pendingAction = pendingActionAfterFail(state.pendingAction, state.gameState?.specialAction, selfPlayerId);
							state.isProcessingAction = false; // Allow sending new action
							break;

						// --- Snap Events ---
						case 'player_snap_success':
							// The card leaves its OWNER's hand (payload.card.user), which is the snapper
							// (payload.user) only when someone snapped their own card. Applying it to the
							// snapper shrank the wrong seat on every screen after an opponent snap
							// (cambia-913); see lib/snapSuccess.ts for the rest of the reasoning.
							if (state.gameState) {
								const ownerId = applySnapSuccess(state.gameState, payload);
								state.lastSnap = {
									nonce: (state.lastSnap?.nonce ?? 0) + 1,
									snapperId: payload.user?.id ?? null,
									ownerId
								};
							}
							break;
						case 'player_snap_move_required':
							// The snapper owes the victim a card back (RULES.md 5, cambia-936). Only
							// the snapper can pay it, so only their client prompts; every other seat
							// learns it from the move itself.
							if (payload.user?.id === selfPlayerId) {
								state.pendingSnapMove = {
									victimId: payload.card?.user?.id ?? '',
									slot: typeof payload.card?.idx === 'number' ? payload.card.idx : 0,
									deadline: typeof payload.payload?.deadline === 'number' ? payload.payload.deadline : null
								};
							}
							break;
						case 'player_snap_move':
							// The card changes hands face down: applySnapMove moves the id and both
							// hand sizes, and no face is shown to anyone (see lib/snapFill.ts).
							if (state.gameState) {
								const parties = applySnapMove(state.gameState, payload);
								state.lastSnapMove = {
									nonce: (state.lastSnapMove?.nonce ?? 0) + 1,
									snapperId: parties.snapperId,
									victimId: parties.victimId,
									auto: payload.payload?.auto === true
								};
								if (payload.user?.id === selfPlayerId) state.pendingSnapMove = null;
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
							// The server drops every unpaid fill when the game ends (endGame ->
							// cancelSnapFills), so the prompt goes with it (cambia-936).
							state.pendingSnapMove = null;
							// Final scores/winner live under the nested GameEvent payload (the service
							// wraps { type, payload: {...} } and the hub re-wraps that as the envelope
							// payload), matching the payload.payload convention used elsewhere in this
							// switch (e.g. player_draw_stockpile's source/stockpileSize).
							state.finalScores = payload.payload?.scores ?? null;
							state.winnerId = payload.payload?.winner ?? null;
							// The round-end reveal (RULES.md 3C, cambia-1542). Every scored seat's
							// hand, so the results can show what each player was holding.
							state.finalHands = readFinalHands(payload.payload?.finalHands);
							break;

						// --- Seat presence (cambia-955) ---
						// A dropped socket now holds its seat for the reconnect grace instead of
						// forfeiting on the spot, so the table has three states to tell apart, not
						// two. The seat fields also arrive on the next private_sync_state; these
						// events are what let the surface react at the moment it happens (and carry
						// the deadline a countdown needs).
						case 'player_reconnecting':
						case 'player_reconnected':
						case 'player_forfeited': {
							const playerId = payload.user?.id;
							if (!playerId) break;
							const kind = type === 'player_reconnecting' ? 'reconnecting'
								: type === 'player_reconnected' ? 'reconnected' : 'forfeited';
							const deadline = typeof payload.payload?.deadline === 'number' ? payload.payload.deadline : null;
							if (state.gameState) {
								const player = state.gameState.players.find(p => p.playerId === playerId);
								if (player) {
									player.connected = kind === 'reconnected';
									player.forfeited = kind === 'forfeited';
									player.reconnectDeadline = kind === 'reconnecting' ? deadline : null;
								}
							}
							state.lastPresence = {
								nonce: (state.lastPresence?.nonce ?? 0) + 1,
								kind,
								playerId,
								deadline
							};
							break;
						}

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
							state.finalHands = null;
							state.abilityReveal = null;
							state.pregamePeek = [];
							state.lastPresence = null;
							state.pendingSnapMove = null;
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
						// adjustedScores computed once in CambiaGame.endGame, game.go), and its
						// lobby_status snapshot is lobbyStore's domain, dual-routed there by
						// useSocket.ts (cambia-763 F2). It used to be a no-op here because game_end
						// had always arrived first. It cannot be any more: a client that reloads into
						// a finished game never saw game_end, and the hub answers its reconnect with
						// this frame alone (cambia-955), so the scores it carries are the only ones
						// the results screen will ever get. Applied unconditionally, since both
						// frames carry the same numbers.
						case 'game_results':
							if (payload.scores && typeof payload.scores === 'object') {
								state.finalScores = payload.scores;
							}
							if (typeof payload.winner === 'string') {
								state.winnerId = payload.winner;
							}
							// The reveal rides this frame too, and for a client that reloaded into
							// the results this is the only copy of it that will ever arrive: the
							// service drops the game from its store right after emitting it, so
							// game_end is long gone (cambia-1542).
							{
								const hands = readFinalHands(payload.finalHands);
								if (hands) state.finalHands = hands;
							}
							if (state.gameState) {
								state.gameState.gameOver = true;
								state.gameState.currentPlayerId = null;
							}
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
export const selectDroppedActionNonce = (state: GameState) => state.droppedActionNonce;
export const selectLastSnap = (state: GameState) => state.lastSnap;
export const selectPendingSnapMove = (state: GameState) => state.pendingSnapMove;
export const selectLastSnapMove = (state: GameState) => state.lastSnapMove;
export const selectLastPresence = (state: GameState) => state.lastPresence;
export const selectFinalScores = (state: GameState) => state.finalScores;
export const selectFinalHands = (state: GameState) => state.finalHands;
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
