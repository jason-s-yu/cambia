/* eslint-disable @typescript-eslint/no-explicit-any */
// src/types/game.ts

import type { HouseRules } from './index';
import { v4 as uuidv4 } from 'uuid'; // Import UUID generator

/** Represents the potentially obfuscated state of a single card */
export interface ObfCard {
	id: string;
	known: boolean; // If true, rank/suit/value are revealed to the requesting client
	rank?: string;
	suit?: string;
	value?: number;
	idx?: number; // Index in hand, relevant for some actions
}

/** Represents the potentially obfuscated state of a single player */
export interface ObfPlayerState {
	playerId: string;
	username: string;
	handSize: number;
	hasCalledCambia: boolean;
	connected: boolean;
	isCurrentTurn: boolean;
	// Revealed only for the player requesting the state
	revealedHand?: ObfCard[];
	drawnCard?: ObfCard | null; // Card currently held after drawing
}

/** Represents the overall game state, potentially obfuscated for a specific client */
export interface ObfGameState {
	gameId: string;
	preGameActive: boolean;
	started: boolean;
	gameOver: boolean;
	currentPlayerId: string | null; // Null if no current player (e.g., before start)
	turnId: number;
	stockpileSize: number;
	discardSize: number;
	discardTop?: ObfCard | null; // Top card of discard pile (always known)
	players: ObfPlayerState[];
	cambiaCalled: boolean;
	cambiaCallerId?: string | null;
	houseRules: HouseRules;
	// Add specific fields for pending special actions if needed by UI
	specialAction?: {
		active: boolean;
		playerId: string;
		cardRank: string;
		// Add other fields based on spec (e.g., peeked card info for King)
	} | null;
}

// --- Action Payloads (Client -> Server) ---

/** Base structure for actions sent by the client */
export interface ClientGameAction {
	type: string;
	special?: string; // For action_special type
	card?: { id: string; idx?: number }; // Primary card for discard, replace, snap, peek
	card1?: { id: string; idx?: number; user?: { id: string } }; // For swap actions
	card2?: { id: string; idx?: number; user?: { id: string } }; // For swap actions
	payload?: any; // Generic payload if needed
}

// --- Action Constructors (Client -> Server) ---

export const drawStockpileAction = (): ClientGameAction => ({ type: 'action_draw_stockpile' });
export const drawDiscardPileAction = (): ClientGameAction => ({ type: 'action_draw_discardpile' });
export const discardAction = (cardId: string): ClientGameAction => ({ type: 'action_discard', card: { id: cardId } });
export const replaceAction = (cardIdToReplace: string, index: number): ClientGameAction => ({ type: 'action_replace', card: { id: cardIdToReplace, idx: index } });
export const snapAction = (cardId: string): ClientGameAction => ({ type: 'action_snap', card: { id: cardId } });
export const callCambiaAction = (): ClientGameAction => ({ type: 'action_cambia' });
export const skipSpecialAction = (): ClientGameAction => ({ type: 'action_special', special: 'skip' });

// Special action constructors (more complex, might need refinement based on UI)
export const peekSelfAction = (cardId: string, index: number): ClientGameAction => ({
	type: 'action_special',
	special: 'peek_self',
	card1: { id: cardId, idx: index } // Use card1 for single target
});

export const peekOtherAction = (cardId: string, index: number, targetUserId: string): ClientGameAction => ({
	type: 'action_special',
	special: 'peek_other',
	card1: { id: cardId, idx: index, user: { id: targetUserId } } // Use card1 for single target
});

export const blindSwapAction = (card1Id: string, idx1: number, owner1Id: string, card2Id: string, idx2: number, owner2Id: string): ClientGameAction => ({
	type: 'action_special',
	special: 'swap_blind',
	card1: { id: card1Id, idx: idx1, user: { id: owner1Id } },
	card2: { id: card2Id, idx: idx2, user: { id: owner2Id } }
});

// King actions require two steps
export const kingPeekAction = (card1Id: string, idx1: number, owner1Id: string, card2Id: string, idx2: number, owner2Id: string): ClientGameAction => ({
	type: 'action_special',
	special: 'swap_peek', // Initial peek step
	card1: { id: card1Id, idx: idx1, user: { id: owner1Id } },
	card2: { id: card2Id, idx: idx2, user: { id: owner2Id } }
});

export const kingSwapConfirmAction = (card1Id: string, idx1: number, owner1Id: string, card2Id: string, idx2: number, owner2Id: string): ClientGameAction => ({
	type: 'action_special',
	special: 'swap_peek_swap', // Confirmation swap step
	card1: { id: card1Id, idx: idx1, user: { id: owner1Id } },
	card2: { id: card2Id, idx: idx2, user: { id: owner2Id } }
});

// --- Event Payloads (Server -> Client) ---

/** Structure for player turn events */
export interface GamePlayerTurnEvent {
	type: 'game_player_turn';
	user: { id: string };
	turn: number;
}

/** Structure for private initial card reveal events */
export interface PrivateInitialCardsEvent {
	type: 'private_initial_cards';
	card1?: ObfCard | null;
	card2?: ObfCard | null;
}

/** Structure for full state sync events */
export interface PrivateSyncStateEvent {
	type: 'private_sync_state';
	state: ObfGameState;
}

/** Structure for player draw events (public) */
export interface PlayerDrawStockpileEvent {
	type: 'player_draw_stockpile';
	user: { id: string };
	card: { id: string }; // Only ID revealed publicly
	payload: {
		stockpileSize?: number;
		source: 'stockpile' | 'discardpile';
		discardSize?: number;
	};
}

/** Structure for private draw events */
export interface PrivateDrawStockpileEvent {
	type: 'private_draw_stockpile';
	card: ObfCard; // Full card details revealed
	payload: {
		source: 'stockpile' | 'discardpile';
	};
}

/** Structure for player discard events (public) */
export interface PlayerDiscardEvent {
	type: 'player_discard';
	user: { id: string };
	card: ObfCard; // Full details revealed on discard
}

/** Structure for player replace events (public) */
export interface PlayerReplaceEvent {
	type: 'player_replace'; // DEPRECATED? Server might just send player_discard
	user: { id: string };
	card: ObfCard; // Card being discarded (full details)
	// replacedWithCardId: string; // ID of the card that took its place in hand
}

/** Structure for special choice events (public) */
export interface PlayerSpecialChoiceEvent {
	type: 'player_special_choice';
	user: { id: string };
	card: { id: string; rank: string }; // ID and rank of the triggering card
	special: string; // e.g., 'peek_self', 'swap_blind'
}

/** Structure for special action events (public) */
export interface PlayerSpecialActionEvent {
	type: 'player_special_action';
	user: { id: string };
	special: string; // e.g., 'peek_self', 'swap_blind', 'skip'
	card1?: { id: string; idx?: number; user?: { id: string } }; // Obfuscated targets
	card2?: { id: string; idx?: number; user?: { id: string } };
}

/** Structure for private special action success events */
export interface PrivateSpecialActionSuccessEvent {
	type: 'private_special_action_success';
	special: string;
	card1?: ObfCard; // Revealed card details
	card2?: ObfCard;
}

/** Structure for private special action fail events */
export interface PrivateSpecialActionFailEvent {
	type: 'private_special_action_fail';
	special?: string;
	message: string;
	card1?: { id: string; idx?: number; user?: { id: string } }; // Original targets
	card2?: { id: string; idx?: number; user?: { id: string } };
}

/** Structure for player snap success events (public) */
export interface PlayerSnapSuccessEvent {
	type: 'player_snap_success';
	user: { id: string };
	card: ObfCard; // Card that was snapped (full details)
}

/** Structure for player snap fail events (public) */
export interface PlayerSnapFailEvent {
	type: 'player_snap_fail';
	user: { id: string };
	card?: ObfCard; // Attempted card (revealed details)
}

/** Structure for public penalty draw events */
export interface PlayerSnapPenaltyEvent {
	type: 'player_snap_penalty';
	user: { id: string };
	card: { id: string }; // Obfuscated card ID drawn
	payload: { count: number; total: number };
}

/** Structure for private penalty draw events */
export interface PrivateSnapPenaltyEvent {
	type: 'private_snap_penalty';
	card: ObfCard; // Revealed penalty card details
	payload: { count: number; total: number };
}

/** Structure for player call Cambia events (public) */
export interface PlayerCambiaEvent {
	type: 'player_cambia';
	user: { id: string };
}

/** Structure for game end events (public) */
export interface GameEndEvent {
	type: 'game_end';
	payload: {
		scores: Record<string, number>; // Map of playerID -> final score
		winner: string; // UUID string of the winner, or NIL UUID if tie/no winner
		caller: string; // UUID string of the player who called Cambia, or NIL UUID
		penaltyApplied: boolean; // Did the Cambia caller penalty get applied?
		winBonusApplied: boolean; // Was a circuit win bonus applied?
	};
}

/** Structure for generic error events (private) */
export interface GameErrorEvent {
	type: 'error';
	message: string;
	code?: string; // Optional error code (e.g., 'invalid_action')
}

// --- Union Type for Server Events ---

export type ServerGameEvent =
	| GamePlayerTurnEvent
	| PrivateInitialCardsEvent
	| PrivateSyncStateEvent
	| PlayerDrawStockpileEvent
	| PrivateDrawStockpileEvent
	| PlayerDiscardEvent
	| PlayerReplaceEvent
	| PlayerSpecialChoiceEvent
	| PlayerSpecialActionEvent
	| PrivateSpecialActionSuccessEvent
	| PrivateSpecialActionFailEvent
	| PlayerSnapSuccessEvent
	| PlayerSnapFailEvent
	| PlayerSnapPenaltyEvent
	| PrivateSnapPenaltyEvent
	| PlayerCambiaEvent
	| GameEndEvent
	| GameErrorEvent; // Add other event types here

// --- Utility Functions ---

/**
 * Helper to create a minimal card representation for actions.
 * @param card The full card object.
 * @param idx Optional index.
 * @returns A simplified object for action payloads.
 */
export function cardToActionPayload(card: ObfCard | null, idx?: number): { id: string; idx?: number } | undefined {
	if (!card) return undefined;
	return { id: card.id, ...(idx !== undefined && { idx }) };
}

/**
 * Helper to create a full card representation with owner.
 * @param card The card object.
 * @param ownerId The owner's UUID.
 * @returns An object suitable for card1/card2 in special actions.
 */
export function cardToSpecialActionPayload(card: ObfCard | null, ownerId: string): { id: string; idx?: number; user?: { id: string } } | undefined {
	if (!card) return undefined;
	return {
		id: card.id,
		...(card.idx !== undefined && { idx: card.idx }),
		...(ownerId && { user: { id: ownerId } })
	};
}