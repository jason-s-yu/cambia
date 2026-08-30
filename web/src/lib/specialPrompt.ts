// src/lib/specialPrompt.ts
// The client's model of a pending special action, and of the two piles a draw event moves
// (cambia-1125).
//
// Both halves of this file exist because the same fact was being written from two places at once.
//
// The prompt. A special action is on screen when the server says one is pending, and it leaves the
// screen when the server settles it or announces a new turn. The store used to move `pendingAction`
// into 'special_action' from a third place as well - every private_special_action_fail, which is
// the generic refusal the server sends for ANY rejected action, not just a special one. So a plain
// "you must draw a card first" refusal put a special-action prompt back on a table that had none,
// and since the refusal carries no rank, it rendered as a bare Skip button under the ordinary
// draw-your-card line, wired to a skip the server would only refuse again.
//
// Whether a skip is even on offer is the server's call too. An ability the engine armed by itself -
// the one a replace triggers under allowReplaceAbilities - cannot be declined, so the prompt for it
// arrives marked `mandatory` (service/internal/game/engine_adapter.go promptEngineArmedAbility) and
// the table must ask for a target instead of offering a skip.
//
// The piles. A stockpile draw is announced twice: player_draw_stockpile to the table carrying the
// server's post-draw counts, then private_draw_stockpile to the drawer carrying the card's face.
// The store applied the counts on both, so the drawer's own screen ran one card light on the
// stockpile from their draw until the next public draw event happened to overwrite it - which is
// the stockpile the two clients disagreed about.

/** A pending special action as the server projects it, in an event or in a sync snapshot. */
export interface SpecialActionView {
	active: boolean;
	playerId?: string;
	cardRank?: string;
	/** Set when the ability cannot be declined; absent means it can. */
	mandatory?: boolean;
}

/** The store's pending-action tag. */
export type PendingAction = string | null;

/**
 * Whether this client may offer to skip the special action on screen.
 *
 * A skip is offered only for an ability the player chose to invoke, never for one the engine armed
 * off their replace: the engine has no action that declines an armed ability, so the server refuses
 * that skip and the prompt would sit there looking dead.
 */
export function canSkipSpecialAction(special: SpecialActionView | null | undefined): boolean {
	if (!special?.active) return false;
	return !special.mandatory;
}

/**
 * The pending-action tag after a private_special_action_fail.
 *
 * The refusal is generic, so it only restores the prompt when this client is genuinely holding one;
 * otherwise the tag it already had stands. A refusal never invents a prompt.
 */
export function pendingActionAfterFail(
	current: PendingAction,
	special: SpecialActionView | null | undefined,
	selfId: string | null | undefined
): PendingAction {
	if (special?.active && !!selfId && special.playerId === selfId) return 'special_action';
	return current === 'special_action' ? null : current;
}

/**
 * The pending-action tag after a private_sync_state or a player_special_choice installed `special`.
 * Returns null when the snapshot holds no prompt for this client.
 */
export function pendingActionForSpecial(
	special: SpecialActionView | null | undefined,
	selfId: string | null | undefined
): PendingAction {
	return special?.active && !!selfId && special.playerId === selfId ? 'special_action' : null;
}

/** The two piles as the client tracks them. */
export interface PileCounts {
	stockpileSize: number;
	discardSize: number;
	discardTop?: unknown;
}

/** The payload half of a draw event. */
export interface DrawPayload {
	source?: string;
	stockpileSize?: number;
	discardSize?: number;
}

/**
 * Applies the pile counts a draw event carries.
 *
 * Only the public player_draw_stockpile moves them, and it moves them to the counts the server
 * sent rather than by a local decrement: the private_draw_stockpile that follows is the drawer's
 * view of the same draw, and counting it a second time is what put the drawer one card behind the
 * rest of the table.
 */
export function applyDrawPileCounts(piles: PileCounts, type: string, payload: DrawPayload | undefined): void {
	if (type !== 'player_draw_stockpile') return;
	if (payload?.source === 'stockpile') {
		piles.stockpileSize = typeof payload?.stockpileSize === 'number' ? payload.stockpileSize : piles.stockpileSize - 1;
		return;
	}
	piles.discardSize = typeof payload?.discardSize === 'number' ? payload.discardSize : piles.discardSize - 1;
	// The card that was on top has left the pile; the next event or sync names the new one.
	piles.discardTop = null;
}
