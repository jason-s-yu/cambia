// src/lib/snapSuccess.ts
// Applies a player_snap_success event to the client's board model (cambia-913).
//
// The event names two different people and the difference is the whole point:
//   - `user` is the SNAPPER, the player who acted;
//   - `card.user` is the OWNER, the hand the card left, and `card.idx` is the slot it sat in
//     WITHIN THAT HAND.
// Both branches of the server emitter fire with the snapper on top (service/internal/game/
// engine_adapter.go handleSnapViaEngine -> emitSnapSuccessEvents), so a client that removes the
// card from `user` takes it out of the snapper's hand even when the server took it out of the
// victim's: after B snapped A's card the table showed B one card short and A one card long on
// every screen, until the next private_sync_state happened to correct it.
//
// The removal is by card id first, slot second. The server sends both, and the id is the stronger
// key: the client's hand model is patched from events and can be a slot out (a penalty draw is
// appended at the server's index, a blind swap moves ids between hands), while the id is minted
// once per card for the life of the game.

/** A card as it appears in an event payload or in a hand model. */
export interface SnapCard {
	id: string;
	idx?: number;
	rank?: string;
	suit?: string;
	value?: number;
	/** Owner of the card. Present on player_snap_success since cambia-913. */
	user?: { id: string };
}

/** The part of a player's board state a snap touches. */
export interface SnapHandView {
	playerId: string;
	handSize: number;
	/** Own hand carries faces; an opponent hand carries id references (cambia-509). */
	revealedHand?: SnapCard[];
}

/** The part of the board a snap touches. */
export interface SnapBoardLike {
	players: SnapHandView[];
	discardSize: number;
	discardTop?: SnapCard | null;
}

/** The event this module consumes. */
export interface SnapSuccessEventLike {
	user?: { id?: string } | null;
	card?: SnapCard | null;
}

/**
 * Whose hand the snapped card left. The card's own owner when the server names one, else the
 * snapper: an own-hand snap is the only case those differ in, so the fallback keeps a client
 * talking to a pre-cambia-913 server exactly as wrong as it was, and no worse.
 */
export function snapOwnerId(ev: SnapSuccessEventLike): string | null {
	return ev.card?.user?.id ?? ev.user?.id ?? null;
}

/**
 * Removes the snapped card from its owner's hand and puts it on the discard pile. Mutates the
 * board in place (it runs inside the store's immer draft). Returns the owner id it acted on, or
 * null when the event named no one this client knows.
 */
export function applySnapSuccess(board: SnapBoardLike, ev: SnapSuccessEventLike): string | null {
	const card = ev.card ?? null;

	// The pile is public and moves whether or not the owner is a seat this client tracks.
	board.discardSize += 1;
	if (card) board.discardTop = card;

	const ownerId = snapOwnerId(ev);
	if (!ownerId || !card) return null;
	const owner = board.players.find((p) => p.playerId === ownerId);
	if (!owner) return null;

	owner.handSize = Math.max(0, owner.handSize - 1);

	const hand = owner.revealedHand;
	if (!hand) return ownerId;

	// Id first, slot second: an id hit is proof, a slot is an assumption about a model patched
	// from events.
	let at = hand.findIndex((c) => c.id === card.id);
	if (at < 0) {
		if (typeof card.idx !== 'number' || card.idx < 0 || card.idx >= hand.length) return ownerId;
		at = card.idx;
	}
	hand.splice(at, 1);
	// Every slot after the hole shifts left, matching the server (engine_adapter.go shifts the
	// engine hand and the UUID mirror the same way).
	for (let i = at; i < hand.length; i++) {
		if (typeof hand[i].idx === 'number') hand[i].idx = (hand[i].idx as number) - 1;
	}
	return ownerId;
}
