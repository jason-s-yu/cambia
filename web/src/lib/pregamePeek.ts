// src/lib/pregamePeek.ts
// Re-applies the pregame peek to the client's own hand (cambia-1094).
//
// No own card is ever persistently face-up: the server sends every own hand slot with the face
// hidden, in every phase, because the physical game turns your peeked cards down at the start and
// leaves you to play the round on memory. private_initial_cards is therefore the ONLY frame that
// ever carries those faces, and nothing can restore them from a snapshot.
//
// That makes every sync during the pregame window a hazard: a peer dropping, a staleness repair, or
// the reconnect resync all replace the board with one whose own slots are face-down, and would end
// the peek before the window is up. So the store holds the faces and calls this for every snapshot
// while preGameActive, and drops them on the first snapshot with preGameActive false - which is the
// sync StartGame broadcasts, and is exactly what turns the cards down at game start.
//
// Matching is by card id ONLY. The id is minted once per card and both frames read it out of the
// same server-side tracker (CardTracker.Players[i].HandUUIDs), so it always hits during the window,
// and unlike a removal - where lib/snapSuccess.ts and lib/snapFill.ts fall back to a slot index -
// the failure here would not be a missing card but a face painted onto the wrong one. A back is a
// correct-if-unhelpful render; a wrong face is a lie the player would go on to play against.

/** A card as it appears in a hand model or in the held peek. */
export interface PeekCard {
	id: string;
	rank?: string;
	suit?: string;
	value?: number;
	idx?: number;
	known?: boolean;
}

/** The part of a player's board state the peek touches. */
export interface PeekHandView {
	playerId: string;
	revealedHand?: PeekCard[];
}

/** The part of the board the peek touches. */
export interface PeekBoardLike {
	players: PeekHandView[];
}

/**
 * Puts the peeked faces back onto `selfId`'s hand. Mutates the board in place (it runs inside the
 * store's immer draft). Slot ids and indices come from the board, which is authoritative for them;
 * only the face comes from the peek. Returns how many slots it faced, which is what a caller (or a
 * test) can check the peek actually landed by.
 */
export function applyPregamePeek(
	board: PeekBoardLike | null | undefined,
	peek: PeekCard[] | null | undefined,
	selfId: string | null | undefined
): number {
	if (!board || !selfId || !peek || peek.length === 0) return 0;
	const self = board.players.find((p) => p.playerId === selfId);
	const hand = self?.revealedHand;
	if (!hand) return 0;

	let faced = 0;
	for (const card of peek) {
		if (!card?.id) continue;
		const at = hand.findIndex((c) => c.id === card.id);
		if (at < 0) continue;
		hand[at] = { ...hand[at], known: true, rank: card.rank, suit: card.suit, value: card.value };
		faced++;
	}
	return faced;
}
