// src/lib/snapFill.ts
// Applies a player_snap_move event to the client's board model (cambia-936).
//
// RULES.md 5: snapping an opponent's card obliges the snapper to move one of their own cards into
// the slot it left. The server settles that as its own step (service/internal/game/snap_fill.go),
// either from the card the snapper names or from the deadline, and announces it with:
//   - `user`          the SNAPPER, whose hand loses the card;
//   - `card.user`     the VICTIM, whose hand gains it, and `card.idx` the slot it lands in;
//   - `payload.fromIdx` the slot it left in the snapper's hand.
// The card travels face down, so the event carries an id and no face: whoever receives it learns
// nothing about it, which is the whole point of the rule.
//
// No sync follows the move, so a client that does not apply it keeps showing the snapper a card
// they no longer hold and the victim an empty slot they no longer have - and worse, would target
// either by a slot number the server has already renumbered.

/** A card as it appears in an event payload or in a hand model. */
export interface FillCard {
	id: string;
	idx?: number;
	rank?: string;
	suit?: string;
	value?: number;
	known?: boolean;
	user?: { id: string };
}

/** The part of a player's board state a fill touches. */
export interface FillHandView {
	playerId: string;
	handSize: number;
	revealedHand?: FillCard[];
}

/** The part of the board a fill touches. */
export interface FillBoardLike {
	players: FillHandView[];
}

/** The event this module consumes. */
export interface SnapMoveEventLike {
	user?: { id?: string } | null;
	card?: FillCard | null;
	payload?: { fromIdx?: number; auto?: boolean } | null;
}

/** Who gave the card and who received it, or nulls where the event named nobody. */
export interface SnapMoveParties {
	snapperId: string | null;
	victimId: string | null;
}

/**
 * Moves the card from the snapper's hand into the victim's at the slot the event names. Mutates
 * the board in place (it runs inside the store's immer draft). Hand sizes move whether or not the
 * seats carry a hand model, since handSize is what the table counts card backs from.
 */
export function applySnapMove(board: FillBoardLike, ev: SnapMoveEventLike): SnapMoveParties {
	const card = ev.card ?? null;
	const snapperId = ev.user?.id ?? null;
	const victimId = card?.user?.id ?? null;
	if (!card || !snapperId || !victimId) return { snapperId, victimId };

	const snapper = board.players.find((p) => p.playerId === snapperId);
	if (snapper) {
		snapper.handSize = Math.max(0, snapper.handSize - 1);
		const hand = snapper.revealedHand;
		if (hand) {
			// Id first, slot second: an id hit is proof, a slot is an assumption about a model
			// patched from events (the same order lib/snapSuccess.ts removes in).
			let at = hand.findIndex((c) => c.id === card.id);
			if (at < 0) {
				const from = ev.payload?.fromIdx;
				at = typeof from === 'number' && from >= 0 && from < hand.length ? from : -1;
			}
			if (at >= 0) {
				hand.splice(at, 1);
				for (let i = at; i < hand.length; i++) {
					if (typeof hand[i].idx === 'number') hand[i].idx = (hand[i].idx as number) - 1;
				}
			}
		}
	}

	const victim = board.players.find((p) => p.playerId === victimId);
	if (victim) {
		victim.handSize += 1;
		const hand = victim.revealedHand;
		if (hand) {
			const named = card.idx;
			const at = typeof named === 'number' && named >= 0 && named <= hand.length ? named : hand.length;
			// Face down on arrival: the receiving player was not shown what they were given, and no
			// other client was either.
			hand.splice(at, 0, { id: card.id, known: false, idx: at });
			for (let i = at + 1; i < hand.length; i++) {
				if (typeof hand[i].idx === 'number') hand[i].idx = (hand[i].idx as number) + 1;
			}
		}
	}

	return { snapperId, victimId };
}
