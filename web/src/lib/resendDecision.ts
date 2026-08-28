// src/lib/resendDecision.ts
// Decides what to do with the last outbound WS message when the hub answers it with a
// sync_state repair (cambia-891).
//
// Hub side (service/internal/hub/hub.go dispatch): every inbound frame carries the client's
// last_seq, and `msg.LastSeq < h.seq` sends the sender a private sync_state and DISCARDS the
// message. The gate sits above the phase switch, so it covers lobby frames (ready, chat,
// update_rules, start_game) as well as game actions. cambia-878 removed the deterministic
// desync that fired this constantly, but the legitimate race remains: a frame sent while a
// broadcast is in flight is still dropped, and until this module the client only repaired its
// seq and let the action vanish.
//
// Safety property this module rests on (proved against the server, not assumed):
//   A sync_state stamped seq S proves the recorded message was dropped whenever S > sentSeq.
//   h.seq is monotonic and dispatch is serialized on the hub's Run goroutine, so if a message
//   sent with last_seq = L had been ACCEPTED (h.seq <= L at its dispatch), then every earlier
//   dispatch saw h.seq <= L too, and any sync_state they produced carries S <= L. Contrapositive:
//   S > L can only be produced after our message was dispatched and rejected. Resending on
//   S > sentSeq therefore never re-applies an action the server already applied.
// The record is the LAST outbound frame of any kind (a game action or a lobby frame): tracking
// only game actions would let a later dropped chat's repair resend an action that had landed.
//
// The seq proof only shows the frame was not applied; it says nothing about what applying it
// now would mean. A frame that addresses a hand SLOT (replace, every ability step) is resolved
// by the server on the index alone, so the decision re-checks the slot as well as the id: see
// CardRef, and the discard-top witness in decideResend.

/** What the socket should do with the recorded message. */
export type ResendVerdict =
	/** Send it again with the repaired seq. */
	| 'resend'
	/** The intent is stale: tell the player it did not go through. */
	| 'notify'
	/** Nothing to say: the frame is not resendable, or this repair does not answer it. */
	| 'drop';

/** Game-phase message types the table sends. */
export const GAME_ACTION_TYPES = new Set([
	'action_draw_stockpile',
	'action_draw_discardpile',
	'action_discard',
	'action_replace',
	'action_snap',
	'action_cambia',
	'action_special'
]);

/**
 * Lobby-phase types the same staleness gate drops. They carry no board intent, so the only
 * question is whether the hub is still in the phase the frame was composed for.
 */
export const LOBBY_ACTION_TYPES = new Set([
	'chat',
	'ready',
	'unready',
	'start_game',
	'update_rules'
]);

/** The subset of the game snapshot the decision reads (structurally satisfied by ObfGameState). */
export interface GameSnapshotLike {
	gameId: string;
	started: boolean;
	gameOver: boolean;
	turnId: number;
	currentPlayerId: string | null;
	cambiaCalled: boolean;
	discardTop?: { id: string } | null;
	players: {
		playerId: string;
		revealedHand?: { id: string; idx?: number }[];
		drawnCard?: { id: string } | null;
	}[];
	specialAction?: { active: boolean; playerId: string; cardRank: string } | null;
}

/** Everything the decision compares, captured at send time and again after the repair. */
export interface TableContext {
	/** null when no game state is loaded (lobby phase). */
	gameId: string | null;
	started: boolean;
	gameOver: boolean;
	turnId: number | null;
	currentPlayerId: string | null;
	discardTopId: string | null;
	/** gameStore.pendingAction: null | 'discard_replace' | 'special_action'. */
	pendingAction: string | null;
	/** Rank of the special action this player owes, else null. */
	specialRank: string | null;
	/** The card this player is holding after a draw, else null. */
	drawnCardId: string | null;
	cambiaCalled: boolean;
	/** Every card id this client can still name: own hand, opponent slots, drawn card, discard top. */
	cardIds: string[];
	/** Hand slot each card sits in now, keyed owner:card. Covers own and opponent hands. */
	slots: Record<string, number>;
	/** Hub phase from the lobby store. */
	phase: string;
	selfId: string | null;
}

/** One outbound frame, kept until the hub either acts on it or answers it with a repair. */
export interface OutboundRecord {
	type: string;
	/** Cards the frame names; each must still be in play, and in its slot, for a resend. */
	cardRefs: CardRef[];
	/** last_seq the frame went out with. */
	sentSeq: number;
	/** 0 for the original send. A frame is resent at most once, then the player is told. */
	attempt: number;
	ctx: TableContext;
}

/**
 * One card a frame names: the id it was composed against and, when the frame addresses a hand
 * slot, that slot and its owner. The slot is load-bearing, not decoration: the server resolves
 * a replace and every ability step by INDEX and never reads the id back
 * (engine_adapter.go handleReplaceViaEngine takes payload["idx"]; special_actions.go
 * parseCardTarget feeds doPeekSelfEngine, doPeekOtherEngine and doSwapBlindEngine the same way),
 * so an id that is still in play at a different slot is a different move.
 */
export interface CardRef {
	id: string;
	/** Hand slot the frame addresses. Absent for id-addressed frames (snap, discard). */
	idx?: number;
	/** Owner the frame names. Absent means the sender's own hand. */
	ownerId?: string;
}

/** Cards named by a client action frame, with the slot each one was addressed at. */
export function cardRefsOf(msg: {
	card?: { id?: string; idx?: number; user?: { id?: string } };
	card1?: { id?: string; idx?: number; user?: { id?: string } };
	card2?: { id?: string; idx?: number; user?: { id?: string } };
}): CardRef[] {
	const refs: CardRef[] = [];
	for (const ref of [msg.card, msg.card1, msg.card2]) {
		if (!ref?.id) continue;
		const out: CardRef = { id: ref.id };
		if (typeof ref.idx === 'number') out.idx = ref.idx;
		if (ref.user?.id) out.ownerId = ref.user.id;
		refs.push(out);
	}
	return refs;
}

/** Key for the slot map: a card is only "the same card" under the same owner. */
export function slotKey(ownerId: string, cardId: string): string {
	return `${ownerId}:${cardId}`;
}

/** Builds the comparison context from the two stores' current state. */
export function tableContext(
	gs: GameSnapshotLike | null | undefined,
	pendingAction: string | null,
	phase: string,
	selfId: string | null | undefined
): TableContext {
	const self = gs?.players.find((p) => p.playerId === selfId);
	const cardIds: string[] = [];
	const slots: Record<string, number> = {};
	if (gs) {
		for (const p of gs.players) {
			for (const [pos, c] of (p.revealedHand ?? []).entries()) {
				cardIds.push(c.id);
				// The server sends the authoritative slot (sync_state.go builds ObfCard.Idx for
				// both the self view and the opponent view); the array position is the fallback
				// for an event-patched hand that lost it.
				slots[slotKey(p.playerId, c.id)] = typeof c.idx === 'number' ? c.idx : pos;
			}
			if (p.drawnCard?.id) cardIds.push(p.drawnCard.id);
		}
		if (gs.discardTop?.id) cardIds.push(gs.discardTop.id);
	}
	const special = gs?.specialAction;
	return {
		gameId: gs?.gameId ?? null,
		started: !!gs?.started,
		gameOver: !!gs?.gameOver,
		turnId: typeof gs?.turnId === 'number' ? gs.turnId : null,
		currentPlayerId: gs?.currentPlayerId ?? null,
		discardTopId: gs?.discardTop?.id ?? null,
		pendingAction,
		specialRank: special?.active && special.playerId === selfId ? special.cardRank : null,
		drawnCardId: self?.drawnCard?.id ?? null,
		cambiaCalled: !!gs?.cambiaCalled,
		cardIds,
		slots,
		phase,
		selfId: selfId ?? null
	};
}

/**
 * Decides what to do with `rec` when a sync_state stamped `syncSeq` arrives and `now` is the
 * client's state after that repair.
 *
 * Every branch that is not a plain resend is a legality question already answered by the
 * server (service/internal/game/game.go HandlePlayerAction), mirrored here so the client does
 * not fire an action the server would reject or, worse, accept with a different meaning: a snap
 * resent after the discard top moved is a WRONG snap and draws a penalty
 * (engine_adapter.go handleSnapViaEngine -> handleSnapFailure).
 */
export function decideResend(rec: OutboundRecord, now: TableContext, syncSeq: number): ResendVerdict {
	const isGameAction = GAME_ACTION_TYPES.has(rec.type);

	// This repair answers an older frame; ours may well have landed. Never resend on it.
	if (syncSeq <= rec.sentSeq) return 'drop';

	// One retry. A frame dropped twice means the table moved on under it.
	if (rec.attempt > 0) return isGameAction ? 'notify' : 'drop';

	if (LOBBY_ACTION_TYPES.has(rec.type)) {
		return now.phase === rec.ctx.phase ? 'resend' : 'drop';
	}
	if (!isGameAction) return 'drop';

	// The table has to still be the same live table.
	if (!now.gameId || now.gameId !== rec.ctx.gameId) return 'drop';
	if (!now.started || now.gameOver) return 'drop';

	// The discard top is the client's witness that no hand has moved. Every way a hand shifts
	// under an in-flight frame is a snap, and a successful snap always pushes the snapped card
	// onto the discard (engine_adapter.go handleSnapViaEngine, both the own-hand and the
	// opponent-hand branch), so an unchanged top means no slot was renumbered. It is checked for
	// every game action, not only the two that read the top, because the client's own slot model
	// is patched from events and can lag; this does not depend on that bookkeeping.
	if (now.discardTopId !== rec.ctx.discardTopId) return 'notify';

	// Every card the frame names must still be in play, under the same owner and in the same
	// slot. A snap removes a card and shifts the rest of that hand left (engine_adapter.go, and
	// gameStore mirrors the splice), and nothing else the frame checks would notice: turnId,
	// currentPlayerId, pendingAction and the drawn card all survive a snap untouched. Resending
	// a slot-addressed frame against a shifted hand hits a card the player never clicked.
	for (const ref of rec.cardRefs) {
		if (!now.cardIds.includes(ref.id)) return 'notify';
		if (ref.idx === undefined) continue;
		const owner = ref.ownerId ?? now.selfId;
		if (!owner) return 'notify';
		if (now.slots[slotKey(owner, ref.id)] !== ref.idx) return 'notify';
	}

	const sameTurn = now.turnId === rec.ctx.turnId && now.currentPlayerId === rec.ctx.currentPlayerId;
	const myTurn = !!now.selfId && now.currentPlayerId === now.selfId;

	switch (rec.type) {
		case 'action_snap':
			// Snapping is legal out of turn, so the turn only has to be unmoved, not ours. The
			// top is the snap's whole subject (a moved top is already a notify above: the same
			// card id against a new top is a wrong snap and a penalty draw).
			if (!sameTurn) return 'notify';
			if (!now.discardTopId) return 'notify';
			if (now.pendingAction !== rec.ctx.pendingAction) return 'notify';
			return 'resend';

		case 'action_draw_discardpile':
			// Taking the discard needs a card there to take.
			if (!now.discardTopId) return 'notify';
			return myTurn && sameTurn && now.pendingAction === null && now.drawnCardId === null
				? 'resend'
				: 'notify';

		case 'action_draw_stockpile':
			return myTurn && sameTurn && now.pendingAction === null && now.drawnCardId === null
				? 'resend'
				: 'notify';

		case 'action_discard':
		case 'action_replace':
			// The server rejects either without a pending draw; the drawn card also has to be the
			// same one the click was about.
			if (!myTurn || !sameTurn) return 'notify';
			if (now.pendingAction !== 'discard_replace') return 'notify';
			if (!now.drawnCardId || now.drawnCardId !== rec.ctx.drawnCardId) return 'notify';
			return 'resend';

		case 'action_special':
			// Ability steps (peek, blind swap, the King's look and its follow-up, skip) are only
			// meaningful against the same pending special.
			if (!myTurn || !sameTurn) return 'notify';
			if (now.pendingAction !== 'special_action') return 'notify';
			if (!now.specialRank || now.specialRank !== rec.ctx.specialRank) return 'notify';
			return 'resend';

		case 'action_cambia':
			// Cambia is called at turn start, before drawing, and only once per game.
			if (!myTurn || !sameTurn) return 'notify';
			if (now.pendingAction !== null || now.drawnCardId !== null) return 'notify';
			if (now.cambiaCalled) return 'notify';
			return 'resend';

		default:
			return 'drop';
	}
}
