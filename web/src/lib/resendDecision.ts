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
// Records cover outbound frames of any kind (game actions and lobby frames alike): tracking only
// game actions would let a later dropped chat's repair resend an action that had landed. They are
// held in a small outbox rather than a single slot, because one repair window can swallow more
// than one frame; see The outbox at the foot of this file.
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
	/** Date.now() at send time. Used only to expire an outbox entry no repair ever answered. */
	sentAt?: number;
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

	// One retry. A frame dropped twice means the table moved on under it. A lobby frame is told
	// about too: a ready or a chat line that never landed is exactly as invisible as a lost
	// action, and the lobby now has somewhere to say so (cambia-913 F4).
	if (rec.attempt > 0) return isGameAction || LOBBY_ACTION_TYPES.has(rec.type) ? 'notify' : 'drop';

	if (LOBBY_ACTION_TYPES.has(rec.type)) {
		// The phase moved under the frame: ready in an open lobby means nothing once the
		// countdown started, and start_game after that is not the same instruction. Nothing can
		// be resent, so the player is told rather than left watching an unchanged lobby.
		return now.phase === rec.ctx.phase ? 'resend' : 'notify';
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

// --- The outbox --------------------------------------------------------------------------
//
// One slot is not enough (cambia-913 F2). The hub answers EVERY discarded frame with its own
// sync_state, and that repair consumes no seq (hub.go sendSyncState), so two frames sent inside
// one repair window are both discarded and both answered, with the same seq on each repair.
// Holding only the newest frame decided that one and lost the older one in silence: the exact
// failure cambia-891 set out to end.
//
// A repair names no frame, so each entry is judged on its own sentSeq. Entries the repair cannot
// speak to (syncSeq <= sentSeq) stay for a later one.
//
// It holds UNANSWERED frames, and that word is doing work (cambia-913 R1). Since a repair names
// no frame, every entry still in the outbox when one arrives is judged against it: an entry the
// hub had already ACCEPTED would be judged too, and neither answer is right. 'notify' tells the
// player an action failed that in fact landed; 'resend' puts an accepted chat line on the wire a
// second time. Acking is what keeps accepted frames out of that decision: see ackedType below.

/** How many unacknowledged frames the outbox holds. Past this the oldest is evicted, silently. */
export const OUTBOX_LIMIT = 8;

/**
 * How long an entry no repair and no ack ever answered stays eligible for a decision. A repair is
 * produced by the same dispatch that discarded the frame, so it lands within one round trip; past
 * a few seconds of silence the frame was accepted by a hub that broadcast nothing for it (today
 * update_rules is the only such frame), and its intent is no longer worth re-deciding.
 */
export const OUTBOX_TTL_MS = 5000;

/** A frame on the wire: what was sent, and what the decision needs in order to judge it. */
export interface OutboxEntry<M = unknown> {
	message: M;
	record: OutboundRecord;
}

/** What one repair does to the outbox. */
export interface OutboxOutcome<M = unknown> {
	/** Entries this repair does not answer; they stay in the outbox, in send order. */
	pending: OutboxEntry<M>[];
	/** Entries to put back on the wire, in send order, each at attempt + 1. */
	resend: OutboxEntry<M>[];
	/** Records the player should be told about. Lobby and game frames are told in different places. */
	notify: OutboundRecord[];
}

/** Appends a sent frame, evicting the oldest once the outbox is full. */
export function recordOutbound<M>(outbox: OutboxEntry<M>[], entry: OutboxEntry<M>): OutboxEntry<M>[] {
	const next = [...outbox, entry];
	return next.length > OUTBOX_LIMIT ? next.slice(next.length - OUTBOX_LIMIT) : next;
}

/**
 * The outbound frame type an inbound server frame proves the hub acted on, or null.
 *
 * The hub never replies to a frame by name: an accepted frame is answered by the events applying
 * it produces, so those events are the acknowledgement. Each case below is the emitter that fires
 * for exactly one client frame, with the actor on it:
 *
 *   player_draw_stockpile   engine_adapter.go emitEventsForAction, User = the actor, and
 *                           payload.source separates the stock draw from the discard draw (the
 *                           discard draw reuses the same event type).
 *   player_discard          fired for a discard (no idx) and for a replace (Card.Idx = the slot
 *                           replaced), which is what tells the two frames apart.
 *   player_cambia           the call.
 *   player_snap_success /
 *   player_snap_fail        User = the SNAPPER on both, so either one answers our snap.
 *   player_special_action   the public half of an ability step; the private success/fail halves
 *                           reach the actor alone (fireEventToPlayer), so they need no User.
 *   chat                    hub.go handleLobbyMsg, payload.userID = the author.
 *   lobby_state             broadcastLobbyUpdate after a ready or an unready; the self entry's
 *                           is_ready is the flag the frame was asking for. A snapshot broadcast
 *                           for some other reason can only ack a frame whose flag already holds,
 *                           which is a frame that asked for nothing.
 *   phase_change countdown  beginCountdown, reached from start_game and from the ready that
 *                           completes an auto-start lobby (MarkUserReadyUnsafe all-ready).
 *
 * update_rules has no case because an accepted one emits nothing at all (handleLobbyMsg applies
 * the rules and broadcasts no snapshot); OUTBOX_TTL_MS is what bounds it.
 */
export function ackedType(type: string, payload: unknown, selfId: string | null): string | null {
	const p = (payload ?? {}) as {
		user?: { id?: string };
		card?: { idx?: number };
		payload?: { source?: string };
		userID?: string;
		phase?: string;
		lobby_status?: { users?: { id?: string; is_ready?: boolean }[] };
	};
	const mine = !!selfId && p.user?.id === selfId;

	switch (type) {
		case 'player_draw_stockpile':
			if (!mine) return null;
			return p.payload?.source === 'discardpile' ? 'action_draw_discardpile' : 'action_draw_stockpile';
		case 'player_discard':
			if (!mine) return null;
			return typeof p.card?.idx === 'number' ? 'action_replace' : 'action_discard';
		case 'player_cambia':
			return mine ? 'action_cambia' : null;
		case 'player_snap_success':
		case 'player_snap_fail':
			return mine ? 'action_snap' : null;
		case 'player_special_action':
			return mine ? 'action_special' : null;
		case 'private_special_action_success':
		case 'private_special_action_fail':
			return 'action_special';
		case 'chat':
			return !!selfId && p.userID === selfId ? 'chat' : null;
		case 'lobby_state': {
			const self = (p.lobby_status?.users ?? []).find((u) => u.id === selfId);
			if (!self) return null;
			return self.is_ready ? 'ready' : 'unready';
		}
		case 'phase_change':
			return p.phase === 'countdown' ? 'start_game' : null;
		default:
			return null;
	}
}

/**
 * Removes the oldest frame this inbound envelope answers. One event acks one frame: two draws
 * cannot both have landed on one broadcast, and the outbox is in send order, so the oldest match
 * is the one that produced it.
 *
 * A countdown also acks a 'ready', not only the 'start_game' that usually causes it: an
 * auto-start lobby begins the countdown on the last ready, and that ready was ours if the hub
 * counted every seat as ready.
 */
export function ackOutbound<M>(
	outbox: OutboxEntry<M>[],
	type: string,
	payload: unknown,
	selfId: string | null
): OutboxEntry<M>[] {
	const acked = ackedType(type, payload, selfId);
	if (!acked) return outbox;
	const alsoReady = acked === 'start_game';
	const at = outbox.findIndex((e) => e.record.type === acked || (alsoReady && e.record.type === 'ready'));
	if (at < 0) return outbox;
	return [...outbox.slice(0, at), ...outbox.slice(at + 1)];
}

/**
 * Decides every outstanding frame against one sync_state repair. `now` is this client's state
 * after the repair was applied; `syncSeq` is the seq the repair carries.
 */
export function resolveOutbox<M>(
	outbox: OutboxEntry<M>[],
	now: TableContext,
	syncSeq: number,
	nowMs: number = Date.now()
): OutboxOutcome<M> {
	const out: OutboxOutcome<M> = { pending: [], resend: [], notify: [] };
	for (const entry of outbox) {
		const rec = entry.record;
		const expired = typeof rec.sentAt === 'number' && nowMs - rec.sentAt > OUTBOX_TTL_MS;
		if (expired) continue;
		// Not an answer to this frame: keep it, a later repair may be.
		if (syncSeq <= rec.sentSeq) {
			out.pending.push(entry);
			continue;
		}
		const verdict = decideResend(rec, now, syncSeq);
		if (verdict === 'resend') out.resend.push(entry);
		else if (verdict === 'notify') out.notify.push(rec);
	}
	return out;
}

/** True for a frame whose notice belongs in the lobby rather than on the table. */
export function isLobbyFrame(rec: OutboundRecord): boolean {
	return LOBBY_ACTION_TYPES.has(rec.type);
}
