// Unit test for the sync_state resend contract (cambia-891).
//
// Run:  npm run test:resend        (node --test, types stripped from the imported .ts)
//
// The two failure modes this pins down are opposite: never resending loses the player's
// action silently (the pre-cambia-891 behaviour), and always resending re-fires an action
// whose meaning has changed: for a snap a penalty draw, for a slot-addressed frame (replace,
// every ability step) a hit on whatever card now sits in that slot, since the server resolves
// those by index and never reads the card id back.

/* global console */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
    decideResend,
    tableContext,
    cardRefsOf,
    recordOutbound,
    resolveOutbox,
    isLobbyFrame,
    OUTBOX_LIMIT,
    OUTBOX_TTL_MS
} from '../src/lib/resendDecision.ts';

const SELF = 'self-id';
const OPP = 'opp-id';

/** A live table: our turn, nothing pending, a 7 on top of the discard. */
function ctx(over = {}) {
    return {
        gameId: 'game-1',
        started: true,
        gameOver: false,
        turnId: 4,
        currentPlayerId: SELF,
        discardTopId: 'discard-7',
        pendingAction: null,
        specialRank: null,
        drawnCardId: null,
        cambiaCalled: false,
        cardIds: ['my-0', 'my-1', 'opp-0', 'discard-7'],
        slots: { [`${SELF}:my-0`]: 0, [`${SELF}:my-1`]: 1, [`${OPP}:opp-0`]: 0 },
        phase: 'in_game',
        selfId: SELF,
        ...over
    };
}

function record(type, over = {}) {
    return { type, cardRefs: [], sentSeq: 10, attempt: 0, sentAt: 1000, ctx: ctx(), ...over };
}

// --- The staleness proof -----------------------------------------------------------------

test('a repair stamped at or below the send seq never resends', () => {
    // h.seq is monotonic and dispatch is serialized, so a sync_state carrying seq <= last_seq
    // cannot have been produced after our frame was rejected: it answers an earlier one, and
    // ours may have been applied. Resending there would double-apply.
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx(), 10), 'drop');
    assert.equal(decideResend(rec, ctx(), 9), 'drop');
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
});

test('a frame is resent at most once, then the player is told', () => {
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }], attempt: 1 });
    assert.equal(decideResend(rec, ctx(), 11), 'notify');
    // A lobby frame is told about too (cambia-913 F4): a chat line dropped twice is as invisible
    // as a lost action, and the lobby now has a place to say so.
    assert.equal(decideResend(record('chat', { attempt: 1 }), ctx(), 11), 'notify');
});

// --- Snap --------------------------------------------------------------------------------

test('snap resends while the discard top and the turn are unmoved', () => {
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
    // Out of turn is fine for a snap: it is legal any time (game.go HandlePlayerAction).
    assert.equal(decideResend(
        record('action_snap', { cardRefs: [{ id: 'my-0' }], ctx: ctx({ currentPlayerId: OPP }) }),
        ctx({ currentPlayerId: OPP }),
        11
    ), 'resend');
});

test('snap notifies once the discard top has moved', () => {
    // Same card id, different meaning: the server would score it against the new top and draw
    // a penalty (engine_adapter.go handleSnapViaEngine -> handleSnapFailure).
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    const moved = ctx({ discardTopId: 'discard-9', cardIds: ['my-0', 'my-1', 'opp-0', 'discard-9'] });
    assert.equal(decideResend(rec, moved, 11), 'notify');
});

test('snap notifies once the turn has moved on', () => {
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx({ turnId: 5, currentPlayerId: OPP }), 11), 'notify');
});

test('snap notifies when the snapped card is no longer in play', () => {
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx({ cardIds: ['my-1', 'opp-0', 'discard-7'] }), 11), 'notify');
});

// --- Draw / discard / replace ------------------------------------------------------------

test('draw resends only on the same turn with nothing drawn yet', () => {
    const rec = record('action_draw_stockpile');
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
    assert.equal(decideResend(rec, ctx({ currentPlayerId: OPP }), 11), 'notify');
    assert.equal(decideResend(rec, ctx({ turnId: 5 }), 11), 'notify');
    // The draw already landed: resending would be a second draw, which the server rejects
    // with "You have already drawn a card this turn."
    assert.equal(decideResend(rec, ctx({ drawnCardId: 'drawn-1', pendingAction: 'discard_replace' }), 11), 'notify');
});

test('taking the discard is bound to the card that was on top', () => {
    const rec = record('action_draw_discardpile');
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
    assert.equal(decideResend(rec, ctx({ discardTopId: 'discard-9', cardIds: ['my-0', 'discard-9'] }), 11), 'notify');
});

test('discard and replace resend only against the same pending draw', () => {
    const drew = ctx({ pendingAction: 'discard_replace', drawnCardId: 'drawn-1', cardIds: ['my-0', 'my-1', 'drawn-1', 'discard-7'] });
    const rec = { type: 'action_discard', cardRefs: [{ id: 'drawn-1' }], sentSeq: 10, attempt: 0, ctx: drew };
    assert.equal(decideResend(rec, drew, 11), 'resend');
    // The turn timer discarded for us: nothing is pending any more.
    assert.equal(decideResend(rec, ctx({ turnId: 5, currentPlayerId: OPP }), 11), 'notify');
    const replace = { ...rec, type: 'action_replace', cardRefs: [{ id: 'my-0' }] };
    assert.equal(decideResend(replace, drew, 11), 'resend');
    assert.equal(decideResend(replace, ctx(), 11), 'notify');
});

// --- Ability steps and Cambia -------------------------------------------------------------

test('ability steps resend only against the same pending special', () => {
    const king = ctx({ pendingAction: 'special_action', specialRank: 'K' });
    const rec = { type: 'action_special', cardRefs: [{ id: 'my-0', idx: 0 }, { id: 'opp-0', idx: 0, ownerId: OPP }], sentSeq: 10, attempt: 0, ctx: king };
    assert.equal(decideResend(rec, king, 11), 'resend');
    assert.equal(decideResend(rec, ctx({ pendingAction: 'special_action', specialRank: 'Q' }), 11), 'notify');
    assert.equal(decideResend(rec, ctx(), 11), 'notify');
});

test('Cambia resends only while it is still callable', () => {
    const rec = record('action_cambia');
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
    assert.equal(decideResend(rec, ctx({ cambiaCalled: true }), 11), 'notify');
    assert.equal(decideResend(rec, ctx({ drawnCardId: 'drawn-1', pendingAction: 'discard_replace' }), 11), 'notify');
});

// --- Slot safety (the shifted-hand resend) ------------------------------------------------

test('a slot-addressed frame notifies when its card has moved slot', () => {
    // The server takes payload["idx"] and never checks the id (engine_adapter.go
    // handleReplaceViaEngine; special_actions.go parseCardTarget for every ability step), so a
    // resend against a hand that shifted under the drop would replace the wrong card. Nothing
    // else in the decision sees it: a snap by either player leaves turnId, currentPlayerId,
    // pendingAction and the drawn card exactly as they were.
    const drew = ctx({ pendingAction: 'discard_replace', drawnCardId: 'drawn-1', cardIds: ['my-0', 'my-1', 'opp-0', 'drawn-1', 'discard-7'] });
    const rec = { type: 'action_replace', cardRefs: [{ id: 'my-1', idx: 1 }], sentSeq: 10, attempt: 0, ctx: drew };
    assert.equal(decideResend(rec, drew, 11), 'resend');

    const shifted = { ...drew, slots: { [`${SELF}:my-1`]: 0, [`${OPP}:opp-0`]: 0 } };
    assert.equal(decideResend(rec, shifted, 11), 'notify');
});

test('an ability step checks the opponent slot under its owner', () => {
    const king = ctx({ pendingAction: 'special_action', specialRank: 'K' });
    const rec = { type: 'action_special', cardRefs: [{ id: 'my-0', idx: 0 }, { id: 'opp-0', idx: 0, ownerId: OPP }], sentSeq: 10, attempt: 0, ctx: king };
    assert.equal(decideResend(rec, king, 11), 'resend');

    // The opponent's hand shifted: same id, slot 1 now.
    const oppShifted = { ...king, slots: { ...king.slots, [`${OPP}:opp-0`]: 1 } };
    assert.equal(decideResend(rec, oppShifted, 11), 'notify');

    // Same slot number, wrong owner: the id has to sit under the owner the frame named.
    const wrongOwner = { ...king, slots: { [`${SELF}:my-0`]: 0, [`${SELF}:opp-0`]: 0 } };
    assert.equal(decideResend(rec, wrongOwner, 11), 'notify');

    // A client hand model that lost its slots fails closed rather than resending blind.
    assert.equal(decideResend(rec, { ...king, slots: {} }, 11), 'notify');
});

test('an id-addressed frame is not slot-checked', () => {
    // A snap carries no idx: the server searches both hands by UUID, so a shifted hand does not
    // change its meaning (only a moved discard top does, and that is checked).
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx({ slots: {} }), 11), 'resend');
});

test('a moved discard top stops every game action, not just the two that read it', () => {
    // The top is the witness that no hand shifted: a successful snap always pushes the snapped
    // card onto the discard. Checking it does not depend on the client's own slot bookkeeping.
    const moved = ctx({ discardTopId: 'discard-9', cardIds: ['my-0', 'my-1', 'opp-0', 'discard-9'] });
    assert.equal(decideResend(record('action_draw_stockpile'), moved, 11), 'notify');
    assert.equal(decideResend(record('action_cambia'), moved, 11), 'notify');
});

// --- Table lifecycle ---------------------------------------------------------------------

test('a finished or replaced game says nothing', () => {
    const rec = record('action_snap', { cardRefs: [{ id: 'my-0' }] });
    assert.equal(decideResend(rec, ctx({ gameOver: true }), 11), 'drop');
    assert.equal(decideResend(rec, ctx({ gameId: 'game-2' }), 11), 'drop');
    assert.equal(decideResend(rec, ctx({ gameId: null, started: false }), 11), 'drop');
});

// --- Lobby frames -------------------------------------------------------------------------

test('lobby frames resend while the phase holds, and are reported once it moves', () => {
    // The staleness gate sits above the phase switch in hub.go dispatch(), so ready, chat,
    // start_game and update_rules are dropped by exactly the same rule. A phase that moved makes
    // the frame unresendable ('ready' in an open lobby is not 'ready' in a countdown), and the
    // player is told rather than left watching a lobby that did not react (cambia-913 F4).
    for (const type of ['chat', 'ready', 'unready', 'start_game', 'update_rules']) {
        const rec = record(type, { ctx: ctx({ phase: 'open' }) });
        assert.equal(decideResend(rec, ctx({ phase: 'open' }), 11), 'resend', type);
        assert.equal(decideResend(rec, ctx({ phase: 'countdown' }), 11), 'notify', type);
        assert.equal(isLobbyFrame(rec), true, type);
    }
    assert.equal(isLobbyFrame(record('action_snap')), false);
});

test('an unknown frame type is never resent', () => {
    assert.equal(decideResend(record('ping'), ctx(), 11), 'drop');
    assert.equal(decideResend(record('ping', { attempt: 1 }), ctx(), 11), 'drop');
});

// --- Context derivation --------------------------------------------------------------------

test('tableContext reads the snapshot the way the table does', () => {
    const gs = {
        gameId: 'game-1',
        started: true,
        gameOver: false,
        turnId: 4,
        currentPlayerId: SELF,
        cambiaCalled: false,
        discardTop: { id: 'discard-7' },
        specialAction: { active: true, playerId: SELF, cardRank: 'K' },
        players: [
            { playerId: SELF, revealedHand: [{ id: 'my-0' }, { id: 'my-1' }], drawnCard: { id: 'drawn-1' } },
            { playerId: OPP, revealedHand: [{ id: 'opp-0', idx: 7 }] }
        ]
    };
    const c = tableContext(gs, 'special_action', 'in_game', SELF);
    assert.equal(c.discardTopId, 'discard-7');
    // Slots come from the server's ObfCard.idx where it is present (sync_state.go fills it for
    // the self view and the opponent view alike) and from the array position otherwise.
    assert.equal(c.slots[`${SELF}:my-1`], 1);
    assert.equal(c.slots[`${OPP}:opp-0`], 7);
    assert.equal(c.drawnCardId, 'drawn-1');
    assert.equal(c.specialRank, 'K');
    assert.deepEqual(c.cardIds.sort(), ['discard-7', 'drawn-1', 'my-0', 'my-1', 'opp-0']);

    // An opponent's pending special is not ours to resend against.
    assert.equal(tableContext({ ...gs, specialAction: { active: true, playerId: OPP, cardRank: 'K' } }, null, 'in_game', SELF).specialRank, null);

    // No game state at all (lobby phase).
    const empty = tableContext(null, null, 'open', SELF);
    assert.equal(empty.gameId, null);
    assert.deepEqual(empty.cardIds, []);
    assert.deepEqual(empty.slots, {});
});

test('cardRefsOf keeps the slot and the owner the frame addressed', () => {
    // A snap names an id only; the server finds it in either hand by UUID.
    assert.deepEqual(cardRefsOf({ type: 'action_snap', card: { id: 'a' } }), [{ id: 'a' }]);
    // A replace names a slot in the sender's own hand; an ability step names both sides.
    assert.deepEqual(cardRefsOf({ card: { id: 'a', idx: 2 } }), [{ id: 'a', idx: 2 }]);
    assert.deepEqual(
        cardRefsOf({ card1: { id: 'a', idx: 1 }, card2: { id: 'b', idx: 3, user: { id: OPP } } }),
        [{ id: 'a', idx: 1 }, { id: 'b', idx: 3, ownerId: OPP }]
    );
    assert.deepEqual(cardRefsOf({ type: 'ready' }), []);
});

// --- The outbox (more than one frame per repair window) -------------------------------------

/** An outbox entry: what went on the wire, plus the record the decision reads. */
function entry(type, over = {}) {
    return { message: { type }, record: record(type, over) };
}

test('both frames dropped in one window are decided, not just the newest', () => {
    // The hub answers every discarded frame with its own sync_state and that repair consumes no
    // seq (hub.go sendSyncState), so a window can swallow two frames and answer both with the
    // same seq. The single slot this replaced decided the newest and lost the older in silence,
    // which is the failure cambia-891 set out to end (cambia-913 F2).
    const snap = entry('action_snap', { cardRefs: [{ id: 'my-0' }] });
    const chat = entry('chat', { ctx: ctx({ phase: 'open' }) });
    const outbox = recordOutbound(recordOutbound([], snap), chat);
    assert.equal(outbox.length, 2);

    const out = resolveOutbox(outbox, ctx({ phase: 'open' }), 11, 1000);
    assert.deepEqual(out.resend.map((e) => e.record.type), ['action_snap', 'chat']);
    assert.deepEqual(out.pending, []);
    assert.deepEqual(out.notify, []);
});

test('each frame is judged on its own intent, not the newest one\'s', () => {
    // A snap whose discard top moved is stale; a chat sent in the same window is not.
    const snap = entry('action_snap', { cardRefs: [{ id: 'my-0' }] });
    const chat = entry('chat', { ctx: ctx({ phase: 'open' }) });
    const moved = ctx({ phase: 'open', discardTopId: 'discard-9', cardIds: ['my-0', 'my-1', 'opp-0', 'discard-9'] });

    const out = resolveOutbox([snap, chat], moved, 11, 1000);
    assert.deepEqual(out.resend.map((e) => e.record.type), ['chat']);
    assert.deepEqual(out.notify.map((r) => r.type), ['action_snap']);
});

test('a repair older than a frame leaves it in the outbox for the next one', () => {
    const older = entry('action_snap', { cardRefs: [{ id: 'my-0' }], sentSeq: 10 });
    const newer = entry('chat', { sentSeq: 12, ctx: ctx({ phase: 'open' }) });

    const out = resolveOutbox([older, newer], ctx({ phase: 'open' }), 11, 1000);
    assert.deepEqual(out.resend.map((e) => e.record.type), ['action_snap']);
    assert.deepEqual(out.pending.map((e) => e.record.type), ['chat'], 'seq 11 cannot answer a frame sent at 12');
    // The later repair answers it.
    const next = resolveOutbox(out.pending, ctx({ phase: 'open' }), 13, 1000);
    assert.deepEqual(next.resend.map((e) => e.record.type), ['chat']);
    assert.deepEqual(next.pending, []);
});

test('the outbox is bounded and keeps the newest frames', () => {
    let outbox = [];
    for (let i = 0; i < OUTBOX_LIMIT + 3; i++) {
        outbox = recordOutbound(outbox, entry('chat', { sentSeq: i, ctx: ctx({ phase: 'open' }) }));
    }
    assert.equal(outbox.length, OUTBOX_LIMIT);
    assert.equal(outbox[0].record.sentSeq, 3, 'the oldest three were evicted');
});

test('a frame no repair ever answered ages out instead of being resurrected', () => {
    // An accepted frame is never answered, so its entry would sit there forever and a repair
    // minutes later would judge an intent nobody remembers forming.
    const old = entry('chat', { sentAt: 1000, ctx: ctx({ phase: 'open' }) });
    const fresh = entry('chat', { sentAt: 1000 + OUTBOX_TTL_MS, ctx: ctx({ phase: 'open' }) });
    const at = 1000 + OUTBOX_TTL_MS + 1;

    const out = resolveOutbox([old, fresh], ctx({ phase: 'open' }), 11, at);
    assert.deepEqual(out.resend.map((e) => e.record.sentAt), [1000 + OUTBOX_TTL_MS]);
    assert.deepEqual(out.notify, [], 'an expired frame is not worth a notice either');
    assert.deepEqual(out.pending, []);
});

test('a repair that answers nothing changes nothing', () => {
    const rec = entry('action_snap', { cardRefs: [{ id: 'my-0' }], sentSeq: 20 });
    const out = resolveOutbox([rec], ctx(), 11, 1000);
    assert.deepEqual(out.pending.length, 1);
    assert.deepEqual(out.resend, []);
    assert.deepEqual(out.notify, []);
    assert.deepEqual(resolveOutbox([], ctx(), 11, 1000), { pending: [], resend: [], notify: [] });
});

test('a dropped game action and a dropped lobby frame are reported separately', () => {
    // The two notices land in different surfaces, so the caller has to be able to tell them
    // apart from the record alone.
    const snap = entry('action_snap', { cardRefs: [{ id: 'my-0' }], attempt: 1 });
    const ready = entry('ready', { attempt: 1, ctx: ctx({ phase: 'open' }) });
    const out = resolveOutbox([snap, ready], ctx({ phase: 'open' }), 11, 1000);
    assert.deepEqual(out.notify.map((r) => [r.type, isLobbyFrame(r)]), [['action_snap', false], ['ready', true]]);
});

console.log('resend decision: all assertions defined');
