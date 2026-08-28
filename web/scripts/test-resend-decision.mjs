// Unit test for the sync_state resend contract (cambia-891).
//
// Run:  npm run test:resend        (node --test, types stripped from the imported .ts)
//
// The two failure modes this pins down are opposite: never resending loses the player's
// action silently (the pre-cambia-891 behaviour), and always resending re-fires an action
// whose meaning has changed, which for a snap is a penalty draw.

/* global console */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decideResend, tableContext, cardRefsOf } from '../src/lib/resendDecision.ts';

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
        phase: 'in_game',
        selfId: SELF,
        ...over
    };
}

function record(type, over = {}) {
    return { type, cardRefs: [], sentSeq: 10, attempt: 0, ctx: ctx(), ...over };
}

// --- The staleness proof -----------------------------------------------------------------

test('a repair stamped at or below the send seq never resends', () => {
    // h.seq is monotonic and dispatch is serialized, so a sync_state carrying seq <= last_seq
    // cannot have been produced after our frame was rejected: it answers an earlier one, and
    // ours may have been applied. Resending there would double-apply.
    const rec = record('action_snap', { cardRefs: ['my-0'] });
    assert.equal(decideResend(rec, ctx(), 10), 'drop');
    assert.equal(decideResend(rec, ctx(), 9), 'drop');
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
});

test('a frame is resent at most once, then the player is told', () => {
    const rec = record('action_snap', { cardRefs: ['my-0'], attempt: 1 });
    assert.equal(decideResend(rec, ctx(), 11), 'notify');
    assert.equal(decideResend(record('chat', { attempt: 1 }), ctx(), 11), 'drop');
});

// --- Snap --------------------------------------------------------------------------------

test('snap resends while the discard top and the turn are unmoved', () => {
    const rec = record('action_snap', { cardRefs: ['my-0'] });
    assert.equal(decideResend(rec, ctx(), 11), 'resend');
    // Out of turn is fine for a snap: it is legal any time (game.go HandlePlayerAction).
    assert.equal(decideResend(
        record('action_snap', { cardRefs: ['my-0'], ctx: ctx({ currentPlayerId: OPP }) }),
        ctx({ currentPlayerId: OPP }),
        11
    ), 'resend');
});

test('snap notifies once the discard top has moved', () => {
    // Same card id, different meaning: the server would score it against the new top and draw
    // a penalty (engine_adapter.go handleSnapViaEngine -> handleSnapFailure).
    const rec = record('action_snap', { cardRefs: ['my-0'] });
    const moved = ctx({ discardTopId: 'discard-9', cardIds: ['my-0', 'my-1', 'opp-0', 'discard-9'] });
    assert.equal(decideResend(rec, moved, 11), 'notify');
});

test('snap notifies once the turn has moved on', () => {
    const rec = record('action_snap', { cardRefs: ['my-0'] });
    assert.equal(decideResend(rec, ctx({ turnId: 5, currentPlayerId: OPP }), 11), 'notify');
});

test('snap notifies when the snapped card is no longer in play', () => {
    const rec = record('action_snap', { cardRefs: ['my-0'] });
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
    const rec = { type: 'action_discard', cardRefs: ['drawn-1'], sentSeq: 10, attempt: 0, ctx: drew };
    assert.equal(decideResend(rec, drew, 11), 'resend');
    // The turn timer discarded for us: nothing is pending any more.
    assert.equal(decideResend(rec, ctx({ turnId: 5, currentPlayerId: OPP }), 11), 'notify');
    const replace = { ...rec, type: 'action_replace', cardRefs: ['my-0'] };
    assert.equal(decideResend(replace, drew, 11), 'resend');
    assert.equal(decideResend(replace, ctx(), 11), 'notify');
});

// --- Ability steps and Cambia -------------------------------------------------------------

test('ability steps resend only against the same pending special', () => {
    const king = ctx({ pendingAction: 'special_action', specialRank: 'K' });
    const rec = { type: 'action_special', cardRefs: ['my-0', 'opp-0'], sentSeq: 10, attempt: 0, ctx: king };
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

// --- Table lifecycle ---------------------------------------------------------------------

test('a finished or replaced game says nothing', () => {
    const rec = record('action_snap', { cardRefs: ['my-0'] });
    assert.equal(decideResend(rec, ctx({ gameOver: true }), 11), 'drop');
    assert.equal(decideResend(rec, ctx({ gameId: 'game-2' }), 11), 'drop');
    assert.equal(decideResend(rec, ctx({ gameId: null, started: false }), 11), 'drop');
});

// --- Lobby frames -------------------------------------------------------------------------

test('lobby frames resend while the phase holds', () => {
    // The staleness gate sits above the phase switch in hub.go dispatch(), so ready, chat,
    // start_game and update_rules are dropped by exactly the same rule.
    for (const type of ['chat', 'ready', 'unready', 'start_game', 'update_rules']) {
        const rec = record(type, { ctx: ctx({ phase: 'open' }) });
        assert.equal(decideResend(rec, ctx({ phase: 'open' }), 11), 'resend', type);
        assert.equal(decideResend(rec, ctx({ phase: 'countdown' }), 11), 'drop', type);
    }
});

test('an unknown frame type is never resent', () => {
    assert.equal(decideResend(record('ping'), ctx(), 11), 'drop');
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
            { playerId: OPP, revealedHand: [{ id: 'opp-0' }] }
        ]
    };
    const c = tableContext(gs, 'special_action', 'in_game', SELF);
    assert.equal(c.discardTopId, 'discard-7');
    assert.equal(c.drawnCardId, 'drawn-1');
    assert.equal(c.specialRank, 'K');
    assert.deepEqual(c.cardIds.sort(), ['discard-7', 'drawn-1', 'my-0', 'my-1', 'opp-0']);

    // An opponent's pending special is not ours to resend against.
    assert.equal(tableContext({ ...gs, specialAction: { active: true, playerId: OPP, cardRank: 'K' } }, null, 'in_game', SELF).specialRank, null);

    // No game state at all (lobby phase).
    const empty = tableContext(null, null, 'open', SELF);
    assert.equal(empty.gameId, null);
    assert.deepEqual(empty.cardIds, []);
});

test('cardRefsOf collects every card the frame names', () => {
    assert.deepEqual(cardRefsOf({ type: 'action_snap', card: { id: 'a' } }), ['a']);
    assert.deepEqual(cardRefsOf({ card1: { id: 'a' }, card2: { id: 'b' } }), ['a', 'b']);
    assert.deepEqual(cardRefsOf({ type: 'ready' }), []);
});

console.log('resend decision: all assertions defined');
