// Unit test for the client hand model after a snap fill (cambia-936).
//
// Run:  npm run test:snap-fill        (node --test, types stripped from the imported .ts)
//
// RULES.md 5 makes an opponent snap an even trade: the snapper takes a card and gives one back
// into the slot it left. The server settles that as its own step and announces it with the snapper
// on top and the victim under the card (service/internal/game/snap_fill.go). A client that does not
// apply it shows the snapper a card they no longer hold and the victim a slot they no longer have,
// and would then target either by a slot number the server has already renumbered.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { applySnapMove } from '../src/lib/snapFill.ts';

const A = 'player-a';
const B = 'player-b';

/** A live two-seat board: A is a card short (they were just snapped), B holds three. */
function board() {
    return {
        players: [
            {
                playerId: A,
                handSize: 2,
                revealedHand: [
                    { id: 'a-0', idx: 0 },
                    { id: 'a-2', idx: 1 }
                ]
            },
            {
                playerId: B,
                handSize: 3,
                revealedHand: [
                    { id: 'b-0', idx: 0 },
                    { id: 'b-1', idx: 1 },
                    { id: 'b-2', idx: 2 }
                ]
            }
        ]
    };
}

/** The event the server fires: B pays A a card, landing in A's slot 1. */
function moveEvent(over = {}) {
    return {
        user: { id: B },
        card: { id: 'b-1', idx: 1, user: { id: A } },
        payload: { fromIdx: 1, auto: false },
        ...over
    };
}

test('the card leaves the snapper and lands in the victim slot the event names', () => {
    const b = board();
    const parties = applySnapMove(b, moveEvent());

    assert.deepEqual(parties, { snapperId: B, victimId: A });

    const snapper = b.players.find((p) => p.playerId === B);
    assert.equal(snapper.handSize, 2);
    assert.deepEqual(snapper.revealedHand.map((c) => c.id), ['b-0', 'b-2']);
    // Every slot after the hole shifts left, matching the server's own hand shift.
    assert.deepEqual(snapper.revealedHand.map((c) => c.idx), [0, 1]);

    const victim = b.players.find((p) => p.playerId === A);
    assert.equal(victim.handSize, 3);
    assert.deepEqual(victim.revealedHand.map((c) => c.id), ['a-0', 'b-1', 'a-2']);
    assert.deepEqual(victim.revealedHand.map((c) => c.idx), [0, 1, 2]);
});

test('the paid card arrives face down', () => {
    // It changes hands unseen: the receiver was shown nothing, and neither was anyone else, so the
    // event carries an id and no face (snap_fill.go applySnapFill).
    const b = board();
    applySnapMove(b, moveEvent());
    const arrived = b.players.find((p) => p.playerId === A).revealedHand[1];
    assert.equal(arrived.known, false);
    assert.equal(arrived.rank, undefined);
});

test('the id is the key and the slot the fallback', () => {
    // The snapper's hand model is patched from events and can be a slot out; the id is minted once
    // per card for the life of the game (the same order lib/snapSuccess.ts removes in).
    const b = board();
    applySnapMove(b, moveEvent({ payload: { fromIdx: 0, auto: false } }));
    assert.deepEqual(b.players.find((p) => p.playerId === B).revealedHand.map((c) => c.id), ['b-0', 'b-2']);

    // With no id match at all, the named slot is what goes.
    const c = board();
    applySnapMove(c, { user: { id: B }, card: { id: 'unknown', idx: 0, user: { id: A } }, payload: { fromIdx: 2 } });
    assert.deepEqual(c.players.find((p) => p.playerId === B).revealedHand.map((c) => c.id), ['b-0', 'b-1']);
});

test('a slot past the end of the hand appends', () => {
    // The victim's hand can move between the snap and the fill (they may replace a card, be
    // snapped again, take a penalty), so the server clamps the target slot and so does this.
    const b = board();
    applySnapMove(b, moveEvent({ card: { id: 'b-1', idx: 9, user: { id: A } } }));
    const victim = b.players.find((p) => p.playerId === A);
    assert.deepEqual(victim.revealedHand.map((c) => c.id), ['a-0', 'a-2', 'b-1']);
    assert.deepEqual(victim.revealedHand.map((c) => c.idx), [0, 1, 2]);
});

test('hand sizes move for seats this client holds no hand model for', () => {
    // handSize is what the table counts card backs from, so it has to move whether or not the seat
    // carries a revealedHand.
    const b = { players: [{ playerId: A, handSize: 2 }, { playerId: B, handSize: 3 }] };
    applySnapMove(b, moveEvent());
    assert.equal(b.players[0].handSize, 3);
    assert.equal(b.players[1].handSize, 2);
});

test('an event naming nobody changes nothing', () => {
    const b = board();
    const before = JSON.stringify(b);
    assert.deepEqual(applySnapMove(b, { user: { id: B }, card: null }), { snapperId: B, victimId: null });
    assert.equal(JSON.stringify(b), before);
});
