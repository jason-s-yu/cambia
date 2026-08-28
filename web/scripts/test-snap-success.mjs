// Unit test for the client hand model after a snap (cambia-913).
//
// Run:  npm run test:snap        (node --test, types stripped from the imported .ts)
//
// player_snap_success names the SNAPPER at the top level and the card's OWNER under the card
// (service/internal/game/engine_adapter.go emitSnapSuccessEvents fires both branches with the
// snapper on top). Removing the card from the snapper is right for an own-hand snap and wrong for
// every opponent snap: the server took the card out of the victim's hand, so the table showed the
// snapper a card short and the victim a card long on every screen until a private_sync_state
// happened to correct it.

/* global console */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { applySnapSuccess, snapOwnerId } from '../src/lib/snapSuccess.ts';

const A = 'player-a';
const B = 'player-b';

/** A live two-seat board: A holds three cards, B four, a 7 on the discard. */
function board() {
    return {
        players: [
            {
                playerId: A,
                handSize: 3,
                revealedHand: [
                    { id: 'a-0', idx: 0 },
                    { id: 'a-1', idx: 1 },
                    { id: 'a-2', idx: 2 }
                ]
            },
            {
                playerId: B,
                handSize: 4,
                revealedHand: [
                    { id: 'b-0', idx: 0 },
                    { id: 'b-1', idx: 1 },
                    { id: 'b-2', idx: 2 },
                    { id: 'b-3', idx: 3 }
                ]
            }
        ],
        discardSize: 5,
        discardTop: { id: 'discard-7', rank: '7' }
    };
}

const seat = (b, id) => b.players.find((p) => p.playerId === id);

// --- The bug this file exists for -------------------------------------------------------

test('B snapping A\'s card takes it out of A\'s hand, not B\'s', () => {
    const b = board();
    applySnapSuccess(b, {
        user: { id: B },
        card: { id: 'a-1', idx: 1, rank: '7', suit: 'S', user: { id: A } }
    });

    assert.equal(seat(b, A).handSize, 2, 'the victim loses a card');
    assert.equal(seat(b, B).handSize, 4, 'the snapper keeps every card');
    assert.deepEqual(seat(b, A).revealedHand.map((c) => c.id), ['a-0', 'a-2']);
    assert.deepEqual(seat(b, B).revealedHand.map((c) => c.id), ['b-0', 'b-1', 'b-2', 'b-3']);
});

test('the slots after the hole renumber, in the owner\'s hand only', () => {
    // The server shifts the engine hand and its UUID mirror left (engine_adapter.go), and a
    // stale idx here would send the next replace or ability step at the wrong card: the server
    // resolves those by index alone.
    const b = board();
    applySnapSuccess(b, { user: { id: B }, card: { id: 'a-0', idx: 0, user: { id: A } } });
    assert.deepEqual(seat(b, A).revealedHand, [{ id: 'a-1', idx: 0 }, { id: 'a-2', idx: 1 }]);
    assert.deepEqual(seat(b, B).revealedHand.map((c) => c.idx), [0, 1, 2, 3]);
});

test('an own-hand snap still shrinks the snapper', () => {
    const b = board();
    applySnapSuccess(b, { user: { id: B }, card: { id: 'b-2', idx: 2, user: { id: B } } });
    assert.equal(seat(b, B).handSize, 3);
    assert.equal(seat(b, A).handSize, 3);
    assert.deepEqual(seat(b, B).revealedHand.map((c) => c.id), ['b-0', 'b-1', 'b-3']);
});

// --- The pile ---------------------------------------------------------------------------

test('the snapped card becomes the discard top for everyone', () => {
    const b = board();
    const card = { id: 'a-1', idx: 1, rank: '7', suit: 'S', user: { id: A } };
    applySnapSuccess(b, { user: { id: B }, card });
    assert.equal(b.discardSize, 6);
    assert.equal(b.discardTop.id, 'a-1');
    assert.equal(b.discardTop.rank, '7');
});

// --- Owner resolution -------------------------------------------------------------------

test('the owner is the card\'s, and the snapper only as a fallback', () => {
    assert.equal(snapOwnerId({ user: { id: B }, card: { id: 'a-1', user: { id: A } } }), A);
    // A server that does not name an owner cannot be second-guessed; the snapper is the old
    // assumption and stays the fallback rather than dropping the update entirely.
    assert.equal(snapOwnerId({ user: { id: B }, card: { id: 'b-1' } }), B);
    assert.equal(snapOwnerId({ card: { id: 'b-1' } }), null);
});

// --- Model drift ------------------------------------------------------------------------

test('the card id wins over a slot the client no longer agrees with', () => {
    // The client's hand model is patched from events and can be a slot out; the id is minted
    // once per card for the life of the game.
    const b = board();
    applySnapSuccess(b, { user: { id: B }, card: { id: 'a-2', idx: 0, user: { id: A } } });
    assert.deepEqual(seat(b, A).revealedHand.map((c) => c.id), ['a-0', 'a-1']);
    assert.equal(seat(b, A).handSize, 2);
});

test('a card this client never saw still moves the count and the pile', () => {
    // Opponent hands carry id references (cambia-509), but a hand model that missed one must
    // not leave the seat showing a card too many.
    const b = board();
    seat(b, A).revealedHand = undefined;
    applySnapSuccess(b, { user: { id: B }, card: { id: 'a-1', idx: 1, user: { id: A } } });
    assert.equal(seat(b, A).handSize, 2);
    assert.equal(b.discardSize, 6);
});

test('an unknown owner leaves every hand alone', () => {
    const b = board();
    applySnapSuccess(b, { user: { id: B }, card: { id: 'x-1', idx: 0, user: { id: 'ghost' } } });
    assert.equal(seat(b, A).handSize, 3);
    assert.equal(seat(b, B).handSize, 4);
    assert.equal(b.discardSize, 6, 'the pile is public and moves regardless');
});

test('hand size never goes negative', () => {
    const b = board();
    seat(b, A).handSize = 0;
    seat(b, A).revealedHand = [];
    applySnapSuccess(b, { user: { id: B }, card: { id: 'a-1', idx: 0, user: { id: A } } });
    assert.equal(seat(b, A).handSize, 0);
});

console.log('snap success: all assertions defined');
