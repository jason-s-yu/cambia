// Unit test for the client-held pregame peek (cambia-1094).
//
// Run:  npm run test:pregame-peek        (node --test, types stripped from the imported .ts)
//
// The pregame peek is the physical game's: you see two cards, they go face-down with the rest, and
// the round is played on memory. The server enforces that by hiding every own hand slot in every
// sync_state, which makes private_initial_cards the only frame that ever carries those faces - so
// the client holds them and re-applies them to each snapshot that lands during the window. Without
// that, a peer dropping mid-peek (or any staleness repair) replaces the board with a face-down one
// and ends the peek early; with it applied past the window, the peek would never end at all.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { applyPregamePeek, nextPregamePeek } from '../src/lib/pregamePeek.ts';

const SELF = 'player-self';
const OPP = 'player-opp';

/** A freshly synced two-seat board: every slot face-down, as the server sends it. */
function board() {
    return {
        players: [
            {
                playerId: SELF,
                revealedHand: [
                    { id: 'self-0', known: false, idx: 0 },
                    { id: 'self-1', known: false, idx: 1 },
                    { id: 'self-2', known: false, idx: 2 },
                    { id: 'self-3', known: false, idx: 3 }
                ]
            },
            {
                playerId: OPP,
                revealedHand: [
                    { id: 'opp-0', known: false, idx: 0 },
                    { id: 'opp-1', known: false, idx: 1 }
                ]
            }
        ]
    };
}

/** The two faces private_initial_cards carried, as the store holds them. */
function peek() {
    return [
        { id: 'self-0', rank: 'K', suit: 'S', value: 13, idx: 0, ownerId: SELF },
        { id: 'self-2', rank: '9', suit: 'H', value: 9, idx: 2, ownerId: SELF }
    ];
}

test('the peeked slots come back face-up and nothing else does', () => {
    const b = board();
    assert.equal(applyPregamePeek(b, peek(), SELF), 2);

    const hand = b.players[0].revealedHand;
    assert.deepEqual(hand[0], { id: 'self-0', known: true, rank: 'K', suit: 'S', value: 13, idx: 0 });
    assert.deepEqual(hand[2], { id: 'self-2', known: true, rank: '9', suit: 'H', value: 9, idx: 2 });
    assert.deepEqual(hand[1], { id: 'self-1', known: false, idx: 1 });
    assert.deepEqual(hand[3], { id: 'self-3', known: false, idx: 3 });
});

test('the slot keeps the id and index the board gave it, not the peek', () => {
    // The board is authoritative for where a card sits; only the face comes from the peek. A peek
    // entry carrying a stale index must not renumber the slot it lands on.
    const b = board();
    const stale = [{ id: 'self-2', rank: '9', suit: 'H', value: 9, idx: 0, ownerId: SELF }];
    assert.equal(applyPregamePeek(b, stale, SELF), 1);

    const hand = b.players[0].revealedHand;
    assert.equal(hand[2].idx, 2, 'the slot keeps its own index');
    assert.equal(hand[2].rank, '9');
    assert.equal(hand[0].known, false, 'the peek must not land on the slot its stale index named');
});

test('a face is never painted onto a card the peek does not name', () => {
    // Id-only matching, deliberately: unlike a removal, a miss here would put a face on the wrong
    // card, and a player would go on to play against it. A back is the safe render.
    const b = board();
    const gone = [{ id: 'self-9', rank: 'A', suit: 'C', value: 1, idx: 1, ownerId: SELF }];
    assert.equal(applyPregamePeek(b, gone, SELF), 0);
    for (const c of b.players[0].revealedHand) assert.equal(c.known, false);
});

test('a peek that names one live card and one missing card applies the live one', () => {
    const b = board();
    const mixed = [
        { id: 'self-9', rank: 'A', suit: 'C', value: 1, idx: 0, ownerId: SELF },
        { id: 'self-1', rank: '4', suit: 'D', value: 4, idx: 1, ownerId: SELF }
    ];
    assert.equal(applyPregamePeek(b, mixed, SELF), 1);
    assert.equal(b.players[0].revealedHand[1].rank, '4');
    assert.equal(b.players[0].revealedHand[0].known, false);
});

test('opponent hands are never touched', () => {
    const b = board();
    const crossSeat = [{ id: 'opp-0', rank: 'Q', suit: 'D', value: 12, idx: 0, ownerId: SELF }];
    assert.equal(applyPregamePeek(b, crossSeat, SELF), 0, 'a peek only ever faces the self hand');
    for (const c of b.players[1].revealedHand) assert.equal(c.known, false);
});

test('an empty peek leaves the board exactly as it found it', () => {
    // This is what a client that never received private_initial_cards holds, and what every seat
    // holds once the window closes. It must be a no-op, not a wipe.
    const b = board();
    const before = JSON.stringify(b);
    assert.equal(applyPregamePeek(b, [], SELF), 0);
    assert.equal(JSON.stringify(b), before);
});

test('a board with no self seat, no hand, or no viewer is a no-op', () => {
    assert.equal(applyPregamePeek(board(), peek(), 'nobody'), 0);
    assert.equal(applyPregamePeek(board(), peek(), null), 0);
    assert.equal(applyPregamePeek(null, peek(), SELF), 0);
    assert.equal(applyPregamePeek(undefined, peek(), SELF), 0);
    assert.equal(applyPregamePeek({ players: [{ playerId: SELF }] }, peek(), SELF), 0);
});

test('re-applying across successive syncs holds the peek up for the whole window', () => {
    // The window is as many snapshots long as it takes: a peer dropping, a staleness repair and the
    // reconnect resync each replace the board, and the same held peek has to survive all of them.
    const held = peek();
    for (let sync = 0; sync < 3; sync++) {
        const fresh = board();
        assert.equal(applyPregamePeek(fresh, held, SELF), 2, `sync ${sync} should re-face both slots`);
        assert.equal(fresh.players[0].revealedHand[0].rank, 'K');
        assert.equal(fresh.players[0].revealedHand[2].rank, '9');
    }
    // The store drops the held peek on the first snapshot with preGameActive false; from then on
    // the same board is left face-down, which is what turns the cards over at game start.
    const afterStart = board();
    assert.equal(applyPregamePeek(afterStart, [], SELF), 0);
    for (const c of afterStart.players[0].revealedHand) assert.equal(c.known, false);
});

// gameStore.ts drives the drop above through nextPregamePeek: private_sync_state assigns
// `state.pregamePeek = nextPregamePeek(state.pregamePeek, payload.state)` ahead of the
// re-apply, rather than checking preGameActive itself. These drive that helper directly with
// the same inputs the store passes it.

test('nextPregamePeek holds the peek while the pregame window is open', () => {
    const held = peek();
    assert.deepEqual(nextPregamePeek(held, { preGameActive: true }), held);
});

test('nextPregamePeek clears the peek on the post-start sync', () => {
    // This is the sync StartGame broadcasts: the first snapshot with preGameActive false. It is
    // what turns the peeked cards face-down on screen.
    assert.deepEqual(nextPregamePeek(peek(), { preGameActive: false }), []);
});

test('nextPregamePeek clears the peek for a sync that omits preGameActive, same as false', () => {
    assert.deepEqual(nextPregamePeek(peek(), {}), []);
    assert.deepEqual(nextPregamePeek(peek(), null), []);
    assert.deepEqual(nextPregamePeek(peek(), undefined), []);
});
