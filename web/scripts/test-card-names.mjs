// Unit test for the spoken card face used in every table accessible name (cambia-959).
//
// Run:  npm run test:names      (node --test, types stripped from the imported .ts)
//
// The table names cards owner, position, face: 'Your card 2: 9 of clubs'. A screen reader
// reads the printed index as a letter, so the face fragment spells the rank out and the
// engine's T and O are already normalized by toDsCardFace.

/* global console */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { toDsCardFace, cardFaceName, cardSlotName, LOCKED_SUFFIX, FORFEITED_SUFFIX } from '../src/components/game/dsCardMap.ts';

const name = (rank, suit) => cardFaceName(toDsCardFace({ id: 'c', known: true, rank, suit }));

/** The face a slot is drawing during a reveal: what revealById hands renderHand. */
const shown = (rank, suit) => toDsCardFace({ id: 'c', known: true, rank, suit });

test('number ranks read as the number and the long suit', () => {
    assert.equal(name('9', 'C'), '9 of clubs');
    assert.equal(name('2', 'H'), '2 of hearts');
    assert.equal(name('T', 'D'), '10 of diamonds');
});

test('court ranks and the ace are spoken, not printed', () => {
    assert.equal(name('A', 'S'), 'ace of spades');
    assert.equal(name('J', 'H'), 'jack of hearts');
    assert.equal(name('Q', 'D'), 'queen of diamonds');
    assert.equal(name('K', 'S'), 'king of spades');
});

test('a joker has no suit to read', () => {
    assert.equal(name('O', 'R'), 'joker');
    assert.equal(name('O', 'B'), 'joker');
});

test('a rank with no suit still names the rank', () => {
    assert.equal(name('K', undefined), 'king');
});

test('an unknown card has no face to name, so the caller says face down', () => {
    assert.equal(cardFaceName(toDsCardFace(null)), null);
    assert.equal(cardFaceName(toDsCardFace(undefined)), null);
    assert.equal(cardFaceName(toDsCardFace({ id: 'c', known: false })), null);
});

// --- Slot names -------------------------------------------------------------------------
//
// The name a hand slot carries has to follow the face the slot is drawing. During the transient
// reveal of an own card (a replace, a 7/8 peek, a King look) the slot goes face-up for the hold,
// and a name still reading 'face down' would tell a screen reader the opposite of what is on the
// felt (cambia-1094, cambia-1124).

test('a revealed face names the card, on either side of the table', () => {
    assert.equal(cardSlotName('Your', 0, shown('9', 'H')), 'Your card 1: 9 of hearts');
    assert.equal(cardSlotName('Your', 2, shown('K', 'S')), 'Your card 3: king of spades');
    assert.equal(cardSlotName('Guest-AA0883AF', 1, shown('T', 'D')), 'Guest-AA0883AF card 2: 10 of diamonds');
});

test('the slot goes back to face down when the reveal ends', () => {
    assert.equal(cardSlotName('Your', 0, null), 'Your card 1, face down');
    assert.equal(cardSlotName('Your', 0, toDsCardFace({ id: 'c', known: false })), 'Your card 1, face down');
    assert.equal(cardSlotName('Guest-AA0883AF', 3, null), 'Guest-AA0883AF card 4, face down');
});

test('slots are named by their 1-based engine slot index', () => {
    // Placement puts slots 0 and 1 on the row nearest their owner, but the spoken index stays the
    // engine slot, so the names run 1, 2, 3, 4 in DOM order (cambia-1095).
    const hand = [null, null, null, null];
    assert.deepEqual(
        hand.map((face, i) => cardSlotName('Your', i, face)),
        ['Your card 1, face down', 'Your card 2, face down', 'Your card 3, face down', 'Your card 4, face down']
    );
});

test('a locked hand keeps the reason it went inert, revealed or not', () => {
    assert.equal(cardSlotName('Your', 0, shown('A', 'S'), 'locked'), 'Your card 1: ace of spades' + LOCKED_SUFFIX);
    assert.equal(cardSlotName('Your', 0, null, 'locked'), 'Your card 1, face down' + LOCKED_SUFFIX);
    assert.equal(LOCKED_SUFFIX, ', locked after calling Cambia');
});

// A seat whose reconnect window closed keeps its cards on the felt to watch with, so the slot
// itself has to say the score behind it stopped counting (cambia-1468, carried from cambia-1237).
test('a forfeited seat says its score stopped counting, revealed or not', () => {
    assert.equal(cardSlotName('Your', 0, shown('A', 'S'), 'forfeited'), 'Your card 1: ace of spades' + FORFEITED_SUFFIX);
    assert.equal(cardSlotName('Rival', 2, null, 'forfeited'), 'Rival card 3, face down' + FORFEITED_SUFFIX);
    assert.equal(FORFEITED_SUFFIX, ', seat forfeited, not scored');
});

// The two states are one suffix, so a seat that called Cambia and then forfeited says the thing
// that outranks: an unscored seat has nothing left for the lock to add.
test('a slot with no state named reads plainly', () => {
    assert.equal(cardSlotName('Your', 0, null, 'live'), 'Your card 1, face down');
    assert.equal(cardSlotName('Your', 0, null), 'Your card 1, face down');
});

console.log('card name checks loaded');
