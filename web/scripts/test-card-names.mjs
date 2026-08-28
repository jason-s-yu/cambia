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

import { toDsCardFace, cardFaceName } from '../src/components/game/dsCardMap.ts';

const name = (rank, suit) => cardFaceName(toDsCardFace({ id: 'c', known: true, rank, suit }));

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

console.log('card name checks loaded');
