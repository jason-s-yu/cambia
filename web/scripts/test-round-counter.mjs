// Unit test for the round counter's visibility rule (cambia-1126 item 2).
//
// Run:  npm run test:round-counter     (node --test, types stripped from the imported .ts)
//
// The defect this pins: the felt and the results card counted rounds off match state alone. A
// matchmade lobby carries the queue's round count with no circuit playing it, so an h2h_rapid
// game read ROUND 0/8 from the first turn to the last, on both seats, and never moved.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { roundCounterLabel } from '../src/lib/roundCounter.ts';

test('a matchmade lobby with no circuit shows no counter', () => {
    // What h2h_rapid actually delivers: total_rounds 8 off the queue config, current_round 0
    // because no round_start is ever emitted, and no circuit anywhere.
    assert.equal(roundCounterLabel({ circuitEnabled: false, totalRounds: 8, currentRound: 0 }), null);
});

test('a circuit counts its rounds', () => {
    assert.equal(roundCounterLabel({ circuitEnabled: true, totalRounds: 8, currentRound: 3 }), 'Round 3/8');
});

test('a single-round match shows no counter, circuit or not', () => {
    assert.equal(roundCounterLabel({ circuitEnabled: true, totalRounds: 1, currentRound: 1 }), null);
    assert.equal(roundCounterLabel({ circuitEnabled: false, totalRounds: 1, currentRound: 0 }), null);
});

test('a lobby or match the client has not read yet shows no counter', () => {
    // lobbyDetails is null before the first lobby_state, and a casual game has no match state at
    // all. Neither is a reason to print a number.
    assert.equal(roundCounterLabel(null), null);
    assert.equal(roundCounterLabel(undefined), null);
    assert.equal(roundCounterLabel({}), null);
    assert.equal(roundCounterLabel({ circuitEnabled: true }), null, 'a circuit with no round count counts nothing');
});

test('a circuit that has yet to start a round still reads as round zero', () => {
    // Not hidden: the circuit is real, so the counter is the real counter and says where the
    // match is. Only a counter with nothing behind it is dropped.
    assert.equal(roundCounterLabel({ circuitEnabled: true, totalRounds: 4, currentRound: 0 }), 'Round 0/4');
});
