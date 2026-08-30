// Unit test for the client's reading of the LockCallerHand house rule (cambia-1069).
//
// Run:  npm run test:lock        (node --test, types stripped from the imported .ts)
//
// The felt used to offer moves the server refuses: after a Cambia call under lockCallerHand the
// caller's cards stayed clickable as snap and swap targets, and the caller's own cards stayed
// clickable as snap picks. The server answers both with a no-penalty refusal on the snap path
// (service/internal/game/engine_adapter.go handLocked, cambia-1043), delivered as the public
// player_snap_fail the table renders as nothing happening. These are the predicates the table
// gates its targets on.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { lockedPlayerId, isHandLocked, canSnapCard } from '../src/lib/handLock.ts';

const A = 'player-a';
const B = 'player-b';
const C = 'player-c';

/** A three-seat board where A has called Cambia and the rule is on. */
function board(overrides = {}) {
    return {
        cambiaCalled: true,
        cambiaCallerId: A,
        players: [
            { playerId: A, hasCalledCambia: true },
            { playerId: B, hasCalledCambia: false },
            { playerId: C, hasCalledCambia: false }
        ],
        houseRules: { lockCallerHand: true },
        ...overrides
    };
}

// --- Which hand is locked ---------------------------------------------------------------

test('the Cambia caller is the locked seat', () => {
    const b = board();
    assert.equal(lockedPlayerId(b), A);
    assert.equal(isHandLocked(b, A), true);
    assert.equal(isHandLocked(b, B), false);
    assert.equal(isHandLocked(b, C), false);
});

test('nothing is locked before Cambia is called', () => {
    const b = board({ cambiaCalled: false, cambiaCallerId: null, players: [{ playerId: A }, { playerId: B }, { playerId: C }] });
    assert.equal(lockedPlayerId(b), null);
    assert.equal(isHandLocked(b, A), false);
});

test('nothing is locked with the rule off', () => {
    const b = board({ houseRules: { lockCallerHand: false } });
    assert.equal(lockedPlayerId(b), null);
    assert.equal(isHandLocked(b, A), false);
    assert.equal(canSnapCard(b, B, A), true);
});

test('an absent lockCallerHand reads as on, the way the server defaults it', () => {
    const b = board({ houseRules: {} });
    assert.equal(lockedPlayerId(b), A);
});

test('the caller is found from hasCalledCambia when no caller id arrived', () => {
    const b = board({ cambiaCallerId: undefined });
    assert.equal(lockedPlayerId(b), A);
    assert.equal(isHandLocked(b, A), true);
});

test('no seat is locked when the caller id names nobody this client knows', () => {
    const b = board({ cambiaCallerId: null, players: [{ playerId: A }, { playerId: B }, { playerId: C }] });
    assert.equal(lockedPlayerId(b), null);
});

test('an absent player id is never locked', () => {
    const b = board();
    assert.equal(isHandLocked(b, null), false);
    assert.equal(isHandLocked(b, undefined), false);
});

// --- Snap gating ------------------------------------------------------------------------

test('nobody can snap a card out of the locked hand', () => {
    const b = board();
    assert.equal(canSnapCard(b, B, A), false);
    assert.equal(canSnapCard(b, C, A), false);
});

test('the caller cannot snap at all, their own hand included', () => {
    const b = board();
    assert.equal(canSnapCard(b, A, A), false);
    assert.equal(canSnapCard(b, A, B), false);
    assert.equal(canSnapCard(b, A, C), false);
});

test('every other snap is still allowed', () => {
    const b = board();
    assert.equal(canSnapCard(b, B, B), true);
    assert.equal(canSnapCard(b, B, C), true);
    assert.equal(canSnapCard(b, C, B), true);
});
