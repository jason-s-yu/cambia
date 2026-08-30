// Unit test for own-hand row placement (cambia-1095).
//
// Run:  npm run test:hand-layout    (node --test, types stripped from the imported .ts)
//
// The engine peeks hand slots 0 and 1 at the deal (engine/game.go, RULES.md section 2), so those
// two slots are the row nearest their owner. The own hand used to be drawn row-major into a
// two-column grid, which put the peeked pair on the TOP row and the unseen pair on the bottom.
//
// The web tree has no component renderer (no vitest, jsdom or testing-library, and node's type
// stripping does not transform JSX), so the placement itself is tested through the pure module the
// table calls, and the wiring is checked against the source of DsGameTable.tsx.

/* global console, URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';

import { ownHandPlacement, handRowCount } from '../src/components/game/handLayout.ts';

const TABLE = new URL('../src/components/game/DsGameTable.tsx', import.meta.url);

test('a four card hand draws the peeked pair on the bottom row', () => {
    // Grid rows count from the top, so a larger row number is further down the screen.
    assert.deepEqual(ownHandPlacement(0, 4), { gridRow: 2, gridColumn: 1 });
    assert.deepEqual(ownHandPlacement(1, 4), { gridRow: 2, gridColumn: 2 });
    assert.deepEqual(ownHandPlacement(2, 4), { gridRow: 1, gridColumn: 1 });
    assert.deepEqual(ownHandPlacement(3, 4), { gridRow: 1, gridColumn: 2 });
    assert.ok(ownHandPlacement(0, 4).gridRow > ownHandPlacement(2, 4).gridRow, 'slot 0 sits below slot 2');
});

test('slots pair off left to right within their row', () => {
    assert.equal(ownHandPlacement(0, 4).gridRow, ownHandPlacement(1, 4).gridRow);
    assert.equal(ownHandPlacement(2, 4).gridRow, ownHandPlacement(3, 4).gridRow);
    assert.equal(ownHandPlacement(0, 4).gridColumn, 1);
    assert.equal(ownHandPlacement(1, 4).gridColumn, 2);
});

test('penalty cards extend the square away from the owner', () => {
    // A snap penalty grows the hand past four (cambia-820); slots 0 and 1 stay nearest.
    assert.equal(handRowCount(6), 3);
    assert.equal(ownHandPlacement(0, 6).gridRow, 3);
    assert.equal(ownHandPlacement(2, 6).gridRow, 2);
    assert.equal(ownHandPlacement(4, 6).gridRow, 1);
    for (const slot of [1, 2, 3, 4, 5]) {
        assert.ok(ownHandPlacement(0, 6).gridRow >= ownHandPlacement(slot, 6).gridRow, 'slot 0 must stay nearest, slot ' + slot);
    }
});

test('an odd hand size leaves the gap on the far row', () => {
    assert.equal(handRowCount(5), 3);
    assert.equal(ownHandPlacement(0, 5).gridRow, 3);
    assert.deepEqual(ownHandPlacement(4, 5), { gridRow: 1, gridColumn: 1 });
});

test('every row is a real grid line, even for a hand of one or none', () => {
    assert.equal(handRowCount(0), 1);
    assert.equal(ownHandPlacement(0, 0).gridRow, 1);
    assert.equal(ownHandPlacement(0, 1).gridRow, 1);
    // A slot drawn past the counted hand size still lands on a positive line.
    assert.ok(ownHandPlacement(3, 1).gridRow >= 1);
});

test('the table places its own cards with the helper and keeps the slot keyed hooks', () => {
    const src = readFileSync(TABLE, 'utf8');
    assert.ok(src.includes("import { ownHandPlacement } from './handLayout';"), 'DsGameTable does not import the placement helper');
    // The known cards and the padding backs both take a placement.
    assert.ok(src.match(/style=\{ownHandPlacement\(/g).length >= 2, 'own hand cards are not all placed');
    // The testId and the spoken name stay keyed by the engine slot index, not by draw order.
    assert.ok(src.includes('testId={`card-${seat}-${i}`}'), 'own card testId is no longer the engine slot index');
    assert.ok(src.includes('`Your card ${i + 1}'), 'own card name is no longer the engine slot index');
    // Opponents keep plain row-major order: their near row is the top one on screen.
    const opponentGrid = src.slice(src.indexOf('{opponents.map('), src.indexOf('{opponents.length === 0'));
    assert.ok(opponentGrid.length > 0, 'opponent hand block not found');
    assert.ok(!opponentGrid.includes('ownHandPlacement('), 'opponent cards must not take the own-hand placement');
});

console.log('hand layout checks loaded');
