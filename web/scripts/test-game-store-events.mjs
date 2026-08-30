// Tests for the gameStore event branches the replace-ability fix changed (cambia-1125).
//
// Run:  npm run test:game-store-events      (node --test)
//
// The helpers in src/lib/specialPrompt.ts are unit-tested next door, but a helper only holds a
// rule if the store actually routes through it. Each of these three branches was a place the store
// wrote a fact it did not own:
//
//   - private_draw_stockpile counted the drawer's own draw against the piles a second time, so the
//     drawer's stockpile ran one card behind the rest of the table until the next public draw.
//   - private_special_action_fail is the server's refusal for ANY action; turning every one of them
//     into a special_action prompt put a rankless prompt and a dead Skip button on a settled table.
//   - a special action that outlived game_player_turn kept prompting for an ability nobody owed.
//
// The store cannot be imported here: it pulls in zustand, immer and the '@/' alias, none of which
// node --test resolves. So the branch bodies are read out of the source and held to their shape,
// and the helper they delegate to is driven directly with the payloads the server sends.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { applyDrawPileCounts, pendingActionAfterFail } from '../src/lib/specialPrompt.ts';

const STORE_PATH = fileURLToPath(new URL('../src/stores/gameStore.ts', import.meta.url));
const STORE = readFileSync(STORE_PATH, 'utf8');
const LINES = STORE.split('\n');

/** The line index of `case '<name>':`, which must appear exactly once. */
function caseLine(name) {
    const re = new RegExp(`^\\s*case '${name}':`);
    const hits = LINES.map((l, i) => (re.test(l) ? i : -1)).filter((i) => i >= 0);
    assert.equal(hits.length, 1, `expected exactly one 'case ${name}:' in gameStore.ts, found ${hits.length}`);
    return hits[0];
}

/**
 * The body of a switch case: every line after its label, up to the next label at the same
 * indentation. A label whose body is empty falls through to the next one.
 */
function caseBody(name) {
    const start = caseLine(name);
    const indent = LINES[start].match(/^\s*/)[0];
    const boundary = new RegExp(`^${indent}(case '|default:)`);
    const out = [];
    for (let i = start + 1; i < LINES.length; i++) {
        if (boundary.test(LINES[i])) break;
        out.push(LINES[i]);
    }
    return out.join('\n');
}

// --- private_draw_stockpile must not touch the pile counts -------------------------------

test('the two draw events share one body that delegates the counts to applyDrawPileCounts', () => {
    // The public event's label falls straight through to the private one, so there is a single
    // body and a single place the counts can be written from.
    assert.equal(caseBody('player_draw_stockpile').trim(), '', 'the public draw label falls through');
    assert.equal(caseLine('private_draw_stockpile'), caseLine('player_draw_stockpile') + 1);

    const body = caseBody('private_draw_stockpile');
    const calls = body.match(/applyDrawPileCounts\(/g) ?? [];
    assert.equal(calls.length, 1, 'the shared draw body applies the pile counts exactly once');
    assert.match(body, /applyDrawPileCounts\(state\.gameState, type, payload\.payload\)/);
});

test('the draw body writes no pile count of its own', () => {
    // A local decrement next to the delegated write is the double count coming back.
    const body = caseBody('private_draw_stockpile');
    assert.doesNotMatch(body, /\bstockpileSize\s*=/, 'the draw branch may not assign stockpileSize');
    assert.doesNotMatch(body, /\bdiscardSize\s*=/, 'nor discardSize');
    assert.doesNotMatch(body, /\bstockpileSize\s*(--|\+\+|-=|\+=)/, 'nor step it');
});

test('the private draw leaves the counts the public draw set', () => {
    const piles = { stockpileSize: 37, discardSize: 3, discardTop: { id: 'top' } };

    applyDrawPileCounts(piles, 'player_draw_stockpile', { source: 'stockpile', stockpileSize: 36 });
    assert.equal(piles.stockpileSize, 36, 'the public event carries the server count');

    applyDrawPileCounts(piles, 'private_draw_stockpile', { source: 'stockpile', stockpileSize: 36 });
    assert.equal(piles.stockpileSize, 36, 'and the private twin of the same draw moves nothing');
    assert.equal(piles.discardSize, 3);
});

// --- private_special_action_fail must not invent a prompt --------------------------------

test('the refusal branch delegates the prompt to pendingActionAfterFail', () => {
    const body = caseBody('private_special_action_fail');
    assert.match(
        body,
        /state\.pendingAction = pendingActionAfterFail\(state\.pendingAction, state\.gameState\?\.specialAction, selfPlayerId\)/,
        'the refusal asks the helper what the prompt should be'
    );
    assert.doesNotMatch(
        body,
        /pendingAction\s*=\s*'special_action'/,
        'and never sets the prompt from the refusal itself'
    );
});

test('a refusal on a table with no pending ability leaves no prompt behind', () => {
    // The dead-Skip-button shape: the server had already settled the ability, the player clicked
    // the deck, the engine refused the draw.
    assert.equal(pendingActionAfterFail(null, null, 'player-self'), null);
    assert.equal(pendingActionAfterFail('special_action', null, 'player-self'), null);
    // Another player's pending ability is not this client's prompt either.
    assert.equal(
        pendingActionAfterFail(null, { active: true, playerId: 'player-opp' }, 'player-self'),
        null
    );
});

// --- game_player_turn clears a stale prompt ----------------------------------------------

test('a new turn clears the special action', () => {
    // The server never announces a turn over a pending ability, so one that survives this event
    // belongs to a turn that is over.
    const body = caseBody('game_player_turn');
    assert.match(body, /state\.gameState\.specialAction = null/, 'the turn clears the special action');
    assert.match(body, /state\.pendingAction = null/, 'and the pending-action tag with it');
});
