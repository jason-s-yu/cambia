// Unit tests for the client's special-action prompt and draw-pile counting (cambia-1125).
//
// Run:  npm run test:special-prompt      (node --test, types stripped from the imported .ts)
//
// Ranked queues play allowReplaceAbilities, so a replace that puts a 7/8/9/T/J/Q/K on the pile
// arms an ability the engine will not let its owner decline. Two client faults turned that into a
// dead table:
//
//   - private_special_action_fail is the server's refusal for ANY action, and the store turned
//     every one of them into a special-action prompt. After the server had already settled the
//     ability, a refused draw put a rankless prompt back on screen: the ordinary "draw from the
//     stock" line with a Skip button under it, wired to a skip the server refused again.
//   - the private draw event was counted against the piles a second time, so the drawer's own
//     screen showed the stockpile one card light until the next public draw overwrote it. That is
//     the 36-vs-37 the two clients disagreed about.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
    applyDrawPileCounts,
    canSkipSpecialAction,
    pendingActionAfterFail,
    pendingActionForSpecial
} from '../src/lib/specialPrompt.ts';

const SELF = 'player-self';
const OPP = 'player-opp';

// --- The skip affordance ----------------------------------------------------------------

test('an ability the player chose to invoke can be skipped', () => {
    assert.equal(canSkipSpecialAction({ active: true, playerId: SELF, cardRank: '7' }), true);
});

test('an ability the engine armed off a replace cannot be skipped', () => {
    assert.equal(
        canSkipSpecialAction({ active: true, playerId: SELF, cardRank: 'T', mandatory: true }),
        false
    );
});

test('no special action means no skip', () => {
    assert.equal(canSkipSpecialAction(null), false);
    assert.equal(canSkipSpecialAction(undefined), false);
    assert.equal(canSkipSpecialAction({ active: false, playerId: SELF, cardRank: 'T' }), false);
});

// --- The prompt a refusal may and may not restore ---------------------------------------

test('a refusal with no pending ability does not invent a prompt', () => {
    // The wedge shape: the server had settled the ability, the player clicked the deck, and the
    // engine refused the draw. Nothing is pending, so nothing goes back on screen.
    assert.equal(pendingActionAfterFail(null, null, SELF), null);
    assert.equal(pendingActionAfterFail('discard_replace', null, SELF), 'discard_replace');
});

test('a refusal drops a prompt the server has already settled', () => {
    assert.equal(pendingActionAfterFail('special_action', null, SELF), null);
    assert.equal(
        pendingActionAfterFail('special_action', { active: false, playerId: SELF, cardRank: 'T' }, SELF),
        null
    );
});

test('a refused target keeps the prompt so the player can pick another', () => {
    const special = { active: true, playerId: SELF, cardRank: 'T', mandatory: true };
    assert.equal(pendingActionAfterFail('special_action', special, SELF), 'special_action');
    assert.equal(pendingActionAfterFail(null, special, SELF), 'special_action');
});

test("another player's pending ability is not this client's prompt", () => {
    const special = { active: true, playerId: OPP, cardRank: 'T' };
    assert.equal(pendingActionAfterFail(null, special, SELF), null);
    assert.equal(pendingActionForSpecial(special, SELF), null);
});

// --- The prompt a sync restores ----------------------------------------------------------

test("a sync taken mid-ability restores this client's prompt", () => {
    const special = { active: true, playerId: SELF, cardRank: 'T', mandatory: true };
    assert.equal(pendingActionForSpecial(special, SELF), 'special_action');
});

test('a sync taken after the ability settled restores no prompt', () => {
    assert.equal(pendingActionForSpecial(null, SELF), null);
    assert.equal(pendingActionForSpecial(undefined, SELF), null);
});

// --- The piles --------------------------------------------------------------------------

/** A live board mid-game: eight cards dealt, seven turns played. */
function piles() {
    return { stockpileSize: 38, discardSize: 8, discardTop: { id: 'discard-top' } };
}

test('the public stockpile draw takes the count the server sent', () => {
    const p = piles();
    applyDrawPileCounts(p, 'player_draw_stockpile', { source: 'stockpile', stockpileSize: 37 });
    assert.equal(p.stockpileSize, 37);
});

test("the drawer's private twin does not count the same draw again", () => {
    // The bug this file exists for: the drawer saw 36 while the rest of the table saw 37.
    const p = piles();
    applyDrawPileCounts(p, 'player_draw_stockpile', { source: 'stockpile', stockpileSize: 37 });
    applyDrawPileCounts(p, 'private_draw_stockpile', { source: 'stockpile' });
    assert.equal(p.stockpileSize, 37);
});

test('a discard-pile draw moves the discard pile, not the stockpile', () => {
    const p = piles();
    applyDrawPileCounts(p, 'player_draw_stockpile', { source: 'discardpile', discardSize: 7 });
    applyDrawPileCounts(p, 'private_draw_stockpile', { source: 'discardpile' });
    assert.equal(p.discardSize, 7);
    assert.equal(p.stockpileSize, 38);
    assert.equal(p.discardTop, null, 'the card that was on top has left the pile');
});

test('a public draw with no count in the payload still moves by one', () => {
    const p = piles();
    applyDrawPileCounts(p, 'player_draw_stockpile', { source: 'stockpile' });
    assert.equal(p.stockpileSize, 37);
});
