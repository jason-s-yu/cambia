// Unit test for how the rule sheet decides which ruleset a lobby is on (cambia-1123).
//
// Run:  npm run test:preset        (node --test, types stripped from the imported .ts)
//
// The defect this pins: the six queue presets are byte-identical to each other, because
// MATCHMAKING.md 5.2 fixes one ruleset for every ranked queue and a preset's player and round
// counts are not rules. Recognising a preset by comparing house rules therefore named whichever
// one the service listed first, and a lobby created from H2H Rapid opened a sheet reading H2H
// Quick. The service records the id; this module has to prefer it.
//
// The preset fixtures below are read out of the Go source that defines them, not hand-copied, so
// a change to the ranked ruleset or the queue list fails here instead of quietly making the test
// describe a service that no longer exists.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { presetMatchesRules, presetFitsLobby, resolvePresetId, CUSTOM_PRESET_VALUE } from '../src/lib/lobbyPreset.ts';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const PRESETS_GO = read('service/internal/lobby/presets.go');
const QUEUES_GO = read('service/internal/matchmaking/validation.go');

/**
 * The queue ids the service offers as presets, in its own order. Read off the QueueConfigs table
 * so the fixture cannot drift from the queues that exist.
 */
function queueIds() {
    const ids = [...QUEUES_GO.matchAll(/^\s*"([a-z0-9_]+)":\s*\{QueueID:/gm)].map((m) => m[1]);
    assert.ok(ids.length >= 3, 'could not read the queue list out of validation.go');
    return ids;
}

/**
 * The four departures from the default house rules that rankedQueueHouseRules applies. Asserted
 * as a set so this test states the ruleset rather than assuming it.
 */
function rankedDepartures() {
    const body = PRESETS_GO.split('func rankedQueueHouseRules')[1];
    assert.ok(body, 'presets.go no longer defines rankedQueueHouseRules');
    const flags = [...body.matchAll(/hr\.(\w+) = (true|false)\n/g)].map(([, key, value]) => [
        key.charAt(0).toLowerCase() + key.slice(1),
        value === 'true'
    ]);
    assert.deepEqual(
        flags.map(([k]) => k).sort(),
        ['allowDrawFromDiscardPile', 'allowReplaceAbilities', 'lockCallerHand', 'snapRace'],
        'the ranked ruleset changed; this test describes a service that no longer exists'
    );
    return Object.fromEntries(flags);
}

const RANKED_RULES = { turnTimerSec: 15, cardsPerPlayer: 4, ...rankedDepartures() };
const DEFAULT_RULES = {
    turnTimerSec: 15,
    cardsPerPlayer: 4,
    allowDrawFromDiscardPile: false,
    allowReplaceAbilities: false,
    lockCallerHand: true,
    snapRace: false
};

const DEFAULT_PRESET = { id: 'default', gameMode: '', houseRules: DEFAULT_RULES, settings: { autoStart: true } };

/** One preset per queue, all holding the same rules, which is the whole problem. */
const QUEUE_PRESETS = queueIds().map((id) => ({
    id,
    gameMode: id.startsWith('ffa4') ? 'group_of_4' : 'head_to_head',
    houseRules: RANKED_RULES,
    settings: { autoStart: true }
}));

const PRESETS = [DEFAULT_PRESET, ...QUEUE_PRESETS];

const AUTO_START = { autoStart: true };

test('the queue presets really are rule-identical, which is why the id has to travel', () => {
    assert.ok(QUEUE_PRESETS.length >= 3, 'expected at least three queue presets');
    for (const preset of QUEUE_PRESETS) {
        assert.deepEqual(preset.houseRules, QUEUE_PRESETS[0].houseRules);
    }
    assert.notDeepEqual(QUEUE_PRESETS[0].houseRules, DEFAULT_RULES);
});

test('a recorded id names the preset, not the first one holding those rules', () => {
    assert.ok(QUEUE_PRESETS.some((p) => p.id === 'h2h_rapid'), 'h2h_rapid must be a configured queue');
    const firstTwoPlayer = QUEUE_PRESETS.find((p) => p.gameMode === 'head_to_head');
    assert.notEqual(firstTwoPlayer.id, 'h2h_rapid',
        'h2h_rapid is listed first now, so this test can no longer tell the two apart');

    const lobby = { gameMode: 'head_to_head', houseRules: RANKED_RULES, settings: AUTO_START };
    assert.equal(resolvePresetId(PRESETS, { ...lobby, presetId: 'h2h_rapid' }), 'h2h_rapid');
    // The defect, still visible in the fallback: identical rules with nothing recorded name
    // whichever preset the service happened to list first.
    assert.equal(resolvePresetId(PRESETS, lobby), firstTwoPlayer.id);
});

test('with no recorded id the rules answer, and the game mode narrows it', () => {
    const twoPlayer = resolvePresetId(PRESETS, {
        gameMode: 'head_to_head',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    const fourPlayer = resolvePresetId(PRESETS, {
        gameMode: 'group_of_4',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.equal(PRESETS.find((p) => p.id === twoPlayer).gameMode, 'head_to_head');
    assert.equal(PRESETS.find((p) => p.id === fourPlayer).gameMode, 'group_of_4',
        'a 4-player lobby must never be named by a 2-player preset');
    assert.notEqual(twoPlayer, fourPlayer);
});

test('an id naming no preset in the list falls back to the rules', () => {
    const resolved = resolvePresetId(PRESETS, {
        presetId: 'a_queue_that_was_retired',
        gameMode: 'head_to_head',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.equal(PRESETS.find((p) => p.id === resolved).gameMode, 'head_to_head');
});

test("a sheet that is nobody's preset resolves to nothing", () => {
    const resolved = resolvePresetId(PRESETS, {
        gameMode: 'head_to_head',
        houseRules: { ...RANKED_RULES, turnTimerSec: 42 },
        settings: AUTO_START
    });
    assert.equal(resolved, null);
});

test('an empty preset list resolves to nothing, recorded id or not', () => {
    assert.equal(resolvePresetId([], { presetId: 'h2h_rapid', houseRules: RANKED_RULES, settings: AUTO_START }), null);
});

test('the default preset fixes no game mode, so it fits any lobby', () => {
    for (const gameMode of ['head_to_head', 'group_of_4', undefined]) {
        assert.equal(presetFitsLobby(DEFAULT_PRESET, { gameMode, houseRules: DEFAULT_RULES, settings: AUTO_START }), true);
    }
});

test('matching compares the rules a preset names and the auto-start setting', () => {
    const preset = QUEUE_PRESETS[0];
    assert.equal(presetMatchesRules(preset, RANKED_RULES, AUTO_START), true);
    assert.equal(presetMatchesRules(preset, { ...RANKED_RULES, snapRace: false }, AUTO_START), false);
    assert.equal(presetMatchesRules(preset, RANKED_RULES, { autoStart: false }), false);
    // Extra keys the preset says nothing about are not a departure from it.
    assert.equal(presetMatchesRules(preset, { ...RANKED_RULES, numDecks: 2 }, AUTO_START), true);
});

test('matching survives the sheet arriving empty or missing', () => {
    const preset = QUEUE_PRESETS[0];
    for (const sheet of [undefined, null, {}, 'not an object']) {
        assert.equal(presetMatchesRules(preset, sheet, AUTO_START), false);
    }
    assert.equal(presetMatchesRules({ ...preset, houseRules: {} }, RANKED_RULES, AUTO_START), false,
        'a preset carrying no rules must not match every sheet ever rendered');
});

test('the custom value is not a preset id', () => {
    assert.equal(PRESETS.some((p) => p.id === CUSTOM_PRESET_VALUE), false);
});
