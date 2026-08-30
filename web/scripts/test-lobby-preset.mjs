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

import {
    presetMatchesRules,
    presetFitsLobby,
    resolvePresetId,
    rulesetRow,
    CUSTOM_PRESET_LABEL,
    CUSTOM_PRESET_VALUE
} from '../src/lib/lobbyPreset.ts';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const PRESETS_GO = read('service/internal/lobby/presets.go');
const QUEUES_GO = read('service/internal/matchmaking/validation.go');

/**
 * The queues the service offers as presets, in its own order, each with the display name the
 * preset carries (lobby/presets.go queuePreset). Read off the QueueConfigs table so the fixture
 * cannot drift from the queues that exist.
 */
function queues() {
    const found = [...QUEUES_GO.matchAll(/^\s*"([a-z0-9_]+)":\s*\{QueueID: "[a-z0-9_]+", DisplayName: "([^"]+)"/gm)]
        .map(([, id, name]) => ({ id, name }));
    assert.ok(found.length >= 3, 'could not read the queue list out of validation.go');
    return found;
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

const DEFAULT_PRESET = {
    id: 'default',
    name: 'Default',
    description: 'The rules a new lobby starts with.',
    gameMode: '',
    houseRules: DEFAULT_RULES,
    settings: { autoStart: true }
};

/** One preset per queue, all holding the same rules, which is the whole problem. */
const QUEUE_PRESETS = queues().map(({ id, name }) => ({
    id,
    name,
    description: 'Ranked queue rules. A custom lobby plays a single round.',
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

test('a recorded id from another table size does not name the lobby', () => {
    // A head-to-head lobby recorded as ffa4_standard is a record no lobby can satisfy: applying a
    // preset used to fill the rules and nothing else, so the id could be recorded against a lobby
    // of the wrong size and then head the sheet FFA-4 Standard beside a Head to head badge
    // (cambia-1099 Q1). The rules answer instead.
    const fourPlayer = QUEUE_PRESETS.find((p) => p.gameMode === 'group_of_4');
    const resolved = resolvePresetId(PRESETS, {
        presetId: fourPlayer.id,
        gameMode: 'head_to_head',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.notEqual(resolved, fourPlayer.id);
    assert.equal(PRESETS.find((p) => p.id === resolved).gameMode, 'head_to_head');
});

test('a recorded id from another table size on rules of its own reads Custom', () => {
    // Nothing to fall back to: the id cannot be true and the rules are nobody's preset either.
    const fourPlayer = QUEUE_PRESETS.find((p) => p.gameMode === 'group_of_4');
    assert.equal(resolvePresetId(PRESETS, {
        presetId: fourPlayer.id,
        gameMode: 'head_to_head',
        houseRules: { ...RANKED_RULES, turnTimerSec: 42 },
        settings: AUTO_START
    }), null);
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

// The Ruleset row (cambia-1123). The row was gated on the viewer being able to change it, so a
// matchmade lobby - locked by definition, and the one sheet whose ruleset the player never chose
// - showed the queue's rules with nothing naming them.

/** A matchmade lobby's sheet as every seat reads it: locked, so nobody can edit. */
const MATCHMADE = {
    presets: PRESETS,
    canEdit: false,
    saved: {
        presetId: 'h2h_rapid',
        gameMode: 'head_to_head',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    }
};

test('a locked sheet names the queue ruleset and offers no selector', () => {
    const row = rulesetRow(MATCHMADE);
    assert.equal(row.kind, 'name', 'a locked sheet must still render the Ruleset row');
    assert.equal(row.name, PRESETS.find((p) => p.id === 'h2h_rapid').name);
    assert.equal(row.options, undefined, 'a viewer who cannot edit is offered no choices');
    assert.equal(row.value, undefined, 'nothing on a locked row is a control value');
});

test('the locked name is the label the host would have picked, not the id', () => {
    const row = rulesetRow(MATCHMADE);
    const host = rulesetRow({
        ...MATCHMADE,
        canEdit: true,
        selectedId: 'h2h_rapid',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.equal(host.kind, 'select');
    assert.equal(row.name, host.options.find((o) => o.value === 'h2h_rapid').label);
    assert.notEqual(row.name, 'h2h_rapid', 'the row must show the display name, never the queue id');
});

test('a locked sheet is named by the recorded id, not by the first preset holding those rules', () => {
    const firstTwoPlayer = QUEUE_PRESETS.find((p) => p.gameMode === 'head_to_head');
    assert.notEqual(firstTwoPlayer.id, 'h2h_rapid', 'this test can no longer tell the two apart');
    assert.equal(rulesetRow(MATCHMADE).name, PRESETS.find((p) => p.id === 'h2h_rapid').name);
});

test('a locked sheet on nobody\'s preset reads Custom', () => {
    const row = rulesetRow({
        ...MATCHMADE,
        saved: { gameMode: 'head_to_head', houseRules: { ...RANKED_RULES, turnTimerSec: 42 }, settings: AUTO_START }
    });
    assert.deepEqual(row, { kind: 'name', name: CUSTOM_PRESET_LABEL });
});

test('an unread preset list drops the row rather than calling the lobby Custom', () => {
    for (const canEdit of [true, false]) {
        assert.deepEqual(rulesetRow({ ...MATCHMADE, presets: [], canEdit }), { kind: 'none' });
    }
});

test('a guest in an unlocked lobby reads the name the host set', () => {
    const row = rulesetRow({
        presets: PRESETS,
        canEdit: false,
        saved: { presetId: 'default', gameMode: 'head_to_head', houseRules: DEFAULT_RULES, settings: AUTO_START }
    });
    assert.deepEqual(row, { kind: 'name', name: DEFAULT_PRESET.name });
});

test('the host of an unlocked lobby keeps the selector, Custom included', () => {
    const onPreset = rulesetRow({
        presets: PRESETS,
        canEdit: true,
        saved: MATCHMADE.saved,
        selectedId: 'h2h_rapid',
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.equal(onPreset.kind, 'select');
    assert.equal(onPreset.value, 'h2h_rapid');
    const fitsTwoPlayer = PRESETS.filter((p) => !p.gameMode || p.gameMode === 'head_to_head');
    assert.deepEqual(onPreset.options.map((o) => o.value), fitsTwoPlayer.map((p) => p.id),
        'a sheet still on its preset is offered the presets it could be on and no Custom entry');
    assert.equal(onPreset.description, PRESETS.find((p) => p.id === 'h2h_rapid').description);

    // One edited rule departs from the preset: the select falls to Custom, which has to be an
    // option before it can be the value.
    const departed = rulesetRow({
        presets: PRESETS,
        canEdit: true,
        saved: MATCHMADE.saved,
        selectedId: 'h2h_rapid',
        houseRules: { ...RANKED_RULES, snapRace: false },
        settings: AUTO_START
    });
    assert.equal(departed.value, CUSTOM_PRESET_VALUE);
    assert.equal(departed.options.at(-1).label, CUSTOM_PRESET_LABEL);
    assert.equal(departed.description, '', 'a departed sheet has no preset description to show');
});

// The table size a preset carries (cambia-1099 Q1). Applying one fills the rules and the
// auto-start setting; it does not reseat the lobby, so a preset naming a player count the lobby
// does not have is not a ruleset it can be on, whatever its rules say.

test('the host is offered only the presets the lobby could be on', () => {
    const offered = (gameMode) => rulesetRow({
        presets: PRESETS,
        canEdit: true,
        saved: { gameMode, houseRules: RANKED_RULES, settings: AUTO_START },
        selectedId: null,
        houseRules: RANKED_RULES,
        settings: AUTO_START
    }).options.map((o) => o.value).filter((v) => v !== CUSTOM_PRESET_VALUE);

    const twoPlayer = offered('head_to_head');
    const fourPlayer = offered('group_of_4');
    for (const preset of QUEUE_PRESETS) {
        const listed = preset.gameMode === 'head_to_head' ? twoPlayer : fourPlayer;
        const withheld = preset.gameMode === 'head_to_head' ? fourPlayer : twoPlayer;
        assert.ok(listed.includes(preset.id), preset.id + ' belongs on a ' + preset.gameMode + ' lobby');
        assert.equal(withheld.includes(preset.id), false,
            preset.id + ' seats a different table and must not be offered');
    }
    // The default preset fixes no player count, so it is on offer either way.
    assert.ok(twoPlayer.includes(DEFAULT_PRESET.id) && fourPlayer.includes(DEFAULT_PRESET.id));
});

test('a recorded id from another table size cannot mislabel a locked sheet', () => {
    // The read-only row takes the recorded id on trust, so a record written before the select was
    // gated is the one thing left that could head a Head to head lobby FFA-4 Standard.
    const fourPlayer = QUEUE_PRESETS.find((p) => p.gameMode === 'group_of_4');
    const row = rulesetRow({ ...MATCHMADE, saved: { ...MATCHMADE.saved, presetId: fourPlayer.id } });
    assert.equal(row.kind, 'name');
    assert.notEqual(row.name, fourPlayer.name);
    assert.equal(PRESETS.find((p) => p.name === row.name).gameMode, 'head_to_head');
});

test('a lobby no preset fits is offered no Ruleset row at all', () => {
    // A select whose only entry is Custom offers nothing, so it goes the way an unread preset
    // list does rather than sitting on the sheet as a one-entry control.
    const row = rulesetRow({
        presets: QUEUE_PRESETS.filter((p) => p.gameMode === 'group_of_4'),
        canEdit: true,
        saved: MATCHMADE.saved,
        selectedId: null,
        houseRules: RANKED_RULES,
        settings: AUTO_START
    });
    assert.deepEqual(row, { kind: 'none' });
});
