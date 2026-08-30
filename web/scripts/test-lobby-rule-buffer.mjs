// Unit test for the rule sheet's edit buffer (cambia-1099 Q3).
//
// Run:  npm run test:rule-buffer     (node --test, types stripped from the imported .ts)
//
// The defect this pins: the sheet seeded its buffer from one effect keyed on the saved lobby and
// on the preset id resolved against GET /lobby/presets. That fetch resolves a beat after mount, so
// the effect ran a second time against a lobby that had not moved, and the reseed took whatever
// the host had typed in the meantime with it.
//
// The sheet has no component test to catch that (no vitest, jsdom or testing-library in the web
// tree), which is the reason the decision lives in a reducer here rather than in the effect.

/* global structuredClone */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ruleBufferReducer, seedRuleBuffer } from '../src/lib/lobbyRuleBuffer.ts';

const LOBBY = {
    houseRules: { turnTimerSec: 15, cardsPerPlayer: 4, snapRace: false },
    circuit: { enabled: false, rules: { targetScore: 100, winBonus: -1 } },
    settings: { autoStart: true }
};

/** The saved lobby as a fresh object, which is what every lobby event hands the sheet. */
const resent = () => structuredClone(LOBBY);

const edit = (state, houseRules) => ruleBufferReducer(state, { type: 'houseRules', houseRules });
const savedLobby = (state, lobby, presetId) => ruleBufferReducer(state, { type: 'saved', lobby, presetId });

/** Runs an action and asserts the reducer wrote nothing into the state it was handed. */
function reduce(state, action) {
    const before = JSON.stringify(state);
    const next = ruleBufferReducer(state, action);
    assert.equal(JSON.stringify(state), before, 'the reducer mutated the state it was given');
    return next;
}

test('a seeded buffer holds the lobby, edited by nobody', () => {
    const state = seedRuleBuffer(resent(), null);
    assert.deepEqual(state.houseRules, LOBBY.houseRules);
    assert.deepEqual(state.circuit, LOBBY.circuit);
    assert.deepEqual(state.settings, LOBBY.settings);
    assert.equal(state.presetId, null);
    assert.equal(state.saveStatus, 'idle');
    assert.equal(state.touched, false);
});

test('an in-flight edit survives the presets fetch landing', () => {
    // The host starts typing before GET /lobby/presets comes back. When it does, the same saved
    // lobby arrives again, now carrying a resolved id: nothing about the lobby moved, so there is
    // nothing here for the sheet to be reset from.
    const typed = edit(seedRuleBuffer(resent(), null), { ...LOBBY.houseRules, turnTimerSec: 30 });
    const after = reduce(typed, { type: 'saved', lobby: resent(), presetId: 'h2h_rapid' });

    assert.equal(after.houseRules.turnTimerSec, 30, 'the edit must outlive the presets fetch');
    assert.deepEqual(after.circuit, LOBBY.circuit);
    assert.deepEqual(after.settings, LOBBY.settings);
    assert.equal(after.touched, true);
});

test('a preset id resolved after mount names a sheet nobody has touched', () => {
    const state = reduce(seedRuleBuffer(resent(), null), { type: 'saved', lobby: resent(), presetId: 'h2h_rapid' });
    assert.equal(state.presetId, 'h2h_rapid', 'the id the fetch resolved still names the lobby');
    assert.deepEqual(state.houseRules, LOBBY.houseRules);
    assert.equal(state.touched, false);
});

test('a real rule change still resets the buffer, the ruleset naming it included', () => {
    const typed = edit(seedRuleBuffer(resent(), 'h2h_rapid'), { ...LOBBY.houseRules, turnTimerSec: 30 });
    const moved = { ...resent(), houseRules: { ...LOBBY.houseRules, cardsPerPlayer: 5 } };
    const after = savedLobby(typed, moved, 'default');

    assert.deepEqual(after.houseRules, moved.houseRules, 'the saved lobby is what the sheet shows');
    assert.equal(after.presetId, 'default');
    assert.equal(after.touched, false);
    assert.equal(after.saveStatus, 'idle');
});

test('a lobby resent with the same rules is not a rule change', () => {
    // A player joining rebuilds LobbyState, so the sheet is handed a new object holding the same
    // rules on every lobby event. Reseeding on object identity would discard an edit per event.
    const typed = edit(seedRuleBuffer(resent(), null), { ...LOBBY.houseRules, snapRace: true });
    assert.equal(savedLobby(typed, resent(), null).houseRules.snapRace, true);
});

test('a circuit or auto-start edit survives the same landing', () => {
    const seeded = seedRuleBuffer(resent(), null);
    const withCircuit = reduce(seeded, { type: 'circuit', circuit: { ...LOBBY.circuit, enabled: true } });
    const withBoth = reduce(withCircuit, { type: 'settings', settings: { autoStart: false } });
    const after = savedLobby(withBoth, resent(), 'default');

    assert.equal(after.circuit.enabled, true);
    assert.equal(after.settings.autoStart, false);
});

test('applying a preset fills the rules, names itself, and leaves circuit scoring alone', () => {
    // A preset cannot express a round count, so it says nothing about circuit scoring: a host who
    // turned it on has not left the preset and must not have it turned back off underneath them.
    const seeded = reduce(seedRuleBuffer(resent(), null), { type: 'circuit', circuit: { ...LOBBY.circuit, enabled: true } });
    const applied = reduce(seeded, {
        type: 'preset',
        presetId: 'ffa4_standard',
        houseRules: { ...LOBBY.houseRules, snapRace: true },
        settings: { autoStart: false }
    });

    assert.equal(applied.houseRules.snapRace, true);
    assert.equal(applied.settings.autoStart, false);
    assert.equal(applied.presetId, 'ffa4_standard');
    assert.equal(applied.circuit.enabled, true, 'applying a preset must not touch circuit scoring');
    assert.equal(applied.touched, true);
});

test('a pick between two rule-identical presets counts as an edit', () => {
    // The queue presets hold the same rules, so this moves no value at all. It is still a change
    // the host made, and a preset id landing after it must not rename the sheet back.
    const state = reduce(seedRuleBuffer(resent(), 'h2h_quick'), {
        type: 'preset',
        presetId: 'h2h_rapid',
        houseRules: resent().houseRules,
        settings: resent().settings
    });
    assert.equal(savedLobby(state, resent(), 'h2h_quick').presetId, 'h2h_rapid');
});

test('every edit clears the saved flag, and Save sets it', () => {
    const saved = reduce(seedRuleBuffer(resent(), null), { type: 'saveStatus', status: 'saved' });
    assert.equal(saved.saveStatus, 'saved');
    assert.equal(edit(saved, { ...LOBBY.houseRules, turnTimerSec: 30 }).saveStatus, 'idle');
    assert.equal(ruleBufferReducer(saved, { type: 'saveStatus', status: 'saved' }), saved,
        'a status that has not moved must not re-render the sheet');
});
