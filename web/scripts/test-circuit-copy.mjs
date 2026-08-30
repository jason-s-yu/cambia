// Unit test for the circuit scoring explainer's copy (cambia-1100).
//
// Run:  npm run test:circuit-copy        (node --test, types stripped from the imported .ts)
//
// The explainer states numbers the server owns: the circuit rule defaults a new
// lobby is created with, the round counts each format runs, the per-round
// subsidy schedule and the window a disconnect holds a seat for. Prose cannot
// import those, so this test reads them out of the Go sources and fails when the
// two drift apart. It also holds the copy to the project's writing rules.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { CIRCUIT_DEFAULTS, CIRCUIT_SCORING_COPY } from '../src/components/lobby/circuitScoringCopy.ts';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const LOBBY_GO = read('service/internal/lobby/lobby.go');
const CIRCUIT_GO = read('engine/circuit.go');
const SCORING_GO = read('engine/scoring.go');
const GAME_GO = read('service/internal/game/game.go');

/** The single match of `re` in `text`, as a number. Absence is a failure, not a skip. */
function num(text, re, what) {
    const m = text.match(re);
    assert.ok(m, `could not read ${what} out of the Go source; the pattern this test reads it with is stale`);
    return Number(m[1]);
}

const section = (id) => {
    const s = CIRCUIT_SCORING_COPY.find((sec) => sec.id === id);
    assert.ok(s, `copy is missing the '${id}' section`);
    return s.body.join(' ');
};

/** Every signed integer in a string, in order. */
const numbers = (text) => (text.match(/-?\d+/g) ?? []).map(Number);

test('the stated defaults are the ones NewLobbyWithDefaults creates a lobby with', () => {
    assert.equal(CIRCUIT_DEFAULTS.targetScore, num(LOBBY_GO, /TargetScore:\s+(-?\d+)/, 'TargetScore'));
    assert.equal(CIRCUIT_DEFAULTS.winBonus, num(LOBBY_GO, /WinBonus:\s+(-?\d+)/, 'WinBonus'));
    assert.equal(CIRCUIT_DEFAULTS.falseCambiaPenalty, num(LOBBY_GO, /FalseCambiaPenalty:\s+(-?\d+)/, 'FalseCambiaPenalty'));
});

test('each default is quoted in the copy, signed the way it is applied', () => {
    const end = section('end');
    const adjustments = section('adjustments');
    assert.match(end, new RegExp(`default ${CIRCUIT_DEFAULTS.targetScore}\\b`));
    assert.match(adjustments, /default -1\b/);
    assert.match(adjustments, /default \+1\b/);
    assert.equal(CIRCUIT_DEFAULTS.winBonus, -1, 'win bonus default moved; the signed forms above are stale');
    assert.equal(CIRCUIT_DEFAULTS.falseCambiaPenalty, 1, 'false Cambia penalty default moved; the signed forms above are stale');
});

test('the round counts are the ones NewCircuit sets per format', () => {
    const quick = num(CIRCUIT_GO, /case CircuitQuick:\s*\n\s*config\.NumRounds = (\d+)/, 'the quick round count');
    const standard = num(CIRCUIT_GO, /case CircuitStandard:\s*\n\s*config\.NumRounds = (\d+)/, 'the standard round count');
    const championship = num(CIRCUIT_GO, /case CircuitChampionship:\s*\n\s*config\.NumRounds = (\d+)/, 'the championship round count');
    const fallback = num(CIRCUIT_GO, /default:\s*\n\s*config\.NumRounds = (\d+)/, 'the unset-format round count');
    const rounds = section('rounds');
    assert.match(rounds, new RegExp(`${quick} for quick`));
    assert.match(rounds, new RegExp(`${standard} for standard`));
    assert.match(rounds, new RegExp(`${championship} for championship`));
    assert.match(rounds, new RegExp(`${fallback} when the lobby names no format`));
});

test('the subsidy schedule reads out ComputeAggressionSubsidy, in its order', () => {
    const schedules = [...SCORING_GO.matchAll(/schedule = \[\]int\{([^}]*)\}/g)].map((m) => numbers(m[1]));
    assert.equal(schedules.length, 3, 'engine/scoring.go no longer holds exactly three subsidy schedules');
    const stated = numbers(section('adjustments'));
    // The section's other figures are the two defaults, which are matched above and are the
    // first numbers in the text; the schedule is the run that follows them.
    const tail = stated.slice(stated.length - schedules.flat().length);
    assert.deepEqual(tail, schedules.flat());
});

test('the disconnect window is the one a circuit game arms', () => {
    // Since cambia-1233 a circuit drop runs the rule sheet's reconnect grace
    // (HouseRules.DisconnectGraceSec; service/internal/game/game.go) rather than a fixed window
    // of its own, so the copy quotes the DefaultHouseRules value as the default.
    const RULES_GO = read('service/internal/game/rules.go');
    const hold = num(RULES_GO, /DisconnectGraceSec:\s+(\d+)/, 'the default reconnect grace');
    const disconnects = section('disconnects');
    assert.match(disconnects, new RegExp(`default ${hold} seconds`));
    assert.match(disconnects, /does not forfeit the seat/);
});

test('the copy covers every topic the explainer exists to answer', () => {
    const ids = CIRCUIT_SCORING_COPY.map((s) => s.id);
    assert.deepEqual(ids, ['rounds', 'end', 'adjustments', 'disconnects', 'toggle']);
    for (const s of CIRCUIT_SCORING_COPY) {
        assert.ok(s.heading.length > 0, `section '${s.id}' has no heading`);
        assert.ok(s.body.length > 0, `section '${s.id}' has no body`);
    }
    // The lowest total winning and the per-lobby scope of the toggle are the two claims a
    // rewrite most easily drops; they are the reason a player opens this dialog.
    assert.match(section('rounds'), /lowest total/);
    assert.match(section('end'), /lowest cumulative score/);
    assert.match(section('toggle'), /per-lobby setting/);
});

test('the target score is not described as ending a circuit', () => {
    // Nothing in the service reads CircuitRules.TargetScore: the round count ends a circuit
    // (engine/circuit.go RecordRound). The copy says so, and this pins it until the server
    // enforces a target.
    assert.match(section('end'), /no circuit ends on it: the round count does/);
});

test('the copy keeps to the project writing rules', () => {
    const text = CIRCUIT_SCORING_COPY.flatMap((s) => [s.heading, ...s.body]).join('\n');
    const emDash = String.fromCharCode(0x2014);
    assert.ok(!text.includes(emDash), 'copy contains an em dash');
    for (const word of ['leverage', 'streamline', 'enhance', 'robust', 'comprehensive', 'utilize', 'facilitate']) {
        assert.ok(!new RegExp(word, 'i').test(text), `copy contains the banned word '${word}'`);
    }
});
