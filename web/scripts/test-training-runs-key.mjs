// Source-structure test for the training runs table's row key (cambia-1099 K5).
//
// Run:  node --test scripts/test-training-runs-key.mjs
//
// The defect this pins: TrainingPage keyed each runs-table row by run.id, but GET /training/runs
// merges dashboard-created runs that exist only as process.json (store.go queryRuns), and those
// carry no run_db row, so their id is 0. Two such runs collided on key "0" and React logged
// "Encountered two children with the same key, 0" on every resource tick (the tick re-renders the
// page). The name is the identity everywhere else (navigate, GetRun, ComparePage), so the row
// key is the name.
//
// Structure rather than behaviour: there is no React renderer in this suite. The keying rule is
// re-stated below as a plain function and driven with the payload shape that broke it.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const PAGE = read('web/src/pages/TrainingPage.tsx');
const STORE_GO = read('service/internal/training/store.go');

test('the runs table rows are keyed by name, not id', () => {
    assert.doesNotMatch(PAGE, /key=\{run\.id\}/, 'id is 0 for every unregistered run');
    assert.match(PAGE, /filtered\.map\(\(run\) => \(\s*<tr\s+key=\{run\.name\}/);
});

test('the name key is unique for a list with two unregistered runs', () => {
    const keyOf = (run) => run.name;
    const runs = [
        { id: 12, name: 'v0.4-prtcfr-x2r-c1' },
        { id: 0, name: 'dash-created-a' },
        { id: 0, name: 'dash-created-b' },
    ];
    const ids = runs.map((r) => r.id);
    assert.notEqual(new Set(ids).size, runs.length, 'the fixture must reproduce the id collision');
    const keys = runs.map(keyOf);
    assert.equal(new Set(keys).size, runs.length, `keys collided: ${keys.join(', ')}`);
});

test('the server still merges process-only runs with no id', () => {
    // If this stops matching, the sentinel changed; re-check whether id became usable as a key.
    assert.match(
        STORE_GO,
        /states, _ := procmgr\.ScanProcessStates\(s\.runsDir\)[\s\S]*?run := Run\{\s*Name:\s*st\.Name,/,
        'queryRuns builds process-only runs without an ID'
    );
    assert.match(STORE_GO, /0 is never a real\s*\n\s*\/\/ id/, 'the Run.ID comment documents the sentinel');
});
