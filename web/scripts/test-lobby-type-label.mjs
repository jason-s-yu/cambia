// Unit and source-structure test for the lobby-type badge label (cambia-1099 K1).
//
// Run:  npm run test:type-label        (node --test, types stripped from the imported .ts)
//
// What it pins: a lobby type the label map does not carry reaches the badge as words, never as
// the raw wire id. The map covers the three types the service validates today, so the fallback
// only fires on a type added later - which is exactly when nobody is watching the badge, and how
// cambia-1086 shipped raw pool ids to the DOM. The source half exists because there is no React
// renderer in this suite to mount the view in.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { humanizeId, gameModeLabel } from '../src/utils/gameMode.ts';

const WEB = fileURLToPath(new URL('../', import.meta.url));
const read = (rel) => readFileSync(WEB + rel, 'utf8');

test('an id renders as title-cased words, on either separator the service uses', () => {
    assert.equal(humanizeId('matchmaking'), 'Matchmaking');
    assert.equal(humanizeId('private_ranked'), 'Private Ranked');
    assert.equal(humanizeId('ranked-h2h'), 'Ranked H2h');
    // Repeated and mixed separators collapse rather than leaving empty words.
    assert.equal(humanizeId('a__b-_c'), 'A B C');
});

test('an absent value names nothing, so the caller can drop the badge', () => {
    assert.equal(humanizeId(''), '');
    assert.equal(humanizeId(null), '');
    assert.equal(humanizeId(undefined), '');
});

test('the game-mode map still wins over the fallback, and still answers for an empty mode', () => {
    assert.equal(gameModeLabel('head_to_head'), 'Head to Head');
    assert.equal(gameModeLabel('some_new_mode'), 'Some New Mode');
    assert.equal(gameModeLabel(''), 'Unknown mode');
});

test('the lobby badge falls back to a label, not to the raw type', () => {
    const view = read('src/components/lobby/DsLobbyView.tsx');
    assert.doesNotMatch(
        view,
        /TYPE_LABELS\[lobbyType\]\s*\?\?\s*lobbyType/,
        'the badge must not fall back to the raw lobby-type id'
    );
    assert.match(
        view,
        /TYPE_LABELS\[lobbyType\]\s*\?\?\s*humanizeId\(lobbyType\)/,
        'the badge label must fall through humanizeId'
    );
});

test('every type the service validates has a written label, so the fallback is the rare path', () => {
    const view = read('src/components/lobby/DsLobbyView.tsx');
    const types = read('../service/internal/handlers/lobby.go').match(
        /validGameTypes\s*=\s*map\[string\]bool\{([^}]*)\}/s
    );
    assert.ok(types, 'validGameTypes not found in service/internal/handlers/lobby.go');
    const names = [...types[1].matchAll(/"([a-z_]+)"\s*:\s*true/g)].map((m) => m[1]);
    assert.ok(names.length >= 3, `expected the service to validate 3+ lobby types, got ${names.length}`);
    for (const name of names) {
        assert.match(
            view,
            new RegExp(`^\\s*${name}:\\s*'`, 'm'),
            `lobby type ${name} has no label in TYPE_LABELS`
        );
    }
});
