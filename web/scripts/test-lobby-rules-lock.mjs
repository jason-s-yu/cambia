// Source-structure test for the rule sheet's lock signal (cambia-1099 K2).
//
// Run:  npm run test:rules-lock        (node --test)
//
// The defect this pins: the sheet decided "these rules are locked" from lobby_type and a mode the
// snapshot never carried, so a standing public or private lobby that queued its party into a
// ranked queue - typed private, moded ranked (cambia-966) - read as editable, and its host was
// offered controls the service refuses on Save. The service now says so itself, in
// lobby_state.rules_locked, computed where update_rules refuses.
//
// Structure rather than unit: the store pulls in zustand and the '@/' alias, neither of which
// node --test resolves, and the two ends of this field live in different languages. What can go
// wrong is one end being renamed, so both ends are read and held to the same wire key.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const HUB_GO = read('service/internal/hub/hub.go');
const LOBBY_GO = read('service/internal/lobby/lobby.go');
const STORE = read('web/src/stores/lobbyStore.ts');
const SHEET = read('web/src/components/lobby/DsMatchSettings.tsx');
const TYPES = read('web/src/types/index.ts');

test('the service computes the lock once, where update_rules refuses', () => {
    assert.match(
        LOBBY_GO,
        /func \(l \*Lobby\) RulesLockedUnsafe\(\) bool \{\s*return l\.Type == "matchmaking" \|\| l\.Mode == "ranked"\s*\}/,
        'RulesLockedUnsafe must carry the matchmaking-or-ranked rule'
    );
    assert.match(
        HUB_GO,
        /locked := h\.Lobby\.RulesLockedUnsafe\(\)/,
        'the update_rules refusal must read the same helper, not its own copy of the expression'
    );
});

test('lobby_state ships the flag, so sync_state does too', () => {
    // Both are built from buildLobbySnapshot; the key is in that one map literal.
    assert.match(HUB_GO, /"rules_locked":\s*lob\.RulesLockedUnsafe\(\)/);
    assert.match(HUB_GO, /payload := h\.buildLobbySnapshot\(userID\)/, 'sync_state must repair from the same snapshot');
});

test('the store maps the flag on both paths a lobby arrives by', () => {
    // lobby_state and the sync_state rebuild (forceSync) each construct LobbyState separately.
    const mapped = [...STORE.matchAll(/rulesLocked:\s*(payload|message)\.rules_locked/g)].map((m) => m[1]);
    assert.deepEqual(mapped.sort(), ['message', 'payload'], 'both lobby_state and sync_state must map rules_locked');
    assert.match(TYPES, /rulesLocked\?:\s*boolean/, 'LobbyState must declare the field');
});

test('the sheet reads the service flag, with the type reading only as a fallback', () => {
    assert.match(
        SHEET,
        /const locked = currentSettings\.rulesLocked \?\?/,
        'the locked check must start from the service flag'
    );
    assert.doesNotMatch(
        SHEET,
        /const locked = currentSettings\.type === 'matchmaking'/,
        'the sheet must not decide the lock from the lobby type alone'
    );
});
