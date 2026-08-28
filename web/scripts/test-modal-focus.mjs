// Unit test for a dialog's initial focus (cambia-935, F6).
//
// Run:  npm run test:modal-focus        (node --test, types stripped from the imported .ts)
//
// ds/core/Modal focused the last control in the action row, which stands in for
// "the confirm action" only while the confirm action is drawn last. A footer
// that ends in a destructive control opened with that control under Enter. The
// default here is the first control that is not destructive, and a caller that
// wants another one names it.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { pickInitialFocus } from '../src/lib/modalFocus.ts';

const DESTRUCTIVE = new Set(['Delete', 'Leave']);
const isDestructive = (el) => DESTRUCTIVE.has(el);

/** The standard action row: a way out, then the action the dialog exists to take. */
const standard = { footer: ['Cancel', 'Create'], body: ['Lobby type', 'House rules'], panel: 'panel' };
/** A destructive confirm, the case the positional rule got wrong. */
const destructive = { footer: ['Cancel', 'Delete'], body: [], panel: 'panel' };

test('the default never lands on a destructive control', () => {
    assert.equal(pickInitialFocus(undefined, destructive, isDestructive), 'Cancel');
});

test('the default is the first footer control, not the last', () => {
    assert.equal(pickInitialFocus(undefined, standard, isDestructive), 'Cancel');
});

test('an all-destructive action row falls through to the body, then the panel', () => {
    assert.equal(
        pickInitialFocus(undefined, { footer: ['Delete', 'Leave'], body: ['Name'], panel: 'panel' }, isDestructive),
        'Name'
    );
    assert.equal(
        pickInitialFocus(undefined, { footer: ['Delete', 'Leave'], body: [], panel: 'panel' }, isDestructive),
        'panel'
    );
});

test('confirm takes the last footer control, dismiss the first', () => {
    assert.equal(pickInitialFocus('confirm', standard, isDestructive), 'Create');
    assert.equal(pickInitialFocus('dismiss', standard, isDestructive), 'Cancel');
});

test('confirm reaches a destructive control only because the caller asked for it', () => {
    assert.equal(pickInitialFocus('confirm', destructive, isDestructive), 'Delete');
});

test('body opens on the first field, panel on nothing', () => {
    assert.equal(pickInitialFocus('body', standard, isDestructive), 'Lobby type');
    assert.equal(pickInitialFocus('panel', standard, isDestructive), 'panel');
});

test('a named target that is not there falls back to the default chain', () => {
    const noFooter = { footer: [], body: ['Name'], panel: 'panel' };
    assert.equal(pickInitialFocus('confirm', noFooter, isDestructive), 'Name');
    assert.equal(pickInitialFocus('dismiss', noFooter, isDestructive), 'Name');
    assert.equal(pickInitialFocus('body', { footer: ['Cancel'], body: [], panel: 'panel' }, isDestructive), 'Cancel');
});

test('an empty dialog resolves to the panel, never to null while a panel exists', () => {
    assert.equal(pickInitialFocus(undefined, { footer: [], body: [], panel: 'panel' }, isDestructive), 'panel');
    assert.equal(pickInitialFocus(undefined, { footer: [], body: [], panel: null }, isDestructive), null);
});

test('without a destructive test the default is still the first footer control', () => {
    assert.equal(pickInitialFocus(undefined, standard), 'Cancel');
});
