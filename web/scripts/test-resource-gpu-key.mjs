// Source-structure test for the resource monitor's list keys (cambia-1099 K4).
//
// Run:  npm run test:gpu-key        (node --test)
//
// The defect this pins: the GPU cards were keyed by gpu.index alone, so a /ws/training/resources
// payload listing one index twice logged "Encountered two children with the same key, 0" on every
// tick and left one card unrendered. The sampler keeps the index unique now (resources.go), and
// the key holds regardless, because the payload comes off a driver this side does not control.
//
// Structure rather than behaviour: there is no React renderer in this suite. The keying rule is
// re-stated below as a plain function and driven with the payload that broke it, so the rule is
// tested even though the component only gets read.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const read = (rel) => readFileSync(REPO + rel, 'utf8');

const MONITOR = read('web/src/components/training/ResourceMonitor.tsx');
const RESOURCES_GO = read('service/internal/training/resources.go');

test('the GPU cards are keyed by index and position, not index alone', () => {
    assert.doesNotMatch(
        MONITOR,
        /<GPUCard key=\{gpu\.index\}/,
        'a bare index key collides whenever the payload repeats one'
    );
    assert.match(MONITOR, /snapshot\.gpus\.map\(\(gpu, i\) => \(/);
    assert.match(MONITOR, /<GPUCard key=\{`\$\{gpu\.index\}-\$\{i\}`\}/);
});

test('the per-process rows carry the same composite', () => {
    // pid comes through parseIntField, which answers 0 for anything it cannot read.
    assert.doesNotMatch(MONITOR, /<div key=\{p\.pid\}/);
    assert.match(MONITOR, /key=\{`\$\{p\.pid\}-\$\{i\}`\}/);
});

test('the composite is unique for a payload that repeats an index', () => {
    const keyOf = (item, i) => `${item.index}-${i}`;
    const gpus = [{ index: 0 }, { index: 0 }, { index: 1 }];
    const keys = gpus.map(keyOf);
    assert.equal(new Set(keys).size, gpus.length, `keys collided: ${keys.join(', ')}`);
});

test('the sampler is the first line: one entry per device index', () => {
    assert.match(
        RESOURCES_GO,
        /if indexTaken\(gpus, idx\) \{\s*continue\s*\}/,
        'parseGPUCSV must drop a row repeating an index already in the list'
    );
    assert.match(
        RESOURCES_GO,
        /idx, ok := parseIndexField\(fields\[0\]\)/,
        'an index that does not parse must not silently become device 0'
    );
});
