// Unit test for the per-tab identity module (cambia-1149).
//
// Run:  npm run test:tab-session        (node --test, types stripped from the imported .ts)
//
// What this pins: a tab that has pinned an identity must send it explicitly on
// every carrier (REST header, WebSocket subprotocol list) and must survive a
// browser that refuses storage entirely; a boot URL's identity request must be
// consumed once and taken back out of the address bar; and a failed request
// must leave the tab on the shared cookie rather than half pinned.
//
// The host objects the module reads (sessionStorage, location, history) are
// fakes installed on globalThis, which is the whole reason lib/tabSession.ts
// reads them through globalThis at call time instead of capturing them.

/* global URL, Buffer, console */

import assert from 'node:assert/strict';
import { test, beforeEach } from 'node:test';

/** A sessionStorage that behaves, and one that does not. */
class FakeStorage {
    constructor() {
        this.map = new Map();
    }
    getItem(key) {
        return this.map.has(key) ? this.map.get(key) : null;
    }
    setItem(key, value) {
        this.map.set(key, String(value));
    }
    removeItem(key) {
        this.map.delete(key);
    }
}

const THROWING_STORAGE = {
    getItem() {
        throw new Error('site data blocked');
    },
    setItem() {
        throw new Error('site data blocked');
    },
    removeItem() {
        throw new Error('site data blocked');
    }
};

/** A history that writes back to the fake location, as a browser's does. */
function installLocation(href) {
    globalThis.location = { get origin() { return new URL(href).origin; }, href };
    globalThis.history = {
        replaceState(_state, _title, url) {
            const next = new URL(url, globalThis.location.href);
            globalThis.location.href = next.href;
        }
    };
}

/** A JWT-shaped token whose payload really does decode to `sub`. */
function fakeToken(sub) {
    const b64url = (obj) =>
        Buffer.from(JSON.stringify(obj)).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    return `${b64url({ alg: 'HS256', typ: 'JWT' })}.${b64url({ sub })}.c2ln`;
}

const ALICE = fakeToken('4f3c1d20-1111-2222-3333-444455556666');

globalThis.sessionStorage = new FakeStorage();
installLocation('http://localhost:5180/dashboard');

const {
    authHeader,
    checkAccountName,
    consumeBootIdentity,
    describeMintFailure,
    getTabLabel,
    getTabNotice,
    getTabSessionEpoch,
    getTabToken,
    handoffHref,
    isPinned,
    pinTab,
    readBootIntent,
    setTabNotice,
    shortSubject,
    subscribeTabSession,
    tokenSubject,
    unpinTab,
    wsProtocols,
    TAB_TOKEN_KEY,
    WS_PROTOCOL,
    WS_TOKEN_PREFIX
} = await import('../src/lib/tabSession.ts');

beforeEach(() => {
    globalThis.sessionStorage = new FakeStorage();
    installLocation('http://localhost:5180/dashboard');
    setTabNotice(null);
});

test('an unpinned tab sends nothing of its own', () => {
    assert.equal(getTabToken(), null);
    assert.equal(isPinned(), false);
    assert.deepEqual(authHeader(), {});
    assert.deepEqual(wsProtocols(), [WS_PROTOCOL]);
});

test('a pinned tab carries its token on both carriers', () => {
    pinTab(ALICE, 'alice');
    assert.equal(isPinned(), true);
    assert.equal(getTabLabel(), 'alice');
    assert.deepEqual(authHeader(), { Authorization: `Bearer ${ALICE}` });
    // The base protocol stays first: the server selects it, and a client that
    // offers protocols and gets none back has to fail the handshake.
    assert.deepEqual(wsProtocols(), [WS_PROTOCOL, WS_TOKEN_PREFIX + ALICE]);
});

test('unpinning returns the tab to the shared cookie, and repeats are silent', () => {
    let epochs = 0;
    const stop = subscribeTabSession(() => {
        epochs += 1;
    });
    pinTab(ALICE, 'alice');
    unpinTab();
    const after = getTabSessionEpoch();
    // Idempotent: lib/axios.ts calls this on every rejected request, and a
    // notification per call would restart every socket on an unpinned tab.
    unpinTab();
    unpinTab();
    stop();
    assert.equal(isPinned(), false);
    assert.deepEqual(authHeader(), {});
    assert.equal(getTabSessionEpoch(), after);
    assert.equal(epochs, 2);
});

test('storage that throws leaves the tab unpinned instead of breaking the app', () => {
    globalThis.sessionStorage = THROWING_STORAGE;
    assert.doesNotThrow(() => pinTab(ALICE, 'alice'));
    assert.equal(getTabToken(), null);
    assert.equal(isPinned(), false);
    assert.deepEqual(wsProtocols(), [WS_PROTOCOL]);
    assert.doesNotThrow(() => unpinTab());
});

test('an empty token is not a pin', () => {
    pinTab('', 'nobody');
    assert.equal(isPinned(), false);
});

test('the subject is read off the token, and shortened for the pill', () => {
    assert.equal(tokenSubject(ALICE), '4f3c1d20-1111-2222-3333-444455556666');
    assert.equal(shortSubject(ALICE), '4f3c1d20');
    // Anything that is not a JWT answers null rather than throwing.
    assert.equal(tokenSubject('not-a-token'), null);
    assert.equal(tokenSubject('a.!!!.c'), null);
});

test('a pin with no label falls back to the token subject', () => {
    pinTab(ALICE, '');
    assert.equal(getTabLabel(), '4f3c1d20');
});

test('?as= names an account, ?as=guest a fresh guest, and both leave a clean URL', () => {
    const named = readBootIntent('http://localhost:5180/lobby/abc?as=Alice&keep=1');
    assert.deepEqual(named.intent, { kind: 'dev-account', name: 'alice' });
    assert.equal(named.cleanedHref, '/lobby/abc?keep=1');
    assert.equal(named.error, null);

    const guest = readBootIntent('http://localhost:5180/dashboard?as=guest');
    assert.deepEqual(guest.intent, { kind: 'guest' });
    assert.equal(guest.cleanedHref, '/dashboard');
});

test('a name the server would refuse is refused here, and still consumed', () => {
    const bad = readBootIntent('http://localhost:5180/?as=Not%20A%20Name!');
    assert.equal(bad.intent, null);
    assert.equal(bad.cleanedHref, '/');
    assert.match(bad.error, /1 to 32 characters/);

    const long = readBootIntent(`http://localhost:5180/?as=${'a'.repeat(33)}`);
    assert.equal(long.intent, null);
    assert.ok(long.error);
});

test('#tab= carries a token, and the fragment does not survive it', () => {
    const handoff = readBootIntent(`http://localhost:5180/lobby/abc#tab=${ALICE}`);
    assert.deepEqual(handoff.intent, { kind: 'token', token: ALICE });
    assert.equal(handoff.cleanedHref, '/lobby/abc');
});

test('?as= wins over a fragment, and a plain URL asks for nothing', () => {
    const both = readBootIntent(`http://localhost:5180/?as=bob#tab=${ALICE}`);
    assert.deepEqual(both.intent, { kind: 'dev-account', name: 'bob' });

    const plain = readBootIntent('http://localhost:5180/dashboard#section');
    assert.equal(plain.intent, null);
    assert.equal(plain.cleanedHref, null);
});

test('boot pins the named account and strips the parameter', async () => {
    installLocation('http://localhost:5180/dashboard?as=alice');
    const asked = [];
    const intent = await consumeBootIdentity(async (want) => {
        asked.push(want);
        return { token: ALICE, label: 'alice' };
    });
    assert.deepEqual(asked, [{ kind: 'dev-account', name: 'alice' }]);
    assert.deepEqual(intent, { kind: 'dev-account', name: 'alice' });
    assert.equal(getTabToken(), ALICE);
    assert.equal(globalThis.location.href, 'http://localhost:5180/dashboard');
});

test('a handoff fragment needs no round trip', async () => {
    installLocation(`http://localhost:5180/lobby/abc#tab=${ALICE}`);
    let minted = false;
    await consumeBootIdentity(async () => {
        minted = true;
        return null;
    });
    assert.equal(minted, false);
    assert.equal(getTabToken(), ALICE);
    // The token must not stay in the address bar: a copied link would hand the
    // identity to whoever it was pasted to.
    assert.equal(globalThis.location.href, 'http://localhost:5180/lobby/abc');
});

test('a boot request that fails leaves the tab on the shared session, with a notice', async () => {
    installLocation('http://localhost:5180/?as=alice');
    const warn = console.warn;
    console.warn = () => {};
    try {
        await consumeBootIdentity(async () => {
            throw Object.assign(new Error('nope'), { response: { status: 404 } });
        });
    } finally {
        console.warn = warn;
    }
    assert.equal(isPinned(), false);
    assert.match(getTabNotice(), /CAMBIA_DEV_ACCOUNTS=1/);
    assert.equal(globalThis.location.href, 'http://localhost:5180/');
});

test('a URL that asks for nothing does not touch the address bar', async () => {
    installLocation('http://localhost:5180/dashboard');
    let touched = false;
    globalThis.history = {
        replaceState() {
            touched = true;
        }
    };
    const intent = await consumeBootIdentity(async () => ({ token: ALICE, label: 'alice' }));
    assert.equal(intent, null);
    assert.equal(touched, false);
    assert.equal(isPinned(), false);
});

test('the 404 notice names the flag; anything else stays generic', () => {
    const off = describeMintFailure({ kind: 'dev-account', name: 'alice' }, { response: { status: 404 } });
    assert.match(off, /CAMBIA_DEV_ACCOUNTS=1/);
    assert.match(describeMintFailure({ kind: 'guest' }, new Error('offline')), /shared session/);
});

test('the handoff link carries the token in the fragment, never the query', () => {
    const href = handoffHref('http://localhost:5180', '/lobby/abc', ALICE);
    assert.equal(href, `http://localhost:5180/lobby/abc#tab=${ALICE}`);
    assert.ok(!href.includes('?'));
});

test('a typed account name is normalised, and a bad one is named as bad', () => {
    assert.deepEqual(checkAccountName('  Alice '), { ok: true, name: 'alice' });
    assert.deepEqual(checkAccountName('a-b_9'), { ok: true, name: 'a-b_9' });
    assert.equal(checkAccountName('').ok, false);
    assert.equal(checkAccountName('has space').ok, false);
    assert.equal(checkAccountName('a'.repeat(33)).ok, false);
    assert.equal(checkAccountName('a'.repeat(32)).ok, true);
});

test('the storage keys are the ones a duplicated tab inherits', () => {
    pinTab(ALICE, 'alice');
    assert.equal(globalThis.sessionStorage.getItem(TAB_TOKEN_KEY), ALICE);
});
