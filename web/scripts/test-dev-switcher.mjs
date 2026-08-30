// Unit and source-structure test for the dev identity switcher (cambia-1149).
//
// Run:  npm run test:dev-switcher        (node --test, types stripped from the imported .ts)
//
// Two halves. The first tests what the panel decides: the identity line, which
// blocks are usable, and where Tab goes. The second reads the files that carry
// the tab token to the wire and asserts they still do, because that wiring is
// what makes two tabs two players and nothing else in this suite would notice
// it going missing: there is no React renderer here to mount the app in.

/* global URL */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { NAMED_ACCOUNTS_OFF, identityLine, panelModel, trapIndex } from '../src/lib/devSwitcher.ts';

const WEB = fileURLToPath(new URL('../', import.meta.url));
const read = (rel) => readFileSync(WEB + rel, 'utf8');

// --- What the panel says --------------------------------------------------

test('the pill states who this tab is and which session it came from', () => {
    assert.equal(
        identityLine({ username: 'alice', authenticated: true, pinned: true, label: 'alice' }).pill,
        'alice (tab)'
    );
    assert.equal(
        identityLine({ username: 'Guest-1A2B', authenticated: true, pinned: false }).pill,
        'Guest-1A2B (shared)'
    );
    assert.equal(identityLine({ authenticated: false, pinned: false }).pill, 'signed out (shared)');
});

test('the pinned label covers the gap before /user/me answers', () => {
    // Pinned, store not caught up yet: the tab says what it asked to be.
    assert.equal(identityLine({ authenticated: false, pinned: true, label: 'bob' }).name, 'bob');
    // Once the service answers, its username wins: a pin that resolved to
    // another account has to say so rather than keep showing what was typed.
    assert.equal(
        identityLine({ username: 'carol', authenticated: true, pinned: true, label: 'bob' }).name,
        'carol'
    );
});

test('a service with the dev routes off still offers guests, and says why', () => {
    const off = panelModel({ devRoutesEnabled: false, pinned: false, authenticated: true });
    assert.equal(off.namedAccounts, false);
    assert.equal(off.namedAccountsHint, NAMED_ACCOUNTS_OFF);
    assert.match(NAMED_ACCOUNTS_OFF, /CAMBIA_DEV_ACCOUNTS=1/);

    const on = panelModel({ devRoutesEnabled: true, pinned: false, authenticated: true });
    assert.equal(on.namedAccounts, true);
    assert.equal(on.namedAccountsHint, null);
});

test('only a pinned tab can hand its identity to a new tab or drop it', () => {
    // The shared session is an HttpOnly cookie: the page cannot read it out, so
    // there is nothing to put in a handoff fragment.
    const shared = panelModel({ devRoutesEnabled: true, pinned: false, authenticated: true });
    assert.equal(shared.canOpenNewTab, false);
    assert.equal(shared.canUnpin, false);

    const pinned = panelModel({ devRoutesEnabled: true, pinned: true, authenticated: true });
    assert.equal(pinned.canOpenNewTab, true);
    assert.equal(pinned.canUnpin, true);
});

test('Tab cycles inside the panel and never walks out of it', () => {
    assert.equal(trapIndex(3, 0, false), 1);
    assert.equal(trapIndex(3, 2, false), 0);
    assert.equal(trapIndex(3, 0, true), 2);
    assert.equal(trapIndex(3, 2, true), 1);
    // Focus on nothing in the panel (a control removed under it): back to an end.
    assert.equal(trapIndex(3, -1, false), 0);
    assert.equal(trapIndex(3, -1, true), 2);
    assert.equal(trapIndex(3, 9, false), 0);
    // An empty panel has nowhere to send focus; the caller leaves the event alone.
    assert.equal(trapIndex(0, -1, false), -1);
});

// --- What actually carries the token --------------------------------------

test("every REST call carries a pinned tab's token", () => {
    const axios = read('src/lib/axios.ts');
    assert.match(axios, /from '@\/lib\/tabSession'/);
    // Request interceptor sets the header from the module, not from a copy of
    // the storage key.
    assert.match(axios, /config\.headers\.set\('Authorization', header\.Authorization\)/);
});

test('a rejected tab token unpins instead of logging the browser out', () => {
    const axios = read('src/lib/axios.ts');
    const branch = axios.split('} else if (isPinned()) {')[1];
    assert.ok(branch, 'the 401 handler no longer has a pinned-tab branch');
    const untilNext = branch.split('} else if')[0];
    assert.match(untilNext, /unpinTab\(\)/);
    // POST /user/logout would end the session every other tab is using.
    assert.ok(!/logout\(\)/.test(untilNext), 'the pinned-tab 401 branch must not force a store logout');
    // Unpinning retires the probe that is in flight, so this branch has to ask
    // again as the cookie identity or the app never leaves its loading state.
    assert.match(untilNext, /checkAuth\(\)/);
});

test('logout in a pinned tab drops the pin and never posts /user/logout', () => {
    const store = read('src/stores/authStore.ts');
    const logout = store.split('logout: async () => {')[1].split('checkAuth:')[0];
    const pinnedBranch = logout.split('if (isPinned()) {')[1].split('}')[0];
    assert.match(pinnedBranch, /unpinTab\(\)/);
    assert.ok(!/logoutUser\(\)/.test(pinnedBranch), 'the pinned branch must not call the logout endpoint');
    // The unpinned path is unchanged: it still ends the shared session.
    assert.match(logout, /await logoutUser\(\)/);
});

test('all three sockets offer the tab token on the handshake', () => {
    for (const file of ['src/hooks/useSocket.ts', 'src/hooks/useTrainingSocket.ts', 'src/hooks/useResourceSocket.ts']) {
        const source = read(file);
        assert.match(source, /new WebSocket\([^)]*wsProtocols\(\)\)/, `${file} does not offer the tab protocol list`);
        // A pin has to redial: the identity a connection handshaked with is
        // fixed for its lifetime.
        assert.match(source, /sessionEpoch/, `${file} does not redial when the tab identity changes`);
    }
});

test('the tab-session header rides the endpoints that would otherwise set a cookie', () => {
    const service = read('src/services/devSessionService.ts');
    assert.match(service, /TAB_SESSION_HEADER\]: 'tab'/);
    for (const path of ['/user/guest', '/user/login']) {
        const call = service.split(`'${path}'`)[1];
        assert.ok(call, `${path} is no longer called from devSessionService`);
        assert.match(call.split(');')[0], /headers: TAB_HEADERS/, `${path} must carry the tab-session header`);
    }
    // A 404 from a service with the flag unset is a state to render, not an
    // error to log on every panel open.
    assert.match(service, /validateStatus: \(status\) => status === 200 \|\| status === 404/);
});

test('the boot identity is consumed before the first /user/me probe', () => {
    const main = read('src/main.tsx');
    const bootAt = main.indexOf('consumeBootIdentity');
    const renderAt = main.indexOf('createRoot');
    assert.ok(bootAt > -1 && renderAt > -1, 'main.tsx no longer boots the tab identity');
    assert.ok(bootAt < renderAt, 'the render must happen inside the boot callback, not before it');
    // App's mount effect fires the probe, so the render is what has to wait.
    assert.match(main, /consumeBootIdentity\(mintForIntent\)[\s\S]{0,60}createRoot/);
});

test('the switcher is mounted for dev builds only', () => {
    const main = read('src/main.tsx');
    assert.match(main, /\{import\.meta\.env\.DEV && <DevSessionSwitcher \/>\}/);
    const component = read('src/components/dev/DevSessionSwitcher.tsx');
    assert.match(component, /if \(!import\.meta\.env\.DEV\) return null;/);
});

test('the panel holds the controls the design calls for', () => {
    const component = read('src/components/dev/DevSessionSwitcher.tsx');
    for (const hook of ['dev-session-pill', 'dev-new-name', 'dev-pin-new', 'dev-new-guest', 'dev-open-new-tab', 'dev-unpin']) {
        assert.match(component, new RegExp(`data-testid='${hook}'`), `the switcher is missing ${hook}`);
    }
    assert.match(component, /aria-expanded=\{open\}/);
    assert.match(component, /aria-labelledby=\{headingId\}/);
    assert.match(component, /event\.key === 'Escape'/);
    // Below Modal and the results overlay, both at 100.
    const zIndex = Number(component.match(/zIndex: (\d+)/)[1]);
    assert.ok(zIndex < 100, `the switcher must sit below modals, got zIndex ${zIndex}`);
    assert.match(component, /position: 'fixed'/);
    assert.match(component, /bottom: 'calc\(var\(--space-3\) \+ env\(safe-area-inset-bottom/);
});

test('a pin or unpin re-reads the identity rather than reloading the page', () => {
    const component = read('src/components/dev/DevSessionSwitcher.tsx');
    assert.match(component, /await checkAuth\(\)/);
    assert.ok(!/location\.reload/.test(component), 'the switcher must not reload the page to apply a pin');
});

test('the dev route is proxied to the service, not answered by Vite', () => {
    const config = read('vite.config.js');
    const paths = config.match(/const API_PROXY_PATHS = \[([^\]]*)\]/)[1];
    assert.match(paths, /'\/dev'/);
});

test('the modules web/scripts loads directly still import nothing', () => {
    // node --test strips types but does not resolve the `@/` alias, so an
    // import here is a test that stops running rather than a test that fails.
    for (const file of ['src/lib/tabSession.ts', 'src/lib/devSwitcher.ts']) {
        assert.ok(!/^import /m.test(read(file)), `${file} must stay dependency-free`);
    }
});
