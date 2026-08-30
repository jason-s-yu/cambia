// internal/auth/token.go
package auth

import (
	"net/http"
	"strings"
)

const (
	// SessionModeHeader lets a caller ask for a tab-scoped session: the token
	// is returned in the response body and no auth_token cookie is set, so the
	// caller holds an identity that is private to one browser tab instead of
	// sharing the origin's cookie jar (cambia-1149).
	SessionModeHeader = "X-Cambia-Session"

	// SessionModeTab is the only SessionModeHeader value with meaning today.
	SessionModeTab = "tab"

	// TokenSubprotocolPrefix carries a JWT through a WebSocket handshake.
	// Browsers cannot set request headers on `new WebSocket`, so a tab-held
	// token rides the subprotocol list instead: the client offers
	// ["cambia", "cambia-token.<jwt>"]. The JWT alphabet (base64url plus ".")
	// is valid RFC 6455 token text. The server never selects this entry - it
	// only ever selects "cambia" - so the carrier stays invisible to the
	// negotiated protocol.
	TokenSubprotocolPrefix = "cambia-token."

	// bearerScheme is the Authorization scheme carrying a JWT on REST calls.
	// RFC 7235 makes the scheme case-insensitive; the token after it is not.
	bearerScheme = "bearer"
)

// IsTabSession reports whether the request asked for a tab-scoped session via
// SessionModeHeader. Handlers that honor it return the minted token in the body
// and write no Set-Cookie, leaving the shared cookie identity of the other tabs
// untouched.
func IsTabSession(r *http.Request) bool {
	return strings.EqualFold(strings.TrimSpace(r.Header.Get(SessionModeHeader)), SessionModeTab)
}

// ExplicitToken returns the JWT a caller sent deliberately, and whether one was
// present at all. Two carriers, in order:
//
//  1. Authorization: Bearer <jwt> - REST calls.
//  2. A Sec-WebSocket-Protocol entry prefixed TokenSubprotocolPrefix - the
//     WebSocket handshake, where headers are not available to the client.
//
// An Authorization header with any other scheme is not ours (a proxy's Basic
// credentials, say) and offers nothing; neither does a Bearer with empty token
// text. Both fall through, so a caller that sent no usable token of ours is
// treated as having sent none rather than being locked out.
//
// The protocol list is parsed here rather than through the websocket library's
// helper so the value's case survives untouched: subprotocol matching is
// case-insensitive, JWT payloads are not.
func ExplicitToken(r *http.Request) (token string, present bool) {
	if authz := strings.TrimSpace(r.Header.Get("Authorization")); authz != "" {
		if scheme, rest, found := strings.Cut(authz, " "); found && strings.EqualFold(scheme, bearerScheme) {
			if t := strings.TrimSpace(rest); t != "" {
				return t, true
			}
		}
	}

	for _, value := range r.Header.Values("Sec-WebSocket-Protocol") {
		for _, entry := range strings.Split(value, ",") {
			entry = strings.TrimSpace(entry)
			if t, found := strings.CutPrefix(entry, TokenSubprotocolPrefix); found && t != "" {
				return t, true
			}
		}
	}

	return "", false
}

// ResolveAuthToken resolves the requesting user's ID from, in order: an
// explicit token (Authorization: Bearer, then the WebSocket handshake carrier),
// then the auth_token cookie(s).
//
// A caller that sends an explicit token has opted out of the cookie for that
// request: if that token does not verify the request fails, the cookie is
// neither consulted nor expired, and no fallback identity is granted. That
// keeps one tab's stale token from either borrowing or destroying the shared
// cookie session the other tabs are still using (cambia-1149).
//
// With no explicit token the behavior is exactly ResolveAuthTokenCookie's,
// including the self-healing expiring Set-Cookie for an invalid cookie; see
// that function's doc comment for the multi-cookie rationale.
//
// sawAny reports whether any credential was offered at all, letting callers
// distinguish "not signed in" from "signed in with something that failed"
// (401 vs 403). ok reports whether userID is authenticated.
func ResolveAuthToken(w http.ResponseWriter, r *http.Request) (userID string, sawAny bool, ok bool) {
	if token, present := ExplicitToken(r); present {
		uid, err := AuthenticateJWT(token)
		if err != nil {
			return "", true, false
		}
		return uid, true, true
	}

	return ResolveAuthTokenCookie(w, r)
}
