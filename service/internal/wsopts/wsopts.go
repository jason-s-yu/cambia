// Package wsopts centralizes the websocket.AcceptOptions shared by every
// upgrade site in the service, so the origin allowlist is configured in one
// place instead of being repeated per handler.
package wsopts

import (
	"os"
	"strings"

	"github.com/coder/websocket"
)

// OriginEnvVar names the environment variable holding the comma-separated
// allowlist. Unset means "no allowlist configured".
const OriginEnvVar = "ALLOWED_WS_ORIGINS"

// Subprotocol is the only subprotocol this service ever selects, on every
// socket. A client may offer more than this one entry - a tab-held JWT rides
// the handshake as auth.TokenSubprotocolPrefix + the token (cambia-1149) - and
// RFC 6455 requires a client that offered protocols to fail the handshake if
// the server selects none, so every Accept site passes this value even where
// the protocol itself carries no meaning.
const Subprotocol = "cambia"

// OriginPatterns returns the value for websocket.AcceptOptions.OriginPatterns.
// An unset ALLOWED_WS_ORIGINS keeps the "*" wildcard that dev runs depend on
// (the Vite dev server and the client are served from different ports).
//
// coder/websocket matches each pattern against the *host* of the Origin header,
// not the full origin URL, so a configured "https://cambia.jasonyu.io" would
// never match. Entries are normalized to bare host[:port] to accept both the
// scheme-qualified form people naturally write and the bare host the matcher
// wants. Same-origin requests are authorized by the library before the patterns
// are consulted, so the allowlist only governs cross-origin upgrades.
func OriginPatterns() []string {
	raw := os.Getenv(OriginEnvVar)
	if strings.TrimSpace(raw) == "" {
		return []string{"*"}
	}

	// A set-but-empty allowlist (only unparseable entries) fails closed: an
	// empty pattern slice means the library allows same-origin upgrades only,
	// which is the safe reading of a misconfigured allowlist. Only a fully
	// unset variable opts into the dev wildcard above.
	patterns := make([]string, 0, strings.Count(raw, ",")+1)
	for _, entry := range strings.Split(raw, ",") {
		if host := normalizeOrigin(entry); host != "" {
			patterns = append(patterns, host)
		}
	}
	return patterns
}

// normalizeOrigin strips an optional scheme and any trailing path from a single
// allowlist entry, leaving the host[:port] form the matcher compares against.
// Wildcard patterns such as "*.jasonyu.io" pass through untouched.
func normalizeOrigin(entry string) string {
	host := strings.TrimSpace(entry)
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+len("://"):]
	}
	if i := strings.IndexByte(host, '/'); i >= 0 {
		host = host[:i]
	}
	return strings.TrimSpace(host)
}

// AcceptOptions builds the accept options for an upgrade offering the given
// subprotocols, with the configured origin allowlist applied.
func AcceptOptions(subprotocols ...string) *websocket.AcceptOptions {
	return &websocket.AcceptOptions{
		Subprotocols:   subprotocols,
		OriginPatterns: OriginPatterns(),
	}
}
