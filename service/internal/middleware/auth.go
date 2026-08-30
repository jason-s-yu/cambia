// internal/middleware/auth.go

package middleware

import (
	"net/http"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// RequireAuth is an HTTP middleware that requires a JWT signed by the server's
// ed25519 key pair, carried by an Authorization: Bearer header, a
// cambia-token.<jwt> WebSocket handshake entry, or the "auth_token" cookie.
// Requests with no credential, or whose credential does not verify, receive a
// 401.
//
// Resolution (carrier precedence, tolerance for stale/duplicate auth_token
// cookies, and self-healing expiring Set-Cookie responses) is delegated to
// auth.ResolveAuthToken, shared with the REST handlers in internal/handlers
// that authenticate directly off the request. See that function's doc comment
// for the precedence rules and the multi-cookie rationale. Compose with
// LogMiddleware as an outer wrapper, e.g.:
//
//	middleware.LogMiddleware(logger)(middleware.RequireAuth(handler))
func RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, _, ok := auth.ResolveAuthToken(w, r); !ok {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
