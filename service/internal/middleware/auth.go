// internal/middleware/auth.go

package middleware

import (
	"net/http"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// RequireAuth is an HTTP middleware that requires a valid "auth_token" cookie
// containing a JWT signed by the server's ed25519 key pair. Requests without an
// auth_token cookie, or where none of the auth_token cookies present verify,
// receive a 401.
//
// Cookie resolution (including tolerance for stale/duplicate auth_token
// cookies and self-healing expiring Set-Cookie responses) is delegated to
// auth.ResolveAuthTokenCookie, shared with the REST handlers in
// internal/handlers that authenticate directly off the Cookie header. See
// that function's doc comment for the multi-cookie rationale. Compose with
// LogMiddleware as an outer wrapper, e.g.:
//
//	middleware.LogMiddleware(logger)(middleware.RequireAuth(handler))
func RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, _, ok := auth.ResolveAuthTokenCookie(w, r); !ok {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
