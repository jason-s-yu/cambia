// internal/middleware/auth.go

package middleware

import (
	"net/http"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// RequireAuth is an HTTP middleware that requires a valid "auth_token" cookie
// containing a JWT signed by the server's ed25519 key pair. Requests without the
// cookie, or with a cookie that fails JWT verification, receive a 401. Compose
// with LogMiddleware as an outer wrapper, e.g.:
//
//	middleware.LogMiddleware(logger)(middleware.RequireAuth(handler))
func RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ck, err := r.Cookie("auth_token")
		if err != nil {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if _, err := auth.AuthenticateJWT(ck.Value); err != nil {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
