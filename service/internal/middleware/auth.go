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
// A request can carry more than one cookie named "auth_token": localhost
// cookies are shared across ports/apps on a multi-app host, and a dev server
// restart that rotates the signing key leaves a stale cookie in the jar
// alongside the fresh one issued at the next login. net/http's r.Cookie only
// ever returns the first match, so a stale or foreign duplicate can shadow a
// valid cookie sent later in the header. RequireAuth instead walks every
// auth_token cookie via r.Cookies() and accepts the request if any of them
// verifies. Any invalid auth_token cookie encountered along the way is
// cleared with an expiring Set-Cookie response so the browser's jar
// self-heals instead of looping on repeated 401s. Compose with LogMiddleware
// as an outer wrapper, e.g.:
//
//	middleware.LogMiddleware(logger)(middleware.RequireAuth(handler))
func RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var authTokenCookies []*http.Cookie
		for _, ck := range r.Cookies() {
			if ck.Name == "auth_token" {
				authTokenCookies = append(authTokenCookies, ck)
			}
		}

		if len(authTokenCookies) == 0 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		authenticated := false
		sawInvalid := false
		for _, ck := range authTokenCookies {
			if _, err := auth.AuthenticateJWT(ck.Value); err == nil {
				authenticated = true
				break
			}
			sawInvalid = true
		}

		if sawInvalid {
			expireAuthTokenCookie(w)
		}

		if !authenticated {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// expireAuthTokenCookie writes a Set-Cookie response that expires the
// "auth_token" cookie at Path "/" (matching the Path used everywhere the
// cookie is issued), so a stale or invalid copy in the browser's cookie jar
// is removed rather than continuing to shadow future valid cookies.
func expireAuthTokenCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:     "auth_token",
		Value:    "",
		HttpOnly: true,
		Path:     "/",
		MaxAge:   -1,
	})
}
