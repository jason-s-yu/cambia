// internal/auth/cookie.go
package auth

import "net/http"

// AuthCookieName is the name of the cookie carrying the signed session JWT.
const AuthCookieName = "auth_token"

// ResolveAuthTokenCookie walks every AuthCookieName cookie present on the
// request and returns the "sub" (user ID) claim of the first one that
// verifies.
//
// A request can carry more than one cookie named "auth_token": localhost
// cookies are shared across ports/apps on a multi-app host, and a dev
// server restart that rotates the signing key leaves a stale cookie in the
// jar alongside the fresh one issued at the next login. net/http's
// r.Cookie, and naive "Cookie" header string-splitting, only ever look at
// the first match, so a stale or foreign duplicate can shadow a valid
// cookie sent later in the header. ResolveAuthTokenCookie instead checks
// every occurrence via r.Cookies() and accepts the first that verifies.
//
// sawAny reports whether at least one auth_token cookie was present on the
// request at all, letting callers distinguish "no cookie" from "cookie(s)
// present but none valid" for status-code purposes (401 vs 403). ok reports
// whether a verifying cookie was found; when ok is true, userID is the
// authenticated user's ID.
//
// If any auth_token cookie on the request failed to verify, w receives an
// expiring Set-Cookie for auth_token so the browser's jar drops the stale
// copy instead of resending it on every subsequent request.
func ResolveAuthTokenCookie(w http.ResponseWriter, r *http.Request) (userID string, sawAny bool, ok bool) {
	sawInvalid := false

	for _, ck := range r.Cookies() {
		if ck.Name != AuthCookieName {
			continue
		}
		sawAny = true

		if uid, err := AuthenticateJWT(ck.Value); err == nil {
			userID = uid
			ok = true
			break
		}
		sawInvalid = true
	}

	if sawInvalid {
		ExpireAuthTokenCookie(w)
	}

	return userID, sawAny, ok
}

// ExpireAuthTokenCookie writes a Set-Cookie response that expires the
// "auth_token" cookie at Path "/" (matching the Path used everywhere the
// cookie is issued), so a stale or invalid copy in the browser's cookie jar
// is removed rather than continuing to shadow future valid cookies.
func ExpireAuthTokenCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:     AuthCookieName,
		Value:    "",
		HttpOnly: true,
		Path:     "/",
		MaxAge:   -1,
	})
}
