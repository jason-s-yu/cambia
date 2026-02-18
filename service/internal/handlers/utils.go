// internal/handlers/utils.go
package handlers

import "strings"

// extractCookieToken extracts a named cookie value from the "Cookie" header string.
// It returns the value of the cookie or an empty string if the cookie is not found.
func extractCookieToken(cookieHeader, cookieName string) string {
	parts := strings.Split(cookieHeader, cookieName+"=")
	if len(parts) < 2 {
		return ""
	}
	token := parts[1]
	// Handle potential subsequent cookies in the header.
	if idx := strings.Index(token, ";"); idx != -1 {
		token = token[:idx]
	}
	return token
}
