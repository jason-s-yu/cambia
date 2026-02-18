// internal/handlers/ws_codes.go
package handlers

// Custom WebSocket close codes used within the lobby and game handlers.
// These provide more specific reasons for closure than standard codes.
const (
	BadSubprotocolError   = 3000 // Client connected with an unsupported subprotocol.
	InvalidAuthTokenError = 3001 // Provided auth token was invalid or expired (used if auth fails before standard HTTP response).
	InvalidUserIDError    = 3002 // User ID derived from token was malformed or invalid.
	InvalidLobbyIDError   = 3003 // Target lobby ID specified in the WS URL does not exist or is invalid.
	// Add more custom codes as needed for specific game/lobby errors.
)
