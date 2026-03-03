// internal/game/emitter.go
package game

import "github.com/google/uuid"

// Emitter is the interface the hub implements to send events to clients.
// The game engine calls Emit/EmitTo instead of holding WebSocket references.
type Emitter interface {
	Emit(eventType string, payload any)
	EmitTo(userID uuid.UUID, eventType string, payload any)
}
