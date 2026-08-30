package models

import (
	"github.com/coder/websocket"
	"github.com/google/uuid"
)

// Player is the service's per-seat record: identity, socket and the hand mirror. Who called Cambia
// is deliberately absent - the engine's CambiaCaller is the only authority on it, and the
// hasCalledCambia a client reads comes from the snapshot projection (game.ObfPlayerState), which
// reads that engine field. A second copy here was never assigned in production, so every check
// against it silently passed (cambia-1118).
type Player struct {
	ID        uuid.UUID       `json:"id"`
	Hand      []*Card         `json:"-"` // Hand is generally kept private
	Connected bool            `json:"connected"`
	Conn      *websocket.Conn `json:"-"`

	User *User `json:"-"`

	// DrawnCard holds the most recently drawn card (not yet discarded or swapped).
	DrawnCard *Card `json:"-"`
}
