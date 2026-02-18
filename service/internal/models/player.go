package models

import (
	"github.com/coder/websocket"
	"github.com/google/uuid"
)

type Player struct {
	ID              uuid.UUID       `json:"id"`
	Hand            []*Card         `json:"-"` // Hand is generally kept private
	Connected       bool            `json:"connected"`
	Conn            *websocket.Conn `json:"-"`
	HasCalledCambia bool            `json:"hasCalledCambia"`

	User *User `json:"-"`

	// DrawnCard holds the most recently drawn card (not yet discarded or swapped).
	DrawnCard *Card `json:"-"`
}
