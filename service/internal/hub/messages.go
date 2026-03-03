// internal/hub/messages.go
package hub

import (
	"encoding/json"

	"github.com/google/uuid"
)

// Envelope is the server→client wire frame. Every message includes a monotonic seq.
type Envelope struct {
	Seq     uint64          `json:"seq"`
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// ClientMsg is the parsed client→server message routed through the hub's incoming channel.
type ClientMsg struct {
	ConnID  uuid.UUID
	UserID  uuid.UUID
	LastSeq uint64          `json:"last_seq"`
	Type    string          `json:"type"`
	Body    json.RawMessage `json:"body,omitempty"`
}
