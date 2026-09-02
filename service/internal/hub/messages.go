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

// ClientMsg is the parsed client→server message routed through the hub's incoming channel. The
// hub's own timers queue synthetic messages (the underscore-prefixed types) on the same channel,
// so those fire through one dispatch path with every client frame.
type ClientMsg struct {
	ConnID  uuid.UUID
	UserID  uuid.UUID
	LastSeq uint64          `json:"last_seq"`
	Type    string          `json:"type"`
	Body    json.RawMessage `json:"body,omitempty"`

	// gen stamps a synthetic message with the generation of the state it was armed for, so a
	// timer that fired after its state was superseded is dropped rather than applied to whatever
	// took its place. Unexported: no client sets it, and ReadPump does not parse it. Used by the
	// post-game reset (Hub.postGameGen, cambia-1238) and by the countdown's game start
	// (Hub.countdownGen, cambia-1557).
	gen uint64

	// internal marks a message the hub built for itself, which is the only kind allowed to name
	// an underscore-prefixed type. Unexported for the same reason gen is: ReadPump parses the
	// frame into the exported fields alone, so nothing off a socket can set it. dispatch used to
	// tell the two apart by the absence of a ConnID and UserID, which fails in the dangerous
	// direction: a socket-side producer that forgot to stamp identity would have walked through,
	// while an internal producer that forgets this flag is refused loudly (cambia-1239).
	internal bool
}
