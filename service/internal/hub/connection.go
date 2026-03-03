// internal/hub/connection.go
package hub

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
)

// Connection represents a single client WebSocket connection to a hub.
type Connection struct {
	ID       uuid.UUID
	UserID   uuid.UUID
	Username string
	IsHost   bool
	ws       *websocket.Conn
	outChan  chan []byte // buffered 32, pre-marshaled JSON
	cancel   context.CancelFunc
}

// NewConnection creates a new Connection.
func NewConnection(id, userID uuid.UUID, username string, isHost bool, ws *websocket.Conn, cancel context.CancelFunc) *Connection {
	return &Connection{
		ID:       id,
		UserID:   userID,
		Username: username,
		IsHost:   isHost,
		ws:       ws,
		outChan:  make(chan []byte, 32),
		cancel:   cancel,
	}
}

// ReadPump reads WebSocket JSON text frames in a loop.
// Parses each frame into a ClientMsg and sends it to the incoming channel.
// Returns when the context is done or a read error occurs.
func (c *Connection) ReadPump(ctx context.Context, incoming chan<- ClientMsg) {
	defer func() {
		log.Printf("hub: ReadPump exiting for conn %s (user %s)", c.ID, c.UserID)
	}()
	for {
		msgType, data, err := c.ws.Read(ctx)
		if err != nil {
			return
		}
		if msgType != websocket.MessageText {
			continue
		}

		// Parse last_seq, type, and optionally body from the raw frame.
		// If "body" is not explicitly provided, fall back to using the entire
		// frame as Body so handlers can parse top-level fields directly.
		var raw struct {
			LastSeq uint64          `json:"last_seq"`
			Type    string          `json:"type"`
			Body    json.RawMessage `json:"body,omitempty"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			log.Printf("hub: invalid JSON from conn %s: %v", c.ID, err)
			continue
		}

		body := raw.Body
		if len(body) == 0 {
			// No explicit "body" field — use the full frame so handlers
			// can parse top-level fields (e.g. { type: "chat", msg: "hi" }).
			body = json.RawMessage(data)
		}

		msg := ClientMsg{
			ConnID:  c.ID,
			UserID:  c.UserID,
			LastSeq: raw.LastSeq,
			Type:    raw.Type,
			Body:    body,
		}

		select {
		case incoming <- msg:
		case <-ctx.Done():
			return
		}
	}
}

// WritePump drains outChan and writes each []byte as a text frame.
// Pings every 30s. Returns when the context is done or outChan is closed.
func (c *Connection) WritePump(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case data, ok := <-c.outChan:
			if !ok {
				return
			}
			writeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			err := c.ws.Write(writeCtx, websocket.MessageText, data)
			cancel()
			if err != nil {
				log.Printf("hub: write error for conn %s: %v", c.ID, err)
				c.cancel()
				return
			}
		case <-ticker.C:
			pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			err := c.ws.Ping(pingCtx)
			cancel()
			if err != nil {
				log.Printf("hub: ping error for conn %s: %v", c.ID, err)
				c.cancel()
				return
			}
		}
	}
}

// Send performs a non-blocking send to outChan.
// If the channel is full, logs a warning and closes the connection.
func (c *Connection) Send(data []byte) {
	select {
	case c.outChan <- data:
	default:
		log.Printf("hub: outChan full for conn %s (user %s), disconnecting", c.ID, c.UserID)
		c.cancel()
	}
}

// Close cancels the connection's context and closes the WebSocket.
func (c *Connection) Close() {
	c.cancel()
	c.ws.Close(websocket.StatusGoingAway, "connection closed")
}

// SendEnvelope marshals an Envelope and sends it via Send.
func (c *Connection) SendEnvelope(env Envelope) {
	data, err := json.Marshal(env)
	if err != nil {
		log.Printf("hub: failed to marshal envelope for conn %s: %v", c.ID, err)
		return
	}
	c.Send(data)
}
