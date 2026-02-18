// internal/handlers/game_ws.go
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/sirupsen/logrus"
)

// GameMessage represents the structure for incoming WebSocket messages during the game phase.
type GameMessage struct {
	Type string `json:"type"`

	// Card represents the primary card involved in an action (discard, snap, replace, peek).
	// Using map[string]interface{} allows flexibility in the client sending optional fields like idx.
	Card map[string]interface{} `json:"card,omitempty"`

	// Card1 and Card2 are used for special actions involving two cards (swaps).
	Card1 map[string]interface{} `json:"card1,omitempty"`
	Card2 map[string]interface{} `json:"card2,omitempty"`

	// Special identifies the specific sub-action for action_special messages
	// (e.g., "peek_self", "skip", "swap_blind", "swap_peek_swap").
	Special string `json:"special,omitempty"`

	// Payload provides a generic container for any additional data, though most actions
	// use specific top-level fields based on the specification.
	Payload map[string]interface{} `json:"payload,omitempty"`
}

// GameWSHandler upgrades the HTTP connection to WebSocket for a specific game instance.
// It authenticates the user, verifies they belong to the game, registers the connection,
// and then starts the read loop to handle incoming game messages.
func GameWSHandler(logger *logrus.Logger, gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Extract Game ID from URL path: /game/ws/{game_id}
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/game/ws/"), "/")
		if len(pathParts) < 1 || pathParts[0] == "" {
			http.Error(w, "Missing game_id in path (/game/ws/{game_id})", http.StatusBadRequest)
			return
		}
		gameIDStr := pathParts[0]
		gameID, err := uuid.Parse(gameIDStr)
		if err != nil {
			http.Error(w, "Invalid game_id format", http.StatusBadRequest)
			return
		}

		// Find the game instance.
		g, ok := gs.GameStore.GetGame(gameID)
		if !ok {
			http.Error(w, "Game not found", http.StatusNotFound)
			return
		}
		if g.GameOver {
			http.Error(w, "Game has already ended", http.StatusGone)
			return
		}

		// Upgrade WebSocket connection.
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			Subprotocols:   []string{"game"},
			OriginPatterns: []string{"*"}, // Adjust for production security.
		})
		if err != nil {
			logger.Warnf("WebSocket accept error for game %s: %v", gameID, err)
			return
		}
		defer c.Close(websocket.StatusInternalError, "Internal server error during handler exit.")

		if c.Subprotocol() != "game" {
			logger.Warnf("Client for game %s connected with invalid subprotocol: %s", gameID, c.Subprotocol())
			c.Close(websocket.StatusPolicyViolation, "Client must use the 'game' subprotocol.")
			return
		}
		logger.Infof("WebSocket connection established for game %s from %s", gameID, r.RemoteAddr)

		// Authenticate user, potentially creating an ephemeral guest user.
		userID, err := EnsureEphemeralUser(w, r)
		if err != nil {
			logger.Warnf("User authentication failed for game %s: %v", gameID, err)
			c.Close(websocket.StatusPolicyViolation, "Authentication failed.")
			return
		}
		logger.Infof("User %s authenticated for game %s", userID, gameID)

		// Verify the authenticated user is a player in this specific game.
		isPlayerInGame := false
		g.Mu.Lock()
		for _, p := range g.Players {
			if p.ID == userID {
				isPlayerInGame = true
				break
			}
		}
		g.Mu.Unlock()
		if !isPlayerInGame {
			logger.Warnf("User %s is not a player in game %s. Closing connection.", userID, gameID)
			c.Close(websocket.StatusPolicyViolation, "You are not a player in this game.")
			return
		}

		// Register broadcast functions if they haven't been set up yet for this game instance.
		// These functions handle sending events to clients, managing locks appropriately.
		g.Mu.Lock()
		if g.BroadcastFn == nil {
			g.BroadcastFn = createBroadcastFunc(g, logger)
		}
		if g.BroadcastToPlayerFn == nil {
			g.BroadcastToPlayerFn = createBroadcastToPlayerFunc(g, logger)
		}
		g.Mu.Unlock()

		// Handle player connection/reconnection within the game logic.
		// This updates the player's connection status and sends initial state.
		g.HandleReconnect(userID, c) // Needs game lock internally.

		// Start reading messages from the client in a blocking loop.
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel() // Ensure context cancellation propagates on exit.

		readGameMessages(ctx, c, g, userID, logger)

		// Cleanup after readGameMessages returns (due to error, closure, or context cancellation).
		logger.Infof("Player %s WebSocket read loop exited for game %s.", userID, gameID)
		g.HandleDisconnect(userID) // Mark player as disconnected in game state.
		logger.Infof("Player %s cleanup complete for game %s.", userID, gameID)
		// The deferred c.Close handles the actual WebSocket closure.
	}
}

// createBroadcastFunc returns a function suitable for CambiaGame.BroadcastFn.
// It marshals the event and sends it asynchronously to all connected players.
func createBroadcastFunc(g *game.CambiaGame, logger *logrus.Logger) func(ev game.GameEvent) {
	return func(ev game.GameEvent) {
		// This function is called *while the game lock is held*.
		// We must release the lock before writing to WebSockets to avoid blocking game logic.

		playersToSend := []*models.Player{}
		g.Mu.Lock() // Acquire lock briefly to get current connected players.
		for _, p := range g.Players {
			if p.Connected && p.Conn != nil {
				playersToSend = append(playersToSend, p)
			}
		}
		g.Mu.Unlock() // Release lock before marshaling and sending.

		msgBytes, err := json.Marshal(ev)
		if err != nil {
			logger.Errorf("Failed to marshal broadcast event (%s) for game %s: %v", ev.Type, g.ID, err)
			return
		}

		// Send asynchronously.
		go func(players []*models.Player, data []byte, gameID uuid.UUID) {
			for _, pl := range players {
				// Check connection status again *before* writing, as it might have changed.
				if pl.Conn != nil {
					ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second) // Write timeout.
					err := pl.Conn.Write(ctx, websocket.MessageText, data)
					cancel()
					if err != nil {
						logger.Warnf("Failed to write broadcast message to player %s in game %s: %v", pl.ID, gameID, err)
						// Consider triggering disconnect handling more formally here if needed.
					}
				}
			}
		}(playersToSend, msgBytes, g.ID)
	}
}

// createBroadcastToPlayerFunc returns a function suitable for CambiaGame.BroadcastToPlayerFn.
// It finds the target player, marshals the event, and sends it asynchronously.
func createBroadcastToPlayerFunc(g *game.CambiaGame, logger *logrus.Logger) func(targetPlayerID uuid.UUID, ev game.GameEvent) {
	return func(targetPlayerID uuid.UUID, ev game.GameEvent) {
		// This function is also called *while the game lock is held*.

		var targetConn *websocket.Conn
		g.Mu.Lock() // Acquire lock briefly to find the target connection.
		for _, pl := range g.Players {
			if pl.ID == targetPlayerID {
				if pl.Connected && pl.Conn != nil {
					targetConn = pl.Conn
				}
				break
			}
		}
		g.Mu.Unlock() // Release lock before marshaling and sending.

		if targetConn != nil {
			msgBytes, err := json.Marshal(ev)
			if err != nil {
				logger.Errorf("Failed to marshal private event (%s) for player %s in game %s: %v", ev.Type, targetPlayerID, g.ID, err)
				return
			}
			// Send asynchronously.
			go func(conn *websocket.Conn, data []byte, playerID uuid.UUID, gameID uuid.UUID) {
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				err := conn.Write(ctx, websocket.MessageText, data)
				cancel()
				if err != nil {
					logger.Warnf("Failed to write private message to player %s in game %s: %v", playerID, gameID, err)
				}
			}(targetConn, msgBytes, targetPlayerID, g.ID)
		}
	}
}

// readGameMessages continuously reads messages from a client's WebSocket connection,
// unmarshals them, validates the action based on game state, and routes them
// to the appropriate game logic handler (HandlePlayerAction or ProcessSpecialAction).
// It operates within the connection's context and exits upon error or cancellation.
func readGameMessages(ctx context.Context, c *websocket.Conn, g *game.CambiaGame, userID uuid.UUID, logger *logrus.Logger) {
	for {
		msgType, data, err := c.Read(ctx)
		if err != nil {
			// Handle read errors (connection closed, context cancelled, etc.)
			status := websocket.CloseStatus(err)
			if status == websocket.StatusNormalClosure || status == websocket.StatusGoingAway {
				logger.Infof("WebSocket closed normally for user %s in game %s.", userID, g.ID)
			} else if strings.Contains(err.Error(), "context canceled") {
				logger.Infof("WebSocket context canceled for user %s in game %s.", userID, g.ID)
			} else {
				logger.Warnf("Error reading from WebSocket for user %s in game %s: %v (Status: %d)", userID, g.ID, err, status)
			}
			return // Exit loop on error/closure/cancelation.
		}

		if msgType != websocket.MessageText {
			logger.Warnf("Received non-text message type %d from user %s in game %s. Ignoring.", msgType, userID, g.ID)
			continue
		}

		var msg GameMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			logger.Warnf("Invalid JSON received from user %s in game %s: %v. Data: %s", userID, g.ID, err, string(data))
			sendWsError(ctx, c, "Invalid JSON format.")
			continue
		}

		logger.Debugf("Received action '%s' from user %s in game %s.", msg.Type, userID, g.ID)

		// Acquire game lock before accessing or modifying game state.
		g.Mu.Lock()

		// Check if game is over before processing action.
		if g.GameOver {
			logger.Warnf("Game %s is over. Ignoring action '%s' from user %s.", g.ID, msg.Type, userID)
			g.Mu.Unlock()
			continue
		}

		// Route the message based on its type.
		switch msg.Type {
		case "action_draw_stockpile", "action_draw_discardpile",
			"action_discard", "action_replace", "action_cambia", "action_snap":
			// Prepare the GameAction struct for standard actions.
			gameAction := models.GameAction{
				ActionType: msg.Type,
				Payload:    make(map[string]interface{}),
			}
			// Populate payload primarily from msg.Card if present, otherwise msg.Payload.
			if msg.Card != nil {
				gameAction.Payload = msg.Card
			} else if msg.Payload != nil {
				gameAction.Payload = msg.Payload
			}
			// Call the main action handler (assumes lock is held).
			g.HandlePlayerAction(userID, gameAction)

		case "action_special":
			// Call the dedicated special action processor (assumes lock is held).
			g.ProcessSpecialAction(userID, msg.Special, msg.Card1, msg.Card2)

		case "ping":
			// Respond to ping immediately without holding the lock for the write.
			g.Mu.Unlock() // Release lock before writing.
			logger.Tracef("Received ping from user %s, sending pong.", userID)
			sendWsMessage(ctx, c, map[string]string{"type": "pong"})
			g.Mu.Lock() // Re-acquire lock after write for loop consistency.

		default:
			logger.Warnf("Unknown action type '%s' from user %s in game %s.", msg.Type, userID, g.ID)
			sendWsError(ctx, c, fmt.Sprintf("Unknown action type: %s", msg.Type))
		}

		// Release the lock after processing the action for this iteration.
		g.Mu.Unlock()

		// Check context cancellation after processing each message.
		select {
		case <-ctx.Done():
			logger.Infof("Context canceled after processing message for user %s in game %s.", userID, g.ID)
			return // Exit loop if context is cancelled.
		default:
			// Continue to the next read iteration.
		}
	}
}

// sendWsMessage marshals a message and sends it to the WebSocket client.
// Includes logging for errors and uses a write timeout.
func sendWsMessage(ctx context.Context, c *websocket.Conn, message interface{}) {
	if c == nil {
		log.Println("Error: Attempted to send WebSocket message on nil connection.")
		return
	}
	msgBytes, err := json.Marshal(message)
	if err != nil {
		log.Printf("Error marshaling WebSocket message: %v", err)
		return
	}

	// Use a dedicated context with timeout for the write operation.
	writeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = c.Write(writeCtx, websocket.MessageText, msgBytes)
	if err != nil {
		status := websocket.CloseStatus(err)
		if status != websocket.StatusNormalClosure && status != websocket.StatusGoingAway && !strings.Contains(err.Error(), "context deadline exceeded") {
			log.Printf("Error writing WebSocket message: %v (Status: %d)", err, status)
		} else if strings.Contains(err.Error(), "context deadline exceeded") {
			log.Printf("Timeout writing WebSocket message: %v", err)
		}
		// Let the read loop handle connection closure detection.
	}
}

// sendWsError sends a structured error message to the client.
func sendWsError(ctx context.Context, c *websocket.Conn, errorMsg string) {
	sendWsMessage(ctx, c, map[string]interface{}{
		"type":    "error",
		"message": errorMsg,
	})
}
