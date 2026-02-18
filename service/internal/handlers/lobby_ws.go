// internal/handlers/lobby_ws.go
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
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/sirupsen/logrus"
)

var GameServerForLobbyWS *GameServer

// LobbyWSHandler sets up the ephemeral in-memory WS flow.
func LobbyWSHandler(logger *logrus.Logger, gs *GameServer) http.HandlerFunc {
	GameServerForLobbyWS = gs // Store reference to GameServer
	return func(w http.ResponseWriter, r *http.Request) {
		remoteAddr := r.RemoteAddr // Get remote address here for initial connection log
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/lobby/ws/"), "/")
		if len(pathParts) < 1 {
			http.Error(w, "missing lobby_id", http.StatusBadRequest)
			return
		}
		lobbyIDStr := pathParts[0]
		lobbyUUID, err := uuid.Parse(lobbyIDStr)
		if err != nil {
			http.Error(w, "invalid lobby_id", http.StatusBadRequest)
			return
		}

		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			Subprotocols:   []string{"lobby"},
			OriginPatterns: []string{"*"}, // Adjust in production
		})
		if err != nil {
			logger.Warnf("websocket accept error: %v", err)
			return
		}
		// Use a more specific close reason in defer if possible, but InternalError is a safe default
		defer c.Close(websocket.StatusInternalError, "handler finished")

		if c.Subprotocol() != "lobby" {
			c.Close(BadSubprotocolError, "client must speak the lobby subprotocol")
			return
		}

		// Authenticate user
		userUUID, err := EnsureEphemeralUser(w, r) // Handle potential cookie setting
		if err != nil {
			logger.Warnf("User authentication failed for lobby %s: %v", lobbyUUID, err)
			c.Close(websocket.StatusPolicyViolation, "Authentication failed.")
			return
		}

		lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
		if !exists {
			c.Close(InvalidLobbyIDError, "lobby does not exist")
			return
		}

		// Check if private lobby and user is invited/present
		lob.Mu.Lock()
		isInvitedOrPresent := false
		if _, ok := lob.Users[userUUID]; ok {
			isInvitedOrPresent = true
		}
		lobbyType := lob.Type
		lob.Mu.Unlock()

		if lobbyType == "private" && !isInvitedOrPresent {
			c.Close(websocket.StatusPolicyViolation, "user not invited to private lobby")
			return
		}

		// Create ephemeral connection
		ctx, cancel := context.WithCancel(r.Context())
		conn := &lobby.LobbyConnection{
			UserID:  userUUID,
			Cancel:  cancel, // Store cancel func to stop associated goroutines
			OutChan: make(chan map[string]interface{}, 10),
			IsHost:  (lob.HostUserID == userUUID),
		}

		// Add connection to lobby (this acquires lock)
		if err := lob.AddConnection(userUUID, conn); err != nil {
			logger.Warnf("failed AddConnection: %v", err)
			c.Close(websocket.StatusPolicyViolation, fmt.Sprintf("AddConnection error: %v", err))
			cancel() // Cancel context if AddConnection fails
			return
		}

		// Log initial connection with remoteAddr
		logger.Infof("User %v (%s) connected to lobby %v", userUUID, remoteAddr, lobbyUUID)

		// Start write pump in a goroutine
		go writePump(ctx, c, conn, logger)

		// Start read pump (blocks until connection closes or error)
		readPump(ctx, c, lob, conn, logger, lobbyUUID) // Pass *lobby.Lobby

		// ---- Cleanup after readPump exits ----
		logger.Infof("User %v readPump exited for lobby %v. Initiating cleanup.", userUUID, lobbyUUID)
		lob.RemoveUser(userUUID) // Ensure user is removed from lobby state (this acquires lock)
		// cancel() should be called automatically when ctx is done.
		// The defer c.Close in the main handler scope ensures the WebSocket connection is closed.
	}
}

// readPump handles incoming messages from the lobby websocket.
// Acquires lobby lock before calling handleLobbyMessage and releases it afterwards,
// unless handleLobbyMessage signals otherwise (e.g., for leave_lobby).
func readPump(ctx context.Context, c *websocket.Conn, lob *lobby.Lobby, conn *lobby.LobbyConnection, logger *logrus.Logger, lobbyID uuid.UUID) {
	logger.Infof("Lobby %s: Starting read pump for user %v", lobbyID, conn.UserID)
	defer logger.Infof("Lobby %s: Exiting read pump for user %v", lobbyID, conn.UserID)

	for {
		select {
		case <-ctx.Done(): // Check context cancellation first
			logger.Infof("Lobby %s: Context cancelled for user %v, stopping read pump.", lobbyID, conn.UserID)
			return
		default:
			// Proceed with reading
		}

		typ, msg, err := c.Read(ctx)
		if err != nil {
			// Handle read errors (connection closed, context cancelled, etc.)
			closeStatus := websocket.CloseStatus(err)
			if closeStatus == websocket.StatusNormalClosure || closeStatus == websocket.StatusGoingAway {
				logger.Infof("Lobby %s: WebSocket closed normally for user %v.", lobbyID, conn.UserID)
			} else if strings.Contains(err.Error(), "context canceled") {
				// Already logged above, just return
			} else {
				logger.Warnf("Lobby %s: Read error for user %v: %v (CloseStatus: %d)", lobbyID, conn.UserID, err, closeStatus)
			}
			return // Exit loop on any read error or context cancellation
		}

		if typ != websocket.MessageText {
			logger.Warnf("Lobby %s: Received non-text message type %d from user %v. Ignoring.", lobbyID, typ, conn.UserID)
			continue
		}

		var packet map[string]interface{}
		if err := json.Unmarshal(msg, &packet); err != nil {
			logger.Warnf("Lobby %s: Invalid json from user %v: %v", lobbyID, conn.UserID, err)
			conn.WriteError("Invalid JSON format") // Send error back to client
			continue
		}

		lockReleasedByHandler := false // Flag to track if handler releases the lock
		shouldStartCountdown := false  // Flag to start countdown after releasing lock

		// Acquire lock before handling the message
		lob.Mu.Lock()

		// Ensure user is still connected before handling
		currentConn, stillConnected := lob.Connections[conn.UserID]
		if !stillConnected || currentConn != conn {
			logger.Warnf("Lobby %s: Ignoring action from user %s who disconnected or reconnected during handling.", lob.ID, conn.UserID)
			lob.Mu.Unlock() // Release lock if skipping
			continue        // Skip handling if user disconnected or this is a stale connection instance
		}

		// Handle the message while holding the lock
		// Pass a callback function to release the lock if needed by the handler
		// Pass a pointer to shouldStartCountdown so handler can signal it
		handleLobbyMessage(packet, lob, conn, logger, &shouldStartCountdown, func() {
			lob.Mu.Unlock()
			lockReleasedByHandler = true
		})

		// Release lock ONLY if the handler did not release it
		if !lockReleasedByHandler {
			lob.Mu.Unlock()
		}

		// Start countdown AFTER releasing the lock if signaled by handler
		if shouldStartCountdown {
			lob.StartCountdown(10, func(l *lobby.Lobby) { // StartCountdown acquires lock internally
				logger.Infof("Lobby %s: Auto-start countdown finished.", l.ID)
				// GameServer must be accessible (e.g., global or passed)
				if GameServerForLobbyWS == nil {
					logger.Errorf("Lobby %s: GameServerForLobbyWS is nil, cannot start game.", l.ID)
					return
				}
				g := GameServerForLobbyWS.NewCambiaGameFromLobby(context.Background(), l) // This needs locking internally or careful handling
				l.Mu.Lock()                                                               // Lock needed to update lobby state after game creation
				l.InGame = true
				l.GameID = g.ID
				gameStartPayload := map[string]interface{}{
					"type":    "game_start",
					"game_id": g.ID.String(),
				}
				l.BroadcastAllUnsafe(gameStartPayload)
				l.Mu.Unlock()
			})
		}
	}
}

// handleLobbyMessage interprets the "type" field for ephemeral lobby logic.
// Assumes the lobby lock is HELD by the caller (readPump).
// Uses the unlockCallback function to release the lock before long operations or returning early.
// Uses shouldStartCountdown pointer to signal if countdown should start after lock release.
func handleLobbyMessage(packet map[string]interface{}, lob *lobby.Lobby, senderConn *lobby.LobbyConnection, logger *logrus.Logger, shouldStartCountdown *bool, unlockCallback func()) {
	action, _ := packet["type"].(string)
	// logger.Debugf("Lobby %s: Handling action '%s' from user %s", lob.ID, action, senderConn.UserID) // Reduce Noise

	// Note: Lock is held by caller (readPump). unlockCallback MUST be called before returning early
	// or before long operations like game creation.

	switch action {
	case "ready":
		// MarkUserReady now calls unsafe and returns bool
		if lob.MarkUserReady(senderConn.UserID) {
			*shouldStartCountdown = true // Signal to caller (readPump) to start countdown after lock release
		}
	case "unready":
		lob.MarkUserUnready(senderConn.UserID) // Calls unsafe internally
	case "invite":
		userIDStr, _ := packet["userID"].(string)
		userToAdd, err := uuid.Parse(userIDStr)
		if err != nil {
			logger.Warnf("Lobby %s: Invalid user ID to invite: %v", lob.ID, packet["userID"])
			senderConn.WriteError("Invalid userID format for invite")
			return // Lock released by readPump
		}
		lob.InviteUser(userToAdd) // Calls unsafe internally
	case "leave_lobby":
		userID := senderConn.UserID // Get user ID before unlocking
		unlockCallback()            // Release the lock held by readPump
		lob.RemoveUser(userID)      // RemoveUser manages its own lock
		return                      // Exit handler immediately, lock already released
	case "chat":
		msg, _ := packet["msg"].(string)
		if msg != "" {
			lob.BroadcastChat(senderConn.UserID, msg) // Calls unsafe internally
		}
	case "update_rules":
		if !senderConn.IsHost {
			senderConn.WriteError("Only the host can update rules")
			return // Lock released by readPump
		}
		if rulesData, ok := packet["rules"].(map[string]interface{}); ok {
			// Call UpdateUnsafe directly as lock is held
			if err := lob.UpdateUnsafe(rulesData); err != nil {
				logger.Warnf("Lobby %s: Error during UpdateUnsafe: %v", lob.ID, err)
				senderConn.WriteError("Failed to apply rule updates.")
			}
			// Broadcast happens inside UpdateUnsafe if changes occurred
		} else {
			logger.Warnf("Lobby %s: Received update_rules without valid 'rules' field from host %s", lob.ID, senderConn.UserID)
			senderConn.WriteError("Invalid payload for update_rules")
			// Lock released by readPump
		}
	case "start_game":
		if !senderConn.IsHost {
			senderConn.WriteError("Only the host can force start")
			return // Lock released by readPump
		}
		if lob.InGame {
			senderConn.WriteError("Game already in progress")
			return // Lock released by readPump
		}
		if !lob.AreAllReadyUnsafe() { // Use unsafe as lock is held
			senderConn.WriteError("Not all users are ready")
			return // Lock released by readPump
		}
		lob.CancelCountdownUnsafe() // Use unsafe as lock is held

		// --- Game Start Sequence ---
		// 1. Prepare necessary info while lock is held
		lobbyID := lob.ID
		hostID := lob.HostUserID
		// lobbyType := lob.Type
		gameMode := lob.GameMode
		houseRules := lob.HouseRules // Deep copy might be needed if rules can change post-start
		circuit := lob.Circuit       // Deep copy?
		// Extract player IDs and connections safely
		playersToStart := make([]*lobby.LobbyConnection, 0, len(lob.Connections))
		for _, conn := range lob.Connections {
			playersToStart = append(playersToStart, conn)
		}

		// 2. Release lock BEFORE potentially long-running game creation
		unlockCallback()
		logger.Infof("Lobby %s: Released lock, attempting game creation...", lobbyID)

		// 3. Ensure GameServer is available
		if GameServerForLobbyWS == nil {
			logger.Errorf("Lobby %s: GameServerForLobbyWS is nil, cannot start game on host command.", lobbyID)
			// How to signal error back? We released the lock. Maybe log and do nothing.
			return
		}

		// 4. Create the game (pass necessary info)
		// Modify NewCambiaGameFromLobby to accept necessary parameters instead of the whole lobby object
		// to avoid passing the locked object. Let's assume it's modified or create a helper.
		g := GameServerForLobbyWS.CreateGameInstance(context.Background(), lobbyID, hostID, gameMode, houseRules, circuit, playersToStart)
		if g == nil {
			logger.Errorf("Lobby %s: Failed to create game instance.", lobbyID)
			// Maybe try to signal an error back? Difficult.
			return
		}
		logger.Infof("Lobby %s: Game instance %s created.", lobbyID, g.ID)

		// 5. Re-acquire lock to update lobby state AFTER game creation
		lob.Mu.Lock()
		// Ensure lobby still exists and wasn't deleted during game creation
		// This check is implicitly handled by the fact that `lob` pointer is still valid
		// if we got here without panicking.

		// Verify host is still connected after game creation
		if _, stillConnected := lob.Connections[senderConn.UserID]; !stillConnected {
			logger.Warnf("Lobby %s: Host %s disconnected during game creation for start_game. Game %s might be orphaned.", lob.ID, senderConn.UserID, g.ID)
			// Clean up game instance?
			GameServerForLobbyWS.GameStore.DeleteGame(g.ID)
			lob.Mu.Unlock()
			return
		}

		// Check if lobby is already in game (e.g., race condition?)
		if lob.InGame {
			logger.Warnf("Lobby %s: Lobby is already marked InGame after game creation attempt. Ignoring.", lob.ID)
			// Clean up potentially duplicate game instance?
			GameServerForLobbyWS.GameStore.DeleteGame(g.ID)
			lob.Mu.Unlock() // Release lock before returning
			return
		}

		// Update lobby state
		lob.InGame = true
		lob.GameID = g.ID // Store game ID
		gameStartPayload := map[string]interface{}{
			"type":    "game_start",
			"game_id": g.ID.String(),
		}
		// Broadcast game start using the lock we just re-acquired
		lob.BroadcastAllUnsafe(gameStartPayload)
		logger.Infof("Lobby %s: Broadcasted game_start for game %s.", lob.ID, g.ID)
		// Lock will be released by the calling readPump after this function returns

	default:
		logger.Warnf("Lobby %s: Unknown action '%s' from user %v", lob.ID, action, senderConn.UserID)
		senderConn.WriteError(fmt.Sprintf("Unknown action type: %s", action))
		// Lock released by readPump
	}
}

// writePump remains largely the same
func writePump(ctx context.Context, c *websocket.Conn, conn *lobby.LobbyConnection, logger *logrus.Logger) {
	ticker := time.NewTicker(30 * time.Second) // Send pings periodically
	defer ticker.Stop()
	// logger.Infof("Lobby: Starting write pump for user %v", conn.UserID) // Reduce noise

	defer func() {
		// logger.Infof("Lobby: Write pump for user %v stopping.", conn.UserID) // Reduce noise
		// Attempt to send a close frame before function exits, useful if readPump hasn't detected closure yet
		// Use a short timeout for this attempt
		_, closeCancel := context.WithTimeout(context.Background(), 1*time.Second)
		_ = c.Close(websocket.StatusGoingAway, "Write pump stopping") // Or StatusNormalClosure if appropriate
		closeCancel()
		// Ensure channel is closed if writePump exits abnormally
		// Check if channel is already closed before trying to close it again
		// This requires careful synchronization or a select{default} approach
		// For simplicity, let RemoveUser handle channel closure robustly.
	}()

	for {
		select {
		case <-ctx.Done():
			// logger.Infof("Lobby: Write pump stopping for user %v due to context done.", conn.UserID) // Reduce noise
			return
		case msg, ok := <-conn.OutChan:
			if !ok {
				// Channel closed, client likely disconnected or removed
				// logger.Infof("Lobby: OutChan closed for user %v, stopping write pump.", conn.UserID) // Reduce noise
				return
			}
			data, err := json.Marshal(msg)
			if err != nil {
				logger.Warnf("Lobby: Failed to marshal outgoing msg for user %v: %v", conn.UserID, err)
				continue
			}

			writeCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // Timeout for write operation based on parent ctx
			err = c.Write(writeCtx, websocket.MessageText, data)
			cancel() // Important to cancel the context associated with the write

			if err != nil {
				logger.Warnf("Lobby: Failed to write to websocket for user %v: %v", conn.UserID, err)
				// If write fails, the connection is likely broken. Exit writePump.
				// readPump should detect the closure via its read error or context cancellation.
				return
			}
		case <-ticker.C:
			// Send ping
			// Increase timeout for ping to 15 seconds
			pingCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
			err := c.Ping(pingCtx)
			cancel() // Important to cancel the context associated with the ping
			if err != nil {
				logger.Warnf("Lobby: Failed to send ping to user %v: %v. Assuming disconnect.", conn.UserID, err)
				// Exit write pump on ping failure. readPump should handle the disconnect.
				return
			}
			// logger.Tracef("Lobby: Sent ping to user %v", conn.UserID) // Optional: Trace logging for successful ping
		}
	}
}

// --- Helper for Game Creation ---
// This function should be part of GameServer or a related service.
// It encapsulates creating the game instance without needing the Lobby lock.
func (gs *GameServer) CreateGameInstance(ctx context.Context, lobbyID, hostID uuid.UUID, gameMode string, houseRules game.HouseRules, circuit game.Circuit, playersToStart []*lobby.LobbyConnection) *game.CambiaGame {
	// 1. Create the basic game instance
	g := game.NewCambiaGame()
	g.LobbyID = lobbyID
	g.HouseRules = houseRules
	g.Circuit = circuit
	// g.GameMode = gameMode // Add GameMode field to CambiaGame if needed

	// 2. Populate players
	var players []*models.Player
	playerIDs := []uuid.UUID{} // Keep track of IDs for OnGameEnd mapping if needed
	for _, conn := range playersToStart {
		// Need user details potentially? For now, just ID.
		players = append(players, &models.Player{
			ID:        conn.UserID,
			Connected: true, // Assume connected at game start
			Hand:      []*models.Card{},
			// Username could be passed from conn.Username if needed by game logic/logging
		})
		playerIDs = append(playerIDs, conn.UserID)
	}
	if len(players) < 2 { // Or other minimum required by gameMode
		log.Printf("Lobby %s: Cannot start game, not enough players (%d).", lobbyID, len(players))
		return nil // Return nil if not enough players
	}
	g.Players = players

	// 3. Attach OnGameEnd callback
	g.OnGameEnd = func(endedLobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int) {
		log.Printf("Game %s ended. Callback executing for lobby %s.", g.ID, endedLobbyID)
		// Find lobby again using the GameServer's store
		lobInstance, exists := gs.LobbyStore.GetLobby(endedLobbyID)
		if !exists {
			log.Printf("Error in OnGameEnd: Lobby %s not found in store.", endedLobbyID)
			gs.GameStore.DeleteGame(g.ID) // Clean up game if lobby is gone
			return
		}

		lobInstance.Mu.Lock() // Lock lobby before modifying
		lobInstance.InGame = false
		lobInstance.GameID = uuid.Nil // Clear game ID reference

		// Reset ready states for connected players
		for uid := range lobInstance.Connections {
			lobInstance.ReadyStates[uid] = false
		}
		// Get current status payload *after* resetting ready states
		statusPayload := lobInstance.GetLobbyStatusPayloadUnsafe()

		// Prepare results message combining results and new lobby status
		resultMsg := map[string]interface{}{
			"type":         "game_results", // Client might need a specific type
			"winner":       winner.String(),
			"scores":       map[string]int{},
			"lobby_status": statusPayload, // Include updated lobby status
		}
		for pid, sc := range scores {
			resultMsg["scores"].(map[string]int)[pid.String()] = sc
		}
		lobInstance.Mu.Unlock() // Unlock before broadcasting

		// Broadcast results back to the lobby
		log.Printf("Broadcasting game end results to lobby %s", endedLobbyID)
		lobInstance.BroadcastAll(resultMsg) // BroadcastAll handles its own locking

		// Clean up game instance from store after results are sent
		gs.GameStore.DeleteGame(g.ID)
		log.Printf("Game %s instance removed from store.", g.ID)
	}

	// 4. Store and start the game
	gs.GameStore.AddGame(g)
	g.BeginPreGame() // Use BeginPreGame to start the flow

	return g
}
