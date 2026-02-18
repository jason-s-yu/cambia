// internal/lobby/lobby.go
package lobby

import (
	"context" // Added for marshaling
	"fmt"
	"log" // Added for logging
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database" // Import database package
	"github.com/jason-s-yu/cambia/service/internal/game"     // Import game package for rules structs
)

// Lobby is an ephemeral grouping of users with chat, rules, ready states, etc.
type Lobby struct {
	ID         uuid.UUID `json:"id"`
	HostUserID uuid.UUID `json:"hostUserID"`
	Type       string    `json:"type"`
	GameMode   string    `json:"gameMode"`

	// Users maps userID -> whether they've joined (true) or only invited (false).
	Users map[uuid.UUID]bool `json:"-"`

	// Connections holds the actual live WebSocket connections for joined users.
	Connections map[uuid.UUID]*LobbyConnection `json:"-"`
	// ReadyStates holds userID -> bool for "is ready".
	ReadyStates map[uuid.UUID]bool `json:"-"`

	GameInstanceCreated bool      `json:"-"`
	GameID              uuid.UUID `json:"gameId,omitempty"` // Include GameID, omitempty if nil
	InGame              bool      `json:"inGame"`

	CountdownTimer *time.Timer `json:"-"`

	// HouseRules is embedded directly from the game package
	HouseRules game.HouseRules `json:"houseRules"` // Use game.HouseRules
	// Circuit settings embedded directly from the game package
	Circuit game.Circuit `json:"circuit"` // Use game.Circuit

	LobbySettings LobbySettings `json:"lobbySettings"`

	// OnEmpty is called if we detect the lobby is empty (0 users) after removing a user.
	// Typically assigned by the code that creates & stores this lobby, e.g. via
	//   lobby.OnEmpty = func(lobbyID uuid.UUID) { store.DeleteLobby(lobbyID) }
	OnEmpty func(lobbyID uuid.UUID) `json:"-"`

	// Mutex to protect concurrent access to lobby state, especially connections and ready states
	Mu sync.Mutex
}

// LobbyConnection is a single user's presence in the lobby.
type LobbyConnection struct {
	UserID   uuid.UUID
	Username string // Added field to store username
	Cancel   func()
	OutChan  chan map[string]interface{}
	IsHost   bool
}

// Write pushes a message onto the user's OutChan non-blockingly. Logs if blocked/dropped.
func (conn *LobbyConnection) Write(msg map[string]interface{}) {
	select {
	case conn.OutChan <- msg:
		// Message sent successfully
	default:
		// Channel is likely closed or full. Log the dropped message type.
		msgType, _ := msg["type"].(string)
		log.Printf("LobbyConnection Write WARNING: OutChan for user %s closed or full. Dropped message type '%s'.", conn.UserID, msgType)
		// Consider further action if this happens frequently (e.g., metrics, faster processing)
	}
}

// WriteError is a convenience to send an error object.
func (conn *LobbyConnection) WriteError(msg string) {
	conn.Write(map[string]interface{}{
		"type":    "error",
		"message": msg,
	})
}

// LobbySettings holds settings specific to the lobby behavior.
type LobbySettings struct {
	AutoStart bool `json:"autoStart"`
}

// NewLobbyWithDefaults creates an ephemeral lobby with default house rules, etc.
func NewLobbyWithDefaults(hostID uuid.UUID) *Lobby {
	lobbyID, _ := uuid.NewRandom()
	// Use the default HouseRules struct definition from game.go
	defaultHouseRules := game.HouseRules{ // Use game.HouseRules
		AllowDrawFromDiscardPile: false,
		AllowReplaceAbilities:    false,
		SnapRace:                 false,
		ForfeitOnDisconnect:      true,
		PenaltyDrawCount:         2, // Default is 2
		AutoKickTurnCount:        3, // Example default
		TurnTimerSec:             15,
	}
	// Use the default Circuit struct definition from game.go
	defaultCircuit := game.Circuit{ // Use game.Circuit
		Enabled: false,
		Rules: game.CircuitRules{ // Use game.CircuitRules
			TargetScore:            100,  // Example default
			WinBonus:               -1,   // Example default
			FalseCambiaPenalty:     1,    // Example default
			FreezeUserOnDisconnect: true, // Example default
		},
	}

	return &Lobby{
		ID:          lobbyID,
		HostUserID:  hostID,
		Type:        "private",      // Default lobby type
		GameMode:    "head_to_head", // Default game mode
		Users:       make(map[uuid.UUID]bool),
		Connections: make(map[uuid.UUID]*LobbyConnection),
		ReadyStates: make(map[uuid.UUID]bool),

		HouseRules: defaultHouseRules, // Assign default game rules
		Circuit:    defaultCircuit,    // Assign default circuit rules

		LobbySettings: LobbySettings{
			AutoStart: true, // Default lobby setting
		},
	}
}

// InviteUser ephemeral sets userID => false in the Users map (private-lobby invitation).
// Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) InviteUser(userID uuid.UUID) {
	lobby.inviteUserUnsafe(userID) // Call unsafe version
}

// inviteUserUnsafe is the internal implementation. Assumes lock is held.
func (lobby *Lobby) inviteUserUnsafe(userID uuid.UUID) {
	// Only add if not already present or joined
	if _, exists := lobby.Users[userID]; !exists {
		lobby.Users[userID] = false // Mark as invited
		log.Printf("Lobby %s: User %s invited.", lobby.ID, userID)
		// Broadcast invite event (using unsafe broadcast as lock is held)
		lobby.BroadcastAllUnsafe(map[string]interface{}{
			"type":      "lobby_invite",
			"invitedID": userID.String(),
		})
	} else {
		log.Printf("Lobby %s: User %s already present or invited.", lobby.ID, userID)
	}
}

// AddConnection ephemeral sets user as "connected," also sets ReadyStates[userID] = false.
// Fetches and stores the username. Acquires lock.
func (lobby *Lobby) AddConnection(userID uuid.UUID, conn *LobbyConnection) error {
	lobby.Mu.Lock() // Lock is needed here as it modifies multiple maps

	// Check if user exists (was invited or previously joined)
	joined, exists := lobby.Users[userID]
	if !exists {
		// If not invited (public lobby or direct join attempt)
		if lobby.Type != "private" {
			lobby.Users[userID] = true // Add directly for non-private
		} else {
			lobby.Mu.Unlock() // Unlock before returning error
			return fmt.Errorf("user %s not invited to the private lobby %s", userID, lobby.ID)
		}
	} else if joined {
		// User is rejoining/replacing connection
		log.Printf("Lobby %s: User %s is re-establishing connection.", lobby.ID, userID)
		// Close existing connection's OutChan if replacing
		if oldConn, ok := lobby.Connections[userID]; ok && oldConn != conn {
			// Safely close channel and cancel context
			close(oldConn.OutChan) // Close channel associated with the old connection
			if oldConn.Cancel != nil {
				oldConn.Cancel() // Trigger context cancel for old connection's goroutines
			}
		}
	}

	// Fetch and store username
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	user, err := database.GetUserByID(ctx, userID)
	cancel() // Ensure context is cancelled after DB call
	if err != nil {
		log.Printf("Lobby %s: Error fetching user %s details: %v. Using default username.", lobby.ID, userID, err)
		conn.Username = fmt.Sprintf("User_%s", userID.String()[:4]) // Fallback username
	} else {
		conn.Username = user.Username
	}

	// Add/update connection and reset ready state
	lobby.Connections[userID] = conn
	lobby.ReadyStates[userID] = false
	lobby.Users[userID] = true // Mark as definitely joined now

	log.Printf("Lobby %s: User %s (%s) connected.", lobby.ID, userID, conn.Username)

	// Prepare state and join payloads while lock is held
	lobbyStatePayload := lobby.getLobbyStatePayloadUnsafe(userID)
	lobbyJoinPayload := lobby.getLobbyJoinPayloadUnsafe(userID)

	lobby.Mu.Unlock() // Unlock BEFORE sending messages/broadcasting

	// Send initial state and broadcast join AFTER releasing the lock
	go func() {
		conn.Write(lobbyStatePayload)        // Send private state first
		lobby.BroadcastAll(lobbyJoinPayload) // Then broadcast join (this method acquires lock again)
	}()

	return nil
}

// RemoveUser ephemeral: remove from Users, Connections, ReadyStates. If empty => call OnEmpty callback.
// Acquires lock.
func (lobby *Lobby) RemoveUser(userID uuid.UUID) {
	lobby.Mu.Lock() // Lock is needed here

	conn, connExists := lobby.Connections[userID]
	if !connExists {
		// User might have been removed already or was only invited
		delete(lobby.Users, userID) // Ensure removal from Users map too
		lobby.Mu.Unlock()           // Unlock before returning
		log.Printf("Lobby %s: Attempted to remove user %s who was not connected.", lobby.ID, userID)
		return
	}

	log.Printf("Lobby %s: Removing user %s.", lobby.ID, userID)

	// Close outgoing channel and cancel context for the removed connection
	// Need to check if OutChan is already closed to avoid panic
	// Do this in a separate goroutine to avoid blocking RemoveUser if Write is blocked
	go func(ch chan map[string]interface{}, cancelFunc func()) {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Lobby %s: Recovered from panic closing OutChan for user %s: %v", lobby.ID, userID, r)
			}
		}()
		close(ch) // Close channel to signal writePump to stop
		if cancelFunc != nil {
			cancelFunc() // Cancel context associated with the connection
		}
	}(conn.OutChan, conn.Cancel)

	delete(lobby.Users, userID)
	delete(lobby.Connections, userID)
	delete(lobby.ReadyStates, userID)

	lobbyLeavePayload := lobby.getLobbyLeavePayloadUnsafe(userID)
	allReady := lobby.AreAllReadyUnsafe()
	isEmpty := len(lobby.Connections) == 0
	onEmptyCallback := lobby.OnEmpty // Get callback while holding lock
	shouldCancelCountdown := lobby.CountdownTimer != nil

	// Cancel countdown while holding lock
	if shouldCancelCountdown {
		lobby.CancelCountdownUnsafe()
	}

	// Check if countdown needs to be restarted (outside lock)
	shouldStartCountdown := allReady && lobby.LobbySettings.AutoStart && !lobby.InGame && len(lobby.Connections) >= 2

	lobby.Mu.Unlock() // Unlock BEFORE broadcasting and potential countdown start/callback

	lobby.BroadcastAll(lobbyLeavePayload) // Broadcast leave notification

	// Start countdown outside the main lock if conditions met
	if shouldStartCountdown {
		// Call the exported method which handles locking
		lobby.StartCountdown(10, func(l *Lobby) {
			log.Printf("Lobby %s: Countdown finished after user removal, initiating game start.", l.ID)
			// Placeholder: Need access to GameServer or similar to actually start
			// gameServer.NewCambiaGameFromLobby(context.Background(), l)
		})
	}

	if isEmpty && onEmptyCallback != nil {
		log.Printf("Lobby %s is now empty. Triggering OnEmpty callback.", lobby.ID)
		onEmptyCallback(lobby.ID)
	}
}

// startCountdownUnsafe begins a countdown. Assumes lock is held.
func (lobby *Lobby) StartCountdownUnsafe(seconds int, callback func(*Lobby)) bool {
	if lobby.InGame || lobby.CountdownTimer != nil {
		log.Printf("Lobby %s: Cannot start countdown (InGame: %v, TimerExists: %v)", lobby.ID, lobby.InGame, lobby.CountdownTimer != nil)
		return false
	}
	if len(lobby.Connections) < 2 { // Don't start countdown with fewer than 2 players
		log.Printf("Lobby %s: Cannot start countdown with fewer than 2 players.", lobby.ID)
		return false
	}

	log.Printf("Lobby %s: Starting %d second countdown.", lobby.ID, seconds)
	lobby.BroadcastAllUnsafe(map[string]interface{}{
		"type":    "lobby_countdown_start",
		"seconds": seconds,
	})

	// Declare timer variable before AfterFunc to capture it correctly
	var timer *time.Timer
	timer = time.AfterFunc(time.Duration(seconds)*time.Second, func() {
		log.Printf("Lobby %s: Countdown finished. Executing callback.", lobby.ID)
		lobby.Mu.Lock()
		// Check if this timer is still the current one before executing callback
		if lobby.CountdownTimer == timer {
			lobby.CountdownTimer = nil // Clear timer ref inside lock
			lobby.Mu.Unlock()
			callback(lobby) // Execute callback outside lock
		} else {
			log.Printf("Lobby %s: Stale countdown timer fired. Ignoring.", lobby.ID)
			lobby.Mu.Unlock() // Release lock if stale timer
		}
	})
	lobby.CountdownTimer = timer // Assign timer after creating it
	return true
}

// StartCountdown starts a countdown. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) StartCountdown(seconds int, callback func(*Lobby)) bool {
	return lobby.StartCountdownUnsafe(seconds, callback)
}

// CancelCountdownUnsafe stops any existing countdown. Assumes lock is held.
func (lobby *Lobby) CancelCountdownUnsafe() {
	if lobby.CountdownTimer != nil {
		log.Printf("Lobby %s: Cancelling countdown.", lobby.ID)
		// Stop returns false if the timer has already fired or been stopped
		if lobby.CountdownTimer.Stop() {
			lobby.CountdownTimer = nil // Clear only if stop was successful
			// Broadcast cancellation only if we successfully stopped it
			lobby.BroadcastAllUnsafe(map[string]interface{}{
				"type": "lobby_countdown_cancel",
			})
		} else {
			// Timer already fired or stopped, might already be nil
			lobby.CountdownTimer = nil // Ensure it's nil
		}
	}
}

// CancelCountdown stops any existing countdown. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) CancelCountdown() {
	lobby.CancelCountdownUnsafe()
}

// MarkUserReadyUnsafe sets a user's ready state to true.
// Assumes lock is held by the caller. Returns true if a countdown should be started.
func (lobby *Lobby) MarkUserReadyUnsafe(userID uuid.UUID) bool {
	conn, ok := lobby.Connections[userID]
	if !ok {
		log.Printf("Lobby %s: Cannot mark non-connected user %s as ready (unsafe).", lobby.ID, userID)
		return false
	}

	if lobby.ReadyStates[userID] {
		return false // Already ready, no change
	}

	lobby.ReadyStates[userID] = true
	log.Printf("Lobby %s: User %s marked as READY (unsafe).", lobby.ID, userID)

	// Broadcast readiness change
	readyPayload := map[string]interface{}{
		"type":     "ready_update",
		"user_id":  userID.String(),
		"username": conn.Username,
		"is_ready": true,
	}
	lobby.BroadcastAllUnsafe(readyPayload)

	// Check if a countdown should start now
	allReady := lobby.AreAllReadyUnsafe()
	shouldStartCountdown := allReady && lobby.LobbySettings.AutoStart && !lobby.InGame && len(lobby.Connections) >= 2
	return shouldStartCountdown
}

// MarkUserReady sets ready state. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) MarkUserReady(userID uuid.UUID) bool { // Return bool to match unsafe version
	shouldStart := lobby.MarkUserReadyUnsafe(userID)

	// Note: The countdown starting logic needs to happen outside this function,
	// typically in the caller (handleLobbyMessage) after the lock is released.
	// Returning the boolean allows the caller to decide.
	return shouldStart
}

// MarkUserUnreadyUnsafe sets a user's ready state to false and cancels countdown.
// Assumes lock is held by the caller.
func (lobby *Lobby) MarkUserUnreadyUnsafe(userID uuid.UUID) {
	conn, ok := lobby.Connections[userID]
	if !ok {
		log.Printf("Lobby %s: Cannot mark non-connected user %s as unready (unsafe).", lobby.ID, userID)
		return
	}

	if !lobby.ReadyStates[userID] {
		return // Already not ready, no change
	}

	lobby.ReadyStates[userID] = false
	log.Printf("Lobby %s: User %s marked as UNREADY (unsafe).", lobby.ID, userID)

	// Broadcast readiness change
	readyPayload := map[string]interface{}{
		"type":     "ready_update",
		"user_id":  userID.String(),
		"username": conn.Username,
		"is_ready": false,
	}
	lobby.BroadcastAllUnsafe(readyPayload)

	// Always cancel countdown if someone becomes unready
	lobby.CancelCountdownUnsafe()
}

// MarkUserUnready sets unready state. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) MarkUserUnready(userID uuid.UUID) {
	lobby.MarkUserUnreadyUnsafe(userID)
}

// AreAllReadyUnsafe checks readiness without acquiring lock. Assumes lock is held.
func (lobby *Lobby) AreAllReadyUnsafe() bool {
	if len(lobby.Connections) < 2 {
		return false
	}
	for userID := range lobby.Connections {
		if !lobby.ReadyStates[userID] {
			return false
		}
	}
	return true
}

// AreAllReady checks readiness (public method, acquires lock).
func (lobby *Lobby) AreAllReady() bool {
	lobby.Mu.Lock() // Lock needed here as it reads map state
	defer lobby.Mu.Unlock()
	return lobby.AreAllReadyUnsafe()
}

// BroadcastAllUnsafe sends message without acquiring lock. Assumes lock is held.
// Sends messages directly to OutChan without spawning a new goroutine.
func (lobby *Lobby) BroadcastAllUnsafe(msg map[string]interface{}) {
	// Create a temporary list of connections to iterate over
	// This avoids holding the lock while potentially blocking on channel sends
	// although conn.Write itself is non-blocking.
	connsToSend := make([]*LobbyConnection, 0, len(lobby.Connections))
	for _, conn := range lobby.Connections {
		connsToSend = append(connsToSend, conn)
	}

	// Send messages outside the lock (but Write is non-blocking anyway)
	for _, conn := range connsToSend {
		// Note: conn.Write handles potential blocking internally with select{default}
		conn.Write(msg)
	}
}

// BroadcastAll sends msg to every connected user. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) BroadcastAll(msg map[string]interface{}) {
	lobby.BroadcastAllUnsafe(msg) // Call the unsafe version
}

// GetLobbyStatusPayloadUnsafe gathers current user status. Assumes lock is held.
func (lobby *Lobby) GetLobbyStatusPayloadUnsafe() map[string]interface{} {
	users := []map[string]interface{}{}
	for userID, conn := range lobby.Connections {
		isReady := lobby.ReadyStates[userID] // Directly access from map
		username := conn.Username            // Use stored username

		users = append(users, map[string]interface{}{
			"id":       userID.String(),
			"username": username,
			"is_host":  conn.IsHost,
			"is_ready": isReady,
		})
	}
	return map[string]interface{}{
		"users": users,
	}
}

// getLobbyJoinPayloadUnsafe prepares the join message payload. Assumes lock is held.
func (lobby *Lobby) getLobbyJoinPayloadUnsafe(userID uuid.UUID) map[string]interface{} {
	isHost := false
	username := "Unknown"
	if conn, ok := lobby.Connections[userID]; ok {
		isHost = conn.IsHost
		username = conn.Username
	}

	return map[string]interface{}{
		"type":         "lobby_update",
		"user_join":    userID.String(),
		"username":     username,
		"is_host":      isHost,
		"lobby_status": lobby.GetLobbyStatusPayloadUnsafe(),
	}
}

// BroadcastJoin notifies that a user joined. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) BroadcastJoin(userID uuid.UUID) {
	payload := lobby.getLobbyJoinPayloadUnsafe(userID)
	lobby.BroadcastAll(payload) // BroadcastAll now calls Unsafe internally
}

// BroadcastReadyState notifies that user changed readiness. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) BroadcastReadyState(userID uuid.UUID, ready bool) {
	username := "Unknown"
	if conn, ok := lobby.Connections[userID]; ok {
		username = conn.Username
	}
	payload := map[string]interface{}{
		"type":     "ready_update",
		"user_id":  userID.String(),
		"username": username, // Include username
		"is_ready": ready,
	}
	lobby.BroadcastAll(payload) // BroadcastAll now calls Unsafe internally
}

// getLobbyLeavePayloadUnsafe prepares the leave message payload. Assumes lock is held.
func (lobby *Lobby) getLobbyLeavePayloadUnsafe(userID uuid.UUID) map[string]interface{} {
	username := "Unknown" // Get username before potential removal
	// Check Connections map for username *before* it's removed
	if conn, ok := lobby.Connections[userID]; ok {
		username = conn.Username
	}

	// Get status *after* user is conceptually removed (caller manages removal timing)
	// Note: getLobbyStatusPayloadUnsafe reads the *current* state
	return map[string]interface{}{
		"type":         "lobby_update",
		"user_left":    userID.String(),
		"username":     username,
		"lobby_status": lobby.GetLobbyStatusPayloadUnsafe(),
	}
}

// BroadcastLeave notifies that a user left. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) BroadcastLeave(userID uuid.UUID) {
	payload := lobby.getLobbyLeavePayloadUnsafe(userID)
	lobby.BroadcastAll(payload) // BroadcastAll now calls Unsafe internally
}

// BroadcastChatUnsafe broadcasts a chat message from userID. Assumes lock is held by the caller.
// It uses the username stored in the sender's LobbyConnection.
func (lobby *Lobby) BroadcastChatUnsafe(senderConn *LobbyConnection, msg string) {
	// Username should already be available on senderConn from AddConnection
	username := senderConn.Username
	if username == "" {
		username = "Unknown" // Fallback just in case
	}

	payload := map[string]interface{}{
		"type":     "chat",
		"user_id":  senderConn.UserID.String(),
		"username": username, // Use username from the connection
		"msg":      msg,
		"ts":       time.Now().Unix(),
	}
	lobby.BroadcastAllUnsafe(payload) // Use unsafe as lock is held
}

// BroadcastChat broadcasts a chat message from userID (public method, acquires lock).
// Now calls the unsafe version assuming caller (e.g., handleLobbyMessage) holds the lock.
func (lobby *Lobby) BroadcastChat(userID uuid.UUID, msg string) {
	conn, ok := lobby.Connections[userID]
	if !ok {
		log.Printf("Lobby %s: Cannot broadcast chat for disconnected user %s", lobby.ID, userID)
		return
	}
	lobby.BroadcastChatUnsafe(conn, msg)
}

// getLobbyStatePayloadUnsafe prepares the full state message. Assumes lock is held.
func (lobby *Lobby) getLobbyStatePayloadUnsafe(userID uuid.UUID) map[string]interface{} {
	isHost := false
	if conn, ok := lobby.Connections[userID]; ok {
		isHost = conn.IsHost
	}

	gameIDStr := ""
	if lobby.GameID != uuid.Nil {
		gameIDStr = lobby.GameID.String()
	}

	// Ensure circuit rules are initialized if circuit is enabled but rules are nil somehow
	circuitState := lobby.Circuit
	if circuitState.Enabled && circuitState.Rules == (game.CircuitRules{}) {
		circuitState.Rules = game.CircuitRules{
			TargetScore: 100, WinBonus: -1, FalseCambiaPenalty: 1, FreezeUserOnDisconnect: true,
		}
		log.Printf("Warning: Lobby %s circuit enabled but rules were zero. Applied defaults.", lobby.ID)
	}

	// Prepare the 'settings' field which includes lobbySettings
	settingsPayload := map[string]interface{}{
		"autoStart": lobby.LobbySettings.AutoStart,
		// Add other lobby-specific settings here if they exist
	}

	return map[string]interface{}{
		"type":         "lobby_state",
		"lobby_id":     lobby.ID.String(),
		"host_id":      lobby.HostUserID.String(),
		"your_id":      userID.String(),
		"your_is_host": isHost,
		"lobby_type":   lobby.Type,
		"game_mode":    lobby.GameMode,
		"in_game":      lobby.InGame,
		"game_id":      gameIDStr,
		"house_rules":  lobby.HouseRules,
		"circuit":      circuitState,
		"settings":     settingsPayload, // Use the combined settings payload
		"lobby_status": lobby.GetLobbyStatusPayloadUnsafe(),
	}
}

// SendLobbyState sends the full current lobby state to a specific user. Calls unsafe assuming caller holds lock.
func (lobby *Lobby) SendLobbyState(userID uuid.UUID) {
	// REMOVED: lobby.Mu.Lock()
	conn, ok := lobby.Connections[userID]
	if !ok {
		// REMOVED: lobby.Mu.Unlock()
		log.Printf("Lobby %s: Cannot send lobby state, user %s not connected.", lobby.ID, userID)
		return // User not connected
	}
	payload := lobby.getLobbyStatePayloadUnsafe(userID)
	// REMOVED: lobby.Mu.Unlock()

	conn.Write(payload) // Send after unlock simulation
}

// BroadcastRulesUpdate notifies all users about updated house rules. Calls unsafe assuming caller holds lock.
func (lobby *Lobby) BroadcastRulesUpdate() {
	// REMOVED: lobby.Mu.Lock()
	lobby.BroadcastRulesUpdateUnsafe() // Call unsafe version
	// REMOVED: lobby.Mu.Unlock()
}

// BroadcastRulesUpdateUnsafe notifies all users about updated rules. Assumes lock is held.
func (lobby *Lobby) BroadcastRulesUpdateUnsafe() {
	// Construct nested rules object
	rulesPayload := map[string]interface{}{
		"house_rules": lobby.HouseRules,    // Read current state
		"circuit":     lobby.Circuit,       // Read current state
		"settings":    lobby.LobbySettings, // Read current state
	}
	// Construct the final message with the nested structure
	payload := map[string]interface{}{
		"type":  "lobby_rules_updated",
		"rules": rulesPayload, // Nest house_rules, circuit, settings under "rules"
	}
	log.Printf("[Lobby %s BroadcastRulesUpdateUnsafe] Broadcasting payload: %+v", lobby.ID, payload)
	lobby.BroadcastAllUnsafe(payload) // Use unsafe as lock is held
}

// UpdateUnsafe applies changes from partial settings updates (rules, circuit, lobby).
// Assumes lock is HELD by the caller. It does NOT acquire the lock itself.
// (Content of this function remains the same as it was already correct)
func (lobby *Lobby) UpdateUnsafe(rules map[string]interface{}) error {
	changed := false // Track if any changes were actually made
	log.Printf("Lobby %s: UpdateUnsafe called with payload: %+v", lobby.ID, rules)

	// Use a temporary copy of HouseRules to perform updates and check for changes
	tempHR := lobby.HouseRules
	if hrData, ok := rules["houseRules"].(map[string]interface{}); ok {
		if err := tempHR.Update(hrData); err != nil {
			log.Printf("Lobby %s: Error updating house rules (unsafe): %v", lobby.ID, err)
			return err // Return error if update fails
		}
		if tempHR != lobby.HouseRules { // Compare updated temp with original
			lobby.HouseRules = tempHR // Apply change if different
			changed = true
		}
	}

	// Use a temporary copy for Circuit settings
	tempCircuit := lobby.Circuit
	madeCircuitChange := false
	if cData, ok := rules["circuit"].(map[string]interface{}); ok {
		if enabled, ok := cData["enabled"].(bool); ok {
			if tempCircuit.Enabled != enabled {
				tempCircuit.Enabled = enabled
				madeCircuitChange = true
			}
		}
		if cRulesData, ok := cData["rules"].(map[string]interface{}); ok {
			if ts, ok := cRulesData["targetScore"].(float64); ok {
				if tempCircuit.Rules.TargetScore != int(ts) {
					tempCircuit.Rules.TargetScore = int(ts)
					madeCircuitChange = true
				}
			}
			if wb, ok := cRulesData["winBonus"].(float64); ok {
				if tempCircuit.Rules.WinBonus != int(wb) {
					tempCircuit.Rules.WinBonus = int(wb)
					madeCircuitChange = true
				}
			}
			if fcp, ok := cRulesData["falseCambiaPenalty"].(float64); ok {
				if tempCircuit.Rules.FalseCambiaPenalty != int(fcp) {
					tempCircuit.Rules.FalseCambiaPenalty = int(fcp)
					madeCircuitChange = true
				}
			}
			if fud, ok := cRulesData["freezeUserOnDisconnect"].(bool); ok {
				if tempCircuit.Rules.FreezeUserOnDisconnect != fud {
					tempCircuit.Rules.FreezeUserOnDisconnect = fud
					madeCircuitChange = true
				}
			}
		}
		if madeCircuitChange {
			lobby.Circuit = tempCircuit // Apply change if different
			changed = true
		}
	}

	// Use a temporary copy for LobbySettings
	tempLS := lobby.LobbySettings
	madeLobbySettingChange := false
	if lsData, ok := rules["settings"].(map[string]interface{}); ok {
		if autoStart, ok := lsData["autoStart"].(bool); ok {
			if tempLS.AutoStart != autoStart {
				tempLS.AutoStart = autoStart
				madeLobbySettingChange = true
			}
		}
		// Add checks for other LobbySettings fields here if needed
		if madeLobbySettingChange {
			lobby.LobbySettings = tempLS // Apply change if different
			// log.Printf("Lobby %s: LobbySettings changed (unsafe).", lobby.ID) // Reduce noise
			changed = true
		}
	}

	// Broadcast if changes were made
	if changed {
		log.Printf("Lobby %s: Rules update detected changes (unsafe). Broadcasting...", lobby.ID) // Reduce noise
		lobby.BroadcastRulesUpdateUnsafe()                                                        // Call unsafe broadcast version
	} else {
		log.Printf("Lobby %s: No rules changes detected after update attempt (unsafe).", lobby.ID) // Reduce noise
	}
	return nil // Return nil on success
}

// Update applies changes. Calls the unsafe version assuming caller holds the lock.
func (lobby *Lobby) Update(rules map[string]interface{}) error {
	return lobby.UpdateUnsafe(rules)
}

func (l *Lobby) GetConnectionsUnsafe() []*LobbyConnection {
	// Assumes lock is held
	conns := make([]*LobbyConnection, 0, len(l.Connections))
	for _, conn := range l.Connections {
		conns = append(conns, conn)
	}
	return conns
}
