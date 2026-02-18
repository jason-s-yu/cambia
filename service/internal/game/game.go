// internal/game/game.go
package game

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/cache"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"

	"github.com/coder/websocket"
)

// OnGameEndFunc defines the signature for a callback function executed when a game ends.
// It receives the lobby ID, the primary winner's ID (can be Nil), and the final scores.
type OnGameEndFunc func(lobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int)

// GameEventType represents the type of a game-related event broadcast via WebSockets.
type GameEventType string

// Constants defining the various GameEvent types used for WebSocket communication.
const (
	EventPlayerSnapSuccess      GameEventType = "player_snap_success"
	EventPlayerSnapFail         GameEventType = "player_snap_fail"
	EventPlayerSnapPenalty      GameEventType = "player_snap_penalty"            // Public: Player drew penalty cards.
	EventPrivateSnapPenalty     GameEventType = "private_snap_penalty"           // Private: Details of penalty cards drawn.
	EventGameReshuffleStockpile GameEventType = "game_reshuffle_stockpile"       // Public: Discard pile was reshuffled into stockpile.
	EventPlayerDrawStockpile    GameEventType = "player_draw_stockpile"          // Public: Player drew a card (ID only).
	EventPrivateDrawStockpile   GameEventType = "private_draw_stockpile"         // Private: Details of the card drawn.
	EventPlayerDiscard          GameEventType = "player_discard"                 // Public: Player discarded a card (details revealed).
	EventPlayerReplace          GameEventType = "player_replace"                 // DEPRECATED? Sends EventPlayerDiscard instead.
	EventPlayerSpecialChoice    GameEventType = "player_special_choice"          // Public: Player can now use a special ability.
	EventPlayerSpecialAction    GameEventType = "player_special_action"          // Public: Player used a special ability (obfuscated details).
	EventPrivateSpecialSuccess  GameEventType = "private_special_action_success" // Private: Details of successful special action.
	EventPrivateSpecialFail     GameEventType = "private_special_action_fail"    // Private: Special action attempt failed.
	EventPlayerCambia           GameEventType = "player_cambia"                  // Public: Player called Cambia.
	EventGamePlayerTurn         GameEventType = "game_player_turn"               // Public: Notification of the current player's turn.
	EventPrivateSyncState       GameEventType = "private_sync_state"             // Private: Full game state sync for a player.
	EventPrivateInitialCards    GameEventType = "private_initial_cards"          // Private: Initial two cards revealed during pre-game.
	EventGameEnd                GameEventType = "game_end"                       // Public: Game has ended, includes results.
)

// EventUser identifies a user within a GameEvent payload.
type EventUser struct {
	ID uuid.UUID `json:"id"`
}

// EventCard identifies a card within a GameEvent payload, optionally including details.
type EventCard struct {
	ID    uuid.UUID  `json:"id"`
	Rank  string     `json:"rank,omitempty"`
	Suit  string     `json:"suit,omitempty"`
	Value int        `json:"value,omitempty"`
	Idx   *int       `json:"idx,omitempty"`  // Index in hand, if relevant.
	User  *EventUser `json:"user,omitempty"` // Owner of the card, if relevant (e.g., for swaps).
}

// GameEvent is the standard structure for broadcasting game state changes and actions.
type GameEvent struct {
	Type    GameEventType `json:"type"`
	User    *EventUser    `json:"user,omitempty"`    // The user initiating or targeted by the event.
	Card    *EventCard    `json:"card,omitempty"`    // Primary card involved.
	Card1   *EventCard    `json:"card1,omitempty"`   // First card in a two-card action (e.g., swap).
	Card2   *EventCard    `json:"card2,omitempty"`   // Second card in a two-card action.
	Special string        `json:"special,omitempty"` // Identifier for the specific special action (e.g., "peek_self").

	Payload map[string]interface{} `json:"payload,omitempty"` // Additional arbitrary data.

	State *ObfGameState `json:"state,omitempty"` // Full obfuscated state for sync events.
}

// SpecialActionState holds temporary information about a pending multi-step special action (e.g., King).
type SpecialActionState struct {
	Active        bool         // Is a special action currently pending?
	PlayerID      uuid.UUID    // Which player must act?
	CardRank      string       // Rank of the card that triggered the action ("K", "Q", etc.).
	FirstStepDone bool         // For King: Has the initial peek step completed?
	Card1         *models.Card // For King: First peeked card.
	Card1Owner    uuid.UUID    // For King: Owner of the first peeked card.
	Card2         *models.Card // For King: Second peeked card.
	Card2Owner    uuid.UUID    // For King: Owner of the second peeked card.
}

// CircuitRules defines parameters for tournament-style play across multiple rounds.
type CircuitRules struct {
	TargetScore            int  `json:"targetScore"`            // Score limit to trigger elimination or end.
	WinBonus               int  `json:"winBonus"`               // Bonus (usually negative) applied to winner's score.
	FalseCambiaPenalty     int  `json:"falseCambiaPenalty"`     // Penalty added if Cambia caller doesn't win.
	FreezeUserOnDisconnect bool `json:"freezeUserOnDisconnect"` // Prevent disconnected users from being auto-kicked.
}

// Circuit wraps the overall circuit settings.
type Circuit struct {
	Enabled bool         `json:"enabled"` // Is circuit mode active?
	Mode    string       `json:"mode"`    // Identifier for the circuit mode (e.g., "circuit_4p").
	Rules   CircuitRules `json:"rules"`   // Specific rules for this circuit.
}

// CambiaGame represents the state and logic for a single instance of the Cambia game.
type CambiaGame struct {
	ID      uuid.UUID // Unique identifier for this game instance.
	LobbyID uuid.UUID // ID of the lobby that created this game.

	HouseRules HouseRules // Configurable game rules.
	Circuit    Circuit    // Circuit mode settings.

	Players []*models.Player // List of players in the game.

	// Engine integration — authoritative game state.
	Engine         engine.GameState                    // The authoritative game state.
	CardTracker    CardUUIDTracker                     // UUID tracking for all cards.
	PlayerToEngine map[uuid.UUID]uint8                 // Service player UUID -> engine index.
	EngineToPlayer [engine.MaxPlayers]uuid.UUID        // Engine index -> service player UUID.

	// Buffered discard flow (ability-choice state).
	pendingDiscardAbilityChoice bool
	pendingDiscardCardID        uuid.UUID

	// Turn Management
	TurnID       int           // Increments each turn, useful for state synchronization and checks.
	TurnDuration time.Duration // Configurable duration for each turn timer.
	turnTimer    *time.Timer   // Active timer for the current turn.
	actionIndex  int           // Sequential index for logging actions via historian.

	// Game Lifecycle State
	Started       bool // Has the game started (after pre-game)?
	GameOver      bool // Has the game finished?
	PreGameActive bool // Is the initial pre-game card reveal phase active?

	lastSeen map[uuid.UUID]time.Time // Tracks last activity time for players (potential future use).
	Mu       sync.Mutex              // Mutex protecting concurrent access to game state.

	// Communication Callbacks
	BroadcastFn         func(ev GameEvent)                     // Sends an event to all connected players.
	BroadcastToPlayerFn func(playerID uuid.UUID, ev GameEvent) // Sends an event to a single player.
	OnGameEnd           OnGameEndFunc                          // Callback executed when the game finishes.

	// Special Action State — kept for backward compatibility with ProcessSpecialAction routing.
	SpecialAction SpecialActionState // Holds state for pending multi-step special actions.

	// Snap State
	snapUsedForThisDiscard bool // Tracks if a snap has succeeded for the current discard (used for SnapRace rule).

	// Timers
	preGameTimer *time.Timer // Timer controlling the duration of the pre-game phase.
}

// NewCambiaGame creates a new game instance with default settings.
// Engine is initialized during BeginPreGame/Deal.
func NewCambiaGame() *CambiaGame {
	id, _ := uuid.NewRandom()
	g := &CambiaGame{
		ID:                     id,
		lastSeen:               make(map[uuid.UUID]time.Time),
		TurnDuration:           15 * time.Second, // Default turn duration.
		snapUsedForThisDiscard: false,
		actionIndex:            0,
		TurnID:                 0,
		PlayerToEngine:         make(map[uuid.UUID]uint8),
		// Initialize HouseRules with standard defaults.
		HouseRules: HouseRules{
			AllowDrawFromDiscardPile: false,
			AllowReplaceAbilities:    false,
			SnapRace:                 false,
			ForfeitOnDisconnect:      true,
			PenaltyDrawCount:         2,
			AutoKickTurnCount:        3,
			TurnTimerSec:             15,
		},
		Circuit: Circuit{Enabled: false}, // Circuit mode disabled by default.
	}
	return g
}

// BeginPreGame starts the initial phase where players see their first two cards.
// Deals cards via engine and schedules the transition to the main game start.
func (g *CambiaGame) BeginPreGame() {
	g.Mu.Lock()
	defer g.Mu.Unlock()

	if g.Started || g.GameOver || g.PreGameActive {
		log.Printf("Game %s: BeginPreGame called in invalid state (Started:%v, Over:%v, PreGame:%v).", g.ID, g.Started, g.GameOver, g.PreGameActive)
		return
	}
	g.PreGameActive = true
	g.logAction(uuid.Nil, "game_pregame_start", nil)

	// Apply turn duration from house rules.
	if g.HouseRules.TurnTimerSec > 0 {
		g.TurnDuration = time.Duration(g.HouseRules.TurnTimerSec) * time.Second
	} else {
		g.TurnDuration = 0 // Disable timer if set to 0.
	}

	// Enforce 2-player engine requirement.
	if len(g.Players) != engine.MaxPlayers {
		log.Printf("Game %s: Engine requires exactly %d players, got %d. Cannot start.", g.ID, engine.MaxPlayers, len(g.Players))
		g.PreGameActive = false
		return
	}

	// Build player <-> engine index mapping.
	for i, p := range g.Players {
		g.PlayerToEngine[p.ID] = uint8(i)
		g.EngineToPlayer[i] = p.ID
	}

	// Initialize engine and deal cards.
	seed := uint64(time.Now().UnixNano())
	engineRules := g.mapHouseRulesToEngine()
	g.Engine = engine.NewGame(seed, engineRules)
	g.Engine.Deal()

	// Initialize card UUID tracker from engine state.
	g.initCardTracker()

	// Sync player Hand fields from engine.
	g.syncPlayerHandsFromEngine()

	// Persist initial state for potential replay/audit.
	g.persistInitialGameState()

	// Privately reveal the initial two cards (indices 0, 1) to each player.
	for _, p := range g.Players {
		engineIdx := g.PlayerToEngine[p.ID]
		peekIdxs := g.Engine.Players[engineIdx].InitialPeek

		makeInitialCard := func(slotIdx uint8) *EventCard {
			cardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[slotIdx]
			card := g.Engine.Players[engineIdx].Hand[slotIdx]
			idx := int(slotIdx)
			return &EventCard{
				ID:    cardUUID,
				Rank:  engineRankToString(card.Rank()),
				Suit:  engineSuitToString(card.Suit()),
				Value: int(card.Value()),
				Idx:   &idx,
				User:  &EventUser{ID: p.ID},
			}
		}

		handLen := g.Engine.Players[engineIdx].HandLen
		if handLen >= 2 {
			g.firePrivateInitialCards(p.ID,
				makeInitialCard(peekIdxs[0]),
				makeInitialCard(peekIdxs[1]),
			)
		} else if handLen == 1 {
			g.firePrivateInitialCards(p.ID, makeInitialCard(0), nil)
		} else {
			log.Printf("Warning: Player %s has 0 cards during pregame reveal in game %s.", p.ID, g.ID)
			g.firePrivateInitialCards(p.ID, nil, nil)
		}
	}

	// Schedule the transition to the main game phase.
	preGameDuration := 10 * time.Second // Standard pre-game duration.
	g.preGameTimer = time.AfterFunc(preGameDuration, func() {
		g.StartGame() // Call StartGame after the timer.
	})
	log.Printf("Game %s: Pre-game phase started. Will transition in %s.", g.ID, preGameDuration)
}

// StartGame transitions the game from the pre-game phase to active play.
// It marks the game as started and initiates the first turn.
func (g *CambiaGame) StartGame() {
	g.Mu.Lock()
	defer g.Mu.Unlock()

	// Ensure StartGame is called in the correct state.
	if g.GameOver || g.Started || !g.PreGameActive {
		log.Printf("Game %s: StartGame called in invalid state (GameOver:%v, Started:%v, PreGameActive:%v). Ignoring.", g.ID, g.GameOver, g.Started, g.PreGameActive)
		return
	}

	// Stop the pre-game timer if it's still running.
	if g.preGameTimer != nil {
		g.preGameTimer.Stop()
		g.preGameTimer = nil
	}

	g.PreGameActive = false
	g.Started = true
	log.Printf("Game %s: Started.", g.ID)
	g.logAction(uuid.Nil, "game_start", nil)

	// Start the turn cycle.
	g.scheduleNextTurnTimer()
	g.broadcastPlayerTurn()
}

// Start is deprecated. Use BeginPreGame instead to initiate the game flow.
// Deprecated: Use BeginPreGame() which handles the pre-game reveal and timer.
func (g *CambiaGame) Start() {
	g.BeginPreGame()
}

// firePrivateInitialCards sends the initial card reveal event to a specific player.
// Assumes lock is held by caller.
func (g *CambiaGame) firePrivateInitialCards(playerID uuid.UUID, card1, card2 *EventCard) {
	if g.BroadcastToPlayerFn == nil {
		log.Println("Warning: BroadcastToPlayerFn is nil, cannot send private initial cards.")
		return
	}
	ev := GameEvent{
		Type:  EventPrivateInitialCards,
		Card1: card1, // EventCard struct already contains details.
		Card2: card2,
	}
	g.BroadcastToPlayerFn(playerID, ev)
}

// persistInitialGameState saves the initial deck order and player hands to the database.
// Assumes lock is held by caller.
func (g *CambiaGame) persistInitialGameState() {
	type initialState struct {
		StockpileSize int                       `json:"stockpileSize"`
		Players       map[string][]*models.Card `json:"players"`
	}

	snap := initialState{
		StockpileSize: int(g.Engine.StockLen),
		Players:       make(map[string][]*models.Card),
	}

	for _, p := range g.Players {
		handCopy := make([]*models.Card, len(p.Hand))
		copy(handCopy, p.Hand)
		snap.Players[p.ID.String()] = handCopy
	}

	if database.DB != nil {
		go database.UpsertInitialGameState(g.ID, snap)
	}
	g.logAction(uuid.Nil, "game_initial_state_saved", map[string]interface{}{"stockpileSize": snap.StockpileSize})
}

// AddPlayer adds a player to the game if not started, or marks them as reconnected.
// Assumes lock is held by caller.
func (g *CambiaGame) AddPlayer(p *models.Player) {
	found := false
	for i, pl := range g.Players {
		if pl.ID == p.ID {
			// Player reconnecting.
			g.Players[i].Conn = p.Conn
			g.Players[i].Connected = true
			g.Players[i].User = p.User // Update user info.
			g.lastSeen[p.ID] = time.Now()
			log.Printf("Game %s: Player %s (%s) reconnected.", g.ID, p.ID, p.User.Username)
			found = true
			// Send sync state on reconnect (handled by HandleReconnect).
			break
		}
	}
	if !found {
		// New player joining (only possible before game starts).
		if !g.Started && !g.PreGameActive {
			g.Players = append(g.Players, p)
			g.lastSeen[p.ID] = time.Now()
			log.Printf("Game %s: Player %s (%s) added.", g.ID, p.ID, p.User.Username)
		} else {
			log.Printf("Game %s: Player %s (%s) cannot be added because game has already started.", g.ID, p.ID, p.User.Username)
			// Optionally close connection or send error.
			if p.Conn != nil {
				p.Conn.Close(websocket.StatusPolicyViolation, "Game already in progress.")
			}
			return
		}
	}
	g.logAction(p.ID, "player_add", map[string]interface{}{"reconnect": found, "username": p.User.Username})
}


// scheduleNextTurnTimer delegates to the engine-aware scheduler.
// Assumes lock is held by caller.
func (g *CambiaGame) scheduleNextTurnTimer() {
	g.scheduleNextTurnTimerEngine()
}

// handleTimeout processes the timeout logic for a player.
// Delegates to engine-aware handler.
// Assumes lock is held by caller.
func (g *CambiaGame) handleTimeout(playerID uuid.UUID) {
	g.handleTimeoutEngine(playerID)
}

// broadcastPlayerTurn notifies all players of the current player's turn.
// Assumes lock is held by caller.
func (g *CambiaGame) broadcastPlayerTurn() {
	g.broadcastPlayerTurnEngine()
}

// fireEvent broadcasts an event to all connected players via the BroadcastFn callback.
// Assumes lock is held by caller.
func (g *CambiaGame) fireEvent(ev GameEvent) {
	if g.BroadcastFn != nil {
		g.BroadcastFn(ev) // Execute the callback.
	} else {
		log.Printf("Warning: Game %s: BroadcastFn is nil, cannot broadcast event type %s.", g.ID, ev.Type)
	}
}

// fireEventToPlayer sends an event to a specific player via the BroadcastToPlayerFn callback.
// Checks if the player is connected before sending.
// Assumes lock is held by caller.
func (g *CambiaGame) fireEventToPlayer(playerID uuid.UUID, ev GameEvent) {
	if g.BroadcastToPlayerFn != nil {
		targetPlayer := g.getPlayerByID(playerID)
		if targetPlayer != nil && targetPlayer.Connected {
			g.BroadcastToPlayerFn(playerID, ev) // Execute the callback.
		} else {
			// Log quietly if player not found or disconnected.
			// log.Printf("Debug: Game %s: Target player %s not found or not connected for private event type %s.", g.ID, playerID, ev.Type)
		}
	} else {
		log.Printf("Warning: Game %s: BroadcastToPlayerFn is nil, cannot send private event type %s to player %s.", g.ID, ev.Type, playerID)
	}
}

// advanceTurn is kept for ProcessSpecialAction compat. Delegates to onTurnAdvanced.
// Assumes lock is held by caller.
func (g *CambiaGame) advanceTurn() {
	g.onTurnAdvanced()
}

// HandleDisconnect marks a player as disconnected and handles game state consequences.
// Assumes lock is held by caller.
func (g *CambiaGame) HandleDisconnect(playerID uuid.UUID) {
	log.Printf("Game %s: Handling disconnect for player %s.", g.ID, playerID)
	g.logAction(playerID, "player_disconnect", nil)

	playerIndex := -1
	found := false
	for i := range g.Players {
		if g.Players[i].ID == playerID {
			if !g.Players[i].Connected {
				log.Printf("Game %s: Player %s already marked as disconnected.", g.ID, playerID)
				return // Already handled.
			}
			g.Players[i].Connected = false
			g.Players[i].Conn = nil // Clear WebSocket connection reference.
			found = true
			playerIndex = i
			break
		}
	}
	if !found {
		log.Printf("Game %s: Disconnected player %s not found.", g.ID, playerID)
		return
	}

	shouldAdvanceTurn := false
	shouldEndGame := false

	if g.Started && !g.GameOver {
		// Check if game ends due to forfeit rule.
		if g.HouseRules.ForfeitOnDisconnect {
			log.Printf("Game %s: Player %s disconnected, forfeiting due to house rules.", g.ID, playerID)
			if g.countConnectedPlayers() <= 1 {
				log.Printf("Game %s: Only %d player(s) left connected after forfeit. Ending game.", g.ID, g.countConnectedPlayers())
				shouldEndGame = true
			}
		} else {
			// If no forfeit, check if the current player disconnected.
			currentPlayerUUID := g.currentPlayerID()
			if playerID == currentPlayerUUID {
				log.Printf("Game %s: Current player %s disconnected. Advancing turn.", g.ID, playerID)
				shouldAdvanceTurn = true
			}
		}
	}
	_ = playerIndex // Used for found check above.

	// Broadcast updated state to remaining players *before* ending or advancing.
	g.broadcastSyncStateToAll()

	if shouldEndGame {
		if !g.GameOver {
			g.EndGame() // End the game immediately.
		}
	} else if shouldAdvanceTurn {
		g.advanceTurn() // Advance turn if current player left.
	}
}

// HandleReconnect marks a player as connected and sends them the current game state.
// Assumes lock is held by caller.
func (g *CambiaGame) HandleReconnect(playerID uuid.UUID, conn *websocket.Conn) {
	log.Printf("Game %s: Handling reconnect for player %s.", g.ID, playerID)

	found := false
	for i := range g.Players {
		if g.Players[i].ID == playerID {
			if g.Players[i].Connected {
				log.Printf("Game %s: Player %s reconnected but was already marked connected.", g.ID, playerID)
				// Update connection object anyway.
			}
			g.Players[i].Connected = true
			g.Players[i].Conn = conn
			g.Players[i].User = g.Players[i].User // Assume User struct is still valid.
			g.lastSeen[playerID] = time.Now()
			found = true

			g.logAction(playerID, "player_reconnect", map[string]interface{}{"username": g.Players[i].User.Username})

			// Send sync state immediately to the reconnected player.
			g.sendSyncState(playerID)

			// Broadcast updated state to others.
			g.broadcastSyncStateToAll()

			// If it was this player's turn, reschedule timer.
			if g.Started && !g.GameOver && g.currentPlayerID() == playerID {
				log.Printf("Game %s: Player %s reconnected on their turn. Rescheduling timer.", g.ID, playerID)
				g.scheduleNextTurnTimer()
			}
			break
		}
	}

	if !found {
		log.Printf("Game %s: Reconnecting player %s not found in game.", g.ID, playerID)
		g.logAction(playerID, "player_reconnect_fail", map[string]interface{}{"reason": "player not found"})
		if conn != nil {
			// Close connection if player isn't actually part of this game.
			conn.Close(websocket.StatusPolicyViolation, "Game not found or you were removed.")
		}
	}
}

// sendSyncState sends the current obfuscated game state to a single player.
// Assumes lock is held by caller.
func (g *CambiaGame) sendSyncState(playerID uuid.UUID) {
	if g.BroadcastToPlayerFn == nil {
		log.Println("Warning: BroadcastToPlayerFn is nil, cannot send sync state.")
		return
	}
	// Generate state specifically for this player.
	state := g.GetCurrentObfuscatedGameState(playerID)
	ev := GameEvent{
		Type:  EventPrivateSyncState,
		State: &state, // Embed the state object.
	}
	g.fireEventToPlayer(playerID, ev) // Uses internal check for connection status.
	// log.Printf("Game %s: Sent sync state to player %s.", g.ID, playerID) // Reduce noise.
}

// broadcastSyncStateToAll sends the obfuscated game state to all currently connected players.
// Assumes lock is held by caller.
func (g *CambiaGame) broadcastSyncStateToAll() {
	if g.BroadcastToPlayerFn == nil {
		log.Println("Warning: BroadcastToPlayerFn is nil, cannot broadcast sync state to all.")
		return
	}
	connectedCount := 0
	for _, p := range g.Players {
		if p.Connected {
			g.sendSyncState(p.ID) // Generate and send state for each connected player.
			connectedCount++
		}
	}
	// log.Printf("Game %s: Broadcasted sync state to %d connected players.", g.ID, connectedCount) // Reduce noise.
}

// countConnectedPlayers returns the number of players currently marked as connected.
// Assumes lock is held by caller.
func (g *CambiaGame) countConnectedPlayers() int {
	count := 0
	for _, p := range g.Players {
		if p.Connected {
			count++
		}
	}
	return count
}


// HandlePlayerAction routes incoming player actions (draw, discard, replace, snap, cambia).
// Validates turn, state, and payload before executing the corresponding handler.
// Assumes lock is held by the caller.
func (g *CambiaGame) HandlePlayerAction(playerID uuid.UUID, action models.GameAction) {
	// --- Basic State Checks ---
	if g.GameOver {
		log.Printf("Game %s: Action %s from %s ignored (game over).", g.ID, action.ActionType, playerID)
		return
	}
	if !g.Started && !g.PreGameActive {
		log.Printf("Game %s: Action %s from %s ignored (game not started).", g.ID, action.ActionType, playerID)
		return
	}
	if g.PreGameActive {
		log.Printf("Game %s: Action %s from %s ignored (pre-game active).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "Cannot perform actions during pre-game reveal."}})
		return
	}

	// --- Player Validation ---
	player := g.getPlayerByID(playerID)
	if player == nil || !player.Connected {
		log.Printf("Game %s: Action %s from non-existent/disconnected player %s ignored.", g.ID, action.ActionType, playerID)
		return
	}

	engineIdx, inEngineMapping := g.PlayerToEngine[playerID]
	if !inEngineMapping {
		log.Printf("Game %s: Action %s from %s ignored (not in engine mapping).", g.ID, action.ActionType, playerID)
		return
	}

	// --- Turn and State Validation ---
	actingPlayer := g.Engine.ActingPlayer()
	isCurrentPlayer := (actingPlayer == engineIdx)

	// Allow snap anytime.
	if action.ActionType != "action_snap" && !isCurrentPlayer {
		log.Printf("Game %s: Action %s from %s ignored (not their turn).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "It's not your turn."}})
		return
	}
	// Check if blocked by pending special action requiring resolution.
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID && action.ActionType != "action_special" && action.ActionType != "action_snap" {
		log.Printf("Game %s: Action %s from %s ignored (special action pending).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You must resolve the special card action first (use action_special with 'skip' or required payload)."}})
		return
	}
	// Prevent drawing twice (check engine state).
	isDrawAction := action.ActionType == "action_draw_stockpile" || action.ActionType == "action_draw_discardpile"
	if g.Engine.Pending.Type == engine.PendingDiscard && g.Engine.Pending.PlayerID == engineIdx && isDrawAction {
		log.Printf("Game %s: Action %s from %s ignored (already drawn).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You have already drawn a card this turn."}})
		return
	}
	// Prevent discard/replace without drawing first.
	isDiscardReplace := action.ActionType == "action_discard" || action.ActionType == "action_replace"
	hasPendingDraw := g.Engine.Pending.Type == engine.PendingDiscard && g.Engine.Pending.PlayerID == engineIdx
	if !hasPendingDraw && isDiscardReplace && !g.pendingDiscardAbilityChoice {
		log.Printf("Game %s: Action %s from %s ignored (must draw first).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You must draw a card first."}})
		return
	}

	// Update last seen time.
	g.lastSeen[playerID] = time.Now()

	// --- Route Action ---
	switch action.ActionType {
	case "action_snap":
		g.handleSnapViaEngine(playerID, engineIdx, action.Payload)
	case "action_draw_stockpile":
		g.applyEngineAction(engine.ActionDrawStockpile, playerID)
	case "action_draw_discardpile":
		g.applyEngineAction(engine.ActionDrawDiscard, playerID)
	case "action_discard":
		g.handleDiscardViaEngine(playerID, engineIdx, action.Payload)
	case "action_replace":
		g.handleReplaceViaEngine(playerID, engineIdx, action.Payload)
	case "action_cambia":
		g.applyEngineAction(engine.ActionCallCambia, playerID)
	// Note: "action_special" is handled directly by ProcessSpecialAction.
	default:
		log.Printf("Game %s: Unknown action type '%s' received from player %s.", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "Unknown action type."}})
	}
}


// rankToSpecial maps card ranks to their corresponding special action identifier string.
// Returns an empty string if the rank has no special ability.
func rankToSpecial(rank string) string {
	switch rank {
	case "7", "8":
		return "peek_self"
	case "9", "T": // T represents Ten.
		return "peek_other"
	case "J", "Q": // Jack, Queen.
		return "swap_blind"
	case "K": // King.
		return "swap_peek" // Initial step for King.
	default:
		return ""
	}
}

// EndGame finalizes the game, computes scores, determines winners, applies bonuses/penalties,
// broadcasts results, and triggers the OnGameEnd callback.
// Assumes lock is held by caller.
func (g *CambiaGame) EndGame() {
	if g.GameOver {
		log.Printf("Game %s: EndGame called, but game is already over.", g.ID)
		return
	}
	g.GameOver = true
	g.Started = false // Mark as inactive.
	log.Printf("Game %s: Ending game. Computing final scores...", g.ID)

	// Stop timers.
	if g.turnTimer != nil {
		g.turnTimer.Stop()
		g.turnTimer = nil
	}
	if g.preGameTimer != nil {
		g.preGameTimer.Stop()
		g.preGameTimer = nil
	}

	// --- Scoring and Winner Determination ---
	// Compute scores from engine hand state.
	finalScores := g.computeScoresFromEngine()
	callerID := g.cambiaCallerID()
	winners, penaltyApplies := g.findWinnersWithCambiaLogicEngine(finalScores, callerID)
	adjustedScores := make(map[uuid.UUID]int)
	for id, score := range finalScores {
		adjustedScores[id] = score
	}

	// Apply Cambia caller penalty if needed.
	if penaltyApplies && callerID != uuid.Nil {
		if _, ok := adjustedScores[callerID]; ok {
			penaltyValue := 1 // Default penalty.
			if g.Circuit.Enabled {
				penaltyValue = g.Circuit.Rules.FalseCambiaPenalty
			}
			adjustedScores[callerID] += penaltyValue
			log.Printf("Game %s: Applying +%d penalty to Cambia caller %s for not winning.", g.ID, penaltyValue, callerID)
		} else {
			log.Printf("Warning: Game %s: Cambia caller %s not found in final scores for penalty.", g.ID, callerID)
		}
	}

	// Apply circuit win bonus if needed.
	winBonusApplied := false
	if g.Circuit.Enabled && g.Circuit.Rules.WinBonus != 0 && len(winners) > 0 {
		winBonus := g.Circuit.Rules.WinBonus
		for _, winnerID := range winners {
			if _, ok := adjustedScores[winnerID]; ok {
				adjustedScores[winnerID] += winBonus
				log.Printf("Game %s: Applying %d win bonus to winner %s.", g.ID, winBonus, winnerID)
				winBonusApplied = true
			}
		}
	}
	// --- End Scoring ---

	g.logAction(uuid.Nil, string(EventGameEnd), map[string]interface{}{
		"scores":         adjustedScores,
		"winners":        winners,
		"caller":         callerID,
		"penaltyApplied": penaltyApplies,
		"winBonus":       g.Circuit.Rules.WinBonus,
	})
	g.persistFinalGameState(adjustedScores, winners)

	// Determine primary winner for event payload.
	var firstWinner uuid.UUID
	if len(winners) > 0 {
		firstWinner = winners[0]
	}

	// Broadcast game end event.
	resultsPayload := map[string]interface{}{
		"scores":          map[string]int{},
		"winner":          firstWinner.String(),
		"caller":          callerID.String(),
		"penaltyApplied":  penaltyApplies,
		"winBonusApplied": winBonusApplied,
	}
	for pid, score := range adjustedScores {
		resultsPayload["scores"].(map[string]int)[pid.String()] = score
	}
	g.fireEvent(GameEvent{
		Type:    EventGameEnd,
		Payload: resultsPayload,
	})

	// Trigger external callback (e.g., update lobby).
	if g.OnGameEnd != nil {
		g.OnGameEnd(g.LobbyID, firstWinner, adjustedScores)
	}

	log.Printf("Game %s: Ended. Winner(s): %v. Final Scores (Adj): %v", g.ID, winners, adjustedScores)
}

// computeScoresFromEngine calculates scores from engine hand state.
// Assumes lock is held by caller.
func (g *CambiaGame) computeScoresFromEngine() map[uuid.UUID]int {
	scores := make(map[uuid.UUID]int)
	for i := uint8(0); i < engine.MaxPlayers; i++ {
		playerUUID := g.EngineToPlayer[i]
		if playerUUID == uuid.Nil {
			continue
		}
		player := g.getPlayerByID(playerUUID)
		if player == nil {
			continue
		}
		// Score only connected players or if disconnect doesn't forfeit.
		if player.Connected || !g.HouseRules.ForfeitOnDisconnect {
			score := 0
			for j := uint8(0); j < g.Engine.Players[i].HandLen; j++ {
				score += int(g.Engine.Players[i].Hand[j].Value())
			}
			scores[playerUUID] = score
		} else {
			log.Printf("Game %s: Player %s score omitted (disconnected/forfeited).", g.ID, playerUUID)
		}
	}
	return scores
}

// findWinnersWithCambiaLogicEngine wraps findWinnersWithCambiaLogic using engine-derived callerID.
func (g *CambiaGame) findWinnersWithCambiaLogicEngine(scores map[uuid.UUID]int, callerID uuid.UUID) ([]uuid.UUID, bool) {
	// Temporarily set CambiaCalled/CambiaCallerID for the existing logic.
	// We use engine state directly, so adapt the call to findWinnersWithCambiaLogic.
	savedCambiaCalled := g.Engine.IsCambiaCalled()
	if len(scores) == 0 {
		return []uuid.UUID{}, false
	}

	lowestScore := -1
	first := true
	for _, score := range scores {
		if first || score < lowestScore {
			lowestScore = score
			first = false
		}
	}

	potentialWinners := []uuid.UUID{}
	for playerID, score := range scores {
		if score == lowestScore {
			potentialWinners = append(potentialWinners, playerID)
		}
	}

	if savedCambiaCalled && callerID != uuid.Nil {
		callerIsPotentialWinner := false
		for _, winnerID := range potentialWinners {
			if winnerID == callerID {
				callerIsPotentialWinner = true
				break
			}
		}
		if callerIsPotentialWinner {
			log.Printf("Game %s: Cambia caller %s won or tied for lowest score (%d).", g.ID, callerID, lowestScore)
			return []uuid.UUID{callerID}, false
		}
		log.Printf("Game %s: Cambia caller %s did not win (Lowest score: %d). Penalty applies.", g.ID, callerID, lowestScore)
		if len(potentialWinners) == 1 {
			return potentialWinners, true
		}
		return []uuid.UUID{}, true
	}
	log.Printf("Game %s: Cambia not called. Lowest score: %d. Winners: %v", g.ID, lowestScore, potentialWinners)
	return potentialWinners, false
}


// persistFinalGameState saves final hands and winners to the database.
// Assumes lock is held by caller.
func (g *CambiaGame) persistFinalGameState(finalScores map[uuid.UUID]int, winners []uuid.UUID) {
	type finalHandCard struct {
		Rank string `json:"rank"`
		Suit string `json:"suit"`
		Val  int    `json:"value"`
	}
	type finalPlayerState struct {
		Hand  []finalHandCard `json:"hand"`
		Score int             `json:"score"`
	}

	snapshot := map[string]interface{}{
		"players": map[string]finalPlayerState{},
		"winners": winners,
	}

	playerStates := snapshot["players"].(map[string]finalPlayerState)
	for i := uint8(0); i < engine.MaxPlayers; i++ {
		playerUUID := g.EngineToPlayer[i]
		if playerUUID == uuid.Nil {
			continue
		}
		score, scoreOk := finalScores[playerUUID]
		if !scoreOk {
			score = -999
		}
		handLen := g.Engine.Players[i].HandLen
		state := finalPlayerState{
			Hand:  make([]finalHandCard, handLen),
			Score: score,
		}
		for j := uint8(0); j < handLen; j++ {
			card := g.Engine.Players[i].Hand[j]
			state.Hand[j] = finalHandCard{
				Rank: engineRankToString(card.Rank()),
				Suit: engineSuitToString(card.Suit()),
				Val:  int(card.Value()),
			}
		}
		playerStates[playerUUID.String()] = state
	}

	if database.DB != nil {
		go database.StoreFinalGameStateInDB(context.Background(), g.ID, snapshot)
	}
}

// removeCardFromPlayerHand removes a specific card instance from a player's hand.
// Returns true if found and removed, false otherwise, and the index where it was found.
// Assumes lock is held by caller.
func (g *CambiaGame) removeCardFromPlayerHand(playerID, cardID uuid.UUID) (bool, int) {
	player := g.getPlayerByID(playerID)
	if player == nil {
		return false, -1
	}
	removedIndex := -1
	for i, c := range player.Hand {
		if c.ID == cardID {
			removedIndex = i
			break
		}
	}
	if removedIndex != -1 {
		player.Hand = append(player.Hand[:removedIndex], player.Hand[removedIndex+1:]...)
		return true, removedIndex
	}
	return false, -1
}

// getPlayerByID finds a player struct by ID within the game's Players slice.
// Returns the player pointer or nil if not found.
// Assumes lock is held by caller.
func (g *CambiaGame) getPlayerByID(playerID uuid.UUID) *models.Player {
	for _, p := range g.Players {
		if p.ID == playerID {
			return p
		}
	}
	return nil
}

// logAction sends game action details to the historian service via Redis queue.
// Increments the internal action index for ordering.
// Assumes lock is held by caller.
func (g *CambiaGame) logAction(actorID uuid.UUID, actionType string, payload map[string]interface{}) {
	g.actionIndex++
	if payload == nil {
		payload = make(map[string]interface{}) // Ensure payload is not nil.
	}
	record := cache.GameActionRecord{
		GameID:        g.ID,
		ActionIndex:   g.actionIndex,
		ActorUserID:   actorID, // Can be Nil for game events.
		ActionType:    actionType,
		ActionPayload: payload,
		Timestamp:     time.Now().UnixMilli(),
	}

	// Asynchronously publish to Redis.
	go func(rec cache.GameActionRecord) {
		// Short timeout for the Redis operation.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		// Check if Redis client is initialized.
		if cache.Rdb == nil {
			// log.Printf("Debug: Redis client (Rdb) is nil. Cannot log action %d for game %s.", rec.ActionIndex, g.ID) // Reduce noise
			return
		}
		if err := cache.PublishGameAction(ctx, rec); err != nil {
			log.Printf("Error: Game %s: Failed publishing action %d ('%s') to Redis: %v", g.ID, rec.ActionIndex, rec.ActionType, err)
		}
	}(record)
}

// ResetTurnTimer restarts the turn timer for the current player.
// Exported for use by special action logic.
// Assumes lock is held by caller.
func (g *CambiaGame) ResetTurnTimer() {
	g.scheduleNextTurnTimer() // Use the internal scheduler.
}

// FireEventPrivateSpecialActionFail helper to send a private failure event for special actions.
// Assumes lock is held by caller.
func (g *CambiaGame) FireEventPrivateSpecialActionFail(userID uuid.UUID, reason string, special string, card1, card2 *EventCard) {
	ev := GameEvent{
		Type:    EventPrivateSpecialFail,
		Special: special,
		Payload: map[string]interface{}{"message": reason},
		Card1:   card1, // Include card info if relevant to the failure.
		Card2:   card2,
	}
	g.fireEventToPlayer(userID, ev)
	g.logAction(userID, string(EventPrivateSpecialFail), map[string]interface{}{"reason": reason, "special": special})
}

// FailSpecialAction clears the pending special action state and advances the turn, sending a failure event.
// Assumes lock is held by caller.
func (g *CambiaGame) FailSpecialAction(userID uuid.UUID, reason string) {
	if !g.SpecialAction.Active || g.SpecialAction.PlayerID != userID {
		log.Printf("Warning: Game %s: FailSpecialAction called for %s but state mismatch (Active:%v, Player:%s). Sending fail event anyway.", g.ID, userID, g.SpecialAction.Active, g.SpecialAction.PlayerID)
		// Send fail event even if state is inconsistent.
		g.FireEventPrivateSpecialActionFail(userID, reason, g.SpecialAction.CardRank, nil, nil)
		// Don't advance turn if state was already inconsistent.
		return
	}
	specialType := rankToSpecial(g.SpecialAction.CardRank) // Get type before clearing.
	log.Printf("Game %s: Failing special action %s for player %s. Reason: %s", g.ID, specialType, userID, reason)

	// Fire the fail event before clearing state.
	g.FireEventPrivateSpecialActionFail(userID, reason, specialType, nil, nil)

	g.SpecialAction = SpecialActionState{} // Clear state.
	g.advanceTurn()                        // Advance turn after failure.
}

// FireEventPrivateSuccess helper to send a private success event for special actions.
// Assumes lock is held by caller.
func (g *CambiaGame) FireEventPrivateSuccess(userID uuid.UUID, special string, c1Ev, c2Ev *EventCard) {
	ev := GameEvent{
		Type:    EventPrivateSpecialSuccess,
		Special: special,
		Card1:   c1Ev, // Include revealed card details.
		Card2:   c2Ev,
	}
	g.fireEventToPlayer(userID, ev)
	// Logging is typically handled within the specific do* action function.
}

// FireEventPlayerSpecialAction helper to broadcast public info about a special action.
// Assumes lock is held by caller.
func (g *CambiaGame) FireEventPlayerSpecialAction(userID uuid.UUID, special string, c1Ev, c2Ev *EventCard) {
	ev := GameEvent{
		Type:    EventPlayerSpecialAction,
		User:    &EventUser{ID: userID},
		Special: special,
		Card1:   c1Ev, // Include obfuscated card details (ID, index, owner).
		Card2:   c2Ev,
	}
	g.fireEvent(ev)
	// Logging is typically handled within the specific do* action function.
}
