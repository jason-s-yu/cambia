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
// It receives the lobby ID, the primary winner's ID (can be Nil), the final (display-adjusted)
// scores, and each participant's username keyed by player ID (cambia-877). usernames is supplied
// here, rather than left for the callback to look up, because endGame calls this synchronously
// while g.mu is still held (see EndGame): any lookup that re-acquires it - including
// GetCurrentObfuscatedGameState - would self-deadlock the caller's own goroutine.
//
// rawScores and cambiaCallerID (cambia-1008/1009) are the pre-WinBonus/FalseCambiaPenalty hand
// scores and the real Cambia caller (uuid.Nil if none): a circuit's cumulative totals must be
// built from these, never from scores, since WinBonus/FalseCambiaPenalty are single-game display
// adjustments with no rulebook standing in circuit scoring (RULES.md has no such knobs).
type OnGameEndFunc func(lobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int, usernames map[uuid.UUID]string, rawScores map[uuid.UUID]int, cambiaCallerID uuid.UUID)

// GameEventType represents the type of a game-related event broadcast via WebSockets.
type GameEventType string

// Constants defining the various GameEvent types used for WebSocket communication.
const (
	EventPlayerSnapSuccess GameEventType = "player_snap_success"
	EventPlayerSnapFail    GameEventType = "player_snap_fail"
	// A successful opponent snap owes the victim a card back (RULES.md 5). The first event opens
	// that obligation - user is the snapper, card.user the victim and card.idx the slot the snapped
	// card left - and the second reports the card that settled it (cambia-936). Both are public:
	// the whole table watched the snap, and the moved card travels face down, so only its id rides
	// the second event.
	EventPlayerSnapMoveRequired GameEventType = "player_snap_move_required"
	EventPlayerSnapMove         GameEventType = "player_snap_move"
	EventPlayerSnapPenalty      GameEventType = "player_snap_penalty"            // Public: Player drew penalty cards.
	EventPrivateSnapPenalty     GameEventType = "private_snap_penalty"           // Private: Details of penalty cards drawn.
	EventGameReshuffleStockpile GameEventType = "game_reshuffle_stockpile"       // Public: Discard pile was reshuffled into stockpile.
	EventPlayerDrawStockpile    GameEventType = "player_draw_stockpile"          // Public: Player drew a card (ID only).
	EventPrivateDrawStockpile   GameEventType = "private_draw_stockpile"         // Private: Details of the card drawn.
	EventPlayerDiscard          GameEventType = "player_discard"                 // Public: Player discarded a card (details revealed).
	EventPlayerSpecialChoice    GameEventType = "player_special_choice"          // Public: Player can now use a special ability.
	EventPlayerSpecialAction    GameEventType = "player_special_action"          // Public: Player used a special ability (obfuscated details).
	EventPrivateSpecialSuccess  GameEventType = "private_special_action_success" // Private: Details of successful special action.
	EventPrivateSpecialFail     GameEventType = "private_special_action_fail"    // Private: Special action attempt failed.
	EventPlayerCambia           GameEventType = "player_cambia"                  // Public: Player called Cambia.
	EventGamePlayerTurn         GameEventType = "game_player_turn"               // Public: Notification of the current player's turn.
	EventPrivateSyncState       GameEventType = "private_sync_state"             // Private: Full game state sync for a player.
	EventPrivateInitialCards    GameEventType = "private_initial_cards"          // Private: Pregame peek cards revealed to their owner.
	EventGameEnd                GameEventType = "game_end"                       // Public: Game has ended, includes results.

	// Disconnect grace (cambia-955). A dropped socket no longer forfeits on the spot: the seat is
	// held for HouseRules.DisconnectGraceSec and these three events report where a player stands
	// in that window. player_reconnecting carries the deadline so clients can count it down;
	// player_forfeited fires when the window closes and is what tells a client to stop showing a
	// seat as merely away.
	EventPlayerReconnecting GameEventType = "player_reconnecting" // Public: Player's socket dropped; their seat is held until the grace expires.
	EventPlayerReconnected  GameEventType = "player_reconnected"  // Public: Player returned inside the grace window.
	EventPlayerForfeited    GameEventType = "player_forfeited"    // Public: Grace window expired; the player forfeited.
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
	// Cards carries a variable-length card list for events whose card count is a house rule
	// rather than a fixed property of the action. private_initial_cards is the only such event:
	// the pregame peek reveals initialViewCount slots, which ranges up to cardsPerPlayer, so the
	// fixed Card1/Card2 pair could not express a three-card peek (cambia-817).
	Cards []*EventCard `json:"cards,omitempty"`

	Payload map[string]interface{} `json:"payload,omitempty"` // Additional arbitrary data.

	State *ObfGameState `json:"state,omitempty"` // Full obfuscated state for sync events.
}

// SpecialActionState holds temporary information about a pending multi-step special action (e.g., King).
type SpecialActionState struct {
	Active   bool      // Is a special action currently pending?
	PlayerID uuid.UUID // Which player must act?
	CardRank string    // Rank of the card that triggered the action ("K", "Q", etc.).
	// Mandatory marks an ability the engine armed by itself, which today means the one
	// engine replace() arms when AllowReplaceAbilities is on. It cannot be declined: the engine
	// folds that decision into the discard action (DiscardNoAbility vs DiscardWithAbility) and its
	// legal set for an armed ability offers targets and nothing else (engine/legal.go
	// legalAbilitySelect), so once it is armed the pending state can only be resolved. Clearing
	// this prompt without resolving it leaves the engine holding the ability forever, refusing
	// every later action at the table (cambia-1125).
	Mandatory     bool
	FirstStepDone bool         // For King: Has the initial peek step completed?
	Card1         *models.Card // For King: First peeked card.
	Card1Owner    uuid.UUID    // For King: Owner of the first peeked card.
	Card2         *models.Card // For King: Second peeked card.
	Card2Owner    uuid.UUID    // For King: Owner of the second peeked card.
}

// MustResolve reports whether the pending ability has to be played out rather than declined: the
// engine armed it (Mandatory) and it is not the King's second step, which the engine does model as
// declinable (ActionKingSwapNo). It is the one predicate three callers had a copy of - the skip
// handler's refusal, the turn timeout's auto-resolve, and the Mandatory flag sync_state projects to
// the client - so a change to when an ability can be declined has one place to land (cambia-1125).
// Only meaningful while Active; a cleared state answers false.
func (s SpecialActionState) MustResolve() bool {
	return s.Mandatory && !(s.CardRank == "K" && s.FirstStepDone)
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
	// mu serializes every access to this game's mutable state. The engine handle, UUID trackers,
	// timers and player slice are not safe for concurrent use, so all public entry points (player
	// actions, lifecycle transitions, and the turn/pre-game/circuit-grace timer callbacks) hold mu.
	// This makes the adapter the single serialization layer over the non-thread-safe engine: the
	// hub Run() goroutine and the game's own timer goroutines never touch state concurrently
	// (cambia-465). Internal helpers annotated "assumes lock is held by caller" run under mu.
	mu sync.Mutex

	ID      uuid.UUID // Unique identifier for this game instance.
	LobbyID uuid.UUID // ID of the lobby that created this game.

	// Lobby provenance, needed to persist a backing lobbies row (games.lobby_id FK) at game
	// start since the ephemeral in-memory Lobby itself is never written to Postgres. Set by
	// the constructor that builds this CambiaGame from a lobby.Lobby (cambia-450).
	HostUserID uuid.UUID // Lobby host, satisfies lobbies.host_user_id (NOT NULL FK to users).
	LobbyType  string    // Lobby type ("private"/"public"/"matchmaking"), satisfies lobbies.type.
	Rated      bool      // Whether this game's results should feed the rating system.

	HouseRules HouseRules // Configurable game rules.
	Circuit    Circuit    // Circuit mode settings.

	Players []*models.Player // List of players in the game.

	// Engine integration - authoritative game state.
	Engine         engine.GameState             // The authoritative game state.
	CardTracker    CardUUIDTracker              // UUID tracking for all cards.
	PlayerToEngine map[uuid.UUID]uint8          // Service player UUID -> engine index.
	EngineToPlayer [engine.MaxPlayers]uuid.UUID // Engine index -> service player UUID.

	// Buffered discard flow (ability-choice state). See buffered_discard.go for the window these
	// four fields describe: the card is announced to every client immediately and only reaches the
	// engine once the ability is resolved or skipped.
	pendingDiscardAbilityChoice bool
	pendingDiscardCardID        uuid.UUID
	// pendingDiscardWindowSnaps counts the cards snapped onto the discard pile while the announced
	// discard was still buffered, so the buffered card can be sunk back underneath them when it
	// applies (cambia-1033).
	pendingDiscardWindowSnaps int
	// applyingAnnouncedDiscard is set only while that buffered action is being applied. It marks the
	// one discard whose card the table already knows about, which is what keeps the apply from
	// announcing it a second time and what triggers the rotation above.
	applyingAnnouncedDiscard bool

	// Turn Management
	TurnID       int           // Increments each turn, useful for state synchronization and checks.
	TurnDuration time.Duration // Configurable duration for each turn timer.
	turnTimer    *time.Timer   // Active timer for the current turn.
	actionIndex  int           // Sequential index for logging actions via historian.

	// PreGameDuration is how long BeginPreGame holds the initial card-reveal phase before
	// StartGame flips Started true. Set from GameServer.PreGameDuration at creation
	// (CreateGameInstance); defaults to 10s here so a game built directly (tests, or any
	// caller that skips CreateGameInstance) still gets a sane duration.
	PreGameDuration time.Duration

	// TurnDeadline is the absolute server-clock time at which the current turn's timer fires.
	// Zero value means no timer is active for the current turn (TurnDuration <= 0, game not
	// started, or game over). Set by scheduleNextTurnTimerEngine, read by broadcastPlayerTurnEngine
	// and getCurrentObfuscatedGameState to let clients derive a live countdown.
	TurnDeadline time.Time

	// Game Lifecycle State
	Started       bool // Has the game started (after pre-game)?
	GameOver      bool // Has the game finished?
	PreGameActive bool // Is the initial pre-game card reveal phase active?

	lastSeen map[uuid.UUID]time.Time // Tracks last activity time for players (potential future use).

	// Communication - all events go through the hub's Emitter.
	Emitter   Emitter       // Set by the hub after game creation; nil-safe (events dropped if unset).
	OnGameEnd OnGameEndFunc // Callback executed when the game finishes.

	// PersistWG, when set (test-only; production games leave it nil, copied from
	// GameServer.PersistWG at creation - see CreateGameInstance), is Add(1)'d once per
	// background DB-write goroutine this game launches - persistInitialGameState's upsert and
	// persistFinalGameState's two - and Done()'d when that goroutine returns. It lets a caller
	// Wait() for every write to finish before proceeding, instead of leaving them to outlive it
	// (cambia-908: an unwaited goroutine from one test's game end raced a later test's
	// database.ConnectDB reassigning the shared pool, under -race).
	//
	// Every Add runs under g.mu, inside BeginPreGame (initial state) or endGame (final state),
	// so a caller holding a happens-before edge to the write it cares about - an observed
	// GameOver, a received game_results broadcast, or its own EndGame call - is guaranteed the
	// matching Add already ran. That edge is required: sync.WaitGroup forbids an Add that takes
	// the counter up from zero from racing a concurrent Wait, and one GameServer's WaitGroup is
	// shared by every game it creates, so a Wait issued for one game and an Add issued by
	// another game's start are ordered only by whatever the caller establishes itself (cambia-942
	// F3; see handlers.awaitGameEndPersistence for the test-side statement of that rule).
	PersistWG *sync.WaitGroup

	// Special Action State - kept for backward compatibility with ProcessSpecialAction routing.
	SpecialAction SpecialActionState // Holds state for pending multi-step special actions.

	// Snap State
	snapUsedForThisDiscard bool // Tracks if a snap has succeeded for the current discard (used for SnapRace rule).

	// snapFills holds the fill each successful opponent snap owes, keyed by the snapper who owes
	// it (RULES.md 5, cambia-936). Keyed rather than singular because two players can each snap a
	// different opponent card off one discard when snapRace is off, and each owes their own card
	// back. snapFillGen stamps every obligation so a timer armed for a settled one cannot act on
	// its successor. See snap_fill.go.
	snapFills   map[uuid.UUID]*snapFillState
	snapFillGen uint64

	// Timers
	preGameTimer *time.Timer // Timer controlling the duration of the pre-game phase.

	// Circuit-mode disconnect handling (T5)
	circuitGraceTimers  map[uuid.UUID]*time.Timer // 60s grace timers per disconnected player
	circuitAIControlled map[uuid.UUID]bool        // Players currently under AI control

	// DisconnectGrace is how long a dropped player keeps their seat before ForfeitOnDisconnect
	// takes it (RULES.md T5, MATCHMAKING.md 8). Derived in BeginPreGame from
	// HouseRules.DisconnectGraceSec, the same way TurnDuration is derived from TurnTimerSec, so a
	// test can shorten it to milliseconds after BeginPreGame. Zero forfeits on the drop itself,
	// which is what the rule did before cambia-955.
	DisconnectGrace time.Duration

	// disconnectGraceTimers holds the armed grace timer per dropped player and graceDeadlines the
	// wall-clock time it fires at, published in sync_state so a client that resyncs mid-window
	// (including the dropped player's own reload) can render the same countdown as everyone else.
	// Both are cleared by a reconnect and by the forfeit itself.
	disconnectGraceTimers map[uuid.UUID]*time.Timer
	graceDeadlines        map[uuid.UUID]time.Time

	// graceGen stamps each arming of a player's window. Timer.Stop cannot un-fire a callback
	// already waiting on g.mu, so a player who drops, returns and drops again inside one grace
	// would otherwise have the first window's stale callback forfeit them against the second
	// window's clock. The callback compares the generation it was armed with and returns.
	graceGen map[uuid.UUID]uint64

	// forfeited records who actually forfeited, which is no longer the same question as who is
	// disconnected: inside the grace window a player is gone but still in the game, and their hand
	// is still scored if the table finishes without them (MATCHMAKING.md 8, "score counts
	// normally"). Scoring reads this rather than Player.Connected (cambia-955).
	forfeited map[uuid.UUID]bool
}

// NewCambiaGame creates a new game instance with default settings.
// Engine is initialized during BeginPreGame/Deal.
func NewCambiaGame() *CambiaGame {
	id, _ := uuid.NewRandom()
	g := &CambiaGame{
		ID:                     id,
		lastSeen:               make(map[uuid.UUID]time.Time),
		TurnDuration:           15 * time.Second, // Default turn duration.
		PreGameDuration:        10 * time.Second, // Default pre-game reveal duration.
		snapUsedForThisDiscard: false,
		actionIndex:            0,
		TurnID:                 0,
		PlayerToEngine:         make(map[uuid.UUID]uint8),
		// Initialize HouseRules with standard defaults.
		HouseRules:          DefaultHouseRules(),
		Circuit:             Circuit{Enabled: false}, // Circuit mode disabled by default.
		circuitGraceTimers:  make(map[uuid.UUID]*time.Timer),
		circuitAIControlled: make(map[uuid.UUID]bool),

		disconnectGraceTimers: make(map[uuid.UUID]*time.Timer),
		graceDeadlines:        make(map[uuid.UUID]time.Time),
		graceGen:              make(map[uuid.UUID]uint64),
		forfeited:             make(map[uuid.UUID]bool),
		snapFills:             make(map[uuid.UUID]*snapFillState),
	}
	return g
}

// BeginPreGame starts the initial phase where players see their first two cards.
// Deals cards via engine and schedules the transition to the main game start.
func (g *CambiaGame) BeginPreGame() {
	g.mu.Lock()
	defer g.mu.Unlock()
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

	// Same derivation for the reconnect grace: 0 means a drop forfeits immediately.
	if g.HouseRules.DisconnectGraceSec > 0 {
		g.DisconnectGrace = time.Duration(g.HouseRules.DisconnectGraceSec) * time.Second
	} else {
		g.DisconnectGrace = 0
	}

	// Validate player count: 2 to engine.MaxPlayers.
	if len(g.Players) < 2 || len(g.Players) > engine.MaxPlayers {
		log.Printf("Game %s: Requires 2-%d players, got %d. Cannot start.", g.ID, engine.MaxPlayers, len(g.Players))
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

	// Emit the initial sync_state to every player before the reveal below. The web client drops
	// all game events while its gameState is nil and only populates it from private_sync_state, so
	// without a sync-first the private_initial_cards reveal (and every later game_player_turn) is
	// discarded and the table never renders (cambia-501). Sending sync first means the subsequent
	// private_initial_cards / game_player_turn events land on already-populated client state.
	g.broadcastSyncStateToAll()

	// Privately reveal each player's pregame peek. This event is the ONLY carrier of those faces:
	// sync_state hides every own card unconditionally (cambia-1094), so a client that misses this
	// frame plays the round blind rather than picking the peek up from the next snapshot.
	for _, p := range g.Players {
		engineIdx := g.PlayerToEngine[p.ID]
		if g.Engine.Players[engineIdx].HandLen == 0 {
			log.Printf("Warning: Player %s has 0 cards during pregame reveal in game %s.", p.ID, g.ID)
		}
		g.firePrivateInitialCards(p.ID, g.pregameInitialCards(p.ID))
	}

	// Schedule the transition to the main game phase. PreGameDuration is configurable (see
	// GameServer.PreGameDuration / CAMBIA_PREGAME_DURATION); fall back to the 10s default if a
	// caller left it unset entirely (zero value).
	preGameDuration := g.PreGameDuration
	if preGameDuration <= 0 {
		preGameDuration = 10 * time.Second
	}
	g.preGameTimer = time.AfterFunc(preGameDuration, func() {
		g.StartGame() // Call StartGame after the timer.
	})
	log.Printf("Game %s: Pre-game phase started. Will transition in %s.", g.ID, preGameDuration)
}

// StartGame transitions the game from the pre-game phase to active play.
// It marks the game as started and initiates the first turn.
func (g *CambiaGame) StartGame() {
	g.mu.Lock()
	defer g.mu.Unlock()
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
	// Re-sync every player now that the game is live: Started flips false->true here and a turn
	// deadline was just armed. game_player_turn alone carries neither the started transition nor a
	// fresh snapshot, so without this the client keeps gameState.started=false and would reject its
	// own turn. Emitting sync before broadcastPlayerTurn keeps the turn event landing on up-to-date
	// state (cambia-501).
	g.broadcastSyncStateToAll()
	g.broadcastPlayerTurn()
}

// Start is deprecated. Use BeginPreGame instead to initiate the game flow.
// Deprecated: Use BeginPreGame() which handles the pre-game reveal and timer.
func (g *CambiaGame) Start() {
	g.BeginPreGame()
}

// pregameInitialCards builds playerID's pregame peek reveal: one EventCard per slot the engine
// peeked, in InitialPeek order. Deal() decides how many slots that is from the initialViewCount
// house rule (clamped to the hand size) and records them in InitialPeek/InitialPeekCount, so the
// count is read from the engine rather than assumed: a lobby that peeks one card, or none at all,
// must not have slot 0 revealed anyway. Returns nil for a player with no engine seat or an empty
// hand. Assumes lock is held by caller.
//
// Called once from BeginPreGame and again from HandleReconnect for a player who returns inside
// the pregame window. Rebuilding rather than caching keeps the reveal honest about the hand as it
// stands; nothing moves cards during the reveal, so the two calls name the same slots.
func (g *CambiaGame) pregameInitialCards(playerID uuid.UUID) []*EventCard {
	engineIdx, ok := g.PlayerToEngine[playerID]
	if !ok {
		return nil
	}
	handLen := g.Engine.Players[engineIdx].HandLen
	if handLen == 0 {
		return nil
	}
	peekCount := g.Engine.Players[engineIdx].InitialPeekCount
	if peekCount > handLen {
		peekCount = handLen
	}
	peekIdxs := g.Engine.Players[engineIdx].InitialPeek

	cards := make([]*EventCard, 0, peekCount)
	for i := uint8(0); i < peekCount; i++ {
		slotIdx := peekIdxs[i]
		cardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[slotIdx]
		card := g.Engine.Players[engineIdx].Hand[slotIdx]
		// Record the pregame peek: this card is now legitimately seen by its owner. Idempotent, so
		// a reconnect re-fire changes nothing.
		g.markCardSeen(engineIdx, cardUUID)
		idx := int(slotIdx)
		cards = append(cards, &EventCard{
			ID:    cardUUID,
			Rank:  engineRankToString(card.Rank()),
			Suit:  engineSuitToString(card.Suit()),
			Value: int(card.Value()),
			Idx:   &idx,
			User:  &EventUser{ID: playerID},
		})
	}
	return cards
}

// firePrivateInitialCards sends the pregame peek reveal to a specific player. cards holds one
// entry per peeked slot, in the engine's InitialPeek order, and is empty when the initialViewCount
// house rule peeks nothing. The whole list ships under Cards: the peek size is a house rule that
// the engine allows up to cardsPerPlayer, so the event cannot assume two (cambia-817).
func (g *CambiaGame) firePrivateInitialCards(playerID uuid.UUID, cards []*EventCard) {
	if g.Emitter == nil {
		return
	}
	ev := GameEvent{
		Type:  EventPrivateInitialCards,
		Cards: cards,
	}
	g.Emitter.EmitTo(playerID, string(EventPrivateInitialCards), ev)
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
		gameID, lobbyID, hostUserID, lobbyType, rated := g.ID, g.LobbyID, g.HostUserID, g.LobbyType, g.Rated
		wg := g.PersistWG
		if wg != nil {
			wg.Add(1)
		}
		go func() {
			if wg != nil {
				defer wg.Done()
			}
			if err := database.UpsertInitialGameState(context.Background(), gameID, lobbyID, hostUserID, lobbyType, rated, snap); err != nil {
				log.Printf("Game %s: failed to persist initial game state: %v", gameID, err)
			}
		}()
	}
	g.logAction(uuid.Nil, "game_initial_state_saved", map[string]interface{}{"stockpileSize": snap.StockpileSize})
}

// AddPlayer adds a player to the game if not started, or marks them as reconnected.
// Public entry point: acquires mu.
func (g *CambiaGame) AddPlayer(p *models.Player) {
	g.mu.Lock()
	defer g.mu.Unlock()
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

// fireEvent broadcasts an event to all connected players via the Emitter.
func (g *CambiaGame) fireEvent(ev GameEvent) {
	if g.Emitter == nil {
		log.Printf("Warning: Game %s: Emitter is nil, cannot broadcast event type %s.", g.ID, ev.Type)
		return
	}
	g.Emitter.Emit(string(ev.Type), ev)
}

// fireEventToPlayer sends an event to a specific player via the Emitter.
// Checks if the player is connected before sending.
func (g *CambiaGame) fireEventToPlayer(playerID uuid.UUID, ev GameEvent) {
	if g.Emitter == nil {
		log.Printf("Warning: Game %s: Emitter is nil, cannot send private event type %s to player %s.", g.ID, ev.Type, playerID)
		return
	}
	targetPlayer := g.getPlayerByID(playerID)
	if targetPlayer != nil && targetPlayer.Connected {
		g.Emitter.EmitTo(playerID, string(ev.Type), ev)
	}
}

// advanceTurn is kept for ProcessSpecialAction compat. Delegates to onTurnAdvanced.
// Assumes lock is held by caller.
func (g *CambiaGame) advanceTurn() {
	g.onTurnAdvanced()
}

// HasPlayer reports whether playerID holds a seat in this game.
// Public entry point: acquires mu.
//
// The hub gates its connect/disconnect wiring on this so those paths only ever reach actual
// participants: a lobby member who arrived after the deal has no seat, and routing their socket
// through HandleDisconnect would write a player_disconnect into the game's action log for
// somebody who was never in the game.
func (g *CambiaGame) HasPlayer(playerID uuid.UUID) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.getPlayerByID(playerID) != nil
}

// HandleDisconnect marks a player as disconnected and handles game state consequences.
// Public entry point: acquires mu.
func (g *CambiaGame) HandleDisconnect(playerID uuid.UUID) {
	g.mu.Lock()
	defer g.mu.Unlock()
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

	// Circuit mode: 60s grace timer instead of immediate forfeit.
	if g.Circuit.Enabled {
		if t, ok := g.circuitGraceTimers[playerID]; ok {
			t.Stop()
		}
		g.circuitGraceTimers[playerID] = time.AfterFunc(60*time.Second, func() {
			g.mu.Lock()
			defer g.mu.Unlock()
			g.circuitAIControlled[playerID] = true
			log.Printf("Game %s: Player %s grace period expired, AI taking over", g.ID, playerID)
		})
		g.broadcastSyncStateToAll()
		return
	}

	shouldAdvanceTurn := false
	shouldForfeit := false

	// PreGameActive counts as being in the game (cambia-955 F1): Started only flips true when
	// StartGame runs at the end of the initial card reveal, but the hub is in PhaseInGame from
	// the deal onwards and routes drops here throughout. Gating on Started alone left that whole
	// window exempt from the forfeit rule, and with scoring keyed on the forfeit set rather than
	// Player.Connected, somebody who abandoned during the peek was then scored as if present.
	if (g.Started || g.PreGameActive) && !g.GameOver {
		// Check if game ends due to forfeit rule.
		if g.HouseRules.ForfeitOnDisconnect {
			// The grace window (cambia-955): the seat is held, the table keeps playing, and the
			// forfeit only lands if nobody comes back. A reload takes a second or two, and before
			// this the drop forfeited inside a few hundred milliseconds, which in a two-player
			// game ended it outright before the returning client could resume (cambia-783).
			if g.DisconnectGrace > 0 {
				g.armDisconnectGrace(playerID)
			} else {
				log.Printf("Game %s: Player %s disconnected, forfeiting due to house rules.", g.ID, playerID)
				shouldForfeit = true
			}
		} else if g.Started {
			// If no forfeit, check if the current player disconnected. Only meaningful once turns
			// exist: during the pregame reveal nobody is on turn yet, so there is nothing to
			// advance past.
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

	if shouldForfeit {
		g.forfeitPlayer(playerID)
	} else if shouldAdvanceTurn {
		g.advanceTurn() // Advance turn if current player left.
	}
}

// armDisconnectGrace starts (or restarts) playerID's reconnect window and tells everyone the seat
// is being held. Assumes lock is held by caller.
//
// The turn timer is deliberately left running for a player inside their window: RULES.md T5 and
// MATCHMAKING.md 8 have the table play on ("AI plays defensively", score counts normally on
// return), and pausing it instead would let anyone freeze a game for the length of the grace by
// pulling their network out. The existing turn timeout - draw and discard the drawn card, hand
// untouched - is that defensive play, so scheduleNextTurnTimerEngine now arms for a disconnected
// player under the forfeit rule rather than declining to.
func (g *CambiaGame) armDisconnectGrace(playerID uuid.UUID) {
	if t, ok := g.disconnectGraceTimers[playerID]; ok {
		t.Stop()
	}
	grace := g.DisconnectGrace
	deadline := time.Now().Add(grace)
	g.graceDeadlines[playerID] = deadline
	g.graceGen[playerID]++
	gen := g.graceGen[playerID]
	log.Printf("Game %s: Player %s disconnected; holding their seat for %s before the forfeit.", g.ID, playerID, grace)
	g.logAction(playerID, string(EventPlayerReconnecting), map[string]interface{}{"graceSeconds": g.HouseRules.DisconnectGraceSec})

	g.disconnectGraceTimers[playerID] = time.AfterFunc(grace, func() {
		g.mu.Lock()
		defer g.mu.Unlock()
		// Stop() cannot un-fire a callback already in flight, so a reconnect that beat this to the
		// lock is caught by the map entry it deleted, and a second drop that re-armed the window
		// in the meantime is caught by the generation.
		if _, still := g.disconnectGraceTimers[playerID]; !still || g.graceGen[playerID] != gen {
			return
		}
		delete(g.disconnectGraceTimers, playerID)
		delete(g.graceDeadlines, playerID)
		// A window armed during the initial card reveal has to be able to land whether or not
		// StartGame has run by the time it expires, so the pregame phase is accepted here too
		// (cambia-955 F1). A grace shorter than the reveal expires with Started still false.
		if g.GameOver || (!g.Started && !g.PreGameActive) {
			return
		}
		log.Printf("Game %s: Player %s did not return within %s. Forfeiting.", g.ID, playerID, grace)
		g.forfeitPlayer(playerID)
	})

	g.fireEvent(GameEvent{
		Type: EventPlayerReconnecting,
		User: &EventUser{ID: playerID},
		Payload: map[string]interface{}{
			"graceSeconds": g.HouseRules.DisconnectGraceSec,
			"deadline":     deadline.UnixMilli(),
			"serverNow":    time.Now().UnixMilli(),
		},
	})
}

// cancelDisconnectGrace closes an open reconnect window without forfeiting. Reports whether one
// was actually open, which is what separates a return inside the window from a socket that comes
// back after the seat was already given up. Assumes lock is held by caller.
func (g *CambiaGame) cancelDisconnectGrace(playerID uuid.UUID) bool {
	t, ok := g.disconnectGraceTimers[playerID]
	if !ok {
		return false
	}
	t.Stop()
	delete(g.disconnectGraceTimers, playerID)
	delete(g.graceDeadlines, playerID)
	return true
}

// forfeitPlayer records the forfeit and runs the consequences: the player drops out of scoring
// (computeScoresFromEngine reads g.forfeited, not Player.Connected) and the game ends if it has
// nobody left to play it. Assumes lock is held by caller.
func (g *CambiaGame) forfeitPlayer(playerID uuid.UUID) {
	if g.GameOver || g.forfeited[playerID] {
		return
	}
	g.forfeited[playerID] = true
	g.logAction(playerID, string(EventPlayerForfeited), nil)
	g.fireEvent(GameEvent{Type: EventPlayerForfeited, User: &EventUser{ID: playerID}})

	if g.countConnectedPlayers() <= 1 {
		log.Printf("Game %s: Only %d player(s) left connected after forfeit. Ending game.", g.ID, g.countConnectedPlayers())
		g.endGame()
		return
	}
	// More than one player is still here, so the table plays on without the forfeited seat; its
	// turns are auto-played by the turn timer exactly as they were during the grace.
	g.broadcastSyncStateToAll()
}

// IsGameOver reports whether this game has finished. Public entry point: acquires mu.
func (g *CambiaGame) IsGameOver() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.GameOver
}

// IsForfeited reports whether playerID has forfeited (their grace window closed, or the rule
// forfeits on the drop itself). Public entry point: acquires mu.
func (g *CambiaGame) IsForfeited(playerID uuid.UUID) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.forfeited[playerID]
}

// ReconnectDeadline returns the wall-clock time playerID's grace window closes, and whether one
// is open at all. Public entry point: acquires mu.
func (g *CambiaGame) ReconnectDeadline(playerID uuid.UUID) (time.Time, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	d, ok := g.graceDeadlines[playerID]
	return d, ok
}

// HandleReconnect marks a player as connected and sends them the current game state.
// Public entry point: acquires mu.
func (g *CambiaGame) HandleReconnect(playerID uuid.UUID, conn *websocket.Conn) {
	g.mu.Lock()
	defer g.mu.Unlock()
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
			// The existing User struct is kept as-is on reconnect.
			g.lastSeen[playerID] = time.Now()
			found = true

			g.logAction(playerID, "player_reconnect", map[string]interface{}{"username": g.Players[i].User.Username})

			// Close the reconnect window before anything is sent: the returning player is back
			// inside it, so the forfeit that was pending on it must not land (cambia-955).
			inGrace := g.cancelDisconnectGrace(playerID)

			// A player who returns to a game that is still running takes their seat back even if
			// the window had already closed: RULES.md T5 forfeits a round to somebody who misses
			// it, not to somebody who was away for part of it, and this is also the pre-cambia-955
			// behaviour, where scoring simply asked who was connected at the final whistle. Once
			// the game is over there is nothing to return to: endGame has already scored it.
			// The initial card reveal is "still running" for this purpose too, so a forfeit that
			// landed during the peek is lifted the same way (cambia-955 F1).
			resumed := false
			if (g.Started || g.PreGameActive) && !g.GameOver && g.forfeited[playerID] {
				delete(g.forfeited, playerID)
				resumed = true
				log.Printf("Game %s: Player %s returned to a game still in progress; their forfeit is lifted.", g.ID, playerID)
			}

			// Send sync state immediately to the reconnected player.
			g.sendSyncState(playerID)

			// A player who returns while the initial reveal is still running gets it again. The
			// snapshot just sent carries no own faces at all (cambia-1094), and private_initial_cards
			// is the only frame that ever carries them, so without this a reload during the peek
			// costs the player the whole reveal for the rest of the round.
			if g.PreGameActive && !g.GameOver {
				g.firePrivateInitialCards(playerID, g.pregameInitialCards(playerID))
			}

			// Broadcast updated state to others.
			g.broadcastSyncStateToAll()

			if inGrace || resumed {
				g.fireEvent(GameEvent{Type: EventPlayerReconnected, User: &EventUser{ID: playerID}})
			}

			// Circuit mode: cancel grace timer, restore player control.
			if g.Circuit.Enabled {
				if t, ok := g.circuitGraceTimers[playerID]; ok {
					t.Stop()
					delete(g.circuitGraceTimers, playerID)
				}
				delete(g.circuitAIControlled, playerID)
			}

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
	if g.Emitter == nil {
		return
	}
	state := g.getCurrentObfuscatedGameState(playerID)
	ev := GameEvent{
		Type:  EventPrivateSyncState,
		State: &state,
	}
	g.fireEventToPlayer(playerID, ev)
}

// broadcastSyncStateToAll sends the obfuscated game state to all currently connected players.
func (g *CambiaGame) broadcastSyncStateToAll() {
	for _, p := range g.Players {
		if p.Connected {
			g.sendSyncState(p.ID)
		}
	}
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
// Public entry point: acquires mu.
func (g *CambiaGame) HandlePlayerAction(playerID uuid.UUID, action models.GameAction) {
	g.mu.Lock()
	defer g.mu.Unlock()
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

	// Allow snap anytime, and the fill a snap owes with it: both are answers to another player's
	// discard, so neither waits for the sender's turn (RULES.md 5).
	if action.ActionType != "action_snap" && action.ActionType != "action_snap_move" && !isCurrentPlayer {
		log.Printf("Game %s: Action %s from %s ignored (not their turn).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "It's not your turn."}})
		return
	}
	// A snapper who took an opponent's card owes them one back before doing anything else
	// (RULES.md 5, cambia-936). Another snap is refused with the rest: it would open a second
	// obligation against a hand that has not paid the first.
	if action.ActionType != "action_snap_move" && g.owesSnapFill(playerID) && !g.dropUnpayableSnapFill(playerID, engineIdx) {
		log.Printf("Game %s: Action %s from %s ignored (snap fill pending).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You must move one of your cards into the slot you snapped first."}})
		return
	}
	// Check if blocked by pending special action requiring resolution.
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID && action.ActionType != "action_special" && action.ActionType != "action_snap" && action.ActionType != "action_snap_move" {
		log.Printf("Game %s: Action %s from %s ignored (special action pending).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You must resolve the special card action first (use action_special with 'skip' or required payload)."}})
		return
	}
	// Whether this player has already drawn and is pending a discard/replace decision.
	hasPendingDraw := g.Engine.Pending.Type == engine.PendingDiscard && g.Engine.Pending.PlayerID == engineIdx

	// Prevent drawing twice (check engine state).
	isDrawAction := action.ActionType == "action_draw_stockpile" || action.ActionType == "action_draw_discardpile"
	if hasPendingDraw && isDrawAction {
		log.Printf("Game %s: Action %s from %s ignored (already drawn).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "You have already drawn a card this turn."}})
		return
	}
	// Cambia must be called instead of drawing, at turn start (RULES.md). A player who has
	// already drawn this turn is still the acting player, so the generic not-your-turn check
	// above doesn't catch this; without this check the engine's raw pending-action error (which
	// leaks an internal state code) would surface instead (cambia-507).
	if hasPendingDraw && action.ActionType == "action_cambia" {
		log.Printf("Game %s: Action %s from %s ignored (already drawn, cannot call Cambia).", g.ID, action.ActionType, playerID)
		g.fireEventToPlayer(playerID, GameEvent{Type: EventPrivateSpecialFail, Payload: map[string]interface{}{"message": "Cambia must be called before drawing."}})
		return
	}
	// Prevent discard/replace without drawing first.
	isDiscardReplace := action.ActionType == "action_discard" || action.ActionType == "action_replace"
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
	case "action_snap_move":
		g.handleSnapMoveViaEngine(playerID, engineIdx, action.Payload)
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

// EndGame is the public entry point for ending a game from outside an already-locked path.
// It acquires mu and delegates to endGame. Internal callers that already hold mu (the action
// apply path and disconnect handling) call endGame directly.
func (g *CambiaGame) EndGame() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.endGame()
}

// endGame finalizes the game, computes scores, determines winners, applies bonuses/penalties,
// broadcasts results, and triggers the OnGameEnd callback.
// Assumes lock is held by caller.
func (g *CambiaGame) endGame() {
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
	// A reconnect window must not outlive the game it belonged to: the callback re-checks
	// GameOver, but leaving the fire pending serves nothing (cambia-955).
	for id, t := range g.disconnectGraceTimers {
		t.Stop()
		delete(g.disconnectGraceTimers, id)
		delete(g.graceDeadlines, id)
	}
	// An unpaid snap fill dies with the game it was owed in: the hands are about to be scored as
	// they stand, so moving a card between them now would rewrite a result already being read.
	g.cancelSnapFills()

	// --- Scoring and Winner Determination ---
	// Compute scores from engine hand state.
	finalScores := g.computeScoresFromEngine()
	callerID := g.cambiaCallerID()
	winners, penaltyApplies := g.findWinnersWithCambiaLogicEngine(finalScores, callerID)
	adjustedScores := make(map[uuid.UUID]int)
	for id, score := range finalScores {
		adjustedScores[id] = score
	}

	// Apply Cambia caller penalty and circuit win bonus below. Both are single-game display
	// adjustments only: they land in adjustedScores (the game_results payload and the value
	// handed to OnGameEnd's scores param) but never reach circuit cumulative scoring, which reads
	// finalScores (raw, pre-adjustment) via OnGameEnd's rawScores param instead (cambia-1009).
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
		usernames := make(map[uuid.UUID]string, len(g.Players))
		for _, p := range g.Players {
			if p.User != nil {
				usernames[p.ID] = p.User.Username
			}
		}
		g.OnGameEnd(g.LobbyID, firstWinner, adjustedScores, usernames, finalScores, callerID)
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
		// Score everyone who did not forfeit. Being disconnected is not the same as having
		// forfeited since cambia-955: a player inside their reconnect window still holds their
		// seat, and a table that finishes without them scores their hand normally
		// (MATCHMAKING.md 8). With the grace at 0 the two coincide, which is the pre-cambia-955
		// behaviour.
		if !g.forfeited[playerUUID] {
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

// ratePerGame reports whether this game's own result feeds the rating system as it ends.
//
// A rated game does, except when it is one round of a circuit: RULES.md T6 and MATCHMAKING.md
// 6.2/6.3 rate a multi-round format strictly once, at its conclusion, from the final cumulative
// scores. The round is still recorded and displayed like any other game; only its rating update
// is withheld, deferring to the single one the circuit's completion triggers
// (handlers.finalizeCircuitRatings -> database.RecordCircuitRatings).
func (g *CambiaGame) ratePerGame() bool {
	return g.Rated && !g.Circuit.Enabled
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
		gameID := g.ID
		wg := g.PersistWG
		if wg != nil {
			wg.Add(2)
		}
		go func() {
			if wg != nil {
				defer wg.Done()
			}
			if err := database.StoreFinalGameStateInDB(context.Background(), gameID, snapshot); err != nil {
				log.Printf("Game %s: failed to persist final game state: %v", gameID, err)
			}
		}()

		// Record game_results and (if rated) apply rating deltas. This is the only production
		// caller of RecordGameAndResults (cambia-450); it previously had none, so ratings never
		// updated after real games despite the pool-aware persistence logic existing.
		players := g.Players
		rated := g.ratePerGame()
		if g.Rated && !rated {
			// Logged separately so a circuit round is distinguishable from a genuinely unrated
			// game downstream, where RecordGameAndResults reports both as "unrated".
			log.Printf("Game %s: circuit round; per-game rating deferred to the circuit's conclusion.", gameID)
		}
		go func() {
			if wg != nil {
				defer wg.Done()
			}
			if err := database.RecordGameAndResults(context.Background(), gameID, players, finalScores, winners, rated); err != nil {
				log.Printf("Game %s: failed to record game results/ratings: %v", gameID, err)
			}
		}()
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

// ResetTurnTimer restarts the turn timer for the current player. Public entry point: acquires mu.
// Internal callers that already hold mu call scheduleNextTurnTimer directly.
func (g *CambiaGame) ResetTurnTimer() {
	g.mu.Lock()
	defer g.mu.Unlock()
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

// RejectSpecialAction rejects an invalid special-action payload without advancing the turn or
// clearing any pending state. It fires the private fail event, then leaves both the active
// SpecialAction and any buffered discard (pendingDiscardAbilityChoice) untouched so the player
// can retry with a valid target, or bail cleanly with "skip" (which resolves the buffered discard
// as a no-ability discard and advances). This keeps engine and service state in lockstep: because
// a rejected payload never resolves the engine's pending discard, the turn is never advanced while
// the engine's ActingPlayer has not moved. That structural invariant makes the King-peek wedge
// (cambia-509) impossible: previously the buffered discard stayed set while advanceTurn rebroadcast
// a turn to the same player, permanently divergent. The turn timer armed when the ability began
// keeps running, so a client that never sends a valid payload still times out cleanly via
// handleTimeoutEngine rather than stalling the game. Assumes lock is held by caller.
func (g *CambiaGame) RejectSpecialAction(userID uuid.UUID, reason string) {
	specialType := rankToSpecial(g.SpecialAction.CardRank)
	log.Printf("Game %s: Rejecting special action %s for player %s (state retained for retry). Reason: %s", g.ID, specialType, userID, reason)
	g.FireEventPrivateSpecialActionFail(userID, reason, specialType, nil, nil)
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

// CircuitAIPlay performs a minimal defensive action for an AI-controlled disconnected player.
// Draw from stockpile and immediately discard (no abilities, no swaps). Public entry point: acquires mu.
func (g *CambiaGame) CircuitAIPlay(playerID uuid.UUID) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.circuitAIControlled[playerID] {
		return
	}
	// The actual implementation depends on how turns are processed.
	// For now, just advance the turn with a no-op.
	// In practice, this would call the engine's draw+discard actions.
	log.Printf("Game %s: AI defensive play for disconnected player %s", g.ID, playerID)
}

// IsCircuitAIControlled returns whether a player is currently under AI control due to disconnect.
// Public entry point: acquires mu.
func (g *CambiaGame) IsCircuitAIControlled(playerID uuid.UUID) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.circuitAIControlled[playerID]
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
