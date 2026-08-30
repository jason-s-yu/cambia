// internal/handlers/api_server.go
package handlers

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// defaultCountdownDuration is the delay between a lobby reaching countdown and the
// game being created. Tests may lower GameServer.CountdownDuration for speed.
const defaultCountdownDuration = 3 * time.Second

// defaultPostGameDuration is how long a lobby shows end-of-game results before its hub returns
// to the open phase for the next game. Tests may lower GameServer.PostGameDuration for speed.
const defaultPostGameDuration = 10 * time.Second

// defaultLobbyIdleTTL bounds a lobby's idle window (cambia-836): the reap deadline where
// defaultLobbyEmptyIdleTTL does not apply, the ceiling on it where it does. A lobby whose game is
// still running is not reaped on it at all (cambia-884). Tests may lower GameServer.LobbyIdleTTL
// for speed.
const defaultLobbyIdleTTL = 45 * time.Minute

// defaultLobbyEmptyIdleTTL is the same window for a lobby with no game in progress (cambia-884).
// Tests may lower GameServer.LobbyEmptyIdleTTL for speed.
const defaultLobbyEmptyIdleTTL = 5 * time.Minute

// defaultPreGameDuration is how long a fresh game holds the initial card-reveal phase before
// StartGame flips it live. Tests may lower GameServer.PreGameDuration for speed.
const defaultPreGameDuration = 10 * time.Second

// GameServer manages the central stores for active lobbies and games.
type GameServer struct {
	Mutex        sync.Mutex
	LobbyStore   *lobby.LobbyStore
	GameStore    *game.GameStore
	CircuitStore *game.CircuitStore
	HubStore     *hub.HubStore
	Matchmaker   *matchmaking.Matchmaker

	// CountdownDuration is copied onto each hub at creation so the lobby -> game
	// countdown length is configurable (production default; shortened in tests).
	CountdownDuration time.Duration

	// PostGameDuration is copied onto each hub at creation so the results-screen interval
	// before the lobby reopens is configurable (production default; shortened in tests).
	PostGameDuration time.Duration

	// LobbyIdleTTL is copied onto each hub at creation and bounds its idle window (cambia-836):
	// the reap deadline while LobbyEmptyIdleTTL is unset or longer, the ceiling on it otherwise,
	// zero to disable reaping. Overridable per deployment via CAMBIA_LOBBY_IDLE_TTL; shortened in
	// tests.
	LobbyIdleTTL time.Duration

	// LobbyEmptyIdleTTL is copied onto each hub alongside LobbyIdleTTL and is the window that
	// reaps while no game is in progress: an abandoned pre-game or post-game lobby is released in
	// minutes rather than sitting out the long TTL (cambia-884). While a game is in progress it is
	// how often the hub reconsiders, so a game that ends after its table dropped is followed by
	// the reap within one of these. Overridable per deployment via CAMBIA_LOBBY_EMPTY_IDLE_TTL;
	// shortened in tests.
	LobbyEmptyIdleTTL time.Duration

	// PreGameDuration is copied onto each CambiaGame at creation (CreateGameInstance) so the
	// pre-game card-reveal window before Started flips true is configurable (production
	// default; shortened in tests). Overridable per deployment via CAMBIA_PREGAME_DURATION.
	PreGameDuration time.Duration

	// PersistWG is copied onto each CambiaGame at creation (CreateGameInstance) as its
	// PersistWG (test-only; nil in production). See game.CambiaGame.PersistWG for what it
	// tracks and cambia-908 for why: it lets a test drain a game's background persistence
	// goroutines before returning instead of leaving them to outlive it. One group is shared by
	// every game this server creates, so a Wait covers all of them, and a caller must have
	// ordered its Wait after every Add it means to cover (cambia-942 F3).
	PersistWG *sync.WaitGroup
}

// NewGameServer initializes a new GameServer with empty, ephemeral stores.
func NewGameServer() *GameServer {
	return &GameServer{
		LobbyStore:        lobby.NewLobbyStore(),
		GameStore:         game.NewGameStore(),
		CircuitStore:      game.NewCircuitStore(),
		HubStore:          hub.NewHubStore(),
		Matchmaker:        matchmaking.NewMatchmaker(),
		CountdownDuration: defaultCountdownDuration,
		PostGameDuration:  defaultPostGameDuration,
		LobbyIdleTTL:      defaultLobbyIdleTTL,
		LobbyEmptyIdleTTL: defaultLobbyEmptyIdleTTL,
		PreGameDuration:   defaultPreGameDuration,
	}
}

// NewCambiaGameFromLobby creates a game instance from a Lobby's current state for the
// given player set. playerIDs is supplied by the caller (the hub passes its currently
// connected players so a mid-countdown disconnect never seats a ghost). The returned game
// is registered in the GameStore with its OnGameEnd callback and Emitter wired, but is NOT
// yet begun: the caller sets any routing it needs and then calls BeginPreGame so the
// pre-game reveal reaches clients through the emitter.
func (gs *GameServer) NewCambiaGameFromLobby(ctx context.Context, lob *lobby.Lobby, playerIDs []uuid.UUID, usernames map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
	lob.Mu.Lock()
	lobbyID := lob.ID
	// The lobbies row this game hangs off has host_user_id NOT NULL with an FK to users, and a
	// matchmade lobby's host is a sentinel with no users row behind it, so a system-hosted lobby
	// records its creator here instead (cambia-1087). Without the substitution the whole
	// UpsertInitialGameState transaction fails the FK and no games row is written for a
	// matchmade match at all, taking its rating with it.
	hostID := lob.HostUserID
	if lob.SystemHostedUnsafe() {
		hostID = lob.CreatorUserID
	}
	lobbyType := lob.Type
	gameMode := lob.GameMode
	rated := lob.Mode == "ranked"
	houseRules := lob.HouseRules
	circuit := lob.Circuit
	queueID := lob.QueueID
	lob.Mu.Unlock()

	// A queue owns the rules of the games it produces. A matchmade lobby has no host setting
	// rules, so what it plays is its queue's preset: the fixed ranked configuration
	// MATCHMAKING.md 5.2 specifies, carrying the reconnect grace from MATCHMAKING.md 8
	// (cambia-955). Taking the whole ruleset rather than the grace alone is what makes the
	// preset the client is shown for a queue the ruleset that queue actually plays (cambia-1088);
	// before this, every queue played game.DefaultHouseRules and the T1C fix in MATCHMAKING.md 5
	// existed only in the document.
	if queueID != "" {
		if preset, known := lobby.GetPreset(queueID); known {
			houseRules = preset.HouseRules
		}
	}

	return gs.CreateGameInstance(ctx, lobbyID, hostID, gameMode, lobbyType, rated, houseRules, circuit, playerIDs, usernames, emitter)
}

// CreateGameInstance creates a game from pre-extracted parameters and registers it in the
// GameStore with its OnGameEnd callback and Emitter wired. playerIDs lists the UUIDs of all
// players joining the game. lobbyType must be a valid lobby_type enum value
// ("private"/"public"/"matchmaking"); rated marks whether results feed the rating system
// (cambia-450). usernames supplies each player's already-known username (sourced from the hub's
// live connections, cambia-877); an id missing from the map falls back to a DB lookup bounded by
// ctx, for a caller with no live connection to read from (e.g. a test harness). emitter is the
// sink for all game events (the owning hub). The game is not begun here: the caller invokes
// BeginPreGame once routing is in place. Returns nil if fewer than two players are supplied.
func (gs *GameServer) CreateGameInstance(ctx context.Context, lobbyID, hostID uuid.UUID, gameMode, lobbyType string, rated bool, houseRules game.HouseRules, circuit game.Circuit, playerIDs []uuid.UUID, usernames map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
	g := game.NewCambiaGame()
	g.LobbyID = lobbyID
	g.HostUserID = hostID
	g.LobbyType = lobbyType
	g.Rated = rated
	if gs.PreGameDuration > 0 {
		g.PreGameDuration = gs.PreGameDuration
	}
	g.PersistWG = gs.PersistWG
	g.Circuit = circuit
	if circuit.Enabled {
		g.HouseRules = houseRules
		g.HouseRules.AllowDrawFromDiscardPile = true
		g.HouseRules.AllowReplaceAbilities = true
		g.HouseRules.ForfeitOnDisconnect = false
	} else {
		g.HouseRules = houseRules
	}

	// usernames is populated from live hub connections by the caller (hub.createAndStartGame);
	// hubFetchUsername (guests included: it resolves to whatever name was generated for them at
	// account creation) is only a fallback for an id with no live connection to read from, so a
	// caller like startTestGame that has no hub in the loop still gets a real answer rather than
	// a panic on a nil map lookup. Either way, Player.User.Username ends up populated: that means
	// ObfPlayerState.username (sync_state) and the game_results roster below both read a real
	// username instead of the zero value CreateGameInstance previously left it at (cambia-877).
	var players []*models.Player
	for _, uid := range playerIDs {
		username, known := usernames[uid]
		if !known {
			username = hubFetchUsername(ctx, uid)
		}
		players = append(players, &models.Player{
			ID:        uid,
			Connected: true,
			Hand:      []*models.Card{},
			User:      &models.User{ID: uid, Username: username},
		})
	}
	if len(players) < 2 {
		log.Printf("Lobby %s: cannot start game, not enough players (%d).", lobbyID, len(players))
		return nil
	}
	g.Players = players
	g.Emitter = emitter

	if circuit.Enabled && gs.CircuitStore != nil {
		existingState, _ := gs.CircuitStore.Get(lobbyID)
		if existingState == nil {
			pIDs := make([]int, len(playerIDs))
			playerMap := make(map[uuid.UUID]int)
			for i, uid := range playerIDs {
				pIDs[i] = i
				playerMap[uid] = i
			}
			cfg := engine.CircuitConfig{
				Format:     engine.CircuitFormat(circuit.Mode),
				NumPlayers: len(playerIDs),
				PlayerIDs:  pIDs,
			}
			circuitState, err := engine.NewCircuit(cfg)
			if err != nil {
				log.Printf("Lobby %s: failed to create circuit state: %v", lobbyID, err)
			} else {
				gs.CircuitStore.Set(lobbyID, circuitState, playerMap)
				log.Printf("Lobby %s: circuit state created (%s, %d rounds).", lobbyID, cfg.Format, circuitState.Config.NumRounds)
			}
		}
	}

	gs.attachOnGameEnd(g, lobbyID)
	gs.GameStore.AddGame(g)
	log.Printf("Created game %s from lobby %s (%d players, rated=%v)", g.ID, lobbyID, len(players), rated)
	return g
}

// tearDownLobby releases every trace of a lobby whose last joined member left. It is the
// lobby's OnEmpty callback (see CreateLobbyHandler) and runs on whichever goroutine released
// that membership, so it touches only lock-guarded stores and the hub's own channel API - never
// hub fields, which belong to the hub's Run goroutine.
//
// The hub is shut down as well as deregistered: it runs for as long as its lobby exists
// (cambia-808), so nothing else would ever stop it, and its Run loop deregisters itself on the
// way out. Deleting the entry here too covers a hub that was registered but never started.
func (gs *GameServer) tearDownLobby(lobbyID uuid.UUID) {
	gs.LobbyStore.DeleteLobby(lobbyID)

	if h, ok := gs.HubStore.GetHub(lobbyID); ok {
		h.Shutdown()
	}
	gs.HubStore.DeleteHub(lobbyID)

	// A lobby that dissolves while queued would otherwise still be matchable, and its circuit
	// standings would outlive every player who could read them.
	if gs.Matchmaker != nil {
		gs.Matchmaker.Dequeue(lobbyID)
	}
	if gs.CircuitStore != nil {
		gs.CircuitStore.Delete(lobbyID)
	}
	log.Printf("Lobby %s torn down: store entry, hub and queue state released.", lobbyID)
}

// hubGameFactory returns the hub.GameFactory a hub uses to build its backing game. It binds
// the GameServer's stores (GameStore/CircuitStore/HubStore) so the hub stays decoupled from
// them: the hub supplies its live lobby and connected player set, the GameServer owns creation.
func (gs *GameServer) hubGameFactory() hub.GameFactory {
	return func(lob *lobby.Lobby, playerIDs []uuid.UUID, usernames map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
		return gs.NewCambiaGameFromLobby(context.Background(), lob, playerIDs, usernames, emitter)
	}
}

// attachOnGameEnd wires the OnGameEnd callback that resets lobby state and emits results.
func (gs *GameServer) attachOnGameEnd(g *game.CambiaGame, lobbyID uuid.UUID) {
	g.OnGameEnd = func(endedLobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int, usernames map[uuid.UUID]string, rawScores map[uuid.UUID]int, cambiaCallerID uuid.UUID) {
		log.Printf("Game %s ended. OnGameEnd executing for lobby %s.", g.ID, endedLobbyID)

		lobInstance, exists := gs.LobbyStore.GetLobby(endedLobbyID)
		if !exists {
			log.Printf("Error in OnGameEnd: Lobby %s not found.", endedLobbyID)
			gs.GameStore.DeleteGame(g.ID)
			return
		}

		lobInstance.Mu.Lock()
		lobInstance.InGame = false
		lobInstance.GameID = uuid.Nil
		for uid := range lobInstance.ReadyStates {
			lobInstance.ReadyStates[uid] = false
		}
		statusPayload := lobInstance.GetLobbyStatusPayloadUnsafe()
		lobInstance.Mu.Unlock()

		// Enrich the roster with usernames sourced from the game's own player list (populated
		// at creation time from the authenticated user, cambia-877), not live hub connections:
		// a forfeiting player's socket is already closed by the time OnGameEnd fires, so a
		// connection-keyed lookup (buildLobbySnapshot's approach for the live lobby_state) would
		// silently miss exactly the player this roster most needs to name correctly. usernames
		// comes from the OnGameEnd callback itself (built while endGame still holds g.mu) rather
		// than a fresh call back into the game here, which would self-deadlock that same lock.
		if users, ok := statusPayload["users"].([]map[string]interface{}); ok {
			for _, u := range users {
				if uidStr, ok := u["id"].(string); ok {
					if uid, err := uuid.Parse(uidStr); err == nil {
						if uname, found := usernames[uid]; found {
							u["username"] = uname
						}
					}
				}
			}
		}

		// Emit game results to the hub.
		h, hasHub := gs.HubStore.GetHub(endedLobbyID)
		if hasHub {
			// Drive the hub to PhasePostGame so clients leave the live table and reach the
			// results screen (mirrors the ranked HandleRoundEnd path, cambia-510). Routed through
			// the hub's own goroutine since OnGameEnd may run on a foreign goroutine (game timers).
			h.NotifyGameEnded()

			resultMsg := map[string]interface{}{
				"type":         "game_results",
				"winner":       winner.String(),
				"scores":       map[string]int{},
				"lobby_status": statusPayload,
			}
			for pid, sc := range scores {
				resultMsg["scores"].(map[string]int)[pid.String()] = sc
			}
			h.Emit("game_results", resultMsg)

			// Circuit round/completion events. RecordRound gets rawScores (pre-WinBonus/
			// FalseCambiaPenalty, cambia-1009) and the real Cambia caller mapped to its engine
			// seat (cambia-1008), not the display-adjusted scores or the -1 "no caller" sentinel
			// this used to hardcode.
			if g.Circuit.Enabled && gs.CircuitStore != nil {
				circuitState, playerMap := gs.CircuitStore.Get(endedLobbyID)
				if circuitState != nil && playerMap != nil {
					engineScores := make(map[int]int)
					for playerUUID, score := range rawScores {
						if engineID, ok := playerMap[playerUUID]; ok {
							engineScores[engineID] = score
						}
					}
					engineCallerID := -1
					if eid, ok := playerMap[cambiaCallerID]; ok {
						engineCallerID = eid
					}
					if err := circuitState.RecordRound(engineScores, engineCallerID); err != nil {
						log.Printf("Circuit round error for lobby %s: %v", endedLobbyID, err)
					} else if circuitState.IsComplete() {
						// The circuit's one rating update, read off the final standings before the
						// state is dropped (RULES.md T6): the rounds themselves rate nothing. The
						// database work runs on its own goroutine, so this does not block OnGameEnd
						// under the game mutex.
						gs.finalizeCircuitRatings(g, circuitState.GetStandings(), playerMap)
						h.Emit("circuit_complete", map[string]interface{}{"standings": circuitState.GetStandings()})
						gs.CircuitStore.Delete(endedLobbyID)
					} else {
						h.Emit("circuit_round", map[string]interface{}{
							"current_round": circuitState.CurrentRound,
							"standings":     circuitState.GetStandings(),
						})
					}
				}
			}
		}

		gs.GameStore.DeleteGame(g.ID)
		log.Printf("Game %s removed from store.", g.ID)
	}
}
