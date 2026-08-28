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

// defaultLobbyIdleTTL is how long a lobby whose game is in progress may hold no live connections
// before its hub reaps it (cambia-836). Tests may lower GameServer.LobbyIdleTTL for speed.
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

	// LobbyIdleTTL is copied onto each hub at creation: how long a lobby whose game is in
	// progress may hold no live connections before its hub tears it down (cambia-836).
	// Overridable per deployment via CAMBIA_LOBBY_IDLE_TTL; shortened in tests.
	LobbyIdleTTL time.Duration

	// LobbyEmptyIdleTTL is copied onto each hub alongside LobbyIdleTTL and is the window that
	// applies while no game is in progress: an abandoned pre-game or post-game lobby is released
	// in minutes rather than sitting out the game grace (cambia-884). Overridable per deployment
	// via CAMBIA_LOBBY_EMPTY_IDLE_TTL; shortened in tests.
	LobbyEmptyIdleTTL time.Duration

	// PreGameDuration is copied onto each CambiaGame at creation (CreateGameInstance) so the
	// pre-game card-reveal window before Started flips true is configurable (production
	// default; shortened in tests). Overridable per deployment via CAMBIA_PREGAME_DURATION.
	PreGameDuration time.Duration
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
func (gs *GameServer) NewCambiaGameFromLobby(ctx context.Context, lob *lobby.Lobby, playerIDs []uuid.UUID, emitter game.Emitter) *game.CambiaGame {
	lob.Mu.Lock()
	lobbyID := lob.ID
	hostID := lob.HostUserID
	lobbyType := lob.Type
	gameMode := lob.GameMode
	rated := lob.Mode == "ranked"
	houseRules := lob.HouseRules
	circuit := lob.Circuit
	lob.Mu.Unlock()

	return gs.CreateGameInstance(ctx, lobbyID, hostID, gameMode, lobbyType, rated, houseRules, circuit, playerIDs, emitter)
}

// CreateGameInstance creates a game from pre-extracted parameters and registers it in the
// GameStore with its OnGameEnd callback and Emitter wired. playerIDs lists the UUIDs of all
// players joining the game. lobbyType must be a valid lobby_type enum value
// ("private"/"public"/"matchmaking"); rated marks whether results feed the rating system
// (cambia-450). emitter is the sink for all game events (the owning hub). The game is not
// begun here: the caller invokes BeginPreGame once routing is in place. Returns nil if
// fewer than two players are supplied.
func (gs *GameServer) CreateGameInstance(ctx context.Context, lobbyID, hostID uuid.UUID, gameMode, lobbyType string, rated bool, houseRules game.HouseRules, circuit game.Circuit, playerIDs []uuid.UUID, emitter game.Emitter) *game.CambiaGame {
	g := game.NewCambiaGame()
	g.LobbyID = lobbyID
	g.HostUserID = hostID
	g.LobbyType = lobbyType
	g.Rated = rated
	if gs.PreGameDuration > 0 {
		g.PreGameDuration = gs.PreGameDuration
	}
	g.Circuit = circuit
	if circuit.Enabled {
		g.HouseRules = houseRules
		g.HouseRules.AllowDrawFromDiscardPile = true
		g.HouseRules.AllowReplaceAbilities = true
		g.HouseRules.ForfeitOnDisconnect = false
	} else {
		g.HouseRules = houseRules
	}

	var players []*models.Player
	for _, uid := range playerIDs {
		players = append(players, &models.Player{
			ID:        uid,
			Connected: true,
			Hand:      []*models.Card{},
			User:      &models.User{ID: uid},
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
	return func(lob *lobby.Lobby, playerIDs []uuid.UUID, emitter game.Emitter) *game.CambiaGame {
		return gs.NewCambiaGameFromLobby(context.Background(), lob, playerIDs, emitter)
	}
}

// attachOnGameEnd wires the OnGameEnd callback that resets lobby state and emits results.
func (gs *GameServer) attachOnGameEnd(g *game.CambiaGame, lobbyID uuid.UUID) {
	g.OnGameEnd = func(endedLobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int) {
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

			// Circuit round/completion events.
			if g.Circuit.Enabled && gs.CircuitStore != nil {
				circuitState, playerMap := gs.CircuitStore.Get(endedLobbyID)
				if circuitState != nil && playerMap != nil {
					engineScores := make(map[int]int)
					for playerUUID, score := range scores {
						if engineID, ok := playerMap[playerUUID]; ok {
							engineScores[engineID] = score
						}
					}
					if err := circuitState.RecordRound(engineScores, -1); err != nil {
						log.Printf("Circuit round error for lobby %s: %v", endedLobbyID, err)
					} else if circuitState.IsComplete() {
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
