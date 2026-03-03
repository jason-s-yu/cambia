// internal/handlers/api_server.go
package handlers

import (
	"context"
	"log"
	"sync"

	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// GameServer manages the central stores for active lobbies and games.
type GameServer struct {
	Mutex        sync.Mutex
	LobbyStore   *lobby.LobbyStore
	GameStore    *game.GameStore
	CircuitStore *game.CircuitStore
	HubStore     *hub.HubStore
	Matchmaker   *matchmaking.Matchmaker
}

// NewGameServer initializes a new GameServer with empty, ephemeral stores.
func NewGameServer() *GameServer {
	return &GameServer{
		LobbyStore:   lobby.NewLobbyStore(),
		GameStore:    game.NewGameStore(),
		CircuitStore: game.NewCircuitStore(),
		HubStore:     hub.NewHubStore(),
		Matchmaker:   matchmaking.NewMatchmaker(),
	}
}

// NewCambiaGameFromLobby creates a game instance from a Lobby's current state.
// Prefer CreateGameInstance for new code; this is kept for backward compatibility.
func (gs *GameServer) NewCambiaGameFromLobby(ctx context.Context, lob *lobby.Lobby) *game.CambiaGame {
	playerIDs := lob.JoinedUsers() // acquires lock internally

	var players []*models.Player
	for _, uid := range playerIDs {
		players = append(players, &models.Player{
			ID:        uid,
			Connected: true,
			Hand:      []*models.Card{},
			User:      &models.User{ID: uid},
		})
	}

	lob.Mu.Lock()
	g := game.NewCambiaGame()
	g.LobbyID = lob.ID
	g.HouseRules = lob.HouseRules
	g.Circuit = lob.Circuit
	g.Players = players
	lobbyID := lob.ID
	lob.Mu.Unlock()

	gs.attachOnGameEnd(g, lobbyID)
	gs.GameStore.AddGame(g)
	g.BeginPreGame()
	log.Printf("Created and started game %s from lobby %s", g.ID, lobbyID)
	return g
}

// CreateGameInstance creates a game from pre-extracted parameters.
// playerIDs lists the UUIDs of all players joining the game.
func (gs *GameServer) CreateGameInstance(ctx context.Context, lobbyID, hostID uuid.UUID, gameMode string, houseRules game.HouseRules, circuit game.Circuit, playerIDs []uuid.UUID) *game.CambiaGame {
	g := game.NewCambiaGame()
	g.LobbyID = lobbyID
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
		})
	}
	if len(players) < 2 {
		log.Printf("Lobby %s: cannot start game, not enough players (%d).", lobbyID, len(players))
		return nil
	}
	g.Players = players

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
	g.BeginPreGame()
	return g
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
