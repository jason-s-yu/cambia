// internal/handlers/api_server.go
package handlers

import (
	"context"
	"log"
	"sync"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// GameServer manages the central stores for active lobbies and games.
// It provides methods for creating game instances from lobbies.
type GameServer struct {
	Mutex      sync.Mutex // Protects access to the stores themselves if needed (currently stores handle internal locking).
	LobbyStore *lobby.LobbyStore
	GameStore  *game.GameStore
}

// NewGameServer initializes a new GameServer with empty, ephemeral stores.
func NewGameServer() *GameServer {
	return &GameServer{
		LobbyStore: lobby.NewLobbyStore(),
		GameStore:  game.NewGameStore(),
	}
}

// NewCambiaGameFromLobby creates a new Cambia game instance based on the state of a lobby.
// It gathers participant IDs, copies lobby rules, sets up game end callbacks,
// adds the game to the store, and starts the pre-game phase.
// This function assumes the lobby lock is HELD by the caller or that lobby state is immutable during this call.
// It's now recommended to use CreateGameInstance which avoids passing the locked lobby object directly.
func (gs *GameServer) NewCambiaGameFromLobby(ctx context.Context, lob *lobby.Lobby) *game.CambiaGame {
	lob.Mu.Lock() // Lock lobby to safely read initial state.
	g := game.NewCambiaGame()
	g.LobbyID = lob.ID
	g.HouseRules = lob.HouseRules // Copy rules from lobby.
	g.Circuit = lob.Circuit       // Copy circuit settings.

	// Gather participants from the lobby's connections map.
	var players []*models.Player
	for userID, conn := range lob.Connections {
		// Need user details potentially? Fetch username from connection.
		players = append(players, &models.Player{
			ID:        userID,
			Connected: true, // Assume connected at game start.
			Hand:      []*models.Card{},
			User:      &models.User{ID: userID, Username: conn.Username}, // Use username stored in LobbyConnection.
		})
	}
	g.Players = players
	lobbyID := lob.ID // Capture lobby ID before unlocking.
	lob.Mu.Unlock()   // Unlock lobby after reading state.

	// Attach OnGameEnd callback function.
	// This callback handles transitioning the lobby back from in-game state
	// and broadcasting results when the game concludes.
	g.OnGameEnd = func(endedLobbyID uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int) {
		log.Printf("Game %s ended. OnGameEnd callback executing for lobby %s.", g.ID, endedLobbyID)
		lobInstance, exists := gs.LobbyStore.GetLobby(endedLobbyID)
		if !exists {
			log.Printf("Error in OnGameEnd: Lobby %s not found in store.", endedLobbyID)
			gs.GameStore.DeleteGame(g.ID) // Clean up game if lobby is gone.
			return
		}

		lobInstance.Mu.Lock() // Lock lobby before modifying its state.
		lobInstance.InGame = false
		lobInstance.GameID = uuid.Nil // Clear game ID reference.

		// Reset ready states for players still connected.
		for uid := range lobInstance.Connections {
			lobInstance.ReadyStates[uid] = false
		}
		// Get current lobby status *after* resetting ready states.
		statusPayload := lobInstance.GetLobbyStatusPayloadUnsafe()
		lobInstance.Mu.Unlock() // Unlock before broadcasting.

		// Prepare and broadcast results message back to the lobby.
		log.Printf("Broadcasting game end results to lobby %s", endedLobbyID)
		resultMsg := map[string]interface{}{
			"type":         "game_results", // Consider a more specific type?
			"winner":       winner.String(),
			"scores":       map[string]int{},
			"lobby_status": statusPayload, // Include updated lobby status.
		}
		for pid, sc := range scores {
			resultMsg["scores"].(map[string]int)[pid.String()] = sc
		}
		lobInstance.BroadcastAll(resultMsg) // BroadcastAll handles its own locking.

		// Clean up the game instance from the store after results are sent.
		gs.GameStore.DeleteGame(g.ID)
		log.Printf("Game %s instance removed from store.", g.ID)
	}

	// Store the newly created game and start its pre-game phase.
	gs.GameStore.AddGame(g)
	g.BeginPreGame()
	log.Printf("Created and started game %s from lobby %s", g.ID, lobbyID)
	return g
}
