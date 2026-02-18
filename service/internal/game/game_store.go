// internal/game/game_store.go
package game

import (
	"sync"

	"github.com/google/uuid"
)

// GameStore manages active CambiaGame instances in memory.
type GameStore struct {
	mu    sync.Mutex
	games map[uuid.UUID]*CambiaGame
}

// NewGameStore creates a new, empty GameStore.
func NewGameStore() *GameStore {
	return &GameStore{
		games: make(map[uuid.UUID]*CambiaGame),
	}
}

// AddGame adds a game instance to the store.
func (s *GameStore) AddGame(game *CambiaGame) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.games[game.ID] = game
}

// GetGame retrieves a game instance by its ID.
func (s *GameStore) GetGame(id uuid.UUID) (*CambiaGame, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g, exists := s.games[id]
	return g, exists
}

// DeleteGame removes a game instance from the store by its ID.
func (s *GameStore) DeleteGame(id uuid.UUID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.games, id)
}

// GetGameByLobbyID finds a game associated with a specific lobby ID.
// Returns the game instance or nil if not found.
func (s *GameStore) GetGameByLobbyID(lobbyID uuid.UUID) *CambiaGame {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, g := range s.games {
		// Check if the game instance has a non-nil LobbyID that matches.
		if g.LobbyID != uuid.Nil && g.LobbyID == lobbyID {
			return g
		}
	}
	return nil
}
