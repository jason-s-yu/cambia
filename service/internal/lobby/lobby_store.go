// internal/lobby/lobby_store.go
package lobby

import (
	"log"
	"sync"

	"github.com/google/uuid"
)

// LobbyStore manages active ephemeral lobbies in memory.
// It provides thread-safe access to add, retrieve, and delete lobbies.
type LobbyStore struct {
	mu      sync.Mutex           // Protects access to the lobbies map.
	lobbies map[uuid.UUID]*Lobby // Map of lobby ID to Lobby object pointer.
}

// NewLobbyStore initializes and returns an empty LobbyStore.
func NewLobbyStore() *LobbyStore {
	return &LobbyStore{
		lobbies: make(map[uuid.UUID]*Lobby),
	}
}

// AddLobby adds a new lobby instance to the store.
// It's recommended to configure the lobby's OnEmpty callback before adding it
// to ensure automatic cleanup when the last user leaves.
func (s *LobbyStore) AddLobby(lobby *Lobby) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.lobbies[lobby.ID]; exists {
		log.Printf("LobbyStore WARNING: Attempted to add lobby %s which already exists.", lobby.ID)
		return // Avoid overwriting existing lobby.
	}
	s.lobbies[lobby.ID] = lobby
	log.Printf("LobbyStore: Added lobby %s.", lobby.ID)
}

// DeleteLobby removes a lobby instance from the store by its ID.
// This is typically called via the lobby's OnEmpty callback.
func (s *LobbyStore) DeleteLobby(id uuid.UUID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.lobbies[id]; exists {
		delete(s.lobbies, id)
		log.Printf("LobbyStore: Deleted lobby %s.", id)
	} else {
		log.Printf("LobbyStore WARNING: Attempted to delete non-existent lobby %s.", id)
	}
}

// GetLobby retrieves a lobby instance from the store by its ID.
// Returns the lobby pointer and a boolean indicating if it was found.
func (s *LobbyStore) GetLobby(id uuid.UUID) (*Lobby, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.lobbies[id]
	return l, ok
}

// GetLobbies returns a copy of the map containing all active lobbies.
// This is primarily used for listing lobbies (e.g., on a dashboard) or debugging.
// Returning a copy prevents race conditions if the caller iterates over the map
// while another goroutine modifies the store.
func (s *LobbyStore) GetLobbies() map[uuid.UUID]*Lobby {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Create a copy of the map to return.
	lobbiesCopy := make(map[uuid.UUID]*Lobby, len(s.lobbies))
	for k, v := range s.lobbies {
		lobbiesCopy[k] = v
	}
	return lobbiesCopy
}
