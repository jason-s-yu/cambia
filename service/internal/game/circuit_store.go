// internal/game/circuit_store.go
package game

import (
	"sync"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
)

// circuitEntry bundles a CircuitState with its service UUID → engine player ID mapping.
type circuitEntry struct {
	State     *engine.CircuitState
	PlayerMap map[uuid.UUID]int // service UUID -> engine player ID (0-based index)
}

// CircuitStore manages active circuit tournament states keyed by lobby ID.
type CircuitStore struct {
	mu      sync.RWMutex
	entries map[uuid.UUID]*circuitEntry
}

func NewCircuitStore() *CircuitStore {
	return &CircuitStore{entries: make(map[uuid.UUID]*circuitEntry)}
}

// Get returns the CircuitState and player map for the given lobby, or (nil, nil) if not found.
func (cs *CircuitStore) Get(lobbyID uuid.UUID) (*engine.CircuitState, map[uuid.UUID]int) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	e := cs.entries[lobbyID]
	if e == nil {
		return nil, nil
	}
	return e.State, e.PlayerMap
}

// Set stores a CircuitState and its UUID→engineID mapping for the given lobby.
func (cs *CircuitStore) Set(lobbyID uuid.UUID, state *engine.CircuitState, playerMap map[uuid.UUID]int) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.entries[lobbyID] = &circuitEntry{State: state, PlayerMap: playerMap}
}

// Delete removes the circuit entry for the given lobby.
func (cs *CircuitStore) Delete(lobbyID uuid.UUID) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	delete(cs.entries, lobbyID)
}
