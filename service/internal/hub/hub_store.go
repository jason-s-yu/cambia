// internal/hub/hub_store.go
package hub

import (
	"sync"

	"github.com/google/uuid"
)

// HubStore is a thread-safe registry of active Hub instances.
type HubStore struct {
	mu   sync.Mutex
	hubs map[uuid.UUID]*Hub // lobbyID → Hub
}

// NewHubStore creates an empty HubStore.
func NewHubStore() *HubStore {
	return &HubStore{
		hubs: make(map[uuid.UUID]*Hub),
	}
}

// CreateHub registers a hub. Overwrites any existing entry for the same ID.
func (s *HubStore) CreateHub(h *Hub) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hubs[h.ID] = h
}

// GetHub returns the hub for a lobby ID, or (nil, false) if not found.
func (s *HubStore) GetHub(id uuid.UUID) (*Hub, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	h, ok := s.hubs[id]
	return h, ok
}

// DeleteHub removes the hub for a lobby ID.
func (s *HubStore) DeleteHub(id uuid.UUID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.hubs, id)
}

// ListPublicHubs returns all hubs whose lobby type is not "private".
func (s *HubStore) ListPublicHubs() []*Hub {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Hub, 0, len(s.hubs))
	for _, h := range s.hubs {
		if h.Lobby != nil && h.Lobby.Type != "private" {
			out = append(out, h)
		}
	}
	return out
}
