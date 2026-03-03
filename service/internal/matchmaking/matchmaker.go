package matchmaking

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// QueuedLobby represents a lobby (party) waiting in a matchmaking queue.
type QueuedLobby struct {
	LobbyID     uuid.UUID
	PlayerCount int       // current party size
	QueueID     string    // e.g. "h2h_rapid", "ffa4_standard"
	TargetCount int       // players needed (2 for h2h, 4 for ffa4)
	AvgRating   float64   // avg party rating
	MaxRD       float64   // worst RD in party
	QueuedAt    time.Time
	IsRanked    bool
}

// MatchedPlayer is a player included in a formed match.
type MatchedPlayer struct {
	UserID   uuid.UUID
	Username string
	LobbyID  uuid.UUID // which lobby they came from
}

// MatchResult is emitted via OnMatchFormed when a match is complete.
type MatchResult struct {
	HostLobbyID uuid.UUID
	Players     []MatchedPlayer
	QueueID     string
	IsRanked    bool
}

// QueueStat holds per-queue statistics.
type QueueStat struct {
	PlayerCount int     `json:"playerCount"`
	AvgWaitSec  float64 `json:"avgWaitSec"`
}

// Matchmaker manages multiple queues and forms matches between lobbies.
type Matchmaker struct {
	mu     sync.Mutex
	queues map[string][]*QueuedLobby // queueID → list sorted by QueuedAt

	// OnMatchFormed is called (under mu released) when a match is formed.
	OnMatchFormed func(result MatchResult)
}

// NewMatchmaker creates a Matchmaker with empty queues.
func NewMatchmaker() *Matchmaker {
	return &Matchmaker{
		queues: make(map[string][]*QueuedLobby),
	}
}

// Enqueue adds a lobby to the appropriate queue.
func (m *Matchmaker) Enqueue(entry *QueuedLobby) error {
	if entry == nil {
		return fmt.Errorf("matchmaking: entry is nil")
	}
	if entry.LobbyID == uuid.Nil {
		return fmt.Errorf("matchmaking: LobbyID is required")
	}
	if entry.QueueID == "" {
		return fmt.Errorf("matchmaking: QueueID is required")
	}
	if entry.PlayerCount <= 0 {
		return fmt.Errorf("matchmaking: PlayerCount must be > 0")
	}
	if entry.TargetCount <= 0 {
		return fmt.Errorf("matchmaking: TargetCount must be > 0")
	}
	if entry.PlayerCount > entry.TargetCount {
		return fmt.Errorf("matchmaking: PlayerCount %d exceeds TargetCount %d", entry.PlayerCount, entry.TargetCount)
	}
	if entry.QueuedAt.IsZero() {
		entry.QueuedAt = time.Now()
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.queues[entry.QueueID] = append(m.queues[entry.QueueID], entry)
	log.Printf("matchmaking: enqueued lobby %s into queue %s (party=%d, target=%d)",
		entry.LobbyID, entry.QueueID, entry.PlayerCount, entry.TargetCount)
	return nil
}

// Dequeue removes a lobby from all queues (e.g. lobby canceled search).
func (m *Matchmaker) Dequeue(lobbyID uuid.UUID) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for qid, entries := range m.queues {
		filtered := entries[:0]
		for _, e := range entries {
			if e.LobbyID != lobbyID {
				filtered = append(filtered, e)
			}
		}
		m.queues[qid] = filtered
	}
}

// Run starts the matchmaking loop. Blocks until ctx is canceled.
func (m *Matchmaker) Run(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.processQueues()
		}
	}
}

// processQueues iterates all queues and attempts to form matches.
func (m *Matchmaker) processQueues() {
	m.mu.Lock()
	queueIDs := make([]string, 0, len(m.queues))
	for qid := range m.queues {
		queueIDs = append(queueIDs, qid)
	}
	m.mu.Unlock()

	for _, qid := range queueIDs {
		m.mu.Lock()
		entries := make([]*QueuedLobby, len(m.queues[qid]))
		copy(entries, m.queues[qid])
		m.mu.Unlock()

		m.tryMatchQueue(qid, entries)
	}
}

// minQuality returns the minimum acceptable match quality based on wait time.
func minQuality(waitSec float64) float64 {
	switch {
	case waitSec < 30:
		return 0.80
	case waitSec < 60:
		return 0.70
	case waitSec < 120:
		return 0.55
	default:
		return 0.40
	}
}

// glicko2Quality computes match quality between two rated lobbies.
// Uses Q = exp(-Δμ² / (2*(c² + φ1² + φ2²))) where c = sqrt(2)*173.7178.
func glicko2Quality(a, b *QueuedLobby) float64 {
	const beta = 173.7178
	c := math.Sqrt2 * beta
	dmu := a.AvgRating - b.AvgRating
	denom := 2 * (c*c + a.MaxRD*a.MaxRD + b.MaxRD*b.MaxRD)
	return math.Exp(-(dmu * dmu) / denom)
}

// tryMatchQueue attempts to form matches within a single queue.
func (m *Matchmaker) tryMatchQueue(queueID string, entries []*QueuedLobby) {
	if len(entries) == 0 {
		return
	}

	// Sort oldest first.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].QueuedAt.Before(entries[j].QueuedAt)
	})

	target := entries[0].TargetCount
	now := time.Now()

	// Track which entries have been matched this pass.
	matched := make(map[uuid.UUID]bool)

	// Try to fill each unmatched entry with partners.
	for i, anchor := range entries {
		if matched[anchor.LobbyID] {
			continue
		}
		if anchor.PlayerCount == target {
			// Solo lobby exactly fills the target — match immediately.
			m.commitMatch(queueID, []*QueuedLobby{anchor}, matched)
			continue
		}

		// Find partners to fill remaining slots.
		need := target - anchor.PlayerCount
		group := []*QueuedLobby{anchor}
		groupSize := anchor.PlayerCount

		anchorWait := now.Sub(anchor.QueuedAt).Seconds()
		minQ := minQuality(anchorWait)

		for j := i + 1; j < len(entries); j++ {
			candidate := entries[j]
			if matched[candidate.LobbyID] {
				continue
			}
			if candidate.PlayerCount > need {
				continue
			}
			// Quality check for ranked H2H (target==2).
			if anchor.IsRanked && target == 2 {
				q := glicko2Quality(anchor, candidate)
				if q < minQ {
					continue
				}
			}
			group = append(group, candidate)
			groupSize += candidate.PlayerCount
			need -= candidate.PlayerCount
			if need == 0 {
				break
			}
		}

		if groupSize == target {
			m.commitMatch(queueID, group, matched)
		}
	}
}

// commitMatch marks entries matched, removes them from the queue, and fires OnMatchFormed.
func (m *Matchmaker) commitMatch(queueID string, group []*QueuedLobby, matched map[uuid.UUID]bool) {
	for _, e := range group {
		matched[e.LobbyID] = true
	}

	hostLobbyID := group[0].LobbyID
	isRanked := group[0].IsRanked

	// Build players list (lobby-level only; per-player details populated by caller).
	var players []MatchedPlayer
	for _, e := range group {
		players = append(players, MatchedPlayer{
			LobbyID: e.LobbyID,
		})
	}

	result := MatchResult{
		HostLobbyID: hostLobbyID,
		Players:     players,
		QueueID:     queueID,
		IsRanked:    isRanked,
	}

	// Remove matched entries from the live queue.
	m.mu.Lock()
	live := m.queues[queueID]
	filtered := live[:0]
	for _, e := range live {
		if !matched[e.LobbyID] {
			filtered = append(filtered, e)
		}
	}
	m.queues[queueID] = filtered
	m.mu.Unlock()

	log.Printf("matchmaking: match formed in queue %s with %d lobbies (host=%s)", queueID, len(group), hostLobbyID)

	if m.OnMatchFormed != nil {
		m.OnMatchFormed(result)
	}
}

// QueueStats returns per-queue statistics.
func (m *Matchmaker) QueueStats() map[string]QueueStat {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	stats := make(map[string]QueueStat, len(m.queues))
	for qid, entries := range m.queues {
		playerCount := 0
		var totalWait float64
		for _, e := range entries {
			playerCount += e.PlayerCount
			totalWait += now.Sub(e.QueuedAt).Seconds()
		}
		var avgWait float64
		if len(entries) > 0 {
			avgWait = totalWait / float64(len(entries))
		}
		stats[qid] = QueueStat{
			PlayerCount: playerCount,
			AvgWaitSec:  avgWait,
		}
	}
	return stats
}
