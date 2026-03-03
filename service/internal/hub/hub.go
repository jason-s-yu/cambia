// internal/hub/hub.go
package hub

import (
	"context"
	"encoding/json"
	"log"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// LobbyPhase represents the current lifecycle state of a hub.
type LobbyPhase int

const (
	PhaseOpen       LobbyPhase = iota // Lobby is open for joining
	PhaseSearching                    // Lobby is in matchmaker queue
	PhaseReadyCheck                   // Ready check in progress
	PhaseCountdown                    // Countdown before game start
	PhaseInGame                       // Game actively being played (casual single game OR ranked round)
	PhaseRoundEnd                     // Between ranked rounds, showing scores
	PhasePostGame                     // Game ended, showing results
	PhaseMatchEnd                     // Match/circuit complete
)

// String returns the wire-format string for a LobbyPhase.
func (p LobbyPhase) String() string {
	switch p {
	case PhaseOpen:
		return "open"
	case PhaseSearching:
		return "searching"
	case PhaseReadyCheck:
		return "ready_check"
	case PhaseCountdown:
		return "countdown"
	case PhaseInGame:
		return "in_game"
	case PhaseRoundEnd:
		return "round_end"
	case PhasePostGame:
		return "post_game"
	case PhaseMatchEnd:
		return "match_end"
	default:
		return "open"
	}
}

// MatchedPlayer represents a player matched by the matchmaker.
type MatchedPlayer struct {
	UserID   uuid.UUID
	Username string
	IsHost   bool
}

// Hub manages a single lobby's lifecycle through a single goroutine.
// All lobby/game state access is serialized through the Run() select loop — no mutex needed.
type Hub struct {
	ID    uuid.UUID
	Phase LobbyPhase

	Lobby *lobby.Lobby
	Game  *game.CambiaGame

	seq   uint64 // monotonic; only accessed from Run() goroutine via nextSeq()
	conns map[uuid.UUID]*Connection // userID → connection

	// Match state (ranked/circuit)
	QueueID  string
	IsRanked bool
	TotalRounds int

	// Multi-round match state
	RoundsPlayed     int
	CumulativeScores map[uuid.UUID]int   // running totals across rounds
	RoundHistory     []map[uuid.UUID]int // per-round scores
	DealerSeatIdx    int                 // rotates each round

	// Matchmaker integration
	matched chan []MatchedPlayer // matchmaker sends matched players here

	// Channels for Run() select loop
	join     chan *Connection
	leave    chan uuid.UUID
	incoming chan ClientMsg
	shutdown chan struct{}
}

// NewHub creates a new hub for the given lobby.
func NewHub(lob *lobby.Lobby) *Hub {
	return &Hub{
		ID:               lob.ID,
		Phase:            PhaseOpen,
		Lobby:            lob,
		conns:            make(map[uuid.UUID]*Connection),
		CumulativeScores: make(map[uuid.UUID]int),
		RoundHistory:     make([]map[uuid.UUID]int, 0),
		matched:          make(chan []MatchedPlayer, 1),
		join:             make(chan *Connection, 8),
		leave:            make(chan uuid.UUID, 8),
		incoming:         make(chan ClientMsg, 64),
		shutdown:         make(chan struct{}),
	}
}

// Run is the hub's main goroutine. It serializes all state access.
// Call this in its own goroutine.
func (h *Hub) Run(ctx context.Context) {
	defer h.cleanup()
	for {
		select {
		case <-ctx.Done():
			return
		case conn := <-h.join:
			h.conns[conn.UserID] = conn
			h.sendLobbyState(conn)
			h.broadcastLobbyUpdate()
		case userID := <-h.leave:
			if conn, ok := h.conns[userID]; ok {
				conn.Close()
				delete(h.conns, userID)
			}
			if len(h.conns) == 0 {
				return // dissolve hub
			}
		case msg := <-h.incoming:
			h.dispatch(msg)
		case players := <-h.matched:
			h.handleMatchFound(players)
		case <-h.shutdown:
			return
		}
	}
}

// cleanup closes all remaining connections when the hub exits.
func (h *Hub) cleanup() {
	for _, conn := range h.conns {
		conn.Close()
	}
}

// dispatch routes a ClientMsg based on the current phase.
func (h *Hub) dispatch(msg ClientMsg) {
	// Sequence check: if client is behind, send a sync snapshot and discard.
	if msg.LastSeq < h.seq && h.seq > 0 {
		h.sendSyncState(msg.UserID)
		return
	}

	// Handle synthetic internal messages regardless of phase.
	if msg.Type == "_start_next_round" && h.Phase == PhaseRoundEnd {
		h.startNextRound()
		return
	}

	switch h.Phase {
	case PhaseOpen, PhaseReadyCheck, PhaseCountdown:
		h.handleLobbyMsg(msg)
	case PhaseSearching:
		h.handleSearchingMsg(msg)
	case PhaseInGame:
		// Handles both casual single game and ranked rounds.
		h.handleGameMsg(msg)
	case PhaseRoundEnd:
		if msg.Type == "chat" {
			h.handleLobbyMsg(msg)
		}
	case PhasePostGame, PhaseMatchEnd:
		// Only allow chat in post-game; ignore game actions.
		if msg.Type == "chat" {
			h.handleLobbyMsg(msg)
		}
	}
}

// handleLobbyMsg handles lobby-phase messages (ready, chat, rules, etc.).
// Runs inside the hub's Run() goroutine — no external lock needed.
func (h *Hub) handleLobbyMsg(msg ClientMsg) {
	conn, ok := h.conns[msg.UserID]
	if !ok {
		return
	}

	switch msg.Type {
	case "ready":
		if h.Lobby.MarkUserReady(msg.UserID) {
			// All ready — transition to ready-check / countdown
			h.Phase = PhaseReadyCheck
			h.Emit("phase_change", map[string]interface{}{"phase": "ready_check"})
		}
		h.broadcastLobbyUpdate()

	case "unready":
		h.Lobby.MarkUserUnready(msg.UserID)
		h.broadcastLobbyUpdate()

	case "invite":
		var payload struct {
			UserID string `json:"userID"`
		}
		if err := json.Unmarshal(msg.Body, &payload); err != nil || payload.UserID == "" {
			conn.SendEnvelope(h.errEnvelope("invalid invite payload"))
			return
		}
		targetID, err := uuid.Parse(payload.UserID)
		if err != nil {
			conn.SendEnvelope(h.errEnvelope("invalid userID format"))
			return
		}
		h.Lobby.InviteUser(targetID)

	case "leave_lobby":
		// Enqueue a leave; hub will handle removal in the main select.
		h.leave <- msg.UserID

	case "chat":
		var payload struct {
			Msg string `json:"msg"`
		}
		if err := json.Unmarshal(msg.Body, &payload); err != nil || payload.Msg == "" {
			return
		}
		h.Emit("chat", map[string]interface{}{
			"userID":   msg.UserID,
			"username": conn.Username,
			"msg":      payload.Msg,
		})

	case "update_rules":
		if !conn.IsHost {
			conn.SendEnvelope(h.errEnvelope("only the host can update rules"))
			return
		}
		var payload struct {
			Rules map[string]interface{} `json:"rules"`
		}
		if err := json.Unmarshal(msg.Body, &payload); err != nil || payload.Rules == nil {
			conn.SendEnvelope(h.errEnvelope("invalid update_rules payload"))
			return
		}
		if err := h.Lobby.UpdateUnsafe(payload.Rules); err != nil {
			log.Printf("hub %s: UpdateUnsafe error: %v", h.ID, err)
			conn.SendEnvelope(h.errEnvelope("failed to apply rule updates"))
		}

	case "start_game":
		if !conn.IsHost {
			conn.SendEnvelope(h.errEnvelope("only the host can start the game"))
			return
		}
		if h.Phase == PhaseInGame {
			conn.SendEnvelope(h.errEnvelope("game already in progress"))
			return
		}
		if !h.Lobby.AreAllReadyUnsafe() {
			conn.SendEnvelope(h.errEnvelope("not all players are ready"))
			return
		}
		h.Lobby.CancelCountdownUnsafe()
		h.Phase = PhaseCountdown
		h.Emit("phase_change", map[string]interface{}{"phase": "countdown"})

	default:
		log.Printf("hub %s: unknown lobby message type %q from user %s", h.ID, msg.Type, msg.UserID)
	}
}

// handleGameMsg routes game-phase messages to the game engine.
// Runs inside the hub's Run() goroutine — no game mutex needed here;
// game.HandlePlayerAction and game.ProcessSpecialAction manage their own locking.
func (h *Hub) handleGameMsg(msg ClientMsg) {
	if h.Game == nil {
		return
	}

	var raw struct {
		Card    map[string]interface{} `json:"card,omitempty"`
		Card1   map[string]interface{} `json:"card1,omitempty"`
		Card2   map[string]interface{} `json:"card2,omitempty"`
		Special string                 `json:"special,omitempty"`
		Payload map[string]interface{} `json:"payload,omitempty"`
	}
	if msg.Body != nil {
		_ = json.Unmarshal(msg.Body, &raw) // best-effort; handlers tolerate nil maps
	}

	switch msg.Type {
	case "action_draw_stockpile", "action_draw_discardpile",
		"action_discard", "action_replace", "action_cambia", "action_snap":
		gameAction := models.GameAction{
			ActionType: msg.Type,
			Payload:    make(map[string]interface{}),
		}
		if raw.Card != nil {
			gameAction.Payload = raw.Card
		} else if raw.Payload != nil {
			gameAction.Payload = raw.Payload
		}
		h.Game.HandlePlayerAction(msg.UserID, gameAction)

	case "action_special":
		h.Game.ProcessSpecialAction(msg.UserID, raw.Special, raw.Card1, raw.Card2)

	case "ping":
		if conn, ok := h.conns[msg.UserID]; ok {
			conn.SendEnvelope(Envelope{Seq: h.nextSeq(), Type: "pong"})
		}

	default:
		log.Printf("hub %s: unknown game message type %q from user %s", h.ID, msg.Type, msg.UserID)
	}
}

// handleSearchingMsg handles messages during the matchmaking search phase.
func (h *Hub) handleSearchingMsg(msg ClientMsg) {
	conn, ok := h.conns[msg.UserID]
	if !ok {
		return
	}
	switch msg.Type {
	case "cancel_search":
		if !conn.IsHost {
			conn.SendEnvelope(h.errEnvelope("only the host can cancel search"))
			return
		}
		h.Phase = PhaseOpen
		if h.Lobby != nil {
			h.Lobby.Mu.Lock()
			h.Lobby.Searching = false
			h.Lobby.Mu.Unlock()
		}
		h.Emit("phase_change", map[string]interface{}{"phase": "open"})
		h.Emit("search_status", map[string]interface{}{"searching": false})
	case "chat":
		h.handleLobbyMsg(msg)
	default:
		log.Printf("hub %s: ignoring message type %q during searching phase", h.ID, msg.Type)
	}
}

// handleMatchFound transitions the hub to ready-check when the matchmaker finds players.
func (h *Hub) handleMatchFound(players []MatchedPlayer) {
	h.Phase = PhaseReadyCheck
	h.Emit("phase_change", map[string]interface{}{"phase": "ready_check"})
	h.Emit("match_found", map[string]interface{}{
		"queue_id":     h.QueueID,
		"total_rounds": h.TotalRounds,
		"is_ranked":    h.IsRanked,
		"players":      players,
	})
}

// Matched returns a send-only channel that the matchmaker uses to deliver matched players.
func (h *Hub) Matched() chan<- []MatchedPlayer {
	return h.matched
}

// HandleRoundEnd is called when a game ends during a ranked multi-round match.
// It records scores, applies aggression subsidies, and either transitions to
// PhaseRoundEnd (more rounds remain) or PhaseMatchEnd (match complete).
func (h *Hub) HandleRoundEnd(scores map[uuid.UUID]int, cambiaCallerID uuid.UUID) {
	h.RoundsPlayed++

	// Store this round's scores.
	roundScores := make(map[uuid.UUID]int)
	for id, score := range scores {
		roundScores[id] = score
	}
	h.RoundHistory = append(h.RoundHistory, roundScores)

	// Build ordered slices for engine calls.
	playerIDs := make([]uuid.UUID, 0, len(scores))
	scoreList := make([]int, 0, len(scores))
	for id, score := range scores {
		playerIDs = append(playerIDs, id)
		scoreList = append(scoreList, score)
	}

	// Compute placements (0-indexed, lower is better).
	ranks := engine.RanksFromScores(scoreList, 3)
	placements := make([]int, len(ranks))
	for i, r := range ranks {
		placements[i] = r - 1
	}

	callerIdx := -1
	for i, id := range playerIDs {
		if id == cambiaCallerID {
			callerIdx = i
			break
		}
	}

	subsidies := engine.ComputeAggressionSubsidy(len(playerIDs), placements, callerIdx)

	// Apply subsidies and accumulate.
	for i, id := range playerIDs {
		roundScores[id] += subsidies[i]
		h.CumulativeScores[id] += roundScores[id]
	}

	// Rotate dealer seat.
	h.DealerSeatIdx = (h.DealerSeatIdx + 1) % len(playerIDs)

	if h.RoundsPlayed >= h.TotalRounds {
		h.Phase = PhaseMatchEnd
		h.Emit("phase_change", map[string]interface{}{"phase": "match_end"})
		h.Emit("match_end", map[string]interface{}{
			"round_scores":      roundScores,
			"cumulative_scores": h.CumulativeScores,
			"round_history":     h.RoundHistory,
			"subsidies":         buildSubsidyMap(playerIDs, subsidies),
			"final":             true,
		})
	} else {
		h.Phase = PhaseRoundEnd
		h.Emit("phase_change", map[string]interface{}{"phase": "round_end"})
		h.Emit("round_end", map[string]interface{}{
			"round":             h.RoundsPlayed,
			"total_rounds":      h.TotalRounds,
			"round_scores":      roundScores,
			"cumulative_scores": h.CumulativeScores,
			"subsidies":         buildSubsidyMap(playerIDs, subsidies),
		})
		// Auto-advance to next round after 10 seconds.
		go func() {
			time.Sleep(10 * time.Second)
			h.incoming <- ClientMsg{Type: "_start_next_round"}
		}()
	}
}

// startNextRound transitions to PhaseInGame for the next ranked round.
// Actual game creation is handled by the GameServer's game creation flow.
func (h *Hub) startNextRound() {
	h.Phase = PhaseInGame
	h.Emit("phase_change", map[string]interface{}{"phase": "in_game"})
	h.Emit("round_start", map[string]interface{}{
		"round":        h.RoundsPlayed + 1,
		"total_rounds": h.TotalRounds,
		"dealer_seat":  h.DealerSeatIdx,
	})
}

// buildSubsidyMap converts parallel playerID/subsidy slices to a string-keyed map.
func buildSubsidyMap(playerIDs []uuid.UUID, subsidies []int) map[string]int {
	m := make(map[string]int)
	for i, id := range playerIDs {
		m[id.String()] = subsidies[i]
	}
	return m
}

// Emit broadcasts an envelope to all connected clients.
func (h *Hub) Emit(eventType string, payload any) {
	raw, err := marshalPayload(payload)
	if err != nil {
		log.Printf("hub %s: Emit marshal error: %v", h.ID, err)
		return
	}
	env := Envelope{Seq: h.nextSeq(), Type: eventType, Payload: raw}
	data, err := json.Marshal(env)
	if err != nil {
		log.Printf("hub %s: Emit envelope marshal error: %v", h.ID, err)
		return
	}
	for _, conn := range h.conns {
		conn.Send(data)
	}
}

// EmitTo sends an envelope only to the connection matching userID.
func (h *Hub) EmitTo(userID uuid.UUID, eventType string, payload any) {
	conn, ok := h.conns[userID]
	if !ok {
		return
	}
	raw, err := marshalPayload(payload)
	if err != nil {
		log.Printf("hub %s: EmitTo marshal error: %v", h.ID, err)
		return
	}
	env := Envelope{Seq: h.nextSeq(), Type: eventType, Payload: raw}
	conn.SendEnvelope(env)
}

// buildLobbySnapshot builds a JSON-friendly lobby state payload for the given user.
// Must be called from within the Run() goroutine (no lock needed on hub state).
func (h *Hub) buildLobbySnapshot(forUserID uuid.UUID) map[string]interface{} {
	lob := h.Lobby
	lobbyStatus := lob.GetLobbyStatusPayloadUnsafe()

	// Enrich user entries with username from connections
	if users, ok := lobbyStatus["users"].([]map[string]interface{}); ok {
		for _, u := range users {
			if uidStr, ok := u["id"].(string); ok {
				uid, err := uuid.Parse(uidStr)
				if err == nil {
					if conn, exists := h.conns[uid]; exists {
						u["username"] = conn.Username
					}
				}
			}
		}
	}

	snapshot := map[string]interface{}{
		"lobby_id":     lob.ID.String(),
		"host_id":      lob.HostUserID.String(),
		"lobby_type":   lob.Type,
		"game_mode":    lob.GameMode,
		"in_game":      lob.InGame,
		"game_id":      lob.GameID.String(),
		"house_rules":  lob.HouseRules,
		"circuit":      lob.Circuit,
		"settings":     lob.LobbySettings,
		"lobby_status": lobbyStatus,
		"phase":        h.Phase.String(),
		"your_id":      forUserID.String(),
		"your_is_host": forUserID == lob.HostUserID,
	}

	if h.IsRanked && h.TotalRounds > 1 {
		snapshot["match_state"] = map[string]interface{}{
			"queue_id":          h.QueueID,
			"is_ranked":         h.IsRanked,
			"total_rounds":      h.TotalRounds,
			"current_round":     h.RoundsPlayed,
			"cumulative_scores": h.CumulativeScores,
			"round_history":     h.RoundHistory,
			"dealer_seat":       h.DealerSeatIdx,
		}
	}

	return snapshot
}

// sendLobbyState sends a full lobby_state snapshot to a single connection.
func (h *Hub) sendLobbyState(conn *Connection) {
	h.EmitTo(conn.UserID, "lobby_state", h.buildLobbySnapshot(conn.UserID))
}

// broadcastLobbyUpdate sends a lobby_state snapshot to all connected users.
func (h *Hub) broadcastLobbyUpdate() {
	for userID, conn := range h.conns {
		h.EmitTo(userID, "lobby_state", h.buildLobbySnapshot(conn.UserID))
	}
}

// sendSyncState sends a full state snapshot to a single user for desync recovery.
func (h *Hub) sendSyncState(userID uuid.UUID) {
	if _, ok := h.conns[userID]; !ok {
		return
	}
	payload := h.buildLobbySnapshot(userID)
	payload["seq"] = h.seq
	h.EmitTo(userID, "sync_state", payload)
}

// errEnvelope builds an error envelope (without consuming seq).
func (h *Hub) errEnvelope(msg string) Envelope {
	raw, _ := json.Marshal(map[string]string{"error": msg})
	return Envelope{Seq: h.nextSeq(), Type: "error", Payload: raw}
}

// Join sends a Connection to the hub's join channel.
func (h *Hub) Join(conn *Connection) {
	h.join <- conn
}

// Leave sends a userID to the hub's leave channel.
func (h *Hub) Leave(userID uuid.UUID) {
	h.leave <- userID
}

// Shutdown signals the hub to stop.
func (h *Hub) Shutdown() {
	close(h.shutdown)
}

// Incoming returns the channel for routing inbound ClientMsgs into the hub.
func (h *Hub) Incoming() chan<- ClientMsg {
	return h.incoming
}

// nextSeq atomically increments and returns the next sequence number.
// Only called from within the Run() goroutine, but atomic for safety.
func (h *Hub) nextSeq() uint64 {
	return atomic.AddUint64(&h.seq, 1)
}

// marshalPayload marshals a value to json.RawMessage.
func marshalPayload(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	return json.Marshal(v)
}
