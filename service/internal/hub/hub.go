// internal/hub/hub.go
package hub

import (
	"context"
	"encoding/json"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// defaultCountdownDuration is the fallback lobby -> game countdown length when the
// GameServer does not override CountdownDuration on the hub.
const defaultCountdownDuration = 3 * time.Second

// GameFactory builds and registers a CambiaGame for the given players, wiring emitter as
// the event sink. The returned game is registered but not begun (the hub calls BeginPreGame
// after routing is in place). Returns nil if the game could not be created (e.g. <2 players).
type GameFactory func(lob *lobby.Lobby, playerIDs []uuid.UUID, emitter game.Emitter) *game.CambiaGame

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
// Phase and lobby/game state are mutated only from the Run() select loop. The conns map is
// the exception: an in-progress CambiaGame emits events from its own timer goroutines
// (turn/pre-game/end timers) through the hub's Emitter, so conns is guarded by connsMu.
type Hub struct {
	ID    uuid.UUID
	Phase LobbyPhase

	Lobby *lobby.Lobby
	Game  *game.CambiaGame

	// CreateGame builds and registers the backing CambiaGame. Injected by the GameServer so
	// the hub stays decoupled from the game/lobby stores. Nil until wired.
	CreateGame GameFactory

	// CountdownDuration is the delay from countdown start to game creation.
	CountdownDuration time.Duration

	// seq is the monotonic per-hub sequence stamped on every server->client envelope. Invariant:
	// one logical broadcast consumes exactly one seq, stamped identically on every recipient's copy
	// (Emit does this inherently; broadcastLobbyUpdate does it via emitToWithSeq). dispatch() rejects
	// inbound msgs whose LastSeq < seq, so a per-recipient seq bump on a broadcast would leave every
	// recipient but the last one spuriously "behind" and drop their next action (cambia-502).
	// Mutated only via nextSeq() (atomic, for cross-goroutine game-timer emits); read unlocked in
	// dispatch(), which runs in the single Run() goroutine.
	seq uint64

	connsMu sync.RWMutex              // guards conns (cross-goroutine emits from game timers)
	conns   map[uuid.UUID]*Connection // userID → connection

	// Match state (ranked/circuit)
	QueueID     string
	IsRanked    bool
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
		ID:                lob.ID,
		Phase:             PhaseOpen,
		Lobby:             lob,
		CountdownDuration: defaultCountdownDuration,
		conns:             make(map[uuid.UUID]*Connection),
		CumulativeScores:  make(map[uuid.UUID]int),
		RoundHistory:      make([]map[uuid.UUID]int, 0),
		matched:           make(chan []MatchedPlayer, 1),
		join:              make(chan *Connection, 8),
		leave:             make(chan uuid.UUID, 8),
		incoming:          make(chan ClientMsg, 64),
		shutdown:          make(chan struct{}),
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
			h.connsMu.Lock()
			h.conns[conn.UserID] = conn
			h.connsMu.Unlock()
			h.sendLobbyState(conn)
			h.broadcastLobbyUpdate()
		case userID := <-h.leave:
			h.connsMu.Lock()
			conn, ok := h.conns[userID]
			if ok {
				delete(h.conns, userID)
			}
			remaining := len(h.conns)
			h.connsMu.Unlock()
			if ok {
				conn.Close()
			}
			if remaining == 0 {
				return // dissolve hub
			}
			h.broadcastLobbyUpdate()
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
	h.connsMu.Lock()
	conns := make([]*Connection, 0, len(h.conns))
	for _, conn := range h.conns {
		conns = append(conns, conn)
	}
	h.conns = make(map[uuid.UUID]*Connection)
	h.connsMu.Unlock()
	for _, conn := range conns {
		conn.Close()
	}
}

// dispatch routes a ClientMsg based on the current phase.
func (h *Hub) dispatch(msg ClientMsg) {
	// Synthetic internal messages (timer callbacks) carry no client seq and must be handled
	// before the sequence check, which would otherwise discard them as stale.
	switch msg.Type {
	case "_begin_game":
		if h.Phase == PhaseCountdown {
			h.beginGame()
		}
		return
	case "_start_next_round":
		if h.Phase == PhaseRoundEnd {
			h.startNextRound()
		}
		return
	case "_game_ended":
		if h.Phase == PhaseInGame {
			h.Phase = PhasePostGame
			h.Emit("phase_change", map[string]interface{}{"phase": "post_game"})
		}
		return
	}

	// Sequence check: if client is behind, send a sync snapshot and discard.
	if msg.LastSeq < h.seq && h.seq > 0 {
		h.sendSyncState(msg.UserID)
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
	conn := h.getConn(msg.UserID)
	if conn == nil {
		return
	}

	switch msg.Type {
	case "ready":
		// MarkUserReady returns true only when every joined user is ready and the lobby is
		// set to auto-start: that is the signal to begin the countdown to game creation.
		allReadyAutoStart := h.Lobby.MarkUserReady(msg.UserID)
		h.broadcastLobbyUpdate()
		if allReadyAutoStart {
			h.beginCountdown()
		}

	case "unready":
		h.Lobby.MarkUserUnready(msg.UserID)
		// Unreadying during the countdown aborts the pending start: the scheduled _begin_game
		// then no-ops because the phase is no longer countdown.
		if h.Phase == PhaseCountdown {
			h.Phase = PhaseOpen
			h.Emit("phase_change", map[string]interface{}{"phase": "open"})
		}
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
		if h.Phase == PhaseCountdown {
			conn.SendEnvelope(h.errEnvelope("game is already starting"))
			return
		}
		if !h.Lobby.AreAllReady() {
			conn.SendEnvelope(h.errEnvelope("not all players are ready"))
			return
		}
		h.beginCountdown()

	default:
		log.Printf("hub %s: unknown lobby message type %q from user %s", h.ID, msg.Type, msg.UserID)
	}
}

// handleGameMsg routes game-phase messages to the game engine. Player actions run here in the
// hub's Run() goroutine, while the game's own timer goroutines (turn/pre-game/end timers) also
// mutate its state; CambiaGame serializes both through its internal mutex (cambia-465), so the
// hub can call HandlePlayerAction/ProcessSpecialAction directly without holding a game lock.
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
		if conn := h.getConn(msg.UserID); conn != nil {
			conn.SendEnvelope(Envelope{Seq: h.nextSeq(), Type: "pong"})
		}

	default:
		log.Printf("hub %s: unknown game message type %q from user %s", h.ID, msg.Type, msg.UserID)
	}
}

// handleSearchingMsg handles messages during the matchmaking search phase.
func (h *Hub) handleSearchingMsg(msg ClientMsg) {
	conn := h.getConn(msg.UserID)
	if conn == nil {
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

// startNextRound transitions to PhaseInGame for the next ranked round and creates that
// round's game via the same path as round one. Reached only through _start_next_round, which
// HandleRoundEnd schedules; that round-end -> HandleRoundEnd link, cumulative scoring, dealer
// rotation and CircuitStore round tracking are the multi-round half that remains unwired
// (cambia-458): this creates the round's game so the mechanism is consistent once that half
// lands, but the scoring pipeline is not driven yet.
func (h *Hub) startNextRound() {
	h.Game = nil // clear the previous round's finished game before creating the next
	h.Phase = PhaseInGame
	h.Emit("phase_change", map[string]interface{}{"phase": "in_game"})
	h.Emit("round_start", map[string]interface{}{
		"round":        h.RoundsPlayed + 1,
		"total_rounds": h.TotalRounds,
		"dealer_seat":  h.DealerSeatIdx,
	})
	if pids := h.connectedPlayerIDs(); len(pids) >= 2 {
		h.createAndStartGame(pids)
	}
}

// beginCountdown enters PhaseCountdown and schedules game creation. Idempotent: a hub already
// counting down or in game is left untouched, so a duplicate ready/start_game cannot stack
// timers. Must run in the Run() goroutine.
func (h *Hub) beginCountdown() {
	if h.Phase == PhaseCountdown || h.Phase == PhaseInGame {
		return
	}
	h.Phase = PhaseCountdown
	seconds := int(h.CountdownDuration / time.Second)
	h.Emit("phase_change", map[string]interface{}{"phase": "countdown", "seconds": seconds})
	h.scheduleGameStart()
}

// scheduleGameStart fires a _begin_game message back into the Run() loop after the countdown
// so that game creation itself runs serialized in the hub goroutine (race-free), not in the
// timer goroutine. A shutdown mid-countdown drops the pending start.
func (h *Hub) scheduleGameStart() {
	d := h.CountdownDuration
	go func() {
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-timer.C:
			select {
			case h.incoming <- ClientMsg{Type: "_begin_game"}:
			case <-h.shutdown:
			}
		case <-h.shutdown:
		}
	}()
}

// beginGame creates and starts the round-one game once the countdown elapses. Guards against
// a second _begin_game (phase already advanced) and against every player leaving during the
// countdown (fewer than two connections aborts back to open). Must run in the Run() goroutine.
func (h *Hub) beginGame() {
	if h.Phase != PhaseCountdown || h.Game != nil {
		return // already started, or a stale timer fired
	}
	pids := h.connectedPlayerIDs()
	if len(pids) < 2 {
		h.abortToOpen("not enough connected players to start")
		return
	}
	h.Phase = PhaseInGame
	h.Emit("phase_change", map[string]interface{}{"phase": "in_game"})
	if !h.createAndStartGame(pids) {
		h.abortToOpen("game creation failed")
	}
}

// abortToOpen rolls the hub back to the open lobby phase after a failed start.
func (h *Hub) abortToOpen(reason string) {
	log.Printf("hub %s: aborting game start: %s", h.ID, reason)
	h.Game = nil
	h.Phase = PhaseOpen
	h.Emit("phase_change", map[string]interface{}{"phase": "open"})
	h.broadcastLobbyUpdate()
}

// createAndStartGame builds the game for playerIDs via the injected factory, routes it as
// h.Game, marks the lobby in-game, emits game_started to all participants, and begins the
// pre-game reveal. game_started precedes BeginPreGame so clients learn the game id before the
// first private card events arrive. Returns false if the factory is unset or returns nil.
// Must run in the Run() goroutine.
func (h *Hub) createAndStartGame(playerIDs []uuid.UUID) bool {
	if h.CreateGame == nil {
		log.Printf("hub %s: no game factory wired; cannot create game", h.ID)
		return false
	}
	if h.Game != nil {
		return false
	}
	g := h.CreateGame(h.Lobby, playerIDs, h)
	if g == nil {
		return false
	}
	h.Game = g

	h.Lobby.Mu.Lock()
	h.Lobby.InGame = true
	h.Lobby.GameID = g.ID
	h.Lobby.GameInstanceCreated = true
	h.Lobby.Mu.Unlock()

	playerStrs := make([]string, len(playerIDs))
	for i, id := range playerIDs {
		playerStrs[i] = id.String()
	}
	h.Emit("game_started", map[string]interface{}{
		"game_id": g.ID.String(),
		"players": playerStrs,
	})

	g.BeginPreGame()
	return true
}

// connectedPlayerIDs returns the user IDs of currently connected participants, host first and
// the remainder in a stable (UUID-sorted) order so seat assignment is deterministic. Using the
// live connection set (not lobby membership) means a player who disconnected during the
// countdown is naturally excluded.
func (h *Hub) connectedPlayerIDs() []uuid.UUID {
	ids := h.connUserIDs()

	h.Lobby.Mu.Lock()
	host := h.Lobby.HostUserID
	h.Lobby.Mu.Unlock()

	sort.Slice(ids, func(i, j int) bool {
		if ids[i] == host {
			return true
		}
		if ids[j] == host {
			return false
		}
		return ids[i].String() < ids[j].String()
	})
	return ids
}

// buildSubsidyMap converts parallel playerID/subsidy slices to a string-keyed map.
func buildSubsidyMap(playerIDs []uuid.UUID, subsidies []int) map[string]int {
	m := make(map[string]int)
	for i, id := range playerIDs {
		m[id.String()] = subsidies[i]
	}
	return m
}

// Emit broadcasts an envelope to all connected clients. Safe to call from the game's timer
// goroutines: the connection set is snapshotted under connsMu before sending.
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
	h.connsMu.RLock()
	conns := make([]*Connection, 0, len(h.conns))
	for _, conn := range h.conns {
		conns = append(conns, conn)
	}
	h.connsMu.RUnlock()
	for _, conn := range conns {
		conn.Send(data)
	}
}

// EmitTo sends an envelope only to the connection matching userID, consuming one seq.
func (h *Hub) EmitTo(userID uuid.UUID, eventType string, payload any) {
	h.emitToWithSeq(userID, h.nextSeq(), eventType, payload)
}

// emitToWithSeq sends an envelope stamped with the caller-supplied seq to userID's connection.
// Broadcasts that fan a single logical event out per-recipient (buildLobbySnapshot is tailored per
// user, so they cannot share one Emit) call this with one nextSeq() value across every recipient,
// keeping the one-seq-per-broadcast invariant documented on Hub.seq (cambia-502).
func (h *Hub) emitToWithSeq(userID uuid.UUID, seq uint64, eventType string, payload any) {
	conn := h.getConn(userID)
	if conn == nil {
		return
	}
	raw, err := marshalPayload(payload)
	if err != nil {
		log.Printf("hub %s: EmitTo marshal error: %v", h.ID, err)
		return
	}
	conn.SendEnvelope(Envelope{Seq: seq, Type: eventType, Payload: raw})
}

// getConn returns the connection for userID, or nil. Acquires connsMu (read).
func (h *Hub) getConn(userID uuid.UUID) *Connection {
	h.connsMu.RLock()
	defer h.connsMu.RUnlock()
	return h.conns[userID]
}

// connUserIDs returns a snapshot of the currently connected user IDs. Acquires connsMu (read).
func (h *Hub) connUserIDs() []uuid.UUID {
	h.connsMu.RLock()
	defer h.connsMu.RUnlock()
	ids := make([]uuid.UUID, 0, len(h.conns))
	for uid := range h.conns {
		ids = append(ids, uid)
	}
	return ids
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
					if conn := h.getConn(uid); conn != nil {
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

// broadcastLobbyUpdate sends a per-user lobby_state snapshot to all connected users. All copies of
// this one logical broadcast share a single seq (see the Hub.seq invariant): otherwise the
// per-recipient seq bump would leave every recipient but the last one behind h.seq, and dispatch()
// would drop their next inbound action as stale (cambia-502).
func (h *Hub) broadcastLobbyUpdate() {
	userIDs := h.connUserIDs()
	if len(userIDs) == 0 {
		return
	}
	seq := h.nextSeq()
	for _, userID := range userIDs {
		h.emitToWithSeq(userID, seq, "lobby_state", h.buildLobbySnapshot(userID))
	}
}

// sendSyncState sends a full state snapshot to a single user for desync recovery.
func (h *Hub) sendSyncState(userID uuid.UUID) {
	if h.getConn(userID) == nil {
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

// NotifyGameEnded queues the post-game phase transition (PhaseInGame -> PhasePostGame) onto the
// hub's incoming channel. The game engine's OnGameEnd callback (see handlers.attachOnGameEnd) can
// run on a goroutine other than the hub's Run() loop (e.g. a turn-timeout timer), so it must not
// mutate h.Phase directly; this mirrors the _start_next_round synthetic-message pattern to route
// the mutation through dispatch() inside Run() instead (cambia-510).
func (h *Hub) NotifyGameEnded() {
	h.incoming <- ClientMsg{Type: "_game_ended"}
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
