// internal/hub/hub.go
package hub

import (
	"context"
	"encoding/json"
	"log"
	"runtime/debug"
	"sort"
	"strings"
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

// defaultPostGameResultsDuration is how long a hub stays in PhasePostGame showing results before
// it returns the lobby to PhaseOpen, used when the GameServer does not override
// PostGameDuration on the hub. It matches the between-rounds interval in HandleRoundEnd.
const defaultPostGameResultsDuration = 10 * time.Second

// defaultIdleTTL is the longest an idle window may run, used when the GameServer does not
// override IdleTTL. What carries a table that all dropped mid-game is not this length but the
// reap-time exemption in handleIdleReap, which never reaps a hub whose game is still running; with
// both defaults in force EmptyIdleTTL is the window that actually reaps (cambia-884).
const defaultIdleTTL = 45 * time.Minute

// defaultEmptyIdleTTL is the same window for a hub with no game in progress, used when the
// GameServer does not override EmptyIdleTTL. A pre-game or post-game lobby whose members have
// all closed their tabs holds nothing worth waiting out the long window for, and until
// cambia-884 it sat in the public list for the best part of an hour.
const defaultEmptyIdleTTL = 5 * time.Minute

// GameFactory builds and registers a CambiaGame for the given players, wiring emitter as
// the event sink. usernames carries the authenticated (or guest-generated) username already
// known for each connected player id, sourced from the hub's own connections (cambia-877): the
// factory reads it instead of re-fetching from the database, since createAndStartGame runs on
// the hub's Run goroutine and a DB round trip there would stall the entire lobby's event loop.
// The returned game is registered but not begun (the hub calls BeginPreGame after routing is in
// place). Returns nil if the game could not be created (e.g. <2 players).
type GameFactory func(lob *lobby.Lobby, playerIDs []uuid.UUID, usernames map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame

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
//
// IsHost is false for every seat of a matchmade match and has been since cambia-1087: the lobby
// holding the match is system-hosted, so the field survives only for the clients that read it
// off match_found. It is not a permission - handleLobbyMsg re-reads the lobby's own HostUserID
// on every host-gated frame.
type MatchedPlayer struct {
	UserID   uuid.UUID
	Username string
	IsHost   bool
}

// MatchNotice tells a searching hub what the matchmaker did with its players.
//
// LobbyID is the lobby the match is played in. The hub whose own id it carries hosts the match
// and runs the ready check; every other hub in the group forwards the notice to its clients so
// they can move there, because a match is formed out of one lobby per party and only one of
// those lobbies can hold the game (cambia-933).
type MatchNotice struct {
	LobbyID uuid.UUID
	Players []MatchedPlayer
}

// SearchState is a requested matchmaking phase change: entering the queue carries that queue's
// parameters, leaving it carries nothing but the flag. See SetSearchState.
type SearchState struct {
	Searching   bool
	QueueID     string
	IsRanked    bool
	TotalRounds int
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

	// PostGameDuration is how long PhasePostGame holds the results screen before the hub
	// returns itself to PhaseOpen (see returnToLobby), for a table that does not close the
	// screen itself first (return_to_lobby, cambia-1238).
	PostGameDuration time.Duration

	// IdleTTL bounds the idle window: it is the reap deadline while EmptyIdleTTL is unset or
	// longer, and the ceiling on it otherwise. Zero disables reaping, whatever EmptyIdleTTL says.
	// It is not what holds a live game's lobby open; handleIdleReap's exemption is (cambia-884).
	IdleTTL time.Duration

	// EmptyIdleTTL is the shorter idle window from cambia-884. With no game in progress it is the
	// reap deadline: a lobby waiting to start, or one whose game has finished, is reclaimed on it
	// rather than sitting out the long TTL. While a game is in progress it is how often the reap
	// decision is reconsidered. Zero, or any value above IdleTTL, leaves IdleTTL in force.
	EmptyIdleTTL time.Duration

	// OnIdle tears the lobby down when the idle window elapses. Injected by the owner and
	// pointed at the same teardown the last deliberate leave runs (cambia-807), so a lobby
	// everyone abandoned is released exactly like one everyone left. Nil disables reaping.
	OnIdle func(lobbyID uuid.UUID)

	// seq is the monotonic per-hub sequence stamped on every server->client envelope. Invariant:
	// only a broadcast consumes a seq, exactly one, stamped identically on every recipient's copy
	// (Emit does this inherently; broadcastLobbyUpdate does it via emitToWithSeq). dispatch()
	// rejects inbound msgs whose LastSeq < seq, so a seq is only usable as a staleness gate if
	// every connected client observes it. A per-recipient bump on a broadcast leaves every
	// recipient but the last one spuriously "behind" and drops their next action (cambia-502);
	// a bump on a frame sent to a single connection (EmitTo, sendSyncState, errEnvelope, pong)
	// leaves every OTHER client behind a number no frame will ever carry to them, so their next
	// action is dropped and only a retry after the repair gets through (cambia-878). Private
	// frames therefore stamp the current seq without consuming one.
	// Mutated only via nextSeq() (atomic, for cross-goroutine game-timer emits); read unlocked in
	// dispatch(), which runs in the single Run() goroutine.
	seq uint64

	connsMu sync.RWMutex              // guards conns (cross-goroutine emits from game timers)
	conns   map[uuid.UUID]*Connection // userID → connection

	// terminalMu guards terminalType/terminalPayload, the last results frame this hub broadcast
	// (game_results, or match_end for a ranked circuit). Emit records it from whichever goroutine
	// ended the game, and notePlayerReconnected replays it to a connection that arrives after the
	// fact, so a player who reloads into a finished game gets the scores instead of an empty
	// "Game over" (cambia-955). Cleared when the next game starts.
	terminalMu      sync.RWMutex
	terminalType    string
	terminalPayload json.RawMessage

	// alive reports whether Run() is still serving this hub. Set when Run starts, cleared
	// when it returns (Shutdown or ctx cancel). Callers outside the Run goroutine need this
	// to tell a hub that can still answer a reconnect from one that can no longer be resumed.
	alive atomic.Bool

	// OnDissolve is called once, from Run()'s exit path, so the owner can drop this hub from
	// its registry. A hub that has stopped serving must not stay discoverable: a WebSocket
	// routed to it would be accepted and then never answered (cambia-808).
	OnDissolve func(hubID uuid.UUID)

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
	matched     chan MatchNotice // matchmaker sends the formed match here
	searchState chan SearchState // the search endpoints send phase changes here

	// idleTimer/idleGen own the idle window. Both belong to the Run() goroutine. Stopping a
	// time.Timer does not un-fire one that already ran, so every fire carries the generation it
	// was armed with and handleIdleReap discards a superseded one: that is what makes a
	// reconnect genuinely reset the clock rather than leave an older timer to fire on the
	// lobby's new occupants.
	idleTimer *time.Timer
	idleGen   uint64

	// postGameGen numbers the results screens, on the same reasoning as idleGen: a timer cannot
	// be un-fired once it has run, and a results screen can now be closed before its own timer
	// fires (cambia-1238). Every arm takes the next number and the queued reset carries it, so a
	// reset that outlived the screen it was armed for is dropped rather than applied to whichever
	// screen is up when it lands. Belongs to the Run() goroutine, like the phase it guards.
	postGameGen uint64

	// idleArmed is the window armIdleReap actually armed the current idleTimer with. handleIdleReap
	// logs this rather than recomputing idleWindow() at fire time: EmptyIdleTTL/IdleTTL are not
	// reassigned once a hub is running in production, but a test - or a future reconfiguration path
	// - that changes them between arm and fire would otherwise leave the reap log naming a window
	// that was never the one actually running (cambia-887 F2).
	idleArmed time.Duration

	// Channels for Run() select loop
	join     chan *Connection
	leave    chan uuid.UUID
	incoming chan ClientMsg
	idleReap chan uint64
	shutdown chan struct{}

	// shutdownOnce guards close(shutdown): Run()'s exit path closes the channel to release
	// everything parked on it, and an owner tearing the hub down closes it too.
	shutdownOnce sync.Once
}

// NewHub creates a new hub for the given lobby.
func NewHub(lob *lobby.Lobby) *Hub {
	return &Hub{
		ID:                lob.ID,
		Phase:             PhaseOpen,
		Lobby:             lob,
		CountdownDuration: defaultCountdownDuration,
		PostGameDuration:  defaultPostGameResultsDuration,
		IdleTTL:           defaultIdleTTL,
		EmptyIdleTTL:      defaultEmptyIdleTTL,
		conns:             make(map[uuid.UUID]*Connection),
		CumulativeScores:  make(map[uuid.UUID]int),
		RoundHistory:      make([]map[uuid.UUID]int, 0),
		matched:           make(chan MatchNotice, 1),
		searchState:       make(chan SearchState, 4),
		join:              make(chan *Connection, 8),
		leave:             make(chan uuid.UUID, 8),
		incoming:          make(chan ClientMsg, 64),
		idleReap:          make(chan uint64, 1),
		shutdown:          make(chan struct{}),
	}
}

// Run is the hub's main goroutine. It serializes all state access.
// Call this in its own goroutine.
//
// The loop outlives its connections. A hub used to return once its last connection left, which
// left it registered but unable to answer anything: the next WebSocket to that lobby was
// accepted and then hung forever (cambia-808). It also threw away state only the hub holds -
// the phase, the routing to a live CambiaGame, and a ranked match's cumulative scores - which
// a restarted hub could not reconstruct. The hub now runs for as long as its lobby exists and
// stops only when the owner tears the lobby down (Shutdown), the idle window elapses with
// nothing connected (cambia-836), or the context is cancelled.
func (h *Hub) Run(ctx context.Context) {
	h.alive.Store(true)
	defer h.exit()
	// A lobby whose members never connect at all is idle from birth, so the first window opens
	// here rather than waiting for a departure.
	h.armIdleReap()
	for {
		if h.runStep(ctx) {
			return
		}
	}
}

// runStep waits for the hub's next event and handles it, reporting whether the loop must end.
//
// Every handler runs under a recover boundary. A panic in one hub's message handling used to
// unwind past Run and kill the process, so a single game's bug took down every other lobby's
// game with it: a 4-player table's opponent-targeting ability panicked the shared service and
// ended every concurrent match (cambia-946). A recovered panic is logged with its stack, this
// hub tells its own clients the session cannot continue, and only this hub dissolves, through
// Run's normal exit path (exit -> OnDissolve -> cleanup).
//
// The recovery leaves no lock held: the game's own mutex is taken and released by defer inside
// CambiaGame's public entry points (HandlePlayerAction, ProcessSpecialAction), so the unwind
// releases it before it ever reaches this boundary, and the hub's own state belongs to this
// goroutine, which is exiting.
func (h *Hub) runStep(ctx context.Context) (stop bool) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		log.Printf("hub %s: panic while handling an event: %v\n%s", h.ID, r, debug.Stack())
		h.reportFatalPanic()
		stop = true
	}()

	select {
	case <-ctx.Done():
		return true
	case conn := <-h.join:
		h.cancelIdleReap()
		h.connsMu.Lock()
		h.conns[conn.UserID] = conn
		h.connsMu.Unlock()
		h.sendLobbyState(conn)
		h.broadcastLobbyUpdate()
		h.notePlayerReconnected(conn.UserID)
	case userID := <-h.leave:
		// Connection-level only: the user keeps their lobby membership, because this fires
		// for a dropped socket just as it does for a deliberate leave (which releases
		// membership over HTTP before signalling the hub). See lobby.RemoveUser.
		h.connsMu.Lock()
		conn, ok := h.conns[userID]
		if ok {
			delete(h.conns, userID)
		}
		h.connsMu.Unlock()
		if ok {
			conn.Close()
		}
		h.broadcastLobbyUpdate()
		h.notePlayerDisconnected(userID)
		if h.connCount() == 0 {
			h.armIdleReap()
		}
	case msg := <-h.incoming:
		h.dispatch(msg)
	case notice := <-h.matched:
		h.handleMatchFound(notice)
	case state := <-h.searchState:
		h.applySearchState(state)
	case gen := <-h.idleReap:
		h.handleIdleReap(gen)
	case <-h.shutdown:
		return true
	}
	return false
}

// reportFatalPanic tells this hub's own clients that their session cannot continue, just before
// the hub dissolves. Best-effort by construction: it runs on the way out of a panic, so it takes
// its own recover boundary rather than risking a second unwind, and it touches nothing but the
// connection set (the game's state is whatever the panic left behind, and is not read here).
func (h *Hub) reportFatalPanic() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("hub %s: panic while reporting a fatal error: %v", h.ID, r)
		}
	}()
	h.Emit("error", map[string]interface{}{
		"code":    "hub_fatal",
		"message": "This game hit an internal error and has ended. Other games are unaffected.",
		"fatal":   true,
	})
}

// notePlayerDisconnected tells a running game that one of its players lost their socket.
// game.HandleDisconnect had no callers at all before this: a mid-game drop never marked the
// player disconnected, so the ForfeitOnDisconnect house rule was dead, circuit grace timers never
// armed, and a turn timer running out was the only thing that ever noticed (cambia-837).
//
// This is a connection-level event and stays one. Lobby membership is untouched, so the player
// keeps their seat and their resume entry and can reconnect into the game: the transient-versus-
// deliberate distinction cambia-807 drew. A deliberate leave cannot reach here mid-game anyway,
// since the leave endpoint refuses one.
//
// Run() goroutine only, and safe there: h.Phase and h.Game belong to it, and HandleDisconnect
// takes the game's own mutex, which nothing on this path holds (connsMu is released above).
func (h *Hub) notePlayerDisconnected(userID uuid.UUID) {
	if h.Phase != PhaseInGame || h.Game == nil || !h.Game.HasPlayer(userID) {
		return
	}
	h.Game.HandleDisconnect(userID)
}

// notePlayerReconnected is the counterpart: it restores the seat, sends the returning player the
// sync state they need to draw the table again, reschedules their turn timer if the game was
// waiting on them, and cancels a circuit grace timer before the AI takes over.
//
// The game is handed no socket of its own. models.Player.Conn is only ever written and never
// read - every event a game sends goes out through the hub Emitter - and passing the raw
// WebSocket would give HandleReconnect's not-a-player path the power to close a connection the
// hub is still serving.
// Since cambia-955 it also answers the other reconnect a hub has to serve: one into a game that
// is already over. The hub holds its finished game for the results interval (cambia-793), so a
// player who reloads in that window is handed the final table and the results frame again. The
// lobby snapshot they get on join carries the phase but no scores, and nothing else re-sends
// them, which is why that reload landed on a "Game over" screen with no winner and no scores.
func (h *Hub) notePlayerReconnected(userID uuid.UUID) {
	if h.Game == nil || !h.Game.HasPlayer(userID) {
		return
	}
	// Phase is not a gate here the way it is on the disconnect side: the reconnect is safe in
	// every phase that still holds a game. In PhaseInGame it cancels a pending forfeit and
	// restores the seat; in PhasePostGame/PhaseMatchEnd the game is finished, so HandleReconnect
	// only re-sends its terminal snapshot (its turn-timer and grace branches are all guarded by
	// Started/GameOver).
	h.Game.HandleReconnect(userID, nil)
	// The phase can still read in_game for the moment between the game ending and the queued
	// _game_ended landing (NotifyGameEnded routes it through this same loop), so the game's own
	// verdict is what decides, with the post-game phases covering a hub whose game was cleared.
	if h.Game.IsGameOver() || h.Phase == PhasePostGame || h.Phase == PhaseMatchEnd {
		h.resendTerminal(userID)
	}
}

// rememberTerminal stores the last results broadcast so a late joiner can be given it.
func (h *Hub) rememberTerminal(eventType string, payload json.RawMessage) {
	h.terminalMu.Lock()
	h.terminalType = eventType
	h.terminalPayload = payload
	h.terminalMu.Unlock()
}

// forgetTerminal drops the stored results. A new game's results are the only ones worth
// re-sending, so the previous round's are cleared the moment one starts.
func (h *Hub) forgetTerminal() {
	h.terminalMu.Lock()
	h.terminalType = ""
	h.terminalPayload = nil
	h.terminalMu.Unlock()
}

// resendTerminal replays the stored results frame to one connection. Private, so it stamps the
// current seq and consumes none (see the Hub.seq invariant).
func (h *Hub) resendTerminal(userID uuid.UUID) {
	h.terminalMu.RLock()
	eventType, payload := h.terminalType, h.terminalPayload
	h.terminalMu.RUnlock()
	if eventType == "" || payload == nil {
		return
	}
	conn := h.getConn(userID)
	if conn == nil {
		return
	}
	conn.SendEnvelope(Envelope{Seq: atomic.LoadUint64(&h.seq), Type: eventType, Payload: payload})
}

// armIdleReap opens a fresh idle window. Called from Run() only, so idleTimer and idleGen need no
// lock. The timer posts back into the Run() loop rather than acting on its own, the same shape as
// scheduleGameStart and schedulePostGameReset (cambia-793): the reap decision reads h.Phase, which
// belongs to the Run goroutine, and a shutdown drops the pending fire.
func (h *Hub) armIdleReap() {
	h.cancelIdleReap()
	if h.IdleTTL <= 0 || h.OnIdle == nil {
		return
	}
	gen := h.idleGen
	window := h.idleWindow()
	h.idleArmed = window
	h.idleTimer = time.AfterFunc(window, func() {
		select {
		case h.idleReap <- gen:
		case <-h.shutdown:
		}
	})
}

// idleWindow returns how long the next window runs: the shorter of the two TTLs, since
// EmptyIdleTTL never lengthens one. A deployment that lowers IdleTTL below it, and every test that
// shortens only IdleTTL, gets IdleTTL.
//
// The window carries two meanings and the fire decides which, not the arm (cambia-884):
//
//   - No game in progress: it is the reap deadline. A lobby waiting to start, or one whose game
//     has finished, holds nothing but membership its members can re-establish by opening a new
//     lobby, and until cambia-884 it sat in the public list for the best part of an hour.
//   - Game in progress: it is a re-check interval, because handleIdleReap declines to reap a live
//     game whatever the window says. Arming the long TTL here instead would put the reap decision
//     out of reach for the rest of it, and since nothing re-arms when a game ends, a table that
//     all dropped mid-game would keep its lobby for the full long window after the game was over:
//     exactly the dead lobby the short window exists to clear.
func (h *Hub) idleWindow() time.Duration {
	if h.EmptyIdleTTL > 0 && h.EmptyIdleTTL < h.IdleTTL {
		return h.EmptyIdleTTL
	}
	return h.IdleTTL
}

// cancelIdleReap closes the current window. The generation bump is the part that matters: Stop()
// cannot un-fire a timer that already ran, so a fire already in flight is invalidated here and
// discarded by handleIdleReap instead of reaping a lobby somebody just reconnected to.
func (h *Hub) cancelIdleReap() {
	if h.idleTimer != nil {
		h.idleTimer.Stop()
		h.idleTimer = nil
	}
	h.idleGen++
}

// handleIdleReap runs the reap decision in the Run() goroutine. A lobby whose members all closed
// their tabs keeps its membership - only a deliberate leave releases that (cambia-807) - so
// nothing else would ever reclaim it, and post-cambia-808 its hub goroutine parks for the life of
// the process. Reaping goes through OnIdle, the same whole-lobby teardown the last leave runs.
//
// A running game is never idle, whatever the socket count: its turn timers and the forfeit rule
// (cambia-837) still have to reach their own end. That exemption, rather than the length of
// IdleTTL, is what carries a live game across a disconnect; the lobby is reconsidered one window
// later, and that window is the shorter of the two, so the reap follows shortly after the game
// ends rather than at the end of a grace nothing needs any more.
func (h *Hub) handleIdleReap(gen uint64) {
	if gen != h.idleGen || h.OnIdle == nil {
		return
	}
	h.idleTimer = nil
	if h.connCount() > 0 {
		return // somebody is here; their departure opens the next window
	}
	if h.inGame() {
		h.armIdleReap()
		return
	}
	// At least one window, and more where an earlier fire found a game still running.
	log.Printf("hub %s: no connections for at least %s and no game in progress; reaping lobby.", h.ID, h.idleArmed)
	h.OnIdle(h.ID)
}

// inGame reports whether a game is under way, from the hub's phase and the lobby's own flag. The
// two are cleared at different moments - OnGameEnd clears the lobby while the hub moves to
// PhasePostGame - so the exemption holds while either says a game is live.
func (h *Hub) inGame() bool {
	if h.Phase == PhaseInGame || h.Phase == PhaseRoundEnd {
		return true
	}
	if h.Lobby == nil {
		return false
	}
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	return h.Lobby.InGame
}

// exit runs on Run()'s way out, in an order the rest of the package depends on: the hub stops
// being discoverable, then publishes that it is no longer alive, then releases everything
// parked on it. Join() reads the shutdown channel after handing over a connection, so closing
// it before cleanup drains the join queue is what guarantees every accepted socket is either
// served or closed.
func (h *Hub) exit() {
	if h.OnDissolve != nil {
		h.OnDissolve(h.ID)
	}
	h.alive.Store(false)
	h.Shutdown()
	h.cleanup()
}

// cleanup closes every connection the hub still owns, including connections that were handed
// to Join but never registered, and releases the idle window.
func (h *Hub) cleanup() {
	h.cancelIdleReap()

	h.connsMu.Lock()
	conns := make([]*Connection, 0, len(h.conns))
	for _, conn := range h.conns {
		conns = append(conns, conn)
	}
	h.conns = make(map[uuid.UUID]*Connection)
	h.connsMu.Unlock()

drained:
	for {
		select {
		case conn := <-h.join:
			conns = append(conns, conn)
		default:
			break drained
		}
	}

	for _, conn := range conns {
		conn.Close()
	}
}

// dispatch routes a ClientMsg based on the current phase.
func (h *Hub) dispatch(msg ClientMsg) {
	// Synthetic internal messages (timer callbacks) carry no client seq and must be handled
	// before the sequence check, which would otherwise discard them as stale.
	//
	// They are the hub's own, and a socket may not name one. ReadPump takes the type straight
	// off the frame and stamps the message with the connection and the authenticated user it
	// arrived on, while a timer builds one with neither, so an underscore type carrying either
	// was typed by a client. Each of these runs a phase transition with no check on the sender:
	// a client naming _return_to_lobby would walk straight past the host gate the results exit
	// is admitted through below (cambia-1238).
	if strings.HasPrefix(msg.Type, "_") && (msg.ConnID != uuid.Nil || msg.UserID != uuid.Nil) {
		log.Printf("hub %s: refusing internal message type %q sent by user %s", h.ID, msg.Type, msg.UserID)
		return
	}

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
			h.schedulePostGameReset()
		}
		return
	case "_return_to_lobby":
		// The results timer firing. It runs only for the screen that armed it: the phase check
		// alone passes a reset armed by an earlier game, which the host's exit left queued
		// behind it, and a hub showing a SECOND game's results is in PhasePostGame just the same
		// (cambia-1238).
		if h.Phase == PhasePostGame && msg.gen == h.postGameGen {
			h.returnToLobby()
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
		// Chat stays open over the results; game actions are ignored, since the game they name
		// is over. return_to_lobby is the table's own way off the results screen: without it the
		// only exit is PostGameDuration elapsing, and PhaseMatchEnd arms no timer at all, so a
		// finished match had no exit (cambia-1238).
		switch msg.Type {
		case "chat":
			h.handleLobbyMsg(msg)
		case "return_to_lobby":
			h.handlePostGameExit(msg)
		}
	}
}

// handleLobbyMsg handles lobby-phase messages (ready, chat, rules, etc.). Runs inside the hub's
// Run() goroutine, so hub state needs no lock, but the lobby does: its Users, ReadyStates and
// rules are shared with the HTTP handlers that join, leave and search on their own goroutines,
// and the lobby's *Unsafe methods assume the caller holds Lobby.Mu.
func (h *Hub) handleLobbyMsg(msg ClientMsg) {
	conn := h.getConn(msg.UserID)
	if conn == nil {
		return
	}

	switch msg.Type {
	case "ready":
		// MarkUserReadyUnsafe returns true only when every joined user is ready and the lobby is
		// set to auto-start: that is the signal to begin the countdown to game creation.
		h.Lobby.Mu.Lock()
		allReadyAutoStart := h.Lobby.MarkUserReadyUnsafe(msg.UserID)
		h.Lobby.Mu.Unlock()
		h.broadcastLobbyUpdate()
		if allReadyAutoStart {
			h.beginCountdown()
		}

	case "unready":
		h.Lobby.Mu.Lock()
		h.Lobby.MarkUserUnreadyUnsafe(msg.UserID)
		h.Lobby.Mu.Unlock()
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
		h.Lobby.Mu.Lock()
		h.Lobby.InviteUser(targetID)
		h.Lobby.Mu.Unlock()

	// No leave here on purpose. Leaving a lobby releases membership, and membership must not
	// be released by anything a lost socket can also trigger; a client that sent a leave frame
	// and closed its socket in the same breath would also race its own disconnect through this
	// loop. POST /lobby/{id}/leave owns the deliberate leave, on the same surface that granted
	// membership in the first place (cambia-807).

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
		// The rules lock is checked before the host gate, not after: a matchmade lobby has no
		// player host at all (cambia-1087), so ordering the checks the other way answered every
		// sender with "only the host can update rules" - true, and a description of a role
		// nobody in that lobby can hold, rather than the actual reason the rules will not move.
		//
		// A ranked or matchmade lobby has its rules fixed by the queue it entered, not by its
		// host: the queue config is what the matchmaker used to pair the players and what the
		// game inherits at creation (NewCambiaGameFromLobby), so a mid-search or post-match rule
		// edit would leave the two sides of the match disagreeing about what they agreed to play
		// (cambia-966). Type and Mode are only ever assigned at creation (CreateLobbyHandler), so
		// this is a stable read for as long as the lobby exists. RulesLockedUnsafe carries the
		// rule, and buildLobbySnapshot ships its answer as rules_locked, so the refusal here and
		// the sheet the client renders cannot disagree (cambia-1099 K2).
		h.Lobby.Mu.Lock()
		locked := h.Lobby.RulesLockedUnsafe()
		h.Lobby.Mu.Unlock()
		if locked {
			conn.SendEnvelope(h.errEnvelope("the rules for this match are locked by its queue and cannot be changed"))
			return
		}
		if !h.isHost(msg.UserID) {
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
		h.Lobby.Mu.Lock()
		err := h.Lobby.UpdateUnsafe(payload.Rules)
		h.Lobby.Mu.Unlock()
		if err != nil {
			// The reason travels with the refusal. Every error UpdateUnsafe returns is a
			// statement about the message that was sent - an unknown preset, a house rule out of
			// range, a ruleset that seats fewer players than the lobby has - and a host told only
			// "failed" has nothing to change before trying again (cambia-1099). The generic text
			// stays as the prefix, since the payload carries one field and the client renders it
			// as it arrives.
			log.Printf("hub %s: UpdateUnsafe error: %v", h.ID, err)
			conn.SendEnvelope(h.errEnvelope("failed to apply rule updates: " + err.Error()))
			return
		}

		// Everyone at the table plays by these rules, so everyone is told. Nothing broadcast after
		// an accepted edit, so every seat but the host's went on rendering the old sheet - house
		// rules, auto-start, preset name, and the game mode a preset now fixes - until a reload or
		// some unrelated roster change refreshed it, while chat on the same socket arrived live
		// (cambia-1099 Q10). The same per-user lobby_state join, leave and match formation send, so
		// there is one shape a client renders a rule sheet from however it moved.
		h.broadcastLobbyUpdate()

	case "start_game":
		// A matchmade lobby starts itself. Its host role belongs to the system (cambia-1087),
		// so there is no player to force the start and none is needed: auto-start is on for
		// every lobby the create handler builds and a queue-backed one cannot turn it off, so
		// the last ready seat begins the countdown through the "ready" case above. Answered
		// with what will start the game rather than with the host gate's wording, which would
		// point at a role nobody holds.
		if h.Lobby != nil && h.Lobby.SystemHosted() {
			conn.SendEnvelope(h.errEnvelope("this match starts on its own once every player is ready"))
			return
		}
		if !h.isHost(msg.UserID) {
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
		"action_discard", "action_replace", "action_cambia", "action_snap", "action_snap_move":
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
		// A pong answers one connection, so it stamps the current seq without consuming one:
		// a keepalive that moved the sequence would stale every other client (cambia-878).
		if conn := h.getConn(msg.UserID); conn != nil {
			conn.SendEnvelope(Envelope{Seq: atomic.LoadUint64(&h.seq), Type: "pong"})
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
		// Still the party leader's call, and deliberately so: the system host lands at match
		// formation (handlers.HandleMatchFormed), which also moves the hub out of
		// PhaseSearching, so a lobby that can reach this case always still has its human leader
		// (cambia-1087). Pulling a party out of the queue is the one host power a quick play
		// party keeps, because the party is theirs until a match exists.
		if !h.isHost(msg.UserID) {
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

// handleMatchFound reacts to a formed match. The hub hosting it runs the ready check; a hub
// whose players are moving to another lobby stops searching and reopens, since it is no longer
// part of any queue. Either way its clients get one match_found naming the lobby the match is
// played in, which is the only thing that tells a non-hosting party where to go (cambia-933).
func (h *Hub) handleMatchFound(notice MatchNotice) {
	destination := notice.LobbyID
	if destination == uuid.Nil {
		destination = h.ID
	}
	hosting := destination == h.ID

	if hosting {
		h.Phase = PhaseReadyCheck
	} else {
		h.Phase = PhaseOpen
	}
	// The lobby has left the queue whichever side of the match it is on; leaving Searching set
	// would have it advertise a search the matchmaker has already resolved.
	if h.Lobby != nil {
		h.Lobby.Mu.Lock()
		h.Lobby.Searching = false
		h.Lobby.Mu.Unlock()
	}

	h.Emit("phase_change", map[string]interface{}{"phase": h.Phase.String()})
	h.Emit("match_found", map[string]interface{}{
		"lobby_id":     destination.String(),
		"queue_id":     h.QueueID,
		"total_rounds": h.TotalRounds,
		"is_ranked":    h.IsRanked,
		"players":      notice.Players,
	})

	// The hosting lobby just lost its player host to the system (cambia-1087), and match_found
	// does not carry lobby state. Without this the party leader who queued keeps a Host badge
	// and a Start game button from the lobby_state they were sent on connect, both of which the
	// service now refuses, until some unrelated event happens to rebroadcast the roster.
	if hosting {
		h.broadcastLobbyUpdate()
	}
}

// Matched returns a send-only channel that the matchmaker uses to deliver a formed match.
func (h *Hub) Matched() chan<- MatchNotice {
	return h.matched
}

// SetSearchState asks the hub to enter or leave the searching phase, carrying the parameters of
// the queue the lobby entered. Callable from any goroutine: the change is applied in the Run
// loop, where Phase, QueueID, IsRanked and TotalRounds belong. The search endpoints used to
// assign those four fields directly from their HTTP goroutine, which raced the Run loop's own
// reads of Phase the moment anyone was connected (cambia-933).
//
// Never blocks: a hub that has stopped serving, or one somehow behind on the queue, drops the
// notice with a log rather than parking the request handler.
func (h *Hub) SetSearchState(state SearchState) {
	select {
	case h.searchState <- state:
	default:
		log.Printf("hub %s: dropped a search state change (searching=%v, queue=%q)", h.ID, state.Searching, state.QueueID)
	}
}

// applySearchState performs the phase change SetSearchState asked for. Run() goroutine only.
func (h *Hub) applySearchState(state SearchState) {
	if state.Searching {
		h.Phase = PhaseSearching
		h.QueueID = state.QueueID
		h.IsRanked = state.IsRanked
		h.TotalRounds = state.TotalRounds
		h.Emit("phase_change", map[string]interface{}{"phase": "searching"})
		h.Emit("search_status", map[string]interface{}{"searching": true, "queue_id": state.QueueID})
		return
	}
	h.Phase = PhaseOpen
	h.Emit("phase_change", map[string]interface{}{"phase": "open"})
	h.Emit("search_status", map[string]interface{}{"searching": false})
}

// UsernameOf returns the display name a connected user authenticated with, or "" when that user
// has no live connection to this hub. Read under connsMu, so it is safe from any goroutine.
func (h *Hub) UsernameOf(userID uuid.UUID) string {
	h.connsMu.RLock()
	defer h.connsMu.RUnlock()
	if conn, ok := h.conns[userID]; ok {
		return conn.Username
	}
	return ""
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

// schedulePostGameReset fires a _return_to_lobby message back into the Run() loop after the
// results interval, so the reset itself runs serialized in the hub goroutine like every other
// phase transition (same pattern as scheduleGameStart). A shutdown drops the pending reset.
// Armed only on the PhaseInGame -> PhasePostGame edge, so one game end arms one timer.
//
// The message carries the generation this arm took, and dispatch fires only on a match: the
// screen it belongs to can be closed by its host before the timer runs (cambia-1238), and the
// queued reset outlives that exit. Must run in the Run() goroutine.
func (h *Hub) schedulePostGameReset() {
	d := h.PostGameDuration
	if d <= 0 {
		d = defaultPostGameResultsDuration
	}
	h.postGameGen++
	gen := h.postGameGen
	go func() {
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-timer.C:
			select {
			case h.incoming <- ClientMsg{Type: "_return_to_lobby", gen: gen}:
			case <-h.shutdown:
			}
		case <-h.shutdown:
		}
	}()
}

// handlePostGameExit runs the client-side half of the post-game return: the table closing the
// results screen instead of sitting out PostGameDuration. It runs the same returnToLobby the
// timer runs, so a lobby that left the results early is in exactly the state one that waited is.
//
// Host-gated, read live off the lobby like every other host-gated action (isHost): the exit
// closes the screen for the whole table, and one seat does not take everybody else's results
// away. A matchmade lobby has no player host to gate on - its host role belongs to the queue and
// your_is_host is false in every seat (cambia-1087) - so the role there widens to the lobby's own
// seats. Gating those tables on the role would leave every seat of a finished match stuck on the
// results screen, which is the bug this control exists to fix, and their seats are the players
// who just played the match. Must run in the Run() goroutine.
func (h *Hub) handlePostGameExit(msg ClientMsg) {
	conn := h.getConn(msg.UserID)
	if conn == nil {
		return
	}
	if !h.mayReturnToLobby(msg.UserID) {
		conn.SendEnvelope(h.errEnvelope("only the host can close the results and reopen the lobby"))
		return
	}
	h.returnToLobby()
}

// mayReturnToLobby reports whether userID may close the results screen: the host, or any seated
// player of a system-hosted lobby (see handlePostGameExit for why the fallback is there). The
// membership read is what keeps the fallback a rule rather than no gate at all - a socket is not
// a seat.
func (h *Hub) mayReturnToLobby(userID uuid.UUID) bool {
	if h.Lobby == nil {
		return false
	}
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	if h.Lobby.SystemHostedUnsafe() {
		return h.Lobby.Users[userID]
	}
	return h.Lobby.HostUserID == userID
}

// returnToLobby ends the post-game results phase: it drops the finished game, clears the lobby's
// in-game flags and ready states, and puts the hub back in PhaseOpen so the existing
// ready -> countdown -> beginGame path can create the next game (cambia-793; PhasePostGame was
// terminal and h.Game was never cleared, so createAndStartGame's guard blocked every later start).
//
// Reached from the results timer and from the table's own exit (handlePostGameExit), which is why
// it is written to be run once per results screen: the caller decides whether the screen is still
// the one it belongs to. Runs for PhaseMatchEnd as well, so a finished match's seats are not
// stranded (nothing arms a timer for that phase), but it touches no circuit or cumulative state:
// RoundsPlayed, CumulativeScores and RoundHistory stay where the match left them, since what
// becomes of a finished ranked match's lobby is the unratified half of cambia-466. Must run in
// the Run() goroutine.
func (h *Hub) returnToLobby() {
	h.Game = nil
	h.Phase = PhaseOpen

	// The lobby is also mutated by the game's OnGameEnd callback on a foreign goroutine, so its
	// own mutex guards this reset (same discipline as createAndStartGame).
	h.Lobby.Mu.Lock()
	h.Lobby.InGame = false
	h.Lobby.GameID = uuid.Nil
	h.Lobby.GameInstanceCreated = false
	for uid := range h.Lobby.ReadyStates {
		h.Lobby.ReadyStates[uid] = false
	}
	h.Lobby.Mu.Unlock()

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

	// Every id in playerIDs is a currently connected participant (connectedPlayerIDs reads the
	// live connection set), so its Username - populated at connect time from the authenticated
	// user (see ws.go's HubWSHandler) - is already known here without touching the database.
	usernames := make(map[uuid.UUID]string, len(playerIDs))
	for _, uid := range playerIDs {
		if conn := h.getConn(uid); conn != nil {
			usernames[uid] = conn.Username
		}
	}
	g := h.CreateGame(h.Lobby, playerIDs, usernames, h)
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
	// A results broadcast is the one frame a client can miss and never recover from any later
	// state: the lobby snapshot carries the phase but no scores, and nothing re-sends it. Keep it
	// so a reconnect can be answered with it (cambia-955); game_started drops it again.
	switch eventType {
	case "game_results", "match_end":
		h.rememberTerminal(eventType, raw)
	case "game_started":
		h.forgetTerminal()
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

// EmitTo sends an envelope to the connection matching userID alone, stamped with the current seq
// and consuming none.
//
// This is the whole private-event surface: the game engine's per-player frames all arrive here
// through the Emitter (private_draw_stockpile, private_snap_penalty,
// private_special_action_success/_fail, private_initial_cards, private_sync_state), as does the
// lobby snapshot a joining connection gets. Consuming a seq for a frame only one connection
// receives leaves every other client behind h.seq holding a number no frame will ever carry to
// them, and dispatch() answers their next action with a sync_state repair and drops it: a failed
// snap cost the other player their next draw, an ability reveal cost them theirs, and only a
// retry after the repair got through (cambia-878). A private frame conveys no shared ordering, so
// it stamps h.seq as it stands, exactly as sendSyncState and errEnvelope do.
//
// A logical broadcast that has to be built per recipient still consumes exactly one seq for all
// of them: those call emitToWithSeq directly with a single nextSeq() (broadcastLobbyUpdate).
func (h *Hub) EmitTo(userID uuid.UUID, eventType string, payload any) {
	h.emitToWithSeq(userID, atomic.LoadUint64(&h.seq), eventType, payload)
}

// emitToWithSeq sends an envelope stamped with the caller-supplied seq to userID's connection.
// Broadcasts that fan a single logical event out per-recipient (buildLobbySnapshot is tailored per
// user, so they cannot share one Emit) call this with one nextSeq() value across every recipient,
// keeping the one-seq-per-broadcast invariant documented on Hub.seq (cambia-502). Genuinely
// private frames call it with the current seq and consume none (cambia-878); EmitTo is that path.
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

// isHost reports whether userID holds the host role right now. Derived from the lobby at
// permission-check time rather than cached on the connection: the role migrates to a remaining
// member when a host leaves (cambia-835), and a flag stamped on the socket at accept time would
// leave the promoted host refused and the departed one still authorised.
func (h *Hub) isHost(userID uuid.UUID) bool {
	if h.Lobby == nil {
		return false
	}
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	return h.Lobby.HostUserID == userID
}

// getConn returns the connection for userID, or nil. Acquires connsMu (read).
func (h *Hub) getConn(userID uuid.UUID) *Connection {
	h.connsMu.RLock()
	defer h.connsMu.RUnlock()
	return h.conns[userID]
}

// connCount returns how many connections the hub currently holds. Acquires connsMu (read).
func (h *Hub) connCount() int {
	h.connsMu.RLock()
	defer h.connsMu.RUnlock()
	return len(h.conns)
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

// buildLobbySnapshot builds a JSON-friendly lobby state payload for the given user. Hub fields
// are read unlocked because this runs in the Run() goroutine that owns them; the lobby is not
// hub-owned, so every lobby field is read under its lock. The HTTP handlers write that state
// from their own goroutines - joining, leaving (cambia-807) and searching all mutate the same
// maps - and iterating lob.Users unlocked against a concurrent delete is a fatal map fault, not
// a stale read. The connection lookups that enrich the roster run after the lobby lock is
// released, so the two locks are taken in sequence and never nested.
func (h *Hub) buildLobbySnapshot(forUserID uuid.UUID) map[string]interface{} {
	lob := h.Lobby

	lob.Mu.Lock()
	lobbyStatus := lob.GetLobbyStatusPayloadUnsafe()
	snapshot := map[string]interface{}{
		"lobby_id":    lob.ID.String(),
		"host_id":     lob.HostUserID.String(),
		"lobby_type":  lob.Type,
		"game_mode":   lob.GameMode,
		"in_game":     lob.InGame,
		"game_id":     lob.GameID.String(),
		"house_rules": lob.HouseRules,
		// preset_id names the ruleset house_rules came from, or is empty for a sheet that is
		// nobody's preset. Sent because the rules cannot answer it: every ranked queue plays the
		// one ruleset MATCHMAKING.md 5.2 fixes, so a client matching house_rules against the
		// preset list picks whichever preset is listed first and shows a lobby created from H2H
		// Rapid as H2H Quick (cambia-1123). Always present, empty string included: an absent key
		// and a recorded empty id read the same to a client, and only one of them means "the
		// service has no answer".
		"preset_id": lob.PresetID,
		// rules_locked is the service's own answer to "may these rules still change", computed
		// where update_rules refuses (Lobby.RulesLockedUnsafe) rather than left to the client to
		// re-derive. It cannot be re-derived from this payload anyway: a public or private lobby
		// that queued its party into a ranked queue is locked by its mode, and mode is not sent,
		// so the sheet offered its host controls that fail on Save (cambia-1099 K2).
		"rules_locked": lob.RulesLockedUnsafe(),
		"circuit":      lob.Circuit,
		"settings":     lob.LobbySettings,
		"lobby_status": lobbyStatus,
		"phase":        h.Phase.String(),
		"your_id":      forUserID.String(),
		"your_is_host": forUserID == lob.HostUserID,
		// system_host says the nil host_id is deliberate rather than missing: a matchmade
		// lobby's host role belongs to the queue, so every seat reads your_is_host false and
		// lobby_status marks no seat is_host (cambia-1087). A client that only checked
		// your_is_host could not tell "somebody else hosts" from "nobody does", and the two
		// call for different copy.
		"system_host": lob.SystemHostedUnsafe(),
	}
	lob.Mu.Unlock()

	// Enrich user entries with username from connections. lobbyStatus is freshly built above
	// and owned by this call, so it is safe to fill in after the lobby lock is released.
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
	// Stamp the current seq WITHOUT consuming one: this is a private repair
	// message, and the whole point is to catch the recipient up to h.seq. If
	// it bumped the sequence, every repair would re-stale every other client,
	// and two clients failing the dispatch() staleness gate concurrently would
	// livelock feeding each other sync_states forever (observed live
	// 2026-08-27: a fresh 2-client lobby drove seq past 19000). Same
	// one-seq-per-broadcast reasoning as cambia-502.
	seq := atomic.LoadUint64(&h.seq)
	payload := h.buildLobbySnapshot(userID)
	payload["seq"] = seq
	h.emitToWithSeq(userID, seq, "sync_state", payload)
}

// errEnvelope builds an error envelope. Like sendSyncState, it is a private
// per-connection reply and must not consume a seq: a rejected action that
// advanced the global sequence would mark every other client stale.
func (h *Hub) errEnvelope(msg string) Envelope {
	raw, _ := json.Marshal(map[string]string{"error": msg})
	return Envelope{Seq: atomic.LoadUint64(&h.seq), Type: "error", Payload: raw}
}

// Join hands a Connection to the hub's Run loop, or closes it if this hub has stopped. Never
// parks a connection on a hub that will not serve it: that hang is the failure cambia-808 is
// about. The post-handover check covers the narrow window where Run exits between the send and
// the return - exit() closes the shutdown channel before cleanup drains the join queue, so a
// connection this sees as shut down is either already closed by cleanup or closed here, and
// closing a connection twice is harmless.
func (h *Hub) Join(conn *Connection) {
	select {
	case h.join <- conn:
	case <-h.shutdown:
		conn.Close()
		return
	}
	select {
	case <-h.shutdown:
		conn.Close()
	default:
	}
}

// Leave asks the hub to drop userID's connection. It does not touch lobby membership: this
// fires for a dropped socket as well as a deliberate leave. Returns as soon as the hub has
// stopped, so a WebSocket handler goroutine never parks on a hub that is gone.
func (h *Hub) Leave(userID uuid.UUID) {
	select {
	case h.leave <- userID:
	case <-h.shutdown:
	}
}

// Shutdown signals the hub to stop. Idempotent: the owner tearing a lobby down and Run()'s own
// exit path both close the channel, and a second close would panic.
func (h *Hub) Shutdown() {
	h.shutdownOnce.Do(func() { close(h.shutdown) })
}

// Alive reports whether the Run() loop is still serving this hub. Safe to call from any
// goroutine. False for a hub whose Run() has returned (its lobby was torn down) and for one
// that was never started, i.e. a hub that would accept a WebSocket without ever answering it.
func (h *Hub) Alive() bool {
	return h.alive.Load()
}

// NotifyGameEnded queues the post-game phase transition (PhaseInGame -> PhasePostGame) onto the
// hub's incoming channel. The game engine's OnGameEnd callback (see handlers.attachOnGameEnd) can
// run on a goroutine other than the hub's Run() loop (e.g. a turn-timeout timer), so it must not
// mutate h.Phase directly; this mirrors the _start_next_round synthetic-message pattern to route
// the mutation through dispatch() inside Run() instead (cambia-510).
//
// The send never blocks the caller. Since cambia-837 the caller can be the Run goroutine itself:
// a drop under ForfeitOnDisconnect ends the game inside the leave case, and a blocking send onto
// a full incoming queue would park the only goroutine that drains it. A full queue hands the send
// to a goroutine that can park harmlessly, so the transition is deferred rather than dropped, and
// released by shutdown if the hub stops first.
func (h *Hub) NotifyGameEnded() {
	select {
	case h.incoming <- ClientMsg{Type: "_game_ended"}:
	default:
		go func() {
			select {
			case h.incoming <- ClientMsg{Type: "_game_ended"}:
			case <-h.shutdown:
			}
		}()
	}
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
