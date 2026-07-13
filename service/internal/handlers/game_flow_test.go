// internal/handlers/game_flow_test.go
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// wsEnvelope mirrors the server->client hub.Envelope wire frame for test decoding.
type wsEnvelope struct {
	Seq     uint64          `json:"seq"`
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// wsTestClient is a minimal cambia hub WS client: it tracks the highest seq seen (so
// outbound messages echo an up-to-date last_seq and survive the hub's staleness check)
// and records every frame for assertion.
type wsTestClient struct {
	t      *testing.T
	conn   *websocket.Conn
	ctx    context.Context
	mu     sync.Mutex
	frames []wsEnvelope
	maxSeq uint64
}

func dialWSClient(t *testing.T, ctx context.Context, serverURL, lobbyID, token string) *wsTestClient {
	t.Helper()
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/ws/" + lobbyID
	hdr := http.Header{}
	hdr.Set("Cookie", "auth_token="+token)
	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPHeader: hdr})
	if err != nil {
		t.Fatalf("dial ws for lobby %s: %v", lobbyID, err)
	}
	c := &wsTestClient{t: t, conn: conn, ctx: ctx}
	go c.readLoop()
	return c
}

func (c *wsTestClient) readLoop() {
	for {
		_, data, err := c.conn.Read(c.ctx)
		if err != nil {
			return
		}
		var env wsEnvelope
		if err := json.Unmarshal(data, &env); err != nil {
			continue
		}
		c.mu.Lock()
		c.frames = append(c.frames, env)
		if env.Seq > c.maxSeq {
			c.maxSeq = env.Seq
		}
		c.mu.Unlock()
	}
}

// settle waits for in-flight frames to arrive so maxSeq reflects the hub's current seq
// before the next send. The test is the only driver, so nothing advances the hub seq
// during this window.
func (c *wsTestClient) settle() {
	time.Sleep(200 * time.Millisecond)
}

// send transmits a typed message echoing the freshest seq observed.
func (c *wsTestClient) send(msgType string) {
	c.mu.Lock()
	seq := c.maxSeq
	c.mu.Unlock()
	frame, _ := json.Marshal(map[string]interface{}{"type": msgType, "last_seq": seq})
	if err := c.conn.Write(c.ctx, websocket.MessageText, frame); err != nil {
		c.t.Fatalf("ws write %q: %v", msgType, err)
	}
}

// sendReliable sends a message and, if the hub bounces it with a sync_state (the client's
// echoed seq was behind the global counter, which private frames to other users advance),
// updates its seq from that snapshot and retries. This mirrors how a real client recovers
// from the staleness gate. Returns once the message is accepted (no fresh sync_state bounce).
func (c *wsTestClient) sendReliable(msgType string) {
	for attempt := 0; attempt < 8; attempt++ {
		before := c.countType("sync_state")
		c.send(msgType)
		deadline := time.Now().Add(400 * time.Millisecond)
		bounced := false
		for time.Now().Before(deadline) {
			if c.countType("sync_state") > before {
				bounced = true // the readLoop has already advanced maxSeq from the snapshot
				break
			}
			time.Sleep(15 * time.Millisecond)
		}
		if !bounced {
			return
		}
	}
	c.t.Fatalf("message %q never accepted after retries", msgType)
}

// waitForType polls recorded frames for one of the given type, up to timeout.
func (c *wsTestClient) waitForType(msgType string, timeout time.Duration) *wsEnvelope {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		for i := range c.frames {
			if c.frames[i].Type == msgType {
				env := c.frames[i]
				c.mu.Unlock()
				return &env
			}
		}
		c.mu.Unlock()
		time.Sleep(25 * time.Millisecond)
	}
	return nil
}

// countType returns how many recorded frames have the given type.
func (c *wsTestClient) countType(msgType string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for i := range c.frames {
		if c.frames[i].Type == msgType {
			n++
		}
	}
	return n
}

// createPublicLobby drives POST /lobby/create and returns the new lobby id.
func createPublicLobby(t *testing.T, gs *GameServer, hostToken string) uuid.UUID {
	t.Helper()
	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBufferString(`{"type":"public","gameMode":"head_to_head"}`))
	req.Header.Set("Cookie", "auth_token="+hostToken)
	w := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create lobby: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var created struct {
		ID uuid.UUID `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created lobby: %v", err)
	}
	return created.ID
}

func (c *wsTestClient) close() {
	c.conn.Close(websocket.StatusNormalClosure, "done")
}

// TestHubLobbyToGameFlow drives the full path over the real WS handlers: two players join a
// public lobby, both ready, the auto-start countdown elapses, and the hub creates an
// engine-backed CambiaGame. It asserts both players receive game_started, the game is
// registered in the GameStore, and the pre-game reveal (private_initial_cards) reaches
// clients through the hub emitter (proving Emitter + h.Game routing are wired).
func TestHubLobbyToGameFlow(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 50 * time.Millisecond // keep the test fast

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Two distinct authenticated users (host = lobby creator).
	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	// Host creates a public lobby (public bypasses the private-invite WS gate).
	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	defer p2.close()

	// Let both joins broadcast so each client's seq is current.
	host.settle()
	p2.settle()

	// Host readies (not all ready yet), then player 2 readies -> auto-start countdown.
	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	// Both players must receive game_started carrying the game id.
	hostStarted := host.waitForType("game_started", 5*time.Second)
	if hostStarted == nil {
		t.Fatalf("host never received game_started")
	}
	p2Started := p2.waitForType("game_started", 5*time.Second)
	if p2Started == nil {
		t.Fatalf("player 2 never received game_started")
	}

	var startedPayload struct {
		GameID  string   `json:"game_id"`
		Players []string `json:"players"`
	}
	if err := json.Unmarshal(hostStarted.Payload, &startedPayload); err != nil {
		t.Fatalf("decode game_started payload: %v", err)
	}
	if startedPayload.GameID == "" {
		t.Fatalf("game_started carried no game_id")
	}
	if len(startedPayload.Players) != 2 {
		t.Fatalf("game_started players = %d, want 2", len(startedPayload.Players))
	}

	// A CambiaGame must exist in the GameStore for this lobby, matching the announced id.
	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered in GameStore for lobby %s", lobbyID)
	}
	if g.ID.String() != startedPayload.GameID {
		t.Fatalf("GameStore game id %s != game_started id %s", g.ID, startedPayload.GameID)
	}
	if len(g.Players) != 2 {
		t.Fatalf("game has %d players, want 2", len(g.Players))
	}

	// The pre-game reveal proves the game's Emitter is the hub and BeginPreGame ran: each
	// player should receive their private initial cards over the same WS connection.
	if host.waitForType("private_initial_cards", 5*time.Second) == nil {
		t.Fatalf("host never received private_initial_cards (emitter not wired?)")
	}
	if p2.waitForType("private_initial_cards", 5*time.Second) == nil {
		t.Fatalf("player 2 never received private_initial_cards (emitter not wired?)")
	}
}

// TestHubRepeatedStartIsIdempotent verifies that extra start_game requests once a game is
// starting are rejected and never create a second game: exactly one game_started is emitted.
func TestHubRepeatedStartIsIdempotent(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 1500 * time.Millisecond // wide window so start_game lands mid-countdown

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	defer p2.close()
	host.settle()
	p2.settle()

	// Both ready -> auto-start countdown begins (1s window). Host then tries start_game
	// mid-countdown; it must be rejected as "already starting", not stack a second game.
	host.sendReliable("ready")
	host.settle()
	p2.sendReliable("ready")
	host.settle()
	host.sendReliable("start_game") // lands during the countdown -> rejected

	rejected := host.waitForType("error", 3*time.Second)
	if rejected == nil {
		t.Fatalf("host never received an error for start_game during countdown")
	}
	var errPayload struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal(rejected.Payload, &errPayload)
	if !strings.Contains(errPayload.Error, "already starting") {
		t.Fatalf("start_game rejection = %q, want 'already starting'", errPayload.Error)
	}

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("host never received game_started")
	}
	host.settle()

	// A game exists, and exactly one game_started was emitted (no duplicate creation).
	if gs.GameStore.GetGameByLobbyID(lobUUID) == nil {
		t.Fatalf("no game registered for lobby")
	}
	if n := host.countType("game_started"); n != 1 {
		t.Fatalf("expected exactly 1 game_started, got %d (duplicate game creation)", n)
	}
}

// TestHubDisconnectDuringCountdownAborts verifies that if a player drops during the
// countdown, game creation aborts (fewer than two live connections) rather than seating a
// ghost: no game is registered and the hub returns to the open phase.
func TestHubDisconnectDuringCountdownAborts(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 800 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	host.settle()
	p2.settle()

	// Both ready -> countdown starts, then player 2 drops before it elapses.
	host.sendReliable("ready")
	host.settle()
	p2.sendReliable("ready")
	p2.close() // disconnect mid-countdown

	// After the countdown would have fired, no game must exist for the lobby.
	if got := host.waitForType("game_started", 1500*time.Millisecond); got != nil {
		t.Fatalf("game_started emitted despite a mid-countdown disconnect")
	}
	if g := gs.GameStore.GetGameByLobbyID(lobUUID); g != nil {
		t.Fatalf("game %s was created despite only one live connection", g.ID)
	}
}
