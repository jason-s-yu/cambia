package engine

import (
	"testing"
	"unsafe"
)

// TestNewGameDeck verifies NewGame creates the correct number of unique cards.
func TestNewGameDeck(t *testing.T) {
	hr := DefaultHouseRules()
	g := NewGame(42, hr)

	wantLen := uint8(52 + hr.NumJokers)
	if g.StockLen != wantLen {
		t.Fatalf("StockLen = %d, want %d", g.StockLen, wantLen)
	}

	seen := make(map[Card]bool)
	for i := uint8(0); i < g.StockLen; i++ {
		c := g.Stockpile[i]
		if c == EmptyCard {
			t.Errorf("Stockpile[%d] is EmptyCard", i)
			continue
		}
		if seen[c] {
			t.Errorf("Duplicate card at index %d: suit=%d rank=%d", i, c.Suit(), c.Rank())
		}
		seen[c] = true
	}

	if len(seen) != int(wantLen) {
		t.Errorf("got %d unique cards, want %d", len(seen), wantLen)
	}

	// Verify 4x13 standard cards + NumJokers jokers.
	standardCount := 0
	jokerCount := 0
	for c := range seen {
		if c.Rank() == RankJoker {
			jokerCount++
		} else {
			standardCount++
		}
	}
	if standardCount != 52 {
		t.Errorf("standardCount = %d, want 52", standardCount)
	}
	if jokerCount != int(hr.NumJokers) {
		t.Errorf("jokerCount = %d, want %d", jokerCount, hr.NumJokers)
	}
}

// TestNewGameNoJokers verifies NewGame with NumJokers=0 creates a 52-card deck.
func TestNewGameNoJokers(t *testing.T) {
	hr := DefaultHouseRules()
	hr.NumJokers = 0
	g := NewGame(42, hr)

	if g.StockLen != 52 {
		t.Fatalf("StockLen = %d, want 52", g.StockLen)
	}

	for i := uint8(0); i < g.StockLen; i++ {
		c := g.Stockpile[i]
		if c.Rank() == RankJoker {
			t.Errorf("found joker at index %d in no-joker deck", i)
		}
	}
}

// TestNewGameSeedZero verifies that seed 0 is corrected to 1.
func TestNewGameSeedZero(t *testing.T) {
	g := NewGame(0, DefaultHouseRules())
	if g.RNG == 0 {
		t.Error("RNG is 0 after seed=0; expected correction to 1")
	}
	if g.RNG != 1 {
		t.Errorf("RNG = %d, want 1 for seed=0", g.RNG)
	}
}

// TestDealCardCounts verifies card counts after Deal.
func TestDealCardCounts(t *testing.T) {
	hr := DefaultHouseRules() // CardsPerPlayer = 4, NumPlayers = 2
	g := NewGame(42, hr)
	g.Deal()

	n := g.NumActivePlayers()

	// Active players should have exactly CardsPerPlayer cards.
	for p := uint8(0); p < n; p++ {
		if g.Players[p].HandLen != hr.CardsPerPlayer {
			t.Errorf("player %d HandLen = %d, want %d", p, g.Players[p].HandLen, hr.CardsPerPlayer)
		}
	}
	// Inactive slots should be empty.
	for p := n; p < MaxPlayers; p++ {
		if g.Players[p].HandLen != 0 {
			t.Errorf("inactive player %d HandLen = %d, want 0", p, g.Players[p].HandLen)
		}
	}

	// Stockpile should have shrunk by CardsPerPlayer*NumPlayers + 1 (flipped to discard).
	dealtCards := uint8(hr.CardsPerPlayer) * n
	expectedStockLen := uint8(52+hr.NumJokers) - dealtCards - 1
	if g.StockLen != expectedStockLen {
		t.Errorf("StockLen = %d, want %d", g.StockLen, expectedStockLen)
	}

	// Discard pile should have exactly 1 card.
	if g.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d, want 1", g.DiscardLen)
	}
}

// TestDealDeterministic verifies that the same seed produces identical results.
func TestDealDeterministic(t *testing.T) {
	hr := DefaultHouseRules()

	g1 := NewGame(99, hr)
	g1.Deal()

	g2 := NewGame(99, hr)
	g2.Deal()

	// Same starting player.
	if g1.CurrentPlayer != g2.CurrentPlayer {
		t.Errorf("CurrentPlayer: %d vs %d", g1.CurrentPlayer, g2.CurrentPlayer)
	}

	// Same discard top.
	if g1.DiscardTop() != g2.DiscardTop() {
		t.Errorf("DiscardTop: %v vs %v", g1.DiscardTop(), g2.DiscardTop())
	}

	// Same hands for each active player.
	n := g1.NumActivePlayers()
	for p := uint8(0); p < n; p++ {
		for c := uint8(0); c < g1.Players[p].HandLen; c++ {
			if g1.Players[p].Hand[c] != g2.Players[p].Hand[c] {
				t.Errorf("player %d card %d: %v vs %v", p, c, g1.Players[p].Hand[c], g2.Players[p].Hand[c])
			}
		}
	}
}

// TestDealDifferentSeeds verifies that different seeds produce different hands.
func TestDealDifferentSeeds(t *testing.T) {
	hr := DefaultHouseRules()

	g1 := NewGame(1, hr)
	g1.Deal()

	g2 := NewGame(2, hr)
	g2.Deal()

	// With high probability, at least one card should differ.
	allSame := true
	n := g1.NumActivePlayers()
	for p := uint8(0); p < n; p++ {
		for c := uint8(0); c < g1.Players[p].HandLen; c++ {
			if g1.Players[p].Hand[c] != g2.Players[p].Hand[c] {
				allSame = false
				break
			}
		}
	}
	if allSame {
		t.Error("seeds 1 and 2 produced identical hands (extremely unlikely if RNG is working)")
	}
}

// TestActingPlayer verifies ActingPlayer returns CurrentPlayer when no snap/pending.
func TestActingPlayer(t *testing.T) {
	g := NewGame(42, DefaultHouseRules())
	g.Deal()

	// No snap, no pending — should return CurrentPlayer.
	if g.Snap.Active || g.Pending.Type != PendingNone {
		t.Skip("unexpected initial snap/pending state")
	}

	got := g.ActingPlayer()
	if got != g.CurrentPlayer {
		t.Errorf("ActingPlayer() = %d, want CurrentPlayer=%d", got, g.CurrentPlayer)
	}

	// Test with pending action set.
	g.Pending.Type = PendingPeekOwn
	g.Pending.PlayerID = 1
	if g.ActingPlayer() != 1 {
		t.Errorf("ActingPlayer() with pending = %d, want 1", g.ActingPlayer())
	}
	g.Pending.Type = PendingNone

	// Test with snap active.
	g.Snap.Active = true
	g.Snap.Snappers[0] = 0
	g.Snap.CurrentSnapperIdx = 0
	if g.ActingPlayer() != 0 {
		t.Errorf("ActingPlayer() with snap = %d, want 0", g.ActingPlayer())
	}
	g.Snap.Active = false
}

// TestDiscardTop verifies DiscardTop returns valid card after deal and EmptyCard when empty.
func TestDiscardTop(t *testing.T) {
	// Empty discard.
	g := NewGame(42, DefaultHouseRules())
	if g.DiscardTop() != EmptyCard {
		t.Errorf("empty DiscardTop() = %v, want EmptyCard", g.DiscardTop())
	}

	// After deal, should be a valid card.
	g.Deal()
	top := g.DiscardTop()
	if top == EmptyCard {
		t.Error("DiscardTop() after Deal returned EmptyCard")
	}
	// Should be a real card (not 0xFF).
	if top.Suit() > SuitBlackJoker {
		t.Errorf("DiscardTop() has invalid suit %d", top.Suit())
	}
}

// TestOpponentOf verifies OpponentOf returns the correct opponent index.
func TestOpponentOf(t *testing.T) {
	g := NewGame(1, DefaultHouseRules())
	if g.OpponentOf(0) != 1 {
		t.Errorf("OpponentOf(0) = %d, want 1", g.OpponentOf(0))
	}
	if g.OpponentOf(1) != 0 {
		t.Errorf("OpponentOf(1) = %d, want 0", g.OpponentOf(1))
	}
}

// TestGameStateSize verifies sizeof(GameState) ≤ 300 bytes.
func TestGameStateSize(t *testing.T) {
	size := unsafe.Sizeof(GameState{})
	const maxSize = 300
	if size > maxSize {
		t.Errorf("sizeof(GameState) = %d, want ≤ %d", size, maxSize)
	}
	t.Logf("sizeof(GameState) = %d bytes", size)
}

// TestPlayerStateSize verifies sizeof(PlayerState) == 16 bytes.
func TestPlayerStateSize(t *testing.T) {
	size := unsafe.Sizeof(PlayerState{})
	const wantSize = 16
	if size != wantSize {
		t.Errorf("sizeof(PlayerState) = %d, want %d", size, wantSize)
	}
}

// TestSnapshotSaveRestore verifies that Save/Restore round-trips the game state.
func TestSnapshotSaveRestore(t *testing.T) {
	g := NewGame(42, DefaultHouseRules())
	g.Deal()

	// Record original values before mutation.
	origPlayer := g.CurrentPlayer
	origTurn := g.TurnNumber
	origFlags := g.Flags
	origRNG := g.RNG
	origStock := g.StockLen
	origDiscard := g.DiscardLen

	snap := g.Save()

	// Mutate several fields.
	g.CurrentPlayer = 1 - g.CurrentPlayer
	g.TurnNumber = 999
	g.Flags |= FlagGameOver
	g.RNG = 0xDEADBEEF
	g.StockLen = 0
	g.DiscardLen = 0

	// Restore from snapshot.
	g.Restore(snap)

	// All fields should match original saved state.
	if g.CurrentPlayer != origPlayer {
		t.Errorf("CurrentPlayer: got %d, want %d", g.CurrentPlayer, origPlayer)
	}
	if g.TurnNumber != origTurn {
		t.Errorf("TurnNumber: got %d, want %d", g.TurnNumber, origTurn)
	}
	if g.Flags != origFlags {
		t.Errorf("Flags: got %d, want %d", g.Flags, origFlags)
	}
	if g.RNG != origRNG {
		t.Errorf("RNG: got %d, want %d", g.RNG, origRNG)
	}
	if g.StockLen != origStock {
		t.Errorf("StockLen: got %d, want %d", g.StockLen, origStock)
	}
	if g.DiscardLen != origDiscard {
		t.Errorf("DiscardLen: got %d, want %d", g.DiscardLen, origDiscard)
	}
}

// TestSnapshotIndependence verifies that a Snapshot is a value copy independent of the live state.
func TestSnapshotIndependence(t *testing.T) {
	g := NewGame(7, DefaultHouseRules())
	g.Deal()

	snap := g.Save()
	origSnapPlayer := GameState(snap).CurrentPlayer

	// Modifying game does not affect snapshot.
	g.CurrentPlayer = 1 - g.CurrentPlayer
	if GameState(snap).CurrentPlayer != origSnapPlayer {
		t.Error("snapshot was mutated when game state changed")
	}

	// Modifying snapshot does not affect game.
	gamePlayerBefore := g.CurrentPlayer
	snap2 := g.Save()
	snap2Player := GameState(snap2).CurrentPlayer
	modSnap := snap2
	gs := (*GameState)(&modSnap)
	gs.CurrentPlayer = 1 - gs.CurrentPlayer
	if g.CurrentPlayer != gamePlayerBefore {
		t.Error("game state was mutated when snapshot was changed")
	}
	_ = snap2Player
}

// BenchmarkSnapshot measures Save/Restore throughput.
func BenchmarkSnapshot(b *testing.B) {
	gs := NewGame(42, DefaultHouseRules())
	gs.Deal()
	snap := gs.Save()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gs.Restore(snap)
	}
}
