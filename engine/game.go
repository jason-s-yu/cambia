// Package engine implements the Cambia card game rules.
//
// This package provides a cache-optimized, zero-allocation game engine
// suitable for both real-time gameplay (via the service adapter) and
// high-throughput CFR training (via cgo/FFI).
package engine

const (
	MaxPlayers  = 6
	MaxHandSize = 6
	DeckSize    = 54
)

// PlayerState holds one player's hand and initial peek information.
// Padded to exactly 16 bytes for cache alignment.
type PlayerState struct {
	Hand        [MaxHandSize]Card // 6 bytes
	HandLen     uint8             // 1 byte
	InitialPeek [2]uint8          // 2 bytes
	_pad        [7]uint8          // 7 bytes → total 6+1+2+7=16
}


// GameState holds the complete, self-contained state of a Cambia game.
// It is a flat value type (no pointers, no slices) for zero-allocation
// use in CFR traversal. sizeof(GameState) ≤ 300 bytes.
type GameState struct {
	Players       [MaxPlayers]PlayerState // 2 * 16 = 32 bytes
	Stockpile     [DeckSize]Card          // 54 bytes
	StockLen      uint8                   // 1 byte
	DiscardPile   [DeckSize]Card          // 54 bytes
	DiscardLen    uint8                   // 1 byte
	CurrentPlayer uint8                   // 1 byte
	TurnNumber    uint16                  // 2 bytes
	Flags         uint16                  // 2 bytes
	Pending       PendingAction           // 6 bytes
	Snap          SnapState               // 6 bytes
	LastAction    LastActionInfo          // 12 bytes (estimated)
	RNG           uint64                  // 8 bytes
	CambiaCaller  int8                    // 1 byte
	TurnsAfterC   uint8                   // 1 byte
	Rules         HouseRules              // ~12 bytes
}

// ---------------------------------------------------------------------------
// Flags bitfield
// ---------------------------------------------------------------------------

const (
	FlagGameOver     uint16 = 1 << 0
	FlagCambiaCalled uint16 = 1 << 1
	FlagGameStarted  uint16 = 1 << 2
)

func (g *GameState) IsGameOver() bool     { return g.Flags&FlagGameOver != 0 }
func (g *GameState) IsCambiaCalled() bool { return g.Flags&FlagCambiaCalled != 0 }

// ---------------------------------------------------------------------------
// xorshift64 RNG — inline, no interface
// ---------------------------------------------------------------------------

func (g *GameState) nextRand() uint64 {
	x := g.RNG
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	g.RNG = x
	return x
}

// randN returns a random number in [0, n).
func (g *GameState) randN(n uint64) uint64 {
	return g.nextRand() % n
}

// ---------------------------------------------------------------------------
// NewGame and Deal
// ---------------------------------------------------------------------------

// NewGame initializes a new GameState with the given seed and rules.
// The deck is built but not yet shuffled or dealt.
func NewGame(seed uint64, rules HouseRules) GameState {
	var g GameState
	g.RNG = seed
	if g.RNG == 0 {
		g.RNG = 1 // xorshift can't start at 0
	}
	g.Rules = rules
	g.CambiaCaller = -1

	// Initialize deck: 4 suits × 13 ranks = 52 + NumJokers jokers.
	idx := 0
	for suit := uint8(0); suit < 4; suit++ {
		for rank := uint8(0); rank <= RankKing; rank++ {
			g.Stockpile[idx] = NewCard(suit, rank)
			idx++
		}
	}
	jokerSuits := [2]uint8{SuitRedJoker, SuitBlackJoker}
	for j := uint8(0); j < rules.NumJokers && j < 2; j++ {
		g.Stockpile[52+int(j)] = NewCard(jokerSuits[j], RankJoker)
	}
	g.StockLen = uint8(52 + rules.NumJokers)

	return g
}

// Deal shuffles the deck and distributes cards to players.
// It also flips the top stockpile card to start the discard pile.
func (g *GameState) Deal() {
	// Fisher-Yates shuffle.
	for i := int(g.StockLen) - 1; i > 0; i-- {
		j := int(g.randN(uint64(i + 1)))
		g.Stockpile[i], g.Stockpile[j] = g.Stockpile[j], g.Stockpile[i]
	}

	n := g.Rules.numPlayers()

	// Deal cards: alternate between players (deal 1 to p0, 1 to p1, ..., repeat).
	for c := uint8(0); c < g.Rules.CardsPerPlayer; c++ {
		for p := uint8(0); p < n; p++ {
			g.StockLen--
			card := g.Stockpile[g.StockLen]
			g.Players[p].Hand[c] = card
			g.Players[p].HandLen++
		}
	}

	// Set initial peek indices (bottom two cards = indices 0, 1).
	for p := uint8(0); p < n; p++ {
		g.Players[p].InitialPeek = [2]uint8{0, 1}
	}

	// Flip top stockpile card to start the discard pile (per RULES.md §2).
	g.StockLen--
	g.DiscardPile[0] = g.Stockpile[g.StockLen]
	g.DiscardLen = 1

	// Pick random starting player.
	g.CurrentPlayer = uint8(g.randN(uint64(n)))
	g.Flags |= FlagGameStarted
}

// ---------------------------------------------------------------------------
// Query methods
// ---------------------------------------------------------------------------

// IsTerminal returns true when the game is over.
func (g *GameState) IsTerminal() bool { return g.Flags&FlagGameOver != 0 }

// ActingPlayer returns the index of the player who must act next.
// During snap resolution or pending actions, that player takes priority.
func (g *GameState) ActingPlayer() uint8 {
	if g.Snap.Active {
		return g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	}
	if g.Pending.Type != PendingNone {
		return g.Pending.PlayerID
	}
	return g.CurrentPlayer
}

// TurnPlayer returns the player whose turn it currently is (ignoring snaps/pending).
func (g *GameState) TurnPlayer() uint8 { return g.CurrentPlayer }

// DiscardTop returns the top card of the discard pile, or EmptyCard if empty.
func (g *GameState) DiscardTop() Card {
	if g.DiscardLen == 0 {
		return EmptyCard
	}
	return g.DiscardPile[g.DiscardLen-1]
}

// OpponentOf returns the player index of the opponent.
// NOTE: Only meaningful for 2-player games. For N-player games, use Opponents().
func (g *GameState) OpponentOf(player uint8) uint8 {
	return 1 - player
}

// NumActivePlayers returns the number of active players in this game.
func (g *GameState) NumActivePlayers() uint8 { return g.Rules.numPlayers() }

// NextPlayer returns the next player after current in turn order.
func (g *GameState) NextPlayer(current uint8) uint8 {
	return (current + 1) % g.Rules.numPlayers()
}

// Opponents returns all player indices except the given player.
func (g *GameState) Opponents(player uint8) []uint8 {
	n := g.Rules.numPlayers()
	opps := make([]uint8, 0, n-1)
	for i := uint8(0); i < n; i++ {
		if i != player {
			opps = append(opps, i)
		}
	}
	return opps
}

// ---------------------------------------------------------------------------
// Snapshot Undo (Save / Restore)
// ---------------------------------------------------------------------------

// Snapshot is a complete value-copy of GameState for undo support.
// No heap allocation, saving and restoring are plain struct copies.
type Snapshot GameState

// Save returns a snapshot of the current game state.
func (g *GameState) Save() Snapshot { return Snapshot(*g) }

// Restore replaces the game state with the given snapshot.
func (g *GameState) Restore(s Snapshot) { *g = GameState(s) }
