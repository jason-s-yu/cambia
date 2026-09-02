package agent

import (
	"math/rand/v2"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// nplayer_legacy_invariance_test.go pins a digest of the legacy 2-player belief path for a
// seeded game: the token stream, the 222-dim legacy encode, the 224-dim EP-PBS encode, and
// the 16-byte infoset key. InitializeNPlayer's knower-mask fix (cambia-1751) writes only
// N-player-only fields -- KnowledgeMask, NPlayerSlotKnown, NPlayerSlotBuckets -- which
// none of Initialize, Update, Encode, EncodeEPPBS, InfosetKey, or TokenStream ever read;
// this digest must stay byte-identical across that change and any future change that
// touches the N-player code paths sharing state.go with the legacy ones.

// twoPlayerDigest replays a fixed-seed 2-player game through the legacy belief path and a
// player-0 token stream, choosing actions with a seeded chooser so the trajectory itself
// is reproducible.
func twoPlayerDigest(t *testing.T) (encode [InputDim]float32, eppbs [EPPBSInputDim]float32, infoset [16]uint8, tokens []int32) {
	t.Helper()
	g := engine.NewGame(20261751, engine.DefaultHouseRules())
	g.Deal()

	a0 := NewAgentState(0, 1, 1, 5)
	a1 := NewAgentState(1, 0, 1, 5)
	a0.Initialize(&g)
	a1.Initialize(&g)

	var ts TokenStream
	if err := ts.Init(&g, 0); err != nil {
		t.Fatalf("token stream init: %v", err)
	}

	rng := rand.New(rand.NewPCG(20261751, 0x1751))
	for step := 0; step < 400 && !g.IsTerminal(); step++ {
		acts := g.LegalActionsList()
		if len(acts) == 0 {
			break
		}
		act := acts[rng.IntN(len(acts))]
		if err := g.ApplyAction(act); err != nil {
			t.Fatalf("step %d: ApplyAction(%d): %v", step, act, err)
		}
		a0.Update(&g)
		a1.Update(&g)
		if err := ts.Observe(&g, 0); err != nil {
			t.Fatalf("step %d: token observe: %v", step, err)
		}
	}

	a0.Encode(engine.CtxStartTurn, -1, &encode)
	a0.EncodeEPPBS(engine.CtxStartTurn, -1, &eppbs)
	infoset = a0.InfosetKey()
	tokens = append(tokens, ts.Tokens[:ts.Length]...)
	return encode, eppbs, infoset, tokens
}

// TestTwoPlayerLegacyDigestInvariance asserts that two independent replays of the same
// seeded 2-player game produce byte-identical digests, and that the digest is unaffected
// by also constructing and discarding an N-player agent (exercising InitializeNPlayer's
// cambia-1751 fix) over the same dealt game in between: the legacy path reads none of the
// N-player-only fields the fix touches, so interleaving it must change nothing.
func TestTwoPlayerLegacyDigestInvariance(t *testing.T) {
	wantEncode, wantEPPBS, wantInfoset, wantTokens := twoPlayerDigest(t)

	if len(wantTokens) == 0 {
		t.Fatal("token stream is empty; digest would be vacuous")
	}

	// Replay again, but exercise the N-player knower-mask fix on a throwaway agent over
	// an identically-seeded, freshly dealt game before taking the second digest.
	g := engine.NewGame(20261751, engine.DefaultHouseRules())
	g.Deal()
	nplayerAgent := NewNPlayerAgentState(0, 2, 1, 5)
	nplayerAgent.InitializeNPlayer(&g)
	if !nplayerAgent.KnowledgeMask[nplayerSlot(1, 0)][1] {
		t.Fatal("sanity: InitializeNPlayer did not seed seat 1's own initial peek")
	}

	gotEncode, gotEPPBS, gotInfoset, gotTokens := twoPlayerDigest(t)

	if wantEncode != gotEncode {
		t.Errorf("222-dim encode digest differs")
	}
	if wantEPPBS != gotEPPBS {
		t.Errorf("224-dim EP-PBS encode digest differs")
	}
	if wantInfoset != gotInfoset {
		t.Errorf("infoset key digest differs: %v vs %v", wantInfoset, gotInfoset)
	}
	if !tokensEqual(wantTokens, gotTokens) {
		t.Errorf("token stream digest differs: %v vs %v", wantTokens, gotTokens)
	}
}

// tokensEqual compares two token streams for byte-for-byte equality.
func tokensEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestNPlayerInitialPeeksSeedEverySeatsKnowerBit asserts the knower axis InitializeNPlayer
// seeds at game start for a 4-seat table: every seat knows exactly its own initial peek
// slots, and nothing else. The mask used to seed only the encoding agent's own peeks,
// which understated what every other seat knows about its own hand from the deal, even
// though the belief about opponents' cards (which the encoding agent never saw) was
// unaffected (cambia-1751).
func TestNPlayerInitialPeeksSeedEverySeatsKnowerBit(t *testing.T) {
	const seats = 4
	g := makeNPlayerGame(t, seats, 4242)
	a := NewNPlayerAgentState(0, seats, 1, 5)
	a.InitializeNPlayer(&g)

	if g.Players[0].InitialPeekCount == 0 {
		t.Fatal("test requires InitialViewCount > 0 to be meaningful")
	}

	for seat := uint8(0); seat < seats; seat++ {
		peeked := map[uint8]bool{}
		for i := uint8(0); i < g.Players[seat].InitialPeekCount; i++ {
			peeked[g.Players[seat].InitialPeek[i]] = true
		}
		for c := uint8(0); c < engine.MaxHandSize; c++ {
			slot := nplayerSlot(seat, c)
			wantKnower := c < g.Players[seat].HandLen && peeked[c]
			if got := a.KnowledgeMask[slot][seat]; got != wantKnower {
				t.Errorf("seat %d slot %d: KnowledgeMask[seat %d] = %v, want %v",
					seat, c, seat, got, wantKnower)
			}
			// No seat other than the card's own owner is recorded as a knower.
			for k := uint8(0); k < engine.MaxPlayers; k++ {
				if k == seat {
					continue
				}
				if a.KnowledgeMask[slot][k] {
					t.Errorf("seat %d slot %d: seat %d wrongly recorded as a knower", seat, c, k)
				}
			}
		}
	}

	// The encoding agent's own peeked buckets are revealed; every other seat's stay
	// unrevealed even though the mask now records that seat as knowing them.
	for seat := uint8(0); seat < seats; seat++ {
		for i := uint8(0); i < g.Players[seat].InitialPeekCount; i++ {
			slot := nplayerSlot(seat, g.Players[seat].InitialPeek[i])
			wantRevealed := seat == a.PlayerID
			if got := a.NPlayerSlotKnown[slot]; got != wantRevealed {
				t.Errorf("seat %d peeked slot %d: NPlayerSlotKnown = %v, want %v",
					seat, slot, got, wantRevealed)
			}
		}
	}
}
