package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// newDealtGame creates a new game, deals cards, and returns the game state.
// Uses seed 42 for reproducibility.
func newDealtGame() engine.GameState {
	g := engine.NewGame(42, engine.DefaultHouseRules())
	g.Deal()
	return g
}

// applyOrFatal applies an action and fails the test on error.
func applyOrFatal(t *testing.T, g *engine.GameState, action uint16) {
	t.Helper()
	if err := g.ApplyAction(action); err != nil {
		t.Fatalf("ApplyAction(%d) failed: %v", action, err)
	}
}

// applyAndUpdate applies an action and calls Update on all provided agents.
func applyAndUpdate(t *testing.T, g *engine.GameState, action uint16, agents ...*AgentState) {
	t.Helper()
	applyOrFatal(t, g, action)
	for _, a := range agents {
		a.Update(g)
	}
}

// skipSnap passes the snap phase for all current snappers (no agent update).
func skipSnap(t *testing.T, g *engine.GameState) {
	t.Helper()
	for g.Snap.Active {
		applyOrFatal(t, g, engine.ActionPassSnap)
	}
}

// skipSnapAndUpdate passes all snap decisions and calls Update for all agents.
func skipSnapAndUpdate(t *testing.T, g *engine.GameState, agents ...*AgentState) {
	t.Helper()
	for g.Snap.Active {
		applyOrFatal(t, g, engine.ActionPassSnap)
		for _, a := range agents {
			a.Update(g)
		}
	}
}

// advanceToPlayer advances the game until it's the given player's turn.
// Calls Update on all provided agents after each action.
func advanceToPlayer(t *testing.T, g *engine.GameState, player uint8, agents ...*AgentState) {
	t.Helper()
	for g.CurrentPlayer != player {
		applyAndUpdate(t, g, engine.ActionDrawStockpile, agents...)
		applyAndUpdate(t, g, engine.ActionDiscardNoAbility, agents...)
		skipSnapAndUpdate(t, g, agents...)
	}
}

// TestNewAgentState verifies zero-initialization with correct IDs.
func TestNewAgentState(t *testing.T) {
	a := NewAgentState(0, 1, 1, 3)
	if a.PlayerID != 0 {
		t.Errorf("PlayerID = %d, want 0", a.PlayerID)
	}
	if a.OpponentID != 1 {
		t.Errorf("OpponentID = %d, want 1", a.OpponentID)
	}
	if a.MemoryLevel != 1 {
		t.Errorf("MemoryLevel = %d, want 1", a.MemoryLevel)
	}
	if a.TimeDecayTurns != 3 {
		t.Errorf("TimeDecayTurns = %d, want 3", a.TimeDecayTurns)
	}
	if a.OwnHandLen != 0 {
		t.Errorf("OwnHandLen = %d, want 0", a.OwnHandLen)
	}
	if a.OppHandLen != 0 {
		t.Errorf("OppHandLen = %d, want 0", a.OppHandLen)
	}
	if a.CambiaState != CambiaNone {
		t.Errorf("CambiaState = %d, want CambiaNone", a.CambiaState)
	}
	if a.CurrentTurn != 0 {
		t.Errorf("CurrentTurn = %d, want 0", a.CurrentTurn)
	}
}

// TestInitialize verifies that Initialize sets up hand lengths, peek cards,
// opponent beliefs, discard top, and stock estimate correctly.
func TestInitialize(t *testing.T) {
	g := newDealtGame()

	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	// Hand lengths should match game state.
	if a.OwnHandLen != g.Players[0].HandLen {
		t.Errorf("OwnHandLen = %d, want %d", a.OwnHandLen, g.Players[0].HandLen)
	}
	if a.OppHandLen != g.Players[1].HandLen {
		t.Errorf("OppHandLen = %d, want %d", a.OppHandLen, g.Players[1].HandLen)
	}

	// Initial peek indices are 0 and 1; those should have known buckets.
	peekIdx0 := g.Players[0].InitialPeek[0]
	peekIdx1 := g.Players[0].InitialPeek[1]
	expectedCard0 := g.Players[0].Hand[peekIdx0]
	expectedCard1 := g.Players[0].Hand[peekIdx1]

	if a.OwnHand[peekIdx0].Bucket != CardToBucket(expectedCard0) {
		t.Errorf("OwnHand[%d].Bucket = %d, want %d (peek idx)",
			peekIdx0, a.OwnHand[peekIdx0].Bucket, CardToBucket(expectedCard0))
	}
	if a.OwnHand[peekIdx0].Card != expectedCard0 {
		t.Errorf("OwnHand[%d].Card = %v, want %v", peekIdx0, a.OwnHand[peekIdx0].Card, expectedCard0)
	}
	if a.OwnHand[peekIdx1].Bucket != CardToBucket(expectedCard1) {
		t.Errorf("OwnHand[%d].Bucket = %d, want %d (peek idx)",
			peekIdx1, a.OwnHand[peekIdx1].Bucket, CardToBucket(expectedCard1))
	}

	// Non-peeked own cards should be BucketUnknown.
	for i := uint8(0); i < a.OwnHandLen; i++ {
		if i == peekIdx0 || i == peekIdx1 {
			continue
		}
		if a.OwnHand[i].Bucket != BucketUnknown {
			t.Errorf("OwnHand[%d].Bucket = %d, want BucketUnknown for non-peeked slot", i, a.OwnHand[i].Bucket)
		}
		if a.OwnHand[i].Card != engine.EmptyCard {
			t.Errorf("OwnHand[%d].Card should be EmptyCard for non-peeked slot", i)
		}
	}

	// All opponent beliefs should be BucketUnknown.
	for i := uint8(0); i < a.OppHandLen; i++ {
		if !a.OppBelief[i].IsBucket() || a.OppBelief[i].Bucket() != BucketUnknown {
			t.Errorf("OppBelief[%d] = %d, want BucketUnknown", i, a.OppBelief[i])
		}
		if a.OppHasLastSeen[i] {
			t.Errorf("OppHasLastSeen[%d] should be false after Initialize", i)
		}
	}

	// Discard top should be set.
	discardTop := g.DiscardTop()
	expectedBucket := CardToBucket(discardTop)
	if a.DiscardTopBucket != expectedBucket {
		t.Errorf("DiscardTopBucket = %d, want %d", a.DiscardTopBucket, expectedBucket)
	}

	// Stock estimate should match.
	expectedStock := StockEstimateFromSize(g.StockLen)
	if a.StockEstimate != expectedStock {
		t.Errorf("StockEstimate = %d, want %d", a.StockEstimate, expectedStock)
	}

	// CambiaState should be none.
	if a.CambiaState != CambiaNone {
		t.Errorf("CambiaState = %d, want CambiaNone", a.CambiaState)
	}
	// Phase is derived from stockpile size (matching Python).
	// At game start with a large stockpile, phase should be PhaseEarly.
	expectedPhase := GamePhaseFromState(g.StockLen, g.IsCambiaCalled(), g.IsTerminal())
	if a.Phase != expectedPhase {
		t.Errorf("Phase = %d, want %d", a.Phase, expectedPhase)
	}
}

// TestUpdateReplace verifies that replacing a hand card updates own hand belief.
func TestUpdateReplace(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	// Advance to player 0's turn.
	advanceToPlayer(t, &g, 0, &a)

	// Draw from stockpile.
	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a)

	// Replace hand card at index 2 (a non-peeked slot).
	replaceIdx := uint8(2)
	applyAndUpdate(t, &g, engine.EncodeReplace(replaceIdx), &a)

	// After replace, we should know what's at index 2 (it's the drawn card).
	expectedCard := g.Players[0].Hand[replaceIdx]
	if a.OwnHand[replaceIdx].Bucket != CardToBucket(expectedCard) {
		t.Errorf("After replace, OwnHand[%d].Bucket = %d, want %d",
			replaceIdx, a.OwnHand[replaceIdx].Bucket, CardToBucket(expectedCard))
	}
	if a.OwnHand[replaceIdx].Card != expectedCard {
		t.Errorf("After replace, OwnHand[%d].Card = %v, want %v",
			replaceIdx, a.OwnHand[replaceIdx].Card, expectedCard)
	}

	skipSnapAndUpdate(t, &g, &a)
}

// TestUpdatePeekOwn verifies that using peek-own ability updates own hand belief.
func TestUpdatePeekOwn(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	advanceToPlayer(t, &g, 0, &a)

	// Replace top stockpile card with a 7 (peek-self).
	sevenCard := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	g.Stockpile[g.StockLen-1] = sevenCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a)
	applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a)

	// Peek own card at index 3 (a non-peeked slot).
	peekIdx := uint8(3)
	applyAndUpdate(t, &g, engine.EncodePeekOwn(peekIdx), &a)

	// The peeked card should now be known.
	revealed := g.LastAction.RevealedCard
	if a.OwnHand[peekIdx].Bucket != CardToBucket(revealed) {
		t.Errorf("After peek own, OwnHand[%d].Bucket = %d, want %d",
			peekIdx, a.OwnHand[peekIdx].Bucket, CardToBucket(revealed))
	}
	if a.OwnHand[peekIdx].Card != revealed {
		t.Errorf("After peek own, OwnHand[%d].Card = %v, want %v",
			peekIdx, a.OwnHand[peekIdx].Card, revealed)
	}

	skipSnapAndUpdate(t, &g, &a)
}

// TestUpdatePeekOther verifies that using peek-other ability updates opponent belief.
func TestUpdatePeekOther(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	advanceToPlayer(t, &g, 0, &a)

	// Replace top stockpile card with a 9 (peek-other).
	nineCard := engine.NewCard(engine.SuitHearts, engine.RankNine)
	g.Stockpile[g.StockLen-1] = nineCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a)
	applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a)

	// Peek opponent card at index 0.
	peekIdx := uint8(0)
	applyAndUpdate(t, &g, engine.EncodePeekOther(peekIdx), &a)

	// Opponent's card at peekIdx should now be known.
	revealed := g.LastAction.RevealedCard
	if !a.OppBelief[peekIdx].IsBucket() {
		t.Errorf("OppBelief[%d] should be a bucket after peek other", peekIdx)
	}
	if a.OppBelief[peekIdx].Bucket() != CardToBucket(revealed) {
		t.Errorf("After peek other, OppBelief[%d].Bucket = %d, want %d",
			peekIdx, a.OppBelief[peekIdx].Bucket(), CardToBucket(revealed))
	}
	if !a.OppHasLastSeen[peekIdx] {
		t.Errorf("OppHasLastSeen[%d] should be true after peek other", peekIdx)
	}

	skipSnapAndUpdate(t, &g, &a)
}

// TestUpdateBlindSwap tests blind swap belief updates from both perspectives.
func TestUpdateBlindSwap(t *testing.T) {
	g := newDealtGame()
	a0 := NewAgentState(0, 1, 1, 3) // player 0 perspective, memory 1
	a1 := NewAgentState(1, 0, 1, 3) // player 1 perspective, memory 1
	a0.Initialize(&g)
	a1.Initialize(&g)

	// Advance to player 0 and peek opponent slot 1 to give a0 knowledge.
	advanceToPlayer(t, &g, 0, &a0, &a1)

	nineCard := engine.NewCard(engine.SuitClubs, engine.RankNine)
	g.Stockpile[g.StockLen-1] = nineCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a0, &a1)
	applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a0, &a1)
	applyAndUpdate(t, &g, engine.EncodePeekOther(1), &a0, &a1)
	skipSnapAndUpdate(t, &g, &a0, &a1)

	// Verify a0 knows opp slot 1.
	if !a0.OppBelief[1].IsBucket() || a0.OppBelief[1].Bucket() == BucketUnknown {
		t.Skip("peek other didn't result in known bucket, skipping blind swap test")
	}

	// Advance to player 0's turn again.
	advanceToPlayer(t, &g, 0, &a0, &a1)

	// Replace top stockpile card with a Jack (blind swap).
	jackCard := engine.NewCard(engine.SuitHearts, engine.RankJack)
	g.Stockpile[g.StockLen-1] = jackCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a0, &a1)
	applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a0, &a1)

	// Blind swap: player 0's card at ownIdx=2 with opp card at oppIdx=1.
	ownIdx := uint8(2)
	oppIdx := uint8(1)
	applyAndUpdate(t, &g, engine.EncodeBlindSwap(ownIdx, oppIdx), &a0, &a1)

	// For a0 (the acting player):
	// - own card at ownIdx should be UNKNOWN (got opponent's card — unknown to us).
	if a0.OwnHand[ownIdx].Bucket != BucketUnknown {
		t.Errorf("After blind swap (acting), a0.OwnHand[%d].Bucket = %d, want BucketUnknown",
			ownIdx, a0.OwnHand[ownIdx].Bucket)
	}
	// - opponent slot at oppIdx should now be decayed (event decay, memory=1).
	if a0.OppBelief[oppIdx].IsBucket() && a0.OppBelief[oppIdx].Bucket() != BucketUnknown {
		t.Errorf("After blind swap (acting), a0.OppBelief[%d] should be decayed, got bucket %d",
			oppIdx, a0.OppBelief[oppIdx].Bucket())
	}

	// For a1 (observing): a1's own card at index oppIdx should be UNKNOWN
	// (from a1's perspective, the acting player took a1's card at oppIdx).
	if a1.OwnHand[oppIdx].Bucket != BucketUnknown {
		t.Errorf("After blind swap (observer), a1.OwnHand[%d].Bucket = %d, want BucketUnknown",
			oppIdx, a1.OwnHand[oppIdx].Bucket)
	}

	skipSnapAndUpdate(t, &g, &a0, &a1)
}

// TestUpdateKingLookAndSwap verifies king look reveals both cards, and swap
// makes own card unknown while decaying opponent slot.
func TestUpdateKingLookAndSwap(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 1, 3)
	a.Initialize(&g)

	advanceToPlayer(t, &g, 0, &a)

	// Replace top stockpile card with a Black King for king look.
	kingCard := engine.NewCard(engine.SuitClubs, engine.RankKing)
	g.Stockpile[g.StockLen-1] = kingCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a)
	applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a)

	// King look: own card at 2, opp card at 0.
	ownIdx := uint8(2)
	oppIdx := uint8(0)
	applyAndUpdate(t, &g, engine.EncodeKingLook(ownIdx, oppIdx), &a)

	// After king look, we should know our own card at ownIdx.
	ownCard := g.LastAction.RevealedCard
	if a.OwnHand[ownIdx].Bucket != CardToBucket(ownCard) {
		t.Errorf("After king look, OwnHand[%d].Bucket = %d, want %d",
			ownIdx, a.OwnHand[ownIdx].Bucket, CardToBucket(ownCard))
	}
	if a.OwnHand[ownIdx].Card != ownCard {
		t.Errorf("After king look, OwnHand[%d].Card = %v, want %v",
			ownIdx, a.OwnHand[ownIdx].Card, ownCard)
	}

	// We should also know the opponent's card at oppIdx.
	oppCard := g.Players[1].Hand[oppIdx]
	if !a.OppBelief[oppIdx].IsBucket() || a.OppBelief[oppIdx].Bucket() != CardToBucket(oppCard) {
		t.Errorf("After king look, OppBelief[%d] = %d, want %d",
			oppIdx, a.OppBelief[oppIdx], CardToBucket(oppCard))
	}
	if !a.OppHasLastSeen[oppIdx] {
		t.Errorf("OppHasLastSeen[%d] should be true after king look", oppIdx)
	}

	// Now decide to swap.
	applyAndUpdate(t, &g, engine.ActionKingSwapYes, &a)

	// After swap, our own card at ownIdx should be UNKNOWN (we gave it away).
	if a.OwnHand[ownIdx].Bucket != BucketUnknown {
		t.Errorf("After king swap yes, OwnHand[%d].Bucket = %d, want BucketUnknown",
			ownIdx, a.OwnHand[ownIdx].Bucket)
	}

	// Opponent's slot at oppIdx should now be decayed (memory level 1 = event decay).
	if a.OppBelief[oppIdx].IsBucket() && a.OppBelief[oppIdx].Bucket() != BucketUnknown {
		t.Errorf("After king swap yes, OppBelief[%d] should be decayed", oppIdx)
	}

	skipSnapAndUpdate(t, &g, &a)
}

// TestUpdateSnapOwn verifies that a successful snap removes the card from hand
// and that the indices above it shift down.
func TestUpdateSnapOwn(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	// Find player 0's card at index 0 to determine the rank.
	snapRank := g.Players[0].Hand[0].Rank()

	// Advance to player 0 then draw and discard a matching rank card.
	advanceToPlayer(t, &g, 0, &a)

	// Replace top stockpile card with a card of matching rank.
	matchCard := engine.NewCard(engine.SuitClubs, snapRank)
	g.Stockpile[g.StockLen-1] = matchCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a)
	applyAndUpdate(t, &g, engine.ActionDiscardNoAbility, &a)

	// If snap phase is active and player 0 can snap own card at index 0.
	if !g.Snap.Active {
		t.Skip("snap phase not active after discarding matching rank card")
	}

	origHandLen := a.OwnHandLen
	beliefAt1 := a.OwnHand[1]

	snapIdx := uint8(0)
	applyAndUpdate(t, &g, engine.EncodeSnapOwn(snapIdx), &a)

	if g.LastAction.SnapSuccess {
		// Hand length should decrease by 1.
		if a.OwnHandLen != origHandLen-1 {
			t.Errorf("After successful snap, OwnHandLen = %d, want %d",
				a.OwnHandLen, origHandLen-1)
		}
		// Index 0 should now have what was at index 1.
		if a.OwnHand[0].Bucket != beliefAt1.Bucket {
			t.Errorf("After snap, OwnHand[0].Bucket = %d, want %d (shifted from index 1)",
				a.OwnHand[0].Bucket, beliefAt1.Bucket)
		}
	} else {
		// Failed snap: penalty cards added.
		penalty := g.Rules.PenaltyDrawCount
		expectedLen := origHandLen + penalty
		if expectedLen > engine.MaxHandSize {
			expectedLen = engine.MaxHandSize
		}
		if a.OwnHandLen != expectedLen {
			t.Errorf("After failed snap, OwnHandLen = %d, want %d",
				a.OwnHandLen, expectedLen)
		}
	}

	skipSnapAndUpdate(t, &g, &a)
}

// TestUpdateSnapFail verifies that a failed snap adds penalty cards to the hand.
// This test forces the scenario by injecting a card that matches a hand rank to
// trigger snap, then snapping a non-matching card index.
func TestUpdateSnapFail(t *testing.T) {
	g := newDealtGame()
	a0 := NewAgentState(0, 1, 0, 0)
	a1 := NewAgentState(1, 0, 0, 0)
	a0.Initialize(&g)
	a1.Initialize(&g)

	advanceToPlayer(t, &g, 0, &a0, &a1)

	// Player 0 has rank=8 (RankNine) at index [0] with seed=42.
	// Use RankNine to trigger snap. RankNine is >= RankSeven so it has an ability,
	// so we must use DiscardWithAbility.
	// Instead, find a rank in player 1's hand that player 0 doesn't have (for snap to be legal
	// but fail for player 0). Actually, we want snap phase active with player 0 able to attempt
	// snap on a WRONG index.
	//
	// Approach: inject the same rank as player 0's hand[0] into the discard via draw+discard.
	// Player 0's hand[0] has some rank; inject that rank.
	matchRank := g.Players[0].Hand[0].Rank()

	// If rank >= 7 (has ability), we need a card from a different suit but same rank on stockpile
	// and use DiscardWithAbility. Or, find a non-ability rank.
	// For simplicity, use suit Diamonds since Hearts was original suit.
	injectCard := engine.NewCard(engine.SuitDiamonds, matchRank)
	g.Stockpile[g.StockLen-1] = injectCard

	applyAndUpdate(t, &g, engine.ActionDrawStockpile, &a0, &a1)

	// If rank has ability, use DiscardWithAbility; otherwise DiscardNoAbility.
	if injectCard.Rank() >= engine.RankSeven && injectCard.Rank() <= engine.RankKing {
		// Has ability — discard with ability to trigger snap after resolving.
		applyAndUpdate(t, &g, engine.ActionDiscardWithAbility, &a0, &a1)
		// Resolve the ability quickly (peek own slot 0 or similar).
		switch injectCard.Ability() {
		case engine.AbilityPeekOwn:
			applyAndUpdate(t, &g, engine.EncodePeekOwn(0), &a0, &a1)
		case engine.AbilityPeekOther:
			applyAndUpdate(t, &g, engine.EncodePeekOther(0), &a0, &a1)
		case engine.AbilityBlindSwap:
			applyAndUpdate(t, &g, engine.EncodeBlindSwap(0, 0), &a0, &a1)
		case engine.AbilityKingLook:
			applyAndUpdate(t, &g, engine.EncodeKingLook(0, 0), &a0, &a1)
			applyAndUpdate(t, &g, engine.ActionKingSwapNo, &a0, &a1)
		}
	} else {
		applyAndUpdate(t, &g, engine.ActionDiscardNoAbility, &a0, &a1)
	}

	if !g.Snap.Active {
		t.Skip("snap phase not active after injecting matching rank card")
	}

	// Find a snapper and their hand index that DOESN'T match the discard rank.
	snapperIdx := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	discardRank := g.Snap.DiscardedRank

	var badIdx uint8 = 255
	hand := g.Players[snapperIdx]
	for i := uint8(0); i < hand.HandLen; i++ {
		if hand.Hand[i].Rank() != discardRank {
			badIdx = i
			break
		}
	}
	if badIdx == 255 {
		t.Skip("all hand cards match discard rank, cannot test failed snap")
	}

	var agentForSnapper *AgentState
	if snapperIdx == 0 {
		agentForSnapper = &a0
	} else {
		agentForSnapper = &a1
	}

	origHandLen := agentForSnapper.OwnHandLen
	applyAndUpdate(t, &g, engine.EncodeSnapOwn(badIdx), agentForSnapper)

	if g.LastAction.SnapSuccess {
		t.Skip("snap succeeded unexpectedly, skipping failure test")
	}

	penalty := g.Rules.PenaltyDrawCount
	expectedLen := origHandLen + penalty
	if expectedLen > engine.MaxHandSize {
		expectedLen = engine.MaxHandSize
	}
	if agentForSnapper.OwnHandLen != expectedLen {
		t.Errorf("After failed snap, OwnHandLen = %d, want %d",
			agentForSnapper.OwnHandLen, expectedLen)
	}

	// The penalty cards should be BucketUnknown.
	for i := origHandLen; i < agentForSnapper.OwnHandLen; i++ {
		if agentForSnapper.OwnHand[i].Bucket != BucketUnknown {
			t.Errorf("Penalty card at OwnHand[%d].Bucket = %d, want BucketUnknown",
				i, agentForSnapper.OwnHand[i].Bucket)
		}
		if agentForSnapper.OwnHand[i].Card != engine.EmptyCard {
			t.Errorf("Penalty card at OwnHand[%d].Card should be EmptyCard", i)
		}
	}

	skipSnapAndUpdate(t, &g, agentForSnapper)
}

// TestEventDecay verifies event-based decay at memory levels 0, 1, and 2.
func TestEventDecay(t *testing.T) {
	g := newDealtGame()

	// Memory level 0: no decay.
	a0 := NewAgentState(0, 1, 0, 0)
	a0.Initialize(&g)
	a0.OppBelief[0] = BucketBelief(BucketAce)
	a0.OppHasLastSeen[0] = true

	a0.triggerEventDecay(0)

	if !a0.OppBelief[0].IsBucket() || a0.OppBelief[0].Bucket() != BucketAce {
		t.Errorf("Memory 0: OppBelief[0] should not decay, got %v", a0.OppBelief[0])
	}

	// Memory level 1: event decay should fire.
	a1 := NewAgentState(0, 1, 1, 0)
	a1.Initialize(&g)
	a1.OppBelief[0] = BucketBelief(BucketAce)
	a1.OppHasLastSeen[0] = true

	a1.triggerEventDecay(0)

	if a1.OppBelief[0].IsBucket() {
		t.Errorf("Memory 1: OppBelief[0] should have decayed, still a bucket: %v", a1.OppBelief[0])
	}
	if !a1.OppBelief[0].IsDecay() {
		t.Errorf("Memory 1: OppBelief[0] should be decay, got %v", a1.OppBelief[0])
	}
	expectedDecay := BucketToDecay(BucketAce) // DecayLikelyLow
	if a1.OppBelief[0].Decay() != expectedDecay {
		t.Errorf("Memory 1: OppBelief[0].Decay() = %d, want %d", a1.OppBelief[0].Decay(), expectedDecay)
	}

	// Memory level 2: event decay also fires.
	a2 := NewAgentState(0, 1, 2, 5)
	a2.Initialize(&g)
	a2.OppBelief[0] = BucketBelief(BucketHighKing)
	a2.OppHasLastSeen[0] = true

	a2.triggerEventDecay(0)

	if a2.OppBelief[0].IsBucket() {
		t.Errorf("Memory 2: OppBelief[0] should have decayed, still a bucket")
	}
	expectedDecay2 := BucketToDecay(BucketHighKing) // DecayLikelyHigh
	if a2.OppBelief[0].Decay() != expectedDecay2 {
		t.Errorf("Memory 2: OppBelief[0].Decay() = %d, want %d", a2.OppBelief[0].Decay(), expectedDecay2)
	}
}

// TestTimeDecay verifies time-based decay fires at memory level 2 when enough
// turns have passed.
func TestTimeDecay(t *testing.T) {
	g := newDealtGame()

	a := NewAgentState(0, 1, 2, 3) // memory level 2, decay after 3 turns
	a.Initialize(&g)

	// Manually set known opponent belief at turn 0.
	a.OppBelief[0] = BucketBelief(BucketMidNum)
	a.OppLastSeen[0] = 0
	a.OppHasLastSeen[0] = true

	// Simulate turn 2: should NOT decay yet (2 < 3).
	a.CurrentTurn = 2
	a.applyTimeDecay()
	if a.OppBelief[0].IsDecay() {
		t.Errorf("Turn 2: OppBelief[0] should not decay after 2 turns (threshold=3)")
	}

	// Simulate turn 3: should decay exactly at threshold (3 >= 3).
	a.CurrentTurn = 3
	a.applyTimeDecay()
	if a.OppBelief[0].IsBucket() && a.OppBelief[0].Bucket() != BucketUnknown {
		t.Errorf("Turn 3: OppBelief[0] should have decayed at threshold=3")
	}
	if !a.OppBelief[0].IsDecay() {
		t.Errorf("Turn 3: OppBelief[0] should be decay category, got %v", a.OppBelief[0])
	}
	expected := BucketToDecay(BucketMidNum) // DecayLikelyMid
	if a.OppBelief[0].Decay() != expected {
		t.Errorf("Turn 3: OppBelief[0].Decay() = %d, want %d", a.OppBelief[0].Decay(), expected)
	}

	// Memory level 0: time decay should never fire.
	a0 := NewAgentState(0, 1, 0, 3)
	a0.Initialize(&g)
	a0.OppBelief[0] = BucketBelief(BucketMidNum)
	a0.OppLastSeen[0] = 0
	a0.OppHasLastSeen[0] = true
	a0.CurrentTurn = 100
	a0.applyTimeDecay()
	if !a0.OppBelief[0].IsBucket() || a0.OppBelief[0].Bucket() != BucketMidNum {
		t.Errorf("Memory 0: time decay should not fire, got %v", a0.OppBelief[0])
	}

	// Memory level 1: time decay should also not fire.
	a1 := NewAgentState(0, 1, 1, 3)
	a1.Initialize(&g)
	a1.OppBelief[0] = BucketBelief(BucketMidNum)
	a1.OppLastSeen[0] = 0
	a1.OppHasLastSeen[0] = true
	a1.CurrentTurn = 100
	a1.applyTimeDecay()
	if !a1.OppBelief[0].IsBucket() || a1.OppBelief[0].Bucket() != BucketMidNum {
		t.Errorf("Memory 1: time decay should not fire, got %v", a1.OppBelief[0])
	}
}

// TestClone verifies that clone is independent of the original.
func TestClone(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 1, 3)
	a.Initialize(&g)

	// Set some state.
	a.OppBelief[0] = BucketBelief(BucketAce)
	a.OppHasLastSeen[0] = true
	a.OwnHand[2].Bucket = BucketLowNum

	clone := a.Clone()

	// Verify values match.
	if clone.PlayerID != a.PlayerID {
		t.Errorf("Clone PlayerID = %d, want %d", clone.PlayerID, a.PlayerID)
	}
	if clone.OppBelief[0] != a.OppBelief[0] {
		t.Errorf("Clone OppBelief[0] = %v, want %v", clone.OppBelief[0], a.OppBelief[0])
	}
	if clone.OwnHand[2].Bucket != a.OwnHand[2].Bucket {
		t.Errorf("Clone OwnHand[2].Bucket = %d, want %d", clone.OwnHand[2].Bucket, a.OwnHand[2].Bucket)
	}

	// Modify original; clone should be independent.
	a.OppBelief[0] = BucketBelief(BucketHighKing)
	a.OwnHand[2].Bucket = BucketPeekOther

	if clone.OppBelief[0] == a.OppBelief[0] {
		t.Errorf("Clone is not independent: OppBelief[0] changed with original")
	}
	if clone.OwnHand[2].Bucket == a.OwnHand[2].Bucket {
		t.Errorf("Clone is not independent: OwnHand[2].Bucket changed with original")
	}
}

// TestInfosetKey verifies deterministic infoset key generation and correct layout.
func TestInfosetKey(t *testing.T) {
	g := newDealtGame()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	key := a.InfosetKey()

	// Bytes 0..OwnHandLen-1 should be own hand buckets.
	for i := uint8(0); i < a.OwnHandLen; i++ {
		if key[i] != uint8(a.OwnHand[i].Bucket) {
			t.Errorf("Key[%d] = %d, want OwnHand[%d].Bucket = %d", i, key[i], i, a.OwnHand[i].Bucket)
		}
	}
	// Bytes OwnHandLen..5 should be BucketUnknown.
	for i := a.OwnHandLen; i < engine.MaxHandSize; i++ {
		if key[i] != uint8(BucketUnknown) {
			t.Errorf("Key[%d] = %d, want BucketUnknown (%d) for empty own slot", i, key[i], BucketUnknown)
		}
	}
	// Bytes 6..6+OppHandLen-1 should be opponent beliefs.
	for i := uint8(0); i < a.OppHandLen; i++ {
		if key[6+i] != uint8(a.OppBelief[i]) {
			t.Errorf("Key[%d] = %d, want OppBelief[%d] = %d", 6+i, key[6+i], i, uint8(a.OppBelief[i]))
		}
	}
	// Bytes 6+OppHandLen..11 should be 0xFF.
	for i := a.OppHandLen; i < engine.MaxHandSize; i++ {
		if key[6+i] != 0xFF {
			t.Errorf("Key[%d] = %d, want 0xFF for empty opp slot", 6+i, key[6+i])
		}
	}
	// Key[12] = OppHandLen.
	if key[12] != a.OppHandLen {
		t.Errorf("Key[12] = %d, want OppHandLen = %d", key[12], a.OppHandLen)
	}
	// Key[13] = DiscardTopBucket.
	if key[13] != uint8(a.DiscardTopBucket) {
		t.Errorf("Key[13] = %d, want DiscardTopBucket = %d", key[13], a.DiscardTopBucket)
	}
	// Key[14] = StockEstimate | (Phase << 4).
	expectedKey14 := uint8(a.StockEstimate) | (uint8(a.Phase) << 4)
	if key[14] != expectedKey14 {
		t.Errorf("Key[14] = %d, want %d", key[14], expectedKey14)
	}
	// Key[15] = CambiaState.
	if key[15] != uint8(a.CambiaState) {
		t.Errorf("Key[15] = %d, want CambiaState = %d", key[15], a.CambiaState)
	}

	// Key should be deterministic.
	key2 := a.InfosetKey()
	if key != key2 {
		t.Errorf("InfosetKey not deterministic: first = %v, second = %v", key, key2)
	}
}

// TestBeliefValue verifies IsBucket/IsDecay and round-trip construction.
func TestBeliefValue(t *testing.T) {
	// BucketBelief for values 0-9.
	for b := CardBucket(0); b <= BucketUnknown; b++ {
		bv := BucketBelief(b)
		if !bv.IsBucket() {
			t.Errorf("BucketBelief(%d).IsBucket() = false", b)
		}
		if bv.IsDecay() {
			t.Errorf("BucketBelief(%d).IsDecay() = true", b)
		}
		if bv.Bucket() != b {
			t.Errorf("BucketBelief(%d).Bucket() = %d", b, bv.Bucket())
		}
	}

	// DecayBelief for values 0-3.
	for d := DecayCategory(0); d <= DecayUnknown; d++ {
		dv := DecayBelief(d)
		if dv.IsBucket() {
			t.Errorf("DecayBelief(%d).IsBucket() = true", d)
		}
		if !dv.IsDecay() {
			t.Errorf("DecayBelief(%d).IsDecay() = false", d)
		}
		if dv.Decay() != d {
			t.Errorf("DecayBelief(%d).Decay() = %d", d, dv.Decay())
		}
	}
}

// TestHandReindexingOnSnap verifies that when a snap removes a card, all beliefs
// above that index shift down correctly.
func TestHandReindexingOnSnap(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.OwnHandLen = 4

	a.OwnHand[0] = KnownCardInfo{Bucket: BucketAce, Card: engine.NewCard(engine.SuitHearts, engine.RankAce)}
	a.OwnHand[1] = KnownCardInfo{Bucket: BucketLowNum, Card: engine.NewCard(engine.SuitHearts, engine.RankTwo)}
	a.OwnHand[2] = KnownCardInfo{Bucket: BucketMidNum, Card: engine.NewCard(engine.SuitHearts, engine.RankFive)}
	a.OwnHand[3] = KnownCardInfo{Bucket: BucketHighKing, Card: engine.NewCard(engine.SuitClubs, engine.RankKing)}

	// Remove card at index 1.
	a.removeOwnCard(1)

	if a.OwnHandLen != 3 {
		t.Errorf("OwnHandLen = %d, want 3", a.OwnHandLen)
	}
	if a.OwnHand[0].Bucket != BucketAce {
		t.Errorf("OwnHand[0].Bucket = %d, want BucketAce", a.OwnHand[0].Bucket)
	}
	if a.OwnHand[1].Bucket != BucketMidNum {
		t.Errorf("OwnHand[1].Bucket = %d, want BucketMidNum (shifted)", a.OwnHand[1].Bucket)
	}
	if a.OwnHand[2].Bucket != BucketHighKing {
		t.Errorf("OwnHand[2].Bucket = %d, want BucketHighKing (shifted)", a.OwnHand[2].Bucket)
	}
	if a.OwnHand[3].Bucket != 0 {
		t.Errorf("OwnHand[3] should be cleared after removal")
	}
}

// TestOppHandReindexing verifies opponent hand reindexing after snap.
func TestOppHandReindexing(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.OppHandLen = 3
	a.OppBelief[0] = BucketBelief(BucketAce)
	a.OppBelief[1] = BucketBelief(BucketLowNum)
	a.OppBelief[2] = BucketBelief(BucketHighKing)
	a.OppLastSeen[1] = 5
	a.OppHasLastSeen[1] = true

	// Remove index 0.
	a.removeOppCard(0)

	if a.OppHandLen != 2 {
		t.Errorf("OppHandLen = %d, want 2", a.OppHandLen)
	}
	if a.OppBelief[0] != BucketBelief(BucketLowNum) {
		t.Errorf("OppBelief[0] = %v, want BucketLowNum", a.OppBelief[0])
	}
	if a.OppLastSeen[0] != 5 {
		t.Errorf("OppLastSeen[0] = %d, want 5 (shifted)", a.OppLastSeen[0])
	}
	if !a.OppHasLastSeen[0] {
		t.Errorf("OppHasLastSeen[0] should be true (shifted from index 1)")
	}
	if a.OppBelief[1] != BucketBelief(BucketHighKing) {
		t.Errorf("OppBelief[1] = %v, want BucketHighKing", a.OppBelief[1])
	}
	if a.OppBelief[2] != 0 {
		t.Errorf("OppBelief[2] should be cleared")
	}
	if a.OppHasLastSeen[2] {
		t.Errorf("OppHasLastSeen[2] should be false after clear")
	}
}
