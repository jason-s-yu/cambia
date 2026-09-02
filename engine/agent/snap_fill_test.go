package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// snap_fill_test.go covers the RULES.md 5 fill: after a successful opponent snap the
// snapper moves one of their own cards into the slot the snap emptied. The card keeps its
// identity, so what the mover knew about it is what they know about its new home; the
// belief used to be dropped on the floor, which handed a peeked card back as unknown
// (cambia-1552).

// snapFillRules returns rules with opponent snapping on, which is what makes the fill
// reachable at all (it is on in the engine defaults and competitive.yaml, off in
// default.yaml and the training configs).
func snapFillRules(seats uint8) engine.HouseRules {
	r := engine.DefaultHouseRules()
	r.NumPlayers = seats
	r.AllowOpponentSnapping = true
	return r
}

// seedHand overwrites a seat's hand.
func seedHand(g *engine.GameState, seat uint8, cards ...engine.Card) {
	for i, c := range cards {
		g.Players[seat].Hand[i] = c
	}
	for i := len(cards); i < engine.MaxHandSize; i++ {
		g.Players[seat].Hand[i] = engine.EmptyCard
	}
	g.Players[seat].HandLen = uint8(len(cards))
}

// openSnapWindow puts one card of the given rank on the discard pile and opens a snap
// window with a single snapper, without disturbing the hands.
func openSnapWindow(g *engine.GameState, snapper uint8, top engine.Card) {
	g.DiscardPile[g.DiscardLen] = top
	g.DiscardLen++
	g.Snap = engine.SnapState{}
	g.Snap.Active = true
	g.Snap.DiscardedRank = top.Rank()
	g.Snap.Snappers[0] = snapper
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	for i := range g.Snap.Commits {
		g.Snap.Commits[i] = engine.SnapCommitNone
	}
}

// TestSnapFillCarriesTheMoversKnowledge peeks a slot, pays the fill with it, and reads the
// belief about the slot it landed in from both sides of the table.
func TestSnapFillCarriesTheMoversKnowledge(t *testing.T) {
	g := engine.NewGame(20261552, snapFillRules(2))
	g.Deal()
	g.CurrentPlayer = 0
	// A King on top of the discard matches nothing, so resolving the peek below opens no
	// snap window of its own.
	g.DiscardPile[0] = engine.NewCard(engine.SuitSpades, engine.RankKing)
	g.DiscardLen = 1
	paid := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		paid,
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	seedHand(&g, 1,
		engine.NewCard(engine.SuitClubs, engine.RankNine),
		engine.NewCard(engine.SuitDiamonds, engine.RankFive),
		engine.NewCard(engine.SuitSpades, engine.RankSix),
		engine.NewCard(engine.SuitHearts, engine.RankEight))

	mover := NewAgentState(0, 1, 1, 3)
	victim := NewAgentState(1, 0, 1, 3)
	mover.Initialize(&g)
	victim.Initialize(&g)

	// The mover peeks the card it will pay with.
	g.Pending.Type = engine.PendingPeekOwn
	g.Pending.PlayerID = 0
	if err := g.ApplyAction(engine.EncodePeekOwn(1)); err != nil {
		t.Fatalf("peek own: %v", err)
	}
	mover.Update(&g)
	victim.Update(&g)
	paidBucket := CardToBucket(paid)
	if mover.OwnHand[1].Bucket != paidBucket {
		t.Fatalf("peek did not record the card: bucket %d, want %d", mover.OwnHand[1].Bucket, paidBucket)
	}
	peekTurn := mover.OwnHand[1].LastSeenTurn

	// The mover snaps the victim's nine, which empties slot 0 of the victim's hand.
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitHearts, engine.RankNine))
	if err := g.ApplyAction(engine.EncodeSnapOpponent(0)); err != nil {
		t.Fatalf("snap opponent: %v", err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatalf("snap did not succeed, so no fill is owed")
	}
	mover.Update(&g)
	victim.Update(&g)

	// The mover pays the fill with the card it peeked.
	if err := g.ApplyAction(engine.EncodeSnapOpponentMove(1, 0)); err != nil {
		t.Fatalf("snap opponent move: %v", err)
	}
	if g.Players[1].Hand[0] != paid {
		t.Fatalf("engine put %v in the vacated slot, want the paid card %v", g.Players[1].Hand[0], paid)
	}
	mover.Update(&g)
	victim.Update(&g)

	// The mover still knows the card: it is the same card, in a hand it can no longer play.
	if got := mover.OppBelief[0]; !got.IsBucket() || got.Bucket() != paidBucket {
		t.Errorf("mover belief about the filled slot = %v, want bucket %d", got, paidBucket)
	}
	if !mover.OppHasLastSeen[0] || mover.OppLastSeen[0] != peekTurn {
		t.Errorf("mover last-seen for the filled slot = (%v, %d), want (true, %d)",
			mover.OppHasLastSeen[0], mover.OppLastSeen[0], peekTurn)
	}
	if tag := mover.SlotTags[OppSlotsStart]; tag != TagPrivOwn {
		t.Errorf("EP-PBS tag for the filled slot = %d, want TagPrivOwn (%d)", tag, TagPrivOwn)
	}
	if b := mover.SlotBuckets[OppSlotsStart]; b != paidBucket {
		t.Errorf("EP-PBS bucket for the filled slot = %d, want %d", b, paidBucket)
	}

	// The victim was handed a card it never saw, so its own belief about the face is
	// unchanged by the fill: the transfer is the MOVER's knowledge, not a public reveal.
	if victim.OwnHand[0].Bucket != BucketUnknown {
		t.Errorf("victim believes it knows the card it was handed (bucket %d)", victim.OwnHand[0].Bucket)
	}
	// The victim did watch the mover peek that card, though, so it knows the mover knows
	// the card it just received: the tag travels with the card (cambia-1690).
	if tag := victim.SlotTags[0]; tag != TagPrivOpp {
		t.Errorf("victim EP-PBS tag for the received slot = %d, want TagPrivOpp (%d)", tag, TagPrivOpp)
	}
}

// TestSnapFillCarriesTheKnowledgeToTheReceiver is the mirror of the case above: the seat
// handed the fill keeps what it already knew about the card, which the receiving side used
// to drop on the floor (cambia-1690).
func TestSnapFillCarriesTheKnowledgeToTheReceiver(t *testing.T) {
	g := engine.NewGame(20261690, snapFillRules(2))
	g.Deal()
	g.CurrentPlayer = 0
	g.DiscardPile[0] = engine.NewCard(engine.SuitSpades, engine.RankKing)
	g.DiscardLen = 1

	// Seat 1 will snap seat 0's slot 0 and pay the fill with its own slot 1.
	paid := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	seedHand(&g, 0, engine.NewCard(engine.SuitClubs, engine.RankFour),
		engine.NewCard(engine.SuitDiamonds, engine.RankFive),
		engine.NewCard(engine.SuitSpades, engine.RankSix))
	seedHand(&g, 1, engine.NewCard(engine.SuitClubs, engine.RankTwo), paid,
		engine.NewCard(engine.SuitDiamonds, engine.RankThree))

	// Seat 0 is the receiver here, so it is seat 0 that peeks the card it will be handed.
	receiver := NewAgentState(0, 1, 1, 3)
	mover := NewAgentState(1, 0, 1, 3)
	receiver.Initialize(&g)
	mover.Initialize(&g)

	paidBucket := CardToBucket(paid)
	g.LastAction = engine.LastActionInfo{}
	g.Pending.Type = engine.PendingPeekOther
	g.Pending.PlayerID = 0
	if err := g.ApplyAction(engine.EncodePeekOther(1)); err != nil {
		t.Fatalf("peek other: %v", err)
	}
	receiver.Update(&g)
	mover.Update(&g)
	if got := receiver.OppBelief[1]; !got.IsBucket() || got.Bucket() != paidBucket {
		t.Fatalf("peek left the receiver's belief about the paid card at %v", got)
	}
	peekTurn := receiver.OppLastSeen[1]

	// Seat 1 snaps seat 0's slot 0, which empties it and owes seat 0 a card.
	g.CurrentPlayer = 1
	openSnapWindow(&g, 1, engine.NewCard(engine.SuitHearts, engine.RankFour))
	if err := g.ApplyAction(engine.EncodeSnapOpponent(0)); err != nil {
		t.Fatalf("snap opponent: %v", err)
	}
	receiver.Update(&g)
	mover.Update(&g)

	if err := g.ApplyAction(engine.EncodeSnapOpponentMove(1, 0)); err != nil {
		t.Fatalf("snap opponent move: %v", err)
	}
	if g.Players[0].Hand[0] != paid {
		t.Fatalf("engine put %v in the vacated slot, want the paid card %v", g.Players[0].Hand[0], paid)
	}
	receiver.Update(&g)
	mover.Update(&g)

	if receiver.OwnHand[0].Bucket != paidBucket {
		t.Errorf("receiver belief about the card it was handed = %d, want %d",
			receiver.OwnHand[0].Bucket, paidBucket)
	}
	if receiver.OwnHand[0].LastSeenTurn != peekTurn {
		t.Errorf("receiver last-seen for the filled slot = %d, want %d",
			receiver.OwnHand[0].LastSeenTurn, peekTurn)
	}
	// The receiver saw the card and never watched the mover look at it, so the slot it
	// landed in is the receiver's alone to know: the tag travels with the card.
	if tag := receiver.SlotTags[0]; tag != TagPrivOwn {
		t.Errorf("receiver EP-PBS tag for the filled slot = %d, want TagPrivOwn (%d)", tag, TagPrivOwn)
	}
	if b := receiver.SlotBuckets[0]; b != paidBucket {
		t.Errorf("receiver EP-PBS bucket for the filled slot = %d, want %d", b, paidBucket)
	}
}

// TestBlindSwapCarriesTheSwappersKnowledge covers the same information rule for the card
// a blind swap sends away: the swap is blind in the card RECEIVED, not the one given.
func TestBlindSwapCarriesTheSwappersKnowledge(t *testing.T) {
	g := engine.NewGame(20261553, snapFillRules(2))
	g.Deal()
	g.CurrentPlayer = 0
	g.DiscardPile[0] = engine.NewCard(engine.SuitSpades, engine.RankKing)
	g.DiscardLen = 1
	sent := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		sent,
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	seedHand(&g, 1,
		engine.NewCard(engine.SuitClubs, engine.RankNine),
		engine.NewCard(engine.SuitDiamonds, engine.RankFive),
		engine.NewCard(engine.SuitSpades, engine.RankSix),
		engine.NewCard(engine.SuitHearts, engine.RankEight))

	mover := NewAgentState(0, 1, 1, 3)
	mover.Initialize(&g)

	g.Pending.Type = engine.PendingPeekOwn
	g.Pending.PlayerID = 0
	if err := g.ApplyAction(engine.EncodePeekOwn(1)); err != nil {
		t.Fatalf("peek own: %v", err)
	}
	mover.Update(&g)
	sentBucket := CardToBucket(sent)
	peekTurn := mover.OwnHand[1].LastSeenTurn

	g.Pending.Type = engine.PendingBlindSwap
	g.Pending.PlayerID = 0
	if err := g.ApplyAction(engine.EncodeBlindSwap(1, 2)); err != nil {
		t.Fatalf("blind swap: %v", err)
	}
	mover.Update(&g)

	if got := mover.OppBelief[2]; !got.IsBucket() || got.Bucket() != sentBucket {
		t.Errorf("belief about the slot the swapper filled = %v, want bucket %d", got, sentBucket)
	}
	if !mover.OppHasLastSeen[2] || mover.OppLastSeen[2] != peekTurn {
		t.Errorf("last-seen for the swapped-into slot = (%v, %d), want (true, %d)",
			mover.OppHasLastSeen[2], mover.OppLastSeen[2], peekTurn)
	}
	if tag := mover.SlotTags[OppSlotsStart+2]; tag != TagPrivOwn {
		t.Errorf("EP-PBS tag for the swapped-into slot = %d, want TagPrivOwn (%d)", tag, TagPrivOwn)
	}
	// The card received is blind, so the swapper's own slot goes unknown.
	if mover.OwnHand[1].Bucket != BucketUnknown {
		t.Errorf("swapper believes it knows the card it received blind (bucket %d)", mover.OwnHand[1].Bucket)
	}
	if tag := mover.SlotTags[1]; tag != TagUnk {
		t.Errorf("EP-PBS tag for the swapper's own slot = %d, want TagUnk (%d)", tag, TagUnk)
	}
}

// TestNPlayerSnapFillCarriesTheMoversKnowledge is the 4-seat half: the N-player model
// tracked the mover losing the card and nothing about where it went.
func TestNPlayerSnapFillCarriesTheMoversKnowledge(t *testing.T) {
	g := engine.NewGame(20261554, snapFillRules(4))
	g.Deal()
	g.CurrentPlayer = 0
	g.DiscardPile[0] = engine.NewCard(engine.SuitSpades, engine.RankKing)
	g.DiscardLen = 1
	paid := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		paid,
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	for seat := uint8(1); seat < 4; seat++ {
		seedHand(&g, seat,
			engine.NewCard(engine.SuitClubs, engine.RankFive),
			engine.NewCard(engine.SuitDiamonds, engine.RankNine),
			engine.NewCard(engine.SuitSpades, engine.RankSix),
			engine.NewCard(engine.SuitHearts, engine.RankEight))
	}

	agents := make([]AgentState, 4)
	for seat := uint8(0); seat < 4; seat++ {
		agents[seat] = NewNPlayerAgentState(seat, 4, 1, 3)
		agents[seat].InitializeNPlayer(&g)
	}
	update := func() {
		for i := range agents {
			agents[i].UpdateNPlayer(&g)
		}
	}

	const (
		victim   = uint8(2) // Opponents(0) = [1, 2, 3], so relative index 1
		oppRel   = uint8(1)
		snapSlot = uint8(1)
		paidSlot = uint8(1)
	)
	moverSlot := nplayerSlot(0, paidSlot)
	filledSlot := nplayerSlot(victim, snapSlot)

	// The mover peeks the card it will pay with.
	g.Pending.Type = engine.PendingPeekOwn
	g.Pending.PlayerID = 0
	if err := g.ApplyNPlayerAction(engine.NPlayerEncodePeekOwn(paidSlot)); err != nil {
		t.Fatalf("peek own: %v", err)
	}
	update()
	paidBucket := CardToBucket(paid)
	if !agents[0].NPlayerSlotKnown[moverSlot] || agents[0].NPlayerSlotBuckets[moverSlot] != paidBucket {
		t.Fatalf("peek did not record the card at slot %d", moverSlot)
	}

	// The mover snaps seat 2's nine and pays the fill with the peeked card.
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitHearts, engine.RankNine))
	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOpponent(snapSlot, oppRel)); err != nil {
		t.Fatalf("snap opponent: %v", err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatalf("snap did not succeed, so no fill is owed")
	}
	update()

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOpponentMove(paidSlot)); err != nil {
		t.Fatalf("snap opponent move: %v", err)
	}
	if g.Players[victim].Hand[snapSlot] != paid {
		t.Fatalf("engine put %v in the vacated slot, want the paid card %v",
			g.Players[victim].Hand[snapSlot], paid)
	}
	update()

	if !agents[0].NPlayerSlotKnown[filledSlot] || agents[0].NPlayerSlotBuckets[filledSlot] != paidBucket {
		t.Errorf("mover knowledge of the filled slot: known=%v bucket=%d, want known with bucket %d",
			agents[0].NPlayerSlotKnown[filledSlot], agents[0].NPlayerSlotBuckets[filledSlot], paidBucket)
	}
	if agents[0].NPlayerSlotKnown[moverSlot] {
		t.Errorf("mover still believes it holds the card it paid away")
	}
	for _, observer := range []uint8{1, 2, 3} {
		ao := &agents[observer]
		if !ao.KnowledgeMask[filledSlot][0] {
			t.Errorf("seat %d: the mover is not recorded as knowing the card it paid", observer)
		}
		if ao.NPlayerSlotKnown[filledSlot] {
			t.Errorf("seat %d learned the identity of a card it never saw", observer)
		}
	}
}

// TestSnapFillCarriesAPublicFaceAsPublic pins the carry rule on the actor's side of the
// fill: the destination slot takes the tag the source slot carried, so a card BOTH seats
// had seen stays public instead of collapsing to the mover's private knowledge, which is
// what forcing TagPrivOwn used to do (cambia-1690).
func TestSnapFillCarriesAPublicFaceAsPublic(t *testing.T) {
	g := engine.NewGame(20261690, snapFillRules(2))
	g.Deal()
	g.CurrentPlayer = 0
	g.DiscardPile[0] = engine.NewCard(engine.SuitSpades, engine.RankKing)
	g.DiscardLen = 1
	paid := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		paid,
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	seedHand(&g, 1,
		engine.NewCard(engine.SuitClubs, engine.RankNine),
		engine.NewCard(engine.SuitDiamonds, engine.RankFive),
		engine.NewCard(engine.SuitSpades, engine.RankSix),
		engine.NewCard(engine.SuitHearts, engine.RankEight))

	mover := NewAgentState(0, 1, 1, 3)
	victim := NewAgentState(1, 0, 1, 3)
	mover.Initialize(&g)
	victim.Initialize(&g)
	paidBucket := CardToBucket(paid)

	// The mover peeks the card it will pay with.
	g.Pending.Type = engine.PendingPeekOwn
	g.Pending.PlayerID = 0
	if err := g.ApplyAction(engine.EncodePeekOwn(1)); err != nil {
		t.Fatalf("peek own: %v", err)
	}
	mover.Update(&g)
	victim.Update(&g)

	// The victim peeks the same card, so both seats have now seen it.
	g.CurrentPlayer = 1
	g.Pending.Type = engine.PendingPeekOther
	g.Pending.PlayerID = 1
	if err := g.ApplyAction(engine.EncodePeekOther(1)); err != nil {
		t.Fatalf("peek other: %v", err)
	}
	mover.Update(&g)
	victim.Update(&g)
	if tag := mover.SlotTags[1]; tag != TagPub {
		t.Fatalf("mover own slot 1 tag = %d after both seats peeked it, want TagPub (%d)", tag, TagPub)
	}

	// The mover snaps the victim's nine and pays the fill with that public card.
	g.CurrentPlayer = 0
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitHearts, engine.RankNine))
	if err := g.ApplyAction(engine.EncodeSnapOpponent(0)); err != nil {
		t.Fatalf("snap opponent: %v", err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatalf("snap did not succeed, so no fill is owed")
	}
	mover.Update(&g)
	victim.Update(&g)

	if err := g.ApplyAction(engine.EncodeSnapOpponentMove(1, 0)); err != nil {
		t.Fatalf("snap opponent move: %v", err)
	}
	if g.Players[1].Hand[0] != paid {
		t.Fatalf("engine put %v in the vacated slot, want the paid card %v", g.Players[1].Hand[0], paid)
	}
	mover.Update(&g)
	victim.Update(&g)

	if tag := mover.SlotTags[OppSlotsStart]; tag != TagPub {
		t.Errorf("mover EP-PBS tag for the filled slot = %d, want TagPub (%d)", tag, TagPub)
	}
	if b := mover.SlotBuckets[OppSlotsStart]; b != paidBucket {
		t.Errorf("mover EP-PBS bucket for the filled slot = %d, want %d", b, paidBucket)
	}
	// The victim saw that card too, so from its seat the slot it received is public and
	// its own belief about the face is the one it peeked.
	if victim.OwnHand[0].Bucket != paidBucket {
		t.Errorf("victim belief about the card it was handed = %d, want %d",
			victim.OwnHand[0].Bucket, paidBucket)
	}
	if tag := victim.SlotTags[0]; tag != TagPub {
		t.Errorf("victim EP-PBS tag for the received slot = %d, want TagPub (%d)", tag, TagPub)
	}
}
