package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// nplayer_king_test.go covers the N-player King look and swap against the belief state of
// three seats at once: the actor, the seat it named, and a bystander (cambia-1548). The
// look used to be recorded as a 2-player BlindSwap, so nplayerProcessKingLook never ran
// above two seats, and the swap read its target seat from LastAction.RevealedOwner, which
// the look sets to the ACTOR, so every N-player King swap exchanged knowledge between two
// slots of the actor's own hand.

// kingLookSetup builds a 4-seat table with a King look armed for seat 0 and returns the
// game plus one agent per seat.
func kingLookSetup(t *testing.T) (engine.GameState, []AgentState) {
	t.Helper()
	rules := engine.DefaultHouseRules()
	rules.NumPlayers = 4
	g := engine.NewGame(20261548, rules)
	g.Deal()
	g.CurrentPlayer = 0

	agents := make([]AgentState, 4)
	for seat := uint8(0); seat < 4; seat++ {
		agents[seat] = NewNPlayerAgentState(seat, 4, 1, 5)
		agents[seat].InitializeNPlayer(&g)
	}

	g.Pending.Type = engine.PendingKingLook
	g.Pending.PlayerID = 0
	return g, agents
}

func updateAll(g *engine.GameState, agents []AgentState) {
	for i := range agents {
		agents[i].UpdateNPlayer(g)
	}
}

func TestNPlayerKingLookAndSwapKnowledge(t *testing.T) {
	g, agents := kingLookSetup(t)

	const (
		actor      = uint8(0)
		ownSlot    = uint8(1)
		targetSlot = uint8(2)
		// Opponents(0) is [1, 2, 3], so relative index 1 names seat 2.
		oppRel = uint8(1)
		target = uint8(2)
	)
	actorSlot := nplayerSlot(actor, ownSlot)
	victimSlot := nplayerSlot(target, targetSlot)
	// The slot the old target-from-RevealedOwner read would have hit instead: the actor's
	// own hand at the opponent's slot index.
	straySlot := nplayerSlot(actor, targetSlot)

	ownCardBefore := g.Players[actor].Hand[ownSlot]
	targetCardBefore := g.Players[target].Hand[targetSlot]

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeKingLook(ownSlot, targetSlot, oppRel)); err != nil {
		t.Fatalf("king look: %v", err)
	}
	if g.LastAction.SwapTargetPlayer() != target {
		t.Fatalf("look recorded SwapTargetPlayer %d, want %d", g.LastAction.SwapTargetPlayer(), target)
	}
	updateAll(&g, agents)

	// The actor's own view: it saw both cards it looked at, and nothing else moved.
	a0 := &agents[actor]
	if !a0.NPlayerSlotKnown[actorSlot] || a0.NPlayerSlotBuckets[actorSlot] != CardToBucket(ownCardBefore) {
		t.Errorf("actor own slot: known=%v bucket=%d, want known with bucket %d",
			a0.NPlayerSlotKnown[actorSlot], a0.NPlayerSlotBuckets[actorSlot], CardToBucket(ownCardBefore))
	}
	if !a0.NPlayerSlotKnown[victimSlot] || a0.NPlayerSlotBuckets[victimSlot] != CardToBucket(targetCardBefore) {
		t.Errorf("actor view of the looked seat: known=%v bucket=%d, want known with bucket %d",
			a0.NPlayerSlotKnown[victimSlot], a0.NPlayerSlotBuckets[victimSlot], CardToBucket(targetCardBefore))
	}
	if a0.NPlayerSlotKnown[straySlot] {
		t.Errorf("actor believes it knows its own slot %d, which the look never touched", targetSlot)
	}

	// The target seat's own view and a bystander's: both learn only WHO looked.
	for _, observer := range []uint8{1, 2, 3} {
		ao := &agents[observer]
		if !ao.KnowledgeMask[actorSlot][actor] || !ao.KnowledgeMask[victimSlot][actor] {
			t.Errorf("seat %d: king look not recorded as seen by the actor (own=%v target=%v)",
				observer, ao.KnowledgeMask[actorSlot][actor], ao.KnowledgeMask[victimSlot][actor])
		}
		if ao.NPlayerSlotKnown[actorSlot] || ao.NPlayerSlotKnown[victimSlot] {
			t.Errorf("seat %d learned a card identity from another seat's king look", observer)
		}
		if ao.KnowledgeMask[straySlot][actor] {
			t.Errorf("seat %d recorded the look against the actor's own slot %d", observer, targetSlot)
		}
	}

	// The swap decision exchanges the two cards the look bound.
	if err := g.ApplyNPlayerAction(engine.NPlayerActionKingSwapYes); err != nil {
		t.Fatalf("king swap: %v", err)
	}
	if g.LastAction.SwapTargetPlayer() != target {
		t.Fatalf("swap recorded SwapTargetPlayer %d, want %d", g.LastAction.SwapTargetPlayer(), target)
	}
	if g.Players[actor].Hand[ownSlot] != targetCardBefore || g.Players[target].Hand[targetSlot] != ownCardBefore {
		t.Fatalf("engine did not swap the pair the look bound")
	}
	updateAll(&g, agents)

	// The actor looked at both cards, so after the swap its belief still matches the table.
	if got, want := a0.NPlayerSlotBuckets[actorSlot], CardToBucket(g.Players[actor].Hand[ownSlot]); !a0.NPlayerSlotKnown[actorSlot] || got != want {
		t.Errorf("after swap, actor own slot: known=%v bucket=%d, want known with bucket %d",
			a0.NPlayerSlotKnown[actorSlot], got, want)
	}
	if got, want := a0.NPlayerSlotBuckets[victimSlot], CardToBucket(g.Players[target].Hand[targetSlot]); !a0.NPlayerSlotKnown[victimSlot] || got != want {
		t.Errorf("after swap, actor view of the target seat: known=%v bucket=%d, want known with bucket %d",
			a0.NPlayerSlotKnown[victimSlot], got, want)
	}
	if a0.NPlayerSlotKnown[straySlot] {
		t.Errorf("the swap moved knowledge into the actor's own slot %d, which never moved", targetSlot)
	}

	// Every other seat records that the actor knows both swapped slots, and still knows
	// neither identity itself.
	for _, observer := range []uint8{1, 2, 3} {
		ao := &agents[observer]
		if !ao.KnowledgeMask[actorSlot][actor] || !ao.KnowledgeMask[victimSlot][actor] {
			t.Errorf("seat %d: after the swap the actor is not recorded as knowing both slots", observer)
		}
		if ao.NPlayerSlotKnown[actorSlot] || ao.NPlayerSlotKnown[victimSlot] {
			t.Errorf("seat %d learned a card identity from another seat's king swap", observer)
		}
		if ao.KnowledgeMask[straySlot][actor] {
			t.Errorf("seat %d recorded the swap against the actor's own slot %d", observer, targetSlot)
		}
	}
}

// TestNPlayerPeekOtherLandsOnTheNamedSeat covers the other half of the decode: the peek
// used to record a 2-player PeekOther index, which drops the opponent the action named.
func TestNPlayerPeekOtherLandsOnTheNamedSeat(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.NumPlayers = 4
	g := engine.NewGame(20261549, rules)
	g.Deal()
	g.CurrentPlayer = 0

	a0 := NewNPlayerAgentState(0, 4, 1, 5)
	a0.InitializeNPlayer(&g)

	const (
		slot   = uint8(2)
		oppRel = uint8(2) // Opponents(0) = [1, 2, 3]
		target = uint8(3)
	)
	g.Pending.Type = engine.PendingPeekOther
	g.Pending.PlayerID = 0
	card := g.Players[target].Hand[slot]

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodePeekOther(slot, oppRel)); err != nil {
		t.Fatalf("peek other: %v", err)
	}
	a0.UpdateNPlayer(&g)

	seen := nplayerSlot(target, slot)
	if !a0.NPlayerSlotKnown[seen] || a0.NPlayerSlotBuckets[seen] != CardToBucket(card) {
		t.Errorf("peeked slot %d: known=%v bucket=%d, want known with bucket %d",
			seen, a0.NPlayerSlotKnown[seen], a0.NPlayerSlotBuckets[seen], CardToBucket(card))
	}
	// The 2-player reading of the same index (PeekOther(slot)) lands on seat 1 slot 2.
	stray := nplayerSlot(1, slot)
	if a0.NPlayerSlotKnown[stray] {
		t.Errorf("peek landed on seat 1 slot %d, which the action never named", slot)
	}
}
