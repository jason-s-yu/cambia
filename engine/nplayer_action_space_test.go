package engine

import (
	"fmt"
	"testing"
)

// nplayer_action_space_test.go pins the action space every handler reachable from
// ApplyNPlayerAction records LastAction.ActionIdx in (cambia-1548). The 146-action
// 2-player space and the 620-action N-player space overlap, so a recorded index carries
// no space of its own: an observer that guesses reads a different action than the one
// applied. Each case below drives one N-player action above two seats and asserts the
// recorded index round-trips through the matching N-player decoder, that the record says
// it is in the N-player space, and (where the ranges overlap) names what the 2-player
// decoders would have made of the same index.

// twoPlayerClaim describes what the 2-player decoders read an index as, or "" when none
// of them claims it.
func twoPlayerClaim(idx uint16) string {
	switch idx {
	case ActionDrawStockpile:
		return "DrawStockpile"
	case ActionDrawDiscard:
		return "DrawDiscard"
	case ActionCallCambia:
		return "CallCambia"
	case ActionDiscardNoAbility:
		return "DiscardNoAbility"
	case ActionDiscardWithAbility:
		return "DiscardWithAbility"
	case ActionKingSwapNo:
		return "KingSwapNo"
	case ActionKingSwapYes:
		return "KingSwapYes"
	case ActionPassSnap:
		return "PassSnap"
	}
	if t, ok := ActionIsReplace(idx); ok {
		return fmt.Sprintf("Replace(%d)", t)
	}
	if t, ok := ActionIsPeekOwn(idx); ok {
		return fmt.Sprintf("PeekOwn(%d)", t)
	}
	if t, ok := ActionIsPeekOther(idx); ok {
		return fmt.Sprintf("PeekOther(%d)", t)
	}
	if own, opp, ok := ActionIsBlindSwap(idx); ok {
		return fmt.Sprintf("BlindSwap(%d,%d)", own, opp)
	}
	if own, opp, ok := ActionIsKingLook(idx); ok {
		return fmt.Sprintf("KingLook(%d,%d)", own, opp)
	}
	if t, ok := ActionIsSnapOwn(idx); ok {
		return fmt.Sprintf("SnapOwn(%d)", t)
	}
	if t, ok := ActionIsSnapOpponent(idx); ok {
		return fmt.Sprintf("SnapOpponent(%d)", t)
	}
	if own, slot, ok := ActionIsSnapOpponentMove(idx); ok {
		return fmt.Sprintf("SnapOpponentMove(%d,%d)", own, slot)
	}
	return ""
}

// setSeatHand overwrites a seat's hand with the given cards.
func setSeatHand(g *GameState, seat uint8, cards ...Card) {
	for i, c := range cards {
		g.Players[seat].Hand[i] = c
	}
	for i := len(cards); i < MaxHandSize; i++ {
		g.Players[seat].Hand[i] = EmptyCard
	}
	g.Players[seat].HandLen = uint8(len(cards))
}

// armSnapPhase opens a snap window with a single snapper, which is the shape that lets a
// race-ON commit resolve on the same action a race-OFF snap resolves on.
func armSnapPhase(g *GameState, snapper uint8, rank uint8) {
	g.Snap = SnapState{}
	g.Snap.Active = true
	g.Snap.DiscardedRank = rank
	g.Snap.Snappers[0] = snapper
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	for i := range g.Snap.Commits {
		g.Snap.Commits[i] = SnapCommitNone
	}
}

func TestNPlayerActionsRecordTheNPlayerSpace(t *testing.T) {
	// A 4-seat table for every case but PassSnap, which also runs at 6 to cover the seat
	// count the ticket names (a 2-player decoder reads N-player PassSnap as BlindSwap, so
	// the acting seat and its opponent count do not change what is recorded).
	// twoPlayer is what the 2-player decoders make of the recorded index, "" when none
	// of them claims it. It is the overlap the namespace tag exists for: where the two
	// spaces disagree it names the action an untagged observer acted on, and where they
	// agree (the shared prefix) it repeats the N-player action, which is what lets those
	// handlers leave the tag to the dispatcher.
	cases := []struct {
		name      string
		seats     uint8
		action    uint16
		twoPlayer string
		setup     func(g *GameState)
		decode    func(t *testing.T, idx uint16)
	}{
		{
			name:      "PeekOther(slot 2, opp 2)",
			twoPlayer: "BlindSwap(1,4)",
			seats:     4,
			action:    NPlayerEncodePeekOther(2, 2),
			setup: func(g *GameState) {
				g.Pending.Type = PendingPeekOther
				g.Pending.PlayerID = 0
			},
			decode: func(t *testing.T, idx uint16) {
				slot, oppRel, ok := NPlayerDecodePeekOther(idx)
				if !ok || slot != 2 || oppRel != 2 {
					t.Errorf("NPlayerDecodePeekOther(%d) = (%d, %d, %v), want (2, 2, true)", idx, slot, oppRel, ok)
				}
			},
		},
		{
			name:      "BlindSwap(own 1, oppSlot 2, opp 2)",
			twoPlayer: "SnapOpponentMove(1,1)",
			seats:     4,
			action:    NPlayerEncodeBlindSwap(1, 2, 2),
			setup: func(g *GameState) {
				g.Pending.Type = PendingBlindSwap
				g.Pending.PlayerID = 0
			},
			decode: func(t *testing.T, idx uint16) {
				own, oppSlot, oppRel, ok := NPlayerDecodeBlindSwap(idx)
				if !ok || own != 1 || oppSlot != 2 || oppRel != 2 {
					t.Errorf("NPlayerDecodeBlindSwap(%d) = (%d, %d, %d, %v), want (1, 2, 2, true)", idx, own, oppSlot, oppRel, ok)
				}
			},
		},
		{
			name:      "KingLook(own 1, oppSlot 2, opp 2)",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerEncodeKingLook(1, 2, 2),
			setup: func(g *GameState) {
				g.Pending.Type = PendingKingLook
				g.Pending.PlayerID = 0
			},
			decode: func(t *testing.T, idx uint16) {
				own, oppSlot, oppRel, ok := NPlayerDecodeKingLook(idx)
				if !ok || own != 1 || oppSlot != 2 || oppRel != 2 {
					t.Errorf("NPlayerDecodeKingLook(%d) = (%d, %d, %d, %v), want (1, 2, 2, true)", idx, own, oppSlot, oppRel, ok)
				}
			},
		},
		{
			name:      "KingSwapYes",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerActionKingSwapYes,
			setup: func(g *GameState) {
				g.Pending.Type = PendingKingDecision
				g.Pending.PlayerID = 0
				g.Pending.Data[0] = 1 // own slot
				g.Pending.Data[1] = 2 // target slot
				g.Pending.Data[2] = uint8(g.Players[0].Hand[1])
				g.Pending.Data[3] = 3 // target seat
			},
			decode: func(t *testing.T, idx uint16) {
				if idx != NPlayerActionKingSwapYes {
					t.Errorf("recorded %d, want NPlayerActionKingSwapYes (%d)", idx, NPlayerActionKingSwapYes)
				}
			},
		},
		{
			name:      "KingSwapNo",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerActionKingSwapNo,
			setup: func(g *GameState) {
				g.Pending.Type = PendingKingDecision
				g.Pending.PlayerID = 0
				g.Pending.Data[0] = 1 // own slot
				g.Pending.Data[1] = 2 // target slot
				g.Pending.Data[2] = uint8(g.Players[0].Hand[1])
				g.Pending.Data[3] = 3 // target seat
			},
			decode: func(t *testing.T, idx uint16) {
				if idx != NPlayerActionKingSwapNo {
					t.Errorf("recorded %d, want NPlayerActionKingSwapNo (%d)", idx, NPlayerActionKingSwapNo)
				}
			},
		},
		{
			// A shared-prefix action: both spaces encode Replace(2) as 7, so what the
			// dispatcher has to get right is the tag, not the index.
			name:      "Replace(2)",
			twoPlayer: "Replace(2)",
			seats:     4,
			action:    NPlayerEncodeReplace(2),
			setup: func(g *GameState) {
				g.Pending.Type = PendingDiscard
				g.Pending.PlayerID = 0
				g.Pending.Data[0] = uint8(NewCard(SuitHearts, RankFour))
				g.Pending.Data[1] = DrawnFromStockpile
			},
			decode: func(t *testing.T, idx uint16) {
				slot, ok := NPlayerDecodeReplace(idx)
				if !ok || slot != 2 {
					t.Errorf("NPlayerDecodeReplace(%d) = (%d, %v), want (2, true)", idx, slot, ok)
				}
			},
		},
		{
			name:      "PeekOwn(3)",
			twoPlayer: "PeekOwn(3)",
			seats:     4,
			action:    NPlayerEncodePeekOwn(3),
			setup: func(g *GameState) {
				g.Pending.Type = PendingPeekOwn
				g.Pending.PlayerID = 0
			},
			decode: func(t *testing.T, idx uint16) {
				slot, ok := NPlayerDecodePeekOwn(idx)
				if !ok || slot != 3 {
					t.Errorf("NPlayerDecodePeekOwn(%d) = (%d, %v), want (3, true)", idx, slot, ok)
				}
			},
		},
		{
			name:      "SnapOwn(3)",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerEncodeSnapOwn(3),
			setup: func(g *GameState) {
				armSnapPhase(g, 0, g.Players[0].Hand[3].Rank())
			},
			decode: func(t *testing.T, idx uint16) {
				slot, ok := NPlayerDecodeSnapOwn(idx)
				if !ok || slot != 3 {
					t.Errorf("NPlayerDecodeSnapOwn(%d) = (%d, %v), want (3, true)", idx, slot, ok)
				}
			},
		},
		{
			name:      "SnapOpponent(slot 1, opp 2)",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerEncodeSnapOpponent(1, 2),
			setup: func(g *GameState) {
				armSnapPhase(g, 0, g.Players[3].Hand[1].Rank())
			},
			decode: func(t *testing.T, idx uint16) {
				slot, oppRel, ok := NPlayerDecodeSnapOpponent(idx)
				if !ok || slot != 1 || oppRel != 2 {
					t.Errorf("NPlayerDecodeSnapOpponent(%d) = (%d, %d, %v), want (1, 2, true)", idx, slot, oppRel, ok)
				}
			},
		},
		{
			name:      "SnapOpponentMove(own 2)",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerEncodeSnapOpponentMove(2),
			setup: func(g *GameState) {
				armSnapPhase(g, 0, g.Players[3].Hand[1].Rank())
				g.Pending.Type = PendingSnapMove
				g.Pending.PlayerID = 0
				g.Pending.Data[0] = 3 // victim seat
				g.Pending.Data[1] = 1 // vacated slot
			},
			decode: func(t *testing.T, idx uint16) {
				own, ok := NPlayerDecodeSnapOpponentMove(idx)
				if !ok || own != 2 {
					t.Errorf("NPlayerDecodeSnapOpponentMove(%d) = (%d, %v), want (2, true)", idx, own, ok)
				}
			},
		},
		{
			name:      "PassSnap at 4 seats",
			twoPlayer: "",
			seats:     4,
			action:    NPlayerActionPassSnap,
			setup: func(g *GameState) {
				armSnapPhase(g, 0, g.Players[0].Hand[0].Rank())
			},
			decode: func(t *testing.T, idx uint16) {
				if idx != NPlayerActionPassSnap {
					t.Errorf("recorded %d, want NPlayerActionPassSnap (%d)", idx, NPlayerActionPassSnap)
				}
			},
		},
		{
			name:      "PassSnap at 6 seats",
			twoPlayer: "",
			seats:     6,
			action:    NPlayerActionPassSnap,
			setup: func(g *GameState) {
				armSnapPhase(g, 0, g.Players[0].Hand[0].Rank())
			},
			decode: func(t *testing.T, idx uint16) {
				if idx != NPlayerActionPassSnap {
					t.Errorf("recorded %d, want NPlayerActionPassSnap (%d)", idx, NPlayerActionPassSnap)
				}
			},
		},
	}

	for _, race := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/race=%v", tc.name, race), func(t *testing.T) {
				rules := DefaultHouseRules()
				rules.NumPlayers = tc.seats
				rules.SnapRace = race
				rules.AllowOpponentSnapping = true
				g := NewGame(20260901, rules)
				g.Deal()
				g.CurrentPlayer = 0
				// A known hand per seat keeps the snap ranks and swap targets stable.
				for seat := uint8(0); seat < tc.seats; seat++ {
					setSeatHand(&g, seat,
						NewCard(SuitHearts, RankTwo+seat),
						NewCard(SuitClubs, RankFive+seat),
						NewCard(SuitSpades, RankSeven),
						NewCard(SuitDiamonds, RankNine))
				}
				tc.setup(&g)

				if err := g.ApplyNPlayerAction(tc.action); err != nil {
					t.Fatalf("ApplyNPlayerAction(%d): %v", tc.action, err)
				}

				got := g.LastAction.ActionIdx
				if got != tc.action {
					t.Fatalf("LastAction.ActionIdx = %d, want the driven index %d", got, tc.action)
				}
				if g.LastAction.ActionSpace() != ActionSpaceNPlayer {
					t.Errorf("LastAction.ActionSpace() = %d, want ActionSpaceNPlayer (%d)",
						g.LastAction.ActionSpace(), ActionSpaceNPlayer)
				}
				tc.decode(t, got)

				// The overlap the tag exists for.
				if claim := twoPlayerClaim(got); claim != tc.twoPlayer {
					t.Errorf("2-player reading of index %d = %q, want %q", got, claim, tc.twoPlayer)
				}
			})
		}
	}
}

// TestNPlayerSharedPrefixDecodesInBothSpaces pins the other half of the tag's contract:
// the first seventeen indices hold the same meaning in both spaces, so the handlers that
// record one of them can leave the tag as the dispatcher set it.
func TestNPlayerSharedPrefixDecodesInBothSpaces(t *testing.T) {
	if NPlayerActionDrawStockpile != ActionDrawStockpile ||
		NPlayerActionDrawDiscard != ActionDrawDiscard ||
		NPlayerActionCallCambia != ActionCallCambia ||
		NPlayerActionDiscardNoAbility != ActionDiscardNoAbility ||
		NPlayerActionDiscardWithAbility != ActionDiscardWithAbility {
		t.Fatal("the five fixed indices must be identical in both spaces")
	}
	for slot := uint8(0); slot < MaxHandSize; slot++ {
		if NPlayerEncodeReplace(slot) != EncodeReplace(slot) {
			t.Errorf("Replace(%d): N-player %d != legacy %d", slot, NPlayerEncodeReplace(slot), EncodeReplace(slot))
		}
		if NPlayerEncodePeekOwn(slot) != EncodePeekOwn(slot) {
			t.Errorf("PeekOwn(%d): N-player %d != legacy %d", slot, NPlayerEncodePeekOwn(slot), EncodePeekOwn(slot))
		}
	}
}

// TestTwoSeatNPlayerDispatchRecordsTheLegacySpace pins the seat-count half of the tag: at
// two seats ApplyNPlayerAction resolves abilities and snaps through the 2-player handlers,
// so what it records is a legacy index and the tag has to say so.
func TestTwoSeatNPlayerDispatchRecordsTheLegacySpace(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	rules.AllowOpponentSnapping = true
	g := NewGame(20260901, rules)
	g.Deal()
	g.CurrentPlayer = 0
	setSeatHand(&g, 0, NewCard(SuitHearts, RankTwo), NewCard(SuitClubs, RankFive))
	setSeatHand(&g, 1, NewCard(SuitSpades, RankSeven), NewCard(SuitDiamonds, RankNine))

	g.Pending.Type = PendingPeekOther
	g.Pending.PlayerID = 0
	if err := g.ApplyNPlayerAction(NPlayerEncodePeekOther(1, 0)); err != nil {
		t.Fatalf("ApplyNPlayerAction: %v", err)
	}
	if g.LastAction.ActionSpace() != ActionSpaceLegacy {
		t.Errorf("ActionSpace() = %d, want ActionSpaceLegacy at two seats", g.LastAction.ActionSpace())
	}
	if got, want := g.LastAction.ActionIdx, EncodePeekOther(1); got != want {
		t.Errorf("ActionIdx = %d, want the legacy index %d", got, want)
	}
}
