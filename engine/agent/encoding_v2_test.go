package agent

import (
	"math"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// TestCategorizeActionBuckets checks the mapping from engine action indices
// onto the 3 v2 semantic categories.
func TestCategorizeActionBuckets(t *testing.T) {
	cases := []struct {
		name     string
		idx      uint16
		wantCat  uint8
		wantSlot uint8
		wantHas  bool
	}{
		{"draw_stockpile", engine.ActionDrawStockpile, ActionCategoryDraw, 0, false},
		{"draw_discard", engine.ActionDrawDiscard, ActionCategoryDraw, 0, false},
		{"call_cambia", engine.ActionCallCambia, ActionCategoryDraw, 0, false},
		{"discard_no_ability", engine.ActionDiscardNoAbility, ActionCategoryDiscard, 0, false},
		{"discard_with_ability", engine.ActionDiscardWithAbility, ActionCategoryDiscard, 0, false},
		{"pass_snap", engine.ActionPassSnap, ActionCategoryAbilityOrSnap, 0, false},
		{"king_swap_yes", engine.ActionKingSwapYes, ActionCategoryAbilityOrSnap, 0, false},
		{"king_swap_no", engine.ActionKingSwapNo, ActionCategoryAbilityOrSnap, 0, false},
		{"replace_3", engine.EncodeReplace(3), ActionCategoryDiscard, 3, true},
		{"peek_own_2", engine.EncodePeekOwn(2), ActionCategoryAbilityOrSnap, 2, true},
		{"peek_other_5", engine.EncodePeekOther(5), ActionCategoryAbilityOrSnap, 5, true},
		{"blind_swap_2_4", engine.EncodeBlindSwap(2, 4), ActionCategoryAbilityOrSnap, 2, true},
		{"king_look_1_3", engine.EncodeKingLook(1, 3), ActionCategoryAbilityOrSnap, 1, true},
		{"snap_own_4", engine.EncodeSnapOwn(4), ActionCategoryAbilityOrSnap, 4, true},
		{"snap_opp_0", engine.EncodeSnapOpponent(0), ActionCategoryAbilityOrSnap, 0, true},
		{"snap_opp_move_5_2", engine.EncodeSnapOpponentMove(5, 2), ActionCategoryAbilityOrSnap, 5, true},
	}
	for _, c := range cases {
		cat, slot, has := CategorizeAction(c.idx)
		if cat != c.wantCat || slot != c.wantSlot || has != c.wantHas {
			t.Errorf("%s: got (cat=%d slot=%d has=%t), want (cat=%d slot=%d has=%t)",
				c.name, cat, slot, has, c.wantCat, c.wantSlot, c.wantHas)
		}
	}
}

// TestActionHistoryRingBuffer drives an AgentState through four applied actions
// for the acting player and checks that the ring buffer keeps only the newest
// three entries, oldest-first.
func TestActionHistoryRingBuffer(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.MaxGameTurns = 50
	g := engine.NewGame(42, rules)
	g.Deal()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	// Simulate 4 self-actions by calling recordActionHistory directly.
	a.recordActionHistory(engine.ActionDrawStockpile, 0)   // cat=draw
	a.recordActionHistory(engine.EncodePeekOwn(1), 0)      // cat=ability_or_snap, slot=1
	a.recordActionHistory(engine.EncodeBlindSwap(3, 2), 0) // cat=ability_or_snap, slot=3
	a.recordActionHistory(engine.EncodeReplace(4), 0)      // cat=discard, slot=4

	if a.OwnActionHistoryLen != ActionHistorySize {
		t.Fatalf("OwnActionHistoryLen=%d, want %d", a.OwnActionHistoryLen, ActionHistorySize)
	}
	// Expected order (oldest first): peek_own(1), blind_swap(3,2), replace(4).
	want := []ActionHistoryEntry{
		{Category: ActionCategoryAbilityOrSnap, TargetSlot: 1, HasSlot: true, Valid: true},
		{Category: ActionCategoryAbilityOrSnap, TargetSlot: 3, HasSlot: true, Valid: true},
		{Category: ActionCategoryDiscard, TargetSlot: 4, HasSlot: true, Valid: true},
	}
	for i, w := range want {
		got := a.OwnActionHistory[i]
		if got != w {
			t.Errorf("entry %d: got %+v, want %+v", i, got, w)
		}
	}

	// Opp buffer must be untouched.
	if a.OppActionHistoryLen != 0 {
		t.Errorf("OppActionHistoryLen=%d, want 0", a.OppActionHistoryLen)
	}
}

// TestEncodeEPPBSInterleavedV2Shape checks the v2 encoding prefix matches the
// v1 interleaved encoding exactly, and that the extra dims live at the correct
// offsets.
func TestEncodeEPPBSInterleavedV2Shape(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.MaxGameTurns = 50
	g := engine.NewGame(7, rules)
	g.Deal()
	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	var v1 [EPPBSInputDim]float32
	var v2 [EPPBSV2InputDim]float32
	a.EncodeEPPBSInterleaved(engine.CtxStartTurn, -1, &v1)
	a.EncodeEPPBSInterleavedV2(engine.CtxStartTurn, -1, &v2)

	for i := 0; i < EPPBSInputDim; i++ {
		if v1[i] != v2[i] {
			t.Fatalf("v2[%d]=%f diverges from v1[%d]=%f", i, v2[i], i, v1[i])
		}
	}

	// Posterior must sum to ~1.0 (no discards yet, so remaining > 0).
	var sum float32
	for i := 0; i < V2CardCountDim; i++ {
		sum += v2[V2CardCountOffset+i]
	}
	if math.Abs(float64(sum)-1.0) > 1e-5 {
		t.Errorf("posterior sum=%f, want ~1.0", sum)
	}

	// Action history window starts all zeros on an agent that has not observed any actions.
	for i := 0; i < V2ActionHistoryDim; i++ {
		if v2[V2ActionHistoryOffset+i] != 0 {
			t.Errorf("action history dim %d = %f, want 0 pre-observation", i, v2[V2ActionHistoryOffset+i])
		}
	}
}

// TestPosteriorReflectsDiscards checks that pushing a discard into the agent's
// histogram shrinks the corresponding bucket in the posterior.
func TestPosteriorReflectsDiscards(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 0
	a.MaxGameTurns = 50

	// No discards yet: baseline posterior.
	var base [EPPBSV2InputDim]float32
	a.EncodeEPPBSInterleavedV2(engine.CtxStartTurn, -1, &base)

	// Register 2 discards of BucketLowNum (bucket 3, deck count 12 -> 10 remaining).
	a.DiscardBucketCounts[BucketLowNum] = 2
	a.TotalDiscardsSeen = 2

	var after [EPPBSV2InputDim]float32
	a.EncodeEPPBSInterleavedV2(engine.CtxStartTurn, -1, &after)

	if after[V2CardCountOffset+int(BucketLowNum)] >= base[V2CardCountOffset+int(BucketLowNum)] {
		t.Errorf("expected LowNum posterior to decrease after discards, got base=%f after=%f",
			base[V2CardCountOffset+int(BucketLowNum)], after[V2CardCountOffset+int(BucketLowNum)])
	}

	// Sum should still be ~1.
	var sum float32
	for i := 0; i < V2CardCountDim; i++ {
		sum += after[V2CardCountOffset+i]
	}
	if math.Abs(float64(sum)-1.0) > 1e-5 {
		t.Errorf("posterior sum after discards = %f, want ~1.0", sum)
	}
}

// TestActionHistoryEncoding writes three own actions and verifies the 24-dim
// window lays them out in the expected order and one-hot positions.
func TestActionHistoryEncoding(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 4
	a.MaxGameTurns = 50

	a.recordActionHistory(engine.EncodePeekOwn(0), 0)    // ability_or_snap(2), slot=0 -> 0/5
	a.recordActionHistory(engine.EncodeReplace(2), 0)    // discard(1), slot=2 -> 2/5
	a.recordActionHistory(engine.ActionDrawStockpile, 0) // draw(0), no slot
	a.recordActionHistory(engine.EncodePeekOther(5), 1)  // opponent action, ability_or_snap(2), slot=5

	var v2 [EPPBSV2InputDim]float32
	a.EncodeEPPBSInterleavedV2(engine.CtxStartTurn, -1, &v2)

	entryDim := V2ActionHistoryEntryDim
	ownBase := V2ActionHistoryOffset

	// Entry 0: ability_or_snap (cat 2), slot 0, slot_norm=0.
	if v2[ownBase+0] != 0 || v2[ownBase+1] != 0 || v2[ownBase+2] != 1.0 {
		t.Errorf("own[0] category mismatch: cats=%v", v2[ownBase:ownBase+3])
	}
	if v2[ownBase+3] != 0 {
		t.Errorf("own[0] slot_norm=%f, want 0", v2[ownBase+3])
	}

	// Entry 1: discard (cat 1), slot 2, slot_norm=2/5.
	if v2[ownBase+entryDim+1] != 1.0 {
		t.Errorf("own[1] category want discard=1.0, got %f", v2[ownBase+entryDim+1])
	}
	want1 := float32(2) / 5.0
	if math.Abs(float64(v2[ownBase+entryDim+3]-want1)) > 1e-6 {
		t.Errorf("own[1] slot_norm=%f, want %f", v2[ownBase+entryDim+3], want1)
	}

	// Entry 2: draw (cat 0), no slot.
	if v2[ownBase+2*entryDim+0] != 1.0 {
		t.Errorf("own[2] category want draw=1.0, got %f", v2[ownBase+2*entryDim+0])
	}
	if v2[ownBase+2*entryDim+3] != 0 {
		t.Errorf("own[2] slot_norm=%f, want 0", v2[ownBase+2*entryDim+3])
	}

	// Opp entry 0: ability_or_snap (cat 2), slot 5, slot_norm=1.0.
	oppBase := V2ActionHistoryOffset + ActionHistorySize*V2ActionHistoryEntryDim
	if v2[oppBase+2] != 1.0 {
		t.Errorf("opp[0] category want ability_or_snap=1.0, got %f", v2[oppBase+2])
	}
	if v2[oppBase+3] != 1.0 {
		t.Errorf("opp[0] slot_norm=%f, want 1.0", v2[oppBase+3])
	}

	// Opp entries 1 and 2 must remain all zeros.
	for i := entryDim; i < 3*entryDim; i++ {
		if v2[oppBase+i] != 0 {
			t.Errorf("opp[%d] dim=%f, want 0", i, v2[oppBase+i])
		}
	}
}

// TestUpdateAppendsToHistory verifies a full engine Update() path appends an
// entry, not just the direct helper.
func TestUpdateAppendsToHistory(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.MaxGameTurns = 50
	g := engine.NewGame(11, rules)
	g.Deal()

	// Deal picks the starting player at random; align the agent's side with it
	// so the test exercises the own-history path regardless of seed.
	startingPlayer := g.CurrentPlayer
	opp := uint8(0)
	if startingPlayer == 0 {
		opp = 1
	}
	a := NewAgentState(startingPlayer, opp, 0, 0)
	a.Initialize(&g)

	if err := g.ApplyAction(engine.ActionDrawStockpile); err != nil {
		t.Fatalf("ApplyAction(DrawStockpile) failed: %v", err)
	}
	a.Update(&g)

	if a.OwnActionHistoryLen != 1 {
		t.Fatalf("OwnActionHistoryLen=%d, want 1 (starting player=%d)", a.OwnActionHistoryLen, startingPlayer)
	}
	got := a.OwnActionHistory[0]
	if got.Category != ActionCategoryDraw || got.HasSlot {
		t.Errorf("Own history entry = %+v, want category=draw, no slot", got)
	}
}
