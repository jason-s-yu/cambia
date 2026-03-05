package main

import (
	"math"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// newTestGame creates a standard game and deals cards.
func newTestGame() engine.GameState {
	g := engine.NewGame(42, engine.DefaultHouseRules())
	g.Deal()
	return g
}

// TestBuildSubgameTree verifies that the tree builds without panicking and
// that leaf indices are assigned starting from 0.
func TestBuildSubgameTree(t *testing.T) {
	g := newTestGame()
	root, numLeaves := BuildSubgameTree(g, 2)

	if root == nil {
		t.Fatal("BuildSubgameTree returned nil root")
	}
	if numLeaves < 0 {
		t.Fatalf("BuildSubgameTree returned negative leaf count: %d", numLeaves)
	}

	// Root should not be terminal or leaf at depth 0 with a fresh game.
	if root.IsTerminal {
		t.Fatal("fresh game root should not be terminal")
	}
	if root.IsLeaf {
		t.Fatal("root should not be a leaf at depth 2 with a fresh game")
	}
	if len(root.Children) == 0 {
		t.Fatal("root should have children")
	}

	// Children should exist and have valid indices.
	for i, child := range root.Children {
		if child.Node == nil {
			t.Fatalf("child %d has nil node", i)
		}
	}

	t.Logf("Tree built: root.Children=%d, numLeaves=%d", len(root.Children), numLeaves)
}

// TestLeafCounting verifies that the leaf count from BuildSubgameTree matches
// the number of states returned by CollectLeafStates.
func TestLeafCounting(t *testing.T) {
	g := newTestGame()
	root, numLeaves := BuildSubgameTree(g, 2)

	leafStates := CollectLeafStates(root)
	if len(leafStates) != numLeaves {
		t.Errorf("BuildSubgameTree reported %d leaves, CollectLeafStates returned %d",
			numLeaves, len(leafStates))
	}
	t.Logf("Leaf count consistent: %d", numLeaves)
}

// TestRegretMatch verifies that regretMatch normalizes positive regrets correctly.
func TestRegretMatch(t *testing.T) {
	regrets := []float32{1, 2, 1}
	strategy := regretMatch(regrets)

	if len(strategy) != 3 {
		t.Fatalf("expected strategy length 3, got %d", len(strategy))
	}

	expected := []float32{0.25, 0.5, 0.25}
	for i, s := range strategy {
		if math.Abs(float64(s-expected[i])) > 1e-6 {
			t.Errorf("strategy[%d] = %f, expected %f", i, s, expected[i])
		}
	}

	// Sum should be 1.
	var total float32
	for _, s := range strategy {
		total += s
	}
	if math.Abs(float64(total-1.0)) > 1e-6 {
		t.Errorf("strategy sum = %f, expected 1.0", total)
	}
}

// TestRegretMatchAllNegative verifies that all-negative regrets yield uniform strategy.
func TestRegretMatchAllNegative(t *testing.T) {
	regrets := []float32{-1.0, -2.0, -0.5}
	strategy := regretMatch(regrets)

	if len(strategy) != 3 {
		t.Fatalf("expected strategy length 3, got %d", len(strategy))
	}

	uniform := float32(1.0) / 3.0
	for i, s := range strategy {
		if math.Abs(float64(s-uniform)) > 1e-6 {
			t.Errorf("strategy[%d] = %f, expected %f (uniform)", i, s, uniform)
		}
	}
}

// TestCFRIterationTerminal verifies that a terminal node returns its utility.
func TestCFRIterationTerminal(t *testing.T) {
	node := &SubgameNode{
		IsTerminal: true,
		Utility:    [2]float32{1.0, -1.0},
		LeafIndex:  -1,
	}

	values := node.CFRIteration([2]float32{1.0, 1.0}, nil, 1)
	if values[0] != 1.0 || values[1] != -1.0 {
		t.Errorf("expected [1.0, -1.0], got [%f, %f]", values[0], values[1])
	}
}

// TestCFRIterationLeaf verifies that a leaf node returns the provided leaf values.
func TestCFRIterationLeaf(t *testing.T) {
	node := &SubgameNode{
		IsLeaf:    true,
		LeafIndex: 1,
	}

	leafValues := []float32{0.5, -0.5, 0.8, -0.8, 0.2, -0.2}
	values := node.CFRIteration([2]float32{1.0, 1.0}, leafValues, 1)
	// LeafIndex=1 → leafValues[2], leafValues[3]
	if math.Abs(float64(values[0]-0.8)) > 1e-6 {
		t.Errorf("expected values[0]=0.8, got %f", values[0])
	}
	if math.Abs(float64(values[1]+0.8)) > 1e-6 {
		t.Errorf("expected values[1]=-0.8, got %f", values[1])
	}
}

// TestCFRConvergence runs 200 CFR iterations on a D=2 tree and verifies
// that the root's AverageStrategy sums to 1.
func TestCFRConvergence(t *testing.T) {
	g := newTestGame()
	root, numLeaves := BuildSubgameTree(g, 2)

	if numLeaves == 0 {
		t.Skip("no leaf nodes — game terminates before depth 2")
	}

	// Provide neutral leaf values (zero-sum, slight bias).
	leafValues := make([]float32, numLeaves*2)
	for i := 0; i < numLeaves; i++ {
		leafValues[i*2] = 0.1   // slight bias toward player 0
		leafValues[i*2+1] = -0.1
	}

	for iter := 1; iter <= 200; iter++ {
		root.CFRIteration([2]float32{1.0, 1.0}, leafValues, iter)
	}

	strategy := root.AverageStrategy()
	if len(strategy) == 0 {
		t.Fatal("AverageStrategy returned empty slice")
	}

	var total float32
	for _, s := range strategy {
		total += s
	}
	if math.Abs(float64(total-1.0)) > 1e-5 {
		t.Errorf("AverageStrategy sum = %f, expected 1.0", total)
	}

	// All probabilities should be non-negative.
	for i, s := range strategy {
		if s < 0 {
			t.Errorf("strategy[%d] = %f, expected >= 0", i, s)
		}
	}

	t.Logf("Converged: strategy len=%d, sum=%f, numLeaves=%d", len(strategy), total, numLeaves)
}

// TestSolveSubgame runs the full solve and verifies the strategy is a valid
// probability distribution.
func TestSolveSubgame(t *testing.T) {
	g := newTestGame()

	// First determine how many leaves there are.
	root, numLeaves := BuildSubgameTree(g, 2)
	_ = root

	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	leafValues := make([]float32, numLeaves*2)
	for i := 0; i < numLeaves; i++ {
		leafValues[i*2] = 0.0
		leafValues[i*2+1] = 0.0
	}

	strategy, rootValues := SolveSubgame(g, 2, 100, leafValues)

	if len(strategy) == 0 {
		t.Fatal("SolveSubgame returned empty strategy")
	}

	var total float32
	for _, s := range strategy {
		total += s
	}
	if math.Abs(float64(total-1.0)) > 1e-5 {
		t.Errorf("strategy sum = %f, expected 1.0", total)
	}

	for i, s := range strategy {
		if s < 0 {
			t.Errorf("strategy[%d] = %f, expected >= 0", i, s)
		}
	}

	t.Logf("SolveSubgame: strategy len=%d, sum=%f, rootValues=%v", len(strategy), total, rootValues)
}

// TestRootValuesConvergence verifies that averaged rootValues converge:
// running 200 iterations vs 100 should produce close values.
func TestRootValuesConvergence(t *testing.T) {
	g := newTestGame()
	_, numLeaves := BuildSubgameTree(g, 2)
	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	leafValues := make([]float32, numLeaves*2)
	for i := 0; i < numLeaves; i++ {
		leafValues[i*2] = 0.3
		leafValues[i*2+1] = -0.3
	}

	_, v100 := SolveSubgame(g, 2, 100, leafValues)
	_, v200 := SolveSubgame(g, 2, 200, leafValues)

	diff0 := math.Abs(float64(v200[0] - v100[0]))
	diff1 := math.Abs(float64(v200[1] - v100[1]))

	// Averaged values should converge; allow tolerance of 0.05.
	if diff0 > 0.05 {
		t.Errorf("rootValues[0] changed too much between 100 and 200 iters: %f vs %f (diff=%f)", v100[0], v200[0], diff0)
	}
	if diff1 > 0.05 {
		t.Errorf("rootValues[1] changed too much between 100 and 200 iters: %f vs %f (diff=%f)", v100[1], v200[1], diff1)
	}

	t.Logf("v100=%v v200=%v diff=[%f,%f]", v100, v200, diff0, diff1)
}

// TestRootValuesAveragedVsSingleIter verifies that averaged rootValues differ
// from the raw first-iteration values in a non-trivial game.
func TestRootValuesAveragedVsSingleIter(t *testing.T) {
	g := newTestGame()
	_, numLeaves := BuildSubgameTree(g, 2)
	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	leafValues := make([]float32, numLeaves*2)
	for i := 0; i < numLeaves; i++ {
		leafValues[i*2] = float32(i%5) * 0.1
		leafValues[i*2+1] = -float32(i%5) * 0.1
	}

	// Single iteration.
	rootSingle, _ := BuildSubgameTree(g, 2)
	singleIter := rootSingle.CFRIteration([2]float32{1.0, 1.0}, leafValues, 1)

	// 100-iteration averaged.
	_, avgValues := SolveSubgame(g, 2, 100, leafValues)

	// They should differ (CFR regret-matching changes strategy over iterations).
	diff := math.Abs(float64(avgValues[0]-singleIter[0])) + math.Abs(float64(avgValues[1]-singleIter[1]))
	t.Logf("singleIter=%v avgValues=%v totalDiff=%f", singleIter, avgValues, diff)

	// If numLeaves > 1 with varied values, averaged should differ from single.
	// We only assert they are not identical when there is meaningful structure.
	if numLeaves > 1 && diff < 1e-9 {
		t.Errorf("averaged rootValues identical to single iteration — averaging may not be working")
	}
}

// ---------------------------------------------------------------------------
// Ranged CFR tests
// ---------------------------------------------------------------------------

// TestCFRIterationRangedTerminal verifies that a terminal node returns utility tiled across hand types.
func TestCFRIterationRangedTerminal(t *testing.T) {
	const numHandTypes = 4
	node := &SubgameNode{
		IsTerminal: true,
		Utility:    [2]float32{1.5, -1.5},
		LeafIndex:  -1,
	}
	ranges := [2][]float32{
		{0.25, 0.25, 0.25, 0.25},
		{0.25, 0.25, 0.25, 0.25},
	}
	result := node.CFRIterationRanged([2]float32{1.0, 1.0}, nil, ranges, numHandTypes, 1)
	for h := 0; h < numHandTypes; h++ {
		if math.Abs(float64(result[0][h]-1.5)) > 1e-6 {
			t.Errorf("result[0][%d] = %f, expected 1.5", h, result[0][h])
		}
		if math.Abs(float64(result[1][h]+1.5)) > 1e-6 {
			t.Errorf("result[1][%d] = %f, expected -1.5", h, result[1][h])
		}
	}
}

// TestCFRIterationRangedLeaf verifies that a leaf node returns correct per-hand-type values.
func TestCFRIterationRangedLeaf(t *testing.T) {
	const numHandTypes = 3
	node := &SubgameNode{
		IsLeaf:    true,
		LeafIndex: 1, // second leaf
	}
	// Layout: [leaf0_p0_h0, leaf0_p0_h1, leaf0_p0_h2, leaf0_p1_h0, leaf0_p1_h1, leaf0_p1_h2,
	//          leaf1_p0_h0, leaf1_p0_h1, leaf1_p0_h2, leaf1_p1_h0, leaf1_p1_h1, leaf1_p1_h2]
	leafValues := []float32{
		0.1, 0.2, 0.3, -0.1, -0.2, -0.3, // leaf 0
		0.4, 0.5, 0.6, -0.4, -0.5, -0.6, // leaf 1
	}
	ranges := [2][]float32{
		{0.33, 0.33, 0.34},
		{0.33, 0.33, 0.34},
	}
	result := node.CFRIterationRanged([2]float32{1.0, 1.0}, leafValues, ranges, numHandTypes, 1)
	expected0 := []float32{0.4, 0.5, 0.6}
	expected1 := []float32{-0.4, -0.5, -0.6}
	for h := 0; h < numHandTypes; h++ {
		if math.Abs(float64(result[0][h]-expected0[h])) > 1e-6 {
			t.Errorf("result[0][%d] = %f, expected %f", h, result[0][h], expected0[h])
		}
		if math.Abs(float64(result[1][h]-expected1[h])) > 1e-6 {
			t.Errorf("result[1][%d] = %f, expected %f", h, result[1][h], expected1[h])
		}
	}
}

// TestSolveSubgameRangedUniform verifies that uniform ranges + uniform leaf values → strategy sums to 1.
func TestSolveSubgameRangedUniform(t *testing.T) {
	const numHandTypes = 3
	g := newTestGame()
	root, numLeaves := BuildSubgameTree(g, 2)
	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	leafValues := make([]float32, numLeaves*2*numHandTypes)
	for i := 0; i < numLeaves*2*numHandTypes; i++ {
		leafValues[i] = 0.1
	}
	uniform := float32(1.0) / float32(numHandTypes)
	ranges := [2][]float32{
		make([]float32, numHandTypes),
		make([]float32, numHandTypes),
	}
	for h := 0; h < numHandTypes; h++ {
		ranges[0][h] = uniform
		ranges[1][h] = uniform
	}

	strategy, _ := SolveSubgameRanged(root, 50, leafValues, ranges, numHandTypes)
	if len(strategy) == 0 {
		t.Fatal("SolveSubgameRanged returned empty strategy")
	}
	var total float32
	for _, s := range strategy {
		total += s
	}
	if math.Abs(float64(total-1.0)) > 1e-5 {
		t.Errorf("strategy sum = %f, expected 1.0", total)
	}
}

// TestSolveSubgameRangedNonUniform verifies that non-uniform leaf values produce non-uniform root CFVs.
func TestSolveSubgameRangedNonUniform(t *testing.T) {
	const numHandTypes = 4
	g := newTestGame()
	root, numLeaves := BuildSubgameTree(g, 2)
	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	leafValues := make([]float32, numLeaves*2*numHandTypes)
	for i := 0; i < numLeaves; i++ {
		for h := 0; h < numHandTypes; h++ {
			leafValues[i*2*numHandTypes+0*numHandTypes+h] = float32(h) * 0.1
			leafValues[i*2*numHandTypes+1*numHandTypes+h] = -float32(h) * 0.1
		}
	}
	uniform := float32(1.0) / float32(numHandTypes)
	ranges := [2][]float32{
		{uniform, uniform, uniform, uniform},
		{uniform, uniform, uniform, uniform},
	}

	_, cfvs := SolveSubgameRanged(root, 50, leafValues, ranges, numHandTypes)
	// Compute variance across hand types for player 0.
	var mean, variance float32
	for h := 0; h < numHandTypes; h++ {
		mean += cfvs[0][h]
	}
	mean /= float32(numHandTypes)
	for h := 0; h < numHandTypes; h++ {
		d := cfvs[0][h] - mean
		variance += d * d
	}
	if variance < 1e-10 {
		t.Errorf("expected non-zero variance in CFVs for non-uniform leaf values, got %f", variance)
	}
}

// TestSolveSubgameRangedDeltaRange verifies that a delta range (one hand type = 1.0)
// matches the scalar SolveSubgame for that hand type.
func TestSolveSubgameRangedDeltaRange(t *testing.T) {
	const numHandTypes = 3
	const deltaHand = 1
	g := newTestGame()

	root, numLeaves := BuildSubgameTree(g, 2)
	if numLeaves == 0 {
		t.Skip("no leaf nodes")
	}

	// Build scalar leaf values using deltaHand slice.
	scalarLeafValues := make([]float32, numLeaves*2)
	rangedLeafValues := make([]float32, numLeaves*2*numHandTypes)
	for i := 0; i < numLeaves; i++ {
		v0 := float32(i%7) * 0.05
		v1 := -v0
		scalarLeafValues[i*2+0] = v0
		scalarLeafValues[i*2+1] = v1
		for h := 0; h < numHandTypes; h++ {
			if h == deltaHand {
				rangedLeafValues[i*2*numHandTypes+0*numHandTypes+h] = v0
				rangedLeafValues[i*2*numHandTypes+1*numHandTypes+h] = v1
			}
		}
	}

	// Delta range: only deltaHand has weight.
	ranges := [2][]float32{
		make([]float32, numHandTypes),
		make([]float32, numHandTypes),
	}
	ranges[0][deltaHand] = 1.0
	ranges[1][deltaHand] = 1.0

	// Solve with ranged version (resets the tree internally).
	_, rangedCFVs := SolveSubgameRanged(root, 100, rangedLeafValues, ranges, numHandTypes)

	// Solve with scalar version (builds fresh tree).
	_, scalarCFVs := SolveSubgame(g, 2, 100, scalarLeafValues)

	// The ranged CFV at deltaHand should match the scalar CFV closely.
	diff0 := math.Abs(float64(rangedCFVs[0][deltaHand] - scalarCFVs[0]))
	diff1 := math.Abs(float64(rangedCFVs[1][deltaHand] - scalarCFVs[1]))
	t.Logf("rangedCFVs[0][%d]=%f scalarCFVs[0]=%f diff=%f", deltaHand, rangedCFVs[0][deltaHand], scalarCFVs[0], diff0)
	t.Logf("rangedCFVs[1][%d]=%f scalarCFVs[1]=%f diff=%f", deltaHand, rangedCFVs[1][deltaHand], scalarCFVs[1], diff1)
	// Allow tolerance since averaging is slightly different (ranged uses last iter CFVs, scalar uses averaged).
	if diff0 > 0.2 {
		t.Errorf("ranged[0][%d]=%f vs scalar[0]=%f: diff too large (%f)", deltaHand, rangedCFVs[0][deltaHand], scalarCFVs[0], diff0)
	}
	if diff1 > 0.2 {
		t.Errorf("ranged[1][%d]=%f vs scalar[1]=%f: diff too large (%f)", deltaHand, rangedCFVs[1][deltaHand], scalarCFVs[1], diff1)
	}
}

// TestAverageStrategyUniform verifies that zero strategy sums return a uniform distribution.
func TestAverageStrategyUniform(t *testing.T) {
	node := &SubgameNode{
		StrategySum: []float32{0, 0, 0},
	}

	strategy := node.AverageStrategy()
	if len(strategy) != 3 {
		t.Fatalf("expected length 3, got %d", len(strategy))
	}

	uniform := float32(1.0) / 3.0
	for i, s := range strategy {
		if math.Abs(float64(s-uniform)) > 1e-6 {
			t.Errorf("strategy[%d] = %f, expected %f (uniform)", i, s, uniform)
		}
	}
}
