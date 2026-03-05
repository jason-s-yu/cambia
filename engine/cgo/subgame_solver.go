package main

import (
	engine "github.com/jason-s-yu/cambia/engine"
)

// SubgameNode represents a node in the subgame tree.
type SubgameNode struct {
	State       engine.GameState
	Player      uint8         // acting player at this node
	IsTerminal  bool
	IsLeaf      bool          // depth limit reached (not terminal)
	Utility     [2]float32    // filled if terminal
	LeafIndex   int           // index into leaf values array (if leaf), -1 otherwise
	Children    []SubgameChild
	RegretSum   []float32     // [numChildren] cumulative regrets
	StrategySum []float32     // [numChildren] cumulative strategy
	NumVisits   int
}

// SubgameChild links a parent node to a child via an action.
type SubgameChild struct {
	ActionIdx uint16         // raw action index (legal action value)
	Node      *SubgameNode
}

// leafCounter is used during tree building to assign sequential leaf indices.
type leafCounter struct {
	count int
}

// BuildSubgameTree builds a game tree from root state to maxDepth.
// Returns the root node and total number of leaf nodes.
func BuildSubgameTree(root engine.GameState, maxDepth int) (*SubgameNode, int) {
	counter := &leafCounter{}
	node := buildNode(root, maxDepth, 0, counter)
	return node, counter.count
}

func buildNode(state engine.GameState, maxDepth, depth int, counter *leafCounter) *SubgameNode {
	node := &SubgameNode{
		State:     state,
		Player:    state.ActingPlayer(),
		LeafIndex: -1,
	}

	if state.IsTerminal() {
		node.IsTerminal = true
		u := state.GetUtility()
		node.Utility[0] = u[0]
		node.Utility[1] = u[1]
		return node
	}

	if depth >= maxDepth {
		node.IsLeaf = true
		node.LeafIndex = counter.count
		counter.count++
		return node
	}

	// Get legal actions as a list.
	mask := state.LegalActions()
	var legalActions []uint16
	for i := uint16(0); i < engine.NumActions; i++ {
		if mask[i/64]>>(i%64)&1 == 1 {
			legalActions = append(legalActions, i)
		}
	}

	if len(legalActions) == 0 {
		// No legal actions but not terminal — treat as leaf.
		node.IsLeaf = true
		node.LeafIndex = counter.count
		counter.count++
		return node
	}

	node.Children = make([]SubgameChild, len(legalActions))
	node.RegretSum = make([]float32, len(legalActions))
	node.StrategySum = make([]float32, len(legalActions))

	for i, actionIdx := range legalActions {
		childState := state // value copy
		if err := childState.ApplyAction(actionIdx); err != nil {
			// If an action fails (shouldn't happen), treat as leaf.
			childNode := &SubgameNode{
				State:     childState,
				Player:    childState.ActingPlayer(),
				IsLeaf:    true,
				LeafIndex: counter.count,
			}
			counter.count++
			node.Children[i] = SubgameChild{ActionIdx: actionIdx, Node: childNode}
			continue
		}
		childNode := buildNode(childState, maxDepth, depth+1, counter)
		node.Children[i] = SubgameChild{ActionIdx: actionIdx, Node: childNode}
	}

	return node
}

// regretMatch computes the current strategy from cumulative regrets.
// Positive regrets normalized to sum to 1; uniform if all <= 0.
func regretMatch(regretSum []float32) []float32 {
	n := len(regretSum)
	strategy := make([]float32, n)
	if n == 0 {
		return strategy
	}

	var positiveSum float32
	for _, r := range regretSum {
		if r > 0 {
			positiveSum += r
		}
	}

	if positiveSum > 0 {
		for i, r := range regretSum {
			if r > 0 {
				strategy[i] = r / positiveSum
			}
		}
	} else {
		// Uniform strategy.
		uniform := float32(1.0) / float32(n)
		for i := range strategy {
			strategy[i] = uniform
		}
	}

	return strategy
}

// CFRIteration performs one CFR traversal, returning counterfactual values [2]float32.
func (node *SubgameNode) CFRIteration(
	reachProbs [2]float32,
	leafValues []float32,
	iterCount int,
) [2]float32 {
	if node.IsTerminal {
		return node.Utility
	}

	if node.IsLeaf {
		if node.LeafIndex >= 0 && node.LeafIndex*2+1 < len(leafValues) {
			return [2]float32{
				leafValues[node.LeafIndex*2],
				leafValues[node.LeafIndex*2+1],
			}
		}
		return [2]float32{0, 0}
	}

	numChildren := len(node.Children)
	if numChildren == 0 {
		return [2]float32{0, 0}
	}

	player := node.Player
	opp := uint8(1 - player)

	strategy := regretMatch(node.RegretSum)

	// Compute child values.
	childValues := make([][2]float32, numChildren)
	var nodeValue [2]float32

	for i, child := range node.Children {
		// Update reach probability for acting player.
		var childReach [2]float32
		childReach[opp] = reachProbs[opp]
		childReach[player] = reachProbs[player] * strategy[i]

		childValues[i] = child.Node.CFRIteration(childReach, leafValues, iterCount)
		nodeValue[0] += strategy[i] * childValues[i][0]
		nodeValue[1] += strategy[i] * childValues[i][1]
	}

	// Update regret sums (counterfactual regret for acting player).
	for i := range node.Children {
		cfRegret := childValues[i][player] - nodeValue[player]
		node.RegretSum[i] += reachProbs[opp] * cfRegret
	}

	// Update strategy sum (weighted by acting player's reach).
	for i := range node.Children {
		node.StrategySum[i] += reachProbs[player] * strategy[i]
	}

	node.NumVisits++
	return nodeValue
}

// AverageStrategy returns the normalized average strategy at this node.
// If all zero, returns uniform.
func (node *SubgameNode) AverageStrategy() []float32 {
	n := len(node.StrategySum)
	strategy := make([]float32, n)
	if n == 0 {
		return strategy
	}

	var total float32
	for _, s := range node.StrategySum {
		total += s
	}

	if total > 0 {
		for i, s := range node.StrategySum {
			strategy[i] = s / total
		}
	} else {
		uniform := float32(1.0) / float32(n)
		for i := range strategy {
			strategy[i] = uniform
		}
	}

	return strategy
}

// CollectLeafStates returns game states at all leaf nodes, ordered by LeafIndex.
func CollectLeafStates(root *SubgameNode) []engine.GameState {
	// First pass: count leaves.
	numLeaves := countLeaves(root)
	states := make([]engine.GameState, numLeaves)
	collectLeaves(root, states)
	return states
}

func countLeaves(node *SubgameNode) int {
	if node.IsLeaf {
		return 1
	}
	if node.IsTerminal {
		return 0
	}
	total := 0
	for _, child := range node.Children {
		total += countLeaves(child.Node)
	}
	return total
}

func collectLeaves(node *SubgameNode, states []engine.GameState) {
	if node.IsLeaf {
		if node.LeafIndex >= 0 && node.LeafIndex < len(states) {
			states[node.LeafIndex] = node.State
		}
		return
	}
	if node.IsTerminal {
		return
	}
	for _, child := range node.Children {
		collectLeaves(child.Node, states)
	}
}

// CFRIterationRanged performs one CFR traversal with range-weighted regrets.
// leafValues layout: [numLeaves * 2 * numHandTypes], indexed as
//   leafValues[leafIdx*2*numHandTypes + player*numHandTypes + handType]
// ranges[p] is the probability distribution over hand types for player p.
// Returns per-hand-type CFVs for both players: [2][]float32 each of length numHandTypes.
func (node *SubgameNode) CFRIterationRanged(
	reachProbs [2]float32,
	leafValues []float32,
	ranges [2][]float32,
	numHandTypes int,
	iterCount int,
) [2][]float32 {
	result := [2][]float32{
		make([]float32, numHandTypes),
		make([]float32, numHandTypes),
	}

	if node.IsTerminal {
		for h := 0; h < numHandTypes; h++ {
			result[0][h] = node.Utility[0]
			result[1][h] = node.Utility[1]
		}
		return result
	}

	if node.IsLeaf {
		if node.LeafIndex >= 0 {
			base := node.LeafIndex * 2 * numHandTypes
			if base+2*numHandTypes <= len(leafValues) {
				for h := 0; h < numHandTypes; h++ {
					result[0][h] = leafValues[base+0*numHandTypes+h]
					result[1][h] = leafValues[base+1*numHandTypes+h]
				}
			}
		}
		return result
	}

	numChildren := len(node.Children)
	if numChildren == 0 {
		return result
	}

	player := node.Player
	opp := uint8(1 - player)

	strategy := regretMatch(node.RegretSum)

	// Compute child values: [numChildren][2][]float32
	childValues := make([][2][]float32, numChildren)
	nodeValue := [2][]float32{
		make([]float32, numHandTypes),
		make([]float32, numHandTypes),
	}

	for i, child := range node.Children {
		var childReach [2]float32
		childReach[opp] = reachProbs[opp]
		childReach[player] = reachProbs[player] * strategy[i]

		childValues[i] = child.Node.CFRIterationRanged(childReach, leafValues, ranges, numHandTypes, iterCount)
		for h := 0; h < numHandTypes; h++ {
			nodeValue[0][h] += strategy[i] * childValues[i][0][h]
			nodeValue[1][h] += strategy[i] * childValues[i][1][h]
		}
	}

	// Update regret sums: range-weighted scalar regret per action.
	for i := range node.Children {
		var cfRegret float32
		for h := 0; h < numHandTypes; h++ {
			cfRegret += ranges[player][h] * (childValues[i][player][h] - nodeValue[player][h])
		}
		node.RegretSum[i] += reachProbs[opp] * cfRegret
	}

	// Update strategy sum (unchanged from scalar version).
	for i := range node.Children {
		node.StrategySum[i] += reachProbs[player] * strategy[i]
	}

	node.NumVisits++
	return nodeValue
}

// SolveSubgameRanged runs CFRIterationRanged on the pre-built tree (entry.root),
// returning the root AverageStrategy and per-hand-type CFVs from the last iteration.
// Reset regret/strategy sums before running to avoid accumulation from prior calls.
func SolveSubgameRanged(
	root *SubgameNode,
	numIterations int,
	leafValues []float32,
	ranges [2][]float32,
	numHandTypes int,
) (strategy []float32, rootCFVs [2][]float32) {
	resetTree(root)

	var lastCFVs [2][]float32
	for i := 1; i <= numIterations; i++ {
		lastCFVs = root.CFRIterationRanged([2]float32{1.0, 1.0}, leafValues, ranges, numHandTypes, i)
	}

	strategy = root.AverageStrategy()
	rootCFVs = lastCFVs
	return strategy, rootCFVs
}

// resetTree zeroes RegretSum, StrategySum, and NumVisits for all nodes in the tree.
func resetTree(node *SubgameNode) {
	if node == nil || node.IsTerminal || node.IsLeaf {
		return
	}
	for i := range node.RegretSum {
		node.RegretSum[i] = 0
	}
	for i := range node.StrategySum {
		node.StrategySum[i] = 0
	}
	node.NumVisits = 0
	for _, child := range node.Children {
		resetTree(child.Node)
	}
}

// SolveSubgame builds the tree, runs CFR iterations, and returns the root strategy and values.
func SolveSubgame(
	root engine.GameState,
	maxDepth int,
	numIterations int,
	leafValues []float32, // pre-computed leaf evaluations [numLeaves * 2]
) (strategy []float32, rootValues [2]float32) {
	rootNode, _ := BuildSubgameTree(root, maxDepth)

	var sumValues [2]float32
	for i := 1; i <= numIterations; i++ {
		iterValues := rootNode.CFRIteration([2]float32{1.0, 1.0}, leafValues, i)
		sumValues[0] += iterValues[0]
		sumValues[1] += iterValues[1]
	}

	strategy = rootNode.AverageStrategy()
	rootValues = [2]float32{sumValues[0] / float32(numIterations), sumValues[1] / float32(numIterations)}
	return strategy, rootValues
}
