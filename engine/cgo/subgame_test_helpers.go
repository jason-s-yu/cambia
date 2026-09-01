package main

/*
#include <stdint.h>
*/
import "C"

import (
	agent "github.com/jason-s-yu/cambia/engine/agent"
)

// Test-only plain-Go-typed wrappers around the subgame cgo exports, mirroring
// clone_test_helpers.go's pattern: Go does not allow `import "C"` in
// _test.go files, so subgame_guard_test.go calls through these instead.
// Unexported and carry no //export directive, so they contribute no symbols
// to libcambia.so's C ABI.

func testSubgameBuild(gameH int32, maxDepth int32) int32 {
	return int32(cambia_subgame_build(C.int32_t(gameH), C.int32_t(maxDepth)))
}

// testSubgameSolve calls cambia_subgame_solve with minimal scratch buffers
// sized for the full 146-action space. Callers only interested in the guard
// return code (0 / -1 / -2) can ignore the buffer contents.
func testSubgameSolve(solverH int32, numIterations int32) int32 {
	leafValues := make([]C.float, 2)
	strategyOut := make([]C.float, agent.NumActions)
	rootValuesOut := make([]C.float, 2)
	return int32(cambia_subgame_solve(
		C.int32_t(solverH),
		C.int32_t(numIterations),
		&leafValues[0],
		&strategyOut[0],
		&rootValuesOut[0],
	))
}

// testSubgameSolveRanged calls cambia_subgame_solve_ranged with minimal
// scratch buffers for a small fixed hand-type count. Callers only interested
// in the guard return code (0 / -1 / -2) can ignore the buffer contents.
func testSubgameSolveRanged(solverH int32, numIterations int32) int32 {
	const numHandTypes = 2
	leafValues := make([]C.float, 2*2*numHandTypes)
	rangeP0 := make([]C.float, numHandTypes)
	rangeP1 := make([]C.float, numHandTypes)
	strategyOut := make([]C.float, agent.NumActions)
	rootCFVsOut := make([]C.float, 2*numHandTypes)
	return int32(cambia_subgame_solve_ranged(
		C.int32_t(solverH),
		C.int32_t(numIterations),
		C.int32_t(numHandTypes),
		&leafValues[0],
		&rangeP0[0],
		&rangeP1[0],
		&strategyOut[0],
		&rootCFVsOut[0],
	))
}

func testSubgameFree(solverH int32) { cambia_subgame_free(C.int32_t(solverH)) }
