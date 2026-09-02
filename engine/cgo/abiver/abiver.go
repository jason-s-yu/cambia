// Package abiver holds the libcambia.so cgo export ABI generation: an
// integer bumped every time an //export function's C-visible signature
// changes (an added, removed, or reordered parameter; a changed parameter or
// return type; or a new or removed export).
//
// It is imported by two independent consumers that must agree without
// talking to each other at build time:
//
//   - engine/cgo (package main) exports it through cambia_abi_generation, so
//     cfr/src/ffi/bridge.py can refuse to run against a libcambia.so built
//     from a stale exports.go (cambia-1689). Before this, an appended
//     constructor argument or a widened read-back record was silently
//     ignored by the SysV calling convention: the mismatch was invisible
//     until something downstream broke on the wrong rules.
//   - runnerd/ingest folds it into the harness's libcambia build-cache key,
//     so a cached .so from an older ABI generation is never handed to a
//     newer worktree.
//
// Bump Generation here whenever an exported signature changes, then
// regenerate engine/cgo/testdata/abi_golden.txt (see engine/cgo/abi_test.go)
// so the golden pin catches a signature change that forgot the bump.
package abiver

// Generation is the current libcambia.so C ABI generation.
const Generation = 1
