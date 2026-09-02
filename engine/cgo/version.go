package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/jason-s-yu/cambia/engine/cgo/abiver"
)

// abiCommit is stamped at build time by `make libcambia` via
// `-ldflags -X github.com/jason-s-yu/cambia/engine/cgo.abiCommit=<sha>`
// (cambia-1689). A build that skips the ldflags stamp (a plain `go build`
// without -ldflags) leaves it empty; cambia_abi_commit reports that as a
// zero-length write rather than guessing.
var abiCommit string

// cambia_abi_generation reports the integer libcambia.so C ABI generation
// (abiver.Generation). bridge.py calls this once at library load and
// refuses to proceed on a mismatch against its own pinned expectation: a
// stale .so otherwise silently ignores an appended trailing argument
// (harmless on the SysV calling convention) or serves a narrower read-back
// record, and only a caller lucky enough to have its own length check (like
// cambia_game_get_house_rules) ever notices, after the game was already
// built with the wrong rules (cambia-1689).
//
//export cambia_abi_generation
func cambia_abi_generation() C.int32_t {
	return C.int32_t(abiver.Generation)
}

// cambia_abi_commit fills out_buf with the git commit hash this .so was
// built from (abiCommit), as ASCII, unterminated. Returns the number of
// bytes written, 0 if the build did not stamp a commit, or -1 if out_buf is
// too small.
//
//export cambia_abi_commit
func cambia_abi_commit(out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	b := []byte(abiCommit)
	if int(buf_len) < len(b) {
		return -1
	}
	if len(b) == 0 {
		return 0
	}
	outSlice := (*[128]C.uint8_t)(unsafe.Pointer(out_buf))
	for i, c := range b {
		outSlice[i] = C.uint8_t(c)
	}
	return C.int32_t(len(b))
}
