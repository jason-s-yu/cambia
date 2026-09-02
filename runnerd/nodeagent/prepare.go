package nodeagent

import (
	"errors"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
)

// specFatalError marks a staging failure that is a property of the job spec at
// the pinned commit rather than of this node (D63): a config render or
// validate rejection, an owned-override rejection, a measure script absent at
// the commit, or a receipt mismatch after a good fetch. It fails the job with
// no retry, because re-placing it on another node reproduces it exactly.
type specFatalError struct {
	reason string
	err    error
}

func (e *specFatalError) Error() string {
	if e.err == nil {
		return e.reason
	}
	if e.reason == "" {
		return e.err.Error()
	}
	return e.reason + ": " + e.err.Error()
}

func (e *specFatalError) Unwrap() error { return e.err }

// isSpecFatal reports whether a staging failure fails the job outright.
func isSpecFatal(err error) bool {
	var sf *specFatalError
	return errors.As(err, &sf)
}

// specFatalCauses are the ingest sentinels that identify a spec-fatal Prepare
// failure (D63): a config render or validate rejection at the pinned commit, an
// owned-override rejection, a receipt mismatch after a good fetch, a signature
// refusal, a spec path that escapes containment, and a commit that is not a
// commit. Each is a property of the job at its commit, so re-placing it on
// another node reproduces it exactly.
//
// The set is closed by design: D32 enumerates the spec-fatal causes, so
// anything else a node's Prepare produces is node-attributable and nacks rather
// than burning the run name (the run-name namespace is global). Ordering the
// classification this way means a new ingest failure mode defaults to a nack,
// which costs a cooldown and a breaker tick, instead of defaulting to a job
// failure, which is unrecoverable without an operator purge.
var specFatalCauses = []error{
	ingest.ErrConfigRender,
	ingest.ErrConfigValidate,
	ingest.ErrOwnedOverride,
	ingest.ErrReceiptMismatch,
	ingest.ErrSignatureVerification,
	ingest.ErrPathEscape,
	ingest.ErrInvalidCommit,
}

// classifyPrepare turns a Prepare error into either a spec-fatal failure or a
// node-attributable one. It compares sentinels rather than error text, so the
// prepare_node_failed signal the D63 circuit breaker counts stays reliable
// across any rewording of the ingest pipeline's messages: a cause that stops
// matching would otherwise silently reclassify a spec-fatal failure as a nack
// and loop the job across every node in the pool.
func classifyPrepare(err error) error {
	if err == nil {
		return nil
	}
	if isSpecFatal(err) {
		return err
	}
	for _, cause := range specFatalCauses {
		if errors.Is(err, cause) {
			return &specFatalError{reason: "prepare failed", err: err}
		}
	}
	return err
}
