package nodeagent

import (
	"errors"
	"strings"
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

// specFatalMarkers are the substrings that identify a spec-fatal Prepare
// failure inside the ingest pipeline's wrapped errors. The set is closed by
// design: D32 enumerates the spec-fatal causes, so anything else a node's
// Prepare produces is node-attributable and nacks rather than burning the run
// name (the run-name namespace is global). Ordering the classification this
// way means a new ingest failure mode defaults to a nack, which costs a
// cooldown and a breaker tick, instead of defaulting to a job failure, which
// is unrecoverable without an operator purge.
var specFatalMarkers = []string{
	"receipt check",
	"render config",
	"render:",
	"validate config",
	"config validate",
	"owned override",
	"override targets harness-owned key",
	"invalid override",
	"script: not found at pinned commit",
	"signature",
}

// classifyPrepare turns a Prepare error into either a spec-fatal failure or a
// node-attributable one. It is applied to the error text because the ingest
// pipeline reports its stages as wrapped strings rather than typed sentinels;
// the marker list is the contract, and a marker that stops matching shows up
// as a job that nacks forever rather than as a silently wrong verdict.
func classifyPrepare(err error) error {
	if err == nil {
		return nil
	}
	if isSpecFatal(err) {
		return err
	}
	text := strings.ToLower(err.Error())
	for _, m := range specFatalMarkers {
		if strings.Contains(text, m) {
			return &specFatalError{reason: "prepare failed", err: err}
		}
	}
	return err
}
