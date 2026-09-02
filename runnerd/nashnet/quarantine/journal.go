package quarantine

// JournalValidator content-validates a verified run_db.sqlite blob before the
// commit transaction folds it into the manifest (D55). The production
// implementation is this package's own Validate (rundbcheck.go, cambia-1717),
// wired as the store's default; the interface is the seam a commit test uses to
// force a verdict or to observe which path the transaction handed over.
//
// path is the verified blob in quarantine, never a promoted file, and
// expectedName is the runs.name the journal must carry: the job id, or the
// resolved spec.target for an evaluate job (D64). A non-nil error means the
// validator could not evaluate the path at all, which is a coordinator-side
// fault rather than a node rejection; every content-level problem comes back as
// a rejected Verdict with a nil error, and the commit collapses all of them to
// the single per-entry reason rundb_invalid, carrying Verdict.Reason and
// Verdict.Detail as the rejection's detail.
type JournalValidator interface {
	Validate(path, expectedName string) (Verdict, error)
}

// ValidatorFunc adapts a plain function to JournalValidator.
type ValidatorFunc func(path, expectedName string) (Verdict, error)

// Validate implements JournalValidator.
func (f ValidatorFunc) Validate(path, expectedName string) (Verdict, error) {
	return f(path, expectedName)
}

// runDBValidator is the store's default: the D55 validator under the caps the
// daemon resolved from RUNNERD_NASHNET_MAX_RUNDB_BYTES.
type runDBValidator struct {
	cfg RunDBConfig
}

// Validate implements JournalValidator.
func (v runDBValidator) Validate(path, expectedName string) (Verdict, error) {
	return ValidateWithConfig(path, expectedName, v.cfg)
}
