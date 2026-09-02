package quarantine

// JournalVerdict is the outcome of content-validating a node-uploaded
// run_db.sqlite before promotion (D55).
type JournalVerdict string

// The two verdicts.
const (
	JournalValid   JournalVerdict = "valid"
	JournalInvalid JournalVerdict = "invalid"
)

// JournalValidator content-validates a verified run_db.sqlite blob before the
// commit transaction folds it into the manifest. Validate is handed the path of
// the verified blob in quarantine, never a promoted file, and must not write to
// it: the real implementation opens with mode=ro&immutable=1.
//
// The real implementation is cambia-1717 (W1-T8), landing as
// runnerd/nashnet/quarantine/rundbcheck.go: a size cap before open
// (RUNNERD_NASHNET_MAX_RUNDB_BYTES, default 256 MiB), the modernc driver
// already linked at harness/rundb_status.go:44, PRAGMA integrity_check, a
// schema subset check, exactly one runs row whose name matches the job (or
// spec.target for an evaluate job, D64), the status enum, row-count caps, and
// the 5s query timeout of rundb_status.go:31. Wire it into Config.Validator at
// merge.
//
// The returned string is a short reason code. It is used verbatim as the
// rejection reason when the verdict is JournalInvalid, and the commit path
// substitutes ReasonRunDBInvalid when it is empty, so a validator that returns
// no detail still produces the reason code D55 names.
type JournalValidator interface {
	Validate(path string) (JournalVerdict, string)
}

// ValidatorFunc adapts a plain function to JournalValidator.
type ValidatorFunc func(path string) (JournalVerdict, string)

// Validate implements JournalValidator.
func (f ValidatorFunc) Validate(path string) (JournalVerdict, string) { return f(path) }
