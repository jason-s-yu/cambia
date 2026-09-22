package nodeagent

import (
	"context"
	"database/sql"
	"os"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
)

// reattach reloads the lease records this node persisted and rebuilds a jobRun
// for each lease it may still be supervising (D37). It starts no goroutine:
// the coordinator decides at register which leases re-bind, so nothing posts
// progress or uploads a byte until that answer is in.
//
// A record whose job never launched is dropped rather than reattached. The
// coordinator's own rule for a lease the node does not name is a requeue at
// the original submit_seq with attempt++ (D32, pre-launch row), which is the
// correct outcome for staging this node lost; naming it would claim a
// supervision that does not exist.
func (a *Agent) reattach() (live []nashnet.LiveLease, pending []*jobRun) {
	for _, rec := range readLeaseRecords(a.cfg.BaseDir) {
		if !rec.Launched {
			removeLeaseRecord(a.cfg.BaseDir, rec.JobID)
			a.log.Printf("dropping pre-launch lease %s for %s across restart", rec.LeaseID, rec.JobID)
			continue
		}
		spec, err := decodeSpec(rec.Spec)
		if err != nil {
			removeLeaseRecord(a.cfg.BaseDir, rec.JobID)
			a.log.Printf("dropping lease %s: %v", rec.LeaseID, err)
			continue
		}
		job := &jobRun{agent: a, rec: rec, spec: spec}
		pending = append(pending, job)
		live = append(live, rec.liveLease())
		a.noteCommit(rec.Commit)
	}
	return live, pending
}

// resumePending starts the reattached jobs the coordinator re-bound and
// orphans the ones it did not. A re-bound job re-diffs its run dir against the
// coordinator's head before resuming uploads, which the uploader does on its
// first Sync, and resumes an interrupted blob at the HEAD offset, which
// uploadOne does per blob (D37).
func (a *Agent) resumePending(ctx context.Context, pending []*jobRun, rebound map[string]bool) {
	for _, job := range pending {
		if !rebound[job.rec.LeaseID] {
			a.log.Printf("lease %s was not rebound; orphaning %s", job.rec.LeaseID, job.rec.JobID)
			_ = a.launcher.Stop(job.spec.Name, false)
			removeLeaseRecord(a.cfg.BaseDir, job.rec.JobID)
			continue
		}
		a.mu.Lock()
		a.active[job.rec.JobID] = job
		a.mu.Unlock()
		// A reattached job occupies the node exactly like a launched one, so
		// admission must see it before the next claim goes out (cambia-723).
		a.slots.Claim(job.spec.Exclusive)

		j := job
		a.wg.Add(1)
		go func() {
			defer a.wg.Done()
			defer a.finishJob(j)
			j.resume(ctx)
		}()
	}
}

// AwaitReattachedExit blocks until a reattached job's process is gone: its row
// shows the exit (adoptedExited), or its run dir is purged out from under it.
// A reattached process was forked by a prior daemon incarnation, so this one
// cannot waitpid it and procmgr's own wait goroutine never runs for it;
// liveness is read off the starttime-validated pid probe behind
// Launcher.Status instead. It is the watch half of the reattach machinery,
// moved here with the launch path (D1), and the coordinator's reattach watcher
// blocks on it; a node's supervise loop applies the same predicate on its poll
// tick, because it must keep renewing the lease while it waits. What to write
// for the job that exited stays with the caller, because the two-witness
// finalizer is coordinator state (D7, D35).
func AwaitReattachedExit(l Launcher, name string, poll time.Duration, stop <-chan struct{}) {
	if poll <= 0 {
		poll = time.Second
	}
	for {
		st := l.Status(name)
		if !st.Found {
			return // run dir gone (purged): nothing left to hold or finalize
		}
		if adoptedExited(st) {
			return
		}
		select {
		case <-stop:
			return
		case <-time.After(poll):
		}
	}
}

// adoptedExited reports whether a row shows the exit of a process this daemon
// did not fork. Nothing here waits on such a process, so its raw status stays
// in the live set after it is gone and the pid-validated effective status is
// what turns terminal. A raw terminal counts too: the incarnation that forked
// the process may have recorded its exit before it went down.
func adoptedExited(st ProcessStatus) bool {
	return st.Found && (isTerminalStatus(st.Status) || isTerminalStatus(st.Effective))
}

// runDBStatusCompleted is the runs.status a trainer writes on a clean end
// (update_run_status(..., "completed") in cfr/src/cfr/prtcfr_trainer.py,
// deep_trainer.py and the others), the only value that certifies a clean exit.
const runDBStatusCompleted = "completed"

// runDBQueryTimeout bounds the journal read so a locked or pathological
// run_db.sqlite cannot wedge a job's finish.
const runDBQueryTimeout = 5 * time.Second

// runDBRunStatus returns the runs.status a job's journal records for name, or
// "" when the journal is absent, unreadable, locked, or holds no row. It reads
// the way the coordinator's finalizer does (harness runDBRunStatus): by name
// first, then the newest row, because an evaluate job writes its single row
// under the evaluated target's name. It opens read-only and never creates the
// file.
func runDBRunStatus(dbPath, name string) string {
	if _, err := os.Stat(dbPath); err != nil {
		return ""
	}
	db, err := sql.Open("sqlite", "file:"+dbPath+"?mode=ro&_pragma=busy_timeout(2000)")
	if err != nil {
		return ""
	}
	defer db.Close()
	ctx, cancel := context.WithTimeout(context.Background(), runDBQueryTimeout)
	defer cancel()

	var status string
	if err := db.QueryRowContext(ctx, "SELECT status FROM runs WHERE name = ?", name).Scan(&status); err == nil {
		return status
	}
	if err := db.QueryRowContext(ctx,
		"SELECT status FROM runs ORDER BY updated_at DESC LIMIT 1").Scan(&status); err != nil {
		return ""
	}
	return status
}
