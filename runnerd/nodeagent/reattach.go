package nodeagent

import (
	"context"

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

		j := job
		a.wg.Add(1)
		go func() {
			defer a.wg.Done()
			defer a.finishJob(j.rec.JobID)
			j.resume(ctx)
		}()
	}
}
