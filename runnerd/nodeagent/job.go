package nodeagent

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// stopKind classifies why a running job is being stopped. The three are not
// interchangeable: a revoke keeps the lease token valid through the
// coordinator's grace period and ends in a canceled result, a gate breach ends
// in preempted (D62), and an orphaning ends in no coordinator write at all
// (D36).
type stopKind int

const (
	stopRevoked stopKind = iota + 1
	stopGate
	stopOrphaned
)

// stopReason carries the classification plus what the terminal result needs.
type stopReason struct {
	kind   stopKind
	force  bool
	detail string
	// gate names the breaching gate for a preemption, rendered as
	// reason "gate:<id>".
	gate           string
	nextEligibleAt *time.Time
}

// jobRun is one lease's lifecycle on this node.
type jobRun struct {
	agent    *Agent
	rec      *leaseRecord
	snapshot nashnet.SnapshotRef
	spec     Spec
	up       *uploader

	mu             sync.Mutex
	phase          string
	pid            int
	startedAt      time.Time
	logOffset      int64
	bytesUploaded  int64
	manifestSeq    int64
	manifestDigest string
	stop           *stopReason
	lastOK         time.Time

	stopOnce sync.Once
	stopCh   chan struct{}
	// syncing serializes the artifact cycle, which runs off the tick goroutine
	// so a multi-gigabyte upload can never starve the progress post that renews
	// the lease. At most one cycle is in flight, because the manifest chain is
	// fast-forward only and two concurrent commits would fight over seq.
	syncing  sync.Mutex
	syncBusy atomic.Bool
	syncDone chan struct{}
}

// runDir is where this job's artifacts live on the node.
func (j *jobRun) runDir() string {
	return filepath.Join(j.agent.cfg.RunsDir, j.rec.JobID)
}

// usesAccelerator reports whether the job's device is an accelerator, the
// input to the concurrency gate's max_accelerator_jobs.
func (j *jobRun) usesAccelerator() bool {
	d := j.spec.device()
	return d != "" && d != "cpu"
}

// requestStop records the first stop request and wakes the tick loop. Later
// requests are dropped: the first classification is the one the result carries.
func (j *jobRun) requestStop(r stopReason) {
	j.stopOnce.Do(func() {
		j.mu.Lock()
		j.stop = &r
		j.mu.Unlock()
		close(j.stopCh)
	})
}

func (j *jobRun) stopRequest() *stopReason {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.stop
}

// init builds the per-lease state both entry points share.
func (j *jobRun) init(phase string) {
	j.stopCh = make(chan struct{})
	j.syncDone = make(chan struct{})
	close(j.syncDone)
	j.lastOK = j.agent.now()
	j.phase = phase
	j.up = &uploader{
		client:     j.agent.client,
		leaseID:    j.rec.LeaseID,
		leaseEpoch: j.rec.LeaseEpoch,
		token:      j.rec.Token,
		runDir:     j.runDir(),
		index:      LoadIndex(indexPath(j.agent.cfg.BaseDir, j.rec.JobID)),
		policy:     j.rec.Policy,
		log:        j.agent.log,
		// An embedded run wrote its artifacts at the destination, so there is
		// nothing to push: the coordinator proves each digest against the file
		// already in the run dir and the commit still validates and records
		// every entry (D40).
		inPlace: j.agent.inPlace,
	}
}

// run drives one freshly claimed lease from claim to result.
func (j *jobRun) run(ctx context.Context) {
	j.init(nashnet.PhaseClaimed)

	spec, err := decodeSpec(j.rec.Spec)
	if err != nil {
		j.agent.log.Printf("lease %s: %v", j.rec.LeaseID, err)
		j.postFailure(ctx, err.Error())
		return
	}
	j.spec = spec

	if !j.stage(ctx) {
		return
	}
	j.supervise(ctx)
}

// resume picks a reattached lease back up (D37): the process is already
// forked (or already exited), so the node re-diffs its run dir against the
// coordinator's head on the next tick and resumes an interrupted upload at
// the HEAD offset.
func (j *jobRun) resume(ctx context.Context) {
	phase := j.rec.Phase
	if nashnet.ValidateNodePhase(phase) != nil {
		phase = nashnet.PhaseRunning
	}
	j.init(phase)
	j.mu.Lock()
	if st := j.agent.launcher.Status(j.spec.Name); st.Found {
		j.pid = st.PID
	}
	j.mu.Unlock()
	j.postProgress(ctx)
	j.supervise(ctx)
}

// stage runs the pre-launch half: fetch, gate check, prepare, launch. It
// returns false when the lease ended before the job started (a nack or a
// spec-fatal failure).
func (j *jobRun) stage(ctx context.Context) bool {
	j.setPhase(nashnet.PhaseFetching)
	j.postProgress(ctx)

	if err := j.fetch(ctx); err != nil {
		if errors.Is(err, ingest.ErrBundlePrereqMissing) {
			// The thin bundle's basis is not in this node's mirror. Drop the
			// advertised commits so the re-claim asks for a full tree (D48).
			j.agent.mu.Lock()
			j.agent.haveCommits = nil
			j.agent.mu.Unlock()
			j.nack(ctx, nashnet.NackBundlePrereqMiss, defaultNackCooldownSeconds, err.Error())
			return false
		}
		if errors.Is(err, errSeedUnavailable) {
			j.nack(ctx, nashnet.NackSeedUnavailable, defaultNackCooldownSeconds, err.Error())
			return false
		}
		j.nack(ctx, nashnet.NackSnapshotFailed, defaultNackCooldownSeconds, err.Error())
		return false
	}
	j.agent.noteCommit(j.rec.Commit)

	// Gate evaluation point two (D46): immediately before Prepare. A gate that
	// passed at claim and no longer holds is a scheduling event, not a job
	// failure, so it returns the claim with a cooldown.
	report, _ := j.agent.gateReport()
	if !report.Admit {
		checks := failingChecks(report)
		detail := "gate breach"
		if len(checks) > 0 {
			detail = "gate " + checks[0].Gate + ": " + checks[0].Detail
		}
		j.nack(ctx, nashnet.NackGateBreach, breachCooldown(report, j.agent.now()), detail)
		return false
	}

	j.setPhase(nashnet.PhasePreparing)
	j.postProgress(ctx)

	if err := j.agent.launcher.Ensure(j.spec.Name, j.spec.Kind); err != nil {
		j.nack(ctx, nashnet.NackPrepareNodeFailed, prepareFailedCooldownSeconds, err.Error())
		return false
	}
	prepared, err := j.agent.env.Prepare(ctx, j.spec.Name, j.rec.Commit, j.spec.Kind,
		j.spec.Config, j.spec.device(), j.spec.WarmStart, j.spec.overridesStr())
	if err != nil {
		if classified := classifyPrepare(err); isSpecFatal(classified) {
			j.postFailure(ctx, classified.Error())
			return false
		}
		j.nack(ctx, nashnet.NackPrepareNodeFailed, prepareFailedCooldownSeconds, err.Error())
		return false
	}

	if j.agent.inPlace {
		// Ingest writes its provenance record under its own name on an
		// embedded run, so the coordinator's env.json (which carries
		// executed_on, D23) and ingest's own copy both survive: the node's is
		// env.node.json, exactly the name a remote node's copy is promoted
		// under (D52, D64).
		j.agent.log.Printf("lease %s: staging %s in place", j.rec.LeaseID, j.spec.Name)
	}
	launch, lerr := BuildLaunch(j.spec, prepared, j.agent.cfg.RunsDir, j.agent.algo)
	if lerr != nil {
		if isSpecFatal(lerr) {
			j.postFailure(ctx, lerr.Error())
			return false
		}
		j.nack(ctx, nashnet.NackPrepareNodeFailed, prepareFailedCooldownSeconds, lerr.Error())
		return false
	}

	pid, serr := j.agent.launcher.Start(j.spec.Name, launch)
	if serr != nil {
		j.nack(ctx, nashnet.NackPrepareNodeFailed, prepareFailedCooldownSeconds, serr.Error())
		return false
	}
	j.mu.Lock()
	j.pid = pid
	j.startedAt = j.agent.now()
	j.mu.Unlock()
	j.setPhase(nashnet.PhaseRunning)

	j.rec.Launched = true
	j.rec.Phase = nashnet.PhaseRunning
	if err := writeLeaseRecord(j.agent.cfg.BaseDir, j.rec); err != nil {
		j.agent.log.Printf("lease %s: persist: %v", j.rec.LeaseID, err)
	}
	j.postProgress(ctx)
	return true
}

// supervise runs the progress-tick loop until the job reaches a terminal
// status, a stop is requested, or the lease is lost.
func (j *jobRun) supervise(ctx context.Context) {
	interval := j.rec.progressInterval()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	poll := time.NewTicker(j.agent.poll)
	defer poll.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-j.stopCh:
			j.applyStop(ctx)
			return
		case <-poll.C:
			if st := j.agent.launcher.Status(j.spec.Name); st.Found && isTerminalStatus(st.Status) {
				j.finish(ctx, st)
				return
			}
		case <-ticker.C:
			if !j.tick(ctx) {
				return
			}
		}
	}
}

// tick is one progress cadence: evaluate gates, push logs, sync artifacts,
// post progress. It returns false when the lease ended.
func (j *jobRun) tick(ctx context.Context) bool {
	// Gate evaluation point three (D46): a breach while running applies that
	// gate's on_breach.
	report, _ := j.agent.gateReport()
	if !report.Admit {
		for _, c := range failingChecks(report) {
			switch c.OnBreach {
			case "stop":
				j.requestStop(stopReason{kind: stopGate, gate: c.Gate, nextEligibleAt: c.NextEligibleAt})
				j.applyStop(ctx)
				return false
			case "drain":
				j.agent.setGateDrain(true)
			}
		}
	}

	j.pushLogs(ctx)
	j.startSync(ctx)

	if _, err := j.postProgress(ctx); err != nil {
		if IsFenced(err) {
			j.requestStop(stopReason{kind: stopOrphaned, detail: err.Error()})
			j.applyStop(ctx)
			return false
		}
	}
	// Loss of contact for two lease TTLs obliges the node to stop, whether or
	// not the coordinator ever answered with a fencing code (D36).
	if j.agent.now().Sub(j.lastContact()) > 2*j.rec.ttl() {
		j.requestStop(stopReason{kind: stopOrphaned, detail: "no coordinator contact for 2x lease ttl"})
		j.applyStop(ctx)
		return false
	}
	return true
}

// startSync runs one artifact cycle off the tick goroutine, at most one at a
// time. Uploading inline would block the progress post that renews the lease,
// so a checkpoint larger than one TTL's worth of bandwidth would cost the node
// the very lease it is uploading under.
func (j *jobRun) startSync(ctx context.Context) {
	if !j.syncBusy.CompareAndSwap(false, true) {
		return
	}
	done := make(chan struct{})
	j.syncDone = done
	go func() {
		defer close(done)
		defer j.syncBusy.Store(false)
		j.syncing.Lock()
		defer j.syncing.Unlock()
		resp, err := j.up.Sync(ctx, false)
		if err != nil {
			if IsFenced(err) {
				j.requestStop(stopReason{kind: stopOrphaned, detail: err.Error()})
				return
			}
			j.agent.log.Printf("lease %s: sync: %v", j.rec.LeaseID, err)
			return
		}
		j.noteCommit(resp)
	}()
}

// awaitSync waits for an in-flight artifact cycle so the final commit is the
// last writer of the manifest chain.
func (j *jobRun) awaitSync(ctx context.Context) {
	done := j.syncDone
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-ctx.Done():
	case <-time.After(stopWait):
	}
}

// applyStop performs the stop the reason names and posts whatever terminal it
// implies.
func (j *jobRun) applyStop(ctx context.Context) {
	r := j.stopRequest()
	if r == nil {
		return
	}
	if j.rec.Launched {
		if err := j.agent.launcher.Stop(j.spec.Name, r.force); err != nil {
			j.agent.log.Printf("lease %s: stop %s: %v", j.rec.LeaseID, j.spec.Name, err)
		}
		j.awaitExit(ctx)
	}
	if r.kind == stopOrphaned {
		j.agent.log.Printf("lease %s: orphaned (%s); local run dir kept for the debug ttl", j.rec.LeaseID, r.detail)
		removeLeaseRecord(j.agent.cfg.BaseDir, j.rec.JobID)
		return
	}
	state := nashnet.ResultCanceled
	lastError := r.detail
	if r.kind == stopGate {
		state = nashnet.ResultPreempted
		lastError = "gate:" + r.gate
	}
	j.postTerminal(ctx, state, nil, lastError)
}

// awaitExit waits for the stopped job's process to reach a terminal status so
// the final manifest hashes settled bytes.
func (j *jobRun) awaitExit(ctx context.Context) {
	deadline := j.agent.now().Add(2 * stopWait)
	for {
		st := j.agent.launcher.Status(j.spec.Name)
		if !st.Found || isTerminalStatus(st.Status) {
			return
		}
		if j.agent.now().After(deadline) {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(j.agent.poll):
		}
	}
}

// stopWait bounds how long the node waits for a stopped job to exit; procmgr's
// own SIGINT grace is 30s.
const stopWait = 35 * time.Second

// finish handles a job that reached a terminal status on its own.
func (j *jobRun) finish(ctx context.Context, st ProcessStatus) {
	j.pushLogs(ctx)
	state := nashnet.ResultStopped
	if st.Status == procmgr.StatusCrashed {
		state = nashnet.ResultCrashed
	}
	j.postTerminal(ctx, state, st.ExitCode, "")
}

// postTerminal commits the final manifest and posts the result. The result is
// the commit point and is gated on the artifact commit (D6), so the manifest
// goes first and its digest travels with the result.
func (j *jobRun) postTerminal(ctx context.Context, state string, exitCode *int, lastError string) {
	j.awaitSync(ctx)
	j.setPhase(nashnet.PhaseUploading)
	j.postProgress(ctx)
	j.syncing.Lock()
	resp, err := j.up.Sync(ctx, true)
	j.syncing.Unlock()
	if err != nil {
		if IsFenced(err) {
			j.agent.log.Printf("lease %s: final commit refused, output orphaned: %v", j.rec.LeaseID, err)
			removeLeaseRecord(j.agent.cfg.BaseDir, j.rec.JobID)
			return
		}
		j.agent.log.Printf("lease %s: final commit: %v", j.rec.LeaseID, err)
	} else {
		j.noteCommit(resp)
	}
	j.setPhase(nashnet.PhaseCommitting)
	j.postProgress(ctx)

	req := nashnet.ResultRequest{
		LeaseEpoch:          j.rec.LeaseEpoch,
		State:               state,
		ExitCode:            exitCode,
		LastError:           lastError,
		FinishedAt:          j.agent.now().UTC().Format(time.RFC3339Nano),
		FinalManifestDigest: j.digest(),
		Attempt:             j.rec.Attempt,
	}
	if _, err := j.agent.client.Result(ctx, j.rec.LeaseID, j.rec.Token, req); err != nil {
		// A result on a superseded lease is terminal for this attempt: the
		// node treats its output as orphaned and keeps the run dir (D6).
		j.agent.log.Printf("lease %s: result: %v", j.rec.LeaseID, err)
	}
	_ = j.up.index.Save()
	removeLeaseRecord(j.agent.cfg.BaseDir, j.rec.JobID)
	if j.spec.Name != "" {
		if err := j.agent.env.Cleanup(j.spec.Name, state == nashnet.ResultCrashed); err != nil {
			j.agent.log.Printf("lease %s: cleanup: %v", j.rec.LeaseID, err)
		}
	}
}

// postFailure fails a job the node cannot run at this commit at all (D63
// spec-fatal). It still commits a final manifest first, because a result is
// gated on one.
func (j *jobRun) postFailure(ctx context.Context, detail string) {
	j.postTerminal(ctx, nashnet.ResultFailed, nil, detail)
}

// nack returns an unrunnable claim (D8). The job goes back to the ready set at
// its original position with no attempt increment, and this node is excluded
// from re-matching it for the cooldown.
func (j *jobRun) nack(ctx context.Context, reason string, cooldown int, detail string) {
	req := nashnet.NackRequest{
		LeaseEpoch:      j.rec.LeaseEpoch,
		Reason:          reason,
		CooldownSeconds: cooldown,
		Detail:          truncate(detail, 512),
	}
	if err := j.agent.client.Nack(ctx, j.rec.LeaseID, j.rec.Token, req); err != nil {
		j.agent.log.Printf("lease %s: nack %s: %v", j.rec.LeaseID, reason, err)
	}
	removeLeaseRecord(j.agent.cfg.BaseDir, j.rec.JobID)
}

// postProgress renews the lease and drives the coordinator's projection (D5).
// The phase is validated against the node vocabulary before it goes out, so a
// value outside it, stopping included, can never leave this node.
func (j *jobRun) postProgress(ctx context.Context) (nashnet.ProgressResponse, error) {
	phase := j.currentPhase()
	if err := nashnet.ValidateNodePhase(phase); err != nil {
		j.agent.log.Printf("lease %s: refusing to report phase %q: %v", j.rec.LeaseID, phase, err)
		return nashnet.ProgressResponse{}, err
	}
	report, _ := j.agent.gateReport()

	j.mu.Lock()
	req := nashnet.ProgressRequest{
		LeaseEpoch:     j.rec.LeaseEpoch,
		Phase:          phase,
		PID:            j.pid,
		ManifestSeq:    j.manifestSeq,
		ManifestDigest: j.manifestDigest,
		LogOffset:      j.logOffset,
		BytesUploaded:  j.bytesUploaded,
		GateReport:     mustJSON(report),
	}
	if !j.startedAt.IsZero() {
		req.StartedAt = j.startedAt.UTC().Format(time.RFC3339Nano)
	}
	j.mu.Unlock()
	req.RunDBRows = countRunDBRows(filepath.Join(j.runDir(), runDBName))

	resp, err := j.agent.client.Progress(ctx, j.rec.LeaseID, j.rec.Token, req)
	if err != nil {
		if IsFenced(err) {
			j.requestStop(stopReason{kind: stopOrphaned, detail: err.Error()})
		}
		return resp, err
	}
	j.mu.Lock()
	j.lastOK = j.agent.now()
	j.mu.Unlock()
	if resp.LeaseDeadline != "" {
		j.rec.Deadline = resp.LeaseDeadline
	}
	// A tick carries the same hold every other call does, so a lift reaches a
	// node holding a lease without waiting for it to fall idle and heartbeat.
	j.agent.applyHold(resp.Hold)
	if resp.Revoke {
		j.requestStop(stopReason{kind: stopRevoked, force: resp.Force, detail: "coordinator revoke"})
	}
	return resp, nil
}

// pushLogs appends whatever the job wrote since the last append, at the exact
// offset the coordinator holds (D54). A 409 offset_mismatch carries the true
// offset, so the node seeks rather than duplicating or losing bytes.
func (j *jobRun) pushLogs(ctx context.Context) {
	if j.agent.inPlace {
		// The embedded node's job writes straight into the file the log route
		// appends to, so pushing would duplicate every line (D40).
		return
	}
	limit := j.rec.Policy.LogBytesPerCall
	if limit <= 0 {
		limit = nashnet.DefaultLogBytesPerCall
	}
	for {
		j.mu.Lock()
		offset := j.logOffset
		j.mu.Unlock()

		chunk, next, err := logTail(logPath(j.runDir()), offset, limit)
		if err != nil {
			j.agent.log.Printf("lease %s: log tail: %v", j.rec.LeaseID, err)
			return
		}
		if len(chunk) == 0 {
			return
		}
		if err := j.agent.client.AppendLog(ctx, j.rec.LeaseID, j.rec.Token, offset, chunk); err != nil {
			if api, ok := AsAPIError(err); ok && api.Code == nashnet.CodeOffsetMismatch {
				j.mu.Lock()
				j.logOffset = api.Offset
				j.mu.Unlock()
				continue
			}
			if IsFenced(err) {
				j.requestStop(stopReason{kind: stopOrphaned, detail: err.Error()})
			}
			j.agent.log.Printf("lease %s: log append: %v", j.rec.LeaseID, err)
			return
		}
		j.mu.Lock()
		j.logOffset = next
		j.bytesUploaded += int64(len(chunk))
		j.mu.Unlock()
		if next-offset < limit {
			return
		}
	}
}

// errSeedUnavailable classifies a seed fetch failure so the nack that follows
// names seed_unavailable rather than snapshot_failed (D8).
var errSeedUnavailable = errors.New("seed unavailable")

// fetch stages the job's inputs: the code snapshot and every granted seed. An
// embedded node shares the coordinator's mirror and runs dir, so both are
// already where the launch will look for them and the whole step is skipped
// (D40); nothing else about the lease changes.
func (j *jobRun) fetch(ctx context.Context) error {
	if j.agent.inPlace {
		return nil
	}
	if err := j.fetchSnapshot(ctx); err != nil {
		return err
	}
	if err := j.fetchSeeds(ctx); err != nil {
		return fmt.Errorf("%w: %v", errSeedUnavailable, err)
	}
	return nil
}

// fetchSnapshot downloads the coordinator-served bundle, verifies its digest,
// and imports it into the node's own mirror. The sha256 is transport integrity
// and cache correctness; the pin is checked by the receipt verification inside
// Prepare, the same code that checks it on the coordinator (D48).
func (j *jobRun) fetchSnapshot(ctx context.Context) error {
	dir := filepath.Join(j.agent.cfg.BaseDir, "snapshots")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	dest := filepath.Join(dir, j.snapshotCacheKey()+bundleExt)
	if err := j.download(ctx, j.snapshotURL(), dest, j.snapshot.SHA256); err != nil {
		return err
	}
	if err := j.agent.env.BundleFetch(ctx, j.rec.JobID, j.rec.Commit, dest); err != nil {
		return err
	}
	return nil
}

// bundleExt is the suffix of a cached bundle on the node, the same one the
// coordinator's own snapshot cache uses.
const bundleExt = ".bundle"

// snapshotCacheKey names the node's cached bundle the way the coordinator
// names the artifact it serves: the pinned commit for a full bundle, and the
// commit plus a digest of the sorted thin basis for a thin one (D48, the
// coordinator's own key in ingest.bundleCacheKey). Keying by job id instead
// pointed two claims of one job at one file whose bytes belonged to neither
// key, so the second claim resumed a complete file and asked for a range past
// the end of what the coordinator serves (cambia-2018). A claim whose snapshot
// ref carries no 40-hex commit falls back to the job id, which is the shape
// this replaces and no worse than it.
func (j *jobRun) snapshotCacheKey() string {
	commit := j.snapshot.Commit
	if commit == "" {
		commit = j.rec.Commit
	}
	if !isCommitSHA(commit) {
		return j.rec.JobID
	}
	if len(j.snapshot.ThinBasis) == 0 {
		return commit
	}
	sorted := append([]string(nil), j.snapshot.ThinBasis...)
	sort.Strings(sorted)
	sum := sha256.Sum256([]byte(strings.Join(sorted, "\n")))
	return commit + "-" + hex.EncodeToString(sum[:])[:16]
}

// isCommitSHA reports whether s is a 40-hex sha, the only commit shape the
// coordinator pins and the only one safe to spell into a cache file name.
func isCommitSHA(s string) bool {
	if len(s) != 40 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// snapshotURL is the coordinator-relative snapshot path, defaulted from the
// lease id when the claim response carried none.
func (j *jobRun) snapshotURL() string {
	if j.snapshot.URL != "" {
		return j.snapshot.URL
	}
	return "/nashnet/leases/" + j.rec.LeaseID + "/snapshot"
}

// fetchSeeds materializes every granted seed entry under the node's own runs
// dir, verifying each digest (D53). A seed named "resume" lands in the job's
// own run dir; any other seed id names the run directory it belongs to, which
// is what an evaluate target, a head-to-head checkpoint, and a measure read
// each resolve against.
func (j *jobRun) fetchSeeds(ctx context.Context) error {
	for _, seed := range j.rec.Seeds {
		root, err := j.seedRoot(seed)
		if err != nil {
			return err
		}
		for _, entry := range seed.Entries {
			if err := pathguard.CheckRel(entry.Path); err != nil {
				return fmt.Errorf("seed %s: %s: %w", seed.SeedID, entry.Path, err)
			}
			dest := filepath.Join(root, filepath.FromSlash(entry.Path))
			if digest, derr := hashFile(dest); derr == nil {
				if digest == entry.SHA256 {
					continue
				}
				// A materialized seed is 0444, so a re-fetch cannot open it for
				// writing; drop the stale copy first.
				if err := os.Remove(dest); err != nil {
					return fmt.Errorf("seed %s/%s: %w", seed.SeedID, entry.Path, err)
				}
			}
			url := fmt.Sprintf("/nashnet/leases/%s/seeds/%s/%s", j.rec.LeaseID, seed.SeedID, entry.Path)
			if err := j.download(ctx, url, dest, entry.SHA256); err != nil {
				return fmt.Errorf("seed %s/%s: %w", seed.SeedID, entry.Path, err)
			}
			// A seeded input is read-only on the node: a job that must write
			// next to one is given its own run dir for that (D64).
			_ = os.Chmod(dest, 0o444)
		}
	}
	return nil
}

// seedRoot resolves where a seed group materializes.
func (j *jobRun) seedRoot(seed nashnet.Seed) (string, error) {
	if seed.SeedID == "resume" {
		return j.runDir(), nil
	}
	if err := procmgr.ValidateName(seed.SeedID); err != nil {
		return "", fmt.Errorf("seed id %q: %w", seed.SeedID, err)
	}
	return filepath.Join(j.agent.cfg.RunsDir, seed.SeedID), nil
}

// download fetches a lease-scoped file, resuming at whatever is already on
// disk, and verifies the expected digest before the file is used. A complete
// copy already on disk is the answer and no request goes out; a resume the
// coordinator answers 416 for is decided the same way, by the digest, and
// restarts from zero when the local bytes are not the artifact (cambia-2018).
func (j *jobRun) download(ctx context.Context, url, dest, wantDigest string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if holdsComplete(dest, wantDigest) {
		return nil
	}
	for attempt := 0; attempt < 2; attempt++ {
		offset := int64(0)
		if fi, err := os.Stat(dest); err == nil {
			offset = fi.Size()
		}
		f, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE, 0o644)
		if err != nil {
			return err
		}
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			f.Close()
			return err
		}
		_, derr := j.agent.client.Download(ctx, j.rec.Token, url, offset, f)
		closeErr := f.Close()
		if derr != nil {
			if errors.Is(derr, errRangeIgnored) {
				_ = os.Remove(dest)
				continue
			}
			if errors.Is(derr, errRangeUnsatisfiable) {
				// The resume offset is at or past the end of the served file.
				// Either the local copy is the whole artifact, which the
				// digest says, or it is not this artifact at all and the fetch
				// starts over from zero.
				if holdsComplete(dest, wantDigest) {
					return nil
				}
				_ = os.Remove(dest)
				continue
			}
			return derr
		}
		if closeErr != nil {
			return closeErr
		}
		got, herr := hashFile(dest)
		if herr != nil {
			return herr
		}
		if wantDigest != "" && got != wantDigest {
			_ = os.Remove(dest)
			if attempt == 0 {
				continue
			}
			return fmt.Errorf("digest mismatch for %s: got %s, want %s", url, got, wantDigest)
		}
		return nil
	}
	return fmt.Errorf("download %s: exhausted retries", url)
}

// holdsComplete reports whether dest already holds the whole artifact the
// grant names, which is the one thing that makes skipping a fetch safe. An
// unreadable file, a missing one, and a grant that names no digest are all
// answered false: without a digest to check against there is nothing to
// validate the local bytes with, so they are re-fetched.
func holdsComplete(dest, wantDigest string) bool {
	if wantDigest == "" {
		return false
	}
	got, err := hashFile(dest)
	return err == nil && got == wantDigest
}

// noteCommit records the manifest head the coordinator acknowledged.
func (j *jobRun) noteCommit(resp quarantine.CommitResponse) {
	if resp.Digest == "" {
		return
	}
	j.mu.Lock()
	j.manifestSeq = resp.Seq
	j.manifestDigest = resp.Digest
	j.mu.Unlock()
}

func (j *jobRun) digest() string {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.manifestDigest
}

func (j *jobRun) setPhase(p string) {
	j.mu.Lock()
	j.phase = p
	j.mu.Unlock()
}

func (j *jobRun) currentPhase() string {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.phase
}

func (j *jobRun) lastContact() time.Time {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.lastOK
}

// isTerminalStatus reports whether a status is one a job does not leave. It
// spans both alphabets: the procmgr terminals and the runnerd-level ones a
// gate or an operator writes.
func isTerminalStatus(status string) bool {
	switch status {
	case procmgr.StatusStopped, procmgr.StatusCrashed, "canceled", "failed", "skipped":
		return true
	}
	return false
}

// countRunDBRows counts the journal rows the node has staged so far, carried
// on progress for operator visibility only; the coordinator counts its own
// after validation (D55). Any failure reports zeros rather than blocking the
// heartbeat this call is attached to.
func countRunDBRows(dbPath string) nashnet.RunDBRows {
	var out nashnet.RunDBRows
	if _, err := os.Stat(dbPath); err != nil {
		return out
	}
	db, err := sql.Open("sqlite", "file:"+dbPath+"?mode=ro&_pragma=busy_timeout(1000)")
	if err != nil {
		return out
	}
	defer db.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = db.QueryRowContext(ctx, "SELECT COUNT(*) FROM checkpoints").Scan(&out.Checkpoints)
	_ = db.QueryRowContext(ctx, "SELECT COUNT(*) FROM eval_results").Scan(&out.Evals)
	return out
}

// truncate bounds a detail string so a nack or a result never carries an
// unbounded error text.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n]
}
