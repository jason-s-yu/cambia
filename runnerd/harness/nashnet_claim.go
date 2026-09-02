package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// handleClaim is POST /nashnet/claim (D2): the long-poll handout of one
// placeable job. It answers 200 with a job and its lease, or 204 with the hold
// that explains the idleness. The acting node is the verified subject; a
// node_id in the body that disagrees is 403 wrong_subject, so a claim cannot be
// recorded under another node's identity (D25).
func (s *Server) handleClaim(w http.ResponseWriter, r *http.Request, nodeID string) {
	p := s.pool
	var req nashnet.ClaimRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if req.NodeID != "" && req.NodeID != nodeID {
		nashnetError(w, http.StatusForbidden, nashnet.CodeWrongSubject, "node_id does not match the verified subject")
		return
	}
	cand, err := p.candidateFor(nodeID, req)
	if err != nil {
		nashnetError(w, http.StatusUnprocessableEntity, "invalid_declaration", err.Error())
		return
	}
	if _, err := p.nodes.ObserveClaim(nodeID, req); err != nil {
		writeRegistryError(w, err)
		return
	}
	if p.effectiveHold(nodeID) != "" {
		// A coordinator-side hold (an operator drain or the D63 breaker) is not
		// the node's own gate, and it is never clearable by the node: a node that
		// could clear its own hold would loop claim, nack, register, claim at
		// request rate while nothing ever accumulated against it.
		p.hold(w, nashnet.HoldNodeGated)
		return
	}
	if !cand.report.Admit {
		// Two-sided admission: the node owns pool admission and says no, so the
		// coordinator does not scan on its behalf (D11). The ready jobs this
		// node would have matched still record the gate hold, so an operator
		// reads a time window rather than a capability gap (D14).
		p.disp.noteGateDenied(cand)
		p.hold(w, nashnet.HoldNodeGated)
		return
	}

	waiter, ok := p.addWaiter()
	if !ok {
		p.hold(w, nashnet.HoldWaitersFull)
		return
	}
	defer p.dropWaiter(waiter)

	wait := clampWait(req.WaitSeconds)
	setDeadlines(w, r, wait+deadlineDefault)
	timer := time.NewTimer(wait)
	defer timer.Stop()
	for {
		resp, hold := p.placeOnce(r.Context(), cand, req)
		if resp != nil {
			writeJSON(w, http.StatusOK, resp)
			return
		}
		if wait <= 0 {
			p.hold(w, hold)
			return
		}
		select {
		case <-waiter.wake:
		case <-timer.C:
			p.hold(w, hold)
			return
		case <-r.Context().Done():
			return
		}
	}
}

// HeaderClaimHold and HeaderClaimRetry are the header names of nashnet's own
// wire vocabulary, re-exported here for the route tests that read them. Both
// sides of the claim read the one spelling, so a rename cannot silently
// degrade every hold to no_match on the node.
const (
	HeaderClaimHold  = nashnet.HeaderClaimHold
	HeaderClaimRetry = nashnet.HeaderClaimRetry
)

// hold answers a claim that placed nothing (D2).
func (p *Pool) hold(w http.ResponseWriter, reason string) {
	if reason == "" {
		reason = nashnet.HoldNoMatch
	}
	retry := p.policy.ProgressIntervalSeconds
	if retry <= 0 {
		retry = nashnet.DefaultProgressIntervalSeconds
	}
	w.Header().Set(HeaderClaimHold, reason)
	w.Header().Set(HeaderClaimRetry, itoa(retry))
	w.Header().Set("Retry-After", itoa(retry))
	w.WriteHeader(http.StatusNoContent)
}

// placeOnce runs one pass of the placement scan and, on a match, resolves the
// seeds and the snapshot outside the placement lock, grants the lease, and
// authors env.json. A job whose seeds cannot be resolved is marked and the scan
// continues, so one bad job never stalls the pool: a claim does not name a job,
// so answering it with an error would hand every node reaching that job in the
// scan an error instead of the next placeable one (D53).
func (p *Pool) placeOnce(ctx context.Context, cand candidate, req nashnet.ClaimRequest) (*nashnet.ClaimResponse, string) {
	skip := map[string]bool{}
	for {
		picked, hold := p.disp.scanForNode(cand, skip, func(jobID string) bool {
			return p.heldFor(cand.nodeID, jobID)
		})
		if picked == nil {
			return nil, hold
		}
		resp, err := p.grantFor(ctx, cand, req, picked)
		p.disp.releasePlacing(picked.jobID)
		if err == nil {
			return resp, ""
		}
		skip[picked.jobID] = true
		var se *seedError
		switch {
		case errors.As(err, &se):
			if se.fatal {
				p.disp.recordPoolTerminal(picked.jobID, StateFailed, "seed_missing: "+se.detail)
			} else {
				p.disp.markUnplaceable(picked.jobID, PlacementSeedMissing, se.detail)
			}
		case errors.Is(err, nashnet.ErrJobLeased):
			// Another claim won the race between the scan and the grant.
		default:
			poolLog("nashnet claim: %s not placed on %s: %v", picked.jobID, cand.nodeID, err)
		}
	}
}

// grantFor turns a reserved pick into a lease: it resolves the grant set (the
// snapshot digest and the seed entries), grants the lease, writes the
// coordinator-authored env.json, and builds the claim response.
func (p *Pool) grantFor(ctx context.Context, cand candidate, req nashnet.ClaimRequest, picked *pick) (*nashnet.ClaimResponse, error) {
	seeds, grants, err := p.resolveSeeds(picked.spec, picked.spec.Resume)
	if err != nil {
		return nil, err
	}
	// The bundle build runs here, outside the placement lock: a cache-miss
	// build forks a whole-repository git bundle create, and one node's cold
	// fetch must not stall every other node's claim (D48). The reservation the
	// scan took is what keeps a second claim off this job meanwhile.
	desc, err := p.resolveSnapshot(ctx, cand.nodeID, picked.jobID, req.HaveCommits)
	if err != nil {
		return nil, err
	}

	grantSet := nashnet.GrantSet{Snapshot: desc.SHA256, Seeds: map[string][]nashnet.GrantEntry{}}
	for _, sd := range seeds {
		entries := make([]nashnet.GrantEntry, 0, len(sd.Entries))
		for _, e := range sd.Entries {
			entries = append(entries, nashnet.GrantEntry{Path: e.Path, SHA256: e.SHA256})
		}
		grantSet.Seeds[sd.SeedID] = entries
	}

	// The attempt this lease runs at is the store's to decide: the prior lease's
	// verdict recorded it, one higher for a requeue and unchanged for a nack
	// (D8, D32). The floor passed here is the scan's, which is the first attempt
	// for a job that has never been leased.
	lease, token, err := p.leases.Grant(nashnet.GrantRequest{
		JobID:      picked.jobID,
		NodeID:     cand.nodeID,
		NodeEpoch:  p.nodeEpoch(cand.nodeID),
		Attempt:    picked.attempt,
		GrantSet:   grantSet,
		MaxRuntime: p.leaseRuntimeCap(cand, picked.spec),
	})
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.snapshots[lease.LeaseID] = desc
	p.quarGrants[lease.LeaseID] = grants
	p.mu.Unlock()

	// env.json carries executed_on for the whole life of a running pool job,
	// which is what v1.0 delivers from Prepare onward (D23).
	if err := p.writeEnvRecord(picked.jobID, lease, picked.spec.Commit, "", nil); err != nil {
		poolLog("nashnet claim: env.json for %s: %v", picked.jobID, err)
	}
	specJSON, err := json.Marshal(picked.spec)
	if err != nil {
		return nil, err
	}
	return &nashnet.ClaimResponse{
		JobID:         lease.JobID,
		LeaseID:       lease.LeaseID,
		LeaseEpoch:    lease.LeaseEpoch,
		LeaseDeadline: rfc3339(lease.Deadline),
		LeaseToken:    token,
		Spec:          specJSON,
		Resume:        picked.spec.Resume,
		Attempt:       lease.Attempt,
		Snapshot: nashnet.SnapshotRef{
			URL:       "/nashnet/leases/" + lease.LeaseID + "/snapshot",
			Commit:    picked.spec.Commit,
			SHA256:    desc.SHA256,
			Size:      desc.Size,
			ThinBasis: desc.Basis,
		},
		Seeds:  seeds,
		Policy: p.policy,
	}, nil
}

// leaseRuntimeCap is the lease-lifetime bound this grant carries (D4): the
// smaller of the job's own max_runtime_hours and the claiming node's declared
// job_policy.max_runtime_hours, with zero meaning the pool's own
// RUNNERD_NASHNET_MAX_LEASE_SECONDS applies alone.
//
// The node's number only ever shortens a lease. The coordinator's bound does
// not depend on it, which is what D4 means by the node policy staying
// node-evaluated: a node that omits the gate, as a hostile one does by
// construction, is held to the pool cap exactly as before.
func (p *Pool) leaseRuntimeCap(cand candidate, spec JobSpec) time.Duration {
	cap := time.Duration(spec.MaxRuntimeHours * float64(time.Hour))
	if h := nodeMaxRuntimeHours(cand.report); h > 0 {
		nodeCap := time.Duration(h * float64(time.Hour))
		if cap <= 0 || nodeCap < cap {
			cap = nodeCap
		}
	}
	return cap
}

// nodeEpoch reads the registry's current epoch for a node, which is the fence a
// lease carries (D4).
func (p *Pool) nodeEpoch(nodeID string) int64 {
	if rec, ok := p.nodes.Get(nodeID); ok {
		return rec.NodeEpoch
	}
	return 0
}

// snapshotDescriptor is a built bundle plus the basis the claim negated, so the
// claim response can echo thin_basis.
type snapshotDescriptor struct {
	ingest.BundleDescriptor
	Basis []string
}

// resolveSnapshot builds or reuses the job's git bundle (D48). A node over its
// build budget is served the cached full-tree bundle rather than a fresh build,
// so a claim-then-nack loop cannot evict the shared bundles out of the LRU.
func (p *Pool) resolveSnapshot(ctx context.Context, nodeID, jobID string, have []string) (snapshotDescriptor, error) {
	if p.bundles == nil {
		return snapshotDescriptor{}, errors.New("no bundle builder configured")
	}
	basis := have
	if len(basis) > 0 && !p.builds.allow(nodeID) {
		basis = nil
	}
	desc, err := p.bundles.BundleCreate(ctx, jobID, basis)
	if err != nil && len(basis) > 0 {
		// A basis the mirror cannot honor is not a claim failure: fall back to
		// the shared full-tree bundle.
		desc, err = p.bundles.BundleCreate(ctx, jobID, nil)
		basis = nil
	}
	if err != nil {
		return snapshotDescriptor{}, err
	}
	return snapshotDescriptor{BundleDescriptor: desc, Basis: basis}, nil
}

// seedError names a seed the placement scan could not resolve. fatal marks the
// spec-fatal case of D53, where the referenced run dir is gone entirely.
type seedError struct {
	detail string
	fatal  bool
}

func (e *seedError) Error() string { return "seed unresolved: " + e.detail }

// resolveSeeds resolves every input a job reads into the lease grant set (D53):
// the job's own promoted run dir on a resume (excluding reservoir/, which never
// leaves the node that wrote it), an evaluate target, head-to-head checkpoints,
// a train warm start, and a measure job's reads. A seed id is the source run's
// name and every entry path is relative to that run dir, so the coordinator
// resolves a fetch through pathguard against its own runs dir and a restart
// rebuilds the mapping from the lease record alone.
func (p *Pool) resolveSeeds(spec JobSpec, resume bool) ([]nashnet.Seed, map[string]quarantine.Grant, error) {
	sources := seedSources(spec, resume)
	grouped := map[string][]nashnet.SeedEntry{}
	kinds := map[string]string{}
	grants := map[string]quarantine.Grant{}

	for _, src := range sources {
		seedID, rel, err := splitSeedRef(src)
		if err != nil {
			return nil, nil, &seedError{detail: src + ": " + err.Error(), fatal: true}
		}
		root := filepath.Join(p.runsDir, seedID)
		if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
			return nil, nil, &seedError{detail: "run dir " + seedID + " is gone", fatal: true}
		}
		abs, err := pathguard.Resolve(p.runsDir, src)
		if err != nil {
			return nil, nil, &seedError{detail: src + ": " + err.Error(), fatal: true}
		}
		info, err := os.Stat(abs)
		if err != nil {
			return nil, nil, &seedError{detail: src + " no longer exists"}
		}
		kind := "file"
		var files []string
		if info.IsDir() {
			kind = "run_dir"
			files, err = seedFiles(abs)
			if err != nil {
				return nil, nil, &seedError{detail: src + ": " + err.Error()}
			}
		} else {
			files = []string{abs}
		}
		if _, ok := kinds[seedID]; !ok || kind == "run_dir" {
			kinds[seedID] = kind
		}
		_ = rel
		for _, f := range files {
			relPath, rerr := filepath.Rel(root, f)
			if rerr != nil {
				continue
			}
			relPath = filepath.ToSlash(relPath)
			if seedEntryExcluded(relPath) {
				continue
			}
			fi, serr := os.Stat(f)
			if serr != nil {
				return nil, nil, &seedError{detail: relPath + " vanished while resolving"}
			}
			sum, herr := sha256File(f)
			if herr != nil {
				return nil, nil, &seedError{detail: relPath + ": " + herr.Error()}
			}
			grouped[seedID] = append(grouped[seedID], nashnet.SeedEntry{
				Path: relPath, Size: fi.Size(), SHA256: sum,
			})
			grants[sum] = quarantine.Grant{Digest: sum, Size: fi.Size(), SourcePath: f}
		}
	}

	ids := make([]string, 0, len(grouped))
	for id := range grouped {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	seeds := make([]nashnet.Seed, 0, len(ids))
	for _, id := range ids {
		entries := grouped[id]
		sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
		seeds = append(seeds, nashnet.Seed{SeedID: id, Kind: kinds[id], Entries: entries})
	}
	return seeds, grants, nil
}

// seedSources lists the runs-dir-relative inputs a job reads, by kind (D53,
// D38). A resume adds the job's own run dir, which is what the reservoir pin of
// D12 is about; every other source is a different run's promoted output.
func seedSources(spec JobSpec, resume bool) []string {
	var out []string
	if resume {
		out = append(out, spec.Name)
	}
	switch spec.Kind {
	case KindEvaluate:
		if spec.Target != "" {
			out = append(out, spec.Target)
		}
	case KindHeadToHead:
		if spec.CheckpointA != "" {
			out = append(out, spec.CheckpointA)
		}
		if spec.CheckpointB != "" {
			out = append(out, spec.CheckpointB)
		}
	case KindMeasure:
		out = append(out, spec.Reads...)
	case KindTrain:
		if spec.WarmStart != "" {
			out = append(out, spec.WarmStart)
		}
	}
	return out
}

// seedEntryExcluded drops the paths a seed never carries. It is exactly the
// coordinator-authored reserved list of D52, read from the quarantine store so
// there is one such list rather than a second copy that could drift: the
// reservoir, whose transfer cost is unbounded and whose only consumer is a
// resume on the same host (D12, D50), plus the placement records and the
// manifest state, none of which is a job input.
func seedEntryExcluded(rel string) bool {
	return quarantine.ReservedPath(rel)
}

// seedFiles lists every regular file under a seed's source directory.
func seedFiles(root string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		out = append(out, path)
		return nil
	})
	return out, err
}

// splitSeedRef splits a runs-dir-relative reference into the source run's name
// and the remainder. The run name is validated as a run name, so a seed id is
// safe as a path segment and a map key.
func splitSeedRef(ref string) (string, string, error) {
	if err := pathguard.CheckRel(ref); err != nil {
		return "", "", err
	}
	clean := filepath.ToSlash(filepath.Clean(ref))
	parts := strings.SplitN(clean, "/", 2)
	if err := procmgr.ValidateName(parts[0]); err != nil {
		return "", "", err
	}
	if len(parts) == 1 {
		return parts[0], "", nil
	}
	return parts[0], parts[1], nil
}

// candidateFor builds the placement candidate for a claim: the declaration
// parsed, bounds-checked, and clamped by the enrollment grant's caps, the gate
// report as reported, and the claim's volatile facts (D9, D46, D47).
func (p *Pool) candidateFor(nodeID string, req nashnet.ClaimRequest) (candidate, error) {
	decl, err := p.declarationFor(nodeID, req.Capabilities)
	if err != nil {
		return candidate{}, err
	}
	var report gates.Report
	if len(req.GateReport) > 0 {
		if err := json.Unmarshal(req.GateReport, &report); err != nil {
			return candidate{}, fmt.Errorf("gate_report: %w", err)
		}
	}
	kinds := map[string]bool{}
	for _, k := range req.Kinds {
		kinds[k] = true
	}
	busy := map[string]bool{}
	for _, id := range req.BusyJobIDs {
		busy[id] = true
	}
	slots := req.SlotsFree
	if slots <= 0 {
		slots = report.SlotsOffered
	}
	if slots > decl.Slots && decl.Slots > 0 {
		slots = decl.Slots
	}
	room := p.maxLeases
	if decl.Slots > 0 && decl.Slots < room {
		room = decl.Slots
	}
	room -= p.leases.LiveCountForNode(nodeID)

	var labels []string
	if g, err := p.grants.Grant(nodeID); err == nil {
		labels = g.Caps.Labels
	}
	return candidate{
		nodeID:      nodeID,
		declaration: decl,
		grantLabels: labels,
		report:      report,
		kinds:       kinds,
		slotsFree:   slots,
		leaseRoom:   room,
		busy:        busy,
	}, nil
}

// declarationFor parses a node's declaration, validates its shape and bounds,
// and clamps it by the enrollment grant before placement reads it. A node
// reporting 1000 slots or a label it was not granted attracts nothing beyond
// what its enrollment allowed (D9, D47).
func (p *Pool) declarationFor(nodeID string, raw json.RawMessage) (capability.Declaration, error) {
	var decl capability.Declaration
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &decl); err != nil {
			return decl, fmt.Errorf("capabilities: %w", err)
		}
		if err := decl.Validate(); err != nil {
			return decl, err
		}
	}
	g, err := p.grants.Grant(nodeID)
	if err != nil {
		return decl, err
	}
	return capability.Clamp(decl, g.Caps), nil
}
