package quarantine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// retiredStamp is written when a lease reaches its result. The tree and its
// receipt are retained for the debug TTL past that moment and are then removed
// whole; because promotion links, removing blobs/ leaves the run dir holding
// the only link and no reference counting is needed (D59).
type retiredStamp struct {
	RetiredAt int64 `json:"retired_at"`
}

// Retire stamps a lease's tree as finished, starting its debug TTL. It is
// called at the result commit (D6) and is idempotent.
func (s *Store) Retire(l Lease) error {
	dir, err := s.leaseDir(l)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dir); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	data, err := json.Marshal(retiredStamp{RetiredAt: s.now().UnixNano()})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, retiredFile), append(data, '\n'), 0o600)
}

// SweepParts reaps in-flight parts idle past the part TTL and returns how many
// it removed. A part is the only upload state there is, so reaping one costs
// the node a re-upload and nothing else.
func (s *Store) SweepParts() (int, error) {
	cutoff := s.now().Add(-s.lim.PartTTL)
	removed := 0
	var firstErr error
	err := s.eachLeaseTree(func(_, _, _, dir string) {
		entries, err := os.ReadDir(filepath.Join(dir, dirParts))
		if err != nil {
			return
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), partSuffix) {
				continue
			}
			fi, err := e.Info()
			if err != nil || fi.ModTime().After(cutoff) {
				continue
			}
			p := filepath.Join(dir, dirParts, e.Name())
			if rmErr := os.Remove(p); rmErr != nil && !os.IsNotExist(rmErr) {
				if firstErr == nil {
					firstErr = rmErr
				}
				continue
			}
			s.forgetPart(dir, e.Name())
			removed++
		}
	})
	if err != nil {
		return removed, err
	}
	return removed, firstErr
}

// PurgeJob removes every lease tree a job ever had, under every node, and
// returns how many it removed. It is what an operator purge calls after the run
// dir is gone (D31): the debug TTL of D59 retains a tree so a receipt outlives
// the result, and a purged run has nothing left for that receipt to describe.
//
// It walks every node rather than the job's current lease alone, because a job
// re-claimed after an expiry or a nack left a tree under each node that ever
// held it, and one skipped tree is quarantine bytes no later sweep attributes
// to anything.
func (s *Store) PurgeJob(jobID string) (int, error) {
	if err := procmgr.ValidateName(jobID); err != nil {
		return 0, fmt.Errorf("quarantine: purge %q: %w", jobID, err)
	}
	nodes, err := os.ReadDir(s.root)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	removed := 0
	var firstErr error
	for _, n := range nodes {
		if !n.IsDir() {
			continue
		}
		jobDir := filepath.Join(s.root, n.Name(), jobID)
		leases, rerr := os.ReadDir(jobDir)
		if rerr != nil {
			continue
		}
		if rmErr := os.RemoveAll(jobDir); rmErr != nil {
			if firstErr == nil {
				firstErr = rmErr
			}
			continue
		}
		for _, le := range leases {
			if !le.IsDir() {
				continue
			}
			s.forgetLease(n.Name(), jobID, le.Name())
			removed++
		}
	}
	return removed, firstErr
}

// LiveLeases reports whether a lease is still held by the lease store. Sweep
// consults it so a tree belonging to a live lease is never reaped underneath an
// upload in flight.
type LiveLeases func(nodeID, jobID, leaseID string) bool

// Sweep removes lease trees that are no longer needed and returns how many it
// removed. A tree goes when its job has no run dir (the run was purged), or
// when the lease is not live and its retirement, or its own last write if it
// was never retired, is older than the debug TTL. That single rule covers both
// D59 cases: the result-plus-TTL reap and the startup sweep of trees whose job
// or lease is gone, with the unknown-lease case still getting the TTL grace
// that makes the receipt readable after the fact.
func (s *Store) Sweep(live LiveLeases) (int, error) {
	if live == nil {
		live = func(string, string, string) bool { return false }
	}
	cutoff := s.now().Add(-s.lim.DebugTTL)
	removed := 0
	var firstErr error
	err := s.eachLeaseTree(func(nodeID, jobID, leaseID, dir string) {
		if live(nodeID, jobID, leaseID) {
			return
		}
		runDir := filepath.Join(s.runsDir, jobID)
		_, runErr := os.Stat(runDir)
		drop := os.IsNotExist(runErr)
		if !drop {
			drop = !s.retainedFor(dir, cutoff)
		}
		if !drop {
			return
		}
		if rmErr := os.RemoveAll(dir); rmErr != nil {
			if firstErr == nil {
				firstErr = rmErr
			}
			return
		}
		s.forgetLease(nodeID, jobID, leaseID)
		removed++
	})
	if err != nil {
		return removed, err
	}
	return removed, firstErr
}

// retainedFor reports whether a lease tree is still inside its debug TTL.
func (s *Store) retainedFor(dir string, cutoff time.Time) bool {
	if data, err := os.ReadFile(filepath.Join(dir, retiredFile)); err == nil {
		var st retiredStamp
		if json.Unmarshal(data, &st) == nil && st.RetiredAt > 0 {
			return time.Unix(0, st.RetiredAt).After(cutoff)
		}
	}
	fi, err := os.Stat(dir)
	if err != nil {
		return false
	}
	return fi.ModTime().After(cutoff)
}

// eachLeaseTree walks quarantine/<node>/<job>/<lease> and calls fn for each
// lease directory. A name that is not a valid run name is skipped rather than
// reaped, because it is not a tree this store wrote.
func (s *Store) eachLeaseTree(fn func(nodeID, jobID, leaseID, dir string)) error {
	nodes, err := os.ReadDir(s.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	for _, n := range nodes {
		if !n.IsDir() {
			continue
		}
		jobs, err := os.ReadDir(filepath.Join(s.root, n.Name()))
		if err != nil {
			continue
		}
		for _, j := range jobs {
			if !j.IsDir() {
				continue
			}
			leases, err := os.ReadDir(filepath.Join(s.root, n.Name(), j.Name()))
			if err != nil {
				continue
			}
			for _, le := range leases {
				if !le.IsDir() {
					continue
				}
				fn(n.Name(), j.Name(), le.Name(), filepath.Join(s.root, n.Name(), j.Name(), le.Name()))
			}
		}
	}
	return nil
}

// forgetLease drops the in-memory state of a reaped lease.
func (s *Store) forgetLease(nodeID, jobID, leaseID string) {
	key := nodeID + "/" + jobID + "/" + leaseID
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.used, key)
	delete(s.stats, key)
	for k := range s.parts {
		if strings.HasPrefix(k, key+"/") {
			delete(s.parts, k)
		}
	}
}

// forgetPart drops the running hash of a reaped part.
func (s *Store) forgetPart(leaseDir, partName string) {
	digest := strings.TrimSuffix(partName, partSuffix)
	rel, err := filepath.Rel(s.root, leaseDir)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.parts, filepath.ToSlash(rel)+"/"+digest)
	delete(s.used, filepath.ToSlash(rel))
}
