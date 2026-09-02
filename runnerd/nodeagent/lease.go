package nodeagent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
)

// leasesDirName is where a node persists its own lease records, 0600 each, so
// a restart can reattach to the jobs it was supervising (D37, D44).
const leasesDirName = "leases"

// leaseRecord is the node's copy of a granted lease: enough to reattach, to
// keep posting progress under the same epoch, and to resume an upload. The
// coordinator holds the authoritative record; this is the credential and the
// bookkeeping the node needs to keep using it.
type leaseRecord struct {
	JobID       string          `json:"job_id"`
	LeaseID     string          `json:"lease_id"`
	LeaseEpoch  int64           `json:"lease_epoch"`
	Token       string          `json:"lease_token"`
	Deadline    string          `json:"lease_deadline"`
	GrantedAt   string          `json:"granted_at"`
	Attempt     int             `json:"attempt"`
	Resume      bool            `json:"resume"`
	Commit      string          `json:"commit"`
	Spec        json.RawMessage `json:"spec"`
	Policy      nashnet.Policy  `json:"policy"`
	Seeds       []nashnet.Seed  `json:"seeds,omitempty"`
	SnapshotURL string          `json:"snapshot_url,omitempty"`
	SnapshotSHA string          `json:"snapshot_sha256,omitempty"`
	// Launched records that this node forked a process for the lease. It is
	// the node's own witness of the pre-launch versus post-launch split the
	// coordinator applies in D7, and it decides whether a restart reattaches
	// to a live row or simply drops the lease.
	Launched bool `json:"launched"`
	// Phase is the last phase reported, so a restarted agent resumes with the
	// phase it left off at rather than reporting claimed for a running job.
	Phase string `json:"phase"`
}

// leasePath is the on-disk location of one lease record.
func leasePath(baseDir, jobID string) string {
	return filepath.Join(baseDir, leasesDirName, jobID+".json")
}

// indexPath is where the hash index for one job is persisted, beside its lease
// record.
func indexPath(baseDir, jobID string) string {
	return filepath.Join(baseDir, leasesDirName, jobID+".index.json")
}

// writeLeaseRecord persists a lease record atomically at 0600.
func writeLeaseRecord(baseDir string, rec *leaseRecord) error {
	path := leasePath(baseDir, rec.JobID)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// removeLeaseRecord drops a lease record and its hash index.
func removeLeaseRecord(baseDir, jobID string) {
	os.Remove(leasePath(baseDir, jobID))
	os.Remove(indexPath(baseDir, jobID))
}

// readLeaseRecords loads every persisted lease record. A record that will not
// decode is skipped rather than failing startup: the coordinator revokes what
// the node does not name in live_leases, so a lost record costs one job, not
// the agent.
func readLeaseRecords(baseDir string) []*leaseRecord {
	dir := filepath.Join(baseDir, leasesDirName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []*leaseRecord
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".index.json") {
			continue
		}
		data, rerr := os.ReadFile(filepath.Join(dir, name))
		if rerr != nil {
			continue
		}
		var rec leaseRecord
		if json.Unmarshal(data, &rec) != nil || rec.JobID == "" || rec.LeaseID == "" {
			continue
		}
		out = append(out, &rec)
	}
	return out
}

// ttl is the lease TTL this record's policy states, defaulted when the policy
// was never recorded.
func (r *leaseRecord) ttl() time.Duration {
	if r.Policy.LeaseTTLSeconds > 0 {
		return time.Duration(r.Policy.LeaseTTLSeconds) * time.Second
	}
	return nashnet.DefaultLeaseTTLSeconds * time.Second
}

// progressInterval is the tick this record's policy states.
func (r *leaseRecord) progressInterval() time.Duration {
	if r.Policy.ProgressIntervalSeconds > 0 {
		return time.Duration(r.Policy.ProgressIntervalSeconds) * time.Second
	}
	return nashnet.DefaultProgressIntervalSeconds * time.Second
}

// liveLease is the entry this record contributes to a register request (D3).
// The hash is taken over the token text exactly as the coordinator stored it.
func (r *leaseRecord) liveLease() nashnet.LiveLease {
	return nashnet.LiveLease{
		LeaseID:   r.LeaseID,
		JobID:     r.JobID,
		TokenHash: nashnet.HashLeaseToken(r.Token),
	}
}
