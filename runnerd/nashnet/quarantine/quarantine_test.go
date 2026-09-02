package quarantine

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"
)

// testClock is the injected clock of D41: no test waits on wall time.
type testClock struct {
	mu sync.Mutex
	at time.Time
}

func newClock() *testClock {
	return &testClock{at: time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)}
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.at.Add(d)
}

// acceptAll is the stub journal validator. The real one is cambia-1717
// (rundbcheck.go); every commit test that cares about the seam injects its own.
var acceptAll = ValidatorFunc(func(string) (JournalVerdict, string) { return JournalValid, "" })

// rejectRunDB is the stub verdict the D55 call-site test drives.
var rejectRunDB = ValidatorFunc(func(string) (JournalVerdict, string) { return JournalInvalid, ReasonRunDBInvalid })

type rig struct {
	store   *Store
	lease   Lease
	clock   *testClock
	quarDir string
	runsDir string
	runDir  string
}

// newRig builds a store over two temp dirs on the same filesystem, which is
// what link mode needs, and returns it with one lease already named.
func newRig(t *testing.T, tune func(*Config)) *rig {
	t.Helper()
	base := t.TempDir()
	r := &rig{
		clock:   newClock(),
		quarDir: filepath.Join(base, "quarantine"),
		runsDir: filepath.Join(base, "runs"),
	}
	r.runDir = filepath.Join(r.runsDir, "job-a")
	cfg := Config{
		QuarantineDir: r.quarDir,
		RunsDir:       r.runsDir,
		Now:           r.clock.Now,
		Validator:     acceptAll,
	}
	if tune != nil {
		tune(&cfg)
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r.store = s
	r.lease = Lease{NodeID: "node-a", JobID: "job-a", LeaseID: "lease-1", Epoch: 1}
	return r
}

func sha256hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// putBlob uploads data in one chunk and returns its digest.
func (r *rig) putBlob(t *testing.T, data []byte) string {
	t.Helper()
	d := sha256hex(data)
	res, err := r.store.AppendChunk(r.lease, d, 0, int64(len(data))-1, int64(len(data)), bytes.NewReader(data))
	if err != nil {
		t.Fatalf("AppendChunk %s: %v", d[:8], err)
	}
	if !res.Verified {
		t.Fatalf("AppendChunk %s: not verified after one full chunk", d[:8])
	}
	return d
}

func entry(path string, data []byte, mtime int64) Entry {
	return Entry{Digest: sha256hex(data), MTime: mtime, Path: path, Size: int64(len(data))}
}

// commit marshals req as the request body, the way the route does, and runs the
// promotion transaction.
func (r *rig) commit(t *testing.T, req CommitRequest) (CommitResponse, error) {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	return r.store.Commit(r.lease, req, body)
}

func (r *rig) mustCommit(t *testing.T, req CommitRequest) CommitResponse {
	t.Helper()
	resp, err := r.commit(t, req)
	if err != nil {
		t.Fatalf("Commit seq %d: %v", req.Seq, err)
	}
	return resp
}

// head reads the folded head off disk.
func (r *rig) head(t *testing.T) Head {
	t.Helper()
	h, err := r.store.ReadHead("job-a")
	if err != nil {
		t.Fatalf("ReadHead: %v", err)
	}
	return h
}

// treeSnapshot lists every path under the roots, so a test can assert that a
// refused request created nothing anywhere on disk.
func treeSnapshot(t *testing.T, roots ...string) []string {
	t.Helper()
	var out []string
	for _, root := range roots {
		err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
			if err != nil {
				if os.IsNotExist(err) {
					return nil
				}
				return err
			}
			out = append(out, p)
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
	sort.Strings(out)
	return out
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}

func rejectionReason(resp CommitResponse, path string) string {
	for _, r := range resp.Rejected {
		if r.Path == path {
			return r.Reason
		}
	}
	return ""
}
