package quarantine

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestSweepPartsReapsIdlePartsOnly covers the part TTL of D59: an idle part is
// reaped, a fresh one and every verified blob survive.
func TestSweepPartsReapsIdlePartsOnly(t *testing.T) {
	r := newRig(t, nil)
	leaseDir := filepath.Join(r.quarDir, "node-a", "job-a", "lease-1")

	stale := bytes.Repeat([]byte("stale"), 40)
	staleDigest := sha256hex(stale)
	if _, err := r.store.AppendChunk(r.lease, staleDigest, 0, 9, int64(len(stale)), bytes.NewReader(stale[:10])); err != nil {
		t.Fatalf("stale part: %v", err)
	}
	blobDigest := r.putBlob(t, []byte("verified content"))

	r.clock.advance(r.store.Limits().PartTTL + time.Hour)
	if err := os.Chtimes(partPath(leaseDir, staleDigest), r.clock.Now().Add(-2*r.store.Limits().PartTTL), r.clock.Now().Add(-2*r.store.Limits().PartTTL)); err != nil {
		t.Fatalf("age the part: %v", err)
	}

	fresh := bytes.Repeat([]byte("fresh"), 40)
	freshDigest := sha256hex(fresh)
	if _, err := r.store.AppendChunk(r.lease, freshDigest, 0, 9, int64(len(fresh)), bytes.NewReader(fresh[:10])); err != nil {
		t.Fatalf("fresh part: %v", err)
	}

	removed, err := r.store.SweepParts()
	if err != nil {
		t.Fatalf("SweepParts: %v", err)
	}
	if removed != 1 {
		t.Fatalf("SweepParts removed %d, want 1", removed)
	}
	if _, err := os.Stat(partPath(leaseDir, staleDigest)); !os.IsNotExist(err) {
		t.Fatalf("the idle part survived: %v", err)
	}
	if _, err := os.Stat(partPath(leaseDir, freshDigest)); err != nil {
		t.Fatalf("a fresh part was reaped: %v", err)
	}
	if _, err := os.Stat(blobPath(leaseDir, blobDigest)); err != nil {
		t.Fatalf("a verified blob was reaped: %v", err)
	}
}

// TestSweepRetainsARetiredLeaseForTheDebugTTL covers the D59 reap rule: the
// tree and its receipt stay readable for the debug TTL after the result and go
// whole afterwards, and a live lease is never touched.
func TestSweepRetainsARetiredLeaseForTheDebugTTL(t *testing.T) {
	r := newRig(t, nil)
	data := []byte("promoted artifact")
	r.putBlob(t, data)
	r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{entry("metrics.jsonl", data, 71)},
	})
	leaseDir := filepath.Join(r.quarDir, "node-a", "job-a", "lease-1")

	live := func(_, _, leaseID string) bool { return leaseID == "lease-1" }
	if removed, err := r.store.Sweep(live); err != nil || removed != 0 {
		t.Fatalf("Sweep over a live lease removed %d (err %v)", removed, err)
	}

	if err := r.store.Retire(r.lease); err != nil {
		t.Fatalf("Retire: %v", err)
	}
	dead := func(string, string, string) bool { return false }
	if removed, err := r.store.Sweep(dead); err != nil || removed != 0 {
		t.Fatalf("Sweep inside the debug TTL removed %d (err %v)", removed, err)
	}
	if _, err := r.store.Receipt(r.lease); err != nil {
		t.Fatalf("the receipt is unreadable inside the debug TTL: %v", err)
	}

	r.clock.advance(r.store.Limits().DebugTTL + time.Hour)
	removed, err := r.store.Sweep(dead)
	if err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	if removed != 1 {
		t.Fatalf("Sweep removed %d trees past the debug TTL, want 1", removed)
	}
	if _, err := os.Stat(leaseDir); !os.IsNotExist(err) {
		t.Fatalf("the retired tree survived past its TTL: %v", err)
	}
	// Promotion linked, so dropping blobs/ leaves the run dir holding the only
	// link and no reference counting is needed.
	if got := mustRead(t, filepath.Join(r.runDir, "metrics.jsonl")); !bytes.Equal(got, data) {
		t.Fatal("reaping the quarantine tree damaged a promoted file")
	}
}

// TestPurgeJobDropsEveryNodesTreeForThatJob covers the purge half of D31: every
// lease tree a job ever had goes, under every node that held one, and no other
// job's tree is touched.
func TestPurgeJobDropsEveryNodesTreeForThatJob(t *testing.T) {
	r := newRig(t, nil)
	r.putBlob(t, []byte("current lease upload"))

	// An earlier lease of the same job on another node, and a neighbouring
	// job's tree that must survive.
	earlier := filepath.Join(r.quarDir, "node-b", "job-a", "lease-0")
	neighbour := filepath.Join(r.quarDir, "node-a", "job-b", "lease-9")
	for _, dir := range []string{earlier, neighbour} {
		if err := os.MkdirAll(filepath.Join(dir, "blobs"), 0o700); err != nil {
			t.Fatal(err)
		}
	}

	removed, err := r.store.PurgeJob("job-a")
	if err != nil {
		t.Fatalf("PurgeJob: %v", err)
	}
	if removed != 2 {
		t.Fatalf("PurgeJob removed %d trees, want both of job-a's", removed)
	}
	for _, dir := range []string{
		filepath.Join(r.quarDir, "node-a", "job-a"),
		filepath.Join(r.quarDir, "node-b", "job-a"),
	} {
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("%s survived the purge: %v", dir, err)
		}
	}
	if _, err := os.Stat(neighbour); err != nil {
		t.Fatalf("PurgeJob removed another job's tree: %v", err)
	}
	if _, err := r.store.PurgeJob("../escape"); err == nil {
		t.Fatal("PurgeJob accepted a job id that is not a safe path segment")
	}
}

// TestSweepDropsATreeWhoseJobHasNoRunDir covers the startup half of D59: a tree
// whose run was purged goes at once, without waiting out the debug TTL.
func TestSweepDropsATreeWhoseJobHasNoRunDir(t *testing.T) {
	r := newRig(t, nil)
	r.putBlob(t, []byte("orphaned upload"))
	leaseDir := filepath.Join(r.quarDir, "node-a", "job-a", "lease-1")
	if _, err := os.Stat(leaseDir); err != nil {
		t.Fatalf("lease tree missing: %v", err)
	}

	removed, err := r.store.Sweep(func(string, string, string) bool { return false })
	if err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	if removed != 1 {
		t.Fatalf("Sweep removed %d, want the orphaned tree", removed)
	}
	if _, err := os.Stat(leaseDir); !os.IsNotExist(err) {
		t.Fatalf("the orphaned tree survived: %v", err)
	}
}
