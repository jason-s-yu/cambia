package nodeagent

import (
	"context"
	"database/sql"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// TestScanRunDirExcludesReservedPaths covers AC7's first half: the uploaded
// set never contains a coordinator-authored or reserved path, and the node's
// own env.json is offered under the promoted name env.node.json.
func TestScanRunDirExcludesReservedPaths(t *testing.T) {
	runDir := t.TempDir()
	files := map[string]string{
		"metrics.jsonl":                   "{}",
		"config.yaml":                     "a: 1",
		"env.json":                        "{}",
		"process.json":                    "{}",
		"jobspec.json":                    "{}",
		"lease.json":                      "{}",
		"run_db.sqlite":                   "db",
		"run_db.sqlite-wal":               "wal",
		"run_db.sqlite-shm":               "shm",
		"scratch.tmp":                     "x",
		"snapshots/prtcfr_iter_10.pt":     "weights",
		"logs/training.log":               "line",
		"reservoir/meta.json":             "{}",
		"reservoir/p0/shard.bin":          "bytes",
		".nashnet/current.json":           "{}",
		".nashnet-tmp/staging":            "x",
		"eval_summary.jsonl":              "{}",
		"evaluations/iter_10/result.json": "{}",
	}
	for rel, body := range files {
		path := filepath.Join(runDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	scanned, err := scanRunDir(runDir)
	if err != nil {
		t.Fatalf("scanRunDir: %v", err)
	}
	got := map[string]bool{}
	for _, f := range scanned {
		got[f.rel] = true
	}

	want := []string{
		"metrics.jsonl", "config.yaml", "run_db.sqlite", envNodeJSONName,
		"snapshots/prtcfr_iter_10.pt", "eval_summary.jsonl", "evaluations/iter_10/result.json",
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("scan dropped %s", w)
		}
	}
	reserved := []string{
		"env.json", "process.json", "jobspec.json", "lease.json",
		"run_db.sqlite-wal", "run_db.sqlite-shm", "scratch.tmp",
		"logs/training.log", "reservoir/meta.json", "reservoir/p0/shard.bin",
		".nashnet/current.json", ".nashnet-tmp/staging",
	}
	for _, r := range reserved {
		if got[r] {
			t.Errorf("reserved path %s survived the scan", r)
		}
		if !quarantine.ReservedPath(r) {
			t.Errorf("quarantine.ReservedPath(%q) is false; the node filter and the coordinator enforcement disagree", r)
		}
	}
}

// TestFoldRunDBTruncatesWAL covers AC7's second half (D22): the journal's WAL
// is folded node-side, before the file is hashed, so exactly one
// self-contained journal file crosses the wire.
func TestFoldRunDBTruncatesWAL(t *testing.T) {
	runDir := t.TempDir()
	dbPath := filepath.Join(runDir, runDBName)
	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("CREATE TABLE runs (id INTEGER PRIMARY KEY, name TEXT);"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		if _, err := db.Exec("INSERT INTO runs (name) VALUES (?)", "row"); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	beforeDigest, err := hashFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := foldRunDB(dbPath); err != nil {
		t.Fatalf("foldRunDB: %v", err)
	}
	if fi, err := os.Stat(dbPath + "-wal"); err == nil && fi.Size() != 0 {
		t.Errorf("wal file is %d bytes after a TRUNCATE checkpoint", fi.Size())
	}
	afterDigest, err := hashFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if beforeDigest == afterDigest {
		t.Logf("journal digest unchanged by the fold (the driver had already checkpointed on close)")
	}

	// Whatever the fold did, the sibling files stay off the wire.
	scanned, err := scanRunDir(runDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range scanned {
		if f.rel != runDBName {
			t.Errorf("unexpected journal sibling offered: %s", f.rel)
		}
	}
}

// TestSyncFoldsBeforeHashing asserts the ordering the design requires: the
// digest a commit carries is taken after the WAL fold, never before.
func TestSyncFoldsBeforeHashing(t *testing.T) {
	stub := newStubCoordinator(t)
	agent, cfg := testAgent(t, stub, nil)
	runDir := filepath.Join(cfg.RunsDir, "job-fold")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(runDir, runDBName)
	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("CREATE TABLE runs (id INTEGER PRIMARY KEY);"); err != nil {
		t.Fatal(err)
	}
	// Leave the connection open so the -wal file is still present when Sync
	// runs, which is the live-training-process case D22 describes.
	defer db.Close()

	stub.mu.Lock()
	stub.token = "lease-token-fold"
	stub.mu.Unlock()
	up := &uploader{
		client: agent.client, leaseID: "01JFOLD00000000000000000A", token: "lease-token-fold",
		runDir: runDir, index: LoadIndex(filepath.Join(t.TempDir(), "idx.json")),
		policy: fastPolicy(), log: log.New(io.Discard, "", 0),
	}
	walBefore := int64(0)
	if fi, serr := os.Stat(dbPath + "-wal"); serr == nil {
		walBefore = fi.Size()
	}
	if walBefore == 0 {
		t.Fatalf("the test did not produce a non-empty wal, so it cannot prove the fold ran")
	}

	resp, err := up.Sync(context.Background(), true)
	if err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if fi, serr := os.Stat(dbPath + "-wal"); serr == nil && fi.Size() != 0 {
		t.Errorf("wal is %d bytes after Sync; the fold did not run before hashing", fi.Size())
	}
	if resp.Digest == "" {
		t.Fatalf("Sync returned no manifest digest")
	}

	_, _, _, commits, _ := stub.snapshotState()
	if len(commits) != 1 {
		t.Fatalf("expected one commit, got %d", len(commits))
	}
	var journal *quarantine.Entry
	for i := range commits[0].Entries {
		if commits[0].Entries[i].Path == runDBName {
			journal = &commits[0].Entries[i]
		}
		if strings.HasSuffix(commits[0].Entries[i].Path, "-wal") || strings.HasSuffix(commits[0].Entries[i].Path, "-shm") {
			t.Errorf("a journal sibling reached the manifest: %s", commits[0].Entries[i].Path)
		}
	}
	if journal == nil {
		t.Fatalf("the journal was not offered: %+v", commits[0].Entries)
	}
	onDisk, err := hashFile(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	if journal.Digest != onDisk {
		t.Errorf("the committed journal digest %s is not the folded file's digest %s", journal.Digest, onDisk)
	}
}

// TestDiffComputesChangesAndDeletes pins the uploader's diff against the
// coordinator's head.
func TestDiffComputesChangesAndDeletes(t *testing.T) {
	head := ManifestHead{Seq: 3, Digest: "d3", Entries: []quarantine.Entry{
		{Path: "a", Digest: "1"},
		{Path: "b", Digest: "2"},
	}}
	local := []quarantine.Entry{
		{Path: "a", Digest: "1"},
		{Path: "c", Digest: "3"},
	}
	changed, deletes := diff(head, local)
	if len(changed) != 1 || changed[0].Path != "c" {
		t.Errorf("changed = %+v, want only c", changed)
	}
	if len(deletes) != 1 || deletes[0] != "b" {
		t.Errorf("deletes = %v, want [b]", deletes)
	}
}
