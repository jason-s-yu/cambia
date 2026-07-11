package harness

import (
	"database/sql"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// TestRunDBCheckpointFoldsWAL creates a WAL-mode sqlite db with committed rows
// sitting in the -wal file, hits the checkpoint endpoint, and asserts the -wal
// file is truncated and the rows are readable straight off the main db file.
func TestRunDBCheckpointFoldsWAL(t *testing.T) {
	r := newRig(t, rigConfig{})
	dir := filepath.Join(r.runsDir, "ckpt-run")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(dir, "run_db.sqlite")

	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=journal_mode(WAL)")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("CREATE TABLE rows (id INTEGER PRIMARY KEY, val TEXT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("INSERT INTO rows (val) VALUES ('a'), ('b'), ('c')"); err != nil {
		t.Fatal(err)
	}
	// Keep this connection open, as a live writer would, so the WAL genuinely
	// carries the committed rows instead of the pool auto-closing and
	// checkpointing them away before the endpoint runs.
	defer db.Close()

	walPath := dbPath + "-wal"
	walBefore, err := os.Stat(walPath)
	if err != nil {
		t.Fatalf("expected a -wal file before checkpoint: %v", err)
	}
	if walBefore.Size() == 0 {
		t.Fatal("expected a non-empty -wal file before checkpoint (nothing to fold)")
	}

	resp := r.do(http.MethodPost, "/harness/jobs/ckpt-run/rundb-checkpoint", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("checkpoint: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	if walAfter, err := os.Stat(walPath); err == nil && walAfter.Size() != 0 {
		t.Fatalf("-wal not truncated after checkpoint: size=%d", walAfter.Size())
	}

	// A fresh read-only connection sees the rows straight off the main db
	// file: nothing but a fully folded-in main file is consulted here.
	check, err := sql.Open("sqlite", "file:"+dbPath+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer check.Close()
	var count int
	if err := check.QueryRow("SELECT count(*) FROM rows").Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("row count in main db file = %d, want 3", count)
	}
}

// TestRunDBCheckpointTraversalRefused mirrors the ValidateName path-traversal
// guard every other job-id route relies on: a name containing ".." is
// rejected before it reaches the filesystem.
func TestRunDBCheckpointTraversalRefused(t *testing.T) {
	r := newRig(t, rigConfig{})
	resp := r.do(http.MethodPost, "/harness/jobs/bad..name/rundb-checkpoint", nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("traversal name: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestRunDBCheckpointSymlinkEscapeRefused pins the pathguard defense-in-depth
// layer beyond ValidateName's lexical check: a run_db.sqlite planted as a
// symlink pointing outside the runs dir must not be followed.
func TestRunDBCheckpointSymlinkEscapeRefused(t *testing.T) {
	r := newRig(t, rigConfig{})
	dir := filepath.Join(r.runsDir, "symlink-run")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(t.TempDir(), "outside.sqlite")
	if err := os.WriteFile(secret, []byte("not a run db"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(dir, "run_db.sqlite")); err != nil {
		t.Fatal(err)
	}

	resp := r.do(http.MethodPost, "/harness/jobs/symlink-run/rundb-checkpoint", nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("symlink escape: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestRunDBCheckpointMissingDBReturns404 covers both an absent run dir and a
// run dir with no run_db.sqlite yet (e.g. a job that has not written its db):
// both are a clean 404, not an error.
func TestRunDBCheckpointMissingDBReturns404(t *testing.T) {
	r := newRig(t, rigConfig{})

	resp := r.do(http.MethodPost, "/harness/jobs/no-such-run/rundb-checkpoint", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("missing run dir: got %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()

	dir := filepath.Join(r.runsDir, "run-without-db")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	resp = r.do(http.MethodPost, "/harness/jobs/run-without-db/rundb-checkpoint", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("missing run_db.sqlite: got %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()
}
