package ingest

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// TestServeRangedFileETagAndRange is AC(5)'s Range half: the ETag header is
// exactly the caller's precomputed value, and a Range request returns the
// exact byte window with a 206 and a correct Content-Range.
func TestServeRangedFileETagAndRange(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact.bin")
	content := []byte("0123456789abcdefghijklmnopqrstuvwxyz")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}
	etag := FormatETag("deadbeefcafe")

	// Full GET: ETag present, full body returned.
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	if err := ServeRangedFile(rr, req, path, etag); err != nil {
		t.Fatalf("ServeRangedFile: %v", err)
	}
	if got := rr.Header().Get("ETag"); got != etag {
		t.Fatalf("ETag = %q, want %q", got, etag)
	}
	if got := rr.Header().Get("Accept-Ranges"); got != "bytes" {
		t.Fatalf("Accept-Ranges = %q, want bytes", got)
	}
	if rr.Body.String() != string(content) {
		t.Fatalf("full body = %q, want %q", rr.Body.String(), content)
	}

	// Range GET: bytes=5-9 returns exactly that 5-byte window with 206.
	rr2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodGet, "/x", nil)
	req2.Header.Set("Range", "bytes=5-9")
	if err := ServeRangedFile(rr2, req2, path, etag); err != nil {
		t.Fatalf("ServeRangedFile (range): %v", err)
	}
	if rr2.Code != http.StatusPartialContent {
		t.Fatalf("status = %d, want %d", rr2.Code, http.StatusPartialContent)
	}
	wantBody := string(content[5:10])
	if rr2.Body.String() != wantBody {
		t.Fatalf("range body = %q, want %q", rr2.Body.String(), wantBody)
	}
	wantRange := fmt.Sprintf("bytes 5-9/%d", len(content))
	if got := rr2.Header().Get("Content-Range"); got != wantRange {
		t.Fatalf("Content-Range = %q, want %q", got, wantRange)
	}

	// A Range request scoped to the tail also lands on the exact bytes, proving
	// the window math is not an artifact of the first case.
	rr3 := httptest.NewRecorder()
	req3 := httptest.NewRequest(http.MethodGet, "/x", nil)
	req3.Header.Set("Range", fmt.Sprintf("bytes=%d-", len(content)-3))
	if err := ServeRangedFile(rr3, req3, path, etag); err != nil {
		t.Fatalf("ServeRangedFile (tail range): %v", err)
	}
	if rr3.Body.String() != string(content[len(content)-3:]) {
		t.Fatalf("tail range body = %q, want %q", rr3.Body.String(), content[len(content)-3:])
	}
}

// TestServeRangedFileWithBundleDigest exercises the actual coordinator path:
// an ETag built from a real BundleCreate descriptor's digest, served from the
// cache file BundleCreate wrote.
func TestServeRangedFileWithBundleDigest(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-serve")

	desc, err := m.BundleCreate(t.Context(), "job-serve", nil)
	if err != nil {
		t.Fatalf("BundleCreate: %v", err)
	}
	etag := FormatETag(desc.SHA256)

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	if err := ServeRangedFile(rr, req, desc.Path, etag); err != nil {
		t.Fatalf("ServeRangedFile: %v", err)
	}
	if got := rr.Header().Get("ETag"); got != etag {
		t.Fatalf("ETag = %q, want %q", got, etag)
	}
	if int64(rr.Body.Len()) != desc.Size {
		t.Fatalf("served %d bytes, want %d (descriptor size)", rr.Body.Len(), desc.Size)
	}
}
