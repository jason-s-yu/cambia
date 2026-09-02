package ingest

import (
	"net/http"
	"os"
	"path/filepath"
)

// FormatETag returns the canonical `"sha256:<hex>"` ETag value (design 3.3,
// D48 code delivery; D53 seed transport) for content whose sha256 digest is
// hexDigest. Callers already hold the digest (a BundleDescriptor.SHA256 or a
// seed manifest entry's recorded sha256), so this never hashes anything
// itself.
func FormatETag(hexDigest string) string {
	return `"sha256:` + hexDigest + `"`
}

// ServeRangedFile serves the file at path as an HTTP response, honoring
// conditional-request and byte-Range semantics via net/http.ServeContent. etag
// is the caller's precomputed ETag (see FormatETag): a bundle cache
// descriptor's digest, or a seed manifest entry's recorded digest. It is never
// recomputed here, so serving a large cached artifact repeatedly never
// re-hashes it, and an interrupted fetch resumes at its byte offset via a
// Range request (net/http.ServeContent answers those from etag/modtime
// without any range-parsing code of our own).
//
// This is the transport primitive shared by the coordinator-served git bundle
// (design 3.3, D48) and the seed transport (D53); it wires no HTTP route
// itself, only the byte-serving behavior both need.
func ServeRangedFile(w http.ResponseWriter, r *http.Request, path, etag string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("ETag", etag)
	http.ServeContent(w, r, filepath.Base(path), info.ModTime(), f)
	return nil
}
