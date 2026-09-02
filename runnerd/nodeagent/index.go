package nodeagent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

// indexEntry is one remembered hash: the identity facts that make a rehash
// unnecessary plus the digest they produced.
type indexEntry struct {
	Size    int64  `json:"size"`
	MTimeNS int64  `json:"mtime_ns"`
	Inode   uint64 `json:"inode"`
	Digest  string `json:"digest"`
}

// Index is the node's (path, size, mtime_ns, inode) -> digest memory (D50). A
// miss costs a rehash and never a wrong digest: every field of the identity
// must match before a remembered digest is reused, and the coordinator
// verifies the digest against the bytes it received regardless.
type Index struct {
	mu      sync.Mutex
	path    string
	entries map[string]indexEntry
	dirty   bool
}

// LoadIndex reads the index persisted beside the lease record, or returns an
// empty one. A missing or corrupt file is not an error: the worst outcome is
// that the next scan rehashes.
func LoadIndex(path string) *Index {
	idx := &Index{path: path, entries: map[string]indexEntry{}}
	data, err := os.ReadFile(path)
	if err != nil {
		return idx
	}
	var entries map[string]indexEntry
	if err := json.Unmarshal(data, &entries); err == nil && entries != nil {
		idx.entries = entries
	}
	return idx
}

// Digest returns the sha256 of the file at abs, reusing the remembered value
// when every identity fact still matches.
func (i *Index) Digest(key, abs string, fi os.FileInfo) (string, error) {
	ident := identity(fi)
	i.mu.Lock()
	if e, ok := i.entries[key]; ok && e.Size == ident.Size && e.MTimeNS == ident.MTimeNS && e.Inode == ident.Inode {
		i.mu.Unlock()
		return e.Digest, nil
	}
	i.mu.Unlock()

	digest, err := hashFile(abs)
	if err != nil {
		return "", err
	}
	ident.Digest = digest
	i.mu.Lock()
	i.entries[key] = ident
	i.dirty = true
	i.mu.Unlock()
	return digest, nil
}

// Forget drops a remembered entry, used when a path leaves the run dir.
func (i *Index) Forget(key string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if _, ok := i.entries[key]; ok {
		delete(i.entries, key)
		i.dirty = true
	}
}

// Save persists the index when it changed. A write failure is reported but
// never fatal: the index is a cache.
func (i *Index) Save() error {
	i.mu.Lock()
	if !i.dirty || i.path == "" {
		i.mu.Unlock()
		return nil
	}
	data, err := json.Marshal(i.entries)
	i.dirty = false
	i.mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(i.path), 0o700); err != nil {
		return err
	}
	tmp := i.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, i.path)
}

// identity extracts the size, mtime, and inode of a stat result.
func identity(fi os.FileInfo) indexEntry {
	e := indexEntry{Size: fi.Size(), MTimeNS: fi.ModTime().UnixNano()}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		e.Inode = st.Ino
	}
	return e
}

// hashFile streams a file through sha256.
func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
