package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// sha256hex returns the lowercase hex sha256 of b.
func sha256hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// sha256File returns the lowercase hex sha256 of a file's bytes.
func sha256File(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return sha256hex(b), nil
}

// touch bumps a cache entry's mtime to now so LRU eviction reflects use. Errors
// are ignored: a failed touch only weakens LRU ordering, never correctness.
func touch(path string, now time.Time) {
	_ = os.Chtimes(path, now, now)
}

// cacheEntry is a cache member with the LRU sort key (mtime) and its logical key
// (dir/file name minus any extension) used for protect-set membership.
type cacheEntry struct {
	path  string
	key   string
	mtime time.Time
}

// evictLRU retains the newest keep members of the removable pool plus every
// protected member (protected keys are kept in addition to the keep budget, per
// design "keep MaxVenvs + any referenced by live jobs"). It removes the oldest
// removable members beyond that budget by mtime. removeFn performs the physical
// removal (os.RemoveAll for dirs, os.Remove for files).
func evictLRU(entries []cacheEntry, keep int, protect map[string]bool, removeFn func(string) error) {
	if len(entries) <= keep {
		return
	}
	// Oldest first.
	sort.Slice(entries, func(i, j int) bool { return entries[i].mtime.Before(entries[j].mtime) })
	// Protected entries are always retained; only the rest form the LRU pool.
	removable := make([]cacheEntry, 0, len(entries))
	for _, e := range entries {
		if protect[e.key] {
			continue
		}
		removable = append(removable, e)
	}
	removeCount := len(removable) - keep // keep the newest `keep` removable ones
	for i := 0; i < len(removable) && removeCount > 0; i++ {
		if err := removeFn(removable[i].path); err == nil {
			removeCount--
		}
	}
}

// listVenvEntries returns the venv cache members (directories under venvsDir).
func (m *Manager) listVenvEntries() []cacheEntry {
	return listDirEntries(m.venvsDir, true, "")
}

// listLibcambiaEntries returns the libcambia cache members (*.so files).
func (m *Manager) listLibcambiaEntries() []cacheEntry {
	return listDirEntries(m.libcambiaDir, false, ".so")
}

// listDirEntries enumerates a cache directory. When wantDir is true it returns
// subdirectories (venvs); otherwise it returns files with the given suffix,
// stripping the suffix to form the logical key (libcambia). A missing directory
// yields no entries.
func listDirEntries(dir string, wantDir bool, suffix string) []cacheEntry {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []cacheEntry
	for _, e := range ents {
		if wantDir != e.IsDir() {
			continue
		}
		name := e.Name()
		if suffix != "" {
			if filepath.Ext(name) != suffix {
				continue
			}
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		key := name
		if suffix != "" {
			key = name[:len(name)-len(suffix)]
		}
		out = append(out, cacheEntry{
			path:  filepath.Join(dir, name),
			key:   key,
			mtime: info.ModTime(),
		})
	}
	return out
}

// evictVenvs trims the venv cache to MaxVenvs, protecting the given keys.
func (m *Manager) evictVenvs(protect map[string]bool) {
	evictLRU(m.listVenvEntries(), m.cfg.MaxVenvs, protect, os.RemoveAll)
}

// evictLibcambia trims the libcambia cache to MaxLibcambia, protecting the given
// engine-tree-sha keys.
func (m *Manager) evictLibcambia(protect map[string]bool) {
	evictLRU(m.listLibcambiaEntries(), m.cfg.MaxLibcambia, protect, os.Remove)
}
