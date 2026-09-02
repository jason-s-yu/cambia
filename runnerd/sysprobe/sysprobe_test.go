package sysprobe

import "testing"

// TestDefaultRAMQuery is a smoke test against the real /proc/meminfo on this
// Linux host: a positive GiB value with no error. The moved function's error
// and parse paths are exercised indirectly by harness's MinFreeRAMCheck
// tests, which inject RAMQueryFunc and never touch this real-filesystem path.
func TestDefaultRAMQuery(t *testing.T) {
	gb, err := DefaultRAMQuery()
	if err != nil {
		t.Fatalf("DefaultRAMQuery: %v", err)
	}
	if gb <= 0 {
		t.Fatalf("DefaultRAMQuery = %v, want > 0", gb)
	}
}

// TestDiskFreeGB checks a real, writable path reports positive free space and
// an unstatfs-able path reads as 0 (matching the pre-move behavior of failing
// open rather than the check).
func TestDiskFreeGB(t *testing.T) {
	dir := t.TempDir()
	if gb := DiskFreeGB(dir); gb <= 0 {
		t.Fatalf("DiskFreeGB(%q) = %v, want > 0", dir, gb)
	}

	missing := dir + "/does/not/exist/at/all"
	if gb := DiskFreeGB(missing); gb != 0 {
		t.Fatalf("DiskFreeGB(%q) = %v, want 0 for a statfs failure", missing, gb)
	}
}
