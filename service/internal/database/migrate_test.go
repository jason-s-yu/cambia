package database

import (
	"testing"
)

// migrationFiles must order by the numeric prefix, not lexicographically:
// plain string order runs 10_ before 2_ once the set grows past nine files.
func TestMigrationFilesVersionOrder(t *testing.T) {
	files, err := migrationFiles()
	if err != nil {
		t.Fatalf("migrationFiles: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("expected the embedded migrations to be non-empty")
	}

	last := -1
	for _, name := range files {
		n, ok := versionPrefix(name)
		if !ok {
			t.Fatalf("migration %q has no numeric version prefix", name)
		}
		if n <= last {
			t.Fatalf("migration %q (version %d) is out of order after version %d: %v", name, n, last, files)
		}
		last = n
	}
	if files[0] != "0_init.sql" {
		t.Fatalf("expected 0_init.sql first, got %q", files[0])
	}
}

func TestVersionPrefix(t *testing.T) {
	cases := []struct {
		name string
		want int
		ok   bool
	}{
		{"0_init.sql", 0, true},
		{"10_later.sql", 10, true},
		{"2_add_thing.sql", 2, true},
		{"no_number.sql", 0, false},
		{"_leading.sql", 0, false},
		{"nounderscore.sql", 0, false},
	}
	for _, tc := range cases {
		got, ok := versionPrefix(tc.name)
		if got != tc.want || ok != tc.ok {
			t.Errorf("versionPrefix(%q) = (%d, %v), want (%d, %v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
}

// Every embedded file must be readable by name, since Migrate reads them back
// out of the FS by the same names migrationFiles returns.
func TestMigrationFilesAreReadable(t *testing.T) {
	files, err := migrationFiles()
	if err != nil {
		t.Fatalf("migrationFiles: %v", err)
	}
	for _, name := range files {
		body, err := readMigration(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if len(body) == 0 {
			t.Fatalf("migration %s is empty", name)
		}
	}
}

// MigrateIfEnabled must be a no-op without RUN_MIGRATIONS, including with a nil
// pool: an unset variable cannot be allowed to log.Fatalf a dev process.
func TestMigrateIfEnabledDisabledIsNoop(t *testing.T) {
	prev := DB
	DB = nil
	defer func() { DB = prev }()

	for _, val := range []string{"", "false", "0", "no"} {
		t.Setenv("RUN_MIGRATIONS", val)
		MigrateIfEnabled() // would log.Fatalf on a nil pool if it ran
	}
}

// Migrate refuses a nil pool rather than panicking.
func TestMigrateNilPool(t *testing.T) {
	if err := Migrate(t.Context(), nil); err == nil {
		t.Fatal("expected an error for a nil pool")
	}
}
