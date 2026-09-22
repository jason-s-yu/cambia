package quarantine

import (
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// runDBPyPath is cfr/src/run_db.py relative to this package directory
// (runnerd/nashnet/quarantine), which is where go test runs: the repo root is
// three levels up. run_db.py's _DDL and _COLUMN_MIGRATIONS are the schema a
// node's journal is written with, so they are what runDBSchema must cover
// (cambia-2358).
const runDBPyPath = "../../../cfr/src/run_db.py"

// genCorpusCommand regenerates the shared fixture corpus from run_db.py.
const genCorpusCommand = "cd cfr && python scripts/gen_rundb_fixtures.py"

// pySchema is what run_db.py can create: the _DDL literal (executable SQL),
// the columns of each CREATE TABLE in it, and the _COLUMN_MIGRATIONS entries
// get_db applies after it.
type pySchema struct {
	ddl        string
	tables     map[string][]string
	migrations map[string][]migrationColumn
}

type migrationColumn struct {
	name string
	decl string
}

// columns is every column get_db can leave on each table: the _DDL columns
// plus the _COLUMN_MIGRATIONS entries.
func (s pySchema) columns() map[string]map[string]bool {
	out := make(map[string]map[string]bool)
	add := func(table, col string) {
		if out[table] == nil {
			out[table] = make(map[string]bool)
		}
		out[table][col] = true
	}
	for table, cols := range s.tables {
		for _, col := range cols {
			add(table, col)
		}
	}
	for table, cols := range s.migrations {
		for _, col := range cols {
			add(table, col.name)
		}
	}
	return out
}

var (
	ddlAssign        = regexp.MustCompile(`(?ms)^_DDL\s*=\s*[rRuU]?"""(.*?)"""`)
	createTableHead  = regexp.MustCompile(`(?i)\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\x60\[]?(\w+)["\x60\]]?\s*\(`)
	createTableAny   = regexp.MustCompile(`(?i)\bCREATE\s+(?:TEMP\w*\s+)?TABLE\b`)
	migrationsAssign = regexp.MustCompile(`(?m)^_COLUMN_MIGRATIONS\b[^=\n]*=\s*\{`)
	migrationToken   = regexp.MustCompile(`["'](\w+)["']\s*:\s*\[|\(\s*["'](\w+)["']\s*,\s*["']([^"']*)["']\s*,?\s*\)`)
	pyStringLiteral  = regexp.MustCompile(`"[^"\n]*"|'[^'\n]*'`)
	evalMigrationKey = regexp.MustCompile(`["']eval_results["']\s*:\s*\[`)
)

// tableConstraintWords open a table-constraint clause inside CREATE TABLE
// rather than a column definition.
var tableConstraintWords = map[string]bool{
	"CONSTRAINT": true, "PRIMARY": true, "UNIQUE": true, "CHECK": true, "FOREIGN": true,
}

// parseRunDBPy extracts the schema from run_db.py's source text. It fails
// rather than under-reporting: a CREATE TABLE or a migration entry it cannot
// read is an error, so a reshaped run_db.py breaks this test loudly instead of
// letting a column slip past the parity check.
func parseRunDBPy(src string) (pySchema, error) {
	s := pySchema{
		tables:     make(map[string][]string),
		migrations: make(map[string][]migrationColumn),
	}

	m := ddlAssign.FindStringSubmatch(src)
	if m == nil {
		return s, errors.New(`no _DDL = """...""" literal found`)
	}
	s.ddl = m[1]
	ddl := stripLineComments(s.ddl, "--")
	for _, loc := range createTableHead.FindAllStringSubmatchIndex(ddl, -1) {
		table := ddl[loc[2]:loc[3]]
		open := loc[1] - 1
		closeAt, err := matchingClose(ddl, open)
		if err != nil {
			return s, fmt.Errorf("CREATE TABLE %s: %w", table, err)
		}
		cols := tableBodyColumns(ddl[open+1 : closeAt])
		if len(cols) == 0 {
			return s, fmt.Errorf("CREATE TABLE %s: no columns parsed", table)
		}
		if _, dup := s.tables[table]; dup {
			return s, fmt.Errorf("CREATE TABLE %s appears twice in _DDL", table)
		}
		s.tables[table] = cols
	}
	if n := len(createTableAny.FindAllStringIndex(ddl, -1)); n != len(s.tables) {
		return s, fmt.Errorf("_DDL has %d CREATE TABLE statement(s) but %d parsed", n, len(s.tables))
	}
	if len(s.tables) == 0 {
		return s, errors.New("_DDL has no CREATE TABLE statements")
	}

	loc := migrationsAssign.FindStringIndex(src)
	if loc == nil {
		return s, errors.New("no _COLUMN_MIGRATIONS = {...} assignment found")
	}
	rest := stripLineComments(src[loc[1]-1:], "#")
	closeAt, err := matchingClose(rest, 0)
	if err != nil {
		return s, fmt.Errorf("_COLUMN_MIGRATIONS: %w", err)
	}
	block := rest[1:closeAt]
	current, tuples := "", 0
	for _, sm := range migrationToken.FindAllStringSubmatch(block, -1) {
		if sm[1] != "" {
			current = sm[1]
			continue
		}
		if current == "" {
			return s, fmt.Errorf("_COLUMN_MIGRATIONS entry (%q, %q) precedes any table key", sm[2], sm[3])
		}
		s.migrations[current] = append(s.migrations[current], migrationColumn{name: sm[2], decl: sm[3]})
		tuples++
	}
	if entries := strings.Count(pyStringLiteral.ReplaceAllString(block, ""), "("); entries != tuples {
		return s, fmt.Errorf(`_COLUMN_MIGRATIONS has %d parenthesized entries but %d parsed as ("column", "type") tuples`, entries, tuples)
	}
	if tuples == 0 {
		return s, errors.New("_COLUMN_MIGRATIONS has no entries")
	}
	return s, nil
}

// tableBodyColumns names the columns in a CREATE TABLE body, skipping
// table-constraint clauses such as UNIQUE(...) and PRIMARY KEY (...).
func tableBodyColumns(body string) []string {
	var cols []string
	for _, item := range splitTopLevel(body) {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		word := item
		if i := strings.IndexAny(item, " \t\r\n("); i >= 0 {
			word = item[:i]
		}
		if tableConstraintWords[strings.ToUpper(word)] {
			continue
		}
		cols = append(cols, strings.Trim(word, "\"`[]"))
	}
	return cols
}

// splitTopLevel splits s at commas outside parentheses and quotes.
func splitTopLevel(s string) []string {
	var parts []string
	depth, start := 0, 0
	var quote byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case quote != 0:
			if c == quote {
				quote = 0
			}
		case c == '\'' || c == '"':
			quote = c
		case c == '(':
			depth++
		case c == ')':
			depth--
		case c == ',' && depth == 0:
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	return append(parts, s[start:])
}

// matchingClose returns the index of the bracket closing the one at s[open],
// ignoring brackets inside quoted strings.
func matchingClose(s string, open int) (int, error) {
	closer := map[byte]byte{'(': ')', '[': ']', '{': '}'}[s[open]]
	if closer == 0 {
		return 0, fmt.Errorf("%q at offset %d is not an opening bracket", s[open], open)
	}
	depth := 0
	var quote byte
	for i := open; i < len(s); i++ {
		c := s[i]
		switch {
		case quote != 0:
			if c == '\\' {
				i++
			} else if c == quote {
				quote = 0
			}
		case c == '\'' || c == '"':
			quote = c
		case c == s[open]:
			depth++
		case c == closer:
			depth--
			if depth == 0 {
				return i, nil
			}
		}
	}
	return 0, fmt.Errorf("unclosed %q at offset %d", s[open], open)
}

// stripLineComments drops everything from marker ("--" for SQL, "#" for
// Python) to the end of each line, unless the marker sits inside a quoted
// string. Line breaks are kept so the text keeps its shape.
func stripLineComments(s, marker string) string {
	var b strings.Builder
	for _, line := range strings.SplitAfter(s, "\n") {
		body := strings.TrimSuffix(line, "\n")
		var quote byte
		for i := 0; i < len(body); i++ {
			c := body[i]
			if quote != 0 {
				if c == '\\' {
					i++
				} else if c == quote {
					quote = 0
				}
				continue
			}
			if c == '\'' || c == '"' {
				quote = c
				continue
			}
			if strings.HasPrefix(body[i:], marker) {
				body = body[:i]
				break
			}
		}
		b.WriteString(body)
		if strings.HasSuffix(line, "\n") {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// schemaGaps lists, as sorted "table.column" strings, every column in have
// that allow does not permit.
func schemaGaps(have map[string]map[string]bool, allow map[string]map[string]bool) []string {
	var gaps []string
	for table, cols := range have {
		for col := range cols {
			if !allow[table][col] {
				gaps = append(gaps, table+"."+col)
			}
		}
	}
	sort.Strings(gaps)
	return gaps
}

// readRunDBPy returns run_db.py's source, skipping the test when the file is
// absent (a runnerd-only checkout) and failing on any other read error.
func readRunDBPy(t *testing.T, path string) string {
	t.Helper()
	body, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		abs, _ := filepath.Abs(path)
		t.Skipf("skipping the run_db.py schema check: %s is absent, so this is not a full monorepo checkout", abs)
	}
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(body)
}

// loadRunDBPySchema parses the real run_db.py and checks the parse found the
// tables every journal is known to carry, so a parser that silently matches
// nothing cannot pass the checks built on it.
func loadRunDBPySchema(t *testing.T) pySchema {
	t.Helper()
	s, err := parseRunDBPy(readRunDBPy(t, runDBPyPath))
	if err != nil {
		t.Fatalf("parsing %s: %v", runDBPyPath, err)
	}
	cols := s.columns()
	for _, want := range []string{"runs.name", "runs.status", "runs.executed_on", "eval_results.baseline"} {
		table, col, _ := strings.Cut(want, ".")
		if !cols[table][col] {
			t.Fatalf("parsing %s found no %s: the parser is not reading the schema", runDBPyPath, want)
		}
	}
	if len(s.migrations["eval_results"]) == 0 {
		t.Fatalf("parsing %s found no eval_results migrations: the parser is not reading _COLUMN_MIGRATIONS", runDBPyPath)
	}
	return s
}

// TestRunDBSchemaParity is cambia-2358 AC2: every column run_db.py can create,
// through _DDL or _COLUMN_MIGRATIONS, is one runDBSchema allows, so the next
// schema change breaks this build rather than every node's journal. It also
// checks every allowed table but runs carries the row-count cap.
func TestRunDBSchemaParity(t *testing.T) {
	s := loadRunDBPySchema(t)
	if gaps := schemaGaps(s.columns(), runDBSchema); len(gaps) > 0 {
		t.Fatalf("cfr/src/run_db.py creates %d column(s) that rundbcheck.go runDBSchema does not allow, "+
			"so every journal carrying them is rejected as rundb_invalid: %s",
			len(gaps), strings.Join(gaps, ", "))
	}

	var want []string
	for table := range runDBSchema {
		if table != "runs" {
			want = append(want, table)
		}
	}
	got := append([]string(nil), runDBChildTables...)
	sort.Strings(want)
	sort.Strings(got)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("runDBChildTables = %v, want every runDBSchema table but runs (%v) so each is row-capped", got, want)
	}
}

// TestRunDBSchemaParityCatchesAPlantedColumn is the other half of AC2: the
// parity check is not vacuous. A column planted into the real run_db.py text
// in each place a schema change can land (a _DDL column, a
// _COLUMN_MIGRATIONS tuple, a whole new _DDL table) shows up as a gap, and
// nothing else changes.
func TestRunDBSchemaParityCatchesAPlantedColumn(t *testing.T) {
	src := readRunDBPy(t, runDBPyPath)
	base, err := parseRunDBPy(src)
	if err != nil {
		t.Fatalf("parsing %s: %v", runDBPyPath, err)
	}

	planted := src
	insertAt := func(what string, at int, text string) {
		t.Helper()
		if at < 0 {
			t.Fatalf("cannot plant %s: its anchor is not in %s", what, runDBPyPath)
		}
		planted = planted[:at] + text + planted[at:]
	}
	ddl := ddlAssign.FindStringSubmatchIndex(planted)
	if ddl == nil {
		t.Fatalf("no _DDL literal in %s", runDBPyPath)
	}
	insertAt("a table", ddl[2], "\nCREATE TABLE IF NOT EXISTS planted_table (\n    id INTEGER PRIMARY KEY,\n    payload TEXT\n);\n")

	runsAt := -1
	for _, loc := range createTableHead.FindAllStringSubmatchIndex(planted, -1) {
		if planted[loc[2]:loc[3]] == "runs" {
			runsAt = loc[1]
			break
		}
	}
	insertAt("a _DDL column", runsAt, "\n    planted_ddl_column TEXT,")

	evalAt := -1
	if mig := migrationsAssign.FindStringIndex(planted); mig != nil {
		if loc := evalMigrationKey.FindStringIndex(planted[mig[1]:]); loc != nil {
			evalAt = mig[1] + loc[1]
		}
	}
	insertAt("a migration tuple", evalAt, "\n        (\"planted_migration_column\", \"INTEGER\"),")

	s, err := parseRunDBPy(planted)
	if err != nil {
		t.Fatalf("parsing the planted run_db.py: %v", err)
	}

	baseGaps := make(map[string]bool)
	for _, g := range schemaGaps(base.columns(), runDBSchema) {
		baseGaps[g] = true
	}
	var added []string
	for _, g := range schemaGaps(s.columns(), runDBSchema) {
		if !baseGaps[g] {
			added = append(added, g)
		}
	}
	want := []string{
		"eval_results.planted_migration_column",
		"planted_table.id",
		"planted_table.payload",
		"runs.planted_ddl_column",
	}
	if strings.Join(added, ",") != strings.Join(want, ",") {
		t.Fatalf("planted columns produced gaps %v, want exactly %v", added, want)
	}
}

// tableColumns reads a table's column names; an absent table reads as empty.
func tableColumns(t *testing.T, db *sql.DB, table string) map[string]bool {
	t.Helper()
	rows, err := db.Query(fmt.Sprintf("PRAGMA table_info(%s)", quoteIdent(table)))
	if err != nil {
		t.Fatalf("PRAGMA table_info(%s): %v", table, err)
	}
	defer rows.Close()
	cols := make(map[string]bool)
	for rows.Next() {
		var cid, notNull, pk int
		var name, typ string
		var dflt interface{}
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk); err != nil {
			t.Fatalf("scanning table_info(%s): %v", table, err)
		}
		cols[name] = true
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterating table_info(%s): %v", table, err)
	}
	return cols
}

// buildJournal writes a sealed run_db.sqlite the way a node's run does:
// run_db.get_db (WAL, the _DDL script, then _migrate_schema's guarded ALTERs),
// one runs row, an eval row filling the migrated columns, and a TRUNCATE
// checkpoint so the whole journal is in the one file that crosses the wire.
func buildJournal(t *testing.T, s pySchema, path, name string) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("opening %s: %v", path, err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	exec := func(query string, args ...interface{}) sql.Result {
		t.Helper()
		res, err := db.Exec(query, args...)
		if err != nil {
			t.Fatalf("%s: %v", strings.SplitN(strings.TrimSpace(query), "\n", 2)[0], err)
		}
		return res
	}

	exec("PRAGMA journal_mode=WAL")
	exec(s.ddl)
	for table, cols := range s.migrations {
		existing := tableColumns(t, db, table)
		if len(existing) == 0 {
			continue
		}
		for _, col := range cols {
			if !existing[col.name] {
				exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", quoteIdent(table), quoteIdent(col.name), col.decl))
			}
		}
	}

	const now = "2026-09-22T00:00:00Z"
	res := exec(`INSERT INTO runs (name, algorithm, status, engine_commit_hash, executed_on, created_at, updated_at)
		VALUES (?, 'prt-cfr', 'completed', '8dcbb41', 'n-9c1f2a7b0d44', ?, ?)`, name, now, now)
	runID, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("reading the runs row id: %v", err)
	}
	exec(`INSERT INTO eval_results (run_id, iteration, baseline, win_rate, games_played,
			policy_errors, engine_errors, belief_protocol, run_seed, served_policy, timestamp)
		VALUES (?, 20, 'imperfect_greedy', 0.61, 5000, 0, 0, 'advancing', '18446744073709551557', 'average_strategy', ?)`,
		runID, now)
	exec("PRAGMA wal_checkpoint(TRUNCATE)")
}

// TestCurrentRunDBSchemaJournalValidates is cambia-2358 AC1: a journal built
// in the test from the current run_db.py schema, including runs.executed_on
// and the migrated eval_results columns, is accepted rather than rejected as
// rundb_invalid.
func TestCurrentRunDBSchemaJournalValidates(t *testing.T) {
	s := loadRunDBPySchema(t)
	path := filepath.Join(t.TempDir(), RunDBPath)
	buildJournal(t, s, path, "job-train-0001")

	v, err := Validate(path, "job-train-0001")
	if err != nil {
		t.Fatalf("Validate: unexpected error: %v", err)
	}
	if !v.Accepted {
		t.Fatalf("a journal on the current run_db.py schema was rejected: Reason=%q Detail=%q", v.Reason, v.Detail)
	}
}

// TestCorpusMatchesRunDBSchema keeps the shared corpus current: every fixture
// the generator builds through get_db carries exactly the columns run_db.py
// creates today. A stale corpus is how the missing executed_on went unseen
// (cambia-2358), since fixtures older than a column cannot exercise it.
func TestCorpusMatchesRunDBSchema(t *testing.T) {
	want := loadRunDBPySchema(t).columns()
	manifest := loadManifest(t)
	checked := 0
	for _, fx := range manifest.Fixtures {
		fx := fx
		// corrupt.sqlite is not a readable database by design.
		if fx.File == nil || fx.Reason == ReasonIntegrityCheck {
			continue
		}
		checked++
		t.Run(*fx.File, func(t *testing.T) {
			path := filepath.Join(testdataDir, *fx.File)
			db, err := sql.Open("sqlite", "file:"+path+"?mode=ro&immutable=1")
			if err != nil {
				t.Fatalf("opening %s: %v", path, err)
			}
			defer db.Close()
			for table, cols := range want {
				got := tableColumns(t, db, table)
				var missing, extra []string
				for col := range cols {
					if !got[col] {
						missing = append(missing, col)
					}
				}
				for col := range got {
					if !cols[col] {
						extra = append(extra, col)
					}
				}
				if len(missing)+len(extra) > 0 {
					sort.Strings(missing)
					sort.Strings(extra)
					t.Errorf("%s table %s is stale against cfr/src/run_db.py (missing %v, extra %v); regenerate the corpus: %s",
						*fx.File, table, missing, extra, genCorpusCommand)
				}
			}
		})
	}
	if checked == 0 {
		t.Fatal("no readable fixtures were checked: the corpus loader is checking nothing")
	}
}
