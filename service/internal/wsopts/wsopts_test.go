package wsopts

import (
	"reflect"
	"testing"
)

func TestOriginPatternsUnsetKeepsWildcard(t *testing.T) {
	t.Setenv(OriginEnvVar, "")

	if got := OriginPatterns(); !reflect.DeepEqual(got, []string{"*"}) {
		t.Fatalf("expected the dev wildcard when %s is unset, got %v", OriginEnvVar, got)
	}
}

func TestOriginPatternsBlankValueKeepsWildcard(t *testing.T) {
	t.Setenv(OriginEnvVar, "   ")

	if got := OriginPatterns(); !reflect.DeepEqual(got, []string{"*"}) {
		t.Fatalf("expected the dev wildcard for a whitespace-only value, got %v", got)
	}
}

// coder/websocket matches patterns against the Origin *host*, so a
// scheme-qualified entry has to be normalized or it can never match.
func TestOriginPatternsNormalizesEntries(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{
			name: "strips scheme",
			raw:  "https://cambia.jasonyu.io",
			want: []string{"cambia.jasonyu.io"},
		},
		{
			name: "keeps bare host",
			raw:  "cambia.jasonyu.io",
			want: []string{"cambia.jasonyu.io"},
		},
		{
			name: "keeps port",
			raw:  "http://localhost:5173",
			want: []string{"localhost:5173"},
		},
		{
			name: "strips trailing path",
			raw:  "https://cambia.jasonyu.io/",
			want: []string{"cambia.jasonyu.io"},
		},
		{
			name: "multiple entries with spaces",
			raw:  "https://cambia.jasonyu.io, http://localhost:5173 ",
			want: []string{"cambia.jasonyu.io", "localhost:5173"},
		},
		{
			name: "wildcard pattern passes through",
			raw:  "*.jasonyu.io",
			want: []string{"*.jasonyu.io"},
		},
		{
			name: "empty entries dropped",
			raw:  "https://cambia.jasonyu.io,,",
			want: []string{"cambia.jasonyu.io"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(OriginEnvVar, tc.raw)
			if got := OriginPatterns(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("OriginPatterns(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}

// A set-but-unparseable allowlist fails closed: an empty pattern slice allows
// same-origin upgrades only. Falling back to the wildcard here would turn a
// config typo into an open cross-origin surface.
func TestOriginPatternsAllEntriesEmptyFailsClosed(t *testing.T) {
	t.Setenv(OriginEnvVar, " , , ")

	if got := OriginPatterns(); len(got) != 0 {
		t.Fatalf("expected an empty (same-origin-only) allowlist, got %v", got)
	}
}

func TestAcceptOptionsCarriesSubprotocolsAndPatterns(t *testing.T) {
	t.Setenv(OriginEnvVar, "https://cambia.jasonyu.io")

	opts := AcceptOptions("cambia")
	if !reflect.DeepEqual(opts.Subprotocols, []string{"cambia"}) {
		t.Fatalf("expected the cambia subprotocol, got %v", opts.Subprotocols)
	}
	if !reflect.DeepEqual(opts.OriginPatterns, []string{"cambia.jasonyu.io"}) {
		t.Fatalf("unexpected origin patterns: %v", opts.OriginPatterns)
	}

	if sub := AcceptOptions().Subprotocols; len(sub) != 0 {
		t.Fatalf("expected no subprotocols when none are offered, got %v", sub)
	}
}
