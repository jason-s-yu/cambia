// internal/handlers/guest_username_test.go
//
// Guards newGuestUsername's output shape (cambia-942 F2). cambia-890 replaced the fixed string
// "Guest" every ephemeral user carried with an id-derived label, but nothing tested the deriving
// function itself: the whole suite stayed green with it reverted to a constant, because the
// end-to-end tests that cover guest naming (username_population_test.go) only assert that two
// guests differ, which any per-user value satisfies. These cases pin the format the web client
// and the ticket's collision analysis both assume.
package handlers

import (
	"strings"
	"testing"

	"github.com/google/uuid"
)

func TestNewGuestUsername(t *testing.T) {
	cases := []struct {
		name string
		id   string
		want string
	}{
		{
			name: "lowercase hex is upper-cased",
			id:   "a1b2c3d4-1111-2222-3333-444455556666",
			want: "Guest-A1B2C3D4",
		},
		{
			name: "digits only",
			id:   "01234567-89ab-cdef-8000-000000000000",
			want: "Guest-01234567",
		},
		{
			name: "nil uuid",
			id:   "00000000-0000-0000-0000-000000000000",
			want: "Guest-00000000",
		},
		{
			name: "all-f prefix",
			id:   "ffffffff-ffff-4fff-bfff-ffffffffffff",
			want: "Guest-FFFFFFFF",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := newGuestUsername(uuid.MustParse(tc.id))
			if got != tc.want {
				t.Fatalf("newGuestUsername(%s) = %q, want %q", tc.id, got, tc.want)
			}
			if len(got) != 14 {
				t.Fatalf("newGuestUsername(%s) = %q: length %d, want 14 (\"Guest-\" + 8 hex chars)", tc.id, got, len(got))
			}
			if !strings.HasPrefix(got, "Guest-") {
				t.Fatalf("newGuestUsername(%s) = %q, want the %q prefix", tc.id, got, "Guest-")
			}
		})
	}
}

// TestNewGuestUsernameIsPerUser is the direct cambia-890 regression: distinct ids must produce
// distinct labels, so a table of guests no longer renders the same name in every seat. It also
// pins the flip side the ticket's collision analysis rests on - the label is exactly the first
// 32 bits of the id, so two ids sharing that prefix are expected to collide.
func TestNewGuestUsernameIsPerUser(t *testing.T) {
	seen := make(map[string]string, 64)
	for i := 0; i < 64; i++ {
		id := uuid.New()
		name := newGuestUsername(id)
		if prev, dup := seen[name]; dup {
			t.Fatalf("guest label %q derived from both %s and %s", name, prev, id)
		}
		seen[name] = id.String()
	}

	sharedPrefixA := uuid.MustParse("deadbeef-0000-4000-8000-000000000001")
	sharedPrefixB := uuid.MustParse("deadbeef-ffff-4fff-bfff-ffffffffffff")
	if a, b := newGuestUsername(sharedPrefixA), newGuestUsername(sharedPrefixB); a != b {
		t.Fatalf("labels for ids sharing their first 32 bits diverged: %q vs %q; the label is documented as hex[:8] of the id", a, b)
	}
}
