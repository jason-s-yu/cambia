// internal/lobby/lobby_json_test.go
//
// Serialization guards for the Lobby fields REST clients read (cambia-942 F2). The cambia-907 L3
// edit that dropped `omitempty` from GameID had no test at all, and a marshal assertion alone
// cannot supply one: encoding/json never treats a fixed-size array as empty, so re-adding the tag
// changes no output. The tag itself is therefore asserted directly, with the marshal cases below
// pinning the behaviour a client depends on (the key is always there) against the other way it
// could break - GameID changing to a pointer or string type, where omitempty would bite.
package lobby

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/uuid"
)

func TestLobbyGameIDJSONTagHasNoOmitempty(t *testing.T) {
	field, ok := reflect.TypeOf(Lobby{}).FieldByName("GameID")
	if !ok {
		t.Fatalf("lobby.Lobby has no GameID field")
	}
	if got, want := field.Tag.Get("json"), "gameId"; got != want {
		t.Fatalf("Lobby.GameID json tag = %q, want %q: uuid.UUID is a [16]byte array, which "+
			"encoding/json never treats as empty, so any option here is a no-op that misleads "+
			"the next reader (cambia-907 L3)", got, want)
	}
}

func TestLobbyMarshalAlwaysCarriesGameID(t *testing.T) {
	gameID := uuid.MustParse("aabb1965-6ef7-400e-8fff-2d16b430d137")
	// Marshalled through a pointer throughout: Lobby embeds a sync.Mutex, so a by-value copy
	// trips vet's copylocks check, and the handlers serialize *Lobby anyway.
	cases := []struct {
		name string
		lob  *Lobby
		want string
	}{
		{
			name: "no game attached yet",
			lob:  &Lobby{ID: uuid.New(), HostUserID: uuid.New(), Type: "public"},
			want: "00000000-0000-0000-0000-000000000000",
		},
		{
			name: "game in progress",
			lob:  &Lobby{ID: uuid.New(), HostUserID: uuid.New(), Type: "public", GameID: gameID, InGame: true},
			want: gameID.String(),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.lob)
			if err != nil {
				t.Fatalf("marshal lobby: %v", err)
			}
			var decoded map[string]json.RawMessage
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Fatalf("decode marshalled lobby: %v", err)
			}
			field, present := decoded["gameId"]
			if !present {
				t.Fatalf("gameId is missing from the serialized lobby: %s", raw)
			}
			var got string
			if err := json.Unmarshal(field, &got); err != nil {
				t.Fatalf("gameId is not a JSON string: %s", field)
			}
			if got != tc.want {
				t.Fatalf("gameId = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestLobbyMarshalOmitsQueueID pins the contrast documented next to GameID and in
// service/doc/rest_api.md: queueID is a plain string, so its omitempty tag does omit the key
// until matchmaking selects a queue.
func TestLobbyMarshalOmitsQueueID(t *testing.T) {
	raw, err := json.Marshal(&Lobby{ID: uuid.New(), Type: "public"})
	if err != nil {
		t.Fatalf("marshal lobby: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode marshalled lobby: %v", err)
	}
	if _, present := decoded["queueID"]; present {
		t.Fatalf("queueID should be omitted while empty: %s", raw)
	}
}
