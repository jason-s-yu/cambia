package capability

import (
	"errors"
	"testing"
)

var errNotFound = errors.New("executable file not found in $PATH")

// TestProbeCanBuildLibcambia covers AC(7): a fixture node with the pinned Go
// toolchain present and no C compiler declares can_build_libcambia: false.
func TestProbeCanBuildLibcambia(t *testing.T) {
	cases := []struct {
		name string
		run  RunFunc
		want bool
	}{
		{
			name: "go toolchain resolves, cc present and runnable",
			run: func(name string, args []string, env []string) ([]byte, error) {
				switch {
				case name == "go" && args[0] == "version":
					return []byte("go version go1.26.0 linux/amd64\n"), nil
				case name == "go" && args[0] == "env":
					return []byte("cc\n"), nil
				case name == "cc" && len(args) > 0 && args[0] == "--version":
					return []byte("cc (Debian 12.2.0)\n"), nil
				}
				return nil, errNotFound
			},
			want: true,
		},
		{
			name: "go toolchain present, no C compiler (AC7 fixture)",
			run: func(name string, args []string, env []string) ([]byte, error) {
				switch {
				case name == "go" && args[0] == "version":
					return []byte("go version go1.26.0 linux/amd64\n"), nil
				case name == "go" && args[0] == "env":
					return []byte("cc\n"), nil
				case name == "cc":
					return nil, errNotFound
				}
				return nil, errNotFound
			},
			want: false,
		},
		{
			name: "pinned go toolchain does not resolve",
			run: func(name string, args []string, env []string) ([]byte, error) {
				if name == "go" && args[0] == "version" {
					return nil, errNotFound
				}
				return []byte("cc\n"), nil
			},
			want: false,
		},
		{
			name: "go env CC resolves to an empty compiler",
			run: func(name string, args []string, env []string) ([]byte, error) {
				switch {
				case name == "go" && args[0] == "version":
					return []byte("go version go1.26.0 linux/amd64\n"), nil
				case name == "go" && args[0] == "env":
					return []byte("\n"), nil
				}
				return nil, errNotFound
			},
			want: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := ProbeCanBuildLibcambia("go1.26.0", c.run); got != c.want {
				t.Errorf("ProbeCanBuildLibcambia = %v, want %v", got, c.want)
			}
		})
	}
}
