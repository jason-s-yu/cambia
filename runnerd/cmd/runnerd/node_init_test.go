package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
)

// TestNodeInitConfigLoadsThroughTheGoLoader pins design D25's "config
// template that validates against the Go loader" (cambia-1725 AC3): the
// exact nodeagent.File buildNodeConfigFile assembles, which is what
// `cambia-runnerd node init` writes to disk, must load cleanly through
// nodeagent.LoadConfig -- the same function --role node calls at startup.
func TestNodeInitConfigLoadsThroughTheGoLoader(t *testing.T) {
	dir := t.TempDir()
	keyPath := filepath.Join(dir, "nashnet-node.key")
	fingerprint := strings.Repeat("ab", 32)

	file := buildNodeConfigFile(
		"n-0123456789ab",
		"https://coordinator.example:8090",
		fingerprint,
		keyPath,
		dir,
		"",
		4,
		[]string{"gpu", "fast"},
	)

	body, err := marshalNodeConfigFile(file)
	if err != nil {
		t.Fatalf("marshalNodeConfigFile: %v", err)
	}
	out := append([]byte(nodeInitBanner), body...)

	configPath := filepath.Join(dir, "nashnet-node.yaml")
	if err := os.WriteFile(configPath, out, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := nodeagent.LoadConfig(configPath)
	if err != nil {
		t.Fatalf("nodeagent.LoadConfig(%s) = %v, want nil (the template must validate against the Go loader)", configPath, err)
	}
	if cfg.NodeID != "n-0123456789ab" {
		t.Errorf("NodeID = %q, want n-0123456789ab", cfg.NodeID)
	}
	if cfg.Coordinator.URL != "https://coordinator.example:8090" {
		t.Errorf("Coordinator.URL = %q", cfg.Coordinator.URL)
	}
	if cfg.Coordinator.CertSHA256 != fingerprint {
		t.Errorf("Coordinator.CertSHA256 = %q, want %q", cfg.Coordinator.CertSHA256, fingerprint)
	}
	if cfg.KeyPath != keyPath {
		t.Errorf("KeyPath = %q, want %q", cfg.KeyPath, keyPath)
	}
	if cfg.BaseDir != dir {
		t.Errorf("BaseDir = %q, want %q", cfg.BaseDir, dir)
	}
	if cfg.RunsDir != filepath.Join(dir, "runs") {
		t.Errorf("RunsDir = %q, want the base_dir default", cfg.RunsDir)
	}
	if cfg.Slots != 4 {
		t.Errorf("Slots = %d, want 4", cfg.Slots)
	}
	if len(cfg.Labels) != 2 || cfg.Labels[0] != "gpu" || cfg.Labels[1] != "fast" {
		t.Errorf("Labels = %v, want [gpu fast]", cfg.Labels)
	}
}

// TestNodeInitConfigRejectsAMalformedFingerprint is the negative twin: a
// malformed fingerprint must fail nodeagent.LoadConfig rather than silently
// pass through, which is exactly what runNodeInit's post-write self-check
// (node_init.go) guards against before an operator ever sees the file.
func TestNodeInitConfigRejectsAMalformedFingerprint(t *testing.T) {
	dir := t.TempDir()
	file := buildNodeConfigFile("n-0123456789ab", "https://coordinator.example:8090", "not-hex",
		filepath.Join(dir, "key"), dir, "", 1, nil)
	body, err := marshalNodeConfigFile(file)
	if err != nil {
		t.Fatalf("marshalNodeConfigFile: %v", err)
	}
	configPath := filepath.Join(dir, "nashnet-node.yaml")
	if err := os.WriteFile(configPath, body, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, err := nodeagent.LoadConfig(configPath); err == nil {
		t.Fatal("LoadConfig accepted a malformed fingerprint, want an error")
	}
}
