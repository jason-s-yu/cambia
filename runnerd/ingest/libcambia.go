package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// libcambiaResult carries the resolved shared library and its provenance.
type libcambiaResult struct {
	path          string // libcambia/<engine-tree-sha>.so
	engineTreeSha string // git rev-parse <sha>:engine
	sha256        string // sha256 of the built .so
}

// ensureLibcambia resolves the libcambia shared library for the pinned commit,
// building it on a cache miss and reusing it on a hit (design 3.5). The cache key
// is the engine/ subtree sha (git rev-parse <sha>:engine), so every commit that
// leaves the engine tree unchanged shares one artifact. The build runs in the
// worktree with GOTOOLCHAIN pinned so a newer host Go cannot silently change the
// compiler. The artifact is consumed via LIBCAMBIA_PATH; worktrees stay clean.
func (m *Manager) ensureLibcambia(ctx context.Context, commit, worktreeDir string) (libcambiaResult, error) {
	engineTreeSha, err := m.git(ctx, "rev-parse", commit+":engine")
	if err != nil {
		return libcambiaResult{}, fmt.Errorf("resolve engine tree sha: %w", err)
	}
	soPath := filepath.Join(m.libcambiaDir, engineTreeSha+".so")

	if _, err := os.Stat(soPath); err == nil {
		touch(soPath, m.now())
		sum, err := sha256File(soPath)
		if err != nil {
			return libcambiaResult{}, err
		}
		return libcambiaResult{path: soPath, engineTreeSha: engineTreeSha, sha256: sum}, nil
	}

	if err := os.MkdirAll(m.libcambiaDir, 0o755); err != nil {
		return libcambiaResult{}, err
	}
	if err := m.buildLibcambia(ctx, worktreeDir, soPath); err != nil {
		return libcambiaResult{}, err
	}
	touch(soPath, m.now())
	sum, err := sha256File(soPath)
	if err != nil {
		return libcambiaResult{}, err
	}
	return libcambiaResult{path: soPath, engineTreeSha: engineTreeSha, sha256: sum}, nil
}

// buildLibcambia compiles the c-shared engine library directly into the cache
// path from the worktree, with GOTOOLCHAIN env-pinned (design 3.5). Building to
// the cache path (not into the worktree) keeps the checkout clean.
func (m *Manager) buildLibcambia(ctx context.Context, worktreeDir, soPath string) error {
	res, err := m.runner.Run(ctx, Command{
		Name: "go",
		Args: []string{"build", "-buildmode=c-shared", "-o", soPath, "./engine/cgo/"},
		Dir:  worktreeDir,
		Env:  []string{"GOTOOLCHAIN=" + goToolchainPin, "CGO_ENABLED=1"},
	})
	if err != nil {
		return fmt.Errorf("go build c-shared: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}
	if _, statErr := os.Stat(soPath); statErr != nil {
		return fmt.Errorf("go build produced no artifact at %s", soPath)
	}
	return nil
}
