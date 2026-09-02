package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jason-s-yu/cambia/engine/cgo/abiver"
)

// libcambiaResult carries the resolved shared library and its provenance.
type libcambiaResult struct {
	path          string // libcambia/<cacheKey>.so
	cacheKey      string // engine-tree-sha, salted with the cgo ABI generation
	engineTreeSha string // git rev-parse <sha>:engine
	sha256        string // sha256 of the built .so
}

// ensureLibcambia resolves the libcambia shared library for the pinned commit,
// building it on a cache miss and reusing it on a hit (design 3.5). The cache key
// is the engine/ subtree sha (git rev-parse <sha>:engine) salted with this
// runnerd build's abiver.Generation (cambia-1689), so every commit that leaves
// the engine tree unchanged shares one artifact, and a cache populated under one
// harness build's understanding of the cgo export ABI is never handed to a
// worktree running under a different one. The engine subtree hash alone already
// changes whenever an exported signature changes (abiver.Generation lives inside
// engine/), so this is a second, human-legible tag on the artifact rather than
// the sole guard: it also isolates the cache across a runnerd upgrade that
// changes what this daemon itself expects of the .so it builds, independent of
// which commit is being ingested. The build runs in the worktree with
// GOTOOLCHAIN pinned so a newer host Go cannot silently change the compiler. The
// artifact is consumed via LIBCAMBIA_PATH; worktrees stay clean.
func (m *Manager) ensureLibcambia(ctx context.Context, commit, worktreeDir string) (libcambiaResult, error) {
	engineTreeSha, err := m.git(ctx, "rev-parse", commit+":engine")
	if err != nil {
		return libcambiaResult{}, fmt.Errorf("resolve engine tree sha: %w", err)
	}
	cacheKey := engineTreeSha + "-abigen" + strconv.Itoa(abiver.Generation)
	soPath := filepath.Join(m.libcambiaDir, cacheKey+".so")

	if _, err := os.Stat(soPath); err == nil {
		touch(soPath, m.now())
		sum, err := sha256File(soPath)
		if err != nil {
			return libcambiaResult{}, err
		}
		return libcambiaResult{path: soPath, cacheKey: cacheKey, engineTreeSha: engineTreeSha, sha256: sum}, nil
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
	return libcambiaResult{path: soPath, cacheKey: cacheKey, engineTreeSha: engineTreeSha, sha256: sum}, nil
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
