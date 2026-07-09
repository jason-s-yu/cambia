package ingest

import (
	"bytes"
	"context"
	"os"
	"os/exec"
)

// Command is one external process invocation. Name is the executable (git, uv,
// python, go, uname, ...), Args its arguments, Dir the working directory (empty
// = inherit), and Env the extra KEY=VALUE entries appended to the inherited
// environment for this invocation only.
type Command struct {
	Name string
	Args []string
	Dir  string
	Env  []string
}

// Result is the captured output of a Command.
type Result struct {
	Stdout []byte
	Stderr []byte
}

// CommandRunner is the single seam through which the ingest package shells out.
// The production implementation (ExecRunner) wraps exec.CommandContext. Tests
// inject a fake that returns canned output for uv/python/go probes while
// delegating git to a real ExecRunner against temp repositories, so the git
// receipt/worktree logic runs against real git and only the toolchain builds are
// faked.
type CommandRunner interface {
	Run(ctx context.Context, c Command) (Result, error)
}

// ExecRunner runs commands with exec.CommandContext, capturing stdout/stderr.
type ExecRunner struct{}

// Run executes c and returns its captured output. When c.Env is non-empty it is
// appended to the current process environment for this invocation.
func (ExecRunner) Run(ctx context.Context, c Command) (Result, error) {
	cmd := exec.CommandContext(ctx, c.Name, c.Args...)
	cmd.Dir = c.Dir
	if len(c.Env) > 0 {
		cmd.Env = append(os.Environ(), c.Env...)
	}
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return Result{Stdout: out.Bytes(), Stderr: errb.Bytes()}, err
}
