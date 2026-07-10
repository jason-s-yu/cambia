"""
src/harness/spec.py

Job-spec parsing and client-side validation for the `cambia harness` CLI
(cambia-256, design 2.6). The runner runs the authoritative preflight; this
first pass fails fast on the client so a bad spec never reaches the control
plane or consumes a mirror push.

Name validation mirrors the Go ValidateName the runner enforces
(runnerd/procmgr/process.go): allowlist `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`
plus an explicit `..` / `/` reject. Path guards on `config`/`checkpoint_*`
mirror design 5.4: reject absolute paths and any `..` segment before the runner
resolves them inside the job worktree.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Kind allowlist (design 1 / 2.6). The runner further restricts train to the
# `prtcfr` algorithm in v1; that is a runner-side preflight, not a client check.
ALLOWED_KINDS = ("train", "evaluate", "head-to-head", "bench")

# Device allowlist (cambia-329). The client stays permissive within this set;
# the runner enforces its own RUNNERD_ALLOWED_DEVICES capability list against
# whatever hosts are actually deployed, which is a runner-side preflight, not
# a client check.
ALLOWED_DEVICES = ("cpu", "cuda", "xpu")

# Mirror of the Go runNameRe: leading alphanumeric, then up to 127 more chars
# from [A-Za-z0-9._-]; 128 max. Combined with the explicit ".."/"/" reject it is
# the path-traversal guard the run name gets before any directory join.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class HarnessSpecError(ValueError):
    """A job spec failed client-side validation."""


def validate_name(name: Any) -> str:
    """Validate a run/job name against the runner's ValidateName rules.

    Raises HarnessSpecError on any name the runner would reject, so a submit
    fails on the client instead of round-tripping a 400.
    """
    if not isinstance(name, str) or not name:
        raise HarnessSpecError("name is empty or not a string")
    if ".." in name or "/" in name:
        raise HarnessSpecError(f"name {name!r} contains a path separator or '..'")
    if not _NAME_RE.match(name):
        raise HarnessSpecError(
            f"name {name!r} must match {_NAME_RE.pattern} "
            "(leading alphanumeric, then A-Za-z0-9._-, max 128 chars)"
        )
    return name


def guard_relpath(value: Any, field_name: str) -> str:
    """Reject absolute paths and any '..' segment on a spec path field.

    The runner re-guards and containment-checks inside the worktree (design
    5.4); this is the client-side fail-fast so an obviously unsafe path never
    reaches the control plane.
    """
    if not isinstance(value, str) or not value:
        raise HarnessSpecError(f"{field_name} is empty or not a string")
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise HarnessSpecError(
            f"{field_name} must be repo-relative, not absolute: {value!r}"
        )
    segments = re.split(r"[\\/]", value)
    if any(seg == ".." for seg in segments):
        raise HarnessSpecError(f"{field_name} must not contain a '..' segment: {value!r}")
    return value


@dataclass
class JobSpec:
    """A parsed + validated harness job spec (design 2.6).

    `commit` is optional in the source yaml: `harness submit` resolves it from
    the clean worktree HEAD and fills it in. If the yaml pins a commit it must
    match the resolved HEAD, so run_db's recorded commit equals what ran.
    """

    kind: str
    name: str
    config: Optional[str] = None
    commit: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    resume: bool = False
    device: str = "cpu"
    checkpoint_a: Optional[str] = None
    checkpoint_b: Optional[str] = None
    target: Optional[str] = None
    games: Optional[int] = None
    priority: str = "normal"
    force: bool = False

    _KNOWN_KEYS = frozenset(
        {
            "kind",
            "name",
            "config",
            "commit",
            "overrides",
            "resume",
            "device",
            "checkpoint_a",
            "checkpoint_b",
            "target",
            "games",
            "priority",
            "force",
        }
    )

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "JobSpec":
        """Parse and validate a raw spec mapping (design 2.6 validation order).

        Order: kind allowlist -> validateName -> path guards (config,
        checkpoints) -> field-shape checks. Unknown keys are rejected so a typo'd
        field is not silently dropped before the runner sees it.
        """
        if not isinstance(raw, dict):
            raise HarnessSpecError("spec must be a mapping")

        unknown = set(raw) - cls._KNOWN_KEYS
        if unknown:
            raise HarnessSpecError(f"unknown spec keys: {sorted(unknown)}")

        kind = raw.get("kind")
        if kind not in ALLOWED_KINDS:
            raise HarnessSpecError(
                f"kind {kind!r} not in allowlist {list(ALLOWED_KINDS)}"
            )

        name = validate_name(raw.get("name"))

        commit = raw.get("commit")
        if commit is not None:
            if not isinstance(commit, str) or not _COMMIT_RE.match(commit):
                raise HarnessSpecError(f"commit must be a 40-hex sha: {commit!r}")

        config = raw.get("config")
        if config is not None:
            guard_relpath(config, "config")

        checkpoint_a = raw.get("checkpoint_a")
        checkpoint_b = raw.get("checkpoint_b")
        if checkpoint_a is not None:
            guard_relpath(checkpoint_a, "checkpoint_a")
        if checkpoint_b is not None:
            guard_relpath(checkpoint_b, "checkpoint_b")

        target = raw.get("target")
        if target is not None:
            guard_relpath(target, "target")

        overrides = raw.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise HarnessSpecError("overrides must be a mapping of dotted keys")
        if any(not isinstance(k, str) or not k for k in overrides):
            raise HarnessSpecError("override keys must be non-empty strings")

        device = raw.get("device", "cpu")
        if device not in ALLOWED_DEVICES:
            raise HarnessSpecError(
                f"device must be one of {list(ALLOWED_DEVICES)}, got {device!r}"
            )

        priority = raw.get("priority", "normal")
        if not isinstance(priority, str) or not priority:
            raise HarnessSpecError("priority must be a non-empty string")

        games = raw.get("games")
        if games is not None:
            if not isinstance(games, int) or isinstance(games, bool) or games <= 0:
                raise HarnessSpecError(f"games must be a positive integer: {games!r}")

        resume = bool(raw.get("resume", False))
        if resume and kind != "train":
            raise HarnessSpecError("resume is valid for train jobs only")

        # Kind-specific requirements: config drives train/evaluate/bench;
        # head-to-head needs both checkpoints; evaluate needs a target (the
        # checkpoint/run-dir it evaluates), forbidden for train.
        # evaluate takes no config: run-dir mode reads the target's own
        # config.yaml, and the runner never forwards --config for that kind.
        if kind in ("train", "bench") and config is None:
            raise HarnessSpecError(f"kind {kind!r} requires a 'config' path")
        if kind == "head-to-head" and (checkpoint_a is None or checkpoint_b is None):
            raise HarnessSpecError("head-to-head requires checkpoint_a and checkpoint_b")
        if kind == "evaluate" and target is None:
            raise HarnessSpecError(
                "evaluate requires a 'target': a runner-local run directory"
            )
        if kind == "train" and target is not None:
            raise HarnessSpecError("target is not valid for kind='train'")

        return cls(
            kind=kind,
            name=name,
            config=config,
            commit=commit,
            overrides=dict(overrides),
            resume=resume,
            device=device,
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            target=target,
            games=games,
            priority=priority,
            force=bool(raw.get("force", False)),
        )

    def to_payload(self, commit: str) -> Dict[str, Any]:
        """Build the POST /harness/jobs body with the resolved commit stamped in.

        Only set fields are emitted so the runner applies its own defaults for
        anything the spec omitted.
        """
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "commit": commit,
            "device": self.device,
            "resume": self.resume,
            "priority": self.priority,
            "force": self.force,
        }
        if self.config is not None:
            payload["config"] = self.config
        if self.overrides:
            payload["overrides"] = self.overrides
        if self.checkpoint_a is not None:
            payload["checkpoint_a"] = self.checkpoint_a
        if self.checkpoint_b is not None:
            payload["checkpoint_b"] = self.checkpoint_b
        if self.target is not None:
            payload["target"] = self.target
        if self.games is not None:
            payload["games"] = self.games
        return payload


def parse_spec_file(path: str) -> JobSpec:
    """Load a spec YAML file and parse it into a validated JobSpec."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise HarnessSpecError(f"spec file is empty: {path}")
    return JobSpec.parse(raw)
