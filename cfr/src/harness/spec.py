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
# measure (design D38, cambia-1072) runs a pinned script instead of a cambia
# subcommand.
ALLOWED_KINDS = ("train", "evaluate", "head-to-head", "bench", "measure")

# The worktree-relative root a measure job's script must resolve under (design
# D38), mirroring runnerd/harness/jobspec.go's measureScriptRoot.
MEASURE_SCRIPT_ROOT = "cfr/scripts/"

# Per-entry byte cap for a measure job's args (design D38: "each entry
# rejected on a NUL or over a length cap"), mirroring runnerd/harness/
# jobspec.go's maxMeasureArgLen so the client rejects exactly what the runner
# would.
MAX_MEASURE_ARG_LEN = 4096

# Device allowlist (cambia-329). The client stays permissive within this set;
# the runner enforces its own RUNNERD_ALLOWED_DEVICES capability list against
# whatever hosts are actually deployed, which is a runner-side preflight, not
# a client check.
ALLOWED_DEVICES = ("cpu", "cuda", "xpu")

# on_failure policies for an `after` dependency (cambia-352). They govern only
# the parent-failure branch; a parent success always runs the dependent. skip is
# the default. Allowed on every kind.
ON_FAILURE_POLICIES = ("skip", "run", "fail")

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
    warm_start: Optional[str] = None
    after: Optional[str] = None
    on_failure: str = "skip"
    # Optional Codebridge hub work-item handle (cambia-353), e.g. "cambia-359".
    # Telemetry-only: it links the job's reflected note to a hub item and is
    # persisted into jobspec.json/env.json so the link survives a run-dir pull.
    # An unlinked job (hub_item omitted) reflects onto the config collector item.
    hub_item: Optional[str] = None
    # Run-alone flag (cambia-655): a timing-sensitive job the runner launches only
    # into an idle daemon and holds every other job around. Default false (a normal
    # job that shares the concurrency pool). Allowed on every kind.
    exclusive: bool = False
    # script is a kind=measure job's worktree-relative pinned script path
    # (design D38), e.g. "cfr/scripts/measure_gate_gap.py". Required for
    # measure, forbidden for every other kind.
    script: Optional[str] = None
    # args is a kind=measure job's argv tail, appended verbatim after the
    # staged script path with no shell involved. Forbidden for every other
    # kind.
    args: List[str] = field(default_factory=list)
    # reads names other run directories a kind=measure job reads as read-only
    # seeds, resolved through pathguard against the runs dir on the runner.
    # Forbidden for every other kind.
    reads: List[str] = field(default_factory=list)

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
            "warm_start",
            "after",
            "on_failure",
            "hub_item",
            "exclusive",
            "script",
            "args",
            "reads",
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

        warm_start = raw.get("warm_start")
        if warm_start is not None:
            guard_relpath(warm_start, "warm_start")

        script = raw.get("script")
        if script is not None and kind != "measure":
            raise HarnessSpecError("script is valid for kind='measure' only")
        if kind == "measure":
            if script is None:
                raise HarnessSpecError("kind 'measure' requires a 'script' path")
            guard_relpath(script, "script")
            if (
                not script.startswith(MEASURE_SCRIPT_ROOT)
                or script == MEASURE_SCRIPT_ROOT
            ):
                raise HarnessSpecError(
                    f"script must resolve under {MEASURE_SCRIPT_ROOT!r}: {script!r}"
                )

        args_raw = raw.get("args")
        if args_raw is not None and kind != "measure":
            raise HarnessSpecError("args is valid for kind='measure' only")
        args: List[str] = []
        if args_raw is not None:
            if not isinstance(args_raw, list) or any(
                not isinstance(a, str) for a in args_raw
            ):
                raise HarnessSpecError("args must be a list of strings")
            for i, a in enumerate(args_raw):
                if "\x00" in a:
                    raise HarnessSpecError(f"args[{i}] contains a NUL byte")
                if len(a) > MAX_MEASURE_ARG_LEN:
                    raise HarnessSpecError(
                        f"args[{i}] exceeds {MAX_MEASURE_ARG_LEN} bytes"
                    )
            args = list(args_raw)

        reads_raw = raw.get("reads")
        if reads_raw is not None and kind != "measure":
            raise HarnessSpecError("reads is valid for kind='measure' only")
        reads: List[str] = []
        if reads_raw is not None:
            if not isinstance(reads_raw, list) or any(
                not isinstance(r, str) for r in reads_raw
            ):
                raise HarnessSpecError("reads must be a list of strings")
            for i, r in enumerate(reads_raw):
                guard_relpath(r, f"reads[{i}]")
            reads = list(reads_raw)

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

        if warm_start is not None and kind != "train":
            raise HarnessSpecError("warm_start is valid for train jobs only")

        # Kind-specific requirements: config drives train/bench/head-to-head;
        # head-to-head also needs both checkpoints; evaluate needs a target
        # (the checkpoint/run-dir it evaluates) instead of config, forbidden
        # for train.
        # evaluate takes no config: run-dir mode reads the target's own
        # config.yaml, and the runner never forwards --config for that kind.
        # head-to-head takes two bare checkpoint files with no run dir of
        # their own to derive rules/agent type from (unlike evaluate's
        # run-dir mode), so `cambia head-to-head` hard-requires --config;
        # config is required here too rather than left to the CLI's exists=True
        # default, which resolves against the job's staged worktree cwd and is
        # never a config that matches the submitted checkpoints.
        if kind in ("train", "bench", "head-to-head") and config is None:
            raise HarnessSpecError(f"kind {kind!r} requires a 'config' path")
        if kind == "head-to-head" and (checkpoint_a is None or checkpoint_b is None):
            raise HarnessSpecError("head-to-head requires checkpoint_a and checkpoint_b")
        if kind == "evaluate" and target is None:
            raise HarnessSpecError(
                "evaluate requires a 'target': a runner-local run directory"
            )
        if kind == "train" and target is not None:
            raise HarnessSpecError("target is not valid for kind='train'")

        # Cross-job dependency (cambia-352): after names a single parent job
        # (same name rules as the job itself); a self-reference is rejected. The
        # runner re-checks that the parent exists. on_failure governs only the
        # failure branch. Both are allowed on every kind.
        after = raw.get("after")
        if after is not None:
            validate_name(after)
            if after == name:
                raise HarnessSpecError("after must not reference the job itself")

        on_failure = raw.get("on_failure", "skip")
        if on_failure not in ON_FAILURE_POLICIES:
            raise HarnessSpecError(
                f"on_failure must be one of {list(ON_FAILURE_POLICIES)}, got {on_failure!r}"
            )

        hub_item = raw.get("hub_item")
        if hub_item is not None and (not isinstance(hub_item, str) or not hub_item):
            raise HarnessSpecError("hub_item must be a non-empty string when set")

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
            warm_start=warm_start,
            after=after,
            on_failure=on_failure,
            hub_item=hub_item,
            exclusive=bool(raw.get("exclusive", False)),
            script=script,
            args=args,
            reads=reads,
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
        if self.warm_start is not None:
            payload["warm_start"] = self.warm_start
        # on_failure is inert without after, so both travel together.
        if self.after is not None:
            payload["after"] = self.after
            payload["on_failure"] = self.on_failure
        # hub_item is telemetry provenance: forwarded so the runner persists it
        # into jobspec.json/env.json for a pulled-run-dir link recovery.
        if self.hub_item is not None:
            payload["hub_item"] = self.hub_item
        # exclusive is only forwarded when set; absent, the runner defaults false.
        if self.exclusive:
            payload["exclusive"] = True
        if self.script is not None:
            payload["script"] = self.script
        if self.args:
            payload["args"] = self.args
        if self.reads:
            payload["reads"] = self.reads
        return payload


def parse_spec_file(path: str) -> JobSpec:
    """Load a spec YAML file and parse it into a validated JobSpec."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise HarnessSpecError(f"spec file is empty: {path}")
    return JobSpec.parse(raw)
