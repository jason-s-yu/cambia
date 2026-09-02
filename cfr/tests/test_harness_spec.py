"""
tests/test_harness_spec.py

Job-spec + name validation (design 2.6) and the submit dirty-tree refusal
(design 3.1), plus harness config loading (design 5).
"""

import subprocess

import pytest

from src.harness.spec import (
    ALLOWED_DEVICES,
    ALLOWED_KINDS,
    MAX_MEASURE_ARG_LEN,
    HarnessSpecError,
    JobSpec,
    guard_relpath,
    validate_name,
)

# ---------------------------------------------------------------------------
# validate_name (mirror of Go ValidateName)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["v0.4-prtcfr-r12", "abc", "A1", "run_1.2", "x" * 128],
)
def test_validate_name_accepts(name):
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "-leading-dash",  # must start alphanumeric
        "_leading",
        ".hidden",
        "a/b",
        "../evil",
        "a..b",
        "name with space",
        "x" * 129,  # over the 128 cap
        "unïcode",
    ],
)
def test_validate_name_rejects(name):
    with pytest.raises(HarnessSpecError):
        validate_name(name)


def test_guard_relpath():
    assert guard_relpath("cfr/config/x.yaml", "config") == "cfr/config/x.yaml"
    for bad in ["/etc/passwd", "../secret", "a/../b", "C:/win"]:
        with pytest.raises(HarnessSpecError):
            guard_relpath(bad, "config")


# ---------------------------------------------------------------------------
# JobSpec.parse (design 2.6 validation order)
# ---------------------------------------------------------------------------


def _train_spec(**over):
    base = {
        "kind": "train",
        "name": "v0.4-prtcfr-r1",
        "config": "cfr/config/prtcfr_prod.yaml",
    }
    base.update(over)
    return base


def test_parse_train_minimal():
    spec = JobSpec.parse(_train_spec())
    assert spec.kind == "train"
    assert spec.name == "v0.4-prtcfr-r1"
    assert spec.device == "cpu"
    assert spec.priority == "normal"


def test_parse_rejects_unknown_kind():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(kind="rl-ppo"))


def test_hub_item_round_trips_spec_to_payload():
    spec = JobSpec.parse(_train_spec(hub_item="cambia-359"))
    assert spec.hub_item == "cambia-359"
    payload = spec.to_payload("a" * 40)
    assert payload["hub_item"] == "cambia-359"


def test_hub_item_omitted_by_default():
    spec = JobSpec.parse(_train_spec())
    assert spec.hub_item is None
    assert "hub_item" not in spec.to_payload("a" * 40)


def test_hub_item_must_be_nonempty_string():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(hub_item=""))
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(hub_item=123))


# exclusive run-alone flag (cambia-655): allowed on every kind, default false,
# forwarded to the payload only when set.
def test_exclusive_defaults_false_and_omitted():
    spec = JobSpec.parse(_train_spec())
    assert spec.exclusive is False
    assert "exclusive" not in spec.to_payload("a" * 40)


def test_exclusive_round_trips_spec_to_payload():
    spec = JobSpec.parse(_train_spec(exclusive=True))
    assert spec.exclusive is True
    assert spec.to_payload("a" * 40)["exclusive"] is True


def test_exclusive_allowed_on_every_kind():
    spec = JobSpec.parse(_eval_spec(exclusive=True))
    assert spec.exclusive is True
    assert spec.to_payload("f" * 40)["exclusive"] is True


def test_parse_kind_allowlist_exact():
    assert set(ALLOWED_KINDS) == {
        "train",
        "evaluate",
        "head-to-head",
        "bench",
        "measure",
    }


def test_parse_rejects_unknown_keys():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(unexpected_field=1))


def test_parse_rejects_absolute_config():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(config="/etc/passwd"))


def test_parse_rejects_traversal_config():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(config="../../secret.yaml"))


def test_parse_commit_shape():
    good = "a" * 40
    spec = JobSpec.parse(_train_spec(commit=good))
    assert spec.commit == good
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(commit="deadbeef"))  # not 40 hex


def test_parse_device_allowlist_exact():
    assert set(ALLOWED_DEVICES) == {"cpu", "cuda", "xpu"}


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_parse_device_accepts_allowlisted(device):
    spec = JobSpec.parse(_train_spec(device=device))
    assert spec.device == device


@pytest.mark.parametrize("device", ["rocm", "mps", "CUDA", "cuda:0", ""])
def test_parse_device_rejects_non_allowlisted(device):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(device=device))


def test_parse_resume_train_only():
    JobSpec.parse(_train_spec(resume=True))  # ok for train
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(
            {"kind": "evaluate", "name": "e1", "config": "c.yaml", "resume": True}
        )


def test_parse_head_to_head_requires_two_checkpoints():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(
            {
                "kind": "head-to-head",
                "name": "h1",
                "config": "cfr/config/prtcfr_prod.yaml",
                "checkpoint_a": "a.pt",
            }
        )
    spec = JobSpec.parse(
        {
            "kind": "head-to-head",
            "name": "h1",
            "config": "cfr/config/prtcfr_prod.yaml",
            "checkpoint_a": "snapshots/a.pt",
            "checkpoint_b": "snapshots/b.pt",
            "games": 5000,
        }
    )
    assert spec.games == 5000


def test_parse_head_to_head_requires_config():
    # Unlike evaluate's run-dir mode, `cambia head-to-head` takes two bare
    # checkpoint files with no run dir to derive rules/agent type from, so it
    # hard-requires --config (cambia-295 item 1 contract: config is required
    # for kind=head-to-head, same as train/bench).
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(
            {
                "kind": "head-to-head",
                "name": "h1",
                "checkpoint_a": "snapshots/a.pt",
                "checkpoint_b": "snapshots/b.pt",
            }
        )


def test_parse_rejects_nonpositive_games():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(
            {
                "kind": "evaluate",
                "name": "e",
                "config": "c.yaml",
                "target": "prior-run",
                "games": 0,
            }
        )


# ---------------------------------------------------------------------------
# target field (cambia-256 review fix: evaluate requires it, train forbids it)
# ---------------------------------------------------------------------------


def _eval_spec(**over):
    base = {
        "kind": "evaluate",
        "name": "eval-1",
        "config": "cfr/config/prtcfr_prod.yaml",
        "target": "v0.4-prtcfr-r12",
    }
    base.update(over)
    return base


def test_parse_evaluate_requires_target():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse({"kind": "evaluate", "name": "e1", "config": "cfr/config/x.yaml"})


def test_parse_evaluate_accepts_dir_and_file_targets():
    spec = JobSpec.parse(_eval_spec(target="v0.4-prtcfr-r12"))
    assert spec.target == "v0.4-prtcfr-r12"

    spec2 = JobSpec.parse(
        _eval_spec(target="v0.4-prtcfr-r12/snapshots/prtcfr_checkpoint_100.pt")
    )
    assert spec2.target == "v0.4-prtcfr-r12/snapshots/prtcfr_checkpoint_100.pt"


def test_parse_train_forbids_target():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(target="v0.4-prtcfr-r12"))


def test_parse_rejects_absolute_target():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_eval_spec(target="/etc/passwd"))


def test_parse_rejects_traversal_target():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_eval_spec(target="../../secret.pt"))


def test_to_payload_includes_target_for_evaluate():
    spec = JobSpec.parse(_eval_spec(games=5000))
    payload = spec.to_payload("f" * 40)
    assert payload["target"] == "v0.4-prtcfr-r12"
    assert payload["kind"] == "evaluate"


def test_to_payload_omits_target_when_unset():
    spec = JobSpec.parse(_train_spec())
    payload = spec.to_payload("f" * 40)
    assert "target" not in payload


# ---------------------------------------------------------------------------
# warm_start field (cambia-334): train-only, same path guards as target/config
# ---------------------------------------------------------------------------


def test_parse_warm_start_accepts_train():
    spec = JobSpec.parse(
        _train_spec(warm_start="v0.4-x2-530/snapshots/prtcfr_snapshot_iter_530.pt")
    )
    assert spec.warm_start == "v0.4-x2-530/snapshots/prtcfr_snapshot_iter_530.pt"


def test_parse_warm_start_train_only():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_eval_spec(warm_start="prior-run/resume_state.json"))


@pytest.mark.parametrize("bad", ["/etc/passwd", "../secret/x.pt", "a/../b.pt"])
def test_parse_warm_start_rejects_unsafe_path(bad):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(warm_start=bad))


def test_to_payload_includes_warm_start_when_set():
    spec = JobSpec.parse(_train_spec(warm_start="prior/resume_state.json"))
    payload = spec.to_payload("f" * 40)
    assert payload["warm_start"] == "prior/resume_state.json"


def test_to_payload_omits_warm_start_when_unset():
    spec = JobSpec.parse(_train_spec())
    payload = spec.to_payload("f" * 40)
    assert "warm_start" not in payload


# ---------------------------------------------------------------------------
# measure kind (design D38, cambia-1072): script/args/reads, measure-only,
# mirroring the target/warm_start kind-scoping tests above.
# ---------------------------------------------------------------------------


def _measure_spec(**over):
    base = {
        "kind": "measure",
        "name": "measure-1",
        "script": "cfr/scripts/measure_gate_gap.py",
    }
    base.update(over)
    return base


def test_parse_measure_minimal():
    spec = JobSpec.parse(_measure_spec())
    assert spec.kind == "measure"
    assert spec.script == "cfr/scripts/measure_gate_gap.py"
    assert spec.args == []
    assert spec.reads == []


def test_parse_measure_requires_script():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse({"kind": "measure", "name": "measure-noscript"})


@pytest.mark.parametrize(
    "bad",
    [
        "/etc/passwd",
        "../secret.py",
        "cfr/other/measure_gate_gap.py",
        "cfr/scripts/",
        "cfr/scripts",
        "cfr/scripts_evil/x.py",
    ],
)
def test_parse_measure_script_must_resolve_under_scripts_root(bad):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(script=bad))


def test_parse_script_forbidden_outside_measure():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(script="cfr/scripts/measure_gate_gap.py"))


def test_parse_measure_args_verbatim():
    spec = JobSpec.parse(_measure_spec(args=["--run", "v0.4-x2r-c1", "--shards", "17"]))
    assert spec.args == ["--run", "v0.4-x2r-c1", "--shards", "17"]


def test_parse_measure_args_rejects_nul_byte():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(args=["a\x00b"]))


def test_parse_measure_args_rejects_over_cap():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(args=["x" * (MAX_MEASURE_ARG_LEN + 1)]))


def test_parse_measure_args_accepts_at_cap():
    spec = JobSpec.parse(_measure_spec(args=["x" * MAX_MEASURE_ARG_LEN]))
    assert spec.args == ["x" * MAX_MEASURE_ARG_LEN]


def test_parse_measure_args_must_be_list_of_strings():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(args="not-a-list"))
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(args=[1, 2]))


def test_parse_args_forbidden_outside_measure():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(args=["--x"]))


def test_parse_measure_reads_accepted():
    spec = JobSpec.parse(_measure_spec(reads=["v0.4-x2r-c1", "v0.4-x2r-c-rep"]))
    assert spec.reads == ["v0.4-x2r-c1", "v0.4-x2r-c-rep"]


@pytest.mark.parametrize("bad", ["/etc/passwd", "../secret", "a/../b"])
def test_parse_measure_reads_rejects_unsafe_path(bad):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(reads=[bad]))


def test_parse_measure_reads_must_be_list_of_strings():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_measure_spec(reads="not-a-list"))


def test_parse_reads_forbidden_outside_measure():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(reads=["some-run"]))


def test_to_payload_includes_measure_fields():
    spec = JobSpec.parse(_measure_spec(args=["--run", "v0.4-x2r-c1"], reads=["seed-run"]))
    payload = spec.to_payload("f" * 40)
    assert payload["kind"] == "measure"
    assert payload["script"] == "cfr/scripts/measure_gate_gap.py"
    assert payload["args"] == ["--run", "v0.4-x2r-c1"]
    assert payload["reads"] == ["seed-run"]


def test_to_payload_omits_args_and_reads_when_unset():
    spec = JobSpec.parse(_measure_spec())
    payload = spec.to_payload("f" * 40)
    assert "args" not in payload
    assert "reads" not in payload


def test_to_payload_omits_script_for_non_measure():
    spec = JobSpec.parse(_train_spec())
    payload = spec.to_payload("f" * 40)
    assert "script" not in payload


# ---------------------------------------------------------------------------
# after / on_failure dependency fields (cambia-352): allowed on every kind, same
# name rules as the job, self-reference rejected, on_failure enum-checked.
# ---------------------------------------------------------------------------


def test_parse_after_accepts_valid_parent():
    spec = JobSpec.parse(_train_spec(after="v0.4-prtcfr-r0"))
    assert spec.after == "v0.4-prtcfr-r0"
    # on_failure defaults to skip.
    assert spec.on_failure == "skip"


def test_parse_after_allowed_on_every_kind():
    spec = JobSpec.parse(_eval_spec(after="prior-run"))
    assert spec.after == "prior-run"


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a..b", "", "-lead", "x" * 129])
def test_parse_after_rejects_bad_name(bad):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(after=bad))


def test_parse_after_rejects_self_reference():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(name="r1", after="r1"))


@pytest.mark.parametrize("policy", ["skip", "run", "fail"])
def test_parse_on_failure_accepts_policies(policy):
    spec = JobSpec.parse(_train_spec(after="parent", on_failure=policy))
    assert spec.on_failure == policy


@pytest.mark.parametrize("bad", ["explode", "SKIP", "", "retry"])
def test_parse_on_failure_rejects_bad_policy(bad):
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(after="parent", on_failure=bad))


def test_to_payload_bundles_after_and_on_failure():
    spec = JobSpec.parse(_train_spec(after="parent", on_failure="run"))
    payload = spec.to_payload("f" * 40)
    assert payload["after"] == "parent"
    assert payload["on_failure"] == "run"


def test_to_payload_omits_after_when_unset():
    payload = JobSpec.parse(_train_spec()).to_payload("f" * 40)
    assert "after" not in payload
    assert "on_failure" not in payload


# ---------------------------------------------------------------------------
# after as a fan-in list (design D29, cambia-1713): the AND-join wire shape.
# The list form runs through the same per-parent validation as the single-
# string form, and the shape given to parse() is the shape to_payload() emits
# unchanged -- a client using the pre-r2 string form stays byte-identical.
# ---------------------------------------------------------------------------


def test_parse_after_accepts_list_of_parents():
    spec = JobSpec.parse(_train_spec(after=["p1", "p2", "p3"]))
    assert spec.after == ["p1", "p2", "p3"]


def test_parse_after_list_validates_every_entry_name():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(after=["p1", "../evil"]))


def test_parse_after_list_rejects_self_reference_anywhere():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(name="r1", after=["p1", "r1", "p2"]))


def test_parse_after_list_rejects_non_string_entries():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(after=["p1", 2]))


def test_parse_after_rejects_non_string_non_list():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(after={"p1": True}))


def test_parse_after_list_allows_empty_list():
    # design D29: a list of 0..N names; an empty list is a valid (if inert)
    # AND-join wire shape, equivalent to no dependency.
    spec = JobSpec.parse(_train_spec(after=[]))
    assert spec.after == []


def test_to_payload_preserves_list_shape():
    spec = JobSpec.parse(_train_spec(after=["p1", "p2"], on_failure="fail"))
    payload = spec.to_payload("f" * 40)
    assert payload["after"] == ["p1", "p2"]
    assert payload["on_failure"] == "fail"


def test_to_payload_preserves_single_string_shape():
    # The pre-r2 wire shape is unchanged: a bare string in, a bare string out,
    # so an old daemon that only understands a single-parent string keeps
    # working against a new client that was not asked for fan-in.
    spec = JobSpec.parse(_train_spec(after="parent"))
    payload = spec.to_payload("f" * 40)
    assert payload["after"] == "parent"
    assert isinstance(payload["after"], str)


def test_parse_overrides_must_be_mapping():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(overrides=["a=b"]))


# ---------------------------------------------------------------------------
# requires block (design D10/D12, cambia-1725): a client-side shape gate over
# runnerd/nashnet/capability.Requires's JSON tags. The coordinator is the
# authoritative validator (D9); this only catches a typo'd key or an
# obviously wrong type before a submit round-trips a 400.
# ---------------------------------------------------------------------------


def test_parse_requires_round_trips_every_known_field():
    block = {
        "device": "cuda",
        "min_vram_gb": 24,
        "min_cores": 8,
        "min_ram_gb": 32.5,
        "min_disk_gb": 100,
        "needs_libcambia": True,
        "node": "n-abc123def456",
        "labels_any": ["gpu", "fast"],
    }
    spec = JobSpec.parse(_train_spec(requires=block))
    assert spec.requires == block
    payload = spec.to_payload("f" * 40)
    assert payload["requires"] == block


def test_parse_requires_defaults_to_none():
    spec = JobSpec.parse(_train_spec())
    assert spec.requires is None
    payload = spec.to_payload("f" * 40)
    assert "requires" not in payload


def test_parse_requires_must_be_mapping():
    with pytest.raises(HarnessSpecError, match="requires must be a mapping"):
        JobSpec.parse(_train_spec(requires=["node=n-a"]))


def test_parse_requires_rejects_unknown_key():
    with pytest.raises(HarnessSpecError, match="unknown requires keys"):
        JobSpec.parse(_train_spec(requires={"nod": "n-a"}))


def test_parse_requires_rejects_bad_device():
    with pytest.raises(HarnessSpecError, match="requires.device"):
        JobSpec.parse(_train_spec(requires={"device": "tpu"}))


@pytest.mark.parametrize("key", ["min_vram_gb", "min_ram_gb", "min_disk_gb"])
def test_parse_requires_rejects_negative_floor(key):
    with pytest.raises(HarnessSpecError, match=f"requires.{key}"):
        JobSpec.parse(_train_spec(requires={key: -1}))


def test_parse_requires_rejects_non_numeric_floor():
    with pytest.raises(HarnessSpecError, match="requires.min_ram_gb"):
        JobSpec.parse(_train_spec(requires={"min_ram_gb": "lots"}))


def test_parse_requires_rejects_bool_as_numeric_floor():
    # isinstance(True, int) is True in Python; a bool must not sneak past the
    # numeric-floor check.
    with pytest.raises(HarnessSpecError, match="requires.min_cores"):
        JobSpec.parse(_train_spec(requires={"min_cores": True}))


def test_parse_requires_rejects_negative_min_cores():
    with pytest.raises(HarnessSpecError, match="requires.min_cores"):
        JobSpec.parse(_train_spec(requires={"min_cores": -1}))


def test_parse_requires_rejects_non_bool_needs_libcambia():
    with pytest.raises(HarnessSpecError, match="needs_libcambia"):
        JobSpec.parse(_train_spec(requires={"needs_libcambia": "yes"}))


def test_parse_requires_validates_node_name():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(requires={"node": "../etc/passwd"}))


def test_parse_requires_rejects_bad_labels_any():
    with pytest.raises(HarnessSpecError, match="labels_any"):
        JobSpec.parse(_train_spec(requires={"labels_any": ["ok", ""]}))
    with pytest.raises(HarnessSpecError, match="labels_any"):
        JobSpec.parse(_train_spec(requires={"labels_any": "gpu"}))


def test_to_payload_omits_requires_when_empty_dict():
    spec = JobSpec.parse(_train_spec(requires={}))
    payload = spec.to_payload("f" * 40)
    assert "requires" not in payload


def test_to_payload_stamps_commit():
    spec = JobSpec.parse(_train_spec(overrides={"prt_cfr.iterations": 500}))
    payload = spec.to_payload("f" * 40)
    assert payload["commit"] == "f" * 40
    assert payload["kind"] == "train"
    assert payload["overrides"] == {"prt_cfr.iterations": 500}
    assert payload["device"] == "cpu"


def test_parse_spec_file(tmp_path):
    from src.harness.spec import parse_spec_file

    p = tmp_path / "job.yaml"
    p.write_text(
        "kind: train\nname: v0.4-prtcfr-r5\nconfig: cfr/config/prtcfr_prod.yaml\n"
    )
    spec = parse_spec_file(str(p))
    assert spec.name == "v0.4-prtcfr-r5"

    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(HarnessSpecError):
        parse_spec_file(str(empty))


# ---------------------------------------------------------------------------
# Dirty-tree refusal on submit (design 3.1), exercised via the CLI's git helpers
# against a real temp git repo.
# ---------------------------------------------------------------------------


def _git(args, cwd):
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.txt").write_text("hello\n")
    _git(["add", "a.txt"], repo)
    _git(["commit", "-qm", "init"], repo)
    return repo


def test_is_dirty_clean_then_dirty(tmp_path):
    from src.harness.cli import _is_dirty

    repo = _init_repo(tmp_path)
    assert _is_dirty(repo) is False
    (repo / "a.txt").write_text("changed\n")
    assert _is_dirty(repo) is True


def test_submit_refuses_dirty_tree(tmp_path, monkeypatch):
    import typer

    import src.harness.cli as cli

    repo = _init_repo(tmp_path)
    (repo / "b.txt").write_text("untracked\n")  # dirty

    # Point the CLI at a config so _load_cfg succeeds, and at the temp repo.
    monkeypatch.setattr(cli, "_load_cfg", lambda c: object())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r1\nconfig: cfr/config/x.yaml\n")

    with pytest.raises(typer.Exit):
        cli.submit(spec_file=spec_file, force=False, config=None)


def test_submit_clean_tree_pushes_and_posts(tmp_path, monkeypatch):
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)  # clean

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)

    pushes = []
    real_git = cli._git

    def fake_git(args, cwd):
        if args[:1] == ["push"]:
            pushes.append(args)
            return ""
        return real_git(args, cwd)

    monkeypatch.setattr(cli, "_git", fake_git)

    posted = {}

    class FakeClient:
        def submit(self, payload, force=False):
            posted["payload"] = payload
            posted["force"] = force
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r1\nconfig: cfr/config/x.yaml\n")

    cli.submit(spec_file=spec_file, force=False, config=None)

    assert len(pushes) == 1
    # push target is <sha>:refs/harness/<name>
    assert pushes[0][-1].endswith(":refs/harness/r1")
    assert posted["payload"]["name"] == "r1"
    assert posted["payload"]["kind"] == "train"
    # commit stamped as the 40-hex HEAD sha
    assert len(posted["payload"]["commit"]) == 40


def test_submit_after_flag_overrides_and_posts(tmp_path, monkeypatch):
    # The --after/--on-failure CLI flags override the spec file and reach the
    # posted payload (cambia-352).
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)  # clean

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)

    real_git = cli._git
    monkeypatch.setattr(
        cli, "_git", lambda a, cwd: "" if a[:1] == ["push"] else real_git(a, cwd)
    )

    posted = {}

    class FakeClient:
        def submit(self, payload, force=False):
            posted["payload"] = payload
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r2\nconfig: cfr/config/x.yaml\n")

    cli.submit(
        spec_file=spec_file,
        force=False,
        after="parent-run",
        on_failure="run",
        config=None,
    )

    assert posted["payload"]["after"] == "parent-run"
    assert posted["payload"]["on_failure"] == "run"


def _submit_harness(tmp_path, monkeypatch, name):
    """Shared submit() test rig: a clean temp repo, a fake client that
    records the posted payload, and a trivial spec file. Returns (cli module,
    spec_file path, posted dict)."""
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)
    real_git = cli._git
    monkeypatch.setattr(
        cli, "_git", lambda a, cwd: "" if a[:1] == ["push"] else real_git(a, cwd)
    )

    posted = {}

    class FakeClient:
        def submit(self, payload, force=False):
            posted["payload"] = payload
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / f"{name}.yaml"
    spec_file.write_text(f"kind: train\nname: {name}\nconfig: cfr/config/x.yaml\n")
    return cli, spec_file, posted


def test_submit_after_repeated_becomes_fan_in_list(tmp_path, monkeypatch):
    # --after given twice matches the fan-in AND-join wire shape (design D29,
    # cambia-1713/cambia-1725): a list of 2+ names, not the pre-r2 string.
    cli, spec_file, posted = _submit_harness(tmp_path, monkeypatch, "r5")

    cli.submit(spec_file=spec_file, force=False, after=["p1", "p2"], config=None)

    assert posted["payload"]["after"] == ["p1", "p2"]


def test_submit_after_single_value_keeps_string_wire_shape(tmp_path, monkeypatch):
    # Exactly one --after (via the real CLI's list form) stays a bare string,
    # byte-identical against a daemon that predates fan-in.
    cli, spec_file, posted = _submit_harness(tmp_path, monkeypatch, "r6")

    cli.submit(spec_file=spec_file, force=False, after=["parent-only"], config=None)

    assert posted["payload"]["after"] == "parent-only"


def test_submit_after_empty_list_is_unset(tmp_path, monkeypatch):
    # typer hands back None when --after is never passed; an explicit empty
    # list (a direct call) must behave identically.
    cli, spec_file, posted = _submit_harness(tmp_path, monkeypatch, "r7")

    cli.submit(spec_file=spec_file, force=False, after=[], config=None)

    assert "after" not in posted["payload"]


def test_submit_require_node_sets_requires_node(tmp_path, monkeypatch):
    # --require-node (cambia-1725) sets requires.node without a spec-file
    # requires: block.
    cli, spec_file, posted = _submit_harness(tmp_path, monkeypatch, "r8")

    cli.submit(
        spec_file=spec_file, force=False, require_node="n-abc123def456", config=None
    )

    assert posted["payload"]["requires"] == {"node": "n-abc123def456"}


def test_submit_require_node_preserves_existing_requires_block(tmp_path, monkeypatch):
    # --require-node merges into an existing spec-file requires: block rather
    # than discarding its other constraints.
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)
    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)
    real_git = cli._git
    monkeypatch.setattr(
        cli, "_git", lambda a, cwd: "" if a[:1] == ["push"] else real_git(a, cwd)
    )

    posted = {}

    class FakeClient:
        def submit(self, payload, force=False):
            posted["payload"] = payload
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "r9.yaml"
    spec_file.write_text(
        "kind: train\nname: r9\nconfig: cfr/config/x.yaml\n"
        "requires:\n  device: cuda\n  min_vram_gb: 16\n"
    )

    cli.submit(
        spec_file=spec_file, force=False, require_node="n-abc123def456", config=None
    )

    assert posted["payload"]["requires"] == {
        "device": "cuda",
        "min_vram_gb": 16,
        "node": "n-abc123def456",
    }


def test_submit_exclusive_flag_overrides_and_posts(tmp_path, monkeypatch):
    # The --exclusive CLI flag forces exclusivity on a spec file that omits it and
    # reaches the posted payload (cambia-655).
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)  # clean

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)

    real_git = cli._git
    monkeypatch.setattr(
        cli, "_git", lambda a, cwd: "" if a[:1] == ["push"] else real_git(a, cwd)
    )

    posted = {}

    class FakeClient:
        def submit(self, payload, force=False):
            posted["payload"] = payload
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r4\nconfig: cfr/config/x.yaml\n")

    cli.submit(spec_file=spec_file, force=False, exclusive=True, config=None)

    assert posted["payload"]["exclusive"] is True


class _FakeCfg:
    class _DP:
        mirror_remote_url = "cambia@runner:/srv/cambia/mirror.git"
        ssh_alias = "runner"
        origin_host = "runner"

    data_plane = _DP()
    require_signed_commit = False


class _FakeCfgSigned(_FakeCfg):
    require_signed_commit = True


# ---------------------------------------------------------------------------
# Optional pre-push verify gate (cambia-551 W2): `require_signed_commit`,
# default off, checked via `git verify-commit` between rev-parse and push.
# ---------------------------------------------------------------------------


def test_is_commit_signed_false_for_unsigned_commit(tmp_path):
    # Real integration check against an actual (unsigned) temp-repo commit:
    # no gpg/ssh signing is configured for _init_repo, so `git verify-commit`
    # genuinely fails and _is_commit_signed must report that honestly.
    from src.harness.cli import _is_commit_signed

    repo = _init_repo(tmp_path)
    sha = _git_rev_parse_head(repo)
    assert _is_commit_signed(sha, repo) is False


def _git_rev_parse_head(repo):
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
    )
    assert proc.returncode == 0
    return proc.stdout.strip()


def test_submit_signed_gate_off_by_default_never_checks(tmp_path, monkeypatch):
    # require_signed_commit=False (the default): submit must not even call the
    # verify helper, and pushes exactly as before.
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfg())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)

    def boom(sha, repo):
        raise AssertionError("verify gate must not run when require_signed_commit is off")

    monkeypatch.setattr(cli, "_is_commit_signed", boom)

    pushes = []
    real_git = cli._git

    def fake_git(args, cwd):
        if args[:1] == ["push"]:
            pushes.append(args)
            return ""
        return real_git(args, cwd)

    monkeypatch.setattr(cli, "_git", fake_git)

    class FakeClient:
        def submit(self, payload, force=False):
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r3\nconfig: cfr/config/x.yaml\n")

    cli.submit(spec_file=spec_file, force=False, config=None)
    assert len(pushes) == 1


def test_submit_signed_gate_blocks_unverifiable_commit(tmp_path, monkeypatch):
    import typer

    import src.harness.cli as cli

    repo = _init_repo(tmp_path)

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfgSigned())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)
    monkeypatch.setattr(cli, "_is_commit_signed", lambda sha, repo: False)

    pushed = []
    real_git = cli._git
    monkeypatch.setattr(
        cli,
        "_git",
        lambda a, cwd: (pushed.append(a) or "") if a[:1] == ["push"] else real_git(a, cwd),
    )

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r4\nconfig: cfr/config/x.yaml\n")

    with pytest.raises(typer.Exit):
        cli.submit(spec_file=spec_file, force=False, config=None)
    assert pushed == []  # aborted before the push, not after


def test_submit_signed_gate_allows_verified_commit(tmp_path, monkeypatch):
    import src.harness.cli as cli

    repo = _init_repo(tmp_path)

    monkeypatch.setattr(cli, "_load_cfg", lambda c: _FakeCfgSigned())
    monkeypatch.setattr(cli, "_repo_root", lambda: repo)
    monkeypatch.setattr(cli, "_is_commit_signed", lambda sha, repo: True)

    pushes = []
    real_git = cli._git

    def fake_git(args, cwd):
        if args[:1] == ["push"]:
            pushes.append(args)
            return ""
        return real_git(args, cwd)

    monkeypatch.setattr(cli, "_git", fake_git)

    class FakeClient:
        def submit(self, payload, force=False):
            return {"job_id": payload["name"], "state": "queued", "queue_pos": 0}

    monkeypatch.setattr(cli, "_build_client", lambda cfg: FakeClient())

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("kind: train\nname: r5\nconfig: cfr/config/x.yaml\n")

    cli.submit(spec_file=spec_file, force=False, config=None)
    assert len(pushes) == 1
