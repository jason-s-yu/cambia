"""
tests/test_harness_spec.py

Job-spec + name validation (design 2.6) and the submit dirty-tree refusal
(design 3.1), plus harness config loading (design 5).
"""

import subprocess

import pytest

from src.harness.spec import (
    ALLOWED_KINDS,
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


def test_parse_kind_allowlist_exact():
    assert set(ALLOWED_KINDS) == {"train", "evaluate", "head-to-head", "bench"}


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


def test_parse_device_must_be_cpu():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(device="cuda"))


def test_parse_resume_train_only():
    JobSpec.parse(_train_spec(resume=True))  # ok for train
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(
            {"kind": "evaluate", "name": "e1", "config": "c.yaml", "resume": True}
        )


def test_parse_head_to_head_requires_two_checkpoints():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse({"kind": "head-to-head", "name": "h1", "checkpoint_a": "a.pt"})
    spec = JobSpec.parse(
        {
            "kind": "head-to-head",
            "name": "h1",
            "checkpoint_a": "snapshots/a.pt",
            "checkpoint_b": "snapshots/b.pt",
            "games": 5000,
        }
    )
    assert spec.games == 5000


def test_parse_rejects_nonpositive_games():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse({"kind": "evaluate", "name": "e", "config": "c.yaml", "games": 0})


def test_parse_overrides_must_be_mapping():
    with pytest.raises(HarnessSpecError):
        JobSpec.parse(_train_spec(overrides=["a=b"]))


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


class _FakeCfg:
    class _DP:
        mirror_remote_url = "cambia@runner:/srv/cambia/mirror.git"
        ssh_alias = "runner"
        origin_host = "runner"

    data_plane = _DP()
