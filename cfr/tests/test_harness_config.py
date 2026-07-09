"""
tests/test_harness_config.py

Harness config loading + validation (design 5.1/5.2).
"""

import pytest

from src.harness import config as cfgmod

_GOOD = {
    "runner": {
        "url": "https://192.0.2.10:8090",
        "cert_fingerprint": "AA:BB" + ":00" * 30,
    },
    "auth": {
        "private_key_path": "/tmp/does-not-need-to-exist/key",
        "subject": "cambia-harness",
        "token_ttl_seconds": 900,
    },
    "data_plane": {
        "ssh_alias": "runner",
        "runner_runs_dir": "/srv/cambia/runs",
        "mirror_remote_name": "runner-mirror",
        "mirror_remote_url": "cambia@runner:/srv/cambia/mirror.git",
    },
    "sync": {"interval_seconds": 60, "local_runs_dir": "runs"},
}


def test_from_dict_ok():
    cfg = cfgmod.from_dict(_GOOD).validate()
    assert cfg.runner.url.startswith("https://")
    assert cfg.data_plane.origin_host == "runner"  # defaults to ssh_alias
    assert len(cfg.normalized_fingerprint()) == 64


def test_validate_rejects_plaintext_url():
    bad = {**_GOOD, "runner": {**_GOOD["runner"], "url": "http://x:8090"}}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(bad).validate()


def test_validate_rejects_bad_fingerprint():
    bad = {**_GOOD, "runner": {**_GOOD["runner"], "cert_fingerprint": "zz"}}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(bad).validate()


def test_validate_caps_token_ttl():
    bad = {**_GOOD, "auth": {**_GOOD["auth"], "token_ttl_seconds": 7200}}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(bad).validate()


def test_missing_required_field():
    bad = {**_GOOD, "runner": {"url": "https://x:8090"}}  # no fingerprint
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(bad)


def test_load_missing_config_raises_with_searched_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMBIA_HARNESS_CONFIG", raising=False)
    # Force every candidate to a nonexistent path.
    monkeypatch.setattr(cfgmod, "DEFAULT_USER_CONFIG", tmp_path / "nope.yaml")
    with pytest.raises(cfgmod.HarnessConfigError) as exc:
        cfgmod.load(str(tmp_path / "also-nope.yaml"))
    assert "Searched" in str(exc.value)


def test_load_from_explicit_path(tmp_path):
    import yaml

    p = tmp_path / "harness.yaml"
    p.write_text(yaml.safe_dump(_GOOD))
    cfg = cfgmod.load(str(p))
    assert cfg.source_path == str(p)
    assert cfg.sync.interval_seconds == 60
