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


def test_require_signed_commit_defaults_off():
    cfg = cfgmod.from_dict(_GOOD).validate()
    assert cfg.require_signed_commit is False


def test_require_signed_commit_parses_true():
    cfg = cfgmod.from_dict({**_GOOD, "require_signed_commit": True}).validate()
    assert cfg.require_signed_commit is True


def test_validate_allows_divergent_data_plane_targets():
    # ssh_alias (rsync) and mirror_remote_url (git push) may intentionally
    # point at different hosts/keys (cambia-551 W2 two-alias pattern); this
    # must not raise.
    divergent = {
        **_GOOD,
        "data_plane": {
            "ssh_alias": "runs-host",
            "runner_runs_dir": "/srv/cambia/runs",
            "mirror_remote_url": "git@mirror-host:/srv/cambia/mirror.git",
        },
    }
    cfg = cfgmod.from_dict(divergent).validate()
    assert cfg.data_plane.ssh_alias == "runs-host"
    assert cfg.data_plane.mirror_remote_url == "git@mirror-host:/srv/cambia/mirror.git"


def test_load_from_explicit_path(tmp_path):
    import yaml

    p = tmp_path / "harness.yaml"
    p.write_text(yaml.safe_dump(_GOOD))
    cfg = cfgmod.load(str(p))
    assert cfg.source_path == str(p)
    assert cfg.sync.interval_seconds == 60


# --- hub reflection section (cambia-353) ---
_HUB = {
    "url": "https://hub.example.com",
    "slug": "cambia-harness-reflect",
    "collector_item": "cambia-439",
    "secret_env": "CAMBIA_HUB_SECRET",
}


def test_absent_hub_section_is_none():
    cfg = cfgmod.from_dict(_GOOD).validate()
    assert cfg.hub is None  # reflection disabled, zero behavior change


def test_hub_section_parses():
    cfg = cfgmod.from_dict({**_GOOD, "hub": dict(_HUB)}).validate()
    assert cfg.hub is not None
    assert cfg.hub.url == "https://hub.example.com"
    assert cfg.hub.collector_item == "cambia-439"
    assert cfg.hub.secret_env == "CAMBIA_HUB_SECRET"


def test_hub_rejects_plaintext_url():
    bad = {**_GOOD, "hub": {**_HUB, "url": "http://hub.example.com"}}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(bad).validate()


def test_hub_requires_exactly_one_secret_source():
    # both -> reject
    both = {**_GOOD, "hub": {**_HUB, "secret_file": "/tmp/s"}}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict(both).validate()
    # neither -> reject
    neither = {k: v for k, v in _HUB.items() if k != "secret_env"}
    with pytest.raises(cfgmod.HarnessConfigError):
        cfgmod.from_dict({**_GOOD, "hub": neither}).validate()


def test_hub_resolve_secret_from_env(monkeypatch):
    cfg = cfgmod.from_dict({**_GOOD, "hub": dict(_HUB)}).validate()
    monkeypatch.setenv("CAMBIA_HUB_SECRET", "  s3cr3t  ")
    assert cfg.hub.resolve_secret() == "s3cr3t"
    monkeypatch.delenv("CAMBIA_HUB_SECRET", raising=False)
    with pytest.raises(cfgmod.HarnessConfigError):
        cfg.hub.resolve_secret()


def test_hub_resolve_secret_from_file(tmp_path):
    sf = tmp_path / "sec"
    sf.write_text("filesecret\n")
    hub = {k: v for k, v in _HUB.items() if k != "secret_env"}
    hub["secret_file"] = str(sf)
    cfg = cfgmod.from_dict({**_GOOD, "hub": hub}).validate()
    assert cfg.hub.resolve_secret() == "filesecret"
