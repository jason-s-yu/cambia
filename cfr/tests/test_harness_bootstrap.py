"""
tests/test_harness_bootstrap.py

`cambia harness init` (cambia-256): keypair generation in the exact formats
transport.py / runnerd/authtoken consume, refuse-without-force semantics, and
harness.yaml scaffolding from the checked-in template.
"""

import stat

import pytest
import yaml

from src.harness import bootstrap


def _pubkey_pem(pub_bytes: bytes) -> bytes:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = Ed25519PublicKey.from_public_bytes(pub_bytes)
    return key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


# ---------------------------------------------------------------------------
# Key format roundtrip against the real transport loader
# ---------------------------------------------------------------------------


def test_private_key_roundtrips_through_transport_loader(tmp_path):
    """The private key init writes must be exactly what
    transport.load_ed25519_private_key + mint_token can sign with, and the
    resulting token must verify against the public key init writes -- proving
    the two files are a matched pair in the formats the rest of the harness
    consumes."""
    import jwt as pyjwt

    from src.harness.transport import TOKEN_AUDIENCE, load_ed25519_private_key, mint_token

    seed, pub = bootstrap.generate_ed25519_keypair()
    priv_path = tmp_path / "jwt_ed25519"
    pub_path = tmp_path / "jwt_ed25519.pub"
    bootstrap.write_private_key(priv_path, seed)
    bootstrap.write_public_key(pub_path, pub)

    loaded = load_ed25519_private_key(str(priv_path))
    token = mint_token(loaded, subject="cambia-harness", ttl_seconds=60)

    pub_bytes_on_disk = pub_path.read_bytes()
    assert pub_bytes_on_disk == pub
    decoded = pyjwt.decode(
        token,
        _pubkey_pem(pub_bytes_on_disk),
        algorithms=["EdDSA"],
        audience=TOKEN_AUDIENCE,
    )
    assert decoded["sub"] == "cambia-harness"


def test_public_key_is_raw_32_bytes():
    """Exact invariant runnerd/authtoken.Load enforces (len == ed25519.PublicKeySize,
    raw bytes, not PEM/DER)."""
    _seed, pub = bootstrap.generate_ed25519_keypair()
    assert isinstance(pub, bytes)
    assert len(pub) == 32


def test_private_key_is_raw_32_byte_seed():
    """One of the three formats transport.load_ed25519_private_key accepts."""
    seed, _pub = bootstrap.generate_ed25519_keypair()
    assert isinstance(seed, bytes)
    assert len(seed) == 32


def test_private_key_written_0600(tmp_path):
    seed, _pub = bootstrap.generate_ed25519_keypair()
    priv_path = tmp_path / "jwt_ed25519"
    bootstrap.write_private_key(priv_path, seed)
    mode = stat.S_IMODE(priv_path.stat().st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# Refusal-without-force
# ---------------------------------------------------------------------------


def test_write_private_key_refuses_existing_without_force(tmp_path):
    seed, _pub = bootstrap.generate_ed25519_keypair()
    priv_path = tmp_path / "jwt_ed25519"
    bootstrap.write_private_key(priv_path, seed)
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.write_private_key(priv_path, seed, force=False)
    # --force overwrites cleanly.
    bootstrap.write_private_key(priv_path, seed, force=True)


def test_write_public_key_refuses_existing_without_force(tmp_path):
    _seed, pub = bootstrap.generate_ed25519_keypair()
    pub_path = tmp_path / "jwt_ed25519.pub"
    bootstrap.write_public_key(pub_path, pub)
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.write_public_key(pub_path, pub, force=False)


def test_write_harness_yaml_refuses_existing_without_force(tmp_path):
    cfg_path = tmp_path / "harness.yaml"
    bootstrap.write_harness_yaml(cfg_path, "a: 1\n")
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.write_harness_yaml(cfg_path, "a: 2\n", force=False)


def test_run_init_refuses_existing_config_before_writing_anything(tmp_path):
    """A pre-existing config (but no keys yet) must refuse BEFORE any file is
    written -- no partial key pair left behind by a later-stage failure."""
    priv = tmp_path / "jwt_ed25519"
    pub = tmp_path / "jwt_ed25519.pub"
    cfg = tmp_path / "harness.yaml"
    cfg.write_text("preexisting: true\n")

    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.run_init(
            private_key_path=priv, public_key_path=pub, config_path=cfg, force=False
        )
    assert not priv.exists()
    assert not pub.exists()
    assert cfg.read_text() == "preexisting: true\n"  # untouched


def test_run_init_refuses_existing_private_key_before_writing_config(tmp_path):
    priv = tmp_path / "jwt_ed25519"
    pub = tmp_path / "jwt_ed25519.pub"
    cfg = tmp_path / "harness.yaml"
    priv.write_bytes(b"x" * 32)

    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.run_init(
            private_key_path=priv, public_key_path=pub, config_path=cfg, force=False
        )
    assert not pub.exists()
    assert not cfg.exists()


def test_run_init_force_overwrites_everything(tmp_path):
    priv = tmp_path / "jwt_ed25519"
    pub = tmp_path / "jwt_ed25519.pub"
    cfg = tmp_path / "harness.yaml"
    priv.write_bytes(b"stale")
    pub.write_bytes(b"stale")
    cfg.write_text("stale: true\n")

    result = bootstrap.run_init(
        private_key_path=priv, public_key_path=pub, config_path=cfg, force=True
    )
    assert result.private_key_path == priv
    assert len(priv.read_bytes()) == 32
    assert len(pub.read_bytes()) == 32
    assert "stale" not in cfg.read_text()


# ---------------------------------------------------------------------------
# Config scaffold correctness
# ---------------------------------------------------------------------------


def test_render_harness_yaml_always_pins_private_key_path():
    template = bootstrap.TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = bootstrap.render_harness_yaml(
        template, private_key_path="/tmp/somewhere/jwt_ed25519"
    )
    data = yaml.safe_load(rendered)
    assert data["auth"]["private_key_path"] == "/tmp/somewhere/jwt_ed25519"


def test_render_harness_yaml_applies_provided_flags():
    template = bootstrap.TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = bootstrap.render_harness_yaml(
        template,
        private_key_path="/tmp/k",
        runner_url="https://10.1.2.3:8090",
        ssh_target="myrunner",
        mirror_url="cambia@myrunner:/srv/cambia/mirror.git",
    )
    data = yaml.safe_load(rendered)
    assert data["runner"]["url"] == "https://10.1.2.3:8090"
    assert data["data_plane"]["ssh_alias"] == "myrunner"
    assert data["data_plane"]["origin_host"] == "myrunner"
    assert (
        data["data_plane"]["mirror_remote_url"]
        == "cambia@myrunner:/srv/cambia/mirror.git"
    )


def test_render_harness_yaml_leaves_template_placeholders_when_omitted():
    template = bootstrap.TEMPLATE_PATH.read_text(encoding="utf-8")
    template_data = yaml.safe_load(template)
    rendered = bootstrap.render_harness_yaml(template, private_key_path="/tmp/k")
    data = yaml.safe_load(rendered)
    # Nothing but private_key_path (and the parse-safety of the file) changes.
    assert data["runner"]["url"] == template_data["runner"]["url"]
    assert data["data_plane"]["ssh_alias"] == template_data["data_plane"]["ssh_alias"]
    assert (
        data["data_plane"]["mirror_remote_url"]
        == template_data["data_plane"]["mirror_remote_url"]
    )


def test_render_harness_yaml_missing_template_key_raises():
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.render_harness_yaml(
            "runner:\n  no_such_key: x\n", private_key_path="/tmp/k"
        )


def test_run_init_produces_config_matching_the_real_config_schema(tmp_path):
    """The scaffolded config must parse cleanly through the real config.py
    from_dict schema (field names, types, required keys). Full .validate() is
    NOT exercised here: cert_fingerprint is never auto-filled (no flag for it;
    the runner isn't deployed yet at init time), so validate() correctly
    rejects the placeholder until the user pastes the real fingerprint -- that
    is expected behavior, not a bug under test."""
    import yaml as yamlmod

    from src.harness import config as cfgmod

    priv = tmp_path / "jwt_ed25519"
    pub = tmp_path / "jwt_ed25519.pub"
    cfg_path = tmp_path / "harness.yaml"

    bootstrap.run_init(
        runner_url="https://10.0.0.5:8090",
        ssh_target="myrunner",
        mirror_url="cambia@myrunner:/srv/cambia/mirror.git",
        private_key_path=priv,
        public_key_path=pub,
        config_path=cfg_path,
    )

    raw = yamlmod.safe_load(cfg_path.read_text())
    cfg = cfgmod.from_dict(raw, source_path=str(cfg_path))
    assert cfg.runner.url == "https://10.0.0.5:8090"
    assert cfg.auth.private_key_path == str(priv)
    assert cfg.data_plane.ssh_alias == "myrunner"
    assert cfg.data_plane.origin_host == "myrunner"
    assert cfg.data_plane.mirror_remote_url == "cambia@myrunner:/srv/cambia/mirror.git"


# ---------------------------------------------------------------------------
# `cambia harness init` end-to-end via the actual CLI, temp HOME only
# ---------------------------------------------------------------------------


def test_harness_init_cli_end_to_end_against_temp_home(tmp_path, monkeypatch):
    """Runs the real `cambia harness init` command through typer's CliRunner.
    HOME/XDG are pinned to a temp dir for the whole invocation so this test can
    never write to the real ~/.config."""
    from typer.testing import CliRunner

    from src.harness.cli import harness_app

    home = tmp_path / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    }

    result = CliRunner().invoke(
        harness_app,
        [
            "init",
            "--runner-url",
            "https://10.9.9.9:8090",
            "--ssh-target",
            "myrunner",
            "--mirror-url",
            "cambia@myrunner:/srv/cambia/mirror.git",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output

    priv_path = home / ".config" / "cambia" / "jwt_ed25519"
    pub_path = home / ".config" / "cambia" / "jwt_ed25519.pub"
    cfg_path = home / ".config" / "cambia" / "harness.yaml"
    assert priv_path.exists()
    assert pub_path.exists()
    assert cfg_path.exists()
    assert stat.S_IMODE(priv_path.stat().st_mode) == 0o600

    data = yaml.safe_load(cfg_path.read_text())
    assert data["runner"]["url"] == "https://10.9.9.9:8090"
    assert data["auth"]["private_key_path"] == str(priv_path)

    # Second run without --force refuses and leaves everything intact.
    result2 = CliRunner().invoke(harness_app, ["init"], env=env)
    assert result2.exit_code != 0
    assert "already exists" in result2.output
