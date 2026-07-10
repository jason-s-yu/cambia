"""
src/harness/bootstrap.py

`cambia harness init` support: bootstraps a first-time client (cambia-256).
Generates the ed25519 signing keypair and scaffolds ~/.config/cambia/harness.yaml
from the checked-in template, in the exact on-disk formats the rest of the
harness client and the runner already consume:

  - private key: a raw 32-byte ed25519 seed, one of the three formats
    src.harness.transport.load_ed25519_private_key accepts, written chmod 0600.
  - public key: raw 32 bytes, the exact format runnerd/authtoken.Load requires;
    the deploy flow ships this file to the runner verbatim.

Non-interactive: every value is a flag or a template placeholder, never a prompt.
"""

import os
import re
from pathlib import Path
from typing import NamedTuple, Optional

# cfr/config/harness.example.yaml, resolved from this file (mirrors config.py's
# repo-relative fallback resolution). Not HOME-dependent, so safe to fix at
# import time unlike the ~/.config/cambia paths below.
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config" / "harness.example.yaml"


def default_config_dir() -> Path:
    """~/.config/cambia, expanded fresh on every call.

    Deliberately NOT a module-level constant: Path.expanduser() reads $HOME at
    call time, and run_init()'s path defaults must resolve against whatever
    HOME is active when init actually runs (including a test's temp HOME), not
    whatever HOME happened to be set when this module was first imported.
    """
    return Path("~/.config/cambia").expanduser()


class BootstrapError(Exception):
    """A precondition for `harness init` was not met, or the template is malformed."""


class InitResult(NamedTuple):
    private_key_path: Path
    public_key_path: Path
    config_path: Path


def generate_ed25519_keypair() -> "tuple[bytes, bytes]":
    """Return (private_seed_32, public_raw_32) for a fresh ed25519 keypair."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return seed, pub


def _refuse_unless_force(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise BootstrapError(f"{path} already exists (use --force to overwrite)")


def write_private_key(path: Path, seed: bytes, force: bool = True) -> None:
    """Write the raw 32-byte ed25519 seed, chmod 0600 from creation (not
    write-then-chmod, which would briefly expose the key at the process umask)."""
    _refuse_unless_force(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, seed)
    finally:
        os.close(fd)


def write_public_key(path: Path, pub: bytes, force: bool = True) -> None:
    """Write the raw 32-byte public key verbatim (the format runnerd/authtoken
    reads and the deploy flow ships as-is)."""
    _refuse_unless_force(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pub)


def _set_yaml_scalar(text: str, key: str, value: str) -> str:
    """Replace a top-level-unique `key: <value>` line's value, keeping the key
    and its indentation. Raises if the template does not contain `key` (a
    template edit that drops a key should fail init loudly, not scaffold a
    silently-incomplete config)."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}:).*$", re.MULTILINE)
    if not pattern.search(text):
        raise BootstrapError(
            f"harness.example.yaml template is missing expected key {key!r}"
        )
    return pattern.sub(lambda m: f"{m.group(1)} {value}", text, count=1)


def render_harness_yaml(
    template_text: str,
    *,
    private_key_path: Path,
    runner_url: Optional[str] = None,
    ssh_target: Optional[str] = None,
    mirror_url: Optional[str] = None,
) -> str:
    """Scaffold harness.yaml text from the template.

    private_key_path is always pinned to wherever `harness init` actually wrote
    the key (not a placeholder). Any of runner_url/ssh_target/mirror_url that is
    omitted leaves the template's own placeholder value in place -- the template
    already documents what to fill in and how (design 5.1's fingerprint command,
    the RFC 5737 example IP), so there is nothing to improve on by inventing a
    second placeholder convention here.
    """
    text = _set_yaml_scalar(template_text, "private_key_path", str(private_key_path))
    if runner_url:
        text = _set_yaml_scalar(text, "url", runner_url)
    if ssh_target:
        text = _set_yaml_scalar(text, "ssh_alias", ssh_target)
        text = _set_yaml_scalar(text, "origin_host", ssh_target)
    if mirror_url:
        text = _set_yaml_scalar(text, "mirror_remote_url", mirror_url)
    return text


def write_harness_yaml(path: Path, text: str, force: bool = True) -> None:
    _refuse_unless_force(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_init(
    *,
    runner_url: Optional[str] = None,
    ssh_target: Optional[str] = None,
    mirror_url: Optional[str] = None,
    force: bool = False,
    private_key_path: Optional[Path] = None,
    public_key_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    template_path: Path = TEMPLATE_PATH,
) -> InitResult:
    """End-to-end `harness init`.

    All three overwrite checks and the config render (which also validates the
    template) run BEFORE anything is written, so a refusal or a malformed
    template never leaves a half-written key pair or config behind.

    private_key_path/public_key_path/config_path default to fresh
    default_config_dir() lookups when omitted (real usage); tests pass explicit
    temp-dir paths so nothing ever touches the real ~/.config/cambia.
    """
    config_dir = default_config_dir()
    private_key_path = private_key_path or (config_dir / "jwt_ed25519")
    public_key_path = public_key_path or (config_dir / "jwt_ed25519.pub")
    config_path = config_path or (config_dir / "harness.yaml")

    _refuse_unless_force(private_key_path, force)
    _refuse_unless_force(public_key_path, force)
    _refuse_unless_force(config_path, force)

    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError(f"failed to read template {template_path}: {exc}") from exc

    rendered = render_harness_yaml(
        template_text,
        private_key_path=private_key_path,
        runner_url=runner_url,
        ssh_target=ssh_target,
        mirror_url=mirror_url,
    )

    seed, pub = generate_ed25519_keypair()
    write_private_key(private_key_path, seed, force=True)
    write_public_key(public_key_path, pub, force=True)
    write_harness_yaml(config_path, rendered, force=True)

    return InitResult(
        private_key_path=private_key_path,
        public_key_path=public_key_path,
        config_path=config_path,
    )
