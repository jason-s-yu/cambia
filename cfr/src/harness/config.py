"""
src/harness/config.py

Client-side harness config (cambia-256, design 5.1/5.2). Holds the runner control
plane URL, its pinned self-signed TLS SHA256 fingerprint, the ed25519 signing
key path (private half never leaves the client), the git mirror remote, the
ssh/rsync data-plane target, and the pull interval.

Resolution order (first that exists wins):
  1. an explicit path (CLI --config / load(path=...))
  2. $CAMBIA_HARNESS_CONFIG
  3. ~/.config/cambia/harness.yaml   (documented default; holds key paths, host-local)
  4. <repo>/cfr/config/harness.yaml  (checked-in override, optional)

A committed template lives at cfr/config/harness.example.yaml. When no config is
found, load() raises HarnessConfigError listing every path it looked at.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_USER_CONFIG = Path("~/.config/cambia/harness.yaml").expanduser()
_MAX_TOKEN_TTL_SECONDS = 3600  # design 5.2: short-lived, <= 1h


class HarnessConfigError(Exception):
    """The harness config is missing, malformed, or internally inconsistent."""


@dataclass
class RunnerConfig:
    url: str  # https://<runner-host>:8090
    cert_fingerprint: str  # sha256 hex of the runner's self-signed cert (5.1)


@dataclass
class AuthConfig:
    private_key_path: str  # ed25519 private key, client-only 0600 (5.2)
    subject: str = "cambia-harness"  # JWT sub claim
    token_ttl_seconds: int = 900  # <= 3600


@dataclass
class DataPlaneConfig:
    ssh_alias: str  # rsync/ssh target, e.g. "runner"
    runner_runs_dir: str  # remote runsDir, e.g. /srv/cambia/runs
    mirror_remote_url: str  # e.g. cambia@runner:/srv/cambia/mirror.git
    origin_host: str  # stamped onto reconciled runs, e.g. "runner"


@dataclass
class SyncConfig:
    interval_seconds: int = 60
    local_runs_dir: str = "runs"  # where synced dirs land (relative to cfr/)


@dataclass
class HarnessConfig:
    runner: RunnerConfig
    auth: AuthConfig
    data_plane: DataPlaneConfig
    sync: SyncConfig
    source_path: Optional[str] = None

    def normalized_fingerprint(self) -> str:
        """Return the pinned cert fingerprint as lowercase hex, colons stripped."""
        fp = self.runner.cert_fingerprint.replace(":", "").strip().lower()
        if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
            raise HarnessConfigError(
                "runner.cert_fingerprint must be a SHA256 hex string (64 hex chars, "
                f"optionally colon-separated), got {self.runner.cert_fingerprint!r}"
            )
        return fp

    def validate(self) -> "HarnessConfig":
        if not self.runner.url.startswith("https://"):
            raise HarnessConfigError(
                f"runner.url must be https:// (plaintext is refused), got {self.runner.url!r}"
            )
        self.normalized_fingerprint()  # raises on a malformed pin
        if self.auth.token_ttl_seconds <= 0:
            raise HarnessConfigError("auth.token_ttl_seconds must be positive")
        if self.auth.token_ttl_seconds > _MAX_TOKEN_TTL_SECONDS:
            raise HarnessConfigError(
                f"auth.token_ttl_seconds must be <= {_MAX_TOKEN_TTL_SECONDS} (design 5.2)"
            )
        if self.sync.interval_seconds <= 0:
            raise HarnessConfigError("sync.interval_seconds must be positive")
        return self


def _candidate_paths(explicit: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    env = os.environ.get("CAMBIA_HARNESS_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(DEFAULT_USER_CONFIG)
    # repo-relative fallback: cfr/config/harness.yaml, resolved from this file.
    repo_cfg = Path(__file__).resolve().parents[2] / "config" / "harness.yaml"
    paths.append(repo_cfg)
    return paths


def load(path: Optional[str] = None) -> HarnessConfig:
    """Load and validate the harness config from the first path that exists.

    Raises HarnessConfigError with the searched paths if none is found, or on a
    malformed / incomplete file.
    """
    import yaml

    candidates = _candidate_paths(path)
    chosen: Optional[Path] = next((p for p in candidates if p.is_file()), None)
    if chosen is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise HarnessConfigError(
            "no harness config found. Copy cfr/config/harness.example.yaml to "
            f"{DEFAULT_USER_CONFIG} and fill it in. Searched:\n  {searched}"
        )

    try:
        with open(chosen, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise HarnessConfigError(f"failed to read {chosen}: {exc}") from exc

    if not isinstance(data, dict):
        raise HarnessConfigError(f"harness config {chosen} must be a mapping")

    return from_dict(data, source_path=str(chosen)).validate()


def from_dict(data: dict, source_path: Optional[str] = None) -> HarnessConfig:
    """Build a HarnessConfig from a plain mapping (also the test entry point)."""

    def _require(section: dict, key: str, where: str):
        if key not in section or section[key] in (None, ""):
            raise HarnessConfigError(f"harness config missing required {where}.{key}")
        return section[key]

    runner = data.get("runner") or {}
    auth = data.get("auth") or {}
    data_plane = data.get("data_plane") or {}
    sync = data.get("sync") or {}
    if (
        not isinstance(runner, dict)
        or not isinstance(auth, dict)
        or not isinstance(data_plane, dict)
    ):
        raise HarnessConfigError("runner/auth/data_plane sections must be mappings")

    runner_cfg = RunnerConfig(
        url=_require(runner, "url", "runner"),
        cert_fingerprint=_require(runner, "cert_fingerprint", "runner"),
    )
    auth_cfg = AuthConfig(
        private_key_path=str(
            Path(_require(auth, "private_key_path", "auth")).expanduser()
        ),
        subject=auth.get("subject", "cambia-harness"),
        token_ttl_seconds=int(auth.get("token_ttl_seconds", 900)),
    )
    dp_cfg = DataPlaneConfig(
        ssh_alias=_require(data_plane, "ssh_alias", "data_plane"),
        runner_runs_dir=_require(data_plane, "runner_runs_dir", "data_plane"),
        mirror_remote_url=_require(data_plane, "mirror_remote_url", "data_plane"),
        origin_host=data_plane.get("origin_host")
        or _require(data_plane, "ssh_alias", "data_plane"),
    )
    sync_cfg = SyncConfig(
        interval_seconds=int(sync.get("interval_seconds", 60)),
        local_runs_dir=str(sync.get("local_runs_dir", "runs")),
    )
    return HarnessConfig(
        runner=runner_cfg,
        auth=auth_cfg,
        data_plane=dp_cfg,
        sync=sync_cfg,
        source_path=source_path,
    )
