"""Client-side nashnet node enrollment: the grant mint (D25, D60).

Enrollment is an operator-signed grant, not a file's presence: the client mints
a compact JWS with its own ed25519 key (the same key operator tokens are signed
with, transport.mint_token) and the operator places it on the coordinator at
``<nodes_dir>/<node_id>.grant``. The coordinator verifies it and never signs, so
a write primitive on its key directory cannot enroll a node.

The node id is derived from the node's public key,
``"n-" + sha256(pubkey).hex()[:12]``, and the coordinator recomputes it at load
and refuses a file whose name disagrees. That makes the id-to-key binding
intrinsic and id squatting impossible, so ``grant_filename`` is the only correct
name for a minted grant.

The Go verifier is ``runnerd/authtoken`` (grants.go); the two sides are pinned
together by the golden fixture under ``runnerd/authtoken/testdata/nashnet/``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

# The grant audience. It is not the operator control-plane audience
# ("cambia-runnerd") and not the node call-token audience ("nashnet-node"): one
# key signs grants and operator tokens, so the audience is what keeps a grant
# from being replayed as a control-plane token.
GRANT_AUDIENCE = "nashnet-enroll"

# Subject format shared with authtoken.NodeIDFromSubject.
NODE_SUBJECT_PREFIX = "node:"
NODE_ID_PREFIX = "n-"
NODE_ID_HEX_LEN = 12

# Default grant validity, paired with authtoken.DefaultGrantLifetime, which is
# also the coordinator's default ceiling on exp - iat. Raising one without the
# other mints grants the coordinator refuses.
DEFAULT_GRANT_LIFETIME_DAYS = 90
DEFAULT_GRANT_LIFETIME = timedelta(days=DEFAULT_GRANT_LIFETIME_DAYS)

# The capability caps a grant may carry (D47). Every field placement reads is
# clamped by one of these; the Go loader decodes the object strictly, so an
# unrecognized key fails the load rather than passing unclamped, and this set is
# checked here to fail at mint time instead.
CAP_FIELDS = frozenset(
    {
        "max_slots",
        "kinds",
        "device_kinds",
        "max_vram_gb",
        "max_cores",
        "max_ram_gb",
        "max_disk_gb",
        "labels",
        "max_lease_bytes",
    }
)

_DURATION_RE = re.compile(r"^(\d+)([smhd])$")
_DURATION_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


class NashnetError(Exception):
    """A grant could not be minted from the given inputs."""


def parse_duration(text: str) -> timedelta:
    """Parse a ``90d`` / ``12h`` / ``30m`` / ``45s`` duration.

    The grant lifetime is an operator setting, so it is expressed the way the
    CLI flag is written rather than as a raw number of seconds.
    """
    m = _DURATION_RE.match(text.strip())
    if not m:
        raise NashnetError(f"duration must be <n>[s|m|h|d], got {text!r}")
    value, unit = int(m.group(1)), m.group(2)
    if value <= 0:
        raise NashnetError(f"duration must be positive, got {text!r}")
    return timedelta(**{_DURATION_UNITS[unit]: value})


def derive_node_id(node_pubkey: bytes) -> str:
    """Return the node id a raw 32-byte ed25519 public key is entitled to."""
    if len(node_pubkey) != 32:
        raise NashnetError(
            f"node public key must be 32 raw bytes, got {len(node_pubkey)}"
        )
    digest = hashlib.sha256(node_pubkey).hexdigest()
    return NODE_ID_PREFIX + digest[:NODE_ID_HEX_LEN]


def grant_filename(node_id: str) -> str:
    """Return the only file name a grant for node_id may be placed under."""
    return f"{node_id}.grant"


def load_node_public_key(path: str | Path) -> bytes:
    """Read a node's public key as raw bytes.

    Accepts the two shapes ``cambia-runnerd node init`` can hand an operator:
    the raw 32-byte ed25519 public key, and the same key base64-encoded (with or
    without padding, standard or url-safe alphabet) as a single text line. The
    private half never leaves the node, so nothing here loads one.
    """
    data = Path(path).expanduser().read_bytes()
    if len(data) == 32:
        return data
    text = data.strip()
    for decode in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            raw = decode(text + b"=" * (-len(text) % 4))
        except (binascii.Error, ValueError):
            continue
        if len(raw) == 32:
            return raw
    raise NashnetError(
        f"unrecognized node public key at {path}: expected 32 raw bytes or "
        f"base64 of 32 bytes, got {len(data)} bytes"
    )


def _validate_caps(caps: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if caps is None:
        return {}
    unknown = sorted(set(caps) - CAP_FIELDS)
    if unknown:
        raise NashnetError(
            f"unknown cap field(s) {unknown}; allowed: {sorted(CAP_FIELDS)}"
        )
    return dict(caps)


def mint_grant(
    private_key,
    node_pubkey: bytes,
    *,
    caps: Optional[Mapping[str, Any]] = None,
    lifetime: Optional[timedelta] = None,
    now: Optional[datetime] = None,
) -> str:
    """Mint an enrollment grant for a node public key.

    Args:
        private_key: the operator's cryptography Ed25519PrivateKey, loaded by
            transport.load_ed25519_private_key. The coordinator verifies the
            grant with the public half it already holds.
        node_pubkey: the node's raw 32-byte ed25519 public key.
        caps: the clamp set (D47); an omitted cap lets the declared value pass
            unclamped, which is the enrolling operator's explicit choice.
        lifetime: grant validity, default DEFAULT_GRANT_LIFETIME. The
            coordinator refuses a lifetime past its own ceiling, whose default
            is the same value.
        now: injectable clock for tests.

    Returns:
        The compact JWS to place at grant_filename(derive_node_id(node_pubkey)).
    """
    import jwt

    node_id = derive_node_id(node_pubkey)
    if lifetime is None:
        lifetime = DEFAULT_GRANT_LIFETIME
    if lifetime <= timedelta(0):
        raise NashnetError(f"grant lifetime must be positive, got {lifetime}")
    issued = now or datetime.now(timezone.utc)
    claims = {
        "aud": GRANT_AUDIENCE,
        "sub": NODE_SUBJECT_PREFIX + node_id,
        "node_pubkey": base64.urlsafe_b64encode(node_pubkey).rstrip(b"=").decode(),
        "caps": _validate_caps(caps),
        "iat": int(issued.timestamp()),
        "exp": int((issued + lifetime).timestamp()),
    }
    return jwt.encode(claims, private_key, algorithm="EdDSA")
