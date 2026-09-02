"""Tests for the nashnet enrollment grant mint (cambia-1711).

The last test writes the golden fixture the Go verifier's test reads
(runnerd/authtoken/testdata/nashnet/), so a change to either side that breaks
the pairing fails in one of the two suites rather than at deploy time.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

jwt = pytest.importorskip("jwt")
ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
from cryptography.hazmat.primitives.serialization import (  # noqa: E402
    Encoding,
    PublicFormat,
)

from src.harness.nashnet import (  # noqa: E402
    DEFAULT_GRANT_LIFETIME,
    DEFAULT_GRANT_LIFETIME_DAYS,
    GRANT_AUDIENCE,
    NODE_SUBJECT_PREFIX,
    NashnetError,
    derive_node_id,
    grant_filename,
    load_node_public_key,
    mint_grant,
    parse_duration,
)

# Deterministic test-only keys. Seeds are hashes of fixed strings so the golden
# fixture is reproducible from this file alone and no real key is checked in.
OPERATOR_SEED = hashlib.sha256(b"cambia nashnet golden operator").digest()
NODE_A_SEED = hashlib.sha256(b"cambia nashnet golden node-a").digest()

# Pinned mint time for the golden grant. The Go test verifies against an
# injected clock inside this window, so neither suite depends on the wall clock.
GOLDEN_IAT = datetime(2026, 9, 1, tzinfo=timezone.utc)

GOLDEN_CAPS = {
    "max_slots": 2,
    "kinds": ["train", "evaluate", "measure"],
    "device_kinds": ["cuda", "cpu"],
    "max_vram_gb": {"cuda": 24.0},
    "max_cores": 16,
    "max_ram_gb": 64.0,
    "max_disk_gb": 512.0,
    "labels": ["gpu"],
    "max_lease_bytes": 68719476736,
}

GOLDEN_DIR = (
    Path(__file__).resolve().parents[2] / "runnerd" / "authtoken" / "testdata" / "nashnet"
)


def _key(seed: bytes):
    return ed25519.Ed25519PrivateKey.from_private_bytes(seed)


def _raw_public(private_key) -> bytes:
    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def test_derive_node_id_is_the_key_hash():
    pub = _raw_public(_key(NODE_A_SEED))
    assert derive_node_id(pub) == "n-" + hashlib.sha256(pub).hexdigest()[:12]
    assert grant_filename(derive_node_id(pub)).endswith(".grant")


def test_derive_node_id_rejects_a_wrong_sized_key():
    with pytest.raises(NashnetError):
        derive_node_id(b"too-short")


def test_mint_grant_carries_the_enrollment_claims():
    op = _key(OPERATOR_SEED)
    node_pub = _raw_public(_key(NODE_A_SEED))
    token = mint_grant(op, node_pub, caps=GOLDEN_CAPS, now=GOLDEN_IAT)
    claims = jwt.decode(
        token,
        op.public_key(),
        algorithms=["EdDSA"],
        audience=GRANT_AUDIENCE,
    )
    node_id = derive_node_id(node_pub)
    assert claims["sub"] == NODE_SUBJECT_PREFIX + node_id
    assert claims["caps"] == GOLDEN_CAPS
    assert claims["exp"] - claims["iat"] == int(DEFAULT_GRANT_LIFETIME.total_seconds())
    assert DEFAULT_GRANT_LIFETIME_DAYS == 90


def test_mint_grant_refuses_an_unknown_cap_field():
    op = _key(OPERATOR_SEED)
    node_pub = _raw_public(_key(NODE_A_SEED))
    with pytest.raises(NashnetError, match="unknown cap"):
        mint_grant(op, node_pub, caps={"max_slot": 2})


def test_mint_grant_refuses_a_nonpositive_lifetime():
    op = _key(OPERATOR_SEED)
    node_pub = _raw_public(_key(NODE_A_SEED))
    with pytest.raises(NashnetError, match="lifetime must be positive"):
        mint_grant(op, node_pub, lifetime=timedelta(0))


def test_parse_duration_covers_the_cli_flag_forms():
    assert parse_duration("90d") == timedelta(days=90)
    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("45s") == timedelta(seconds=45)
    for bad in ("90", "0d", "-1d", "90w", ""):
        with pytest.raises(NashnetError):
            parse_duration(bad)


def test_load_node_public_key_accepts_raw_and_base64(tmp_path):
    import base64

    pub = _raw_public(_key(NODE_A_SEED))
    raw_path = tmp_path / "node.pub"
    raw_path.write_bytes(pub)
    assert load_node_public_key(raw_path) == pub

    b64_path = tmp_path / "node.pub.b64"
    b64_path.write_text(base64.urlsafe_b64encode(pub).decode() + "\n")
    assert load_node_public_key(b64_path) == pub

    bad = tmp_path / "bad.pub"
    bad.write_text("not a key\n")
    with pytest.raises(NashnetError):
        load_node_public_key(bad)


@pytest.mark.skipif(
    not GOLDEN_DIR.parents[1].exists(),
    reason="runnerd/authtoken is not present in this checkout",
)
def test_golden_grant_fixture_is_current():
    """Mint the fixture the Go loader verifies, and hold it byte-stable.

    Set CAMBIA_UPDATE_GOLDEN=1 to rewrite it after a deliberate change to the
    claim set or a JWT-library encoding change; the Go side asserts the fixture
    still verifies and still derives the same node id.
    """
    op = _key(OPERATOR_SEED)
    node_key = _key(NODE_A_SEED)
    node_pub = _raw_public(node_key)
    node_id = derive_node_id(node_pub)
    token = mint_grant(op, node_pub, caps=GOLDEN_CAPS, now=GOLDEN_IAT)
    meta = {
        "note": (
            "Minted by cfr/tests/test_harness_nashnet.py. Test-only keys, "
            "deterministic seeds; regenerate with CAMBIA_UPDATE_GOLDEN=1."
        ),
        "node_name": "node-a",
        "node_id": node_id,
        "subject": NODE_SUBJECT_PREFIX + node_id,
        "grant_file": grant_filename(node_id),
        "operator_pubkey_hex": _raw_public(op).hex(),
        "node_seed_hex": NODE_A_SEED.hex(),
        "node_pubkey_hex": node_pub.hex(),
        "iat": int(GOLDEN_IAT.timestamp()),
        "exp": int((GOLDEN_IAT + DEFAULT_GRANT_LIFETIME).timestamp()),
        "lifetime_seconds": int(DEFAULT_GRANT_LIFETIME.total_seconds()),
        "caps": GOLDEN_CAPS,
    }
    files = {
        grant_filename(node_id): token + "\n",
        "golden.json": json.dumps(meta, indent=2, sort_keys=True) + "\n",
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = GOLDEN_DIR / name
        if os.environ.get("CAMBIA_UPDATE_GOLDEN") or not path.exists():
            path.write_text(content)
            continue
        assert path.read_text() == content, (
            f"{path} is stale; rerun with CAMBIA_UPDATE_GOLDEN=1 if the change "
            f"to the grant claims is deliberate"
        )
