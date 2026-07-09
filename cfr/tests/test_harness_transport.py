"""
tests/test_harness_transport.py

Transport-layer coverage (cambia-256, design 5.1/5.2):
  - JWT mint claim/expiry shape (EdDSA, sub/aud/iat/nbf/exp, TTL cap)
  - TLS fingerprint pinning against a real localhost TLS server: correct fp
    accepted, wrong fp refused, plaintext refused.
  - WS log-stream pin-before-auth: the Bearer token never leaves the socket on a
    wrong-fingerprint peer (H2).
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.harness.transport import (
    NBF_BACKDATE_SECONDS,
    TOKEN_AUDIENCE,
    CertificatePinError,
    ControlPlaneTransport,
    InsecureSchemeError,
    TransportError,
    load_ed25519_private_key,
    mint_token,
    normalize_fingerprint,
    open_log_stream,
)
from tests.harness_tls_util import (
    RecordingServer,
    RecordingTLSSocketServer,
    make_self_signed,
)

# ---------------------------------------------------------------------------
# JWT mint (design 5.2)
# ---------------------------------------------------------------------------


def _new_ed25519(tmp_path, layout="pem"):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    path = tmp_path / f"key_{layout}"
    if layout == "pem":
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    elif layout == "seed32":
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        )
    elif layout == "go64":
        seed = key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        pub = key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        path.write_bytes(seed + pub)  # Go ed25519.PrivateKey layout
    return key, str(path)


def _pubkey_pem(key):
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def test_mint_token_claim_shape(tmp_path):
    import jwt

    key, path = _new_ed25519(tmp_path, "pem")
    loaded = load_ed25519_private_key(path)
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
    token = mint_token(loaded, subject="cambia-harness", ttl_seconds=900, now=now)

    # verify_exp disabled: the fixed `now` may predate the wall clock; this test
    # asserts claim SHAPE, not liveness (expiry is covered separately).
    decoded = jwt.decode(
        token,
        _pubkey_pem(key),
        algorithms=["EdDSA"],
        audience=TOKEN_AUDIENCE,
        options={"verify_exp": False},
    )
    assert decoded["sub"] == "cambia-harness"
    assert decoded["aud"] == TOKEN_AUDIENCE
    assert decoded["iat"] == int(now.timestamp())
    # nbf is backdated by the clock-skew guard so a runner whose clock trails
    # the mint host still accepts a fresh token.
    assert decoded["nbf"] == int(
        (now - timedelta(seconds=NBF_BACKDATE_SECONDS)).timestamp()
    )
    assert decoded["exp"] == int((now + timedelta(seconds=900)).timestamp())
    # EdDSA header
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "EdDSA"


def test_mint_token_audience_claim(tmp_path):
    """M3 (mint half): every token carries aud=cambia-runnerd. A verifier that
    checks the audience rejects a token minted for a different one."""
    import jwt

    key, path = _new_ed25519(tmp_path, "pem")
    loaded = load_ed25519_private_key(path)
    token = mint_token(loaded, subject="s", ttl_seconds=60)

    decoded = jwt.decode(
        token, _pubkey_pem(key), algorithms=["EdDSA"], audience=TOKEN_AUDIENCE
    )
    assert decoded["aud"] == TOKEN_AUDIENCE
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(
            token, _pubkey_pem(key), algorithms=["EdDSA"], audience="some-other-service"
        )


def test_mint_token_expiry_enforced(tmp_path):
    import jwt

    key, path = _new_ed25519(tmp_path, "pem")
    loaded = load_ed25519_private_key(path)
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    token = mint_token(loaded, subject="s", ttl_seconds=60, now=past)
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, _pubkey_pem(key), algorithms=["EdDSA"], audience=TOKEN_AUDIENCE)


def test_mint_token_ttl_capped(tmp_path):
    key, path = _new_ed25519(tmp_path, "pem")
    loaded = load_ed25519_private_key(path)
    with pytest.raises(TransportError):
        mint_token(loaded, subject="s", ttl_seconds=3601)
    with pytest.raises(TransportError):
        mint_token(loaded, subject="s", ttl_seconds=0)
    with pytest.raises(TransportError):
        mint_token(loaded, subject="", ttl_seconds=60)


@pytest.mark.parametrize("layout", ["pem", "seed32", "go64"])
def test_load_key_formats_produce_same_signing_key(tmp_path, layout):
    import jwt

    key, path = _new_ed25519(tmp_path, layout)
    loaded = load_ed25519_private_key(path)
    token = mint_token(loaded, subject="s", ttl_seconds=60)
    # The original key's public half must verify a token the loaded key signed.
    decoded = jwt.decode(
        token, _pubkey_pem(key), algorithms=["EdDSA"], audience=TOKEN_AUDIENCE
    )
    assert decoded["sub"] == "s"


def test_load_key_rejects_garbage(tmp_path):
    p = tmp_path / "bad"
    p.write_bytes(b"not-a-key")
    with pytest.raises(TransportError):
        load_ed25519_private_key(str(p))


# ---------------------------------------------------------------------------
# TLS fingerprint pinning (design 5.1)
# ---------------------------------------------------------------------------


def test_pin_correct_fingerprint_accepted(tmp_path):
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/health"): (200, {"ok": True})}
    with RecordingServer(cert, key, routes) as srv:
        t = ControlPlaneTransport(srv.base_url, fp)
        status, payload = t.request("GET", "/harness/health", token="tok")
    assert status == 200
    assert payload == {"ok": True}


def test_pin_wrong_fingerprint_refused(tmp_path):
    cert, key, _fp = make_self_signed(tmp_path)
    wrong = "ab" * 32  # valid-shaped but not the server's cert
    routes = {("GET", "/harness/health"): (200, {"ok": True})}
    with RecordingServer(cert, key, routes) as srv:
        t = ControlPlaneTransport(srv.base_url, wrong)
        with pytest.raises(CertificatePinError):
            t.request("GET", "/harness/health", token="tok")


def test_plaintext_scheme_refused(tmp_path):
    # A http:// control-plane URL is refused before any connection.
    with pytest.raises(InsecureSchemeError):
        ControlPlaneTransport("http://127.0.0.1:8090", "ab" * 32)


def test_plaintext_server_handshake_fails(tmp_path):
    # A plaintext server behind an https:// URL fails the TLS handshake (never a
    # silent plaintext exchange).
    cert, key, fp = make_self_signed(tmp_path)
    routes = {("GET", "/harness/health"): (200, {"ok": True})}
    with RecordingServer(None, None, routes) as srv:  # plaintext server
        https_url = srv.base_url.replace("http://", "https://")
        t = ControlPlaneTransport(https_url, fp)
        with pytest.raises(Exception) as exc:
            t.request("GET", "/harness/health", token="tok")
        assert not isinstance(exc.value, AssertionError)


def test_normalize_fingerprint():
    raw = "AA:BB:CC:DD" + ":00" * 28
    assert normalize_fingerprint(raw) == raw.replace(":", "").lower()
    with pytest.raises(TransportError):
        normalize_fingerprint("tooshort")
    with pytest.raises(TransportError):
        normalize_fingerprint("zz" * 32)


# ---------------------------------------------------------------------------
# WS log-stream pin-before-auth (H2, design 5.1)
# ---------------------------------------------------------------------------


def test_ws_pin_mismatch_sends_no_upgrade(tmp_path):
    """A wrong-fingerprint peer must be rejected AFTER the TLS handshake but
    BEFORE the WS upgrade (which carries the Bearer). The recording server proves
    the handshake completed yet zero application bytes arrived, so the token
    never left the client."""
    cert, key, _fp = make_self_signed(tmp_path)
    wrong = "cd" * 32  # valid-shaped fingerprint, not the server's cert
    with RecordingTLSSocketServer(cert, key) as srv:
        with pytest.raises(CertificatePinError):
            open_log_stream(
                srv.base_url, wrong, token="super-secret-bearer", job_id="job1"
            )
    # TLS completed (the client had to read the peer cert to check the pin) but
    # the client refused to send the upgrade request -> no Bearer on the wire.
    assert srv.handshake_ok.is_set()
    assert srv.received == []


def test_ws_pin_match_sends_upgrade_with_bearer(tmp_path):
    """On a matching pin the client proceeds to the WS upgrade, sending it over
    the already-verified TLS channel. The server replies with a non-101 so the
    handshake fails fast; we only assert the upgrade (with the Bearer) was
    transmitted after the pin passed."""
    cert, key, fp = make_self_signed(tmp_path)
    resp = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    with RecordingTLSSocketServer(cert, key, respond=resp) as srv:
        with pytest.raises(Exception) as exc:
            open_log_stream(srv.base_url, fp, token="super-secret-bearer", job_id="job1")
        # It failed on the WS handshake, NOT on the pin (the pin matched).
        assert not isinstance(exc.value, CertificatePinError)
    assert srv.handshake_ok.is_set()
    sent = b"".join(srv.received)
    assert b"GET /ws/harness/jobs/job1/logs" in sent
    assert b"Authorization: Bearer super-secret-bearer" in sent
