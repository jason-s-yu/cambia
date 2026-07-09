"""
src/harness/transport.py

Control-plane transport for the client `harness` CLI (cambia-256, design
5.1/5.2).

TLS (5.1): the runner serves a self-signed cert. The client pins it by SHA256
fingerprint from config: an SSL context with CERT_NONE (no CA chain, no hostname
check) plus a manual SHA256 check of the peer leaf cert's DER against the pin.
Plaintext is refused (https:// scheme only; a plaintext peer fails the
handshake). A CA-bundle path is deliberately not used: pinning one self-signed
cert is tighter than trusting a CA that could issue others.

JWT (5.2): short-lived EdDSA tokens are minted per invocation from the ed25519
private key at a client-local path. The private key never leaves the client and tokens
are never persisted. The runner loads only the public half and verifies exactly
as the Go auth package (SigningMethodEdDSA, "sub" required). Claims: sub, iat,
nbf, exp; exp is bounded by the config TTL (<= 1h).
"""

import hashlib
import hmac
import http.client
import json
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

# Hard ceiling mirrored from config._MAX_TOKEN_TTL_SECONDS (design 5.2). Kept
# here too so mint_token is safe even if called with a raw ttl.
MAX_TOKEN_TTL_SECONDS = 3600


class TransportError(Exception):
    """Base class for transport-layer failures."""


class CertificatePinError(TransportError):
    """The peer cert's SHA256 fingerprint did not match the pinned value."""


class InsecureSchemeError(TransportError):
    """A non-https URL was supplied; plaintext control-plane traffic is refused."""


# ---------------------------------------------------------------------------
# JWT minting (design 5.2)
# ---------------------------------------------------------------------------


def load_ed25519_private_key(path: str):
    """Load an ed25519 private key from a client-local path.

    Accepts, in order of detection:
      - PEM (PKCS8 "-----BEGIN ... KEY-----")
      - 32 raw bytes (the ed25519 seed)
      - 64 raw bytes (Go `ed25519.PrivateKey` layout: seed || public; the Go
        auth package writes this via os.WriteFile of the raw key) -> first 32
        bytes are the seed
    Returns a cryptography Ed25519PrivateKey. The key bytes never leave this
    process.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    data = Path(path).expanduser().read_bytes()
    if data[:11] == b"-----BEGIN ":
        return load_pem_private_key(data, password=None)
    if len(data) == 32:
        return Ed25519PrivateKey.from_private_bytes(data)
    if len(data) == 64:
        # Go ed25519.PrivateKey is seed(32) || public(32); the seed alone yields
        # the identical signing key.
        return Ed25519PrivateKey.from_private_bytes(data[:32])
    raise TransportError(
        f"unrecognized ed25519 private key at {path}: expected PEM, 32, or 64 raw "
        f"bytes, got {len(data)} bytes"
    )


def mint_token(
    private_key,
    subject: str,
    ttl_seconds: int,
    now: Optional[datetime] = None,
) -> str:
    """Mint a short-lived EdDSA JWT for one control-plane invocation.

    Args:
        private_key: a cryptography Ed25519PrivateKey (from load_ed25519_private_key).
        subject: the "sub" claim; the runner requires a non-empty sub.
        ttl_seconds: token lifetime; must be in (0, 3600].
        now: injectable clock for tests.
    """
    import jwt

    if not subject:
        raise TransportError("JWT subject (sub) must be non-empty")
    if ttl_seconds <= 0 or ttl_seconds > MAX_TOKEN_TTL_SECONDS:
        raise TransportError(
            f"token ttl must be in (0, {MAX_TOKEN_TTL_SECONDS}], got {ttl_seconds}"
        )
    issued = now or datetime.now(timezone.utc)
    exp = issued + timedelta(seconds=ttl_seconds)
    claims = {
        "sub": subject,
        "iat": int(issued.timestamp()),
        "nbf": int(issued.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(claims, private_key, algorithm="EdDSA")


# ---------------------------------------------------------------------------
# TLS fingerprint pinning (design 5.1)
# ---------------------------------------------------------------------------


def normalize_fingerprint(fp: str) -> str:
    """Lowercase hex, colons stripped; validates a 64-char SHA256 hex string."""
    norm = fp.replace(":", "").strip().lower()
    if len(norm) != 64 or any(c not in "0123456789abcdef" for c in norm):
        raise TransportError(f"cert fingerprint must be SHA256 hex, got {fp!r}")
    return norm


def sha256_fingerprint(der_bytes: bytes) -> str:
    """SHA256 hex of a DER-encoded cert (matches openssl -fingerprint -sha256)."""
    return hashlib.sha256(der_bytes).hexdigest()


def _pinned_ssl_context() -> ssl.SSLContext:
    """An SSL context that skips CA + hostname checks so the fingerprint pin is
    the sole trust anchor (design 5.1)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _verify_peer(sock: ssl.SSLSocket, pinned: str) -> None:
    der = sock.getpeercert(binary_form=True)
    if not der:
        raise CertificatePinError("peer presented no certificate")
    actual = sha256_fingerprint(der)
    if not hmac.compare_digest(actual, pinned):
        raise CertificatePinError(
            f"peer cert fingerprint {actual} does not match pinned {pinned}"
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that checks the peer cert fingerprint right after the
    handshake and before any request bytes are sent."""

    def __init__(self, host: str, port: int, pinned_fp: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=_pinned_ssl_context())
        self._pinned_fp = pinned_fp

    def connect(self) -> None:  # type: ignore[override]
        super().connect()
        _verify_peer(self.sock, self._pinned_fp)


class ControlPlaneTransport:
    """Minimal JSON-over-HTTPS transport to the runner control plane.

    Every request opens a fresh pinned connection and attaches a Bearer token.
    The token is provided by the caller (minted per CLI invocation, never
    persisted).
    """

    def __init__(self, base_url: str, fingerprint: str, timeout: float = 30.0):
        parsed = urlparse(base_url)
        if parsed.scheme != "https":
            raise InsecureSchemeError(
                f"control-plane URL must be https://, got {base_url!r}"
            )
        if not parsed.hostname:
            raise TransportError(f"control-plane URL has no host: {base_url!r}")
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._fp = normalize_fingerprint(fingerprint)
        self._timeout = timeout

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def new_connection(self) -> _PinnedHTTPSConnection:
        return _PinnedHTTPSConnection(self._host, self._port, self._fp, self._timeout)

    def request(
        self,
        method: str,
        path: str,
        token: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any]:
        """Issue one request; returns (status_code, parsed_json_or_text)."""
        conn = self.new_connection()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        data: Optional[bytes] = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        finally:
            conn.close()
        text = raw.decode("utf-8", errors="replace")
        parsed: Any = text
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
        return status, parsed


def open_log_stream(
    base_url: str,
    fingerprint: str,
    token: str,
    job_id: str,
    origin: Optional[str] = None,
):
    """Open a pinned wss connection to /ws/harness/jobs/{id}/logs.

    Returns a websockets sync ClientConnection; the caller iterates messages and
    closes it. The peer cert fingerprint is verified against the pin immediately
    after connect, before any frames are read.
    """
    from websockets.sync.client import connect as ws_connect

    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise InsecureSchemeError(f"control-plane URL must be https://, got {base_url!r}")
    pinned = normalize_fingerprint(fingerprint)
    host = parsed.hostname
    port = parsed.port or 443
    ws_url = f"wss://{host}:{port}/ws/harness/jobs/{job_id}/logs"
    headers = {"Authorization": f"Bearer {token}"}
    conn = ws_connect(
        ws_url,
        ssl=_pinned_ssl_context(),
        additional_headers=headers,
        origin=origin,
    )
    try:
        sock = conn.socket  # ssl-wrapped socket for a wss connection
        _verify_peer(sock, pinned)
    except Exception:
        conn.close()
        raise
    return conn
