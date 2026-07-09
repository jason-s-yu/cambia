"""
tests/harness_tls_util.py

Shared helpers for the harness transport/client tests: a self-signed cert
generator and a threaded localhost HTTPS server. No real network: everything
binds 127.0.0.1 on an ephemeral port.
"""

import hashlib
import json
import ssl
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple


def make_self_signed(tmp_path, cn: str = "127.0.0.1") -> Tuple[str, str, str]:
    """Generate a self-signed ed25519 cert; return (cert_path, key_path, sha256_hex).

    The returned fingerprint is sha256 of the DER cert, exactly what a pinned
    client computes from getpeercert(binary_form=True).
    """
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    # RSA key (TLS server certs need an algorithm the stdlib ssl server accepts
    # broadly; ed25519 server certs are fine too but RSA maximizes portability).
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(cn))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    fingerprint = hashlib.sha256(der).hexdigest()

    cert_path = tmp_path / "server.crt"
    key_path = tmp_path / "server.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return str(cert_path), str(key_path), fingerprint


class RecordingServer:
    """A threaded HTTPS (or plaintext) server that records requests and returns
    canned responses keyed by (method, path-prefix)."""

    def __init__(
        self,
        cert_path: Optional[str],
        key_path: Optional[str],
        routes: Dict[Tuple[str, str], Tuple[int, Any]],
    ):
        self.requests: List[Dict[str, Any]] = []
        self._routes = routes
        recorder = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _dispatch(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                recorder.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k: v for k, v in self.headers.items()},
                        "body": body.decode("utf-8") if body else "",
                    }
                )
                status, payload = 404, {"error": "no route"}
                for (m, prefix), (st, pl) in routes.items():
                    if self.command == m and self.path.split("?")[0].startswith(prefix):
                        status, payload = st, pl
                        break
                data = json.dumps(payload).encode("utf-8") if payload is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            do_GET = _dispatch
            do_POST = _dispatch
            do_DELETE = _dispatch

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        if cert_path and key_path:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_path, key_path)
            self._server.socket = ctx.wrap_socket(self._server.socket, server_side=True)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        scheme = "https" if isinstance(self._server.socket, ssl.SSLSocket) else "http"
        return f"{scheme}://127.0.0.1:{self.port}"
