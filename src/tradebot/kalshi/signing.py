from __future__ import annotations

import base64
import re
from dataclasses import dataclass

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def load_rsa_private_key_from_file(path: str) -> rsa.RSAPrivateKey:
    with open(path, "rb") as key_file:
        raw = key_file.read()

    # Some users store the key in a file that contains extra text (e.g. "API Key:").
    # Try to extract the PEM block if present.
    m = re.search(
        rb"-----BEGIN ([A-Z ]+)-----[\s\S]+?-----END \1-----",
        raw,
    )
    pem = m.group(0) if m else raw

    return serialization.load_pem_private_key(pem, password=None, backend=default_backend())


def create_kalshi_signature(
    private_key: rsa.RSAPrivateKey,
    *,
    timestamp_ms: str,
    method: str,
    path: str,
) -> str:
    # Important: strip query parameters before signing
    path_without_query = path.split("?")[0]
    message = f"{timestamp_ms}{method.upper()}{path_without_query}".encode("utf-8")

    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


@dataclass(frozen=True)
class KalshiAuth:
    key_id: str
    private_key: rsa.RSAPrivateKey
